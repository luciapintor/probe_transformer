"""
train.py
========
Training loop per il ProbeEncoder con Supervised Contrastive Loss.

Problema chiave: batch balancing per SupCon
-------------------------------------------
La SupCon Loss richiede che ogni batch contenga ALMENO DUE probe per
classe (altrimenti un anchor non ha positivi e non contribuisce alla loss).
Con un DataLoader standard e shuffle casuale, le classi con pochi campioni
(es. label con 12 probe) rischiano di non essere rappresentate in molte batch.

Soluzione: BalancedBatchSampler
  - Per ogni batch, campiona prima K classi a caso
  - Poi campiona M probe per ogni classe (con resampling se la classe
    ha meno di M campioni)
  - Batch size effettivo = K * M
  - Garantisce almeno M positivi per anchor nella batch

Valore consigliato: K=20 classi, M=8 probe per classe -> batch di 160.
Con 39 classi totali e batch di 160, ogni batch copre ~51% delle classi.
"""

import random
import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.BalancedBatchSampler import BalancedBatchSampler
from utils.preprocessing import load_csv, build_datasets
from utils.model import ProbeEncoder, TransformerConfig
from utils.losses import CombinedLoss
from utils.train_epoch import train_epoch
from utils.validate_epoch import validate_epoch


# -----------------------------------------------------------------------
# Funzione principale di training
# -----------------------------------------------------------------------

def train(
    csv_path: str,                  # path al CSV delle probe request
    output_path: str = "probe_encoder.pt", # dove salvare il checkpoint migliore
    epochs: int = 100,              # epoche di training
    n_classes_per_batch: int = 20,  # classi per batch nel BalancedBatchSampler
    n_samples_per_class: int = 8,   # probe per classe per batch
    lr: float = 5e-4,               # learning rate per AdamW
    weight_decay: float = 1e-4,     # weight decay per AdamW
    temperature: float = 0.1,       # temperatura della SupCon Loss
    ce_weight: float = 0.5,         # peso della Cross Entropy ausiliaria (0 = solo SupCon)
    d_model: int = 128,             # dimensione del modello Transformer
    num_layers: int = 3,            # numero di layer del Transformer
    embed_dim: int = 64,            # dimensione embedding finale
    nhead: int = 4,                 # numero di teste multi-head nel Transformer
    pooling: str = "mean",          # pooling: "mean" o "cls"
    dropout: float = 0.1,           # dropout rate nel Transformer
    val_fraction: float = 0.15,     # frazione del dataset per validation
    test_fraction: float = 0.10,    # frazione del dataset per test
    device_str: str = "auto",       # "auto", "cpu" o "cuda"
    seed: int = 42,                 # seed per riproducibilità
):
    # Seed per riproducibilità
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Device
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"Device: {device}")

    # ----------------------------------------------------------------
    # Caricamento dati
    # ----------------------------------------------------------------
    X, y, meta = load_csv(csv_path)
    train_ds, val_ds, test_ds = build_datasets(
        X, y, val_fraction=val_fraction, test_fraction=test_fraction, seed=seed
    )

    # BalancedBatchSampler seleziona gli indici di K classi e M probe per batch, 
    # garantendo almeno M positivi per anchor. 
    # Serve per bilanciare le classi e migliorare la qualità del training.
    balanced_sampler = BalancedBatchSampler(
        labels=train_ds.y.numpy(),
        n_classes_per_batch=n_classes_per_batch,
        n_samples_per_class=n_samples_per_class,
    )
    
    # train_loader usa il BalancedBatchSampler per generare batch bilanciati
    # Ad ogni loop di training riceve una sequenza di indici dal balanced_sampler
    # e li assembla in un tensore.
    train_loader = DataLoader(
        train_ds,
        batch_sampler=balanced_sampler,  # batch_size gestito dal sampler
        num_workers=0,                   # 0 per evitare warning su CPU
        pin_memory=(device.type == "cuda"),
    )

    # DataLoader per validation: batch standard, niente shuffling
    val_loader = DataLoader(
        val_ds,
        batch_size=256,                 # batch standard, più grande per velocità
        shuffle=False,                  # ordine fisso: riproducibilità delle metriche
        num_workers=0,
    )

    batch_size_eff = n_classes_per_batch * n_samples_per_class
    print(f"Batch size effettivo: {batch_size_eff} "
          f"({n_classes_per_batch} classi x {n_samples_per_class} campioni)")
    print(f"Batch per epoca: {len(balanced_sampler)}")

    # ----------------------------------------------------------------
    # Modello
    # ----------------------------------------------------------------
    config = TransformerConfig(
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=d_model * 2,
        embed_dim=embed_dim,
        dropout=dropout,
        pooling=pooling,
    )
    model = ProbeEncoder(config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parametri trainable: {n_params:,}")

    # ----------------------------------------------------------------
    # Loss, ottimizzatore, scheduler
    # ----------------------------------------------------------------
    loss_fn = CombinedLoss(
        embed_dim=embed_dim,
        n_classes=meta['n_classes'],
        temperature=temperature,
        supcon_weight=1.0,
        ce_weight=ce_weight,
    ).to(device)

    # AdamW: come Adam ma con weight decay corretto (decoupled)
    # Ottimizza sia encoder sia loss_fn (che ha il classifier ausiliario)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(loss_fn.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )

    # Cosine annealing: riduce lr gradualmente da lr a lr_min
    # Migliore di StepLR per training lunghi con SupCon
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=lr * 0.01,   # lr minima = 1% della lr iniziale
    )

    # ----------------------------------------------------------------
    # Training loop
    # ----------------------------------------------------------------
    best_delta = -float('inf')   # criteri: massimizza delta_sim in validation
    best_epoch = 0

    print(f"\n{'Epoch':>6} | {'train_loss':>10} | {'val_loss':>8} | "
          f"{'same_sim':>8} | {'diff_sim':>8} | {'delta_sim':>9}")
    print("-" * 65)

    for epoch in range(1, epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_metrics   = validate_epoch(model, val_loader, loss_fn, device)
        scheduler.step()

        print(
            f"{epoch:6d} | {train_metrics['total']:10.4f} | "
            f"{val_metrics['total']:8.4f} | "
            f"{val_metrics['same_sim']:8.3f} | "
            f"{val_metrics['diff_sim']:8.3f} | "
            f"{val_metrics['delta_sim']:9.3f}"
        )

        # Salva il checkpoint con la migliore separazione degli embedding
        if val_metrics['delta_sim'] > best_delta:
            best_delta = val_metrics['delta_sim']
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'config': config.__dict__,
                'model_state': model.state_dict(),
                'val_metrics': val_metrics,
                'meta': meta,
            }, output_path)
            print(f"         -> checkpoint salvato (delta_sim={best_delta:.3f})")

    print(f"\nTraining completato. Miglior epoca: {best_epoch} "
          f"(delta_sim={best_delta:.3f})")
    print(f"Modello salvato in: {output_path}")
    return best_delta


# -----------------------------------------------------------------------
# Funzione principale per esecuzione da riga di comando
# -----------------------------------------------------------------------

if __name__ == "__main__":
    print("Addestra ProbeEncoder con Supervised Contrastive Learning")
    csv_path = "dataset/dataset_merged_probes_csv/data_with_label/all_A_full.csv"
    output_path = "models/probe_encoder.pt"
    epochs = 20                     # Epoche di training (consigliato 50-100 per risultati stabili)
    n_classes_per_batch=20          # Classi per batch (BalancedBatchSampler)
    n_samples_per_class=8           # Campioni per classe per batch
    lr = 5e-4                       # Learning rate per AdamW
    temperature = 0.1               # Temperatura della SupCon Loss
    ce_weight = 0.5                 # Peso della Cross Entropy ausiliaria (0 = solo SupCon)
    d_model = 128                   # Dimensione del modello Transformer
    num_layers = 3                  # Numero di layer del Transformer
    embed_dim = 64                  # Dimensione embedding finale
    pooling = "mean"                # Pooling: "mean" o "cls"
    device = "auto"                 # "auto", "cpu" o "cuda"
    seed = 42                       # Seed per riproducibilità
   
    train(
        csv_path=csv_path,
        output_path=output_path,
        epochs=epochs,
        n_classes_per_batch=n_classes_per_batch,
        n_samples_per_class=n_samples_per_class,
        lr=lr,
        temperature=temperature,
        ce_weight=ce_weight,
        d_model=d_model,
        num_layers=num_layers,
        embed_dim=embed_dim,
        pooling=pooling,
        device_str=device,
        seed=seed,
    )
