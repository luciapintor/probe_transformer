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

Uso da riga di comando:
    python train.py 

Uso da codice:
    from train import train
    train(csv_path="all_A_full.csv", epochs=50)
"""

import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import defaultdict

from utils.BalancedBatchSampler import BalancedBatchSampler
from utils.preprocessing import load_csv, build_datasets
from utils.model import ProbeEncoder, TransformerConfig
from utils.losses import CombinedLoss

# -----------------------------------------------------------------------
# Training e validation step
# -----------------------------------------------------------------------

def train_epoch(
    model: ProbeEncoder,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: CombinedLoss,
    device: torch.device,
    grad_clip: float = 1.0,
) -> dict:
    """
    Esegue una singola epoca di training.
    Restituisce un dizionario con le loss medie (total, supcon, ce).
    """
    model.train()
    totals = defaultdict(float)
    n_batches = 0

    for x, labels in loader:
        x      = x.to(device)
        labels = labels.to(device)

        # Forward pass: ottieni embedding L2-normalizzati
        z = model(x, normalize=True)

        # Calcola la loss
        loss, components = loss_fn(z, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping: evita esplosione dei gradienti,
        # specialmente utile con transformer e learning rate alte
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        optimizer.step()

        # Accumula metriche
        for k, v in components.items():
            totals[k] += v
        n_batches += 1

    return {k: v / n_batches for k, v in totals.items()}


@torch.no_grad()
def validate_epoch(
    model: ProbeEncoder,
    loader: DataLoader,
    loss_fn: CombinedLoss,
    device: torch.device,
) -> dict:
    """
    Esegue la validation sull'intero val set.
    Calcola loss + accuracy di clustering (k-NN a 1 vicino sull'embedding).
    """
    model.eval()
    totals = defaultdict(float)
    n_batches = 0

    # Raccoglie tutti gli embedding per calcolare metriche di separazione
    all_z, all_labels = [], []

    for x, labels in loader:
        x      = x.to(device)
        labels = labels.to(device)
        z      = model(x, normalize=True)

        loss, components = loss_fn(z, labels)

        for k, v in components.items():
            totals[k] += v
        n_batches += 1

        all_z.append(z.cpu())
        all_labels.append(labels.cpu())

    metrics = {k: v / n_batches for k, v in totals.items()}

    # --- Metrica di separazione: delta cosine similarity ---
    # Misura quanto gli embedding dello stesso device sono più simili
    # tra loro rispetto a probe di device diversi.
    # Un delta > 0.3 indica buona separazione; > 0.5 è ottimo.
    Z = torch.cat(all_z).numpy()
    L = torch.cat(all_labels).numpy()

    same_sim, diff_sim = [], []
    # Campiona max 2000 coppie per velocità (calcolo O(N^2) è lento su CPU)
    n = min(len(Z), 500)
    idx = np.random.choice(len(Z), n, replace=False)
    Z_sub, L_sub = Z[idx], L[idx]

    sim_matrix = Z_sub @ Z_sub.T   # cosine similarity (già L2-norm)
    for i in range(n):
        for j in range(i + 1, n):
            s = float(sim_matrix[i, j])
            if L_sub[i] == L_sub[j]:
                same_sim.append(s)
            else:
                diff_sim.append(s)

    if same_sim and diff_sim:
        metrics['same_sim']  = float(np.mean(same_sim))
        metrics['diff_sim']  = float(np.mean(diff_sim))
        metrics['delta_sim'] = metrics['same_sim'] - metrics['diff_sim']
    else:
        metrics['same_sim'] = metrics['diff_sim'] = metrics['delta_sim'] = 0.0

    return metrics


# -----------------------------------------------------------------------
# Funzione principale di training
# -----------------------------------------------------------------------

def train(
    csv_path: str,
    output_path: str = "probe_encoder.pt",
    epochs: int = 100,
    n_classes_per_batch: int = 20,
    n_samples_per_class: int = 8,
    lr: float = 5e-4,
    weight_decay: float = 1e-4,
    temperature: float = 0.1,
    ce_weight: float = 0.5,
    d_model: int = 128,
    num_layers: int = 3,
    embed_dim: int = 64,
    nhead: int = 4,
    pooling: str = "mean",
    dropout: float = 0.1,
    val_fraction: float = 0.15,
    test_fraction: float = 0.10,
    device_str: str = "auto",
    seed: int = 42,
):
    """
    Funzione di training completa. Parametri principali:

    csv_path            : path al CSV delle probe request
    output_path         : dove salvare il checkpoint migliore
    epochs              : epoche di training
    n_classes_per_batch : classi per batch nel BalancedBatchSampler
    n_samples_per_class : probe per classe per batch
    lr                  : learning rate per AdamW
    temperature         : temperatura della SupCon Loss
    ce_weight           : peso della Cross Entropy ausiliaria (0 = solo SupCon)
    embed_dim           : dimensione embedding finale
    """
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

    # DataLoader per training con BalancedBatchSampler
    # batch_sampler sostituisce batch_size + shuffle (non compatibili insieme)
    balanced_sampler = BalancedBatchSampler(
        labels=train_ds.y.numpy(),
        n_classes_per_batch=n_classes_per_batch,
        n_samples_per_class=n_samples_per_class,
    )
    train_loader = DataLoader(
        train_ds,
        batch_sampler=balanced_sampler,  # batch_size gestito dal sampler
        num_workers=0,                   # 0 per evitare warning su CPU
        pin_memory=(device.type == "cuda"),
    )

    # DataLoader per validation: batch standard, niente shuffling
    val_loader = DataLoader(
        val_ds,
        batch_size=256,
        shuffle=False,
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
