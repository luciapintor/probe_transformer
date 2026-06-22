"""
inference.py
============
Inferenza su un CSV di probe request con un ProbeEncoder addestrato.

Funzionalità
------------
1. Carica il modello da checkpoint
2. Estrae embedding per tutte le probe nel CSV
3. Applica DBSCAN (o HDBSCAN se disponibile) per il clustering
4. Salva i risultati in un CSV e/o JSON
5. Stampa statistiche di qualità se il CSV contiene la colonna 'label'

Uso da riga di comando:
    python inference.py \
        --model probe_encoder.pt \
        --csv all_A_full.csv \
        --output clusters.csv \
        --eps 0.25 \
        --min_samples 3

Uso da codice:
    from inference import load_encoder, extract_embeddings, cluster
    encoder = load_encoder("probe_encoder.pt", device)
    Z = extract_embeddings(encoder, X, device)
    labels = cluster(Z, eps=0.25)
"""

import argparse
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from utils.preprocessing import load_csv, FEATURE_DIM
from utils.model import ProbeEncoder, TransformerConfig


# -----------------------------------------------------------------------
# Caricamento modello
# -----------------------------------------------------------------------

def load_encoder(
    checkpoint_path: str,
    device: torch.device,
) -> ProbeEncoder:
    """
    Carica un ProbeEncoder da un checkpoint salvato durante il training.
    Il checkpoint contiene la configurazione del modello, quindi non
    è necessario passare i parametri manualmente.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = TransformerConfig(**ckpt['config'])
    model = ProbeEncoder(config).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"Modello caricato da: {checkpoint_path} "
          f"(epoca {ckpt.get('epoch', '?')}, "
          f"delta_sim={ckpt.get('val_metrics', {}).get('delta_sim', '?'):.3f})")
    return model


# -----------------------------------------------------------------------
# Estrazione embedding
# -----------------------------------------------------------------------

@torch.no_grad()
def extract_embeddings(
    model: ProbeEncoder,
    X: np.ndarray,
    device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    """
    Estrae gli embedding L2-normalizzati per tutte le probe in X.

    Parametri
    ---------
    X : (N, FEATURE_DIM) array di feature
    device : torch.device
    batch_size : dimensione del batch per l'inferenza

    Restituisce
    -----------
    Z : (N, embed_dim) array numpy, valori in [-1, 1], norma L2 = 1
    """
    X_tensor = torch.tensor(X, dtype=torch.float32)
    dataset   = TensorDataset(X_tensor)
    loader    = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_z = []
    for (batch_x,) in loader:
        batch_x = batch_x.to(device)
        z = model(batch_x, normalize=True)
        all_z.append(z.cpu().numpy())

    Z = np.concatenate(all_z, axis=0)
    print(f"Embedding estratti: shape={Z.shape}, norma media={np.linalg.norm(Z, axis=1).mean():.4f}")
    return Z


# -----------------------------------------------------------------------
# Clustering
# -----------------------------------------------------------------------

def cluster_dbscan(
    Z: np.ndarray,
    eps: float = 0.25,
    min_samples: int = 3,
) -> np.ndarray:
    """
    Applica DBSCAN sugli embedding con metrica cosine.

    Con embedding L2-normalizzati la distanza coseno è:
        d(a, b) = 1 - a·b

    Quindi eps=0.25 significa che due probe sono nello stesso cluster
    se la loro cosine similarity è >= 0.75.

    Parametri da calibrare
    ----------------------
    eps         : raggio del vicinato. Valore suggerito: 0.2-0.4.
                  Usa tune_eps() per trovare il valore ottimale.
    min_samples : min probe per formare un core point.
                  Con dataset grandi, 3-5 è ragionevole.
                  Con dataset piccoli o device con poche probe, usa 2.

    Restituisce
    -----------
    labels : (N,) array int, -1 = rumore (non clusterizzato)
    """
    db = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine', n_jobs=-1)
    labels = db.fit_predict(Z)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    print(f"DBSCAN: {n_clusters} cluster, {n_noise} probe non clusterizzate "
          f"({100*n_noise/len(Z):.1f}%)")
    return labels


def tune_eps(
    Z: np.ndarray,
    true_labels: np.ndarray,
    eps_range: list = None,
    min_samples: int = 3,
) -> float:
    """
    Cerca il valore di eps che massimizza l'Adjusted Rand Index (ARI)
    rispetto alle label reali dei device.

    Utile per calibrare DBSCAN quando si hanno label di ground truth.
    In produzione (senza label) usare la curva k-distance invece.

    Restituisce il valore di eps ottimale.
    """
    if eps_range is None:
        eps_range = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

    print(f"\nTuning eps (min_samples={min_samples}):")
    print(f"{'eps':>6} | {'n_cluster':>9} | {'ARI':>8} | {'NMI':>8} | {'noise%':>7}")
    print("-" * 50)

    best_ari = -1.0
    best_eps = eps_range[0]

    for eps in eps_range:
        pred = cluster_dbscan(Z, eps=eps, min_samples=min_samples)
        # Esclude le probe di rumore (-1) dal calcolo delle metriche
        valid = pred != -1
        if valid.sum() < 10:
            continue

        ari = adjusted_rand_score(true_labels[valid], pred[valid])
        nmi = normalized_mutual_info_score(true_labels[valid], pred[valid])
        n_cl = len(set(pred)) - 1
        noise_pct = 100 * (1 - valid.mean())

        print(f"{eps:6.2f} | {n_cl:9d} | {ari:8.4f} | {nmi:8.4f} | {noise_pct:6.1f}%")

        if ari > best_ari:
            best_ari = ari
            best_eps = eps

    print(f"\nMiglior eps={best_eps:.2f} (ARI={best_ari:.4f})")
    return best_eps


# -----------------------------------------------------------------------
# Analisi della separazione degli embedding
# -----------------------------------------------------------------------

def embedding_separation_stats(
    Z: np.ndarray,
    true_labels: np.ndarray,
    n_sample: int = 1000,
) -> dict:
    """
    Calcola statistiche di separazione degli embedding rispetto alle label.

    Campiona n_sample probe per efficienza (O(N^2) è lento su dataset grandi).

    Restituisce un dizionario con:
      same_mean  : cosine similarity media tra probe dello stesso device
      diff_mean  : cosine similarity media tra probe di device diversi
      delta      : same_mean - diff_mean (target: > 0.3)
      same_std   : deviazione standard della similarità intra-classe
      diff_std   : deviazione standard della similarità inter-classe
    """
    n = min(n_sample, len(Z))
    idx = np.random.choice(len(Z), n, replace=False)
    Z_sub = Z[idx]
    L_sub = true_labels[idx]

    # Matrice di similarità coseno (embedding già normalizzati)
    sim = Z_sub @ Z_sub.T   # (n, n)

    same_sims, diff_sims = [], []
    for i in range(n):
        for j in range(i + 1, n):
            s = float(sim[i, j])
            if L_sub[i] == L_sub[j]:
                same_sims.append(s)
            else:
                diff_sims.append(s)

    stats = {
        'same_mean': float(np.mean(same_sims)) if same_sims else 0.0,
        'diff_mean': float(np.mean(diff_sims)) if diff_sims else 0.0,
        'same_std':  float(np.std(same_sims))  if same_sims else 0.0,
        'diff_std':  float(np.std(diff_sims))  if diff_sims else 0.0,
    }
    stats['delta'] = stats['same_mean'] - stats['diff_mean']

    print(f"\nSeparazione embedding:")
    print(f"  Stessa sorgente : mean={stats['same_mean']:.3f} ± {stats['same_std']:.3f}")
    print(f"  Sorgente diversa: mean={stats['diff_mean']:.3f} ± {stats['diff_std']:.3f}")
    print(f"  Delta           : {stats['delta']:.3f}")
    return stats


# -----------------------------------------------------------------------
# Funzione principale di inferenza
# -----------------------------------------------------------------------

def run_inference(
    model_path: str,
    csv_path: str,
    output_path: str = "clusters.csv",
    eps: float = 0.25,
    min_samples: int = 3,
    tune: bool = False,
    device_str: str = "auto",
    batch_size: int = 512,
):
    """
    Pipeline completa di inferenza.

    Se il CSV ha la colonna 'label', calcola anche ARI, NMI e statistiche
    di separazione per valutare la qualità del clustering.
    """
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    # Carica dati e modello
    X, y, meta = load_csv(csv_path)
    model = load_encoder(model_path, device)

    # Estrai embedding
    Z = extract_embeddings(model, X, device, batch_size=batch_size)

    # Statistiche di separazione (richiede label)
    stats = embedding_separation_stats(Z, y)

    # Tuning eps (opzionale, richiede label)
    if tune:
        eps = tune_eps(Z, y, min_samples=min_samples)

    # Clustering
    pred_labels = cluster_dbscan(Z, eps=eps, min_samples=min_samples)

    # Metriche di qualità (se abbiamo le label)
    valid = pred_labels != -1
    if valid.sum() > 0:
        ari = adjusted_rand_score(y[valid], pred_labels[valid])
        nmi = normalized_mutual_info_score(y[valid], pred_labels[valid])
        print(f"\nMetriche clustering (probe non-rumore):")
        print(f"  ARI: {ari:.4f}  (1.0 = perfetto, 0.0 = casuale)")
        print(f"  NMI: {nmi:.4f}  (1.0 = perfetto, 0.0 = casuale)")

    # Salva risultati
    df_out = pd.DataFrame({
        'true_label':    y,
        'pred_cluster':  pred_labels,
        'embedding_dim0': Z[:, 0],   # prime 2 dim per debug/visualizzazione
        'embedding_dim1': Z[:, 1],
    })
    df_out.to_csv(output_path, index=False)
    print(f"\nRisultati salvati in: {output_path}")

    return Z, pred_labels, stats

# -----------------------------------------------------------------------
# Funzione principale per esecuzione da riga di comando
# -----------------------------------------------------------------------

if __name__ == "__main__":
    print("Inferenza ProbeEncoder")
    model_path = "models/probe_encoder.pt"
    csv_path = "dataset/dataset_merged_probes_csv/data_with_label/all_A_full.csv"
    output_path = "outputs/clusters.csv"
    eps = 0.25
    min_samples = 3
    tune = False
    device_str = "auto"
    batch_size = 512

    run_inference(
        model_path=model_path,
        csv_path=csv_path,
        output_path=output_path,
        eps=eps,
        min_samples=min_samples,
        tune=tune,
        device_str=device_str,
        batch_size=batch_size,
    )
