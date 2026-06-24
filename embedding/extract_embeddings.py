import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from utils.model import ProbeEncoder

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
