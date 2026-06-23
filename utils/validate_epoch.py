import numpy as np
import torch
from torch.utils.data import DataLoader
from collections import defaultdict

from utils.model import ProbeEncoder
from utils.losses import CombinedLoss



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
