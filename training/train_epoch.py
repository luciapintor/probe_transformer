import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import defaultdict

from utils.model import ProbeEncoder
from training.losses import CombinedLoss

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



