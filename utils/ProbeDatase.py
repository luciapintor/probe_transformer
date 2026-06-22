import numpy as np
import torch
from torch.utils.data import Dataset

# -----------------------------------------------------------------------
# Dataset PyTorch
# -----------------------------------------------------------------------

class ProbeDataset(Dataset):
    """
    Dataset PyTorch per le probe request.

    Ogni item restituisce (features, label) dove:
      features : tensore float32 di shape (FEATURE_DIM,)
      label    : int64 scalare, indice del device

    La label è necessaria per la Supervised Contrastive Loss,
    che usa tutte le probe della stessa classe come positivi.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        X : (N, FEATURE_DIM) array di feature già preprocessate
        y : (N,) array di label intere
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]