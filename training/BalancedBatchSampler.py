import random
import numpy as np
from torch.utils.data import Sampler
from collections import defaultdict
from typing import Iterator

# -----------------------------------------------------------------------
# BalancedBatchSampler
# -----------------------------------------------------------------------

class BalancedBatchSampler(Sampler):
    """
    Sampler che costruisce batch bilanciati per classe.

    Ogni batch contiene esattamente:
      n_classes_per_batch * n_samples_per_class campioni

    Parametri
    ---------
    labels : array-like di int
        Label di ogni campione nel dataset
    n_classes_per_batch : int
        Quante classi distinte includere in ogni batch.
        Più alto -> più negativi per anchor -> loss più informativa.
        Deve essere <= numero totale di classi.
    n_samples_per_class : int
        Quante probe per classe per batch.
        Deve essere >= 2 per garantire almeno un positivo per anchor.
        Con resampling: se una classe ha meno campioni, vengono ripetuti.
    """

    def __init__(
        self,
        labels: np.ndarray,
        n_classes_per_batch: int = 20,
        n_samples_per_class: int = 8,
    ):
        self.labels = np.array(labels)
        self.n_classes_per_batch = n_classes_per_batch
        self.n_samples_per_class = n_samples_per_class

        # Indici per ogni classe
        self.class_indices: dict[int, list[int]] = defaultdict(list)
        for idx, label in enumerate(self.labels):
            self.class_indices[int(label)].append(idx)

        self.classes = list(self.class_indices.keys())
        self.batch_size = n_classes_per_batch * n_samples_per_class

        # Numero di batch per epoca: approssimato sul numero totale di campioni
        self.n_batches = len(self.labels) // self.batch_size

    def __iter__(self) -> Iterator[list[int]]:
        """
        Genera indici per ogni batch. Ogni chiamata a __iter__ (cioè ogni
        epoca) produce una nuova sequenza casuale di batch.
        """
        for _ in range(self.n_batches):
            batch_indices = []

            # Scegli n_classes_per_batch classi a caso
            selected_classes = random.sample(
                self.classes,
                min(self.n_classes_per_batch, len(self.classes))
            )

            for cls in selected_classes:
                indices = self.class_indices[cls]

                if len(indices) >= self.n_samples_per_class:
                    # Campiona senza restituzione
                    sampled = random.sample(indices, self.n_samples_per_class)
                else:
                    # Resampling: la classe ha pochi campioni, li ripete
                    sampled = random.choices(indices, k=self.n_samples_per_class)

                batch_indices.extend(sampled)

            # Mescola l'ordine all'interno della batch (non per classe)
            random.shuffle(batch_indices)
            yield batch_indices

    def __len__(self) -> int:
        return self.n_batches

