"""
preprocessing.py
================
Converte il CSV delle probe request in tensori PyTorch pronti per il
transformer.

Struttura del CSV di input
--------------------------
Ogni riga è una probe request con campi:
  - label       : int, ground truth del device (39 classi distinte)
  - is_ios      : 0/1, flag sistema operativo
  - timestamp   : stringa datetime (non usata come feature)
  - mac         : stringa MAC randomizzato (non usata come feature)
  - ie0         : SSID in formato hex separato da ':' (es. "43:61:73:65...")
                  NaN se wildcard probe (assenza SSID)
  - ie1         : lista Python come stringa (es. "[2, 4, 11, 22]")
                  Supported Rates
  - ie50        : lista Python come stringa
                  Extended Supported Rates
  - ie3         : int, canale DS Parameter Set
                  NON discriminante (lo stesso device usa canali multipli)
  - ie45_*      : numerici float, sottocampi HT Capabilities
                  NaN se HT Capabilities assente
  - ie127_0     : lista Python come stringa, Extended Capabilities
                  lunghezza variabile (1-10 elementi), padding necessario
  - ie221_oui_* : float, OUI dei vendor specific element
  - ie221_type_*: float, type dei vendor specific element
  - ie191       : float, VHT Capabilities (un unico valore scalare)
  - ie107_*     : sottocampi Interworking (molto sparsi, 96% NaN)

Scelte di design
----------------
1. ie3 (canale) viene ESCLUSO: varia liberamente per lo stesso device,
   non porta informazione di fingerprinting.

2. ie0 (SSID) viene incluso come feature BINARIA (presente/assente)
   e come hash dell'SSID decodificato. L'SSID grezzo non va usato
   direttamente perché la stessa rete può essere vista da device diversi,
   ma la combinazione "questo device ha cercato questo SSID" può essere
   discriminante in sessioni di cattura limitate.

3. ie1, ie50, ie127_0 sono liste di lunghezza variabile. Le
   rappresentiamo con multi-hot encoding su un vocabolario fisso di
   valori osservati nel dataset, più la lunghezza come feature
   separata. Questo preserva l'informazione senza perdere la struttura.

4. I NaN vengono imputati con un valore sentinella (-1.0) DIVERSO
   da zero, perché "campo assente" è un'informazione di fingerprint
   forte (es. un device senza HT Capabilities è riconoscibile proprio
   per questa assenza).

5. L'output finale è un vettore flat di dimensione M per
   ogni probe. Ogni probe è un singolo "campione", non una sequenza.
   Il transformer lo tratta come un token unico e confronta probe
   diverse all'interno della stessa batch.

Approccio alternativo mantenuto come opzione: "feature group tokens",
in cui ogni IE diventa un token separato nella sequenza. Questo è
configurabile tramite il parametro `as_sequence` nel dataset.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from utils.ProbeDataset import ProbeDataset
from utils.feature_schema import build_probe_schema

# -----------------------------------------------------------------------
# Funzione di preprocessing dell'intero DataFrame
# -----------------------------------------------------------------------

def preprocess_dataframe(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Converte l'intero DataFrame in:
      X : np.ndarray di shape (N, M) con le feature di ogni probe
      y : np.ndarray di shape (N,) con le label intere dei device
    """
    
    # Schema completo (equivale al preprocessing originale, 86 feature)
    schema = build_probe_schema()
    X = schema.transform(df)        # (N, 86)
    
    # Rimappa le label in indici contigui 0..n-1
    # (nel CSV le label sono 1,2,3,...,62,... non necessariamente contigue)
    raw_labels = df['label'].values
    unique_labels = sorted(set(raw_labels))
    label_to_idx = {lbl: idx for idx, lbl in enumerate(unique_labels)}
    y = np.array([label_to_idx[l] for l in raw_labels], dtype=np.int64)

    return X.astype(np.float32), y


def load_csv(path: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Carica il CSV, applica il preprocessing e restituisce:
      X       : (N, M) float32
      y       : (N,) int64
      meta    : dizionario con informazioni utili (n_classes, label_map, ecc.)
    """
    df = pd.read_csv(path)
    print(f"CSV caricato: {len(df)} righe, {df['label'].nunique()} device unici")

    X, y = preprocess_dataframe(df)

    # Mappa inversa idx -> label originale (utile per interpretare i risultati)
    raw_labels = df['label'].values
    unique_labels = sorted(set(raw_labels))
    idx_to_label = {idx: lbl for idx, lbl in enumerate(unique_labels)}

    meta = {
        'n_classes': len(unique_labels),
        'n_samples': len(X),
        'feature_dim': X.shape[1],
        'idx_to_label': idx_to_label,
        'label_to_idx': {v: k for k, v in idx_to_label.items()},
    }

    print(f"Feature dim: {X.shape[1]} | Classi: {meta['n_classes']}")
    return X, y, meta


def build_datasets(
    X: np.ndarray,
    y: np.ndarray,
    val_fraction: float = 0.15,
    test_fraction: float = 0.10,
    seed: int = 42,
) -> tuple[ProbeDataset, ProbeDataset, ProbeDataset]:
    """
    Divide i dati in train / val / test con split stratificato per label.

    Lo split stratificato è importante perché le classi sono molto
    sbilanciate (da 12 a 10734 probe per device): senza stratificazione,
    le classi con pochi campioni potrebbero finire tutte in train o
    tutte in val.

    Restituisce (train_dataset, val_dataset, test_dataset).
    """

    # Split train vs (val + test)
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y,
        test_size=(val_fraction + test_fraction),
        stratify=y,
        random_state=seed,
    )

    # Split val vs test
    relative_test = test_fraction / (val_fraction + test_fraction)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp,
        test_size=relative_test,
        stratify=y_tmp,
        random_state=seed,
    )

    print(f"Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    return (
        ProbeDataset(X_train, y_train),
        ProbeDataset(X_val,   y_val),
        ProbeDataset(X_test,  y_test),
    )
