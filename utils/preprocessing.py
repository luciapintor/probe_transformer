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

5. L'output finale è un vettore flat di dimensione FEATURE_DIM per
   ogni probe. Ogni probe è un singolo "campione", non una sequenza.
   Il transformer lo tratta come un token unico e confronta probe
   diverse all'interno della stessa batch.

Approccio alternativo mantenuto come opzione: "feature group tokens",
in cui ogni IE diventa un token separato nella sequenza. Questo è
configurabile tramite il parametro `as_sequence` nel dataset.
"""

import ast
import numpy as np
import pandas as pd
from typing import Optional
from sklearn.model_selection import train_test_split

from utils.ProbeDatase import ProbeDataset


# -----------------------------------------------------------------------
# Vocabolari fissi per le colonne lista
# Costruiti dall'analisi del dataset reale.
# -----------------------------------------------------------------------

# ie1: tutti gli 8 valori unici di lista osservati nel dataset
# Usiamo il set di tutti i valori interi che compaiono in qualsiasi lista
IE1_VOCAB = sorted({2, 4, 11, 12, 18, 22, 24, 36, 48, 72, 96, 108,
                    130, 132, 139, 150})  # 16 valori possibili

# ie50: valori analoghi (extended rates)
IE50_VOCAB = sorted({12, 18, 24, 36, 48, 72, 96, 108})  # 8 valori

# ie127_0: ogni elemento è un intero (bitmask), lunghezza max 10
# Non usiamo vocabolario ma padding a lunghezza fissa
IE127_MAX_LEN = 10

# Numero di vendor specific (OUI+type) tracciati nel CSV
N_VENDOR = 4   # ie221_oui_0..3 / ie221_type_0..3


def _parse_list_column(value) -> list:
    """
    Converte una stringa come "[2, 4, 11, 22]" in lista Python.
    Restituisce lista vuota se il valore è NaN o non parsabile.
    """
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return value
    try:
        result = ast.literal_eval(str(value))
        return result if isinstance(result, list) else []
    except (ValueError, SyntaxError):
        return []


def _multihot(values: list, vocab: list) -> np.ndarray:
    """
    Crea un vettore multi-hot di lunghezza len(vocab).
    Ogni posizione è 1.0 se il valore corrispondente è presente nella lista.
    
    Esempio:
        values = [2, 11], vocab = [2, 4, 11, 22]
        -> [1, 0, 1, 0]
    """
    vec = np.zeros(len(vocab), dtype=np.float32)
    value_set = set(values)
    for i, v in enumerate(vocab):
        if v in value_set:
            vec[i] = 1.0
    return vec


def _pad_list(values: list, max_len: int, fill: float = -1.0) -> np.ndarray:
    """
    Porta una lista a lunghezza fissa con padding.
    Il valore di fill è -1.0 per distinguere padding da zero reale.
    I valori vengono normalizzati nell'intervallo [-1, 1] tramite tanh.
    """
    vec = np.full(max_len, fill, dtype=np.float32)
    for i, v in enumerate(values[:max_len]):
        # tanh comprime qualsiasi intero in (-1, 1)
        # valori tipici di ie127 sono 0-32832, tanh/255 li distribuisce bene
        vec[i] = float(np.tanh(v / 255.0))
    return vec


def _decode_ssid(hex_str: str) -> Optional[str]:
    """
    Decodifica SSID da formato hex separato da ':'.
    Restituisce None se la stringa è vuota o non decodificabile.
    """
    if pd.isna(hex_str) or hex_str == '':
        return None
    try:
        raw = hex_str.replace(':', '')
        return bytes.fromhex(raw).decode('utf-8', errors='replace')
    except Exception:
        return None


def _ssid_features(hex_str) -> np.ndarray:
    """
    Estrae 3 feature dall'SSID:
      [0] presente/assente (0.0 o 1.0)
      [1] lunghezza normalizzata (0..1, max 32 caratteri)
      [2] hash dello SSID normalizzato (fingerprint parziale)
    
    Nota: l'SSID è potenzialmente discriminante (un device che cerca
    sempre la stessa rete rivela abitudini), ma va usato con cautela
    in generalizzazione cross-sessione.
    """
    ssid = _decode_ssid(hex_str)
    if ssid is None:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)
    h = hash(ssid) & 0xFFFFFFFF
    return np.array([
        1.0,                          # presente
        min(len(ssid), 32) / 32.0,   # lunghezza norm.
        (h / 0xFFFFFFFF) * 2 - 1,    # hash in [-1, 1]
    ], dtype=np.float32)


def _ie45_features(row: pd.Series) -> np.ndarray:
    """
    Estrae le feature HT Capabilities (ie45_*).
    Sono già numeriche nel CSV; gestiamo solo i NaN con sentinella -1.0.
    
    Colonne: ampduparam, asel, capabilities, txbf,
             mcsset_txunequalmod, mcsset_txrxmcsnotequal, mcsset_txmaxss,
             mcsset_txsetdefined, mcsset_highestdatarate,
             rxbitmask_0to7, _8to15, _16to23, _24to31,
             _32, _33to38, _39to52, _53to76
    """
    cols = [
        'ie45_ampduparam', 'ie45_asel', 'ie45_capabilities', 'ie45_txbf',
        'ie45_mcsset_txunequalmod', 'ie45_mcsset_txrxmcsnotequal',
        'ie45_mcsset_txmaxss', 'ie45_mcsset_txsetdefined',
        'ie45_mcsset_highestdatarate',
        'ie45_rxbitmask_0to7', 'ie45_rxbitmask_8to15',
        'ie45_rxbitmask_16to23', 'ie45_rxbitmask_24to31',
        'ie45_rxbitmask_32', 'ie45_rxbitmask_33to38',
        'ie45_rxbitmask_39to52', 'ie45_rxbitmask_53to76',
    ]
    vec = np.zeros(len(cols), dtype=np.float32)
    for i, col in enumerate(cols):
        v = row.get(col, np.nan)
        if pd.isna(v):
            vec[i] = -1.0   # sentinella: campo assente
        else:
            # normalizza con tanh; i valori più grandi (bitmask 255)
            # vengono compressi in (0, 1)
            vec[i] = float(np.tanh(v / 255.0))
    return vec


def _ie221_features(row: pd.Series) -> np.ndarray:
    """
    Estrae le feature Vendor Specific (ie221_oui_* e ie221_type_*).
    OUI e type insieme identificano il vendor; li trattiamo come
    feature categoriche hash-encoded.
    
    Per ogni slot (0..3):
      [0] presente/assente
      [1] OUI normalizzato (hash in [-1,1])
      [2] type normalizzato
    -> 4 slot * 3 = 12 feature totali
    """
    vec = np.zeros(N_VENDOR * 3, dtype=np.float32)
    for i in range(N_VENDOR):
        oui = row.get(f'ie221_oui_{i}', np.nan)
        typ = row.get(f'ie221_type_{i}', np.nan)
        base = i * 3
        if pd.isna(oui):
            # slot assente: tutto a -1 per distinguere da "oui=0"
            vec[base:base+3] = -1.0
        else:
            vec[base]   = 1.0                                # presente
            vec[base+1] = float(np.tanh(oui / 1e6))         # OUI norm.
            vec[base+2] = float(np.tanh(typ / 10.0)) if not pd.isna(typ) else -1.0
    return vec


def _ie107_features(row: pd.Series) -> np.ndarray:
    """
    Estrae le feature Interworking (ie107_*).
    Molto sparse (96% NaN), ma quando presenti sono discriminanti.
    
    Colonne: access_network_type, asra, internet, esr, uesa
    """
    cols = ['ie107_access_network_type', 'ie107_asra',
            'ie107_internet', 'ie107_esr', 'ie107_uesa']
    vec = np.zeros(len(cols), dtype=np.float32)
    for i, col in enumerate(cols):
        v = row.get(col, np.nan)
        vec[i] = -1.0 if pd.isna(v) else float(np.tanh(v / 10.0))
    return vec


def probe_row_to_vector(row: pd.Series) -> np.ndarray:
    """
    Converte una singola riga del CSV in un vettore numpy flat.
    
    Composizione del vettore finale:
    
      [A] is_ios            : 1 feature   (0/1)
      [B] SSID features     : 3 feature   (presente, lunghezza, hash)
      [C] ie1 multi-hot     : 16 feature  (supported rates)
      [D] ie1 lunghezza     : 1 feature
      [E] ie50 multi-hot    : 8 feature   (extended rates)
      [F] ie50 presente     : 1 feature   (distingue assente da vuoto)
      [G] ie50 lunghezza    : 1 feature
      [H] ie127_0 padded    : 10 feature  (extended capabilities, tanh)
      [I] ie127_0 lunghezza : 1 feature
      [J] ie127_1 padded    : 9 feature   (secondo blocco ext. cap., padding)
      [K] ie45 features     : 17 feature  (HT capabilities)
      [L] ie221 features    : 12 feature  (vendor specific, 4 slot)
      [M] ie191 presente    : 1 feature
      [N] ie191 valore      : 1 feature   (VHT capabilities)
      [O] ie107 features    : 5 feature   (interworking)
                                         ---------------
                              TOTALE    : 87 feature
    
    Le feature vengono concatenate nello stesso ordine ogni volta,
    così la posizione nel vettore è sempre la stessa feature.
    """
    parts = []

    # --- [A] is_ios ---
    # Flag binario già presente nel CSV
    parts.append(np.array([float(row.get('is_ios', 0))], dtype=np.float32))

    # --- [B] SSID ---
    parts.append(_ssid_features(row.get('ie0', np.nan)))

    # --- [C, D] ie1: Supported Rates ---
    ie1_list = _parse_list_column(row.get('ie1', np.nan))
    parts.append(_multihot(ie1_list, IE1_VOCAB))               # multi-hot
    parts.append(np.array([len(ie1_list) / 8.0], dtype=np.float32))  # lunghezza norm.

    # --- [E, F, G] ie50: Extended Supported Rates ---
    ie50_raw = row.get('ie50', np.nan)
    ie50_present = 0.0 if pd.isna(ie50_raw) else 1.0
    ie50_list = _parse_list_column(ie50_raw)
    parts.append(_multihot(ie50_list, IE50_VOCAB))
    parts.append(np.array([ie50_present], dtype=np.float32))
    parts.append(np.array([len(ie50_list) / 8.0], dtype=np.float32))

    # --- [H, I] ie127_0: Extended Capabilities ---
    ie127_list = _parse_list_column(row.get('ie127_0', np.nan))
    parts.append(_pad_list(ie127_list, IE127_MAX_LEN))
    parts.append(np.array([len(ie127_list) / IE127_MAX_LEN], dtype=np.float32))

    # --- [J] ie127_1: secondo blocco extended capabilities ---
    # È una lista come ie127_0 (lunghezza 4 o 9), presente nel 12.5% dei casi.
    # Padding a lunghezza fissa 9; se assente rimane tutto -1.0 (sentinella).
    IE127_1_MAX_LEN = 9
    ie127_1_list = _parse_list_column(row.get('ie127_1', np.nan))
    parts.append(_pad_list(ie127_1_list, IE127_1_MAX_LEN))

    # --- [K] ie45: HT Capabilities ---
    parts.append(_ie45_features(row))

    # --- [L] ie221: Vendor Specific ---
    parts.append(_ie221_features(row))

    # --- [M, N] ie191: VHT Capabilities ---
    ie191 = row.get('ie191', np.nan)
    if pd.isna(ie191):
        # Assente: sentinella -1 su entrambe le feature
        parts.append(np.array([-1.0, -1.0], dtype=np.float32))
    else:
        parts.append(np.array([1.0, float(np.tanh(ie191 / 1e9))], dtype=np.float32))

    # --- [O] ie107: Interworking ---
    parts.append(_ie107_features(row))

    # Concatena tutto in un unico vettore flat
    return np.concatenate(parts)   # shape: (79,)


# Dimensione attesa del vettore (usata come costante negli altri moduli)
FEATURE_DIM = 87


# -----------------------------------------------------------------------
# Funzione di preprocessing dell'intero DataFrame
# -----------------------------------------------------------------------

def preprocess_dataframe(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Converte l'intero DataFrame in:
      X : np.ndarray di shape (N, FEATURE_DIM) con le feature di ogni probe
      y : np.ndarray di shape (N,) con le label intere dei device

    Le label vengono rimappate in [0, n_classes-1] per comodità con
    PyTorch CrossEntropyLoss e SupCon loss.
    """
    # Converti ogni riga in vettore di feature
    X = np.stack([probe_row_to_vector(row) for _, row in df.iterrows()])

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
      X       : (N, FEATURE_DIM) float32
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
        'feature_dim': FEATURE_DIM,
        'idx_to_label': idx_to_label,
        'label_to_idx': {v: k for k, v in idx_to_label.items()},
    }

    print(f"Feature dim: {FEATURE_DIM} | Classi: {meta['n_classes']}")
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
