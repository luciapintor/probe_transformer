"""
feature_schema.py
=================
Sistema dichiarativo per la definizione delle feature.

Problema che risolve
--------------------
La funzione probe_row_to_vector() originale ha la logica di estrazione
di ogni feature scritta direttamente nel corpo della funzione. Aggiungere
o rimuovere una feature richiede di modificare il codice della funzione,
aggiornare FEATURE_DIM a mano, e ricordarsi l'ordine di concatenazione.

Approccio alternativo: schema dichiarativo
------------------------------------------
Ogni feature (o gruppo di feature) viene descritta da un oggetto
FeatureExtractor che sa:
  - da quale colonna (o colonne) del CSV leggere il valore
  - come trasformarlo in un array numpy
  - quante feature produce (output_dim)

La funzione di conversione diventa un loop generico:

    vector = np.concatenate([f.extract(row) for f in schema])

Aggiungere una feature significa aggiungere un oggetto alla lista.
FEATURE_DIM viene calcolato automaticamente come somma degli output_dim.

Tipi di extractor disponibili
------------------------------
  ScalarExtractor     : colonna scalare numerica, NaN -> sentinella
  BinaryExtractor     : colonna 0/1 o bool
  TanhExtractor       : scalare normalizzato con tanh(x / scale)
  MultiHotExtractor   : colonna lista -> multi-hot su vocabolario fisso
  PaddedListExtractor : colonna lista -> padding a lunghezza fissa con tanh
  SSIDExtractor       : colonna SSID hex -> (presente, lunghezza, hash)
  VendorExtractor     : coppie (oui, type) -> (presente, oui_norm, type_norm)
  CustomExtractor     : funzione arbitraria fornita dall'utente

Esempio di uso
--------------
Definire uno schema personalizzato per un nuovo scenario:

    from feature_schema import (
        BinaryExtractor, MultiHotExtractor, PaddedListExtractor,
        TanhExtractor, FeatureSchema
    )

    MY_SCHEMA = FeatureSchema([
        BinaryExtractor('is_ios'),
        MultiHotExtractor('ie1', vocab=[2, 4, 11, 22, 12, 18, 24, 36]),
        PaddedListExtractor('ie127_0', max_len=10),
        TanhExtractor('ie45_capabilities', scale=255.0),
    ])

    # Converti una riga
    vec = MY_SCHEMA.extract(row)          # np.ndarray di shape (MY_SCHEMA.dim,)

    # Converti un intero DataFrame
    X = MY_SCHEMA.transform(df)           # np.ndarray di shape (N, MY_SCHEMA.dim)

    # Salva/carica lo schema (per usarlo in inferenza con gli stessi parametri)
    MY_SCHEMA.save('schema.json')
    schema = FeatureSchema.load('schema.json')
"""

from __future__ import annotations

import ast
import json
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional


# -----------------------------------------------------------------------
# Classe base
# -----------------------------------------------------------------------

class FeatureExtractor(ABC):
    """
    Classe base per tutti gli extractor.

    Ogni extractor deve implementare:
      - output_dim : int, numero di feature prodotte
      - extract(row) -> np.ndarray di shape (output_dim,)

    Deve anche implementare to_dict() / from_dict() per la
    serializzazione dello schema in JSON.
    """

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Numero di valori float prodotti da questo extractor."""

    @abstractmethod
    def extract(self, row: pd.Series) -> np.ndarray:
        """
        Estrae le feature dalla riga e restituisce un array float32
        di lunghezza self.output_dim.
        """

    @abstractmethod
    def to_dict(self) -> dict:
        """Serializza l'extractor in un dizionario JSON-compatibile."""

    @classmethod
    @abstractmethod
    def from_dict(cls, d: dict) -> FeatureExtractor:
        """Deserializza l'extractor da un dizionario."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dim={self.output_dim})"


# -----------------------------------------------------------------------
# Extractor concreti
# -----------------------------------------------------------------------

class BinaryExtractor(FeatureExtractor):
    """
    Colonna booleana o 0/1. NaN viene trattato come 0.0.

    Produce 1 feature.

    Esempio:
        BinaryExtractor('is_ios')
        -> row['is_ios'] = 1  =>  [1.0]
        -> row['is_ios'] = NaN => [0.0]
    """

    def __init__(self, column: str):
        self.column = column

    @property
    def output_dim(self) -> int:
        return 1

    def extract(self, row: pd.Series) -> np.ndarray:
        v = row.get(self.column, np.nan)
        val = 0.0 if pd.isna(v) else float(bool(v))
        return np.array([val], dtype=np.float32)

    def to_dict(self) -> dict:
        return {'type': 'binary', 'column': self.column}

    @classmethod
    def from_dict(cls, d: dict) -> BinaryExtractor:
        return cls(d['column'])


class ScalarExtractor(FeatureExtractor):
    """
    Colonna scalare numerica con sentinella per NaN.

    Produce 1 feature. Il valore viene restituito as-is (senza normalizzazione).
    Usare TanhExtractor se il range dei valori è ampio.

    Parametri
    ---------
    column    : nome della colonna
    nan_value : valore da usare quando il campo è NaN (default -1.0,
                sentinella distinguibile da zero)

    Esempio:
        ScalarExtractor('ie3', nan_value=-1.0)
        -> row['ie3'] = 6.0  => [6.0]
        -> row['ie3'] = NaN  => [-1.0]
    """

    def __init__(self, column: str, nan_value: float = -1.0):
        self.column = column
        self.nan_value = nan_value

    @property
    def output_dim(self) -> int:
        return 1

    def extract(self, row: pd.Series) -> np.ndarray:
        v = row.get(self.column, np.nan)
        val = self.nan_value if pd.isna(v) else float(v)
        return np.array([val], dtype=np.float32)

    def to_dict(self) -> dict:
        return {'type': 'scalar', 'column': self.column, 'nan_value': self.nan_value}

    @classmethod
    def from_dict(cls, d: dict) -> ScalarExtractor:
        return cls(d['column'], d.get('nan_value', -1.0))


class TanhExtractor(FeatureExtractor):
    """
    Colonna scalare normalizzata con tanh(x / scale).

    Produce 1 feature compressa in (-1, 1).
    Utile quando i valori hanno range ampio e non uniforme (bitmask, OUI...).

    NaN -> nan_value (default -1.0, che è fuori dal range di tanh su valori
    positivi, quindi è riconoscibile come sentinella).

    Parametri
    ---------
    column    : nome della colonna
    scale     : divisore prima di tanh. Scegliere ~ metà del valore massimo
                atteso, così i valori tipici vengono mappati in (-0.76, 0.76)
                lasciando margine per outlier.
    nan_value : sentinella per NaN

    Esempio:
        TanhExtractor('ie45_capabilities', scale=255.0)
        -> row['ie45_capabilities'] = 45.0  => [tanh(45/255)] = [0.173]
        -> row['ie45_capabilities'] = NaN   => [-1.0]
    """

    def __init__(self, column: str, scale: float = 1.0, nan_value: float = -1.0):
        self.column = column
        self.scale = scale
        self.nan_value = nan_value

    @property
    def output_dim(self) -> int:
        return 1

    def extract(self, row: pd.Series) -> np.ndarray:
        v = row.get(self.column, np.nan)
        if pd.isna(v):
            return np.array([self.nan_value], dtype=np.float32)
        return np.array([float(np.tanh(float(v) / self.scale))], dtype=np.float32)

    def to_dict(self) -> dict:
        return {
            'type': 'tanh',
            'column': self.column,
            'scale': self.scale,
            'nan_value': self.nan_value,
        }

    @classmethod
    def from_dict(cls, d: dict) -> TanhExtractor:
        return cls(d['column'], d.get('scale', 1.0), d.get('nan_value', -1.0))


class MultiHotExtractor(FeatureExtractor):
    """
    Colonna che contiene una lista Python (come stringa nel CSV) ->
    vettore multi-hot su un vocabolario fisso.

    Produce len(vocab) feature, più opzionalmente la lunghezza normalizzata
    della lista come feature aggiuntiva (include_length=True).

    Il vocabolario deve essere definito a priori (dall'analisi del dataset).
    Valori nella lista non presenti nel vocabolario vengono ignorati.

    Parametri
    ---------
    column         : nome della colonna
    vocab          : lista ordinata di valori possibili
    include_length : se True, aggiunge 1 feature con len(lista)/max_len
    max_len        : denominatore per la normalizzazione della lunghezza

    Esempio:
        MultiHotExtractor('ie1', vocab=[2, 4, 11, 22], include_length=True, max_len=8)
        -> row['ie1'] = '[2, 11]'  => [1, 0, 1, 0, 0.25]
        -> row['ie1'] = NaN        => [0, 0, 0, 0, 0.0]
    """

    def __init__(
        self,
        column: str,
        vocab: list,
        include_length: bool = True,
        max_len: int = 8,
    ):
        self.column = column
        self.vocab = vocab
        self.include_length = include_length
        self.max_len = max_len
        self._vocab_set = set(vocab)   # per lookup O(1)

    @property
    def output_dim(self) -> int:
        return len(self.vocab) + (1 if self.include_length else 0)

    def extract(self, row: pd.Series) -> np.ndarray:
        raw = row.get(self.column, np.nan)
        values = _parse_list(raw)

        # Multi-hot
        vec = np.zeros(len(self.vocab), dtype=np.float32)
        for v in values:
            if v in self._vocab_set:
                vec[self.vocab.index(v)] = 1.0

        if self.include_length:
            length_feat = np.array(
                [len(values) / max(self.max_len, 1)], dtype=np.float32
            )
            return np.concatenate([vec, length_feat])
        return vec

    def to_dict(self) -> dict:
        return {
            'type': 'multihot',
            'column': self.column,
            'vocab': self.vocab,
            'include_length': self.include_length,
            'max_len': self.max_len,
        }

    @classmethod
    def from_dict(cls, d: dict) -> MultiHotExtractor:
        return cls(
            d['column'], d['vocab'],
            d.get('include_length', True),
            d.get('max_len', 8),
        )


class PaddedListExtractor(FeatureExtractor):
    """
    Colonna lista -> array di lunghezza fissa con padding.

    I valori vengono normalizzati con tanh(x / scale).
    Le posizioni oltre la lunghezza della lista vengono riempite
    con nan_value (default -1.0, sentinella distinguibile da tanh(0)=0).

    Opzionalmente include la lunghezza normalizzata come feature aggiuntiva.

    Parametri
    ---------
    column         : nome della colonna
    max_len        : lunghezza del vettore output (tronca o padda)
    scale          : divisore per la normalizzazione tanh
    nan_value      : valore per le posizioni di padding
    include_length : se True, aggiunge 1 feature con len(lista)/max_len

    Esempio:
        PaddedListExtractor('ie127_0', max_len=5, scale=255.0)
        -> row['ie127_0'] = '[0, 0, 8, 64]'  => [0., 0., 0.031, 0.245, -1.0]
        -> row['ie127_0'] = NaN               => [-1., -1., -1., -1., -1.]
    """

    def __init__(
        self,
        column: str,
        max_len: int,
        scale: float = 255.0,
        nan_value: float = -1.0,
        include_length: bool = True,
    ):
        self.column = column
        self.max_len = max_len
        self.scale = scale
        self.nan_value = nan_value
        self.include_length = include_length

    @property
    def output_dim(self) -> int:
        return self.max_len + (1 if self.include_length else 0)

    def extract(self, row: pd.Series) -> np.ndarray:
        values = _parse_list(row.get(self.column, np.nan))

        vec = np.full(self.max_len, self.nan_value, dtype=np.float32)
        for i, v in enumerate(values[:self.max_len]):
            vec[i] = float(np.tanh(float(v) / self.scale))

        if self.include_length:
            length_feat = np.array([len(values) / self.max_len], dtype=np.float32)
            return np.concatenate([vec, length_feat])
        return vec

    def to_dict(self) -> dict:
        return {
            'type': 'padded_list',
            'column': self.column,
            'max_len': self.max_len,
            'scale': self.scale,
            'nan_value': self.nan_value,
            'include_length': self.include_length,
        }

    @classmethod
    def from_dict(cls, d: dict) -> PaddedListExtractor:
        return cls(
            d['column'], d['max_len'],
            d.get('scale', 255.0),
            d.get('nan_value', -1.0),
            d.get('include_length', True),
        )


class MultiScalarExtractor(FeatureExtractor):
    """
    Gruppo di colonne scalari numeriche trattate insieme.
    Equivale a len(columns) TanhExtractor in sequenza, ma più comodo
    quando le colonne condividono la stessa logica di normalizzazione
    (es. tutti i sottocampi di ie45_*).

    Produce len(columns) feature.

    Parametri
    ---------
    columns   : lista di nomi di colonne
    scale     : divisore per tanh (uguale per tutte le colonne)
    nan_value : sentinella per NaN (uguale per tutte le colonne)

    Esempio:
        MultiScalarExtractor(
            ['ie45_ampduparam', 'ie45_asel', 'ie45_capabilities'],
            scale=255.0
        )
        -> [tanh(v0/255), tanh(v1/255), tanh(v2/255)]
           con -1.0 dove il campo è NaN
    """

    def __init__(
        self,
        columns: list[str],
        scale: float = 255.0,
        nan_value: float = -1.0,
    ):
        self.columns = columns
        self.scale = scale
        self.nan_value = nan_value

    @property
    def output_dim(self) -> int:
        return len(self.columns)

    def extract(self, row: pd.Series) -> np.ndarray:
        vec = np.zeros(len(self.columns), dtype=np.float32)
        for i, col in enumerate(self.columns):
            v = row.get(col, np.nan)
            vec[i] = self.nan_value if pd.isna(v) else float(np.tanh(float(v) / self.scale))
        return vec

    def to_dict(self) -> dict:
        return {
            'type': 'multi_scalar',
            'columns': self.columns,
            'scale': self.scale,
            'nan_value': self.nan_value,
        }

    @classmethod
    def from_dict(cls, d: dict) -> MultiScalarExtractor:
        return cls(d['columns'], d.get('scale', 255.0), d.get('nan_value', -1.0))


class SSIDExtractor(FeatureExtractor):
    """
    Colonna SSID in formato hex separato da ':' (es. "43:61:73:65...").

    Produce 3 feature:
      [0] presente/assente (0.0 o 1.0)
      [1] lunghezza SSID normalizzata (0..1, max 32 caratteri)
      [2] hash dell'SSID decodificato normalizzato in [-1, 1]

    NaN o stringa vuota -> [0.0, 0.0, 0.0]

    Nota: l'hash è deterministico solo all'interno della stessa sessione
    Python (hash() usa un seed casuale per sicurezza). Per riproducibilità
    cross-sessione usare hashlib invece di hash().
    """

    def __init__(self, column: str = 'ie0', max_ssid_len: int = 32):
        self.column = column
        self.max_ssid_len = max_ssid_len

    @property
    def output_dim(self) -> int:
        return 3

    def extract(self, row: pd.Series) -> np.ndarray:
        raw = row.get(self.column, np.nan)
        ssid = _decode_ssid_hex(raw)
        if ssid is None:
            return np.zeros(3, dtype=np.float32)
        h = hash(ssid) & 0xFFFFFFFF
        return np.array([
            1.0,
            min(len(ssid), self.max_ssid_len) / self.max_ssid_len,
            (h / 0xFFFFFFFF) * 2 - 1,
        ], dtype=np.float32)

    def to_dict(self) -> dict:
        return {'type': 'ssid', 'column': self.column, 'max_ssid_len': self.max_ssid_len}

    @classmethod
    def from_dict(cls, d: dict) -> SSIDExtractor:
        return cls(d.get('column', 'ie0'), d.get('max_ssid_len', 32))


class VendorExtractor(FeatureExtractor):
    """
    Coppie di colonne (oui_col, type_col) per vendor specific elements.

    Per ogni coppia produce 3 feature:
      [0] presente/assente
      [1] OUI normalizzato con tanh(oui / oui_scale)
      [2] type normalizzato con tanh(type / type_scale)

    Produce n_slots * 3 feature totali.

    Parametri
    ---------
    oui_columns  : lista di nomi colonne OUI (es. ['ie221_oui_0', 'ie221_oui_1'])
    type_columns : lista di nomi colonne type (stessa lunghezza di oui_columns)
    oui_scale    : scala per normalizzazione OUI (default 1e6)
    type_scale   : scala per normalizzazione type (default 10.0)

    Esempio:
        VendorExtractor(
            oui_columns=['ie221_oui_0', 'ie221_oui_1'],
            type_columns=['ie221_type_0', 'ie221_type_1'],
        )
        -> 6 feature: [pres_0, oui_0, type_0, pres_1, oui_1, type_1]
    """

    def __init__(
        self,
        oui_columns: list[str],
        type_columns: list[str],
        oui_scale: float = 1e6,
        type_scale: float = 10.0,
    ):
        assert len(oui_columns) == len(type_columns), (
            "oui_columns e type_columns devono avere la stessa lunghezza"
        )
        self.oui_columns = oui_columns
        self.type_columns = type_columns
        self.oui_scale = oui_scale
        self.type_scale = type_scale

    @property
    def output_dim(self) -> int:
        return len(self.oui_columns) * 3

    def extract(self, row: pd.Series) -> np.ndarray:
        vec = np.zeros(self.output_dim, dtype=np.float32)
        for i, (oui_col, type_col) in enumerate(zip(self.oui_columns, self.type_columns)):
            oui = row.get(oui_col, np.nan)
            typ = row.get(type_col, np.nan)
            base = i * 3
            if pd.isna(oui):
                vec[base:base+3] = -1.0   # slot assente
            else:
                vec[base]   = 1.0
                vec[base+1] = float(np.tanh(float(oui) / self.oui_scale))
                vec[base+2] = float(np.tanh(float(typ) / self.type_scale)) if not pd.isna(typ) else -1.0
        return vec

    def to_dict(self) -> dict:
        return {
            'type': 'vendor',
            'oui_columns': self.oui_columns,
            'type_columns': self.type_columns,
            'oui_scale': self.oui_scale,
            'type_scale': self.type_scale,
        }

    @classmethod
    def from_dict(cls, d: dict) -> VendorExtractor:
        return cls(
            d['oui_columns'], d['type_columns'],
            d.get('oui_scale', 1e6), d.get('type_scale', 10.0),
        )


class CustomExtractor(FeatureExtractor):
    """
    Extractor con funzione arbitraria fornita dall'utente.

    Utile per trasformazioni che non rientrano negli altri tipi.
    NON serializzabile in JSON (la funzione non può essere salvata).
    Per la serializzazione, implementare un extractor dedicato.

    Parametri
    ---------
    fn        : funzione (row: pd.Series) -> np.ndarray di shape (dim,)
    dim       : output_dim della funzione
    name      : nome descrittivo (usato solo in __repr__)

    Esempio:
        def my_fn(row):
            v = row.get('my_col', 0)
            return np.array([v * 2, v ** 2], dtype=np.float32)

        CustomExtractor(my_fn, dim=2, name='my_col_squared')
    """

    def __init__(self, fn: Callable[[pd.Series], np.ndarray], dim: int, name: str = 'custom'):
        self.fn = fn
        self._dim = dim
        self.name = name

    @property
    def output_dim(self) -> int:
        return self._dim

    def extract(self, row: pd.Series) -> np.ndarray:
        result = self.fn(row)
        assert len(result) == self._dim, (
            f"CustomExtractor '{self.name}': attesi {self._dim} valori, "
            f"ricevuti {len(result)}"
        )
        return result.astype(np.float32)

    def to_dict(self) -> dict:
        raise NotImplementedError(
            "CustomExtractor non è serializzabile. "
            "Implementa un extractor dedicato con to_dict() e from_dict()."
        )

    @classmethod
    def from_dict(cls, d: dict) -> CustomExtractor:
        raise NotImplementedError("CustomExtractor non è deserializzabile da JSON.")

    def __repr__(self) -> str:
        return f"CustomExtractor(name='{self.name}', dim={self._dim})"


# -----------------------------------------------------------------------
# FeatureSchema: contenitore ordinato di extractor
# -----------------------------------------------------------------------


class VhtExtractor(FeatureExtractor):
    """
    Colonna VHT Capabilities (ie191): scalare float o NaN.

    Produce 2 feature:
      [0] presente/assente (1.0 o -1.0)
      [1] valore normalizzato con tanh(x / scale), oppure -1.0 se assente

    La sentinella -1.0 su entrambe le feature distingue il caso
    "assente" da qualsiasi valore reale (tanh e' sempre in (-1,1)).

    Parametri
    ---------
    column : nome della colonna (default 'ie191')
    scale  : divisore per tanh (default 1e9, ordine di grandezza di ie191)
    """

    def __init__(self, column: str = 'ie191', scale: float = 1e9):
        self.column = column
        self.scale = scale

    @property
    def output_dim(self) -> int:
        return 2

    def extract(self, row: pd.Series) -> np.ndarray:
        v = row.get(self.column, np.nan)
        if pd.isna(v):
            return np.array([-1.0, -1.0], dtype=np.float32)
        return np.array([1.0, float(np.tanh(float(v) / self.scale))], dtype=np.float32)

    def to_dict(self) -> dict:
        return {'type': 'vht', 'column': self.column, 'scale': self.scale}

    @classmethod
    def from_dict(cls, d: dict) -> VhtExtractor:
        return cls(d.get('column', 'ie191'), d.get('scale', 1e9))

# Registro dei tipi di extractor per la deserializzazione da JSON
_EXTRACTOR_REGISTRY: dict[str, type] = {
    'binary':       BinaryExtractor,
    'scalar':       ScalarExtractor,
    'tanh':         TanhExtractor,
    'multihot':     MultiHotExtractor,
    'padded_list':  PaddedListExtractor,
    'multi_scalar': MultiScalarExtractor,
    'ssid':         SSIDExtractor,
    'vendor':       VendorExtractor,
    'vht':          VhtExtractor,
}


class FeatureSchema:
    """
    Contenitore ordinato di FeatureExtractor.

    Uso principale
    --------------
        schema = FeatureSchema([
            BinaryExtractor('is_ios'),
            SSIDExtractor('ie0'),
            MultiHotExtractor('ie1', vocab=[2, 4, 11, 22]),
            ...
        ])

        vec = schema.extract(row)      # (dim,) array
        X   = schema.transform(df)     # (N, dim) array
        schema.save('schema.json')
        schema2 = FeatureSchema.load('schema.json')

    Attributi
    ---------
    dim      : dimensione totale del vettore output (calcolata automaticamente)
    groups   : lista di (nome, slice) per identificare i gruppi nel vettore
               (compatibile con FEATURE_GROUPS in model.py)
    """

    def __init__(self, extractors: list[FeatureExtractor], names: Optional[list[str]] = None):
        """
        Parametri
        ---------
        extractors : lista ordinata di FeatureExtractor
        names      : nomi opzionali per ogni extractor (usati in groups e repr).
                     Se None, usa il tipo + indice (es. 'BinaryExtractor_0').
        """
        self.extractors = extractors

        # Nomi degli extractor (per debug e per FEATURE_GROUPS)
        if names is None:
            names = [
                f"{type(e).__name__}_{i}" for i, e in enumerate(extractors)
            ]
        self.names = names

        # Calcola la dimensione totale e i gruppi (slice) di ogni extractor
        self._dim = sum(e.output_dim for e in extractors)
        self._groups = self._build_groups()

    def _build_groups(self) -> list[tuple[str, slice]]:
        """
        Costruisce la lista (nome, slice) compatibile con FEATURE_GROUPS
        in model.py, così il transformer sa dove inizia e finisce ogni
        gruppo di feature nel vettore flat.
        """
        groups = []
        offset = 0
        for name, extractor in zip(self.names, self.extractors):
            end = offset + extractor.output_dim
            groups.append((name, slice(offset, end)))
            offset = end
        return groups

    @property
    def dim(self) -> int:
        """Dimensione totale del vettore prodotto dallo schema."""
        return self._dim

    @property
    def groups(self) -> list[tuple[str, slice]]:
        """
        Lista (nome, slice) per ogni extractor.
        Passare come FEATURE_GROUPS in model.py per aggiornare
        automaticamente i token del transformer.
        """
        return self._groups

    def extract(self, row: pd.Series) -> np.ndarray:
        """
        Converte una singola riga in un vettore flat di shape (dim,).
        Equivale alla vecchia probe_row_to_vector().
        """
        parts = [e.extract(row) for e in self.extractors]
        return np.concatenate(parts)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Converte un intero DataFrame in una matrice (N, dim).
        Equivale alla vecchia np.stack([probe_row_to_vector(row) for ...]).
        """
        return np.stack([self.extract(row) for _, row in df.iterrows()])

    def summary(self) -> str:
        """
        Stampa la composizione del vettore, utile per documentare lo schema.

        Esempio di output:
            Schema (dim=87):
              [  0:  1]  BinaryExtractor_0        (1)   is_ios
              [  1:  4]  SSIDExtractor_1           (3)   ie0
              ...
        """
        lines = [f"Schema (dim={self.dim}):"]
        for (name, sl), extractor in zip(self._groups, self.extractors):
            lines.append(
                f"  [{sl.start:3d}:{sl.stop:3d}]  "
                f"{type(extractor).__name__:<25s} ({extractor.output_dim:2d})  "
                f"{name}"
            )
        return "\n".join(lines)

    def save(self, path: str) -> None:
        """
        Serializza lo schema in un file JSON.
        Gli extractor di tipo Custom non sono serializzabili e causano errore.
        """
        data = {
            'names': self.names,
            'extractors': [e.to_dict() for e in self.extractors],
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> FeatureSchema:
        """
        Carica uno schema da un file JSON salvato con save().
        """
        with open(path) as f:
            data = json.load(f)

        extractors = []
        for d in data['extractors']:
            type_key = d['type']
            if type_key not in _EXTRACTOR_REGISTRY:
                raise ValueError(
                    f"Tipo di extractor sconosciuto: '{type_key}'. "
                    f"Tipi disponibili: {list(_EXTRACTOR_REGISTRY.keys())}"
                )
            extractors.append(_EXTRACTOR_REGISTRY[type_key].from_dict(d))

        return cls(extractors, names=data.get('names'))

    def __repr__(self) -> str:
        return self.summary()


# -----------------------------------------------------------------------
# Funzioni di utilità interne (usate dagli extractor)
# -----------------------------------------------------------------------

def _parse_list(value: Any) -> list:
    """
    Converte una stringa come "[2, 4, 11, 22]" in lista Python.
    Restituisce lista vuota se il valore è NaN o non parsabile.
    """
    if pd.isna(value) if not isinstance(value, (list, str)) else False:
        return []
    if isinstance(value, list):
        return value
    try:
        result = ast.literal_eval(str(value))
        return result if isinstance(result, list) else []
    except (ValueError, SyntaxError):
        return []


def _decode_ssid_hex(raw: Any) -> Optional[str]:
    """
    Decodifica un SSID in formato hex separato da ':'.
    Restituisce None se il valore è NaN, vuoto o non decodificabile.
    """
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None
    if isinstance(raw, str) and raw == '':
        return None
    try:
        return bytes.fromhex(str(raw).replace(':', '')).decode('utf-8', errors='replace')
    except Exception:
        return None


# -----------------------------------------------------------------------
# Schema predefinito per il dataset probe request (all_A_full.csv)
# -----------------------------------------------------------------------

def build_probe_schema() -> FeatureSchema:
    """
    Costruisce lo schema corrispondente al preprocessing originale
    del dataset all_A_full.csv (87 feature, stessa struttura di
    probe_row_to_vector() in preprocessing.py).

    Usare questa funzione come punto di partenza per creare varianti:

        schema = build_probe_schema()

        # Variante senza SSID:
        schema_no_ssid = FeatureSchema(
            [e for e in schema.extractors if not isinstance(e, SSIDExtractor)],
        )
    """
    return FeatureSchema(
        extractors=[
            # [A] Flag sistema operativo (1 feature)
            BinaryExtractor('is_ios'),

            # [B] SSID: presente/assente, lunghezza, hash (3 feature)
            SSIDExtractor('ie0', max_ssid_len=32),

            # [C,D] Supported Rates: multi-hot + lunghezza (17 feature)
            MultiHotExtractor(
                'ie1',
                vocab=sorted({2, 4, 11, 12, 18, 22, 24, 36, 48, 72, 96, 108, 130, 132, 139, 150}),
                include_length=True,
                max_len=8,
            ),

            # [E,F,G] Extended Supported Rates: multi-hot + presente + lunghezza (10 feature)
            MultiHotExtractor(
                'ie50',
                vocab=sorted({12, 18, 24, 36, 48, 72, 96, 108}),
                include_length=True,
                max_len=8,
            ),

            # [H,I] Extended Capabilities blocco 0: padded + lunghezza (11 feature)
            PaddedListExtractor('ie127_0', max_len=10, scale=255.0, include_length=True),

            # [J] Extended Capabilities blocco 1: padded senza lunghezza (9 feature)
            PaddedListExtractor('ie127_1', max_len=9, scale=255.0, include_length=False),

            # [K] HT Capabilities: 17 sottocampi scalari (17 feature)
            MultiScalarExtractor(
                columns=[
                    'ie45_ampduparam', 'ie45_asel', 'ie45_capabilities', 'ie45_txbf',
                    'ie45_mcsset_txunequalmod', 'ie45_mcsset_txrxmcsnotequal',
                    'ie45_mcsset_txmaxss', 'ie45_mcsset_txsetdefined',
                    'ie45_mcsset_highestdatarate',
                    'ie45_rxbitmask_0to7', 'ie45_rxbitmask_8to15',
                    'ie45_rxbitmask_16to23', 'ie45_rxbitmask_24to31',
                    'ie45_rxbitmask_32', 'ie45_rxbitmask_33to38',
                    'ie45_rxbitmask_39to52', 'ie45_rxbitmask_53to76',
                ],
                scale=255.0,
            ),

            # [L] Vendor Specific: 4 slot OUI+type (12 feature)
            VendorExtractor(
                oui_columns=['ie221_oui_0', 'ie221_oui_1', 'ie221_oui_2', 'ie221_oui_3'],
                type_columns=['ie221_type_0', 'ie221_type_1', 'ie221_type_2', 'ie221_type_3'],
                oui_scale=1e6,
                type_scale=10.0,
            ),

            # [M,N] VHT Capabilities: presente + valore (2 feature)
            # ie191 assente -> [-1, -1]; presente -> [1, tanh(val/1e9)]
            VhtExtractor('ie191', scale=1e9),

            # [O] Interworking: 5 sottocampi scalari (5 feature)
            MultiScalarExtractor(
                columns=[
                    'ie107_access_network_type', 'ie107_asra',
                    'ie107_internet', 'ie107_esr', 'ie107_uesa',
                ],
                scale=10.0,
            ),
        ],
        names=[
            'ios_ssid_flag', 'ssid', 'ie1_rates', 'ie50_ext_rates',
            'ie127_0_ext_cap', 'ie127_1_ext_cap',
            'ie45_ht_cap', 'ie221_vendor', 'ie191_vht_cap', 'ie107_interworking',
        ],
    )


# -----------------------------------------------------------------------
# Feature-level tokenization
# -----------------------------------------------------------------------

class FeatureToken:
    """
    Descrive un singolo token a livello di feature singola.

    Ogni feature scalare nel vettore flat diventa un token separato
    per il transformer. Questo oggetto porta con sé:
      - il nome semantico della feature (es. 'ie45_ht_cap__feat_2')
      - la posizione (slice) nel vettore flat
      - il nome dell'IE di appartenenza (es. 'ie45_ht_cap')

    Il nome semantico è usato dal token_type_embed in model.py:
    invece di un positional embedding (che cambierebbe se lo schema
    cambia), ogni feature riceve un embedding basato sul proprio nome,
    che rimane stabile indipendentemente da quante altre feature ci sono.

    Convenzione di naming:
      '{ie_name}__{feat_idx}'
    dove feat_idx è l'indice della feature all'interno del gruppo IE.
    Esempio: 'ie45_ht_cap__0', 'ie45_ht_cap__1', ..., 'ie45_ht_cap__16'
    """

    def __init__(self, name: str, position: int, ie_name: str):
        """
        name     : nome semantico univoco della feature token
        position : indice nel vettore flat (corrisponde a slice(position, position+1))
        ie_name  : nome dell'IE (extractor) di appartenenza
        """
        self.name     = name
        self.position = position
        self.ie_name  = ie_name

    def __repr__(self) -> str:
        return f"FeatureToken('{self.name}', pos={self.position}, ie='{self.ie_name}')"


# Aggiunta alla classe FeatureSchema (monkey-patching per non riscrivere il file)
def _feature_tokens(self) -> list:
    """
    Restituisce la lista di FeatureToken, uno per ogni feature scalare
    nel vettore flat, in ordine di posizione.

    Ogni token ha un nome semantico costruito come '{ie_name}__{feat_idx}'.
    Questo permette al transformer di ricevere ogni feature come token
    separato, con un embedding semantico basato sul nome invece che
    sulla posizione.

    Esempio per uno schema con ie1 (17 feature) e ie45 (17 feature):
        [
          FeatureToken('ie1_rates__0',  pos=0,  ie='ie1_rates'),
          FeatureToken('ie1_rates__1',  pos=1,  ie='ie1_rates'),
          ...
          FeatureToken('ie1_rates__16', pos=16, ie='ie1_rates'),
          FeatureToken('ie45_ht_cap__0', pos=17, ie='ie45_ht_cap'),
          ...
        ]

    I token di IE mancanti (assenti nel dataset) vengono comunque
    inclusi nella lista ma contrassegnati come missing in forward(),
    usando il missing_token learnable del modello.
    """
    tokens = []
    for ie_name, (_, sl) in zip(self.names, self._groups):
        n_feats = sl.stop - sl.start
        for feat_idx in range(n_feats):
            tokens.append(FeatureToken(
                name     = f"{ie_name}__{feat_idx}",
                position = sl.start + feat_idx,
                ie_name  = ie_name,
            ))
    return tokens

FeatureSchema.feature_tokens = _feature_tokens


def _from_dict_schema(cls, d: dict, names: list[str] | None = None) -> "FeatureSchema":
    """
    Costruttore alternativo da dizionario {nome: extractor}.

    L'ordine degli IE nel dizionario (Python 3.7+ garantisce inserimento)
    determina l'ordine delle feature nel vettore flat e dei token
    nella sequenza del transformer.

    Esempio:
        schema = FeatureSchema.from_ie_dict({
            'ie1':  MultiHotExtractor('ie1', vocab=[2, 4, 11, 22]),
            'ie45': MultiScalarExtractor(['ie45_capabilities'], scale=255.0),
            'ie221': VendorExtractor(...),
        })
    """
    extractors = list(d.values())
    names      = list(d.keys())
    return cls(extractors, names=names)

FeatureSchema.from_ie_dict = classmethod(lambda cls, d: _from_dict_schema(cls, d))