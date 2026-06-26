"""
model.py
========
Encoder transformer con feature-level tokenization.

Idea
----
Invece di raggruppare le feature per IE (un token per IE), ogni singola
feature scalare nel vettore flat diventa un token separato nella sequenza
del transformer. Con 86 feature si hanno 86 token.

Vantaggi rispetto alla group-level tokenization:
  - L'attention opera direttamente tra feature individuali, senza la
    mediazione del raggruppamento. Il modello può imparare correlazioni
    come "ie1_rates__3 e ie45_ht_cap__0 tendono a co-occorrere nei
    device Apple" senza dover prima aggregare all'interno del gruppo.
  - La granularità è massima: nessuna perdita di informazione nel
    passaggio group_dim -> d_model.

Svantaggi:
  - Sequenza più lunga (86 token invece di 10): più operazioni di attention,
    ma con d_model piccolo e num_layers basso rimane gestibile.
  - Meno interpretabile: è più difficile leggere i pesi di attention
    quando i token sono singole feature numeriche.

Token type embedding (per nome, non per posizione)
--------------------------------------------------
Ogni feature ha un nome semantico costruito come '{ie_name}__{feat_idx}'
(es. 'ie45_ht_cap__2'). Il modello impara un embedding vettoriale per
ogni nome, così:
  - 'ie45_ht_cap__2' riceve sempre lo stesso embedding indipendentemente
    da quante altre feature ci sono (schema variabile tra dataset).
  - Se un IE è assente nel dataset, le sue feature vengono sostituite
    dal missing_token learnable, che il modello impara a interpretare
    come "questa feature non era disponibile".

IE di appartenenza come embedding aggiuntivo (ie_type_embed)
-------------------------------------------------------------
Oltre al token embedding per nome, ogni feature riceve anche un
embedding che segnala il suo IE di appartenenza (es. 'ie45_ht_cap').
Questo aiuta il modello a raggruppare implicitamente le feature dello
stesso IE senza imporre un raggruppamento esplicito nella sequenza.
L'embedding finale di ogni token è:

    token = proj(feature_value) + token_name_embed + ie_type_embed

Schema del forward pass:
  x (B, n_active_features)
  -> per ogni feature nel vocabolario globale:
       se attiva: Linear(1, d_model)(x[:, pos])
       se assente: missing_token
  -> (B, n_global_features, d_model)
  -> + token_name_embed[nome] + ie_type_embed[ie_name]
  -> TransformerEncoder
  -> mean pooling o CLS
  -> projection head
  -> L2 normalize -> z (B, embed_dim)

Gestione degli IE mancanti
---------------------------
Come nel precedente modello a gruppi, `active_ie_names` in forward()
specifica quali IE sono presenti nel batch. Le feature degli IE assenti
ricevono il missing_token. L'input x deve contenere solo le feature
degli IE attivi, nell'ordine del vocabolario globale.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from feature_schema import FeatureSchema, FeatureToken


@dataclass
class TransformerConfig:
    """
    Iperparametri dell'architettura.

    d_model         : dimensione interna. Divisibile per nhead.
                      Con feature-level tokenization e 86 token,
                      valori più piccoli (64-128) sono sufficienti.
    nhead           : attention head. Con 86 token, 4-8 head funzionano bene.
    num_layers      : layer del transformer. 2-4 per sequenze corte.
    dim_feedforward : FF interno (convenzione: 2x-4x d_model).
    embed_dim       : dimensione embedding finale z.
    dropout         : dropout nel transformer e nel projection head.
    pooling         : "mean" (default) o "cls".
    """
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 3
    dim_feedforward: int = 256
    embed_dim: int = 64
    dropout: float = 0.1
    pooling: str = "mean"


class ProbeEncoder(nn.Module):
    """
    Encoder con feature-level tokenization e vocabolario globale.

    Ogni feature scalare nel vettore flat è un token separato.
    Il modello viene costruito su un vocabolario globale di tutte
    le feature possibili. In inferenza, le feature degli IE mancanti
    vengono sostituite dal missing_token learnable.

    Costruzione
    -----------
    Richiede lo schema globale (o la lista di FeatureToken globali)
    per costruire:
      - una Linear(1, d_model) per ogni feature nel vocabolario
      - il token_name_embed (un vettore per nome feature)
      - il ie_type_embed (un vettore per nome IE)

    Parametri
    ---------
    config         : TransformerConfig
    global_schema  : FeatureSchema che definisce il vocabolario completo.
                     Il modello usa global_schema.feature_tokens() per
                     ricavare la lista di tutti i token possibili.
    global_tokens  : lista di FeatureToken se global_schema è None
                     (per ricostruire da checkpoint senza schema).
    """

    def __init__(
        self,
        config: TransformerConfig,
        global_schema: "FeatureSchema | None" = None,
        global_tokens: list | None = None,
    ):
        super().__init__()
        self.config = config

        # --- Vocabolario globale di FeatureToken ---
        if global_schema is not None:
            self._global_tokens: list = global_schema.feature_tokens()
        elif global_tokens is not None:
            self._global_tokens = global_tokens
        else:
            raise ValueError("Fornire 'global_schema' oppure 'global_tokens'.")

        n_global = len(self._global_tokens)

        # Indici rapidi per nome feature e nome IE
        self._feat_name_to_idx: dict[str, int] = {
            t.name: i for i, t in enumerate(self._global_tokens)
        }
        # Vocabolario degli IE: nomi unici nell'ordine di prima comparsa
        seen_ie: dict[str, int] = {}
        for t in self._global_tokens:
            if t.ie_name not in seen_ie:
                seen_ie[t.ie_name] = len(seen_ie)
        self._ie_name_to_idx: dict[str, int] = seen_ie

        # --- Proiezione per ogni feature: Linear(1, d_model) ---
        # Una Linear separata per ogni feature permette al modello di
        # imparare una scala e un bias specifici per ciascuna.
        # Con 86 feature sono 86 Linear(1, d_model): piccole ma tante.
        # Alternativa più compatta: un unico Linear(1, d_model) condiviso
        # + token_name_embed per differenziare. Scegliamo Linear separate
        # perché le feature hanno scale molto diverse (binarie, tanh, hash).
        self.feat_projs = nn.ModuleList([
            nn.Linear(1, config.d_model)
            for _ in range(n_global)
        ])

        # --- missing_token ---
        # Vettore learnable (d_model,) per le feature degli IE assenti.
        # Condiviso tra tutte le feature mancanti (un unico segnale di assenza).
        self.missing_token = nn.Parameter(torch.zeros(config.d_model))

        # --- CLS token (opzionale) ---
        if config.pooling == "cls":
            self.cls_token = nn.Parameter(torch.randn(1, 1, config.d_model))

        # --- Token name embedding ---
        # Un vettore per ogni nome feature (es. 'ie45_ht_cap__2').
        # Stabile al cambiamento di schema: 'ie45_ht_cap__2' riceve
        # sempre lo stesso embedding indipendentemente dalla posizione.
        n_embed_names = n_global + (1 if config.pooling == "cls" else 0)
        self._embed_names: list[str] = (
            ["__cls__"] if config.pooling == "cls" else []
        ) + [t.name for t in self._global_tokens]
        self._embed_name_to_idx: dict[str, int] = {
            name: i for i, name in enumerate(self._embed_names)
        }
        self.token_name_embed = nn.Embedding(n_embed_names, config.d_model)

        # --- IE type embedding ---
        # Un vettore per ogni IE (es. 'ie45_ht_cap').
        # Tutte le feature dello stesso IE condividono questo embedding,
        # che aiuta il modello a raggruppare implicitamente le feature
        # correlate senza imporre un raggruppamento esplicito.
        # Il CLS token usa l'IE riservato '__cls__'.
        ie_vocab = (
            {"__cls__": 0} if config.pooling == "cls" else {}
        )
        for ie_name, idx in self._ie_name_to_idx.items():
            ie_vocab[ie_name] = idx + (1 if config.pooling == "cls" else 0)
        self._ie_vocab: dict[str, int] = ie_vocab
        self.ie_type_embed = nn.Embedding(len(ie_vocab), config.d_model)

        # --- Transformer Encoder ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers,
            enable_nested_tensor=False,
        )

        # --- Projection head ---
        self.proj_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.embed_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
        nn.init.zeros_(self.missing_token)

    # ------------------------------------------------------------------
    # Proprietà
    # ------------------------------------------------------------------

    @property
    def n_global_features(self) -> int:
        """Numero di feature nel vocabolario globale (= lunghezza sequenza)."""
        return len(self._global_tokens)

    @property
    def global_dim(self) -> int:
        """Dimensione del vettore con TUTTE le feature attive."""
        return len(self._global_tokens)

    @property
    def global_ie_names(self) -> list[str]:
        """Nomi degli IE nel vocabolario globale, in ordine canonico."""
        return list(self._ie_name_to_idx.keys())

    def active_dim(self, active_ie_names: list[str]) -> int:
        """
        Calcola la dimensione del vettore di input per un sottoinsieme
        di IE attivi. Utile per verificare l'input prima di forward().
        """
        return sum(
            1 for t in self._global_tokens
            if t.ie_name in set(active_ie_names)
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        active_ie_names: list[str] | None = None,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Parametri
        ---------
        x : (B, n_active_features) float32
            Vettore con le feature degli IE attivi, nell'ordine in cui
            compaiono nel vocabolario globale (non nell'ordine di
            active_ie_names). Ogni feature è uno scalare.
            Se active_ie_names è None, x deve contenere tutte le feature
            (x.size(1) == n_global_features).

        active_ie_names : lista dei nomi degli IE presenti nel batch.
            None = tutti gli IE del vocabolario sono presenti.
            Esempio: ['ie1_rates', 'ie45_ht_cap'] se gli altri IE mancano.

        normalize : L2-normalizza l'output se True.

        Restituisce
        -----------
        z : (B, embed_dim)
        """
        B = x.size(0)

        if active_ie_names is None:
            active_ie_set = set(self._ie_name_to_idx.keys())
        else:
            active_ie_set = set(active_ie_names)

        # ------------------------------------------------------------------
        # Costruisce la sequenza di token in ordine canonico del vocabolario
        # ------------------------------------------------------------------
        # Per ogni feature nel vocabolario globale:
        #   - se il suo IE è attivo: proietta il valore scalare con feat_projs[i]
        #   - se il suo IE è assente: usa missing_token
        #
        # Le feature in x sono nell'ordine canonico (solo quelle degli IE attivi).
        # x_offset scorre x leggendo una feature alla volta.

        token_list = []
        x_offset = 0

        for i, feat_token in enumerate(self._global_tokens):
            if feat_token.ie_name in active_ie_set:
                # Feature attiva: legge il valore scalare e lo proietta
                # x[:, x_offset] ha shape (B,); unsqueeze(-1) -> (B, 1)
                feat_val = x[:, x_offset].unsqueeze(-1)       # (B, 1)
                token    = self.feat_projs[i](feat_val)        # (B, d_model)
                x_offset += 1
            else:
                # Feature assente: usa missing_token per tutte le feature
                # di questo IE (non solo per l'IE nel suo insieme)
                token = self.missing_token.unsqueeze(0).expand(B, -1)  # (B, d_model)

            token_list.append(token)

        assert x_offset == x.size(1), (
            f"x ha {x.size(1)} feature ma gli IE attivi ne richiedono {x_offset}. "
            f"Verificare che active_ie_names corrisponda alle colonne in x "
            f"e che le feature siano nell'ordine del vocabolario globale."
        )

        tokens = torch.stack(token_list, dim=1)   # (B, n_global, d_model)

        # ------------------------------------------------------------------
        # CLS token (opzionale)
        # ------------------------------------------------------------------
        if self.config.pooling == "cls":
            cls    = self.cls_token.expand(B, -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)

        # ------------------------------------------------------------------
        # Token name embedding + IE type embedding
        # ------------------------------------------------------------------
        # Ogni token riceve la somma di due embedding:
        #   1. token_name_embed: specifico per questa feature ('ie45_ht_cap__2')
        #   2. ie_type_embed: condiviso tra tutte le feature dello stesso IE
        #
        # La somma permette al modello di distinguere le feature sia
        # individualmente che per appartenenza all'IE.

        embed_indices = torch.tensor(
            [self._embed_name_to_idx[n] for n in self._embed_names],
            dtype=torch.long, device=x.device,
        ).unsqueeze(0)   # (1, seq_len)

        ie_indices = torch.tensor(
            [self._ie_vocab.get("__cls__", 0)] * (1 if self.config.pooling == "cls" else 0)
            + [self._ie_vocab[t.ie_name] for t in self._global_tokens],
            dtype=torch.long, device=x.device,
        ).unsqueeze(0)   # (1, seq_len)

        tokens = (
            tokens
            + self.token_name_embed(embed_indices)
            + self.ie_type_embed(ie_indices)
        )

        # ------------------------------------------------------------------
        # Transformer + pooling + projection
        # ------------------------------------------------------------------
        out = self.transformer(tokens)   # (B, seq_len, d_model)

        if self.config.pooling == "cls":
            pooled = out[:, 0, :]
        else:
            pooled = out.mean(dim=1)

        z = self.proj_head(pooled)
        if normalize:
            z = F.normalize(z, dim=-1)
        return z