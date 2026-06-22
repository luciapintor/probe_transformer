"""
model.py
========
Encoder transformer per probe request Wi-Fi.

Idea di base
------------
Ogni probe è un vettore flat di FEATURE_DIM (79) feature numeriche.
Vogliamo un encoder f(x) -> z tale che:
  - z_i ~= z_j  se probe i e j provengono dallo stesso device
  - z_i far da z_j  se provengono da device diversi

L'encoder viene addestrato con Supervised Contrastive Loss, che
usa le label reali per definire quali coppie sono positive.

Architettura: "Feature Group Tokenizer + Transformer Encoder"
-------------------------------------------------------------
Invece di trattare il vettore flat come un singolo token (che darebbe
al transformer nessuna struttura su cui ragionare con l'attention),
dividiamo le 79 feature in gruppi logici basati sull'IE di provenienza.
Ogni gruppo diventa un token separato.

Questo permette all'attention di imparare QUALI IE sono più informativi
per distinguere i device, e come i diversi IE si relazionano tra loro.

Gruppi di token:
  token 0 : is_ios + SSID features       (4 feature)
  token 1 : ie1  supported rates         (17 feature: 16 multi-hot + lunghezza)
  token 2 : ie50 extended rates          (10 feature: 8 multi-hot + presente + lunghezza)
  token 3 : ie127 extended capabilities  (12 feature: 10 padded + lunghezza + ie127_1)
  token 4 : ie45  HT capabilities        (17 feature)
  token 5 : ie221 vendor specific        (12 feature)
  token 6 : ie191 VHT capabilities       (2 feature)
  token 7 : ie107 interworking           (5 feature)

Ogni token ha dimensione diversa -> una Linear per gruppo porta tutti
a d_model prima del transformer.

Schema del forward pass:
  x (B, FEATURE_DIM)
  -> split in 8 gruppi -> 8 Linear -> (B, 8, d_model)
  -> + positional embedding (learnable, 8 posizioni)
  -> TransformerEncoder (N layer, self-attention tra i token)
  -> mean pooling dei token -> (B, d_model)
  -> projection head -> (B, embed_dim)
  -> L2 normalize -> z (B, embed_dim)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field


# -----------------------------------------------------------------------
# Definizione dei gruppi di feature (token)
# Le dimensioni devono sommare a FEATURE_DIM = 79
# -----------------------------------------------------------------------

# Ogni entry: (nome, slice) dove slice indica le posizioni nel vettore flat
FEATURE_GROUPS = [
    ("ios_ssid",   slice(0,  4)),    #  4 feature: is_ios + SSID (3)
    ("ie1",        slice(4,  21)),   # 17 feature: 16 multi-hot + 1 lunghezza
    ("ie50",       slice(21, 31)),   # 10 feature: 8 multi-hot + presente + lunghezza
    ("ie127_0",    slice(31, 42)),   # 11 feature: 10 padded + lunghezza
    ("ie127_1",    slice(42, 51)),   #  9 feature: padded a lunghezza 9
    ("ie45",       slice(51, 68)),   # 17 feature: 17 subcampi HT
    ("ie221",      slice(68, 80)),   # 12 feature: 4 slot * 3
    ("ie191",      slice(80, 82)),   #  2 feature: presente + valore
    ("ie107",      slice(82, 87)),   #  5 feature: 5 subcampi interworking
]

# Verifica che i gruppi coprano esattamente FEATURE_DIM feature senza buchi
_EXPECTED_FEATURE_DIM = 87
_covered = sum(s.stop - s.start for _, s in FEATURE_GROUPS)
assert _covered == _EXPECTED_FEATURE_DIM, (
    f"I gruppi coprono {_covered} feature, attese {_EXPECTED_FEATURE_DIM}"
)

N_TOKENS = len(FEATURE_GROUPS)   # 9 token


@dataclass
class TransformerConfig:
    """
    Parametri dell'architettura. Tutti hanno un default ragionevole
    per il dataset in questione (~32k probe, 39 device, 79 feature).

    d_model : dimensione interna del transformer. Deve essere divisibile
              per nhead. Valori tipici: 64, 128, 256.
    nhead   : numero di attention head. Più heads -> più pattern paralleli.
              Con 8 token, 4 head sono abbondanti.
    num_layers : numero di TransformerEncoderLayer. 2-4 è sufficiente per
                 sequenze corte (8 token). Più layer -> più capacità ma
                 rischio overfitting su dataset piccoli.
    dim_feedforward : dimensione del FF interno di ogni layer.
                      Convenzione: 2x o 4x d_model.
    embed_dim : dimensione dello spazio di embedding finale z.
                Più piccolo -> più compresso, più facile da clusterizzare.
                Valori tipici: 32, 64, 128.
    dropout   : dropout applicato sia nel transformer sia nel projection head.
    pooling   : come aggregare gli 8 token in output.
                "mean" = media di tutti i token (robusto)
                "cls"  = token CLS aggiunto in testa (stile BERT)
    """
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 3
    dim_feedforward: int = 256
    embed_dim: int = 64
    dropout: float = 0.1
    pooling: str = "mean"   # "mean" | "cls"


class ProbeEncoder(nn.Module):
    """
    Encoder che mappa una probe request -> embedding vettore normalizzato.

    Uso tipico:
        encoder = ProbeEncoder(TransformerConfig())
        z = encoder(x)   # x: (B, 79), z: (B, embed_dim) L2-norm
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        # --- Proiezioni per-gruppo ---
        # Ogni gruppo di feature ha dimensione diversa; una Linear
        # separata porta ciascun gruppo a d_model.
        # Usiamo nn.ModuleList per registrare correttamente i parametri.
        self.group_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(s.stop - s.start, config.d_model),
                nn.LayerNorm(config.d_model),
                nn.GELU(),
            )
            for _, s in FEATURE_GROUPS
        ])

        # --- CLS token (opzionale, se pooling == "cls") ---
        if config.pooling == "cls":
            # Learnable vector aggiunto come primo token della sequenza
            self.cls_token = nn.Parameter(torch.randn(1, 1, config.d_model))

        # --- Positional embedding ---
        # Learnable: il modello impara l'importanza relativa della posizione
        # di ogni IE nella sequenza (es. ie45 è sempre token 4).
        n_pos = N_TOKENS + (1 if config.pooling == "cls" else 0)
        self.pos_embed = nn.Embedding(n_pos, config.d_model)

        # --- Transformer Encoder ---
        # Pre-LN (norm_first=True) è più stabile in training con lr alte
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,    # (batch, seq, feature) invece di (seq, batch, feature)
            norm_first=True,     # Pre-LN: LayerNorm prima dell'attention
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers,
            enable_nested_tensor=False,
        )

        # --- Projection head ---
        # Mappa d_model -> embed_dim dopo il pooling.
        # Due layer con GELU: il primo espande leggermente, il secondo proietta.
        # In letteratura (SimCLR, SupCon) un projection head non-lineare
        # migliora la qualità degli embedding rispetto a una proiezione lineare.
        self.proj_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.embed_dim),
        )

        # Inizializzazione dei pesi
        self._init_weights()

    def _init_weights(self):
        """
        Inizializzazione dei pesi con Xavier uniform per le Linear,
        che funziona bene con GELU e riduce la varianza dei gradienti
        nei primi passi di training.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def forward(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """
        Parametri
        ---------
        x         : (B, FEATURE_DIM) tensore di feature float32
        normalize : se True, L2-normalizza l'embedding output.
                    Deve essere True durante training con SupCon Loss.
                    Può essere False se si vuole l'embedding grezzo.

        Restituisce
        -----------
        z : (B, embed_dim) embedding delle probe
        """
        B = x.size(0)

        # 1. Dividi il vettore flat in gruppi e proietta ciascuno a d_model
        #    Risultato: lista di (B, d_model), poi stack -> (B, N_TOKENS, d_model)
        token_list = []
        for i, (_, s) in enumerate(FEATURE_GROUPS):
            group_feat = x[:, s]            # (B, group_dim)
            token = self.group_projs[i](group_feat)  # (B, d_model)
            token_list.append(token)
        tokens = torch.stack(token_list, dim=1)   # (B, N_TOKENS, d_model)

        # 2. Aggiungi CLS token se richiesto
        if self.config.pooling == "cls":
            cls = self.cls_token.expand(B, -1, -1)   # (B, 1, d_model)
            tokens = torch.cat([cls, tokens], dim=1)  # (B, N_TOKENS+1, d_model)

        # 3. Aggiungi positional embedding
        #    Le posizioni sono 0, 1, ..., N_TOKENS (o N_TOKENS+1 con CLS)
        seq_len = tokens.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)  # (1, seq_len)
        tokens = tokens + self.pos_embed(positions)   # broadcast su batch

        # 4. Transformer Encoder
        #    Nessuna padding mask: tutte le probe hanno sempre tutti i token
        #    (anche gli IE assenti diventano token con valori -1.0, non vengono
        #     mascherati, perché l'assenza stessa è un'informazione)
        out = self.transformer(tokens)   # (B, seq_len, d_model)

        # 5. Pooling: aggrega i token in un singolo vettore per probe
        if self.config.pooling == "cls":
            # Usa solo il CLS token (posizione 0), che ha "visto" tutti gli altri
            # tramite self-attention
            pooled = out[:, 0, :]    # (B, d_model)
        else:
            # Media di tutti i token: robusta, funziona bene con sequenze corte
            pooled = out.mean(dim=1)  # (B, d_model)

        # 6. Projection head -> embedding finale
        z = self.proj_head(pooled)   # (B, embed_dim)

        # 7. L2-normalizzazione: porta z sulla sfera unitaria
        #    Necessario per la SupCon Loss e per DBSCAN con metrica cosine
        if normalize:
            z = F.normalize(z, dim=-1)

        return z
