"""
losses.py
=========
Supervised Contrastive Loss (SupCon) per il training del ProbeEncoder.

Perché SupCon invece di NT-Xent self-supervised?
-------------------------------------------------
Nel dataset abbiamo le label reali dei device (39 classi). Questo
permette di usare una versione SUPERVISIONATA della loss contrastiva:
  - per ogni probe (anchor), TUTTI i campioni della stessa classe
    nella batch sono considerati positivi (non solo un augmented copy)
  - i campioni di classi diverse sono negativi

SupCon è più efficace di NT-Xent perché:
1. Sfrutta più coppie positive per batch (tutte le probe dello stesso device)
2. È robusta allo sbilanciamento delle classi (il termine di normalizzazione
   dipende dal numero di positivi per anchor)
3. Generalizza meglio perché impara da variazioni reali tra probe dello stesso
   device, non da augmentazioni artificiali

Formulazione matematica
-----------------------
Data una batch di N probe con embedding L2-normalizzati z_1..z_N e label y_1..y_N:

  L = -1/N * sum_i [ 1/|P(i)| * sum_{p in P(i)} log(
        exp(z_i · z_p / tau) /
        sum_{a != i} exp(z_i · z_a / tau)
      )]

dove P(i) = {j : y_j == y_i, j != i} è l'insieme dei positivi per l'anchor i.

Se P(i) è vuoto (nessun'altra probe dello stesso device nella batch),
quell'anchor viene ignorato nel calcolo della loss.

Riferimento: Khosla et al., "Supervised Contrastive Learning", NeurIPS 2020.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss.

    Parametri
    ---------
    temperature : float
        Scala la "durezza" del contrasto.
        - Valori piccoli (0.05-0.1): contrasto molto netto, ma può
          portare a instabilità numerica e collasso dei negativi
        - Valori grandi (0.5-1.0): contrasto morbido, convergenza
          più lenta ma più stabile
        Suggerimento iniziale: 0.1 per questo dataset.

    base_temperature : float
        Temperatura di riferimento per la normalizzazione della scala
        della loss (convenzione dal paper originale, default = 0.07).
        Nella pratica, impostarlo uguale a temperature semplifica il tuning.
    """

    def __init__(self, temperature: float = 0.1, base_temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(
        self,
        embeddings: torch.Tensor,  # (N, D) embedding L2-normalizzati
        labels: torch.Tensor,      # (N,) label intere del device
    ) -> torch.Tensor:
        """
        Calcola la SupCon Loss per una batch.

        Restituisce lo scalare di loss. Restituisce 0 se nessun anchor
        ha almeno un positivo nella batch (batch troppo piccola o con
        device tutti distinti).
        """
        N = embeddings.size(0)
        device = embeddings.device

        # --- Matrice di similarità coseno ---
        # Gli embedding sono già L2-normalizzati, quindi il prodotto scalare
        # è direttamente la cosine similarity in [-1, 1].
        # Dividendo per temperature otteniamo i logit per la softmax.
        sim = torch.mm(embeddings, embeddings.T) / self.temperature  # (N, N)

        # --- Maschera dei positivi ---
        # positive_mask[i][j] = True se y_i == y_j AND i != j
        labels_col = labels.unsqueeze(1)           # (N, 1)
        labels_row = labels.unsqueeze(0)           # (1, N)
        positive_mask = (labels_col == labels_row) # (N, N) bool
        # Rimuovi la diagonale (una probe non è positiva di se stessa)
        eye_mask = torch.eye(N, dtype=torch.bool, device=device)
        positive_mask = positive_mask & ~eye_mask  # (N, N)

        # --- Conta i positivi per anchor ---
        # n_positives[i] = numero di probe dello stesso device in batch (escluso i)
        n_positives = positive_mask.sum(dim=1).float()  # (N,)

        # Gli anchor senza positivi nella batch vengono ignorati
        # (non contribuiscono alla loss)
        valid_anchors = n_positives > 0  # (N,) bool

        if not valid_anchors.any():
            # Nessun anchor valido: la batch non contiene coppie positive.
            # Restituisce 0 con gradiente per non bloccare il training.
            return torch.tensor(0.0, requires_grad=True, device=device)

        # --- Maschera da escludere dalla somma dei denominatori ---
        # Il denominatore somma su tutti i j != i (positivi E negativi).
        # Escludiamo solo la diagonale.
        neg_mask = ~eye_mask  # (N, N): True = includi nel denominatore

        # --- Log-softmax per stabilità numerica ---
        # Sottraiamo il max per riga prima dell'exp (log-sum-exp trick)
        # Mascheriamo la diagonale portandola a -inf prima del softmax
        sim_masked = sim.masked_fill(eye_mask, float('-inf'))
        log_denom = torch.logsumexp(sim_masked, dim=1, keepdim=True)  # (N, 1)

        # log_prob[i][j] = log P(j | i) = sim[i][j] - log_denom[i]
        log_prob = sim - log_denom  # (N, N)

        # --- Loss per anchor ---
        # Per ogni anchor i, media del log-prob sui positivi
        # Moltiplica per positive_mask (float) per sommare solo sui positivi
        sum_log_prob_pos = (positive_mask.float() * log_prob).sum(dim=1)  # (N,)
        # Normalizza per il numero di positivi (evita bias verso classi grandi)
        mean_log_prob_pos = sum_log_prob_pos / n_positives.clamp(min=1)    # (N,)

        # --- Loss finale ---
        # Media sugli anchor validi, con fattore di scala temperature/base_temperature
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss[valid_anchors].mean()

        return loss


class CombinedLoss(nn.Module):
    """
    Loss combinata: SupCon + CrossEntropy su un classification head ausiliario.

    Aggiungere una CrossEntropy classification head durante il training
    può stabilizzare l'addestramento, specialmente nelle prime epoche
    quando gli embedding non sono ancora ben separati.

    La CE loss è su un layer lineare separato (non fa parte dell'encoder
    finale): dopo il training, il classification head viene scartato e
    si usano solo gli embedding del ProbeEncoder.

    Parametri
    ---------
    supcon_weight : peso della SupCon Loss (default 1.0)
    ce_weight     : peso della CrossEntropy Loss (default 0.5)
                    Impostare a 0 per usare solo SupCon.
    """

    def __init__(
        self,
        embed_dim: int,
        n_classes: int,
        temperature: float = 0.1,
        supcon_weight: float = 1.0,
        ce_weight: float = 0.5,
    ):
        super().__init__()
        self.supcon = SupConLoss(temperature=temperature)
        self.ce_weight = ce_weight
        self.supcon_weight = supcon_weight

        # Classification head ausiliario: linear probe sull'embedding
        # Non fa parte del ProbeEncoder, viene usato solo durante training
        if ce_weight > 0:
            self.classifier = nn.Linear(embed_dim, n_classes)
        else:
            self.classifier = None

    def forward(
        self,
        embeddings: torch.Tensor,  # (N, embed_dim) L2-norm
        labels: torch.Tensor,      # (N,) label
    ) -> tuple[torch.Tensor, dict]:
        """
        Restituisce (loss_totale, dict_con_componenti) per il logging.
        """
        # Componente SupCon
        loss_supcon = self.supcon(embeddings, labels)

        # Componente CrossEntropy (opzionale)
        if self.classifier is not None and self.ce_weight > 0:
            logits = self.classifier(embeddings)    # (N, n_classes)
            loss_ce = F.cross_entropy(logits, labels)
        else:
            loss_ce = torch.tensor(0.0, device=embeddings.device)

        total = self.supcon_weight * loss_supcon + self.ce_weight * loss_ce

        return total, {
            'supcon': loss_supcon.item(),
            'ce': loss_ce.item(),
            'total': total.item(),
        }
