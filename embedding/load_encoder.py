import torch

from utils.model import ProbeEncoder, TransformerConfig

# -----------------------------------------------------------------------
# Caricamento modello
# -----------------------------------------------------------------------

def load_encoder(
    checkpoint_path: str,
    device: torch.device,
) -> ProbeEncoder:
    """
    Carica un ProbeEncoder da un checkpoint salvato durante il training.
    Il checkpoint contiene la configurazione del modello, quindi non
    è necessario passare i parametri manualmente.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = TransformerConfig(**ckpt['config'])
    model = ProbeEncoder(config).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"Modello caricato da: {checkpoint_path} "
          f"(epoca {ckpt.get('epoch', '?')}, "
          f"delta_sim={ckpt.get('val_metrics', {}).get('delta_sim', '?'):.3f})")
    return model