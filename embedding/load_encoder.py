import torch

from utils.model import ProbeEncoder, TransformerConfig
from utils.preprocessing import load_csv
from utils.feature_schema import build_probe_schema, FeatureToken

# -----------------------------------------------------------------------
# Caricamento modello
# -----------------------------------------------------------------------

def load_encoder(
    checkpoint_path: str,
    device: torch.device,
) -> ProbeEncoder:
    """
    Carica un ProbeEncoder da un checkpoint salvato durante il training.
 
    Il checkpoint contiene sia la configurazione del transformer (config)
    sia la struttura delle feature (feature_groups, feature_dim), quindi
    non è necessario avere lo schema in memoria durante l'inferenza.
 
    I checkpoint salvati con train.py includono automaticamente
    feature_groups e feature_dim. Per checkpoint più vecchi che non li
    contengono, viene usato il fallback a build_probe_schema().
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = TransformerConfig(**ckpt['config'])
 
    if 'global_tokens' in ckpt:
        # Ricostruisce FeatureToken dalla lista di dizionari salvata nel checkpoint.
        
        global_tokens = [
            FeatureToken(d['name'], d['position'], d['ie_name'])
            for d in ckpt['global_tokens']
        ]
        model = ProbeEncoder(config, global_tokens=global_tokens).to(device)
    else:
        # Fallback per checkpoint vecchi.
        print("Attenzione: checkpoint senza global_tokens, uso schema di default.")
        model = ProbeEncoder(config, global_schema=build_probe_schema()).to(device)
 
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    delta = ckpt.get('val_metrics', {}).get('delta_sim', '?')
    delta_str = f"{delta:.3f}" if isinstance(delta, float) else delta
    print(f"Modello caricato da: {checkpoint_path} "
          f"(epoca {ckpt.get('epoch', '?')}, delta_sim={delta_str})")
    return model
 