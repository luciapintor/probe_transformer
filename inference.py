"""
inference.py
============
Inferenza su un CSV di probe request con un ProbeEncoder addestrato.

Funzionalità
------------
1. Carica il modello da checkpoint
2. Estrae embedding per tutte le probe nel CSV
3. Applica DBSCAN (o HDBSCAN se disponibile) per il clustering
4. Salva i risultati in un CSV e/o JSON
5. Stampa statistiche di qualità se il CSV contiene la colonna 'label'

Uso da riga di comando:
    python inference.py \
        --model probe_encoder.pt \
        --csv all_A_full.csv \
        --output clusters.csv \
        --eps 0.25 \
        --min_samples 3

Uso da codice:
    from inference import load_encoder, extract_embeddings, cluster
    encoder = load_encoder("probe_encoder.pt", device)
    Z = extract_embeddings(encoder, X, device)
    labels = cluster(Z, eps=0.25)
"""

import torch

from embedding.load_encoder import load_encoder
from embedding.extract_embeddings import extract_embeddings, embedding_separation_stats
from embedding.cluster_dbscan import use_dbscan
from utils.preprocessing import load_csv


# -----------------------------------------------------------------------
# Funzione principale di inferenza
# -----------------------------------------------------------------------

def run_inference(
    model_path: str,
    csv_path: str,
    excluded_features: list[str],
    output_baseline_path: str = "baseline_clusters.csv",
    output_embeddings_path: str = "clusters.csv",
    eps: float = 0.25,
    min_samples: int = 3,
    tune: bool = False,
    device_str: str = "auto",
    batch_size: int = 512,
):
    """
    Pipeline completa di inferenza.

    Se il CSV ha la colonna 'label', calcola anche ARI, NMI e statistiche
    di separazione per valutare la qualità del clustering.
    """
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    # Carica dati e modello
    X, y, meta = load_csv(csv_path, excluded_features=excluded_features)
    model = load_encoder(model_path, device)
    
    # Usa DBSCAN senza embeddings come baseline per il confronto
    print("\n--- Clustering baseline (feature originali) ---")
    baseline_labels = use_dbscan(X, y, eps=eps, min_samples=min_samples, tune=tune, output_path=output_baseline_path)  
    
    # Estrai embedding
    print("\n--- Estrazione embedding ---")
    Z = extract_embeddings(model, X, device, batch_size=batch_size)

    # Statistiche di separazione (richiede label)
    stats = embedding_separation_stats(Z, y)

    # Clustering sugli embedding

    pred_labels = use_dbscan(Z, y, eps=eps, min_samples=min_samples, tune=tune, output_path=output_embeddings_path)    

    return Z, pred_labels, stats

# -----------------------------------------------------------------------
# Funzione principale per esecuzione da riga di comando
# -----------------------------------------------------------------------

if __name__ == "__main__":
    from config import csv_path, model_path, excluded_features, baseline_path, embeddings_path 
        
    print("Inferenza ProbeEncoder")
    eps = 0.25
    min_samples = 3
    tune = False
    device_str = "auto"
    batch_size = 512

    run_inference(
        model_path=model_path,
        csv_path=csv_path,
        excluded_features=excluded_features,
        output_baseline_path=baseline_path,
        output_embeddings_path=embeddings_path,
        eps=eps,
        min_samples=min_samples,
        tune=tune,
        device_str=device_str,
        batch_size=batch_size,
    )
