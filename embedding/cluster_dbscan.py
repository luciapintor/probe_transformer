import numpy as np
import pandas as pd

from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


# -----------------------------------------------------------------------
# Clustering
# -----------------------------------------------------------------------

def cluster_dbscan(
    Z: np.ndarray,
    eps: float = 0.25,
    min_samples: int = 3,
    metric: str = 'cosine',
) -> np.ndarray:
    """
    Applica DBSCAN sugli embedding con metrica cosine.

    Con embedding L2-normalizzati la distanza coseno è:
        d(a, b) = 1 - a·b

    Quindi eps=0.25 significa che due probe sono nello stesso cluster
    se la loro cosine similarity è >= 0.75.

    Parametri da calibrare
    ----------------------
    eps         : raggio del vicinato. Valore suggerito: 0.2-0.4.
                  Usa tune_eps() per trovare il valore ottimale.
    min_samples : min probe per formare un core point.
                  Con dataset grandi, 3-5 è ragionevole.
                  Con dataset piccoli o device con poche probe, usa 2.

    Restituisce
    -----------
    labels : (N,) array int, -1 = rumore (non clusterizzato)
    """
    db = DBSCAN(eps=eps, min_samples=min_samples, metric=metric, n_jobs=-1)
    labels = db.fit_predict(Z)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    print(f"DBSCAN: {n_clusters} cluster, {n_noise} probe non clusterizzate "
          f"({100*n_noise/len(Z):.1f}%)")
    return labels

def use_dbscan(X, y, eps=0.25, min_samples=3, tune=False, output_path="clusters.csv", metric='cosine'):
    """
    Funzione per usare DBSCAN.
    """
    # Tuning eps (opzionale, richiede label)
    if tune:
        eps = tune_eps(X, y, min_samples=min_samples)

    # Clustering
    pred_labels = cluster_dbscan(X, eps=eps, min_samples=min_samples, metric=metric)

    # Metriche di qualità (se abbiamo le label)
    valid = pred_labels != -1
    if valid.sum() > 0:
        ari = adjusted_rand_score(y[valid], pred_labels[valid])
        nmi = normalized_mutual_info_score(y[valid], pred_labels[valid])
        print(f"\nMetriche clustering (probe non-rumore):")
        print(f"  ARI: {ari:.4f}  (1.0 = perfetto, 0.0 = casuale)")
        print(f"  NMI: {nmi:.4f}  (1.0 = perfetto, 0.0 = casuale)")

    # Salva risultati
    df_out = pd.DataFrame({
        'true_label':    y,
        'pred_cluster':  pred_labels,
        'embedding_dim0': X[:, 0],   # prime 2 dim per debug/visualizzazione
        'embedding_dim1': X[:, 1],
    })
    df_out.to_csv(output_path, index=False)
    print(f"\nRisultati salvati in: {output_path}")
    
    return pred_labels

def tune_eps(
    Z: np.ndarray,
    true_labels: np.ndarray,
    eps_range: list = None,
    min_samples: int = 3,
) -> float:
    """
    Cerca il valore di eps che massimizza l'Adjusted Rand Index (ARI)
    rispetto alle label reali dei device.

    Utile per calibrare DBSCAN quando si hanno label di ground truth.
    In produzione (senza label) usare la curva k-distance invece.

    Restituisce il valore di eps ottimale.
    """
    if eps_range is None:
        eps_range = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

    print(f"\nTuning eps (min_samples={min_samples}):")
    print(f"{'eps':>6} | {'n_cluster':>9} | {'ARI':>8} | {'NMI':>8} | {'noise%':>7}")
    print("-" * 50)

    best_ari = -1.0
    best_eps = eps_range[0]

    for eps in eps_range:
        pred = cluster_dbscan(Z, eps=eps, min_samples=min_samples)
        # Esclude le probe di rumore (-1) dal calcolo delle metriche
        valid = pred != -1
        if valid.sum() < 10:
            continue

        ari = adjusted_rand_score(true_labels[valid], pred[valid])
        nmi = normalized_mutual_info_score(true_labels[valid], pred[valid])
        n_cl = len(set(pred)) - 1
        noise_pct = 100 * (1 - valid.mean())

        print(f"{eps:6.2f} | {n_cl:9d} | {ari:8.4f} | {nmi:8.4f} | {noise_pct:6.1f}%")

        if ari > best_ari:
            best_ari = ari
            best_eps = eps

    print(f"\nMiglior eps={best_eps:.2f} (ARI={best_ari:.4f})")
    return best_eps
