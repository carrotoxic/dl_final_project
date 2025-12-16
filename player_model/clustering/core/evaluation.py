import numpy as np
from sklearn.metrics import silhouette_score
from typing import Dict, Tuple


def compute_silhouette_score(features: np.ndarray, labels: np.ndarray) -> float:
    """Compute Silhouette score for clustering."""
    if len(np.unique(labels)) < 2:
        return -1.0
    
    try:
        return silhouette_score(features, labels)
    except Exception:
        return -1.0


def evaluate_clustering(
    features: np.ndarray,
    labels: np.ndarray,
    method: str,
    n_clusters: int = None
) -> Dict[str, float]:
    """Evaluate clustering with multiple metrics."""
    unique_labels = np.unique(labels[labels >= 0])
    n_clusters_found = len(unique_labels)
    
    results = {
        "silhouette_score": compute_silhouette_score(features, labels),
        "n_clusters_found": n_clusters_found,
        "n_noise": int(np.sum(labels == -1)) if -1 in labels else 0,
    }
    
    if n_clusters is not None:
        results["n_clusters_requested"] = n_clusters
    
    return results

