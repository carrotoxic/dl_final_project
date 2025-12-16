from typing import Literal
import numpy as np
from sklearn.cluster import (
    KMeans,
    AgglomerativeClustering,
    DBSCAN,
    SpectralClustering,
)
from sklearn.mixture import GaussianMixture


CLUSTERING_METHODS = ["kmeans", "gmm", "agglomerative", "dbscan", "spectral"]


def create_clustering_method(
    method: Literal["kmeans", "gmm", "agglomerative", "dbscan", "spectral"],
    n_clusters: int = None,
    **kwargs
):
    """Create clustering model based on method name."""
    if method == "kmeans":
        return KMeans(
            n_clusters=n_clusters,
            n_init=100,
            max_iter=10000,
            tol=0.0001,
            random_state=42,
            **kwargs
        )
    elif method == "gmm":
        return GaussianMixture(
            n_components=n_clusters,
            n_init=100,
            max_iter=10000,
            covariance_type='full',
            reg_covar=1e-6,
            random_state=42,
            **kwargs
        )
    elif method == "agglomerative":
        return AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage='ward',
            **kwargs
        )
    elif method == "dbscan":
        eps = kwargs.get('eps', 0.5)
        min_samples = kwargs.get('min_samples', 5)
        return DBSCAN(eps=eps, min_samples=min_samples, **{k: v for k, v in kwargs.items() if k not in ['eps', 'min_samples']})
    elif method == "spectral":
        return SpectralClustering(
            n_clusters=n_clusters,
            random_state=42,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown method: {method}. Must be one of {CLUSTERING_METHODS}")


def fit_predict(clustering_method, features: np.ndarray) -> np.ndarray:
    """Fit clusterer and return labels."""
    if features.dtype != np.float64 and (hasattr(clustering_method, 'n_components') or isinstance(clustering_method, GaussianMixture)):
        features = features.astype(np.float64)
    
    if hasattr(clustering_method, 'fit_predict'):
        labels = clustering_method.fit_predict(features)
    else:
        clustering_method.fit(features)
        labels = clustering_method.labels_
    
    return labels

