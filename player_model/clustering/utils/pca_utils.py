import numpy as np
from sklearn.decomposition import PCA


def compute_pca_projection(features: np.ndarray, n_components: int = 2):
    """Compute PCA projection"""
    pca = PCA(n_components=n_components)
    projection = pca.fit_transform(features)
    return pca, projection


def get_top_players(player_ids: list, n_top: int = 10):
    """Get top N players by frequency"""
    from collections import Counter
    player_counts = Counter(player_ids)
    top_players = [pid for pid, _ in player_counts.most_common(n_top)]
    top_mask = [pid in top_players for pid in player_ids]
    return top_players, top_mask

