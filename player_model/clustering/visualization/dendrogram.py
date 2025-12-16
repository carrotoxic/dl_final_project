from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering


def plot_dendrogram(
    features: np.ndarray,
    clusterer: AgglomerativeClustering,
    save_path: Path,
    max_display: int = 100
):
    """Plot dendrogram for AgglomerativeClustering."""
    if features.shape[0] > max_display:
        indices = np.random.choice(features.shape[0], max_display, replace=False)
        features_subset = features[indices]
    else:
        features_subset = features
        indices = np.arange(features.shape[0])
    
    linkage_matrix = linkage(features_subset, method='ward')
    
    plt.figure(figsize=(12, 6))
    dendrogram(
        linkage_matrix,
        leaf_rotation=90,
        leaf_font_size=8,
        truncate_mode='level',
        p=10 if features.shape[0] > max_display else None,
    )
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')
    plt.title('Hierarchical Clustering Dendrogram')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved dendrogram to: {save_path}")

