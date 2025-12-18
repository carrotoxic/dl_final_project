import numpy as np
import matplotlib.colors as mcolors


SOFT_COLORS = [
    '#FF69B4',  # Hot Pink
    '#00BFFF',  # Deep Sky Blue
    '#00CED1',  # Dark Turquoise
    '#FFD700',  # Gold
    '#9370DB',  # Medium Purple
    '#1E90FF',  # Dodger Blue
    '#FF8C00',  # Dark Orange
    '#32CD32',  # Lime Green
    '#FF6347',  # Tomato
    '#4169E1',  # Royal Blue
    '#BA55D3',  # Medium Orchid
    '#FFA500',  # Orange
    '#20B2AA',  # Light Sea Green
    '#FF1493',  # Deep Pink
    '#00FA9A',  # Medium Spring Green
]


def get_cluster_colors(n_clusters: int, labels: np.ndarray = None) -> dict:
    """Get consistent color mapping for clusters using soft colors."""
    colors = SOFT_COLORS[:n_clusters]
    
    if labels is not None:
        unique_labels = sorted(np.unique(labels[labels >= 0]))
        color_map = {label: colors[i % len(colors)] for i, label in enumerate(unique_labels)}
        if -1 in labels:
            color_map[-1] = '#CCCCCC'
        return color_map
    
    return {i: colors[i % len(colors)] for i in range(n_clusters)}


def get_color_list(labels: np.ndarray, color_map: dict) -> list:
    """Convert labels to color list."""
    return [color_map.get(label, '#CCCCCC') for label in labels]

