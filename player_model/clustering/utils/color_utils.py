import numpy as np
import matplotlib.colors as mcolors


SOFT_COLORS = [
    '#FFB6C1',  # Light Pink
    '#87CEEB',  # Sky Blue
    '#98D8C8',  # Mint Green
    '#F7DC6F',  # Soft Yellow
    '#BB8FCE',  # Lavender
    '#85C1E2',  # Light Blue
    '#F8C471',  # Peach
    '#82E0AA',  # Light Green
    '#F1948A',  # Coral
    '#AED6F1',  # Powder Blue
    '#D7BDE2',  # Light Purple
    '#F9E79F',  # Pale Yellow
    '#A9DFBF',  # Seafoam
    '#F5B7B1',  # Rose
    '#A3E4D7',  # Turquoise
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

