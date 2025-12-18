from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from collections import Counter

from ..utils.pca_utils import compute_pca_projection, get_top_players
from ..utils.color_utils import get_cluster_colors, get_color_list


def plot_clusters_2d(features: np.ndarray, labels: np.ndarray, save_path: Path, title: str = None):
    """Plot 2D cluster visualization with soft colors"""
    _, projection_2d = compute_pca_projection(features, n_components=2)
    unique_labels = sorted(np.unique(labels[labels >= 0]))
    color_map = get_cluster_colors(len(unique_labels), labels)
    colors = get_color_list(labels, color_map)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        projection_2d[:, 0], projection_2d[:, 1],
        c=colors, s=15, alpha=0.8, edgecolors='white', linewidths=0.3
    )
    
    for k in unique_labels:
        idx = labels == k
        if not np.any(idx):
            continue
        cx = projection_2d[idx, 0].mean()
        cy = projection_2d[idx, 1].mean()
        n_points = int(idx.sum())
        plt.text(
            cx, cy, f"C{k}\n(n={n_points})",
            fontsize=10, fontweight="bold", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8, ec=color_map[k], lw=2),
        )
    
    plt.xlabel("PC1", fontsize=12)
    plt.ylabel("PC2", fontsize=12)
    plt.title(title or "Latent Space (PCA 2D, colored by cluster)", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_clusters_3d(features: np.ndarray, labels: np.ndarray, save_path: Path, title: str = None):
    """Plot 3D cluster visualization with soft colors"""
    _, projection_3d = compute_pca_projection(features, n_components=3)
    unique_labels = sorted(np.unique(labels[labels >= 0]))
    color_map = get_cluster_colors(len(unique_labels), labels)
    colors = get_color_list(labels, color_map)
    
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")
    
    ax.scatter(
        projection_3d[:, 0], projection_3d[:, 1], projection_3d[:, 2],
        c=colors, s=10, alpha=0.8, edgecolors='white', linewidths=0.2
    )
    
    ax.set_xlabel("PC1", fontsize=11)
    ax.set_ylabel("PC2", fontsize=11)
    ax.set_zlabel("PC3", fontsize=11)
    ax.set_title(title or "Latent Space (PCA 3D, colored by cluster)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_players_2d(features: np.ndarray, player_ids: list, save_path: Path, n_top: int = 5):
    """Plot 2D player visualization"""
    _, projection_2d = compute_pca_projection(features, n_components=2)
    top_players, top_mask = get_top_players(player_ids, n_top=n_top)
    
    filtered_projection = projection_2d[top_mask]
    filtered_player_ids = [pid for pid, mask in zip(player_ids, top_mask) if mask]
    
    color_map = get_cluster_colors(len(top_players))
    player_to_idx = {pid: i for i, pid in enumerate(top_players)}
    colors = [color_map[player_to_idx[pid]] for pid in filtered_player_ids]
    
    plt.figure(figsize=(10, 8))
    plt.scatter(
        filtered_projection[:, 0], filtered_projection[:, 1],
        c=colors, s=15, alpha=0.8, edgecolors='white', linewidths=0.3
    )
    
    handles = [Patch(facecolor=color_map[player_to_idx[pid]], label=pid) for pid in top_players]
    plt.legend(handles=handles, loc="best", fontsize=9)
    plt.xlabel("PC1", fontsize=12)
    plt.ylabel("PC2", fontsize=12)
    plt.title(f"Latent Space (PCA 2D, top {len(top_players)} players)", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_players_3d(features: np.ndarray, player_ids: list, save_path: Path, n_top: int = 5):
    """Plot 3D player visualization"""
    _, projection_3d = compute_pca_projection(features, n_components=3)
    top_players, top_mask = get_top_players(player_ids, n_top=n_top)
    
    filtered_projection = projection_3d[top_mask]
    filtered_player_ids = [pid for pid, mask in zip(player_ids, top_mask) if mask]
    
    color_map = get_cluster_colors(len(top_players))
    player_to_idx = {pid: i for i, pid in enumerate(top_players)}
    colors = [color_map[player_to_idx[pid]] for pid in filtered_player_ids]
    
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")
    
    ax.scatter(
        filtered_projection[:, 0], filtered_projection[:, 1], projection_3d[top_mask, 2],
        c=colors, s=10, alpha=0.8, edgecolors='white', linewidths=0.2
    )
    
    ax.set_xlabel("PC1", fontsize=11)
    ax.set_ylabel("PC2", fontsize=11)
    ax.set_zlabel("PC3", fontsize=11)
    ax.set_title(f"Latent Space (PCA 3D, top {len(top_players)} players)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def extract_player_type(path_str: str) -> int:
    """Extract player type from path"""
    path_lower = str(path_str).lower()
    if "runner" in path_lower:
        return 0
    elif "killer" in path_lower:
        return 1
    elif "collector" in path_lower:
        return 2
    return -1


def plot_player_types_2d(features: np.ndarray, meta: list, save_path: Path):
    """Plot 2D player type visualization"""
    _, projection_2d = compute_pca_projection(features, n_components=2)
    
    type_indices = [extract_player_type(m.get("path", "")) for m in meta]
    valid_mask = np.array([t >= 0 for t in type_indices])
    
    if not np.any(valid_mask):
        return
    
    filtered_projection = projection_2d[valid_mask]
    filtered_types = np.array([type_indices[i] for i in range(len(type_indices)) if valid_mask[i]])
    
    type_colors = ['#FF69B4', '#00BFFF', '#00CED1']
    colors = [type_colors[t] for t in filtered_types]
    
    plt.figure(figsize=(10, 8))
    plt.scatter(
        filtered_projection[:, 0], filtered_projection[:, 1],
        c=colors, s=15, alpha=0.8, edgecolors='white', linewidths=0.3
    )
    
    legend_elements = [
        Patch(facecolor=type_colors[0], label="Runner"),
        Patch(facecolor=type_colors[1], label="Killer"),
        Patch(facecolor=type_colors[2], label="Collector"),
    ]
    plt.legend(handles=legend_elements, loc="best")
    plt.xlabel("PC1", fontsize=12)
    plt.ylabel("PC2", fontsize=12)
    plt.title("Latent Space (PCA 2D, colored by player type)", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_player_types_3d(features: np.ndarray, meta: list, save_path: Path):
    """Plot 3D player type visualization"""
    _, projection_3d = compute_pca_projection(features, n_components=3)
    
    type_indices = [extract_player_type(m.get("path", "")) for m in meta]
    valid_mask = np.array([t >= 0 for t in type_indices])
    
    if not np.any(valid_mask):
        return
    
    filtered_projection = projection_3d[valid_mask]
    filtered_types = np.array([type_indices[i] for i in range(len(type_indices)) if valid_mask[i]])
    
    type_colors = ['#FF69B4', '#00BFFF', '#00CED1']
    colors = [type_colors[t] for t in filtered_types]
    
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")
    
    ax.scatter(
        filtered_projection[:, 0], filtered_projection[:, 1], filtered_projection[:, 2],
        c=colors, s=10, alpha=0.8, edgecolors='white', linewidths=0.2
    )
    
    legend_elements = [
        Patch(facecolor=type_colors[0], label="Runner"),
        Patch(facecolor=type_colors[1], label="Killer"),
        Patch(facecolor=type_colors[2], label="Collector"),
    ]
    ax.legend(handles=legend_elements, loc="best")
    ax.set_xlabel("PC1", fontsize=11)
    ax.set_ylabel("PC2", fontsize=11)
    ax.set_zlabel("PC3", fontsize=11)
    ax.set_title("Latent Space (PCA 3D, colored by player type)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

