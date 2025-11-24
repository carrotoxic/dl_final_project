import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from .auto_encoder.config import data_config, model_config, train_config
from .auto_encoder.datasets.trajectory_datasets import TrajectoryDataset, trajectory_collate_fn
from .auto_encoder.models.lstm_autoencoder import LSTMAutoencoder


def extract_latent(device: torch.device, normalize: bool = True):
    """
    Extract latent vectors from all traces using the trained LSTM Autoencoder.
    Args:
        device: torch device
        normalize: If True, normalize latents using StandardScaler (mean=0, std=1) for better PCA visualization
    Returns:
        latents: np.ndarray [N, latent_dim] (normalized if normalize=True)
        meta: list[dict]  metadata for each sample (player_id, path, length)
    """
    from sklearn.preprocessing import StandardScaler
    
    # Dataset (use all data. train/val split is not done here)
    dataset = TrajectoryDataset(data_config, normalize=True)
    dataloader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=False,
        collate_fn=trajectory_collate_fn,
        num_workers=1,
        pin_memory=True,
    )

    # Load model
    model = LSTMAutoencoder(model_config).to(device)
    ckpt_path = train_config.model_save_path
    # weights_only=False is needed for PyTorch 2.6+ when checkpoint contains custom classes
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    all_latents = []
    meta = []

    with torch.no_grad():
        for batch in dataloader:
            trajectories = batch["trajectories"].to(device)
            lengths = batch["lengths"].to(device)
            player_ids = batch["player_ids"]
            paths = batch["paths"]

            # [B, latent_dim]
            z = model.encode(trajectories, lengths)

            all_latents.append(z.cpu().numpy())
            for pid, pth, L in zip(player_ids, paths, lengths.cpu().tolist()):
                meta.append(
                    {
                        "player_id": pid,
                        "path": str(pth),
                        "length": L,
                    }
                )

    latents = np.concatenate(all_latents, axis=0)  # [N, latent_dim]
    
    # Normalize latents for better PCA visualization
    if normalize:
        print("[INFO] Normalizing latent vectors (StandardScaler) for better PCA visualization...")
        scaler = StandardScaler()
        latents = scaler.fit_transform(latents)
        print(f"[INFO] Latent vectors normalized: mean={latents.mean(axis=0).mean():.6f}, std={latents.std(axis=0).mean():.6f}")
    
    return latents, meta


def extract_all_features(device: torch.device):
    """
    Extract all features: latent vectors + game statistics.
    Returns:
        features: np.ndarray [N, feature_dim] - combined feature vector
        latents: np.ndarray [N, latent_dim] - latent vectors only (for visualization)
        meta: list[dict] - metadata for each sample
    """
    from sklearn.preprocessing import StandardScaler
    
    # First extract latent vectors
    latents, meta = extract_latent(device, normalize=True)
    
    # Load additional features from JSON files
    print("[INFO] Loading additional features from JSON files...")
    status_list = []
    numeric_features = []
    valid_indices = []
    
    for i, m in enumerate(meta):
        path = Path(m["path"])
        try:
            with path.open("r") as f:
                data = json.load(f)
            
            # Extract features
            status = data.get("status", "UNKNOWN")
            completing_ratio = data.get("completing-ratio", 0.0)
            kills = data.get("#kills", 0)
            kills_by_fire = data.get("#kills-by-fire", 0)
            kills_by_stomp = data.get("#kills-by-stomp", 0)
            kills_by_shell = data.get("#kills-by-shell", 0)
            lives = data.get("lives", 0)
            
            status_list.append(status)
            numeric_features.append([
                completing_ratio,
                float(kills),
                float(kills_by_fire),
                float(kills_by_stomp),
                float(kills_by_shell),
                float(lives),
            ])
            valid_indices.append(i)
            
            # Update meta with additional info
            m.update({
                "status": status,
                "completing_ratio": completing_ratio,
                "kills": kills,
                "kills_by_fire": kills_by_fire,
                "kills_by_stomp": kills_by_stomp,
                "kills_by_shell": kills_by_shell,
                "lives": lives,
            })
        except Exception as e:
            print(f"[WARN] Failed to load features from {path}: {e}")
            continue
    
    if len(numeric_features) != len(latents):
        print(f"[WARN] Mismatch: {len(numeric_features)} feature vectors vs {len(latents)} latent vectors")
        # Filter to only valid indices
        latents = latents[valid_indices]
        meta = [meta[i] for i in valid_indices]
    
    # One-hot encode status
    status_onehot = []
    for status in status_list:
        if status == "WIN":
            status_onehot.append([1.0, 0.0, 0.0])
        elif status == "LOSE":
            status_onehot.append([0.0, 1.0, 0.0])
        elif status == "TIME_OUT":
            status_onehot.append([0.0, 0.0, 1.0])
        else:
            # Unknown status - default to LOSE
            status_onehot.append([0.0, 1.0, 0.0])
    
    status_onehot = np.array(status_onehot, dtype=np.float32)  # [N, 3]
    numeric_features = np.array(numeric_features, dtype=np.float32)  # [N, 6]
    
    # Normalize numeric features (status one-hot doesn't need normalization)
    scaler = StandardScaler()
    numeric_features_scaled = scaler.fit_transform(numeric_features)
    
    # Combine: latent + status_onehot + numeric_features
    additional_features = np.concatenate([status_onehot, numeric_features_scaled], axis=1)  # [N, 3 + 6 = 9]
    
    # Combine latent vectors with additional features
    features = np.concatenate([latents, additional_features], axis=1)  # [N, latent_dim + 9]
    
    print(f"[INFO] Combined features shape: {features.shape} (latent: {latents.shape[1]}, additional: {additional_features.shape[1]})")
    print(f"[INFO]   - Status one-hot: 3 dims (WIN, LOSE, timeout)")
    print(f"[INFO]   - Numeric features: 6 dims (completing_ratio, kills, kills_by_fire, kills_by_stomp, kills_by_shell, lives)")
    
    return features, latents, meta


# ============================================================================
# Plotting Functions - All use the same PCA space
# ============================================================================

def plot_latent_2d_clusters(
    features: np.ndarray,
    labels: np.ndarray,
    save_path: Path,
):
    """
    Plot the 2D latent projection (PCA): color = cluster ID, show Ck label at the center.
    Uses PCA fitted on all data to ensure consistent visualization.
    """
    print("[INFO] plotting 2D latent projection by CLUSTER (PCA)...")
    
    _, projection_2d = compute_pca_projection(features, n_components=2)
    
    # Debug: Print actual counts per cluster being plotted
    unique_labels = np.unique(labels)
    print(f"[DEBUG] Total points to plot: {len(labels)}")
    for k in unique_labels:
        idx = labels == k
        n_points = int(idx.sum())
        print(f"[DEBUG] Cluster {k}: {n_points} points being plotted")

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        projection_2d[:, 0],
        projection_2d[:, 1],
        c=labels,
        s=10,
        alpha=0.8,
        cmap="tab10",
    )
    plt.colorbar(scatter, label="Cluster ID")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Latent Space (PCA 2D, colored by cluster)")

    # Draw the center of each cluster and the label Ck
    for k in unique_labels:
        idx = labels == k
        if not np.any(idx):
            continue
        cx = projection_2d[idx, 0].mean()
        cy = projection_2d[idx, 1].mean()
        n_points = int(idx.sum())
        plt.text(
            cx,
            cy,
            f"C{k} (n={n_points})",
            fontsize=9,
            fontweight="bold",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7),
        )

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[INFO] saved 2D cluster plot to: {save_path}")


def plot_latent_2d_players(
    features: np.ndarray,
    player_ids: list[str],
    save_path: Path,
):
    """
    Plot the 2D latent projection (PCA): color = player ID
    Shows only top 5 players by data size.
    Uses PCA fitted on all data to ensure consistent visualization.
    """
    print("[INFO] plotting 2D latent projection by PLAYER (PCA)...")
    
    # Compute PCA on all data
    _, projection_2d = compute_pca_projection(features, n_components=2)
    
    # Get top players
    top_players, top_mask = get_top_players(player_ids, n_top=5)
    print(f"[INFO] Showing top {len(top_players)} players by data size: {top_players}")

    # Filter to only top players
    filtered_projection = projection_2d[top_mask]
    filtered_player_ids = [pid for pid, mask in zip(player_ids, top_mask) if mask]

    # Create color array for top players
    cmap = plt.get_cmap("tab20")
    player_to_idx = {pid: i for i, pid in enumerate(top_players)}
    colors = [cmap(player_to_idx[pid] % cmap.N) for pid in filtered_player_ids]

    plt.figure(figsize=(8, 6))
    plt.scatter(
        filtered_projection[:, 0],
        filtered_projection[:, 1],
        c=colors,
        s=10,
        alpha=0.8,
    )
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"Latent Space (PCA 2D, colored by player - top {len(top_players)} players)")

    # Show legend for top players
    from matplotlib.patches import Patch
    handles = []
    for pid in top_players:
        # Find first occurrence of this player to get their color
        for idx, p in enumerate(filtered_player_ids):
            if p == pid:
                color = colors[idx]
                handles.append(Patch(facecolor=color, label=pid))
                break
    
    plt.legend(handles=handles, title="Players", loc="best", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[INFO] saved 2D player plot to: {save_path}")


def plot_latent_3d_players(
    features: np.ndarray,
    player_ids: list[str],
    save_path: Path,
):
    """
    Plot the 3D latent projection (PCA): color = player ID
    Shows only top 5 players by data size.
    Uses PCA fitted on all data to ensure consistent visualization.
    """
    print("[INFO] plotting 3D latent projection by PLAYER (PCA)...")
    
    if features.shape[1] < 3:
        print("[WARN] feature_dim < 3, 3D PCA may not be able to visualize properly.")
    
    # Compute PCA on all data
    _, projection_3d = compute_pca_projection(features, n_components=3)
    
    # Get top players
    top_players, top_mask = get_top_players(player_ids, n_top=5)
    print(f"[INFO] Showing top {len(top_players)} players by data size: {top_players}")

    # Filter to only top players
    filtered_projection = projection_3d[top_mask]
    filtered_player_ids = [pid for pid, mask in zip(player_ids, top_mask) if mask]

    # Create color array for top players
    cmap = plt.get_cmap("tab20")
    player_to_idx = {pid: i for i, pid in enumerate(top_players)}
    colors = [cmap(player_to_idx[pid] % cmap.N) for pid in filtered_player_ids]

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")  # type: ignore

    ax.scatter(
        filtered_projection[:, 0],
        filtered_projection[:, 1],
        filtered_projection[:, 2],
        c=colors,
        s=6,
        alpha=0.8,
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title(f"Latent Space (PCA 3D, colored by player - top {len(top_players)} players)")

    # Show legend for top players
    from matplotlib.patches import Patch
    handles = []
    for pid in top_players:
        # Find first occurrence of this player to get their color
        for idx, p in enumerate(filtered_player_ids):
            if p == pid:
                color = colors[idx]
                handles.append(Patch(facecolor=color, label=pid))
                break
    
    ax.legend(handles=handles, title="Players", loc="best", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[INFO] saved 3D player plot to: {save_path}")


def plot_latent_3d_clusters(
    features: np.ndarray,
    labels: np.ndarray,
    save_path: Path,
):
    """
    Plot the 3D latent projection (PCA): color = cluster ID
    Uses PCA fitted on all data to ensure consistent visualization.
    """
    print("[INFO] plotting 3D latent projection by CLUSTER (PCA)...")
    if features.shape[1] < 3:
        print("[WARN] feature_dim < 3, 3D PCA may not be able to visualize properly.")

    _, projection_3d = compute_pca_projection(features, n_components=3)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")  # type: ignore

    p = ax.scatter(
        projection_3d[:, 0],
        projection_3d[:, 1],
        projection_3d[:, 2],
        c=labels,
        s=6,
        alpha=0.8,
        cmap="tab10",
    )
    fig.colorbar(p, ax=ax, label="Cluster ID")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title("Latent Space (PCA 3D, colored by cluster)")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[INFO] saved 3D cluster plot to: {save_path}")


# ============================================================================
# Helper Functions for Plotting
# ============================================================================

def extract_player_type(path_str: str) -> str | None:
    """
    Extract player type from path string.
    Returns 'runner', 'killer', or 'collector' if found, None otherwise.
    """
    path_lower = path_str.lower()
    if "runner" in path_lower:
        return "runner"
    elif "killer" in path_lower:
        return "killer"
    elif "collector" in path_lower:
        return "collector"
    return None


def compute_pca_projection(features: np.ndarray, n_components: int = 2, random_state: int = 42):
    """
    Compute PCA projection on all features.
    This ensures all plots use the same PCA space.
    
    Args:
        features: np.ndarray [N, feature_dim] - all features
        n_components: number of PCA components (2 or 3)
        random_state: random seed for reproducibility
        
    Returns:
        pca: fitted PCA object
        projection: np.ndarray [N, n_components] - PCA projection of all data
    """
    from sklearn.decomposition import PCA
    
    pca = PCA(n_components=n_components, random_state=random_state)
    projection = pca.fit_transform(features)
    return pca, projection


def get_top_players(player_ids: list[str], n_top: int = 5):
    """
    Get top N players by data size.
    
    Returns:
        top_players: list of player IDs (sorted)
        player_mask: boolean array indicating if player is in top N
    """
    from collections import Counter
    
    player_counts = Counter(player_ids)
    top_players = sorted(
        sorted(player_counts.items(), key=lambda x: x[1], reverse=True)[:n_top],
        key=lambda x: x[0]  # Sort by player ID for consistency
    )
    top_player_ids = [pid for pid, _ in top_players]
    player_mask = np.array([pid in top_player_ids for pid in player_ids])
    
    return top_player_ids, player_mask


def get_player_type_indices(meta: list[dict]):
    """
    Get indices of traces that have player types in their filename.
    
    Returns:
        valid_indices: list of indices with player types
        player_types: list of player type strings (runner/killer/collector)
    """
    player_types = []
    valid_indices = []
    for i, m in enumerate(meta):
        player_type = extract_player_type(m["path"])
        if player_type is not None:
            player_types.append(player_type)
            valid_indices.append(i)
    return valid_indices, player_types


def plot_latent_2d_player_types(
    features: np.ndarray,
    meta: list[dict],
    save_path: Path,
):
    """
    Plot the 2D latent projection (PCA): color = player type (runner, killer, collector).
    Shows only traces that have a player type in the filename.
    Uses PCA fitted on all data to ensure consistent visualization.
    """
    print("[INFO] plotting 2D latent projection by PLAYER TYPE (PCA)...")
    
    # Compute PCA on all data
    _, projection_2d = compute_pca_projection(features, n_components=2)
    
    # Get player type indices
    valid_indices, player_types = get_player_type_indices(meta)
    
    if not valid_indices:
        print("[WARN] No traces found with player type (runner/killer/collector) in filename. Skipping plot.")
        return
    
    print(f"[INFO] Found {len(valid_indices)} traces with player types: {dict(zip(*np.unique(player_types, return_counts=True)))}")
    print(f"[INFO] Using PCA fitted on all {len(features)} samples, showing {len(valid_indices)} player type traces")

    # Filter to only player type traces
    filtered_projection = projection_2d[valid_indices]

    # Map player type to color
    type_to_color = {"runner": 0, "killer": 1, "collector": 2}
    type_colors = [type_to_color[pt] for pt in player_types]
    
    # Use a discrete colormap for the 3 types
    cmap = plt.get_cmap("Set1")
    vmin, vmax = -0.5, 2.5

    plt.figure(figsize=(8, 6))
    
    # Plot player type traces with colors
    scatter = plt.scatter(
        filtered_projection[:, 0],
        filtered_projection[:, 1],
        c=type_colors,
        s=10,
        alpha=0.8,
        cmap="Set1",
        vmin=vmin,
        vmax=vmax,
    )
    
    # Create custom legend
    from matplotlib.patches import Patch
    from matplotlib.colors import Normalize
    norm = Normalize(vmin=vmin, vmax=vmax)
    legend_elements = [
        Patch(facecolor=cmap(norm(0)), label="Runner"),
        Patch(facecolor=cmap(norm(1)), label="Killer"),
        Patch(facecolor=cmap(norm(2)), label="Collector"),
    ]
    plt.legend(handles=legend_elements, loc="best")
    
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Latent Space (PCA 2D, colored by player type)")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[INFO] saved 2D player type plot to: {save_path}")


def plot_latent_3d_player_types(
    features: np.ndarray,
    meta: list[dict],
    save_path: Path,
):
    """
    Plot the 3D latent projection (PCA): color = player type (runner, killer, collector).
    Shows only traces that have a player type in the filename.
    Uses PCA fitted on all data to ensure consistent visualization.
    """
    print("[INFO] plotting 3D latent projection by PLAYER TYPE (PCA)...")
    
    if features.shape[1] < 3:
        print("[WARN] feature_dim < 3, 3D PCA may not be able to visualize properly.")
    
    # Compute PCA on all data
    _, projection_3d = compute_pca_projection(features, n_components=3)
    
    # Get player type indices
    valid_indices, player_types = get_player_type_indices(meta)
    
    if not valid_indices:
        print("[WARN] No traces found with player type (runner/killer/collector) in filename. Skipping plot.")
        return
    
    print(f"[INFO] Found {len(valid_indices)} traces with player types: {dict(zip(*np.unique(player_types, return_counts=True)))}")
    print(f"[INFO] Using PCA fitted on all {len(features)} samples, showing {len(valid_indices)} player type traces")

    # Filter to only player type traces
    filtered_projection = projection_3d[valid_indices]

    # Map player type to color
    type_to_color = {"runner": 0, "killer": 1, "collector": 2}
    type_colors = [type_to_color[pt] for pt in player_types]
    
    # Use a discrete colormap for the 3 types
    cmap = plt.get_cmap("Set1")
    vmin, vmax = -0.5, 2.5

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")  # type: ignore
    
    # Plot player type traces with colors
    p = ax.scatter(
        filtered_projection[:, 0],
        filtered_projection[:, 1],
        filtered_projection[:, 2],
        c=type_colors,
        s=6,
        alpha=0.8,
        cmap="Set1",
        vmin=vmin,
        vmax=vmax,
    )
    
    # Create custom legend
    from matplotlib.patches import Patch
    from matplotlib.colors import Normalize
    norm = Normalize(vmin=vmin, vmax=vmax)
    legend_elements = [
        Patch(facecolor=cmap(norm(0)), label="Runner"),
        Patch(facecolor=cmap(norm(1)), label="Killer"),
        Patch(facecolor=cmap(norm(2)), label="Collector"),
    ]
    ax.legend(handles=legend_elements, loc="best")
    
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title("Latent Space (PCA 3D, colored by player type)")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[INFO] saved 3D player type plot to: {save_path}")


def run_kmeans(latents: np.ndarray, n_clusters: int, random_state: int = 42):
    from sklearn.cluster import KMeans

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init="auto",
    )
    labels = kmeans.fit_predict(latents)
    return labels, kmeans


def aggregate_by_player(meta, labels):
    """
    Create a representative cluster for each player based on the cluster labels for each trace.
    Here, we use the "majority vote (most frequent cluster)" as the representative cluster.
    """
    per_player = defaultdict(list)
    for m, c in zip(meta, labels):
        per_player[m["player_id"]].append(int(c))

    player_clusters = {}
    for pid, clusters in per_player.items():
        # Use the most frequent value as the representative
        vals, counts = np.unique(clusters, return_counts=True)
        majority_cluster = int(vals[np.argmax(counts)])
        player_clusters[pid] = {
            "majority_cluster": majority_cluster,
            "cluster_hist": {int(v): int(c) for v, c in zip(vals, counts)},
            "num_traces": int(len(clusters)),
        }
    return player_clusters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_clusters",
        type=int,
        default=8,
        help="Number of clusters (k-means k)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="clusters",
        help="Directory to save the clustering results",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["latent_only", "combined"],
        default="combined",
        help="Clustering mode: 'latent_only' uses only trajectory latents, 'combined' uses latents + game statistics",
    )
    parser.add_argument(
        "--no_timeout",
        action="store_true",
        help="Filter out TIME_OUT traces before clustering/PCA",
    )
    args = parser.parse_args()

    device = torch.device(train_config.device if torch.cuda.is_available() else "cpu")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Extract features based on mode
    if args.mode == "latent_only":
        print("[INFO] Extracting trajectory latent vectors only...")
        latents, meta = extract_latent(device, normalize=True)
        features = latents  # Use latents directly for clustering
        print(f"[INFO] Using {features.shape[1]} latent dimensions for clustering")
    else:  # combined
        print("[INFO] Extracting all features (latent vectors + game statistics)...")
        features, latents, meta = extract_all_features(device)
        print(f"[INFO] Using {features.shape[1]} combined features for clustering (latent: {latents.shape[1]}, additional: {features.shape[1] - latents.shape[1]})")

    if args.no_timeout:
        if args.mode == "latent_only":
            for m in meta:
                path = Path(m["path"])
                try:
                    with path.open("r") as f:
                        data = json.load(f)
                    m["status"] = data.get("status", "UNKNOWN")
                except:
                    m["status"] = "UNKNOWN"
        
        valid_mask = np.array([m.get("status") != "TIME_OUT" for m in meta])
        features = features[valid_mask]
        latents = latents[valid_mask]
        meta = [m for m, v in zip(meta, valid_mask) if v]
    
    # Extract the player_id list
    player_ids = [m["player_id"] for m in meta]

    # k-means clustering
    from sklearn.cluster import KMeans
    print(f"[INFO] Running KMeans clustering with k={args.n_clusters}...")
    kmeans = KMeans(n_clusters=args.n_clusters, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(features)

    # Show simple statistics for each cluster (for a rough idea)
    print("\n[INFO] cluster statistics:")
    for k in range(args.n_clusters):
        idx = labels == k
        n_points = int(idx.sum())
        players_in_cluster = sorted({player_ids[i] for i in range(len(player_ids)) if idx[i]})
        print(f"  - Cluster {k}: {n_points} points, {len(players_in_cluster)} players")

    # Save the results (use args.output_dir, not hardcoded "clusters")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create mode-specific subdirectory for organized file storage
    mode_suffix = "latent_only" if args.mode == "latent_only" else "combined"
    mode_dir = output_dir / mode_suffix
    mode_dir.mkdir(parents=True, exist_ok=True)

    # Save the cluster assignments for each sample
    sample_clusters_path = mode_dir / f"trajectory_clusters_k{args.n_clusters}.json"
    with sample_clusters_path.open("w") as f:
        json.dump(
            [
                {
                    "player_id": m["player_id"],
                    "path": str(m["path"]),
                    "length": m["length"],
                    "cluster": int(c),
                    "status": m.get("status", "UNKNOWN"),
                    "completing_ratio": m.get("completing_ratio", 0.0),
                    "kills": m.get("kills", 0),
                    "kills_by_fire": m.get("kills_by_fire", 0),
                    "kills_by_stomp": m.get("kills_by_stomp", 0),
                    "kills_by_shell": m.get("kills_by_shell", 0),
                    "lives": m.get("lives", 0),
                }
                for m, c in zip(meta, labels)
            ],
            f,
            indent=2,
        )
    print(f"[INFO] saved sample-level clusters to: {sample_clusters_path}")

    # Aggregate the cluster distributions for each player (if needed)
    player_clusters = defaultdict(list)
    for m, c in zip(meta, labels):
        player_clusters[m["player_id"]].append(int(c))
    player_clusters_path = mode_dir / f"player_clusters_k{args.n_clusters}.json"
    with player_clusters_path.open("w") as f:
        json.dump(player_clusters, f, indent=2)
    print(f"[INFO] saved player-level clusters to: {player_clusters_path}")

    # Save the raw latent vectors, features used for clustering, and centers
    np.save(mode_dir / "latents.npy", latents)
    if args.mode == "combined":
        np.save(mode_dir / "combined_features.npy", features)
    else:
        np.save(mode_dir / "latent_features.npy", features)
    np.save(mode_dir / f"kmeans_centers_k{args.n_clusters}.npy", kmeans.cluster_centers_)

    # ==== Start plotting ====
    plot_2d_cluster_path = mode_dir / f"latent_pca_2d_clusters_k{args.n_clusters}.png"
    plot_2d_player_path = mode_dir / f"latent_pca_2d_players_k{args.n_clusters}.png"
    plot_3d_cluster_path = mode_dir / f"latent_pca_3d_clusters_k{args.n_clusters}.png"
    plot_3d_player_path = mode_dir / f"latent_pca_3d_players_k{args.n_clusters}.png"
    plot_2d_type_path = mode_dir / f"latent_pca_2d_player_types_k{args.n_clusters}.png"
    plot_3d_type_path = mode_dir / f"latent_pca_3d_player_types_k{args.n_clusters}.png"

    plot_latent_2d_clusters(features, labels, plot_2d_cluster_path)
    plot_latent_2d_players(features, player_ids, plot_2d_player_path)
    plot_latent_3d_clusters(features, labels, plot_3d_cluster_path)
    plot_latent_3d_players(features, player_ids, plot_3d_player_path)
    plot_latent_2d_player_types(features, meta, plot_2d_type_path)
    plot_latent_3d_player_types(features, meta, plot_3d_type_path)

    print("[INFO] done.")

if __name__ == "__main__":
    main()
