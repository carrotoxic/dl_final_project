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


def extract_latent(device: torch.device):
    """
    Extract latent vectors from all traces using the trained LSTM Autoencoder.
    Returns:
        latents: np.ndarray [N, latent_dim]
        meta: list[dict]  metadata for each sample (player_id, path, length)
    """
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
    latents, meta = extract_latent(device)
    
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
        elif status == "timeout":
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


def plot_latent_2d_clusters(
    latents: np.ndarray,
    labels: np.ndarray,
    save_path: Path,
):
    """
    Plot the 2D latent projection (PCA): color = cluster ID, show Ck label at the center.
    """
    from sklearn.decomposition import PCA

    print("[INFO] plotting 2D latent projection by CLUSTER (PCA)...")
    pca = PCA(n_components=2, random_state=42)
    lat_2d = pca.fit_transform(latents)  # [N, 2]

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        lat_2d[:, 0],
        lat_2d[:, 1],
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
    unique_labels = np.unique(labels)
    for k in unique_labels:
        idx = labels == k
        if not np.any(idx):
            continue
        cx = lat_2d[idx, 0].mean()
        cy = lat_2d[idx, 1].mean()
        plt.text(
            cx,
            cy,
            f"C{k}",
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
    latents: np.ndarray,
    player_ids: list[str],
    save_path: Path,
):
    """
    Plot the 2D latent projection (PCA): color = player ID
    This plot is to see which points belong to the same player.
    Only samples 5 players for visualization.
    """
    from sklearn.decomposition import PCA

    print("[INFO] plotting 2D latent projection by PLAYER (PCA)...")
    
    # Count data points per player and select top 5 by data size
    from collections import Counter
    player_counts = Counter(player_ids)
    # Sort by count (descending) and take top 5
    top_players = sorted(player_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    sampled_players = sorted([pid for pid, _ in top_players])
    player_sizes = {pid: count for pid, count in top_players}
    
    # Filter latents and player_ids to only include sampled players
    mask = np.array([pid in sampled_players for pid in player_ids])
    filtered_latents = latents[mask]
    filtered_player_ids = [pid for pid, m in zip(player_ids, mask) if m]
    
    print(f"[INFO] sampling top {len(sampled_players)} players by data size: {[(pid, player_sizes[pid]) for pid in sampled_players]}")
    
    pca = PCA(n_components=2, random_state=42)
    lat_2d = pca.fit_transform(filtered_latents)  # [N, 2]

    # Map player_id -> index for sampled players
    player_to_idx = {pid: i for i, pid in enumerate(sampled_players)}
    player_indices = np.array([player_to_idx[pid] for pid in filtered_player_ids])

    # Color map: loop through the number of players using tab20
    cmap = plt.get_cmap("tab20")
    colors = [cmap(i % cmap.N) for i in player_indices]

    plt.figure(figsize=(8, 6))
    plt.scatter(
        lat_2d[:, 0],
        lat_2d[:, 1],
        c=colors,
        s=10,
        alpha=0.8,
    )
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"Latent Space (PCA 2D, colored by player - top {len(sampled_players)} players by data size)")

    # Show legend for all sampled players
    handles = []
    labels = []
    for pid in sampled_players:
        idx = [j for j, p in enumerate(filtered_player_ids) if p == pid]
        if not idx:
            continue
        color = colors[idx[0]]
        h = plt.Line2D(
            [], [], marker="o", linestyle="", markersize=5, color=color
        )
        handles.append(h)
        labels.append(pid)
    if handles:
        plt.legend(
            handles,
            labels,
            title="Sampled players",
            loc="best",
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[INFO] saved 2D player plot to: {save_path}")


def plot_latent_3d_players(
    latents: np.ndarray,
    player_ids: list[str],
    save_path: Path,
):
    """
    Plot the 3D latent projection (PCA): color = player ID
    This plot is to see which points belong to the same player.
    Only samples top 5 players by data size for visualization.
    """
    from sklearn.decomposition import PCA

    print("[INFO] plotting 3D latent projection by PLAYER (PCA)...")
    
    if latents.shape[1] < 3:
        print("[WARN] latent_dim < 3, 3D PCA may not be able to visualize properly.")
    
    # Count data points per player and select top 5 by data size
    from collections import Counter
    player_counts = Counter(player_ids)
    # Sort by count (descending) and take top 5
    top_players = sorted(player_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    sampled_players = sorted([pid for pid, _ in top_players])
    player_sizes = {pid: count for pid, count in top_players}
    
    # Filter latents and player_ids to only include sampled players
    mask = np.array([pid in sampled_players for pid in player_ids])
    filtered_latents = latents[mask]
    filtered_player_ids = [pid for pid, m in zip(player_ids, mask) if m]
    
    print(f"[INFO] sampling top {len(sampled_players)} players by data size: {[(pid, player_sizes[pid]) for pid in sampled_players]}")
    
    pca = PCA(n_components=3, random_state=42)
    lat_3d = pca.fit_transform(filtered_latents)  # [N, 3]

    # Map player_id -> index for sampled players
    player_to_idx = {pid: i for i, pid in enumerate(sampled_players)}
    player_indices = np.array([player_to_idx[pid] for pid in filtered_player_ids])

    # Color map: loop through the number of players using tab20
    cmap = plt.get_cmap("tab20")
    colors = [cmap(i % cmap.N) for i in player_indices]

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")  # type: ignore

    p = ax.scatter(
        lat_3d[:, 0],
        lat_3d[:, 1],
        lat_3d[:, 2],
        c=colors,
        s=6,
        alpha=0.8,
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title(f"Latent Space (PCA 3D, colored by player - top {len(sampled_players)} players by data size)")

    # Show legend for all sampled players
    handles = []
    labels_legend = []
    for pid in sampled_players:
        idx = [j for j, p in enumerate(filtered_player_ids) if p == pid]
        if not idx:
            continue
        color = colors[idx[0]]
        h = plt.Line2D(
            [], [], marker="o", linestyle="", markersize=5, color=color
        )
        handles.append(h)
        labels_legend.append(pid)
    if handles:
        ax.legend(
            handles,
            labels_legend,
            title="Sampled players",
            loc="best",
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[INFO] saved 3D player plot to: {save_path}")


def plot_latent_3d_clusters(
    latents: np.ndarray,
    labels: np.ndarray,
    save_path: Path,
):
    """
    Plot the 3D latent projection (PCA): color = cluster ID
    """
    from sklearn.decomposition import PCA

    print("[INFO] plotting 3D latent projection by CLUSTER (PCA)...")
    if latents.shape[1] < 3:
        print("[WARN] latent_dim < 3, 3D PCA may not be able to visualize properly.")

    pca = PCA(n_components=3, random_state=42)
    lat_3d = pca.fit_transform(latents)  # [N, 3]

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")  # type: ignore

    p = ax.scatter(
        lat_3d[:, 0],
        lat_3d[:, 1],
        lat_3d[:, 2],
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


def plot_latent_2d_player_types(
    latents: np.ndarray,
    meta: list[dict],
    save_path: Path,
):
    """
    Plot the 2D latent projection (PCA): color = player type (runner, killer, collector).
    Only includes traces that have a player type in the filename.
    """
    from sklearn.decomposition import PCA

    print("[INFO] plotting 2D latent projection by PLAYER TYPE (PCA)...")
    
    # Extract player types and filter
    player_types = []
    valid_indices = []
    for i, m in enumerate(meta):
        player_type = extract_player_type(m["path"])
        if player_type is not None:
            player_types.append(player_type)
            valid_indices.append(i)
    
    if not valid_indices:
        print("[WARN] No traces found with player type (runner/killer/collector) in filename. Skipping plot.")
        return
    
    # Filter latents to only include valid traces
    filtered_latents = latents[valid_indices]
    
    print(f"[INFO] Found {len(filtered_latents)} traces with player types: {dict(zip(*np.unique(player_types, return_counts=True)))}")
    
    pca = PCA(n_components=2, random_state=42)
    lat_2d = pca.fit_transform(filtered_latents)  # [N, 2]

    # Map player type to color
    type_to_color = {"runner": 0, "killer": 1, "collector": 2}
    type_colors = [type_to_color[pt] for pt in player_types]
    
    # Use a discrete colormap for the 3 types
    cmap = plt.get_cmap("Set1")
    vmin, vmax = -0.5, 2.5

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        lat_2d[:, 0],
        lat_2d[:, 1],
        c=type_colors,
        s=10,
        alpha=0.8,
        cmap="Set1",
        vmin=vmin,
        vmax=vmax,
    )
    
    # Create custom legend with normalized colors to match scatter plot
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
    latents: np.ndarray,
    meta: list[dict],
    save_path: Path,
):
    """
    Plot the 3D latent projection (PCA): color = player type (runner, killer, collector).
    Only includes traces that have a player type in the filename.
    """
    from sklearn.decomposition import PCA

    print("[INFO] plotting 3D latent projection by PLAYER TYPE (PCA)...")
    
    if latents.shape[1] < 3:
        print("[WARN] latent_dim < 3, 3D PCA may not be able to visualize properly.")
    
    # Extract player types and filter
    player_types = []
    valid_indices = []
    for i, m in enumerate(meta):
        player_type = extract_player_type(m["path"])
        if player_type is not None:
            player_types.append(player_type)
            valid_indices.append(i)
    
    if not valid_indices:
        print("[WARN] No traces found with player type (runner/killer/collector) in filename. Skipping plot.")
        return
    
    # Filter latents to only include valid traces
    filtered_latents = latents[valid_indices]
    
    print(f"[INFO] Found {len(filtered_latents)} traces with player types: {dict(zip(*np.unique(player_types, return_counts=True)))}")
    
    pca = PCA(n_components=3, random_state=42)
    lat_3d = pca.fit_transform(filtered_latents)  # [N, 3]

    # Map player type to color
    type_to_color = {"runner": 0, "killer": 1, "collector": 2}
    type_colors = [type_to_color[pt] for pt in player_types]
    
    # Use a discrete colormap for the 3 types
    cmap = plt.get_cmap("Set1")
    vmin, vmax = -0.5, 2.5

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")  # type: ignore

    p = ax.scatter(
        lat_3d[:, 0],
        lat_3d[:, 1],
        lat_3d[:, 2],
        c=type_colors,
        s=6,
        alpha=0.8,
        cmap="Set1",
        vmin=vmin,
        vmax=vmax,
    )
    
    # Create custom legend with normalized colors to match scatter plot
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
    args = parser.parse_args()

    device = torch.device(train_config.device if torch.cuda.is_available() else "cpu")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) extract all features (latent + game statistics)
    print("[INFO] extracting all features (latent vectors + game statistics)...")
    features, latents, meta = extract_all_features(device)

    # Extract the player_id list
    player_ids = [m["player_id"] for m in meta]

    # k-means clustering on combined features
    from sklearn.cluster import KMeans
    print(f"[INFO] Clustering with {features.shape[1]} features per sample...")
    kmeans = KMeans(n_clusters=args.n_clusters, random_state=42)
    labels = kmeans.fit_predict(features)

    # Show simple statistics for each cluster (for a rough idea)
    print("\n[INFO] cluster statistics:")
    for k in range(args.n_clusters):
        idx = labels == k
        n_points = int(idx.sum())
        players_in_cluster = sorted({player_ids[i] for i in range(len(player_ids)) if idx[i]})
        print(f"  - Cluster {k}: {n_points} points, {len(players_in_cluster)} players")

    # Save the results
    output_dir = Path("clusters")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save the cluster assignments for each sample
    sample_clusters_path = output_dir / f"trajectory_clusters_k{args.n_clusters}.json"
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
    player_clusters_path = output_dir / f"player_clusters_k{args.n_clusters}.json"
    with player_clusters_path.open("w") as f:
        json.dump(player_clusters, f, indent=2)
    print(f"[INFO] saved player-level clusters to: {player_clusters_path}")

    # Save the raw latent vectors, combined features, and centers
    np.save(output_dir / "latents.npy", latents)
    np.save(output_dir / "combined_features.npy", features)
    np.save(output_dir / f"kmeans_centers_k{args.n_clusters}.npy", kmeans.cluster_centers_)

    # ==== Start plotting ====
    plot_2d_cluster_path = output_dir / f"latent_pca_2d_clusters_k{args.n_clusters}.png"
    plot_2d_player_path = output_dir / f"latent_pca_2d_players_k{args.n_clusters}.png"
    plot_3d_cluster_path = output_dir / f"latent_pca_3d_clusters_k{args.n_clusters}.png"
    plot_3d_player_path = output_dir / f"latent_pca_3d_players_k{args.n_clusters}.png"
    plot_2d_type_path = output_dir / f"latent_pca_2d_player_types_k{args.n_clusters}.png"
    plot_3d_type_path = output_dir / f"latent_pca_3d_player_types_k{args.n_clusters}.png"

    plot_latent_2d_clusters(latents, labels, plot_2d_cluster_path)
    plot_latent_2d_players(latents, player_ids, plot_2d_player_path)
    plot_latent_3d_clusters(latents, labels, plot_3d_cluster_path)
    plot_latent_3d_players(latents, player_ids, plot_3d_player_path)
    plot_latent_2d_player_types(latents, meta, plot_2d_type_path)
    plot_latent_3d_player_types(latents, meta, plot_3d_type_path)

    print("[INFO] done.")

if __name__ == "__main__":
    main()
