import argparse
import json
from collections import Counter
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn_extra.cluster import KMedoids
from sklearn.manifold import MDS

from fastdtw import fastdtw

from .auto_encoder.config import data_config


# =========================
# DTW (fastdtw)
# =========================

def fastdtw_distance(trace1: np.ndarray, trace2: np.ndarray) -> float:
    """
    Use fastdtw to calculate the DTW distance between two traces.
    Args:
        trace1: shape = (T, D) or (T,) numpy array
        trace2: shape = (T, D) or (T,) numpy array
    """
    # fastdtw can accept columns of scalar or vector values
    # use L2 norm as the distance function
    def euclidean(a, b):
        a = np.asarray(a)
        b = np.asarray(b)
        return np.linalg.norm(a - b)

    distance, _ = fastdtw(trace1, trace2, dist=euclidean)
    return float(distance)


def pairwise_dtw_matrix_fast(traces: List[np.ndarray]) -> np.ndarray:
    """
    Use fastdtw to calculate the DTW distance matrix for all pairs.

    Args:
        traces: list of np.ndarray, each element shape=(T_i, D) or (T_i,)

    Returns:
        dist_mat: (N, N) symmetric distance matrix
    """
    n = len(traces)
    dist_mat = np.zeros((n, n), dtype=float)

    print(f"[INFO] Computing fastdtw distance matrix for {n} traces ...")
    for i in range(n):
        if (i + 1) % 20 == 0 or i == n - 1:
            print(f"  Progress: {i+1}/{n}")
        for j in range(i + 1, n):
            d = fastdtw_distance(traces[i], traces[j])
            dist_mat[i, j] = d
            dist_mat[j, i] = d

    return dist_mat


# =========================
# Data loading
# =========================

def load_trajectories(cfg=None, max_files: int = None) -> Tuple[List[np.ndarray], List[Dict]]:
    """
    Load raw trajectories from JSON files.

    Returns:
        traces: list[np.ndarray]
        meta: list[dict] (player_id, path, length) metadata
    """
    if cfg is None:
        cfg = data_config

    data_root = Path(cfg.data_root)
    files = sorted(data_root.glob("*.json"))

    if max_files is not None:
        files = files[:max_files]

    traces: List[np.ndarray] = []
    meta: List[Dict] = []

    print(f"[INFO] Loading trajectories from {data_root} ...")
    for f in files:
        try:
            with f.open("r") as fp:
                data = json.load(fp)
        except Exception:
            continue

        trace_data = data.get(cfg.trace_key, None)
        if trace_data is None or len(trace_data) < cfg.min_length:
            continue

        trace = np.array(trace_data, dtype=np.float32)

        stem = f.stem
        if "_" in stem:
            player_id = stem.split("_")[0]
        else:
            player_id = stem

        traces.append(trace)
        meta.append(
            {
                "player_id": player_id,
                "path": str(f),
                "length": len(trace),
            }
        )

    print(f"[INFO] Loaded {len(traces)} valid trajectories")
    return traces, meta


# =========================
# Clustering (precomputed)
# =========================

def cluster_traces_with_precomputed_dtw(
    dist_mat: np.ndarray,
    n_clusters: int,
    random_state: int = 42,
):
    """
    Use precomputed DTW distance matrix to cluster with KMedoids.

    Args:
        dist_mat: (N, N) symmetric DTW distance matrix
        n_clusters: number of clusters
    """
    print(f"[INFO] Running KMedoids with precomputed fastdtw distance (k={n_clusters}) ...")
    kmedoids = KMedoids(
        n_clusters=n_clusters,
        metric="precomputed",
        random_state=random_state,
    )
    labels = kmedoids.fit_predict(dist_mat)
    return labels, kmedoids


# =========================
# Visualization utilities (MDS)
# =========================

def plot_dtw_2d_clusters(
    dist_mat: np.ndarray,
    labels: np.ndarray,
    save_path: Path,
):
    """
    Perform MDS(2D) on the DTW distance matrix and color the clusters.
    """
    print("[INFO] Computing 2D MDS embedding from DTW distance matrix (clusters) ...")
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
    coords_2d = mds.fit_transform(dist_mat)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        coords_2d[:, 0],
        coords_2d[:, 1],
        c=labels,
        s=10,
        alpha=0.8,
        cmap="tab10",
    )
    plt.colorbar(scatter, label="Cluster ID")
    plt.xlabel("MDS Dim 1")
    plt.ylabel("MDS Dim 2")
    plt.title("DTW Distance Space (MDS 2D, colored by cluster)")

    unique_labels = np.unique(labels)
    for k in unique_labels:
        idx = labels == k
        if not np.any(idx):
            continue
        cx = coords_2d[idx, 0].mean()
        cy = coords_2d[idx, 1].mean()
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


def plot_dtw_3d_clusters(
    dist_mat: np.ndarray,
    labels: np.ndarray,
    save_path: Path,
):
    """
    Perform MDS(3D) on the DTW distance matrix and color the clusters.
    """
    print("[INFO] Computing 3D MDS embedding from DTW distance matrix (clusters) ...")
    mds = MDS(n_components=3, dissimilarity="precomputed", random_state=42)
    coords_3d = mds.fit_transform(dist_mat)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")  # type: ignore

    p = ax.scatter(
        coords_3d[:, 0],
        coords_3d[:, 1],
        coords_3d[:, 2],
        c=labels,
        s=6,
        alpha=0.8,
        cmap="tab10",
    )
    fig.colorbar(p, ax=ax, label="Cluster ID")
    ax.set_xlabel("MDS Dim 1")
    ax.set_ylabel("MDS Dim 2")
    ax.set_zlabel("MDS Dim 3")
    ax.set_title("DTW Distance Space (MDS 3D, colored by cluster)")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[INFO] saved 3D cluster plot to: {save_path}")


def plot_dtw_2d_players(
    dist_mat: np.ndarray,
    player_ids: List[str],
    save_path: Path,
):
    """
    Perform MDS(2D) on the DTW distance matrix and color the players.
    """
    print("[INFO] Computing 2D MDS embedding from DTW distance matrix (players) ...")
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
    coords_2d = mds.fit_transform(dist_mat)

    player_counts = Counter(player_ids)
    top_players = sorted(player_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    sampled_players = sorted([pid for pid, _ in top_players])
    player_sizes = {pid: count for pid, count in top_players}

    mask = np.array([pid in sampled_players for pid in player_ids])
    filtered_coords = coords_2d[mask]
    filtered_player_ids = [pid for pid, m in zip(player_ids, mask) if m]

    print(f"[INFO] sampling top {len(sampled_players)} players by data size: {[(pid, player_sizes[pid]) for pid in sampled_players]}")

    player_to_idx = {pid: i for i, pid in enumerate(sampled_players)}
    player_indices = np.array([player_to_idx[pid] for pid in filtered_player_ids])

    cmap = plt.get_cmap("tab20")
    colors = [cmap(i % cmap.N) for i in player_indices]

    plt.figure(figsize=(8, 6))
    plt.scatter(
        filtered_coords[:, 0],
        filtered_coords[:, 1],
        c=colors,
        s=10,
        alpha=0.8,
    )
    plt.xlabel("MDS Dim 1")
    plt.ylabel("MDS Dim 2")
    plt.title(f"DTW Distance Space (MDS 2D, colored by player - top {len(sampled_players)} players)")

    handles = []
    labels_legend = []
    for pid in sampled_players:
        idx = [j for j, p in enumerate(filtered_player_ids) if p == pid]
        if not idx:
            continue
        color = colors[idx[0]]
        h = plt.Line2D([], [], marker="o", linestyle="", markersize=5, color=color)
        handles.append(h)
        labels_legend.append(pid)
    if handles:
        plt.legend(handles, labels_legend, title="Sampled players", loc="best", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[INFO] saved 2D player plot to: {save_path}")


def extract_player_type(path_str: str) -> str | None:
    path_lower = path_str.lower()
    if "runner" in path_lower:
        return "runner"
    elif "killer" in path_lower:
        return "killer"
    elif "collector" in path_lower:
        return "collector"
    return None


def plot_dtw_2d_player_types(
    dist_mat: np.ndarray,
    meta: List[Dict],
    save_path: Path,
):
    """
    Perform MDS(2D) on the DTW distance matrix and color the player types.
    """
    print("[INFO] Computing 2D MDS embedding from DTW distance matrix (player types) ...")

    player_types = []
    valid_indices = []
    for i, m in enumerate(meta):
        ptype = extract_player_type(m["path"])
        if ptype is not None:
            player_types.append(ptype)
            valid_indices.append(i)

    if not valid_indices:
        print("[WARN] No traces found with player type. Skipping plot.")
        return

    filtered_dist_mat = dist_mat[np.ix_(valid_indices, valid_indices)]
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
    coords_2d = mds.fit_transform(filtered_dist_mat)

    print(f"[INFO] Found {len(coords_2d)} traces with player types: {dict(zip(*np.unique(player_types, return_counts=True)))}")

    type_to_color = {"runner": 0, "killer": 1, "collector": 2}
    type_colors = [type_to_color[pt] for pt in player_types]

    cmap = plt.get_cmap("Set1")
    vmin, vmax = -0.5, 2.5

    from matplotlib.colors import Normalize
    norm = Normalize(vmin=vmin, vmax=vmax)

    plt.figure(figsize=(8, 6))
    plt.scatter(
        coords_2d[:, 0],
        coords_2d[:, 1],
        c=type_colors,
        s=10,
        alpha=0.8,
        cmap="Set1",
        vmin=vmin,
        vmax=vmax,
    )

    legend_elements = [
        Patch(facecolor=cmap(norm(0)), label="Runner"),
        Patch(facecolor=cmap(norm(1)), label="Killer"),
        Patch(facecolor=cmap(norm(2)), label="Collector"),
    ]
    plt.legend(handles=legend_elements, loc="best")
    plt.xlabel("MDS Dim 1")
    plt.ylabel("MDS Dim 2")
    plt.title("DTW Distance Space (MDS 2D, colored by player type)")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[INFO] saved 2D player type plot to: {save_path}")


def plot_dtw_3d_player_types(
    dist_mat: np.ndarray,
    meta: List[Dict],
    save_path: Path,
):
    """
    Perform MDS(3D) on the DTW distance matrix and color the player types.
    """
    print("[INFO] Computing 3D MDS embedding from DTW distance matrix (player types) ...")

    player_types = []
    valid_indices = []
    for i, m in enumerate(meta):
        ptype = extract_player_type(m["path"])
        if ptype is not None:
            player_types.append(ptype)
            valid_indices.append(i)

    if not valid_indices:
        print("[WARN] No traces found with player type. Skipping plot.")
        return

    filtered_dist_mat = dist_mat[np.ix_(valid_indices, valid_indices)]
    mds = MDS(n_components=3, dissimilarity="precomputed", random_state=42)
    coords_3d = mds.fit_transform(filtered_dist_mat)

    print(f"[INFO] Found {len(coords_3d)} traces with player types: {dict(zip(*np.unique(player_types, return_counts=True)))}")

    type_to_color = {"runner": 0, "killer": 1, "collector": 2}
    type_colors = [type_to_color[pt] for pt in player_types]

    cmap = plt.get_cmap("Set1")
    vmin, vmax = -0.5, 2.5
    from matplotlib.colors import Normalize
    norm = Normalize(vmin=vmin, vmax=vmax)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")  # type: ignore

    p = ax.scatter(
        coords_3d[:, 0],
        coords_3d[:, 1],
        coords_3d[:, 2],
        c=type_colors,
        s=6,
        alpha=0.8,
        cmap="Set1",
        vmin=vmin,
        vmax=vmax,
    )

    legend_elements = [
        Patch(facecolor=cmap(norm(0)), label="Runner"),
        Patch(facecolor=cmap(norm(1)), label="Killer"),
        Patch(facecolor=cmap(norm(2)), label="Collector"),
    ]
    ax.legend(handles=legend_elements, loc="best")

    ax.set_xlabel("MDS Dim 1")
    ax.set_ylabel("MDS Dim 2")
    ax.set_zlabel("MDS Dim 3")
    ax.set_title("DTW Distance Space (MDS 3D, colored by player type)")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[INFO] saved 3D player type plot to: {save_path}")


# =========================
# main
# =========================

def main():
    parser = argparse.ArgumentParser(description="Cluster trajectories using fastdtw + precomputed KMedoids")
    parser.add_argument(
        "--n_clusters",
        type=int,
        default=8,
        help="Number of clusters (k)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="clusters_dtw_fast",
        help="Directory to save clustering results",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Maximum number of trajectory files to process (for testing)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load trajectories
    print("[INFO] Loading trajectory data ...")
    traces, meta = load_trajectories(max_files=args.max_files)
    if len(traces) == 0:
        print("[ERROR] No trajectories loaded. Exiting.")
        return

    player_ids = [m["player_id"] for m in meta]

    # 2) Compute distance matrix with fastdtw
    dist_mat = pairwise_dtw_matrix_fast(traces)

    # 3) Cluster with KMedoids using precomputed distance
    labels, kmedoids = cluster_traces_with_precomputed_dtw(
        dist_mat,
        n_clusters=args.n_clusters,
        random_state=42,
    )

    # 4) Cluster statistics
    print("\n[INFO] Cluster statistics:")
    for k in range(args.n_clusters):
        idx = labels == k
        n_points = int(idx.sum())
        players_in_cluster = sorted({player_ids[i] for i in range(len(player_ids)) if idx[i]})
        print(f"  - Cluster {k}: {n_points} points, {len(players_in_cluster)} players")

    # 5) Save results
    sample_clusters_path = output_dir / f"fastdtw_trajectory_clusters_k{args.n_clusters}.json"
    with sample_clusters_path.open("w") as f:
        json.dump(
            [
                {
                    "player_id": m["player_id"],
                    "path": str(m["path"]),
                    "length": m["length"],
                    "cluster": int(c),
                }
                for m, c in zip(meta, labels)
            ],
            f,
            indent=2,
        )
    print(f"[INFO] saved sample-level clusters to: {sample_clusters_path}")

    np.save(output_dir / f"fastdtw_distance_matrix_k{args.n_clusters}.npy", dist_mat)

    # 6) Visualization
    print("\n[INFO] Generating plots ...")
    plot_2d_cluster_path = output_dir / f"fastdtw_mds_2d_clusters_k{args.n_clusters}.png"
    plot_3d_cluster_path = output_dir / f"fastdtw_mds_3d_clusters_k{args.n_clusters}.png"
    plot_2d_player_path = output_dir / f"fastdtw_mds_2d_players_k{args.n_clusters}.png"
    plot_2d_type_path = output_dir / f"fastdtw_mds_2d_player_types_k{args.n_clusters}.png"
    plot_3d_type_path = output_dir / f"fastdtw_mds_3d_player_types_k{args.n_clusters}.png"

    plot_dtw_2d_clusters(dist_mat, labels, plot_2d_cluster_path)
    plot_dtw_3d_clusters(dist_mat, labels, plot_3d_cluster_path)
    plot_dtw_2d_players(dist_mat, player_ids, plot_2d_player_path)
    plot_dtw_2d_player_types(dist_mat, meta, plot_2d_type_path)
    plot_dtw_3d_player_types(dist_mat, meta, plot_3d_type_path)

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
