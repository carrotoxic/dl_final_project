import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .auto_encoder.config import data_config, model_config, train_config
from .auto_encoder.datasets.trajectory_datasets import TrajectoryDataset, trajectory_collate_fn
from .auto_encoder.models.lstm_autoencoder import LSTMAutoencoder


def extract_latent(device: torch.device, normalize: bool = True):
    """
    Extract latent vectors from all traces using the trained LSTM Autoencoder.
    
    Args:
        device: torch device
        normalize: If True, normalize latents using StandardScaler
        
    Returns:
        latents: np.ndarray [N, latent_dim] (normalized if normalize=True)
        meta: list[dict] metadata for each sample (player_id, path, length)
    """
    from sklearn.preprocessing import StandardScaler
    
    dataset = TrajectoryDataset(data_config, normalize=True)
    dataloader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=False,
        collate_fn=trajectory_collate_fn,
        num_workers=1,
        pin_memory=True,
    )

    model = LSTMAutoencoder(model_config).to(device)
    ckpt_path = train_config.model_save_path
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

            z = model.encode(trajectories, lengths)
            all_latents.append(z.cpu().numpy())
            
            for pid, pth, L in zip(player_ids, paths, lengths.cpu().tolist()):
                meta.append({
                    "player_id": pid,
                    "path": str(pth),
                    "length": L,
                })

    latents = np.concatenate(all_latents, axis=0)
    
    if normalize:
        scaler = StandardScaler()
        latents = scaler.fit_transform(latents)
    
    return latents, meta


def extract_all_features(device: torch.device):
    """
    Extract all features: latent vectors + game statistics.
    
    Returns:
        features: np.ndarray [N, feature_dim] - combined feature vector
        latents: np.ndarray [N, latent_dim] - latent vectors only
        meta: list[dict] - metadata for each sample
        non_trajectory_features: np.ndarray [N, 9] - game statistics (status one-hot + numeric features)
    """
    from sklearn.preprocessing import StandardScaler
    
    latents, meta = extract_latent(device, normalize=True)
    
    print("[INFO] Loading additional features from JSON files...")
    status_list = []
    numeric_features = []
    valid_indices = []
    
    for i, m in enumerate(meta):
        path = Path(m["path"])
        try:
            with path.open("r") as f:
                data = json.load(f)
            
            status = data.get("status", "UNKNOWN")
            completing_ratio = data.get("completing-ratio", 0.0)
            kills = data.get("#kills", 0)
            kills_by_fire = data.get("#kills-by-fire", 0)
            kills_by_stomp = data.get("#kills-by-stomp", 0)
            kills_by_shell = data.get("#kills-by-shell", 0)
            lives = data.get("lives", 0)
            coins = data.get("#coins", data.get("currentCoins", 0))  # Try both field names
            
            status_list.append(status)
            numeric_features.append([
                completing_ratio,
                float(kills),
                float(kills_by_fire),
                float(kills_by_stomp),
                float(kills_by_shell),
                float(lives),
                float(coins),
            ])
            valid_indices.append(i)
            
            m.update({
                "status": status,
                "completing_ratio": completing_ratio,
                "kills": kills,
                "kills_by_fire": kills_by_fire,
                "kills_by_stomp": kills_by_stomp,
                "kills_by_shell": kills_by_shell,
                "lives": lives,
                "coins": coins,
            })
        except Exception as e:
            print(f"[WARN] Failed to load features from {path}: {e}")
            continue
    
    if len(numeric_features) != len(latents):
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
            status_onehot.append([0.0, 1.0, 0.0])
    
    status_onehot = np.array(status_onehot, dtype=np.float32)
    numeric_features = np.array(numeric_features, dtype=np.float32)
    
    # Normalize numeric features
    scaler = StandardScaler()
    numeric_features_scaled = scaler.fit_transform(numeric_features)
    
    # Combine: status_onehot + numeric_features (non-trajectory features)
    non_trajectory_features = np.concatenate([status_onehot, numeric_features_scaled], axis=1)
    
    # Combine latent vectors with additional features
    features = np.concatenate([latents, non_trajectory_features], axis=1)
    
    print(f"[INFO] Combined features shape: {features.shape}")
    print(f"[INFO]   - Latent: {latents.shape[1]} dims")
    print(f"[INFO]   - Non-trajectory: {non_trajectory_features.shape[1]} dims (status: 3, numeric: 7)")
    
    return features, latents, meta, non_trajectory_features


def extract_player_type(path_str: str) -> str | None:
    """Extract player type from path string."""
    path_lower = path_str.lower()
    if "runner" in path_lower:
        return "runner"
    elif "killer" in path_lower:
        return "killer"
    elif "collector" in path_lower:
        return "collector"
    return None


def assign_players_to_clusters(
    meta: list[dict],
    labels: np.ndarray,
    non_trajectory_features: np.ndarray | None = None,
) -> list[dict]:
    """
    For each player, count trajectories in each cluster and assign to majority cluster.
    
    Args:
        meta: List of metadata dictionaries
        labels: Cluster labels for each trajectory
        non_trajectory_features: Optional array of non-trajectory features (None for latent_only mode)
    
    Returns:
        List of dicts with player_id, cluster, mean_non_trajectory_features, player_type_counts
    """
    # Group trajectories by player
    player_trajectories = defaultdict(list)
    for i, m in enumerate(meta):
        player_id = m["player_id"]
        # Extract raw (non-normalized) numeric features from metadata
        # (Always extract from metadata, regardless of mode, for CSV output)
        raw_features = [
            1.0 if m.get("status") == "WIN" else 0.0,
            1.0 if m.get("status") == "LOSE" else 0.0,
            1.0 if m.get("status") == "TIME_OUT" else 0.0,
            m.get("completing_ratio", 0.0),
            float(m.get("kills", 0)),
            float(m.get("kills_by_fire", 0)),
            float(m.get("kills_by_stomp", 0)),
            float(m.get("kills_by_shell", 0)),
            float(m.get("lives", 0)),
            float(m.get("coins", 0)),
        ]
        player_trajectories[player_id].append({
            "index": i,
            "cluster": int(labels[i]),
            "player_type": extract_player_type(m["path"]),
            "raw_features": raw_features,  # Use raw (non-normalized) values
            "length": m["length"],
        })
    
    results = []
    for player_id, trajectories in player_trajectories.items():
        # Count trajectories per cluster
        cluster_counts = Counter(t["cluster"] for t in trajectories)
        majority_cluster = cluster_counts.most_common(1)[0][0]
        
        # Calculate mean non-trajectory features for this player (using raw values)
        player_features = np.array([t["raw_features"] for t in trajectories])
        mean_features = player_features.mean(axis=0).tolist()
        
        # Calculate mean trajectory length for this player
        trajectory_lengths = [t["length"] for t in trajectories]
        mean_trajectory_length = np.mean(trajectory_lengths)
        
        # Total number of runs (trajectories) for this player
        total_runs = len(trajectories)
        
        # Count trajectories by player type and calculate ratios
        player_type_counts = Counter(t["player_type"] for t in trajectories if t["player_type"] is not None)
        total_typed_trajectories = sum(player_type_counts.values())
        
        # Calculate ratios (percentages) for each player type
        if total_typed_trajectories > 0:
            runner_ratio = player_type_counts.get("runner", 0) / total_typed_trajectories
            killer_ratio = player_type_counts.get("killer", 0) / total_typed_trajectories
            collector_ratio = player_type_counts.get("collector", 0) / total_typed_trajectories
        else:
            runner_ratio = killer_ratio = collector_ratio = 0.0
        
        results.append({
            "player_id": player_id,
            "cluster": majority_cluster,
            "mean_non_trajectory_features": mean_features,
            "mean_trajectory_length": mean_trajectory_length,
            "total_runs": total_runs,
            "runner_ratio": runner_ratio,
            "killer_ratio": killer_ratio,
            "collector_ratio": collector_ratio,
        })
    
    return sorted(results, key=lambda x: (x["cluster"], x["player_id"]))


def save_results_to_csv(results: list[dict], output_path: Path, n_clusters: int):
    """Save clustering results to CSV file."""
    # Create feature column names
    feature_names = [
        "status_WIN", "status_LOSE", "status_timeout",
        "completing_ratio", "kills", "kills_by_fire",
        "kills_by_stomp", "kills_by_shell", "lives", "coins"
    ]
    
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        
        # Header
        header = [
            "player_id",
            "cluster",
        ] + [f"mean_{name}" for name in feature_names] + [
            "mean_trajectory_length",
            "total_runs",
            "runner_ratio",
            "killer_ratio",
            "collector_ratio",
        ]
        writer.writerow(header)
        
        # Data rows
        for result in results:
            row = [
                result["player_id"],
                result["cluster"],
            ] + result["mean_non_trajectory_features"] + [
                result["mean_trajectory_length"],
                result["total_runs"],
                result["runner_ratio"],
                result["killer_ratio"],
                result["collector_ratio"],
            ]
            writer.writerow(row)
    
    print(f"[INFO] Saved results to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Cluster players using combined features and assign each player to majority cluster"
    )
    parser.add_argument(
        "--n_clusters",
        type=int,
        default=None,
        help="Number of clusters (k-means k). If not provided, will loop from 2 to 10.",
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
        "--no_time_out",
        action="store_true",
        help="Filter out TIME_OUT traces before clustering/PCA",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["kmeans", "gmm"],
        default="kmeans",
        help="Clustering method: 'kmeans' or 'gmm' (Gaussian Mixture Model)",
    )
    args = parser.parse_args()

    device = torch.device(train_config.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create method-specific subdirectory (kmeans or gmm)
    method_dir = output_dir / args.method
    method_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine cluster range
    if args.n_clusters is None:
        cluster_range = range(2, 11)  # 2 to 10
    else:
        cluster_range = [args.n_clusters]

    # Extract features based on mode
    if args.mode == "latent_only":
        print("[INFO] Extracting trajectory latent vectors only...")
        latents, meta = extract_latent(device, normalize=True)
        features = latents
        non_trajectory_features = None
        print(f"[INFO] Using {features.shape[1]} latent dimensions for clustering")
        
        # Load game statistics into metadata for CSV output (even though not used for clustering)
        print("[INFO] Loading game statistics for CSV output...")
        for m in meta:
            path = Path(m["path"])
            try:
                with path.open("r") as f:
                    data = json.load(f)
                m.update({
                    "status": data.get("status", "UNKNOWN"),
                    "completing_ratio": data.get("completing-ratio", 0.0),
                    "kills": data.get("#kills", 0),
                    "kills_by_fire": data.get("#kills-by-fire", 0),
                    "kills_by_stomp": data.get("#kills-by-stomp", 0),
                    "kills_by_shell": data.get("#kills-by-shell", 0),
                    "lives": data.get("lives", 0),
                    "coins": data.get("#coins", data.get("currentCoins", 0)),
                })
            except Exception as e:
                print(f"[WARN] Failed to load features from {path}: {e}")
                # Set defaults if loading fails
                m.update({
                    "status": "UNKNOWN",
                    "completing_ratio": 0.0,
                    "kills": 0,
                    "kills_by_fire": 0,
                    "kills_by_stomp": 0,
                    "kills_by_shell": 0,
                    "lives": 0,
                    "coins": 0,
                })
    else:  # combined
        print("[INFO] Extracting combined features (latent vectors + game statistics)...")
        features, latents, meta, non_trajectory_features = extract_all_features(device)
        print(f"[INFO] Using {features.shape[1]} combined features for clustering (latent: {latents.shape[1]}, additional: {features.shape[1] - latents.shape[1]})")

    if args.no_time_out:
        valid_mask = np.array([m.get("status") != "TIME_OUT" for m in meta])
        features = features[valid_mask]
        latents = latents[valid_mask]
        meta = [m for m, v in zip(meta, valid_mask) if v]
        if non_trajectory_features is not None:
            non_trajectory_features = non_trajectory_features[valid_mask]

    # Loop through cluster numbers
    for n_clusters in cluster_range:
        print(f"\n{'='*80}")
        print(f"Processing k={n_clusters}")
        print(f"{'='*80}")
        
        # Create mode-specific subdirectory for this k
        mode_dir = method_dir / args.mode / f"k{n_clusters}"
        mode_dir.mkdir(parents=True, exist_ok=True)
        
        # Clustering
        if args.method == "kmeans":
            from sklearn.cluster import KMeans
            print(f"[INFO] Running KMeans clustering with k={n_clusters}...")
            print(f"[INFO] KMeans parameters: n_init=100, max_iter=10000, tol=0.0001")
            clusterer = KMeans(
                n_clusters=n_clusters,
                n_init=100,              # 100 restarts as per paper
                max_iter=10000,          # Maximum iterations as per paper
                tol=0.0001,              # Tolerance as per paper
                random_state=42,
            )
        else:  # gmm
            from sklearn.mixture import GaussianMixture
            print(f"[INFO] Running GMM clustering with k={n_clusters}...")
            print(f"[INFO] GMM parameters: n_init=100, max_iter=10000, covariance_type='full'")
            clusterer = GaussianMixture(
                n_components=n_clusters,
                n_init=100,              # 100 restarts as per paper
                max_iter=10000,          # Maximum iterations as per paper
                covariance_type='full',   # Full covariance as per paper
                random_state=42,
            )
        
        labels = clusterer.fit_predict(features)

        # Show cluster statistics
        print("\n[INFO] Cluster statistics:")
        for k in range(n_clusters):
            idx = labels == k
            n_points = int(idx.sum())
            player_ids = [m["player_id"] for m in meta]
            players_in_cluster = len(set(player_ids[i] for i in range(len(player_ids)) if idx[i]))
            print(f"  - Cluster {k}: {n_points} trajectories, {players_in_cluster} players")

        # Save trajectory-level cluster assignments (for statistics consistency)
        trajectory_clusters_path = mode_dir / f"trajectory_clusters_k{n_clusters}.json"
        with trajectory_clusters_path.open("w") as f:
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
                        "coins": m.get("coins", 0),
                    }
                    for m, c in zip(meta, labels)
                ],
                f,
                indent=2,
            )
        print(f"[INFO] Saved trajectory-level clusters to: {trajectory_clusters_path}")

        # Assign players to clusters
        print("\n[INFO] Assigning players to clusters...")
        results = assign_players_to_clusters(meta, labels, non_trajectory_features)
        
        # Save results to CSV with mode in filename
        output_path = mode_dir / f"player_clusters_{args.mode}_k{n_clusters}.csv"
        save_results_to_csv(results, output_path, n_clusters)
        
        print(f"[INFO] Processed {len(results)} players for k={n_clusters}")
    
    print(f"\n{'='*80}")
    print(f"[INFO] Done processing all cluster numbers.")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

