import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def load_cluster_data(csv_path: Path, json_path: Path | None = None):
    """
    Load cluster data from CSV file (player-level) and optionally JSON file (trajectory-level).
    
    Returns:
        players_by_cluster: dict mapping cluster_id -> list of player rows from CSV
        trajectories_by_cluster: dict mapping cluster_id -> count of trajectories (from JSON if available)
    """
    players_by_cluster = defaultdict(list)
    
    with csv_path.open("r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cluster = int(row["cluster"])
            players_by_cluster[cluster].append(row)
    
    # Load trajectory-level counts from JSON if available
    trajectories_by_cluster = defaultdict(int)
    if json_path and json_path.exists():
        with json_path.open("r") as f:
            trajectory_data = json.load(f)
            print(f"[DEBUG] Loaded {len(trajectory_data)} trajectories from JSON")
            for traj in trajectory_data:
                cluster = int(traj["cluster"])
                trajectories_by_cluster[cluster] += 1
            print(f"[DEBUG] Trajectory distribution: {dict(trajectories_by_cluster)}")
    else:
        # Fallback: sum total_runs from players (less accurate if players split across clusters)
        print(f"[DEBUG] JSON not found, using player-level counts")
        for cluster_id, players in players_by_cluster.items():
            trajectories_by_cluster[cluster_id] = sum(int(p.get("total_runs", 0)) for p in players)
    
    return players_by_cluster, trajectories_by_cluster


def calculate_cluster_statistics(players_by_cluster: dict, trajectories_by_cluster: dict):
    """Calculate statistics for each cluster."""
    feature_names = [
        "mean_status_WIN", "mean_status_LOSE", "mean_status_timeout",
        "mean_completing_ratio", "mean_kills", "mean_kills_by_fire",
        "mean_kills_by_stomp", "mean_kills_by_shell", "mean_lives"
    ]
    
    statistics = {}
    
    for cluster_id in sorted(players_by_cluster.keys()):
        players = players_by_cluster[cluster_id]
        n_players = len(players)
        
        # Extract feature values
        feature_values = {name: [] for name in feature_names}
        trajectory_lengths = []
        total_runs = []
        runner_ratios = []
        killer_ratios = []
        collector_ratios = []
        
        for player in players:
            for name in feature_names:
                feature_values[name].append(float(player[name]))
            if "mean_trajectory_length" in player:
                trajectory_lengths.append(float(player["mean_trajectory_length"]))
            if "total_runs" in player:
                total_runs.append(int(player["total_runs"]))
            # Handle both old (count) and new (ratio) column names
            if "runner_ratio" in player:
                runner_ratios.append(float(player["runner_ratio"]))
                killer_ratios.append(float(player["killer_ratio"]))
                collector_ratios.append(float(player["collector_ratio"]))
            elif "runner_count" in player:
                # Legacy support: convert counts to ratios if needed
                total_typed = int(player.get("runner_count", 0)) + int(player.get("killer_count", 0)) + int(player.get("collector_count", 0))
                if total_typed > 0:
                    runner_ratios.append(int(player["runner_count"]) / total_typed)
                    killer_ratios.append(int(player["killer_count"]) / total_typed)
                    collector_ratios.append(int(player["collector_count"]) / total_typed)
                else:
                    runner_ratios.append(0.0)
                    killer_ratios.append(0.0)
                    collector_ratios.append(0.0)
        
        # Calculate statistics
        # Use trajectory count from JSON if available, otherwise sum from players
        total_trajectories = trajectories_by_cluster.get(cluster_id, sum(total_runs) if total_runs else 0)
        
        stats = {
            "cluster_id": cluster_id,
            "n_players": n_players,
            "total_trajectories": total_trajectories,  # From trajectory-level JSON (accurate)
            "features": {},
            "player_types": {
                "avg_runner_ratio": np.mean(runner_ratios) if runner_ratios else 0.0,
                "avg_killer_ratio": np.mean(killer_ratios) if killer_ratios else 0.0,
                "avg_collector_ratio": np.mean(collector_ratios) if collector_ratios else 0.0,
            },
            "total_runs_from_players": sum(total_runs) if total_runs else 0,  # Sum from player CSV (may differ)
            "avg_runs_per_player": np.mean(total_runs) if total_runs else 0.0,
        }
        
        # Calculate feature statistics
        for name in feature_names:
            values = np.array(feature_values[name])
            stats["features"][name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
        
        # Calculate trajectory length statistics
        if trajectory_lengths:
            traj_lengths = np.array(trajectory_lengths)
            stats["trajectory_length"] = {
                "mean": float(np.mean(traj_lengths)),
                "std": float(np.std(traj_lengths)),
                "min": float(np.min(traj_lengths)),
                "max": float(np.max(traj_lengths)),
            }
        else:
            stats["trajectory_length"] = None
        
        statistics[cluster_id] = stats
    
    return statistics


def print_cluster_statistics(statistics: dict):
    """Print formatted cluster statistics."""
    print("\n" + "=" * 80)
    print("CLUSTER STATISTICS")
    print("=" * 80)
    
    for cluster_id in sorted(statistics.keys()):
        stats = statistics[cluster_id]
        
        print(f"\n{'─' * 80}")
        print(f"Cluster {cluster_id}")
        print(f"{'─' * 80}")
        print(f"Number of players:   {stats['n_players']}")
        print(f"Total trajectories:  {stats['total_trajectories']}")
        
        # Player type statistics
        print(f"\nPlayer Type Distribution (ratios):")
        pt = stats["player_types"]
        print(f"  Average runner ratio:    {pt['avg_runner_ratio']:.4f} ({pt['avg_runner_ratio']*100:.2f}%)")
        print(f"  Average killer ratio:    {pt['avg_killer_ratio']:.4f} ({pt['avg_killer_ratio']*100:.2f}%)")
        print(f"  Average collector ratio: {pt['avg_collector_ratio']:.4f} ({pt['avg_collector_ratio']*100:.2f}%)")
        
        # Average runs per player
        print(f"\nTrajectory Statistics:")
        print(f"  Average runs per player:  {stats['avg_runs_per_player']:.2f}")
        
        # Trajectory length statistics
        if stats["trajectory_length"] is not None:
            tl = stats["trajectory_length"]
            print(f"\nTrajectory Length:")
            print(f"  Mean: {tl['mean']:7.2f} ± {tl['std']:.2f} [{tl['min']:.2f}, {tl['max']:.2f}]")
        
        # Feature statistics
        print(f"\nFeature Statistics (mean ± std, [min, max]):")
        features = stats["features"]
        
        # Status features
        print(f"  Status WIN:       {features['mean_status_WIN']['mean']:7.4f} ± {features['mean_status_WIN']['std']:.4f} "
              f"[{features['mean_status_WIN']['min']:.4f}, {features['mean_status_WIN']['max']:.4f}]")
        print(f"  Status LOSE:      {features['mean_status_LOSE']['mean']:7.4f} ± {features['mean_status_LOSE']['std']:.4f} "
              f"[{features['mean_status_LOSE']['min']:.4f}, {features['mean_status_LOSE']['max']:.4f}]")
        print(f"  Status timeout:   {features['mean_status_timeout']['mean']:7.4f} ± {features['mean_status_timeout']['std']:.4f} "
              f"[{features['mean_status_timeout']['min']:.4f}, {features['mean_status_timeout']['max']:.4f}]")
        
        # Game statistics
        print(f"  Completing ratio: {features['mean_completing_ratio']['mean']:7.4f} ± {features['mean_completing_ratio']['std']:.4f} "
              f"[{features['mean_completing_ratio']['min']:.4f}, {features['mean_completing_ratio']['max']:.4f}]")
        print(f"  Kills:            {features['mean_kills']['mean']:7.4f} ± {features['mean_kills']['std']:.4f} "
              f"[{features['mean_kills']['min']:.4f}, {features['mean_kills']['max']:.4f}]")
        print(f"  Kills by fire:    {features['mean_kills_by_fire']['mean']:7.4f} ± {features['mean_kills_by_fire']['std']:.4f} "
              f"[{features['mean_kills_by_fire']['min']:.4f}, {features['mean_kills_by_fire']['max']:.4f}]")
        print(f"  Kills by stomp:   {features['mean_kills_by_stomp']['mean']:7.4f} ± {features['mean_kills_by_stomp']['std']:.4f} "
              f"[{features['mean_kills_by_stomp']['min']:.4f}, {features['mean_kills_by_stomp']['max']:.4f}]")
        print(f"  Kills by shell:   {features['mean_kills_by_shell']['mean']:7.4f} ± {features['mean_kills_by_shell']['std']:.4f} "
              f"[{features['mean_kills_by_shell']['min']:.4f}, {features['mean_kills_by_shell']['max']:.4f}]")
        print(f"  Lives:            {features['mean_lives']['mean']:7.4f} ± {features['mean_lives']['std']:.4f} "
              f"[{features['mean_lives']['min']:.4f}, {features['mean_lives']['max']:.4f}]")
    
    print(f"\n{'─' * 80}")
    print("Summary:")
    total_players = sum(s["n_players"] for s in statistics.values())
    total_trajectories = sum(s["total_trajectories"] for s in statistics.values())
    print(f"Total clusters:     {len(statistics)}")
    print(f"Total players:       {total_players}")
    print(f"Total trajectories:  {total_trajectories}")
    print(f"{'─' * 80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Print statistics for clusters created by player_clustering.py"
    )
    parser.add_argument(
        "--csv_file",
        type=str,
        required=True,
        help="Path to the player_clusters CSV file (e.g., clusters/player_clusters_k4.csv)",
    )
    parser.add_argument(
        "--json_file",
        type=str,
        default=None,
        help="Path to the trajectory_clusters JSON file (e.g., clusters/combined/trajectory_clusters_k4.json). "
             "If not provided, will try to infer from CSV path or use player-level counts.",
    )
    args = parser.parse_args()
    
    csv_path = Path(args.csv_file)
    if not csv_path.exists():
        print(f"[ERROR] CSV file not found: {csv_path}")
        return
    
    # Try to find JSON file if not provided
    json_path = None
    if args.json_file:
        json_path = Path(args.json_file)
    else:
        # Try to infer JSON path from CSV path
        # CSV: clusters/player_clusters_combined_k4.csv or clusters/player_clusters_latent_only_k4.csv
        # JSON: clusters/combined/trajectory_clusters_k4.json or clusters/latent_only/trajectory_clusters_k4.json
        csv_name = csv_path.stem  # e.g., player_clusters_combined_k4 or player_clusters_latent_only_k4
        if "player_clusters" in csv_name:
            # Extract mode and k value from filename
            # Pattern: player_clusters_{mode}_k{value}
            import re
            match = re.match(r"player_clusters_(combined|latent_only)_k(\d+)", csv_name)
            if match:
                mode = match.group(1)
                k_value = match.group(2)
                # Try mode-specific subdirectory first (from player_clustering.py and visualization)
                potential_json = csv_path.parent / mode / f"trajectory_clusters_k{k_value}.json"
                if potential_json.exists():
                    json_path = potential_json
                else:
                    # Fallback: try same directory as CSV
                    potential_json = csv_path.parent / f"trajectory_clusters_k{k_value}.json"
                    if potential_json.exists():
                        json_path = potential_json
            else:
                # Legacy format: player_clusters_k4 (no mode)
                k_value = csv_name.replace("player_clusters_k", "")
                if k_value != csv_name:  # Only if replacement happened
                    # First try same directory as CSV
                    potential_json = csv_path.parent / f"trajectory_clusters_k{k_value}.json"
                    if potential_json.exists():
                        json_path = potential_json
                    else:
                        # Then try combined/latent_only subdirectories
                        for mode in ["combined", "latent_only"]:
                            potential_json = csv_path.parent / mode / f"trajectory_clusters_k{k_value}.json"
                            if potential_json.exists():
                                json_path = potential_json
                                break
    
    print(f"[INFO] Loading cluster data from {csv_path}...")
    if json_path and json_path.exists():
        print(f"[INFO] Using trajectory-level data from {json_path} for accurate trajectory counts")
    else:
        print(f"[WARN] Trajectory-level JSON not found. Using player-level counts (may be inaccurate if players split across clusters)")
    
    players_by_cluster, trajectories_by_cluster = load_cluster_data(csv_path, json_path)
    
    # Debug: Print trajectory counts from JSON
    if json_path and json_path.exists():
        print(f"\n[DEBUG] Trajectory counts per cluster from JSON:")
        for cluster_id in sorted(trajectories_by_cluster.keys()):
            print(f"  Cluster {cluster_id}: {trajectories_by_cluster[cluster_id]} trajectories")
    
    print(f"[INFO] Calculating statistics for {len(players_by_cluster)} clusters...")
    statistics = calculate_cluster_statistics(players_by_cluster, trajectories_by_cluster)
    
    print_cluster_statistics(statistics)


if __name__ == "__main__":
    main()

