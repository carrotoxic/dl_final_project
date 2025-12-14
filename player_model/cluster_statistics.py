import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


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
            for traj in trajectory_data:
                cluster = int(traj["cluster"])
                trajectories_by_cluster[cluster] += 1
    else:
        # Fallback: sum total_runs from players (less accurate if players split across clusters)
        for cluster_id, players in players_by_cluster.items():
            trajectories_by_cluster[cluster_id] = sum(int(p.get("total_runs", 0)) for p in players)
    
    return players_by_cluster, trajectories_by_cluster


def calculate_cluster_statistics(players_by_cluster: dict, trajectories_by_cluster: dict):
    """Calculate statistics for each cluster."""
    feature_names = [
        "mean_status_WIN", "mean_status_LOSE", "mean_status_timeout",
        "mean_completing_ratio", "mean_kills", "mean_kills_by_fire",
        "mean_kills_by_stomp", "mean_kills_by_shell", "mean_lives", "mean_coins"
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
                # Handle missing coins field (for backward compatibility)
                if name == "mean_coins" and name not in player:
                    feature_values[name].append(0.0)
                else:
                    feature_values[name].append(float(player.get(name, 0.0)))
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
        total_trajectories = trajectories_by_cluster.get(cluster_id, sum(total_runs) if total_runs else 0)
        
        stats = {
            "cluster_id": cluster_id,
            "n_players": n_players,
            "total_trajectories": total_trajectories,
            "features": {},
            "player_types": {
                "avg_runner_ratio": np.mean(runner_ratios) if runner_ratios else 0.0,
                "avg_killer_ratio": np.mean(killer_ratios) if killer_ratios else 0.0,
                "avg_collector_ratio": np.mean(collector_ratios) if collector_ratios else 0.0,
            },
            "total_runs_from_players": sum(total_runs) if total_runs else 0,
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


def create_comparison_plots(statistics: dict, output_path: Path):
    """Create matplotlib visualizations comparing clusters across key features."""
    n_clusters = len(statistics)
    cluster_ids = sorted(statistics.keys())
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Color palette for clusters
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    
    # 1. Win Rate (mean_status_WIN)
    ax1 = fig.add_subplot(gs[0, 0])
    win_means = [statistics[c]["features"]["mean_status_WIN"]["mean"] for c in cluster_ids]
    win_stds = [statistics[c]["features"]["mean_status_WIN"]["std"] for c in cluster_ids]
    bars1 = ax1.bar(range(n_clusters), win_means, yerr=win_stds, color=colors, alpha=0.7, capsize=5)
    ax1.set_xlabel("Cluster", fontsize=10)
    ax1.set_ylabel("Win Rate", fontsize=10)
    ax1.set_title("Win Rate by Cluster", fontsize=12, fontweight="bold")
    ax1.set_xticks(range(n_clusters))
    ax1.set_xticklabels([f"C{c}" for c in cluster_ids])
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim([0, max(win_means) * 1.2 if max(win_means) > 0 else 1.0])
    
    # 2. Completion Rate
    ax2 = fig.add_subplot(gs[0, 1])
    comp_means = [statistics[c]["features"]["mean_completing_ratio"]["mean"] for c in cluster_ids]
    comp_stds = [statistics[c]["features"]["mean_completing_ratio"]["std"] for c in cluster_ids]
    bars2 = ax2.bar(range(n_clusters), comp_means, yerr=comp_stds, color=colors, alpha=0.7, capsize=5)
    ax2.set_xlabel("Cluster", fontsize=10)
    ax2.set_ylabel("Completion Rate", fontsize=10)
    ax2.set_title("Completion Rate by Cluster", fontsize=12, fontweight="bold")
    ax2.set_xticks(range(n_clusters))
    ax2.set_xticklabels([f"C{c}" for c in cluster_ids])
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim([0, 1.1])
    
    # 3. Coin Collection
    ax3 = fig.add_subplot(gs[0, 2])
    coin_means = [statistics[c]["features"].get("mean_coins", {}).get("mean", 0.0) for c in cluster_ids]
    coin_stds = [statistics[c]["features"].get("mean_coins", {}).get("std", 0.0) for c in cluster_ids]
    bars3 = ax3.bar(range(n_clusters), coin_means, yerr=coin_stds, color=colors, alpha=0.7, capsize=5)
    ax3.set_xlabel("Cluster", fontsize=10)
    ax3.set_ylabel("Mean Coins", fontsize=10)
    ax3.set_title("Coin Collection by Cluster", fontsize=12, fontweight="bold")
    ax3.set_xticks(range(n_clusters))
    ax3.set_xticklabels([f"C{c}" for c in cluster_ids])
    ax3.grid(axis='y', alpha=0.3)
    if max(coin_means) > 0:
        ax3.set_ylim([0, max(coin_means) * 1.2])
    
    # 4. Total Kills
    ax4 = fig.add_subplot(gs[1, 0])
    kill_means = [statistics[c]["features"]["mean_kills"]["mean"] for c in cluster_ids]
    kill_stds = [statistics[c]["features"]["mean_kills"]["std"] for c in cluster_ids]
    bars4 = ax4.bar(range(n_clusters), kill_means, yerr=kill_stds, color=colors, alpha=0.7, capsize=5)
    ax4.set_xlabel("Cluster", fontsize=10)
    ax4.set_ylabel("Mean Kills", fontsize=10)
    ax4.set_title("Kills by Cluster", fontsize=12, fontweight="bold")
    ax4.set_xticks(range(n_clusters))
    ax4.set_xticklabels([f"C{c}" for c in cluster_ids])
    ax4.grid(axis='y', alpha=0.3)
    if max(kill_means) > 0:
        ax4.set_ylim([0, max(kill_means) * 1.2])
    
    # 5. Trajectory Length
    ax5 = fig.add_subplot(gs[1, 1])
    traj_means = []
    traj_stds = []
    for c in cluster_ids:
        if statistics[c]["trajectory_length"] is not None:
            traj_means.append(statistics[c]["trajectory_length"]["mean"])
            traj_stds.append(statistics[c]["trajectory_length"]["std"])
        else:
            traj_means.append(0.0)
            traj_stds.append(0.0)
    bars5 = ax5.bar(range(n_clusters), traj_means, yerr=traj_stds, color=colors, alpha=0.7, capsize=5)
    ax5.set_xlabel("Cluster", fontsize=10)
    ax5.set_ylabel("Mean Trajectory Length", fontsize=10)
    ax5.set_title("Trajectory Length by Cluster", fontsize=12, fontweight="bold")
    ax5.set_xticks(range(n_clusters))
    ax5.set_xticklabels([f"C{c}" for c in cluster_ids])
    ax5.grid(axis='y', alpha=0.3)
    if max(traj_means) > 0:
        ax5.set_ylim([0, max(traj_means) * 1.2])
    
    # 6. Player Type Distribution (Stacked Bar)
    ax6 = fig.add_subplot(gs[1, 2])
    runner_ratios = [statistics[c]["player_types"]["avg_runner_ratio"] for c in cluster_ids]
    killer_ratios = [statistics[c]["player_types"]["avg_killer_ratio"] for c in cluster_ids]
    collector_ratios = [statistics[c]["player_types"]["avg_collector_ratio"] for c in cluster_ids]
    x_pos = np.arange(n_clusters)
    width = 0.6
    ax6.bar(x_pos, runner_ratios, width, label='Runner', color='#2ecc71', alpha=0.7)
    ax6.bar(x_pos, killer_ratios, width, bottom=runner_ratios, label='Killer', color='#e74c3c', alpha=0.7)
    ax6.bar(x_pos, collector_ratios, width, bottom=np.array(runner_ratios) + np.array(killer_ratios), 
            label='Collector', color='#f39c12', alpha=0.7)
    ax6.set_xlabel("Cluster", fontsize=10)
    ax6.set_ylabel("Player Type Ratio", fontsize=10)
    ax6.set_title("Player Type Distribution by Cluster", fontsize=12, fontweight="bold")
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels([f"C{c}" for c in cluster_ids])
    ax6.legend(loc='upper right', fontsize=9)
    ax6.set_ylim([0, 1.1])
    ax6.grid(axis='y', alpha=0.3)
    
    # 7. Number of Players
    ax7 = fig.add_subplot(gs[2, 0])
    n_players = [statistics[c]["n_players"] for c in cluster_ids]
    bars7 = ax7.bar(range(n_clusters), n_players, color=colors, alpha=0.7)
    ax7.set_xlabel("Cluster", fontsize=10)
    ax7.set_ylabel("Number of Players", fontsize=10)
    ax7.set_title("Players per Cluster", fontsize=12, fontweight="bold")
    ax7.set_xticks(range(n_clusters))
    ax7.set_xticklabels([f"C{c}" for c in cluster_ids])
    ax7.grid(axis='y', alpha=0.3)
    # Add value labels on bars
    for i, v in enumerate(n_players):
        ax7.text(i, v, str(v), ha='center', va='bottom', fontsize=9)
    
    # 8. Total Trajectories
    ax8 = fig.add_subplot(gs[2, 1])
    n_trajs = [statistics[c]["total_trajectories"] for c in cluster_ids]
    bars8 = ax8.bar(range(n_clusters), n_trajs, color=colors, alpha=0.7)
    ax8.set_xlabel("Cluster", fontsize=10)
    ax8.set_ylabel("Total Trajectories", fontsize=10)
    ax8.set_title("Trajectories per Cluster", fontsize=12, fontweight="bold")
    ax8.set_xticks(range(n_clusters))
    ax8.set_xticklabels([f"C{c}" for c in cluster_ids])
    ax8.grid(axis='y', alpha=0.3)
    # Add value labels on bars
    for i, v in enumerate(n_trajs):
        ax8.text(i, v, str(v), ha='center', va='bottom', fontsize=9)
    
    # 9. Kill Methods Comparison
    ax9 = fig.add_subplot(gs[2, 2])
    fire_means = [statistics[c]["features"]["mean_kills_by_fire"]["mean"] for c in cluster_ids]
    stomp_means = [statistics[c]["features"]["mean_kills_by_stomp"]["mean"] for c in cluster_ids]
    shell_means = [statistics[c]["features"]["mean_kills_by_shell"]["mean"] for c in cluster_ids]
    x_pos = np.arange(n_clusters)
    width = 0.25
    ax9.bar(x_pos - width, fire_means, width, label='Fire', color='#e74c3c', alpha=0.7)
    ax9.bar(x_pos, stomp_means, width, label='Stomp', color='#3498db', alpha=0.7)
    ax9.bar(x_pos + width, shell_means, width, label='Shell', color='#9b59b6', alpha=0.7)
    ax9.set_xlabel("Cluster", fontsize=10)
    ax9.set_ylabel("Mean Kills", fontsize=10)
    ax9.set_title("Kill Methods by Cluster", fontsize=12, fontweight="bold")
    ax9.set_xticks(x_pos)
    ax9.set_xticklabels([f"C{c}" for c in cluster_ids])
    ax9.legend(loc='upper right', fontsize=9)
    ax9.grid(axis='y', alpha=0.3)
    
    plt.suptitle("Cluster Comparison Statistics", fontsize=16, fontweight="bold", y=0.995)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved comparison plot to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate matplotlib visualizations for cluster statistics"
    )
    parser.add_argument(
        "--csv_file",
        type=str,
        required=True,
        help="Path to the player_clusters CSV file",
    )
    parser.add_argument(
        "--json_file",
        type=str,
        default=None,
        help="Path to the trajectory_clusters JSON file. If not provided, will try to infer from CSV path.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for the visualization. If not provided, will save next to CSV file.",
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
        # New path structure: clusters/{method}/{mode}/k{n}/player_clusters_{mode}_k{n}.csv
        # JSON: clusters/{method}/{mode}/k{n}/trajectory_clusters_k{n}.json
        csv_name = csv_path.stem  # e.g., player_clusters_combined_k4
        import re
        match = re.match(r"player_clusters_(combined|latent_only)_k(\d+)", csv_name)
        if match:
            mode = match.group(1)
            k_value = match.group(2)
            # JSON should be in the same directory as CSV
            potential_json = csv_path.parent / f"trajectory_clusters_k{k_value}.json"
            if potential_json.exists():
                json_path = potential_json
    
    print(f"[INFO] Loading cluster data from {csv_path}...")
    if json_path and json_path.exists():
        print(f"[INFO] Using trajectory-level data from {json_path}")
    else:
        print(f"[WARN] Trajectory-level JSON not found. Using player-level counts.")
    
    players_by_cluster, trajectories_by_cluster = load_cluster_data(csv_path, json_path)
    
    print(f"[INFO] Calculating statistics for {len(players_by_cluster)} clusters...")
    statistics = calculate_cluster_statistics(players_by_cluster, trajectories_by_cluster)
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = csv_path.parent / f"cluster_statistics_{csv_path.stem}.png"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create visualizations
    create_comparison_plots(statistics, output_path)
    
    print(f"[INFO] Done. Visualization saved to: {output_path}")


if __name__ == "__main__":
    main()
