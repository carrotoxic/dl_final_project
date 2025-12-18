import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np


def _extract_player_type(path_str: str) -> str | None:
    p = str(path_str).lower()
    if "runner" in p:
        return "runner"
    if "killer" in p:
        return "killer"
    if "collector" in p:
        return "collector"
    return None


def _assign_players_to_clusters(meta: List[Dict], labels: np.ndarray) -> List[Dict]:
    """Aggregate trajectory-level labels into a player-level label + summary features."""
    player_trajectories = defaultdict(list)
    for i, m in enumerate(meta):
        pid = m["player_id"]
        raw_features = [
            1.0 if m.get("status") == "WIN" else 0.0,
            1.0 if m.get("status") == "LOSE" else 0.0,
            1.0 if m.get("status") == "TIME_OUT" else 0.0,
            float(m.get("completing_ratio", 0.0)),
            float(m.get("kills", 0)),
            float(m.get("kills_by_fire", 0)),
            float(m.get("kills_by_stomp", 0)),
            float(m.get("kills_by_shell", 0)),
            float(m.get("lives", 0)),
            float(m.get("coins", 0)),
        ]
        player_trajectories[pid].append(
            {
                "index": i,
                "cluster": int(labels[i]),
                "player_type": _extract_player_type(m.get("path", "")),
                "raw_features": raw_features,
                "length": int(m.get("length", 0)),
            }
        )

    results = []
    for pid, trajectories in player_trajectories.items():
        cluster_counts = Counter(t["cluster"] for t in trajectories)
        majority_cluster = cluster_counts.most_common(1)[0][0]

        feat_arr = np.array([t["raw_features"] for t in trajectories], dtype=np.float64)
        mean_features = feat_arr.mean(axis=0).tolist()

        lengths = [t["length"] for t in trajectories]
        mean_length = float(np.mean(lengths)) if lengths else 0.0

        player_type_counts = Counter(t["player_type"] for t in trajectories if t["player_type"] is not None)
        total_typed = sum(player_type_counts.values())
        if total_typed > 0:
            runner_ratio = player_type_counts.get("runner", 0) / total_typed
            killer_ratio = player_type_counts.get("killer", 0) / total_typed
            collector_ratio = player_type_counts.get("collector", 0) / total_typed
        else:
            runner_ratio = killer_ratio = collector_ratio = 0.0

        results.append(
            {
                "player_id": pid,
                "cluster": majority_cluster,
                "mean_non_trajectory_features": mean_features,
                "mean_trajectory_length": mean_length,
                "total_runs": len(trajectories),
                "runner_ratio": runner_ratio,
                "killer_ratio": killer_ratio,
                "collector_ratio": collector_ratio,
            }
        )

    return sorted(results, key=lambda x: (x["cluster"], x["player_id"]))


def save_clustering_results(
    labels: np.ndarray,
    meta: List[Dict],
    output_dir: Path,
    method: str,
    model: str,
    n_clusters: int,
    evaluation: Dict[str, float] | None = None,
):
    """Save clustering results to JSON and CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / f"trajectory_clusters_k{n_clusters}.json"
    with json_path.open("w") as f:
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

    # Player-level aggregation is derived from meta + trajectory labels.
    # non_trajectory_features is already baked into meta fields above.
    results = _assign_players_to_clusters(meta, labels)

    csv_path = output_dir / f"player_clusters_k{n_clusters}.csv"
    fieldnames = [
        "player_id",
        "cluster",
        "mean_status_WIN",
        "mean_status_LOSE",
        "mean_status_timeout",
        "mean_completing_ratio",
        "mean_kills",
        "mean_kills_by_fire",
        "mean_kills_by_stomp",
        "mean_kills_by_shell",
        "mean_lives",
        "mean_coins",
        "mean_trajectory_length",
        "total_runs",
        "runner_ratio",
        "killer_ratio",
        "collector_ratio",
    ]

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            features = result["mean_non_trajectory_features"]
            writer.writerow(
                {
                    "player_id": result["player_id"],
                    "cluster": result["cluster"],
                    "mean_status_WIN": features[0],
                    "mean_status_LOSE": features[1],
                    "mean_status_timeout": features[2],
                    "mean_completing_ratio": features[3],
                    "mean_kills": features[4],
                    "mean_kills_by_fire": features[5],
                    "mean_kills_by_stomp": features[6],
                    "mean_kills_by_shell": features[7],
                    "mean_lives": features[8],
                    "mean_coins": features[9],
                    "mean_trajectory_length": result["mean_trajectory_length"],
                    "total_runs": result["total_runs"],
                    "runner_ratio": result["runner_ratio"],
                    "killer_ratio": result["killer_ratio"],
                    "collector_ratio": result["collector_ratio"],
                }
            )

    if evaluation:
        eval_path = output_dir / f"evaluation_k{n_clusters}.json"
        with eval_path.open("w") as f:
            json.dump(evaluation, f, indent=2)

    print(f"[INFO] Saved results to {output_dir}")


def save_evaluation_results(all_evaluations: Dict[str, Dict], output_path: Path):
    """Save aggregated evaluation results across all methods and cluster numbers."""
    with output_path.open("w") as f:
        json.dump(all_evaluations, f, indent=2)
    print(f"[INFO] Saved evaluation summary to {output_path}")


