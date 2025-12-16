import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .core import create_clustering_method, fit_predict, evaluate_clustering, CLUSTERING_METHODS
from .data import extract_features
from .visualization import (
    plot_clusters_2d, plot_clusters_3d,
    plot_players_2d, plot_players_3d,
    plot_player_types_2d, plot_player_types_3d,
    plot_dendrogram,
    plot_cluster_statistics,
)
from .utils.save_results import save_clustering_results, save_evaluation_results


def run_clustering(
    features: np.ndarray,
    meta: list,
    non_trajectory_features: np.ndarray,
    method: str,
    n_clusters: int,
    output_dir: Path,
    model: str,
    mode: str,
):
    """Run clustering for a single method and cluster number"""
    print(f"\n{'='*80}")
    print(f"Method: {method.upper()}, K={n_clusters}")
    print(f"{'='*80}")
    
    result_dir = output_dir / method / model / mode / f"k{n_clusters}"
    result_dir.mkdir(parents=True, exist_ok=True)
    
    clustering_method = create_clustering_method(method, n_clusters)
    labels = fit_predict(clustering_method, features)
    
    unique_labels = np.unique(labels[labels >= 0])
    n_clusters_found = len(unique_labels)
    
    print(f"[INFO] Found {n_clusters_found} clusters")
    for k in unique_labels:
        n_points = int(np.sum(labels == k))
        print(f"  - Cluster {k}: {n_points} trajectories")
    
    evaluation = evaluate_clustering(features, labels, method, n_clusters)
    silhouette = evaluation.get("silhouette_score", -1.0)
    print(f"[INFO] Silhouette Score: {silhouette:.4f}")
    
    save_clustering_results(
        labels, meta, non_trajectory_features,
        result_dir, method, model, mode, n_clusters, evaluation
    )
    
    player_ids = [m["player_id"] for m in meta]
    
    plot_clusters_2d(
        features, labels,
        result_dir / f"clusters_2d_k{n_clusters}.png",
        title=f"{method.upper()} Clustering (K={n_clusters}, Silhouette={silhouette:.3f})"
    )
    plot_clusters_3d(
        features, labels,
        result_dir / f"clusters_3d_k{n_clusters}.png",
        title=f"{method.upper()} Clustering (K={n_clusters})"
    )
    plot_players_2d(features, player_ids, result_dir / f"players_2d_k{n_clusters}.png")
    plot_players_3d(features, player_ids, result_dir / f"players_3d_k{n_clusters}.png")
    plot_player_types_2d(features, meta, result_dir / f"player_types_2d_k{n_clusters}.png")
    plot_player_types_3d(features, meta, result_dir / f"player_types_3d_k{n_clusters}.png")

    plot_cluster_statistics(
        meta=meta,
        labels=labels,
        save_path=result_dir / f"cluster_statistics_k{n_clusters}.png",
        title=f"{method.upper()} Cluster Statistics (K={n_clusters}, Silhouette={silhouette:.3f})",
    )
    
    if method == "agglomerative":
        plot_dendrogram(
            features, clustering_method,
            result_dir / f"dendrogram_k{n_clusters}.png"
        )
    
    np.save(result_dir / "labels.npy", labels)
    np.save(result_dir / "features.npy", features)
    
    return evaluation


def main():
    parser = argparse.ArgumentParser(description="Cluster players with multiple methods")
    parser.add_argument("--methods", nargs="+", choices=CLUSTERING_METHODS, default=["kmeans"],
                        help="Clustering methods to use")
    parser.add_argument("--n_clusters", nargs="+", type=int, default=None,
                        help="Cluster numbers (default: 2-10)")
    parser.add_argument("--output_dir", type=str, default="clustering_results",
                        help="Output directory")
    parser.add_argument("--mode", type=str, choices=["latent_only", "combined"], default="combined")
    # no_timeout filtering removed
    parser.add_argument("--model", type=str, choices=["lstm", "fusion", "transformer"], default="lstm")
    parser.add_argument("--fusion_model_path", type=str, default=None)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.n_clusters is None:
        cluster_numbers = list(range(2, 11))
    else:
        cluster_numbers = args.n_clusters
    
    print(f"[INFO] Extracting features (model={args.model}, mode={args.mode})...")
    features, latents, meta, non_trajectory_features = extract_features(
        device, args.model, args.mode,
        Path(args.fusion_model_path) if args.fusion_model_path else None
    )
    
    mode_name = "fusion" if args.model == "fusion" else args.mode
    
    all_evaluations = {}
    
    for method in args.methods:
        method_evaluations = {}
        
        for n_clusters in cluster_numbers:
            if method == "dbscan":
                print(f"[INFO] DBSCAN doesn't use n_clusters, using default parameters")
                n_clusters = None
            
            try:
                evaluation = run_clustering(
                    features, meta, non_trajectory_features,
                    method, n_clusters, output_dir, args.model, mode_name
                )
                method_evaluations[str(n_clusters) if n_clusters else "auto"] = evaluation
            except Exception as e:
                print(f"[ERROR] Failed for {method} k={n_clusters}: {e}")
                continue
        
        all_evaluations[method] = method_evaluations
    
    eval_summary_path = output_dir / "evaluation_summary.json"
    save_evaluation_results(all_evaluations, eval_summary_path)
    
    print(f"\n{'='*80}")
    print(f"[INFO] All clustering complete!")
    print(f"[INFO] Results saved to: {output_dir}")
    print(f"[INFO] Evaluation summary: {eval_summary_path}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

