from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt

from ..utils.color_utils import get_cluster_colors


def _mean_std(vals: List[float]) -> tuple[float, float]:
    if not vals:
        return 0.0, 0.0
    arr = np.asarray(vals, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=0))


def compute_cluster_statistics(meta: List[Dict], labels: np.ndarray) -> Dict[int, Dict[str, float]]:
    """Compute simple per-cluster mean statistics from trajectory-level meta."""
    labels = np.asarray(labels)
    cluster_ids = sorted(np.unique(labels).tolist())

    stats: Dict[int, Dict[str, float]] = {}
    for cid in cluster_ids:
        idx = np.where(labels == cid)[0].tolist()
        rows = [meta[i] for i in idx]

        n = len(rows)
        win_rate = sum(1 for r in rows if r.get("status") == "WIN") / n if n else 0.0
        lose_rate = sum(1 for r in rows if r.get("status") == "LOSE") / n if n else 0.0

        comp_vals = [float(r.get("completing_ratio", 0.0)) for r in rows]
        kills_vals = [float(r.get("kills", 0.0)) for r in rows]
        coins_vals = [float(r.get("coins", 0.0)) for r in rows]
        len_vals = [float(r.get("length", 0.0)) for r in rows]

        comp_mean, comp_std = _mean_std(comp_vals)
        kills_mean, kills_std = _mean_std(kills_vals)
        coins_mean, coins_std = _mean_std(coins_vals)
        len_mean, len_std = _mean_std(len_vals)

        stats[int(cid)] = {
            "n_trajectories": float(n),
            "win_rate": float(win_rate),
            "lose_rate": float(lose_rate),
            "completion_mean": comp_mean,
            "completion_std": comp_std,
            "kills_mean": kills_mean,
            "kills_std": kills_std,
            "coins_mean": coins_mean,
            "coins_std": coins_std,
            "length_mean": len_mean,
            "length_std": len_std,
        }
    return stats


def plot_cluster_statistics(
    meta: List[Dict],
    labels: np.ndarray,
    save_path: Path,
    title: str | None = None,
) -> None:
    """Plot per-cluster summary statistics using the same cluster colors as scatter plots."""
    labels = np.asarray(labels)
    stats = compute_cluster_statistics(meta, labels)

    cluster_ids = sorted(stats.keys())
    if not cluster_ids:
        # Nothing to plot
        return

    # Positive cluster labels (ignore noise=-1 for palette sizing)
    pos_labels = sorted(np.unique(labels[labels >= 0]).tolist())
    color_map = get_cluster_colors(len(pos_labels), labels)
    colors = [color_map.get(cid, "#CCCCCC") for cid in cluster_ids]

    x = np.arange(len(cluster_ids))
    xticklabels = [f"C{cid}" if cid >= 0 else "noise" for cid in cluster_ids]

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    axes = axes.flatten()
    for ax in axes:
        ax.set_axisbelow(True)  # grid behind bars

    def bar(
        ax,
        values,
        yerr=None,
        ylabel: str = "",
        title_txt: str = "",
        ylim: tuple[float, float] | None = None,
        fmt: str = "{:.2f}",
    ):
        bars = ax.bar(
            x,
            values,
            yerr=yerr,
            color=colors,
            alpha=0.85,
            capsize=3 if yerr is not None else 0,
            edgecolor="white",
            linewidth=0.6,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(xticklabels, rotation=0)
        ax.set_ylabel(ylabel)
        ax.set_title(title_txt, fontweight="bold")
        ax.grid(axis="y", alpha=0.25)

        if ylim is not None:
            ax.set_ylim(*ylim)

        # Annotate bar tops with values (small Font)
        for bar_obj, v in zip(bars, values):
            height = bar_obj.get_height()
            ax.text(
                bar_obj.get_x() + bar_obj.get_width() / 2.0,
                height,
                fmt.format(v),
                ha="center",
                va="bottom",
                fontsize=8,
            )

    # Win rate (0–1)
    bar(
        axes[0],
        [stats[c]["win_rate"] for c in cluster_ids],
        ylabel="rate",
        title_txt="Win rate",
        ylim=(0.0, 1.0),
        fmt="{:.2f}",
    )

    # Completion ratio
    bar(
        axes[1],
        [stats[c]["completion_mean"] for c in cluster_ids],
        yerr=[stats[c]["completion_std"] for c in cluster_ids],
        ylabel="mean ± std",
        title_txt="Completion ratio",
        fmt="{:.2f}",
    )

    # Coins
    bar(
        axes[2],
        [stats[c]["coins_mean"] for c in cluster_ids],
        yerr=[stats[c]["coins_std"] for c in cluster_ids],
        ylabel="mean ± std",
        title_txt="Coins",
        fmt="{:.1f}",
    )

    # Kills
    bar(
        axes[3],
        [stats[c]["kills_mean"] for c in cluster_ids],
        yerr=[stats[c]["kills_std"] for c in cluster_ids],
        ylabel="mean ± std",
        title_txt="Kills",
        fmt="{:.1f}",
    )

    # Trajectory length
    bar(
        axes[4],
        [stats[c]["length_mean"] for c in cluster_ids],
        yerr=[stats[c]["length_std"] for c in cluster_ids],
        ylabel="mean ± std",
        title_txt="Trajectory length",
        fmt="{:.0f}",
    )

    # Number of trajectories
    bar(
        axes[5],
        [stats[c]["n_trajectories"] for c in cluster_ids],
        ylabel="count",
        title_txt="Trajectories per cluster",
        fmt="{:.0f}",
    )

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")
        fig.subplots_adjust(top=0.88)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)



