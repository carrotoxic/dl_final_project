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
        coins_ratio_vals = [float(r.get("coins_ratio", 0.0)) for r in rows]
        kills_ratio_vals = [float(r.get("kills_ratio", 0.0)) for r in rows]
        jumps_vals = [float(r.get("jumps", 0.0)) for r in rows]
        bumps_vals = [float(r.get("bumps", 0.0)) for r in rows]
        kicks_vals = [float(r.get("kicks", 0.0)) for r in rows]
        hurts_vals = [float(r.get("hurts", 0.0)) for r in rows]
        lives_vals = [float(r.get("lives", 0.0)) for r in rows]
        len_vals = [float(r.get("length", 0.0)) for r in rows]

        comp_mean, comp_std = _mean_std(comp_vals)
        coins_ratio_mean, coins_ratio_std = _mean_std(coins_ratio_vals)
        kills_ratio_mean, kills_ratio_std = _mean_std(kills_ratio_vals)
        jumps_mean, jumps_std = _mean_std(jumps_vals)
        bumps_mean, bumps_std = _mean_std(bumps_vals)
        kicks_mean, kicks_std = _mean_std(kicks_vals)
        hurts_mean, hurts_std = _mean_std(hurts_vals)
        lives_mean, lives_std = _mean_std(lives_vals)
        len_mean, len_std = _mean_std(len_vals)

        stats[int(cid)] = {
            "n_trajectories": float(n),
            "win_rate": float(win_rate),
            "lose_rate": float(lose_rate),
            "completion_mean": comp_mean,
            "completion_std": comp_std,
            "coins_ratio_mean": coins_ratio_mean,
            "coins_ratio_std": coins_ratio_std,
            "kills_ratio_mean": kills_ratio_mean,
            "kills_ratio_std": kills_ratio_std,
            "jumps_mean": jumps_mean,
            "jumps_std": jumps_std,
            "bumps_mean": bumps_mean,
            "bumps_std": bumps_std,
            "kicks_mean": kicks_mean,
            "kicks_std": kicks_std,
            "hurts_mean": hurts_mean,
            "hurts_std": hurts_std,
            "lives_mean": lives_mean,
            "lives_std": lives_std,
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

    fig, axes = plt.subplots(2, 5, figsize=(20, 7))
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

    # Coins collecting ratio
    bar(
        axes[2],
        [stats[c]["coins_ratio_mean"] for c in cluster_ids],
        yerr=[stats[c]["coins_ratio_std"] for c in cluster_ids],
        ylabel="mean ± std",
        title_txt="Coins collecting ratio",
        fmt="{:.2f}",
    )

    # Kills ratio
    bar(
        axes[3],
        [stats[c]["kills_ratio_mean"] for c in cluster_ids],
        yerr=[stats[c]["kills_ratio_std"] for c in cluster_ids],
        ylabel="mean ± std",
        title_txt="Kills ratio",
        fmt="{:.2f}",
    )

    # Jumps
    bar(
        axes[4],
        [stats[c]["jumps_mean"] for c in cluster_ids],
        yerr=[stats[c]["jumps_std"] for c in cluster_ids],
        ylabel="mean ± std",
        title_txt="Jumps",
        fmt="{:.0f}",
    )

    # Bumps
    bar(
        axes[5],
        [stats[c]["bumps_mean"] for c in cluster_ids],
        yerr=[stats[c]["bumps_std"] for c in cluster_ids],
        ylabel="mean ± std",
        title_txt="Bumps",
        fmt="{:.0f}",
    )

    # Kicks
    bar(
        axes[6],
        [stats[c]["kicks_mean"] for c in cluster_ids],
        yerr=[stats[c]["kicks_std"] for c in cluster_ids],
        ylabel="mean ± std",
        title_txt="Kicks",
        fmt="{:.0f}",
    )

    # Hurts
    bar(
        axes[7],
        [stats[c]["hurts_mean"] for c in cluster_ids],
        yerr=[stats[c]["hurts_std"] for c in cluster_ids],
        ylabel="mean ± std",
        title_txt="Hurts",
        fmt="{:.0f}",
    )

    # Lives
    bar(
        axes[8],
        [stats[c]["lives_mean"] for c in cluster_ids],
        yerr=[stats[c]["lives_std"] for c in cluster_ids],
        ylabel="mean ± std",
        title_txt="Lives",
        fmt="{:.1f}",
    )

    # Trajectory length
    bar(
        axes[9],
        [stats[c]["length_mean"] for c in cluster_ids],
        yerr=[stats[c]["length_std"] for c in cluster_ids],
        ylabel="mean ± std",
        title_txt="Trajectory length",
        fmt="{:.0f}",
    )

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")
        fig.subplots_adjust(top=0.88)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)



