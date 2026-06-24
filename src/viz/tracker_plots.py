"""
src/viz/tracker_plots.py — Paper-ready visualizations for tracker analysis.

Every plotting function here follows the same contract:
  • Accepts a DataFrame from src/analysis/*
  • Returns the matplotlib Figure (NOT plt.show())
  • Has a `save_path` argument for one-line export
  • Applies project styling via apply_style()

Why no plt.show()?
  Returning the Figure means the function works identically in
  notebooks (Jupyter auto-displays returned figures) AND in batch
  scripts (which need to save without a display). Calling plt.show()
  inside the function would force interactive mode.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    FIGURES_DIR, PROFILES, PROFILE_LABELS, PROFILE_COLORS,
)


# ─────────────────────────────────────────────────────────────────────
# STYLING
# ─────────────────────────────────────────────────────────────────────
def apply_style() -> None:
    """Apply project-wide matplotlib style.

    Call this at the top of any notebook or script that produces
    figures. Settings here are tuned for academic publication:
      • Sans-serif font readable at small sizes
      • Larger default font sizes than matplotlib's defaults
      • White background (journal figures rarely use grids)
      • Reasonable line widths that survive PDF compression
    """
    plt.rcParams.update({
        'font.family':       'sans-serif',
        'font.sans-serif':   ['DejaVu Sans', 'Arial', 'Helvetica'],
        'font.size':         11,
        'axes.titlesize':    13,
        'axes.labelsize':    12,
        'xtick.labelsize':   10,
        'ytick.labelsize':   10,
        'legend.fontsize':   10,
        'figure.titlesize':  14,
        'axes.spines.top':   False,   # remove top spine — cleaner look
        'axes.spines.right': False,
        'axes.grid':         True,
        'grid.alpha':        0.3,
        'grid.linestyle':    '--',
        'figure.dpi':        110,     # screen render
        'savefig.dpi':       300,     # publication render
        'savefig.bbox':      'tight',
        'savefig.facecolor': 'white',
    })


def _save_if_requested(fig: Figure, save_path: Optional[Path | str]) -> None:
    """Internal helper: save figure if a path was provided."""
    if save_path is None:
        return
    save_path = Path(save_path)
    if not save_path.is_absolute():
        save_path = FIGURES_DIR / save_path
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    print(f"  ✓ Saved figure → {save_path}")


# ─────────────────────────────────────────────────────────────────────
# CORE PLOT TYPES
# ─────────────────────────────────────────────────────────────────────
def plot_prevalence_bars(
    df: pd.DataFrame,
    value_col: str = 'n_unique_etld1',
    ylabel: str = 'Unique third-party eTLD+1s',
    title: str = 'Tracking breadth by seeded history profile',
    save_path: Optional[Path | str] = None,
) -> Figure:
    """Bar chart of a per-profile metric.

    Args:
        df: DataFrame from tracker_prevalence_by_profile() or similar,
            with at least 'profile' and `value_col` columns.
        value_col: Which column to plot.
        save_path: Optional file path (relative paths go in FIGURES_DIR).
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    # Preserve PROFILES ordering, not whatever order the DataFrame has.
    df_sorted = df.set_index('profile').reindex(PROFILES).reset_index()

    colors = [PROFILE_COLORS[p] for p in df_sorted['profile']]
    bars = ax.bar(
        [PROFILE_LABELS[p] for p in df_sorted['profile']],
        df_sorted[value_col],
        color=colors,
        edgecolor='black',
        linewidth=0.5,
    )

    # Value labels above each bar — reviewers always want exact numbers.
    max_val = df_sorted[value_col].max()
    for bar, val in zip(bars, df_sorted[value_col]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max_val * 0.01,
            f'{val:.0f}' if float(val) == int(float(val)) else f'{val:.2f}',
            ha='center', va='bottom', fontsize=10,
        )

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, max_val * 1.12)   # 12% headroom for labels
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    _save_if_requested(fig, save_path)
    return fig


def plot_tracker_heatmap(
    df: pd.DataFrame,
    top_n: int = 25,
    title: str = 'Top trackers across profiles',
    save_path: Optional[Path | str] = None,
) -> Figure:
    """Heatmap of tracker presence (%) across profiles.

    Args:
        df: Long-format DataFrame from tracker_frequency_table(),
            with columns: profile, etld1, pct_of_visits.
        top_n: Number of trackers to show (by total prevalence).

    The heatmap is often the single most informative figure in a
    tracking study — at a glance, you can see which trackers
    discriminate between profiles and which are universal.
    """
    apply_style()
    # Pivot to wide format for the heatmap.
    pivot = df.pivot(index='etld1', columns='profile', values='pct_of_visits')
    pivot = pivot.reindex(columns=PROFILES).fillna(0)
    # Sort trackers by total prevalence so the most common are at top.
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index[:top_n]]

    fig, ax = plt.subplots(figsize=(8, max(6, top_n * 0.3)))
    im = ax.imshow(pivot.values, aspect='auto', cmap='YlOrRd', vmin=0, vmax=100)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([PROFILE_LABELS[p] for p in pivot.columns],
                       rotation=30, ha='right')
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)

    # Cell annotations. Skip if grid is too dense.
    if top_n <= 30:
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = float(cast(Any, pivot.iloc[i, j]))
                # White text on dark cells, black on light. Threshold at 50%.
                color = 'white' if val > 50 else 'black'
                ax.text(j, i, f'{val:.0f}', ha='center', va='center',
                        color=color, fontsize=8)

    cbar = plt.colorbar(im, ax=ax, label='% of visits with tracker')
    ax.set_title(title)
    ax.grid(False)   # heatmaps shouldn't have grid lines
    plt.tight_layout()
    _save_if_requested(fig, save_path)
    return fig


def plot_jaccard_matrix(
    matrix: pd.DataFrame,
    title: str = 'Tracker set similarity (Jaccard) between profiles',
    save_path: Optional[Path | str] = None,
) -> Figure:
    """Square heatmap of Jaccard similarities between profile pairs."""
    apply_style()
    fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(matrix.values, cmap='YlOrRd', vmin=0, vmax=1)
    labels = [PROFILE_LABELS.get(p, p) for p in matrix.columns]
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(labels)

    for i in range(len(matrix.index)):
        for j in range(len(matrix.columns)):
            val = float(cast(Any, matrix.iloc[i, j]))
            color = 'white' if val > 0.6 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color)

    plt.colorbar(im, ax=ax, label='Jaccard similarity')
    ax.set_title(title)
    ax.grid(False)
    plt.tight_layout()
    _save_if_requested(fig, save_path)
    return fig


def plot_differential_trackers(
    df: pd.DataFrame,
    profile_label: str,
    top_n: int = 20,
    save_path: Optional[Path | str] = None,
) -> Figure:
    """Horizontal bar chart of trackers with highest lift in profile_a.

    Args:
        df: DataFrame from differential_trackers(), sorted by lift desc.
        profile_label: For the title (e.g., "shopping vs control").
    """
    apply_style()
    top = df.head(top_n).iloc[::-1]   # reverse so highest is at top of plot

    fig, ax = plt.subplots(figsize=(9, max(4, top_n * 0.3)))

    ax.barh(top['etld1'], top['lift'],
            color=PROFILE_COLORS.get(profile_label.split()[0], '#999999'),
            edgecolor='black', linewidth=0.5)

    for i, (lift, va, vb) in enumerate(zip(top['lift'], top['visits_a'], top['visits_b'])):
        ax.text(lift + 0.05, i, f'  {va}/{vb} (×{lift:.1f})',
                va='center', fontsize=9)

    ax.set_xlabel('Lift (visits in seeded ÷ visits in control)')
    ax.set_title(f'Top {top_n} differential trackers: {profile_label}')
    ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5,
               label='No difference')
    ax.legend(loc='lower right')
    plt.tight_layout()
    _save_if_requested(fig, save_path)
    return fig


def plot_distribution_comparison(
    df: pd.DataFrame,
    value_col: str = 'metric_value',
    metric_name: str = 'Metric',
    log_scale: bool = True,
    save_path: Optional[Path | str] = None,
) -> Figure:
    """Overlay histograms of a per-visit metric across profiles.

    Args:
        df: DataFrame with 'profile' and `value_col` columns
            (typically the raw output of a METRIC_SQL query).
        log_scale: Use log y-axis. Most OpenWPM metrics are heavy-tailed,
            so log is almost always the right choice.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(10, 5))

    # Compute shared bins so histograms are directly comparable.
    all_vals = pd.to_numeric(df[value_col], errors='coerce').dropna().to_numpy(dtype=float)
    if len(all_vals) == 0:
        ax.set_title("No data")
        return fig
    bins = np.linspace(0, np.percentile(all_vals, 99), 50)

    for profile in PROFILES:
        data = df[df['profile'] == profile][value_col]
        if data.empty:
            continue
        ax.hist(data, bins=bins, alpha=0.5,
                label=PROFILE_LABELS[profile],
                color=PROFILE_COLORS[profile])

    ax.set_xlabel(metric_name)
    ax.set_ylabel('Number of visits')
    ax.set_title(f'Distribution of {metric_name} by profile')
    if log_scale:
        ax.set_yscale('log')
    ax.legend()
    plt.tight_layout()
    _save_if_requested(fig, save_path)
    return fig