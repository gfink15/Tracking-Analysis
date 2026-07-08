"""
src/viz/cookie_plots.py — Visualizations for cookie behavior analysis.

All functions return matplotlib Figures and accept an optional
save_path. Styling delegated to apply_style() from tracker_plots.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.figure import Figure

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    FIGURES_DIR, PROFILES, PROFILE_LABELS, PROFILE_COLORS,
)
# Reuse the styling and save helper from tracker_plots — single source.
from src.viz.tracker_plots import apply_style, _save_if_requested


# Lifespan bucket order matters for stacked bars: shortest at bottom,
# longest at top so the "more persistent = more concerning" reads
# visually upward.
LIFESPAN_ORDER = ['session', '<1d', '1-7d', '7-30d', '30-365d', '1y+']

# Distinct colors for lifespan buckets. Going from cool (short-lived,
# benign) to warm (long-lived, retargeting-relevant) maps lifespan
# concern level to color temperature.
LIFESPAN_COLORS = {
    'session':   '#3498DB',   # cool blue — ephemeral
    '<1d':       '#2ECC71',   # green — short-lived
    '1-7d':      '#F1C40F',   # yellow — weekly
    '7-30d':     '#E67E22',   # orange — monthly
    '30-365d':   '#E74C3C',   # red — long-lived
    '1y+':       '#8E44AD',   # purple — very long-lived
}


# ─────────────────────────────────────────────────────────────────────
# COOKIE COUNTS — first vs third party
# ─────────────────────────────────────────────────────────────────────
def plot_first_vs_third_party(
    df: pd.DataFrame,
    title: str = 'Cookies set by profile (first vs third party)',
    save_path: Optional[Path | str] = None,
) -> Figure:
    """Stacked bar of first-party vs third-party cookie counts per profile.

    Args:
        df: DataFrame from cookie_counts_by_profile() with columns
            'profile', 'n_first_party', 'n_third_party'.

    The stacked-bar form lets readers see BOTH the absolute volume
    (total bar height) AND the composition (first vs third split).
    A profile with 200 cookies, 90% third-party, is qualitatively
    different from one with 200 cookies, 30% third-party.
    """
    apply_style()
    # Reindex to canonical profile order
    df_sorted = df.set_index('profile').reindex(PROFILES).reset_index()

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(df_sorted))
    width = 0.6

    # First-party at the bottom
    bars_fp = ax.bar(
        x, df_sorted['n_first_party'],
        width, label='First-party',
        color='#7F8C8D', edgecolor='black', linewidth=0.5,
    )
    # Third-party stacked on top
    bars_tp = ax.bar(
        x, df_sorted['n_third_party'],
        width, bottom=df_sorted['n_first_party'],
        label='Third-party',
        color='#E74C3C', edgecolor='black', linewidth=0.5,
    )

    # Annotate each segment with its count
    for i, (fp, tp) in enumerate(zip(
        df_sorted['n_first_party'], df_sorted['n_third_party']
    )):
        if fp > 0:
            ax.text(i, fp / 2, f'{fp:,}',
                    ha='center', va='center',
                    color='white', fontsize=10, fontweight='bold')
        if tp > 0:
            ax.text(i, fp + tp / 2, f'{tp:,}',
                    ha='center', va='center',
                    color='white', fontsize=10, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([PROFILE_LABELS[p] for p in df_sorted['profile']],
                       rotation=15, ha='right')
    ax.set_ylabel('Number of cookies set')
    ax.set_title(title)
    ax.legend(loc='upper left')
    plt.tight_layout()
    _save_if_requested(fig, save_path)
    return fig


# ─────────────────────────────────────────────────────────────────────
# LIFESPAN DISTRIBUTION
# ─────────────────────────────────────────────────────────────────────
def plot_lifespan_distribution(
    df: pd.DataFrame,
    title: str = 'Third-party cookie lifespan distribution by profile',
    normalize: bool = True,
    save_path: Optional[Path | str] = None,
) -> Figure:
    """Stacked bar showing lifespan-bucket composition per profile.

    Args:
        df: Long-format DataFrame from cookie_lifespan_distribution(),
            with columns 'profile', 'lifespan_bucket', 'n_cookies'.
        normalize: If True, plot percentages (each bar sums to 100%);
            if False, plot raw counts. Percentages let you compare
            COMPOSITION even when profiles have very different totals.
    """
    apply_style()
    # Pivot to wide: rows = profiles, columns = lifespan buckets
    pivot = (df.pivot(index='profile', columns='lifespan_bucket',
                      values='n_cookies')
               .reindex(index=PROFILES)
               .reindex(columns=LIFESPAN_ORDER)
               .fillna(0))

    if normalize:
        pivot = pivot.div(pivot.sum(axis=1), axis=0) * 100
        ylabel = '% of third-party cookies'
    else:
        ylabel = 'Number of third-party cookies'

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(pivot.index))
    width = 0.6
    bottom = np.zeros(len(pivot.index))

    for bucket in LIFESPAN_ORDER:
        values = pivot[bucket].to_numpy(dtype=float)
        ax.bar(
            x, values, width,
            bottom=bottom, label=bucket,
            color=LIFESPAN_COLORS[bucket],
            edgecolor='black', linewidth=0.3,
        )
        # Annotate segments large enough to fit a label
        for i, (v, b) in enumerate(zip(values, bottom)):
            if v > (5 if normalize else pivot.to_numpy().max() * 0.03):
                ax.text(i, b + v / 2,
                        f'{v:.0f}%' if normalize else f'{int(v)}',
                        ha='center', va='center',
                        color='white', fontsize=9, fontweight='bold')
        bottom += values

    ax.set_xticks(x)
    ax.set_xticklabels([PROFILE_LABELS[p] for p in pivot.index],
                       rotation=15, ha='right')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    # Reverse legend so it matches stack order (longest on top)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1],
              title='Lifespan', loc='center left',
              bbox_to_anchor=(1.0, 0.5))
    plt.tight_layout()
    _save_if_requested(fig, save_path)
    return fig


# ─────────────────────────────────────────────────────────────────────
# RETARGETING NETWORK PRESENCE
# ─────────────────────────────────────────────────────────────────────
def plot_retargeting_presence(
    df: pd.DataFrame,
    metric: str = 'n_cookies',
    title: Optional[str] = None,
    save_path: Optional[Path | str] = None,
) -> Figure:
    """Grouped bar chart: retargeting networks × profiles.

    Args:
        df: DataFrame from retargeting_cookie_presence() with columns
            'profile', 'retargeter', 'n_cookies', 'n_visits_affected'.
        metric: Which column to plot — 'n_cookies' (volume) or
            'n_visits_affected' (reach).

    This figure is often the most striking in a tracking study: if
    one profile shows 5-10× more Criteo cookies than another, the
    figure makes that obvious at a glance.
    """
    apply_style()
    pivot = (df.pivot(index='retargeter', columns='profile', values=metric)
               .reindex(columns=PROFILES)
               .fillna(0))
    # Sort retargeters by total volume so the most prevalent are on top
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=True).index]

    fig, ax = plt.subplots(figsize=(10, max(4, len(pivot) * 0.4)))
    y = np.arange(len(pivot.index))
    height = 0.8 / len(PROFILES)

    for i, profile in enumerate(PROFILES):
        offset = (i - len(PROFILES) / 2 + 0.5) * height
        if profile not in pivot.columns:
            continue
        ax.barh(
            y + offset, pivot[profile].to_numpy(dtype=float), height,
            label=PROFILE_LABELS[profile],
            color=PROFILE_COLORS[profile],
            edgecolor='black', linewidth=0.3,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel('Number of cookies' if metric == 'n_cookies'
                  else 'Visits affected')
    ax.set_title(title or f'Retargeting network presence ({metric})')
    ax.legend(loc='lower right')
    plt.tight_layout()
    _save_if_requested(fig, save_path)
    return fig


# ─────────────────────────────────────────────────────────────────────
# COOKIE-SYNC HEATMAP
# ─────────────────────────────────────────────────────────────────────
def plot_sync_summary(
    df: pd.DataFrame,
    title: str = 'Cookie sync activity by profile',
    save_path: Optional[Path | str] = None,
) -> Figure:
    """Simple bar chart of cookie-sync event counts per profile.

    Args:
        df: DataFrame from cookie_sync_summary() with columns
            'profile', 'n_sync_events', 'n_visits_with_syncs',
            'avg_hosts_per_sync'.

    Two side-by-side bars per profile: total events and visits
    affected. The ratio between them is also informative
    (high events / low visits = concentrated syncing on a few sites;
    similar values = syncing spread evenly).
    """
    apply_style()
    df_sorted = df.set_index('profile').reindex(PROFILES).reset_index()

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(df_sorted))
    width = 0.38

    bars_events = ax.bar(
        x - width / 2, df_sorted['n_sync_events'], width,
        label='Total sync events',
        color='#E74C3C', edgecolor='black', linewidth=0.5,
    )
    bars_visits = ax.bar(
        x + width / 2, df_sorted['n_visits_with_syncs'], width,
        label='Visits with ≥1 sync',
        color='#3498DB', edgecolor='black', linewidth=0.5,
    )

    for bars in (bars_events, bars_visits):
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h * 1.01,
                        f'{int(h)}', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([PROFILE_LABELS[p] for p in df_sorted['profile']],
                       rotation=15, ha='right')
    ax.set_ylabel('Count')
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    _save_if_requested(fig, save_path)
    return fig


# ─────────────────────────────────────────────────────────────────────
# FINGERPRINTING SUMMARY
# ─────────────────────────────────────────────────────────────────────
def plot_fingerprinter_summary(
    df: pd.DataFrame,
    title: str = 'Fingerprinting scripts encountered by profile',
    save_path: Optional[Path | str] = None,
) -> Figure:
    """Grouped bars: fingerprinting technique counts per profile.

    Args:
        df: DataFrame from fingerprinter_summary() with columns
            'profile', 'n_canvas_fp_scripts', 'n_audio_fp_scripts',
            'n_navigator_fp_scripts'.

    Even though this technically belongs in a fingerprinting_plots
    module, it's small enough to live here and reuses the same
    grouped-bar pattern. Split into its own file later if the
    fingerprinting analysis grows.
    """
    apply_style()
    df_sorted = df.set_index('profile').reindex(PROFILES).reset_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(df_sorted))
    width = 0.27

    techniques = [
        ('n_canvas_fp_scripts',    'Canvas',     '#E74C3C'),
        ('n_audio_fp_scripts',     'Audio',      '#3498DB'),
        ('n_navigator_fp_scripts', 'Navigator',  '#2ECC71'),
    ]

    for i, (col, label, color) in enumerate(techniques):
        offset = (i - len(techniques) / 2 + 0.5) * width
        bars = ax.bar(
            x + offset, df_sorted[col], width,
            label=label, color=color,
            edgecolor='black', linewidth=0.5,
        )
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h * 1.02,
                        f'{int(h)}', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([PROFILE_LABELS[p] for p in df_sorted['profile']],
                       rotation=15, ha='right')
    ax.set_ylabel('Number of distinct fingerprinting scripts')
    ax.set_title(title)
    ax.legend(title='Technique')
    plt.tight_layout()
    _save_if_requested(fig, save_path)
    return fig