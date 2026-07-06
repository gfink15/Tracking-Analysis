"""
src/viz/ad_plots.py — Visualizations for ad content analysis.

Companion to src/analysis/ads.py and src/analysis/topic_modeling.py.
All functions return matplotlib Figures and accept save_path.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    FIGURES_DIR, PROFILES, PROFILE_LABELS, PROFILE_COLORS,
)
from src.viz.tracker_plots import apply_style, _save_if_requested


# ─────────────────────────────────────────────────────────────────────
# AD VOLUME
# ─────────────────────────────────────────────────────────────────────
def plot_ad_volume(
    df: pd.DataFrame,
    metric: str = 'ads_per_visit',
    title: Optional[str] = None,
    save_path: Optional[Path | str] = None,
) -> Figure:
    """Bar chart of ad volume metric by profile.

    Args:
        df: DataFrame from ad_counts_by_profile()
        metric: 'n_ads', 'ads_per_visit', or 'pct_visits_with_ads'
    """
    apply_style()
    df_sorted = df.set_index('profile').reindex(PROFILES).reset_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        [PROFILE_LABELS[p] for p in df_sorted['profile']],
        df_sorted[metric],
        color=[PROFILE_COLORS[p] for p in df_sorted['profile']],
        edgecolor='black', linewidth=0.5,
    )

    max_val = df_sorted[metric].max()
    for bar, val in zip(bars, df_sorted[metric]):
        # Format integers without decimals, floats with two
        label = f'{val:.0f}' if val == int(val) else f'{val:.2f}'
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max_val * 0.01,
                label, ha='center', va='bottom', fontsize=10)

    ylabel_map = {
        'n_ads':              'Total ads captured',
        'ads_per_visit':      'Ads per page visit',
        'pct_visits_with_ads': '% of visits with ≥1 ad',
    }
    ax.set_ylabel(ylabel_map.get(metric, metric))
    ax.set_title(title or f'Ad volume by profile ({metric})')
    ax.set_ylim(0, max_val * 1.12)
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    _save_if_requested(fig, save_path)
    return fig


# ─────────────────────────────────────────────────────────────────────
# NETWORK DISTRIBUTION HEATMAP
# ─────────────────────────────────────────────────────────────────────
def plot_network_heatmap(
    pivot: pd.DataFrame,
    title: str = 'Advertiser network distribution (% of profile\'s ads)',
    save_path: Optional[Path | str] = None,
) -> Figure:
    """Heatmap of network shares per profile.

    Args:
        pivot: Wide DataFrame from top_advertiser_networks() —
            rows = networks, cols = profiles, cells = percentages.

    Reads vertically: 'shopping' column shows the composition of
    its ads by network. Reads horizontally: each row shows how
    that network's share differs across profiles.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(8, max(5, len(pivot) * 0.4)))
    pivot_values = pivot.to_numpy(dtype=float)

    im = ax.imshow(pivot_values, aspect='auto', cmap='YlOrRd',
                   vmin=0, vmax=max(50, pivot_values.max()))

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([PROFILE_LABELS[p] for p in pivot.columns],
                       rotation=30, ha='right')
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    # Cell annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot_values[i, j]
            color = 'white' if val > 30 else 'black'
            ax.text(j, i, f'{val:.0f}%',
                    ha='center', va='center', color=color, fontsize=9)

    plt.colorbar(im, ax=ax, label='% of profile\'s ads')
    ax.set_title(title)
    ax.grid(False)
    plt.tight_layout()
    _save_if_requested(fig, save_path)
    return fig


# ─────────────────────────────────────────────────────────────────────
# KEYWORD CATEGORY COMPARISON
# ─────────────────────────────────────────────────────────────────────
def plot_keyword_categories(
    df: pd.DataFrame,
    title: str = 'Ad content categories by profile (keyword-based)',
    save_path: Optional[Path | str] = None,
) -> Figure:
    """Grouped bars: categories × profiles, showing % of ads matching.

    Args:
        df: Long-format DataFrame from keyword_category_counts()
    """
    apply_style()
    pivot = (df.pivot(index='category', columns='profile',
                      values='pct_of_ads')
             .reindex(columns=PROFILES)
             .fillna(0))
    # Sort categories by max share (most varied first)
    pivot = pivot.loc[pivot.max(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(pivot.index))
    width = 0.8 / len(PROFILES)

    for i, profile in enumerate(PROFILES):
        if profile not in pivot.columns:
            continue
        offset = (i - len(PROFILES) / 2 + 0.5) * width
        bars = ax.bar(
            x + offset, pivot[profile].to_numpy(dtype=float), width,
            label=PROFILE_LABELS[profile],
            color=PROFILE_COLORS[profile],
            edgecolor='black', linewidth=0.3,
        )
        for bar in bars:
            h = bar.get_height()
            if h > 1:  # don't annotate tiny bars
                ax.text(bar.get_x() + bar.get_width() / 2, h * 1.02,
                        f'{h:.0f}', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=20, ha='right')
    ax.set_ylabel('% of ads with category keyword')
    ax.set_title(title)
    ax.legend(loc='upper right')
    plt.tight_layout()
    _save_if_requested(fig, save_path)
    return fig


# ─────────────────────────────────────────────────────────────────────
# TOPIC MODELING VISUALIZATION
# ─────────────────────────────────────────────────────────────────────
def plot_topic_heatmap(
    df: pd.DataFrame,
    top_n_topics: int = 20,
    title: str = 'Topic distribution across profiles',
    save_path: Optional[Path | str] = None,
) -> Figure:
    """Heatmap: discovered topics × profiles.

    Args:
        df: Long-format DataFrame from topic_distribution_by_profile()
            with columns profile, topic, keywords, pct_of_profile.
        top_n_topics: Show only the N most prevalent topics (overall).
            Topic models often produce 50-100 topics; showing them
            all in one heatmap is unreadable.
    """
    apply_style()
    # Drop outlier topic for cleaner visualization
    df = df[df['topic'] != -1].copy()
    pivot = (df.pivot(index='topic', columns='profile',
                      values='pct_of_profile')
             .reindex(columns=PROFILES)
             .fillna(0))

    # Top topics by total share across all profiles
    total = pivot.sum(axis=1).sort_values(ascending=False)
    pivot = pivot.loc[total.head(top_n_topics).index]
    pivot_values = pivot.to_numpy(dtype=float)

    # Replace topic IDs with keyword summaries for y-axis labels
    keywords_lookup = (df.drop_duplicates('topic')
                        .set_index('topic')['keywords'])
    y_labels = [
        f"#{t}: {keywords_lookup.get(t, '')[:50]}"
        for t in pivot.index
    ]

    fig, ax = plt.subplots(figsize=(9, max(6, top_n_topics * 0.32)))
    im = ax.imshow(pivot_values, aspect='auto', cmap='YlOrRd',
                   vmin=0, vmax=pivot_values.max())

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([PROFILE_LABELS[p] for p in pivot.columns],
                       rotation=30, ha='right')
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(y_labels, fontsize=9)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot_values[i, j]
            if val > 0.5:
                color = 'white' if val > pivot_values.max() * 0.5 else 'black'
                ax.text(j, i, f'{val:.1f}',
                        ha='center', va='center',
                        color=color, fontsize=8)

    plt.colorbar(im, ax=ax, label='% of profile\'s ads')
    ax.set_title(title)
    ax.grid(False)
    plt.tight_layout()
    _save_if_requested(fig, save_path)
    return fig


def plot_differential_topics(
    df: pd.DataFrame,
    profile_label: str,
    top_n: int = 15,
    save_path: Optional[Path | str] = None,
) -> Figure:
    """Horizontal bar chart of topics most overrepresented in profile_a.

    Args:
        df: DataFrame from differential_topics(), sorted by lift desc.
        profile_label: For the title (e.g., "shopping vs control")
    """
    apply_style()
    top = df.head(top_n).iloc[::-1]   # reverse for top-at-top

    # Truncate keywords for label readability
    labels = [
        f"#{int(t)}: {kw[:40]}{'…' if len(kw) > 40 else ''}"
        for t, kw in zip(top['topic_id'], top['keywords'])
    ]

    fig, ax = plt.subplots(figsize=(11, max(4, top_n * 0.35)))
    profile_key = profile_label.split()[0]
    color = PROFILE_COLORS.get(profile_key, '#999999')
    ax.barh(labels, top['lift'].to_numpy(dtype=float), color=color,
            edgecolor='black', linewidth=0.5)

    for i, (lift, pa, pb) in enumerate(zip(
        top['lift'], top['pct_a'], top['pct_b']
    )):
        lift = float(lift)
        pa = float(pa)
        pb = float(pb)
        ax.text(lift + 0.05, i,
                f'  {pa:.1f}% / {pb:.1f}% (×{lift:.1f})',
                va='center', fontsize=9)

    ax.set_xlabel('Lift (% in seeded ÷ % in control)')
    ax.set_title(f'Top {top_n} overrepresented topics: {profile_label}')
    ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5,
               label='No difference')
    ax.legend(loc='lower right')
    plt.tight_layout()
    _save_if_requested(fig, save_path)
    return fig


# ─────────────────────────────────────────────────────────────────────
# TRACKING ↔ AD CORRELATION SCATTER
# ─────────────────────────────────────────────────────────────────────
def plot_tracking_vs_ads_scatter(
    df: pd.DataFrame,
    title: str = 'Tracking intensity vs. ad volume per visit',
    save_path: Optional[Path | str] = None,
) -> Figure:
    """Scatter of trackers vs ads per visit, colored by profile.

    Args:
        df: DataFrame from ads_per_visit_with_tracking() with
            columns profile, n_trackers, n_ads.

    A strong positive slope confirms the "more tracking → more ads"
    hypothesis at the visit level. Differing slopes across profiles
    (e.g., shopping shows steeper relationship) suggest profile-
    specific ad-network responsiveness.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(9, 6))

    for profile in PROFILES:
        sub = df[df['profile'] == profile]
        if sub.empty:
            continue
        ax.scatter(sub['n_trackers'], sub['n_ads'],
                   alpha=0.5, s=40,
                   label=PROFILE_LABELS[profile],
                   color=PROFILE_COLORS[profile],
                   edgecolor='black', linewidth=0.3)

    ax.set_xlabel('Unique trackers contacted per visit')
    ax.set_ylabel('High-confidence ads detected per visit')
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    _save_if_requested(fig, save_path)
    return fig

def plot_category_comparison(cat_df: pd.DataFrame, save_path=None):
    """Grouped bar chart: % of ads in each category, by profile."""
    df = cat_df[cat_df['ad_category'] != 'Unknown']
    pct = (pd.crosstab(df['profile'], df['ad_category'], normalize='index') * 100)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    pct.T.plot(kind='bar', ax=ax,
               color=[PROFILE_COLORS.get(p, 'gray') for p in pct.index])
    ax.set_ylabel('% of OCR-tagged ads')
    ax.set_xlabel('Category')
    ax.set_title('Ad Content Category Distribution by Profile', fontweight='bold')
    ax.legend(title='Profile', labels=[PROFILE_LABELS.get(p, p) for p in pct.index])
    plt.xticks(rotation=0)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    return fig


def plot_differential_targeting(diff_matrix: pd.DataFrame, save_path=None):
    """
    Diverging heatmap: shopping% − control% per (network, category).
    Red = network over-served category to shopping profile (retargeting evidence).
    Blue = network suppressed category for shopping profile.
    """
    vmax = max(abs(diff_matrix.values.min()), abs(diff_matrix.values.max()))
    fig, ax = plt.subplots(figsize=(11, 7))
    sns.heatmap(diff_matrix, annot=True, fmt=".1f",
                cmap="RdBu_r", center=0, vmin=-vmax, vmax=vmax,
                cbar_kws={'label': 'Δ% (shopping − control)'},
                linewidths=0.5, linecolor='white', ax=ax)
    ax.set_title('Differential Targeting: Which Networks Shifted Their Content?',
                 fontweight='bold', pad=15)
    ax.set_xlabel('Content Category')
    ax.set_ylabel('Ad Network')
    plt.figtext(0.5, -0.02,
                'Red cells = network served MORE of this category to shopping profile (retargeting evidence)',
                ha='center', fontsize=9, style='italic')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    return fig


def plot_network_specialization(cat_df: pd.DataFrame, top_networks: list, save_path=None):
    """
    Stacked bar of each network's category mix in shopping profile.
    Networks dominated by one color = specialists; balanced = generalists.
    """
    sub = cat_df[(cat_df['profile'] == 'shopping') &
                 (cat_df['advertiser_network'].isin(top_networks)) &
                 (cat_df['ad_category'] != 'Unknown')]
    mat = pd.crosstab(sub['advertiser_network'], sub['ad_category'], normalize='index') * 100
    mat = mat.reindex(top_networks).dropna(how='all')
    
    fig, ax = plt.subplots(figsize=(11, 6))
    mat.plot(kind='barh', stacked=True, ax=ax, colormap='tab10', width=0.7)
    ax.set_xlabel('% of Network\'s Ads')
    ax.set_ylabel('Ad Network')
    ax.set_title('Network Specialization (Shopping Profile)', fontweight='bold')
    ax.set_xlim(0, 100)
    ax.legend(title='Category', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    return fig

def plot_category_heatmap(cat_matrix: pd.DataFrame,
                          title: str = "VLM Ad Categories — % Share by Profile",
                          figsize: tuple = (8, 9),
                          cmap: str = "YlOrRd",
                          save_path=None) -> Figure:
    """Heatmap of category × profile % share.

    Args:
        cat_matrix: Output of ads.category_matrix() — rows=categories,
                    cols=profiles, values=% share.
        title: Figure title.
        figsize: Figure dimensions.
        cmap: Matplotlib colormap name.
        save_path: Optional path to save the figure.

    Returns:
        The matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cat_matrix,
        annot=True,
        fmt=".1f",
        cmap=cmap,
        cbar_kws={'label': '% of profile ads'},
        linewidths=0.5,
        linecolor='white',
        ax=ax,
    )
    ax.set_title(title, fontsize=13, pad=12)
    ax.set_xlabel("Profile")
    ax.set_ylabel("VLM Ad Category")
    plt.setp(ax.get_xticklabels(), rotation=0)
    plt.setp(ax.get_yticklabels(), rotation=0)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    return fig


def plot_targeting_delta(delta_df: pd.DataFrame,
                         title: str = "Category Targeting Delta\n"
                                      "(shopping − control, percentage points)",
                         figsize: tuple = (9, 8),
                         save_path=None) -> Figure:
    """Diverging horizontal bar chart for shopping-vs-control category deltas.

    Positive bars = over-served to shopping profile.
    Negative bars = over-served to control profile.

    Args:
        delta_df: Output of ads.targeting_delta() — index=category,
                  single column of signed percentage-point differences.
        title: Figure title.
        figsize: Figure dimensions.
        save_path: Optional path to save the figure.

    Returns:
        The matplotlib Figure object.
    """
    # Extract the single column (function returns a 1-col DataFrame)
    if isinstance(delta_df, pd.DataFrame):
        series = delta_df.iloc[:, 0]
    else:
        series = delta_df

    series = series.sort_values()

    # Color-code: positive = shopping over-served (blue), negative = control (red)
    colors = ['#c0392b' if v < 0 else '#2c7fb8' for v in series.values]

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(series.index, series.values, color=colors, edgecolor='black',
            linewidth=0.6)

    # Reference line at 0
    ax.axvline(0, color='black', linewidth=0.8, linestyle='-')

    # Annotate each bar with its numeric value
    for i, v in enumerate(series.values):
        offset = 0.15 if v >= 0 else -0.15
        ha = 'left' if v >= 0 else 'right'
        ax.text(v + offset, i, f"{v:+.1f}", va='center', ha=ha, fontsize=9)

    ax.set_xlabel("Percentage-point difference (shopping − control)")
    ax.set_ylabel("VLM Ad Category")
    ax.set_title(title, fontsize=13, pad=12)

    # Add subtle legend via annotations
    xmax = max(abs(series.min()), abs(series.max())) * 1.25
    ax.set_xlim(-xmax, xmax)
    ax.text(xmax * 0.95, -0.8, "→ over-served to shopping",
            ha='right', fontsize=8, style='italic', color='#2c7fb8')
    ax.text(-xmax * 0.95, -0.8, "over-served to control ←",
            ha='left', fontsize=8, style='italic', color='#c0392b')

    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    return fig