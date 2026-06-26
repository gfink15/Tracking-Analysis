# Visualization Layer

This document covers the plotting modules in `src/viz/` and how they
work together with notebooks to produce publication-quality figures.

## Table of Contents

1. [Design Contract](#design-contract)
2. [Project Styling: `apply_style()`](#project-styling-apply_style)
3. [`tracker_plots.py`](#tracker_plotspy)
4. [`cookie_plots.py`](#cookie_plotspy)
5. [`ad_plots.py`](#ad_plotspy)
6. [Notebook Patterns](#notebook-patterns)
7. [Common Issues](#common-issues)

---

## Design Contract

Every plotting function in `src/viz/` follows this contract:

```python
def plot_X(
    df: pd.DataFrame,           # data from src/analysis/
    ...,                         # plot-specific params
    title: str = "...",         # default title, overridable
    save_path: Path | str | None = None,
) -> plt.Figure:                # Figure, NOT None
    apply_style()
    fig, ax = plt.subplots(...)
    # ... build the plot ...
    _save_if_requested(fig, save_path)
    return fig

Why These Choices?

Return Figure, never plt.show(): Returning the Figure object means the function works identically in Jupyter (which auto-displays returned figures) AND in batch scripts (which save without a display). Calling plt.show() inside would force interactive mode.

Optional save_path: One uniform mechanism for saving. Relative paths resolve to FIGURES_DIR; absolute paths used as-is.

apply_style() is explicit, not auto: Putting plt.rcParams.update at module level executes on import, which is surprising. Making it opt-in via a function call keeps imports side-effect-free.
Project Styling: apply_style()

Defined in src/viz/tracker_plots.py and reused by all other plot modules.
Settings Applied
Setting	Value	Purpose
font.family	'sans-serif'	Readable at small sizes
font.sans-serif	['DejaVu Sans', 'Arial', 'Helvetica']	Fallback chain
font.size	11	Default text size
axes.titlesize	13	Plot titles
axes.labelsize	12	Axis labels
axes.spines.top/right	False	Cleaner look
axes.grid	True	Reference lines
grid.alpha	0.3	Subtle grid
figure.dpi	110	Screen render
savefig.dpi	300	Publication render
savefig.bbox	'tight'	Auto-crop white space
Helper: _save_if_requested(fig, save_path)

Internal helper used by every plotting function:

    Returns early if save_path is None
    Resolves relative paths against FIGURES_DIR
    Creates parent directory if needed
    Saves with the styled DPI

tracker_plots.py

Purpose: Visualizations for src/analysis/trackers.py output.
Functions
plot_prevalence_bars(df, value_col, ylabel, title, save_path)

Bar chart of a per-profile metric. Used for the headline "unique trackers per profile" figure.

Key parameters:

    df from tracker_prevalence_by_profile()
    value_col — which column to plot (default 'n_unique_etld1')

Bar colors come from config.PROFILE_COLORS, ensuring consistency across every figure.

Value labels: Each bar gets a text annotation with the exact value, with 12% headroom above the tallest bar so labels don't clip.
plot_tracker_heatmap(df, top_n, title, save_path)

Heatmap of tracker presence (%) across profiles. Often the single most informative figure in a tracking study.

Sizing: Figure height scales with top_n (max(6, top_n * 0.3)) so labels never overlap.

Cell annotations: Only added when top_n <= 30 to avoid clutter on dense heatmaps. Text color flips between black and white based on cell darkness ('white' if val > 50 else 'black').
plot_jaccard_matrix(matrix, title, save_path)

Square heatmap of pairwise Jaccard similarities. Every cell annotated; diagonals always 1.0.
plot_differential_trackers(df, profile_label, top_n, save_path)

Horizontal bar chart of trackers with highest lift in one profile vs. another. Each bar annotated with visits_a/visits_b (×lift) for full context.

Reference line: Dashed vertical line at x=1.0 marks "no difference" — bars to the right indicate overrepresentation in the seeded profile.
plot_distribution_comparison(df, value_col, metric_name, log_scale, save_path)

Overlay histograms across profiles. Uses shared bins computed from the 99th percentile to keep extreme outliers from compressing the visible range. Log y-scale by default because OpenWPM metrics are heavy-tailed.
cookie_plots.py

Purpose: Visualizations for src/analysis/cookies.py output.
Module-Level Constants
LIFESPAN_ORDER: list[str]

Canonical ordering for lifespan buckets in stacked bars (shortest at bottom, longest at top). Ordering matters because the "more persistent = more concerning" reading should be visually upward.
LIFESPAN_COLORS: dict[str, str]

Color encoding from cool (blue, ephemeral) to warm (purple, very long-lived). This deliberately maps lifespan-concern to color temperature so readers' eyes catch the privacy implication intuitively.
Functions
plot_first_vs_third_party(df, title, save_path)

Stacked bar of first-party vs. third-party cookies. Stacking lets readers see both absolute volume (bar height) AND composition (first/third split) at once.
plot_lifespan_distribution(df, title, normalize, save_path)

Stacked bar of lifespan composition per profile.

normalize=True (default): Each bar sums to 100%. Composition comparisons work even when profiles have very different totals.

normalize=False: Raw counts. Use when absolute volume matters.

Legend reversal: The legend is reversed so it reads top-to-bottom in the same order as the stacked bars (longest lifespan at top).
plot_retargeting_presence(df, metric, title, save_path)

Grouped horizontal bar chart: retargeting networks × profiles. Horizontal orientation chosen because retargeter names are long.

Sorting: Retargeters sorted by total volume so the most prevalent are on top — eye-tracking principle.
plot_sync_summary(df, title, save_path)

Two-bar grouped chart per profile: total sync events vs. visits with syncs. The ratio between them is informative on its own.
plot_fingerprinter_summary(df, title, save_path)

Grouped bars: fingerprinting techniques per profile (canvas, audio, navigator). Lives in cookie_plots.py for now because it follows the same pattern; split into its own fingerprinting_plots.py if that analysis grows.
ad_plots.py

Purpose: Visualizations for src/analysis/ads.py and src/analysis/topic_modeling.py.
Functions
plot_ad_volume(df, metric, title, save_path)

Bar chart of ad volume by profile. Adaptive y-label based on metric parameter ('n_ads', 'ads_per_visit', 'pct_visits_with_ads').
plot_network_heatmap(pivot, title, save_path)

Heatmap of advertiser network shares per profile. Read columns vertically for profile composition; read rows horizontally for network distribution across profiles.
plot_keyword_categories(df, title, save_path)

Grouped bar chart of keyword-based categories × profiles. Categories sorted by max share so the most varied appear leftmost.

Annotation guard: Bars below 1% are not annotated to reduce clutter.
plot_topic_heatmap(df, top_n_topics, title, save_path)

Heatmap of BERTopic-discovered topics × profiles.

Topic labels: Show "#<id>: <keywords[:50]>" so users see both the canonical topic ID (for citation) and the interpretive keywords.

Outlier topic dropped: BERTopic's -1 cluster (unclassifiable documents) is filtered out before plotting.
plot_differential_topics(df, profile_label, top_n, save_path)

Horizontal bar chart of topics overrepresented in one profile. Each bar annotated with pct_a / pct_b (×lift).
plot_tracking_vs_ads_scatter(df, title, save_path)

Scatter of trackers vs. ads per visit, colored by profile. Reveals both the overall correlation AND group-specific patterns simultaneously.
Notebook Patterns
Standard Notebook Header
python

%load_ext autoreload
%autoreload 2

import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))

import pandas as pd
from config import PROFILES, ALPHA, FIGURES_DIR
from src.utils.db import db_session

# import analysis + viz modules as needed
from src.analysis.trackers import tracker_prevalence_by_profile
from src.viz.tracker_plots import apply_style, plot_prevalence_bars

apply_style()
pd.set_option('display.max_columns', 50)

# Per-notebook figure subdirectory
FIG_DIR = FIGURES_DIR / "02_tracking_comparison"
FIG_DIR.mkdir(parents=True, exist_ok=True)

Why autoreload: Edits to src/ files take effect without restarting the kernel. The single biggest productivity feature for iterative analysis.

Why per-notebook FIG_DIR: Organizes outputs so notebook 02's figures don't collide with notebook 03's.
Standard Cell Structure
python

# %% [markdown]
# ## Section X: Research question
# (explain what this section investigates)

# %% — compute
df = analysis_function(...)
print(df.to_string(index=False))

# %% — visualize
plot_function(df, save_path=FIG_DIR / 'figXX_descriptive_name.pdf')

# %% [markdown]
# **Interpretation:** (state what the result means)

This rhythm (markdown question → compute → display → visualize → markdown interpretation) is what turns a notebook from a code dump into a readable research narrative.
Common Issues
Figures don't save (no file appears)

Cause: save_path is None, OR you passed a relative path that's resolving somewhere unexpected.

Fix:

    Pass save_path=FIG_DIR / 'fig01.pdf' explicitly
    Use absolute paths if in doubt: save_path='/full/path/fig.pdf'
    Check the printout — every save logs ✓ Saved figure → <path>

Bars/colors in wrong order

Cause: Forgot to reindex to PROFILES. DataFrames come back from SQL in whatever order DuckDB chooses.

Fix: Every plot function does this internally:
python

df_sorted = df.set_index('profile').reindex(PROFILES).reset_index()

If you're writing custom plots, do the same.
Heatmap labels unreadable

Cause: Too many rows for the figure size.

Fix: Pass top_n to limit rows, or modify the function's figsize calculation. Default scaling is max(6, top_n * 0.3) which works up to ~30 rows.
"Figure size N exceeds maximum"

Cause: Computed figure height exceeded matplotlib's max DPI×inches limit (typically when top_n > 100).

Fix: Either reduce top_n or split into multiple plots. Topic modeling with 200+ topics commonly hits this.
Colors look wrong / inconsistent

Cause: Custom plotting code that doesn't pull from PROFILE_COLORS.

Fix: Always use color=PROFILE_COLORS[profile]. This is the single source of truth.
Plots appear with old data after re-ingestion

Cause: DuckDB connection cached in the Jupyter kernel still points at old data, OR analysis.duckdb views are stale.

Fix:

    Run python scripts/init_database.py
    Restart Jupyter kernel (Kernel → Restart)
    Re-run cells
