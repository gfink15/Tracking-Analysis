# src/viz/pixel_plots.py
"""
Visualizations for the pixel-vs-ads analysis.
Consumes CSVs written by scripts/run_pixel_analysis.py.
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
from matplotlib.figure import Figure
from scipy.stats import chi2_contingency

logger = logging.getLogger(__name__)
sns.set_theme(style="whitegrid", context="talk")

PIXEL_PALETTE = {True: "#d62728", False: "#1f77b4"}   # red = pixel, blue = no pixel
PIXEL_LABEL   = {True: "Pixel present", False: "No pixel"}


# ---------------------------------------------------------------------------
# Chart 1 — Category distribution, pixel vs no-pixel, per profile
# ---------------------------------------------------------------------------
def plot_category_by_pixel(csv_path: Path, out_path: Optional[Path] = None,
                            top_n: int = 12) -> Figure:
    """Grouped horizontal bars: within-group % share of each category,
    split by pixel presence. One subplot per profile."""
    df = pd.read_csv(csv_path)
    if df.empty:
        logger.warning("%s is empty; skipping chart", csv_path)
        return plt.figure()

    # If the script normalized already, use pct_within_group; otherwise compute it
    if "pct_within_group" not in df.columns:
        totals = df.groupby(["profile", "site_has_pixel"])["n_ads"].transform("sum")
        df["pct_within_group"] = 100.0 * df["n_ads"] / totals

    # Coerce boolean if it came through as string
    if df["site_has_pixel"].dtype == object:
        df["site_has_pixel"] = df["site_has_pixel"].map(
            {"True": True, "False": False, True: True, False: False}
        )

    top_cats = (df.groupby("category")["n_ads"].sum()
                  .nlargest(top_n).index.tolist())
    df = df[df["category"].isin(top_cats)].copy()
    df["pixel_label"] = df["site_has_pixel"].map(PIXEL_LABEL)

    profiles = sorted(df["profile"].unique())
    fig, axes = plt.subplots(
        1, len(profiles),
        figsize=(6.5 * len(profiles), 0.55 * len(top_cats) + 2),
        sharey=True,
    )
    if len(profiles) == 1:
        axes = [axes]

    for ax, profile in zip(axes, profiles):
        sub = df[df["profile"] == profile]
        sns.barplot(
            data=sub, y="category", x="pct_within_group",
            hue="pixel_label", ax=ax, order=top_cats,
            palette={PIXEL_LABEL[True]: PIXEL_PALETTE[True],
                     PIXEL_LABEL[False]: PIXEL_PALETTE[False]},
        )
        ax.set_title(f"{profile}")
        ax.set_xlabel("Share of ads within group (%)")
        ax.set_ylabel("")
        ax.legend(title="", loc="lower right", fontsize=10)

    fig.suptitle("Ad Category Distribution: Pixel Present vs Absent",
                 y=1.02, fontsize=16, fontweight="bold")
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Chart 2 — Targeting Lift (headline chart)
# ---------------------------------------------------------------------------
def plot_targeting_lift(csv_path: Path, out_path: Optional[Path] = None) -> Figure:
    """Two-panel figure:
       (a) % on-target ads, pixel vs no-pixel, per profile, with chi-square stars
       (b) Targeting-lift ratio with 95% CI error bars
    """
    df = pd.read_csv(csv_path)
    if df.empty:
        return plt.figure()

    if df["site_has_pixel"].dtype == object:
        df["site_has_pixel"] = df["site_has_pixel"].map(
            {"True": True, "False": False, True: True, False: False}
        )
    df["pixel_label"] = df["site_has_pixel"].map(PIXEL_LABEL)

    # Chi-square per profile
    sig_markers = {}
    for profile, sub in df.groupby("profile"):
        try:
            w = sub[sub["site_has_pixel"]].iloc[0]
            n = sub[~sub["site_has_pixel"]].iloc[0]
            table = np.array([
                [w["n_on_target"], w["n_ads"] - w["n_on_target"]],
                [n["n_on_target"], n["n_ads"] - n["n_on_target"]],
            ])
            _, p_val, _, _ = chi2_contingency(table)
            sig_markers[profile] = (
                "***" if p_val < 0.001 else # type: ignore
                "**"  if p_val < 0.01  else # type: ignore
                "*"   if p_val < 0.05  else "ns" # type: ignore
            )
        except (IndexError, ValueError, KeyError):
            sig_markers[profile] = "n/a"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Panel A
    sns.barplot(
        data=df, x="profile", y="pct_on_target", hue="pixel_label", ax=ax1,
        palette={PIXEL_LABEL[True]: PIXEL_PALETTE[True],
                 PIXEL_LABEL[False]: PIXEL_PALETTE[False]},
    )
    ax1.set_title("On-Target Ad Rate by Pixel Presence")
    ax1.set_ylabel("% of ads matching persona interests")
    ax1.set_xlabel("")
    ax1.legend(title="", loc="upper left")
    ax1.tick_params(axis="x", rotation=20)

    y_max = df["pct_on_target"].max()
    for i, profile in enumerate(sorted(df["profile"].unique())):
        ax1.text(i, y_max * 1.08, sig_markers.get(profile, ""),
                 ha="center", fontsize=14, fontweight="bold")

    # Panel B — lift ratio with CI
    lift_rows = []
    for profile, sub in df.groupby("profile"):
        try:
            w = sub[sub["site_has_pixel"]].iloc[0]
            n = sub[~sub["site_has_pixel"]].iloc[0]
            if n["pct_on_target"] > 0:
                lift = w["pct_on_target"] / n["pct_on_target"]
                p1 = w["n_on_target"] / max(w["n_ads"], 1)
                p2 = n["n_on_target"] / max(n["n_ads"], 1)
                se = np.sqrt(
                    p1 * (1 - p1) / max(w["n_ads"], 1) +
                    p2 * (1 - p2) / max(n["n_ads"], 1)
                )
                lift_rows.append({
                    "profile": profile, "lift": lift,
                    "ci_low":  max(lift - 1.96 * se * lift, 0),
                    "ci_high": lift + 1.96 * se * lift,
                })
        except (IndexError, KeyError):
            pass

    lift_df = pd.DataFrame(lift_rows)
    if not lift_df.empty:
        yerr = [lift_df["lift"] - lift_df["ci_low"],
                lift_df["ci_high"] - lift_df["lift"]]
        ax2.bar(lift_df["profile"], lift_df["lift"], yerr=yerr,
                capsize=8, color="#2ca02c", alpha=0.85, edgecolor="black")
        ax2.axhline(1.0, color="black", linestyle="--", linewidth=1,
                    label="No effect (lift = 1)")
        ax2.set_title("Targeting Lift  (pixel ÷ no-pixel)")
        ax2.set_ylabel("Lift ratio")
        ax2.set_xlabel("")
        ax2.tick_params(axis="x", rotation=20)
        ax2.legend(loc="upper left")

    fig.suptitle("Does Pixel Presence Improve Ad Targeting?",
                 y=1.03, fontsize=16, fontweight="bold")
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Chart 3 — Per-platform targeting accuracy
# ---------------------------------------------------------------------------
def plot_targeting_by_platform(csv_path: Path, out_path: Optional[Path] = None) -> Figure:
    df = pd.read_csv(csv_path)
    if df.empty:
        return plt.figure()

    if df["present"].dtype == object:
        df["present"] = df["present"].map(
            {"True": True, "False": False, True: True, False: False}
        )
    df["status"] = df["present"].map({True: "Platform present",
                                       False: "Platform absent"})

    g = sns.catplot(
        data=df, kind="bar",
        x="platform", y="pct_on_target",
        hue="status", col="profile",
        palette={"Platform present": PIXEL_PALETTE[True],
                 "Platform absent":  PIXEL_PALETTE[False]},
        height=6, aspect=1.05, sharey=True,
    )
    g.set_axis_labels("Pixel Platform", "% on-target ads")
    g.set_titles("{col_name}")
    for ax in g.axes.flat:
        ax.tick_params(axis="x", rotation=30)
    g.fig.suptitle("Targeting Accuracy by Specific Pixel Platform",
                   y=1.03, fontsize=16, fontweight="bold")
    g.fig.tight_layout()

    if out_path:
        g.fig.savefig(out_path, dpi=200, bbox_inches="tight")
    return g.fig


# ---------------------------------------------------------------------------
# Chart 4 — Category × platform heatmap
# ---------------------------------------------------------------------------
def plot_category_by_platform_heatmap(csv_path: Path,
                                       out_path: Optional[Path] = None) -> Figure:
    """Which platforms' pixels co-occur with which ad categories?
    One heatmap per profile."""
    df = pd.read_csv(csv_path)
    if df.empty:
        return plt.figure()

    profiles = sorted(df["profile"].unique())
    fig, axes = plt.subplots(1, len(profiles),
                             figsize=(6 * len(profiles), 8), sharey=True)
    if len(profiles) == 1:
        axes = [axes]

    for ax, profile in zip(axes, profiles):
        sub = df[df["profile"] == profile]
        pivot = sub.pivot_table(
            index="category", columns="platform",
            values="n_ads_with_platform", aggfunc="sum", fill_value=0,
        )
        # Normalize to % share per platform column for readability
        pivot_pct = 100 * pivot / pivot.sum(axis=0).replace(0, 1)
        sns.heatmap(pivot_pct, ax=ax, cmap="Reds", annot=True, fmt=".0f",
                    cbar_kws={"label": "% of ads (column)"})
        ax.set_title(f"{profile}")
        ax.set_xlabel("Platform")
        ax.set_ylabel("Category" if ax is axes[0] else "")

    fig.suptitle("Ad Category Share by Pixel Platform",
                 y=1.02, fontsize=16, fontweight="bold")
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Chart 5 — Seeded-site impact (cross-site tracking)
# ---------------------------------------------------------------------------
def plot_seeded_site_impact(csv_path: Path,
                             out_path: Optional[Path] = None) -> Figure:
    df = pd.read_csv(csv_path)
    if df.empty:
        return plt.figure()

    for col in ("site_was_seeded_with_pixel", "site_has_pixel"):
        if df[col].dtype == object:
            df[col] = df[col].map({"True": True, "False": False,
                                    True: True, False: False})

    def label(row):
        s = "Seeded+Pixel" if row["site_was_seeded_with_pixel"] else "Not seeded"
        p = "meas. pixel"  if row["site_has_pixel"] else "no meas. pixel"
        return f"{s}\n({p})"
    df["condition"] = df.apply(label, axis=1)

    fig, ax = plt.subplots(figsize=(13, 6))
    sns.barplot(data=df, x="condition", y="pct_on_target", hue="profile",
                ax=ax, palette="Set2")
    ax.set_title("Cross-Site Tracking: Seeded-With-Pixel Sites vs Others")
    ax.set_ylabel("% on-target ads")
    ax.set_xlabel("")
    ax.legend(title="Profile", loc="upper right", ncol=2)

    # Sample-size labels above bars
    for i, bar in enumerate(ax.patches):
        h = bar.get_height() # type: ignore
        if not np.isnan(h) and h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5, # type: ignore
                    "", ha="center", fontsize=9)  # placeholder — extend if desired

    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
    return fig

def plot_intensity_response(csv_path: Path, out_path: Optional[Path] = None) -> Figure:
    """
    Dose-response chart: on-target ad rate as a function of tracker density.
    One line per profile; x-axis = entity bucket; y-axis = % on-target.
    """
    df = pd.read_csv(csv_path)
    if df.empty:
        return plt.figure()

    # Preserve bucket ordering
    bucket_order = ["0 entities", "1 entity", "2-3 entities",
                    "4-6 entities", "7+ entities"]
    df["entity_bucket"] = pd.Categorical(
        df["entity_bucket"], categories=bucket_order, ordered=True
    )

    fig, ax = plt.subplots(figsize=(12, 7))

    # Control gets a distinct treatment — it's the null hypothesis
    for profile, sub in df.groupby("profile"):
        sub = sub.sort_values("entity_bucket")
        is_control = profile == "control"
        ax.plot(
            sub["entity_bucket"].astype(str),
            sub["pct_on_target"],
            marker="o", markersize=10, linewidth=2.5,
            linestyle="--" if is_control else "-",
            color="gray" if is_control else None,
            alpha=0.6 if is_control else 1.0,
            label=f"{profile} (n={sub['n_ads'].sum():,})",
        )

    ax.set_title("Dose–Response: Tracker Density vs Targeting Accuracy",
                 fontsize=15, fontweight="bold")
    ax.set_xlabel("Distinct tracker entities on serving site")
    ax.set_ylabel("% of ads matching persona interests")
    ax.legend(title="Persona", loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
    return fig

# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def render_all(
    csv_dir: Path = Path("artifacts/ad_tracker_analysis_outputs"),
    out_dir: Path = Path("artifacts/figures"),
) -> Path:
    """Render every chart from the CSVs produced by run_pixel_analysis.py."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / "pixel_vs_ads_report.pdf"

    plan = [
        ("category_by_pixel",     "category_by_pixel.csv",     plot_category_by_pixel),
        ("targeting_lift",        "targeting_accuracy.csv",    plot_targeting_lift),
        ("targeting_by_platform", "targeting_by_platform.csv", plot_targeting_by_platform),
        ("category_by_platform",  "category_by_platform.csv",  plot_category_by_platform_heatmap),
        ("seeded_site_impact",    "seeded_site_impact.csv",    plot_seeded_site_impact),
        ("intensity_response", "targeting_intensity.csv", plot_intensity_response),
    ]

    with PdfPages(pdf_path) as pdf:
        for name, csv_name, fn in plan:
            csv_path = csv_dir / csv_name
            if not csv_path.exists():
                logger.warning("Missing %s — skipping %s", csv_path, name)
                continue
            fig = fn(csv_path, out_path=out_dir / f"{name}.png")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print(f"✅ Report: {pdf_path}")
    print(f"✅ PNGs:   {out_dir}")
    return pdf_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    render_all()