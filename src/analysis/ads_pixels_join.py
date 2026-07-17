# src/analysis/ads_pixels_join.py
"""
Join layer: ads_enriched × pixel presence.

Produces the analytical tables needed to answer:
  1. Do sites with ad pixels serve different ad categories than sites without?
  2. Does pixel presence correlate with better persona-targeting accuracy?
  3. Which specific ad platforms (Meta, DoubleClick, etc.) correlate with
     which ad categories?

Uses DuckDB throughout for consistency with the rest of the pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional

import duckdb
import pandas as pd

from src.analysis.pixels import (
    etld_plus_one,
    extract_pixels_from_parquet,
    aggregate_pixels_by_site,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Register pixel tables in DuckDB
# ---------------------------------------------------------------------------
def register_pixel_tables(
    con: duckdb.DuckDBPyConnection,
    http_requests_parquet_glob: str,
    profile_col: str = "profile",
) -> None:
    """
    Build the measurement-side pixel tables inside the DuckDB session.

    Creates:
      - pixel_hits_raw:  one row per pixel request (long format)
      - site_pixels:     one row per (profile, top_level_etld1)

    Parameters
    ----------
    http_requests_parquet_glob : str
        Glob pattern for the http_requests parquet files, e.g.
        "research_data/parquet/*/http_requests.parquet"
    """
    logger.info("Loading http_requests from %s", http_requests_parquet_glob)

    # Pull requests into pandas so we can run the Python-side classifier.
    # For very large datasets we'd batch this, but for a persona-scale
    # crawl it's fine in memory.
    req_df = con.execute(f"""
        SELECT {profile_col} AS profile,
               visit_id,
               url,
               top_level_url
        FROM read_parquet('{http_requests_parquet_glob}', union_by_name=true)
    """).df()

    logger.info("Classifying %d requests...", len(req_df))
    hits_all = []
    for profile, group in req_df.groupby("profile"):
        hits = extract_pixels_from_parquet(group)
        if not hits.empty:
            hits.insert(0, "profile", profile)
            hits_all.append(hits)

    if not hits_all:
        logger.warning("No pixel hits found in measurement data.")
        pixel_hits = pd.DataFrame(columns=[
            "profile", "visit_id", "top_level_url", "top_level_etld1",
            "request_url", "request_etld1", "pixel_type",
            "matched_domain", "path_confirmed",
        ])
    else:
        pixel_hits = pd.concat(hits_all, ignore_index=True)

    con.register("pixel_hits_raw", pixel_hits)

    # Site-level aggregation (per profile, per site)
    if not pixel_hits.empty:
        site_pixels = aggregate_pixels_by_site(
            pixel_hits,
            group_cols=["profile", "top_level_etld1"],
        )
    else:
        site_pixels = pd.DataFrame(columns=[
            "profile", "top_level_etld1", "has_pixel",
            "n_pixel_hits", "n_distinct_pixel_types", "pixel_types",
            "has_meta", "has_tiktok", "has_doubleclick",
            "has_google_ads", "has_criteo",
        ])

    con.register("site_pixels", site_pixels)
    logger.info("Registered site_pixels with %d rows.", len(site_pixels))


# ---------------------------------------------------------------------------
# Register seeding-side pixel exposure
# ---------------------------------------------------------------------------
def register_seeding_pixels(
    con: duckdb.DuckDBPyConnection,
    seeding_pixel_hits: pd.DataFrame,
) -> None:
    """
    Register the pixel hits observed during PROFILE-BUILD crawls.
    Expects the DataFrame produced by extract_pixels_from_sqlite() with
    columns including [persona, top_level_etld1, pixel_type, ...].

    Creates:
      - seeded_site_pixels: one row per (persona, top_level_etld1)
                            for sites in the seeding history that had pixels
    """
    if seeding_pixel_hits.empty:
        seeded = pd.DataFrame(columns=[
            "persona", "top_level_etld1", "has_pixel",
            "n_pixel_hits", "pixel_types",
        ])
    else:
        seeded = aggregate_pixels_by_site(
            seeding_pixel_hits,
            group_cols=["persona", "top_level_etld1"],
        )

    con.register("seeded_site_pixels", seeded)
    logger.info("Registered seeded_site_pixels with %d rows.", len(seeded))


# ---------------------------------------------------------------------------
# The main join view
# ---------------------------------------------------------------------------
def create_ads_with_pixel_context(
    con: duckdb.DuckDBPyConnection,
    ads_enriched_view: str = "ads_enriched",
    min_confidence: float = 0.7,
    require_networks_agree: bool = False,
) -> None:
    """
    Build ads_with_pixel_context: every valid ad annotated with
    pixel presence on its serving site AND on the corresponding
    seeded site (if that site was in the persona's history).

    Filters applied:
      - is_valid_ad = TRUE
      - vlm_confidence >= min_confidence
      - category IS NOT NULL
      - (optional) networks_agree = TRUE
    """
    # UDF for eTLD+1 so we can compute it inside the SQL
    con.create_function("etld1", etld_plus_one, [str], str)

    agree_filter = "AND ae.networks_agree = TRUE" if require_networks_agree else ""

    con.execute(f"""
        CREATE OR REPLACE VIEW ads_with_pixel_context AS
        WITH ads_base AS (
            SELECT
                ae.profile,
                ae.visit_id,
                ae.page_url,
                etld1(ae.page_url) AS page_etld1,
                ae.ad_hash,
                ae.advertiser_network,
                ae.capture_network,
                ae.networks_agree,
                ae.confidence,
                ae.category,
                ae.product,
                ae.brand,
                ae.vlm_confidence
            FROM {ads_enriched_view} ae
            WHERE ae.is_valid_ad = TRUE
              AND ae.category IS NOT NULL
              {agree_filter}
        )
        SELECT
            a.*,
            -- Measurement-side pixel presence on the site serving the ad
            COALESCE(sp.has_pixel, FALSE)      AS site_has_pixel,
            COALESCE(sp.n_pixel_hits, 0)       AS site_pixel_hits,
            COALESCE(sp.n_distinct_pixel_types, 0) AS site_distinct_pixels,
            sp.pixel_types                     AS site_pixel_types,
            COALESCE(sp.has_meta, FALSE)       AS site_has_meta,
            COALESCE(sp.has_tiktok, FALSE)     AS site_has_tiktok,
            COALESCE(sp.has_doubleclick, FALSE) AS site_has_doubleclick,
            COALESCE(sp.has_google_ads, FALSE) AS site_has_google_ads,
            COALESCE(sp.has_criteo, FALSE)     AS site_has_criteo,

            -- Seeding-side: was this same eTLD+1 in the persona's history
            -- AND did it fire pixels back then?
            COALESCE(ssp.has_pixel, FALSE)     AS site_was_seeded_with_pixel,
            ssp.pixel_types                    AS seeded_pixel_types
        FROM ads_base a
        LEFT JOIN site_pixels sp
            ON a.profile = sp.profile
           AND a.page_etld1 = sp.top_level_etld1
        LEFT JOIN seeded_site_pixels ssp
            ON a.profile = ssp.persona
           AND a.page_etld1 = ssp.top_level_etld1
    """)

    n_ads_result = con.execute("SELECT COUNT(*) FROM ads_with_pixel_context").fetchone()
    n_ads = n_ads_result[0] if n_ads_result else 0
    n_with_pixel_result = con.execute(
        "SELECT COUNT(*) FROM ads_with_pixel_context WHERE site_has_pixel"
    ).fetchone()
    n_with_pixel = n_with_pixel_result[0] if n_with_pixel_result else 0
    logger.info(
        "Created ads_with_pixel_context: %d ads (%d on sites with ad pixels, %.1f%%)",
        n_ads, n_with_pixel, 100.0 * n_with_pixel / max(n_ads, 1)
    )


# ---------------------------------------------------------------------------
# Persona-affinity scoring
# ---------------------------------------------------------------------------
def register_persona_affinity(
    con: duckdb.DuckDBPyConnection,
    affinity_map: dict[str, Iterable[str]],
) -> None:
    """
    Register a persona -> on-target categories mapping as a DuckDB table.

    Parameters
    ----------
    affinity_map : dict
        e.g. {"shopping": ["Apparel", "Retail", "Electronics", ...],
              "control":  []}   # control has no expected affinity
    """
    rows = []
    for persona, cats in affinity_map.items():
        for cat in cats:
            rows.append({"profile": persona, "on_target_category": cat})
    df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["profile", "on_target_category"]
    )
    con.register("persona_affinity", df)
    logger.info("Registered persona_affinity with %d (persona, category) pairs.",
                len(df))


def create_ads_scored(con: duckdb.DuckDBPyConnection) -> None:
    """
    Add an `is_on_target` boolean to each ad based on persona affinity.
    Requires persona_affinity to already be registered.
    """
    con.execute("""
        CREATE OR REPLACE VIEW ads_scored AS
        SELECT
            a.*,
            CASE
                WHEN pa.on_target_category IS NOT NULL THEN TRUE
                ELSE FALSE
            END AS is_on_target
        FROM ads_with_pixel_context a
        LEFT JOIN persona_affinity pa
            ON a.profile = pa.profile
           AND a.category = pa.on_target_category
    """)
    logger.info("Created ads_scored view.")


# ---------------------------------------------------------------------------
# Summary queries — the actual research outputs
# ---------------------------------------------------------------------------
def category_distribution_by_pixel(
    con: duckdb.DuckDBPyConnection,
    profile: Optional[str] = None,
    normalize: bool = True,
) -> pd.DataFrame:
    """
    Category × pixel-presence distribution.

    Returns a long-format DataFrame with columns:
        [profile,] category, site_has_pixel, n_ads, pct_within_group

    This is the primary table for your grouped-bar comparison plot.
    """
    where = f"WHERE profile = '{profile}'" if profile else ""
    df = con.execute(f"""
        SELECT
            profile,
            category,
            site_has_pixel,
            COUNT(*) AS n_ads
        FROM ads_with_pixel_context
        {where}
        GROUP BY profile, category, site_has_pixel
        ORDER BY profile, category, site_has_pixel
    """).df()

    if normalize and not df.empty:
        # Percent within each (profile, pixel-status) group so the two
        # bars are directly comparable regardless of sample size.
        totals = df.groupby(["profile", "site_has_pixel"])["n_ads"].transform("sum")
        df["pct_within_group"] = 100.0 * df["n_ads"] / totals
    else:
        df["pct_within_group"] = None

    return df


def category_distribution_by_platform(
    con: duckdb.DuckDBPyConnection,
    profile: Optional[str] = None,
) -> pd.DataFrame:
    """
    Break out category distribution by SPECIFIC ad platform
    (Meta vs DoubleClick vs Criteo, etc.).

    Useful for spotting which platform's pixels are associated with
    which kinds of ads.
    """
    where = f"WHERE profile = '{profile}'" if profile else ""
    return con.execute(f"""
        WITH platform_flags AS (
            SELECT
                profile, category,
                site_has_meta, site_has_doubleclick, site_has_google_ads,
                site_has_criteo, site_has_tiktok
            FROM ads_with_pixel_context
            {where}
        ),
        unpivoted AS (
            SELECT profile, category, 'Meta' AS platform, site_has_meta AS present FROM platform_flags
            UNION ALL SELECT profile, category, 'DoubleClick', site_has_doubleclick FROM platform_flags
            UNION ALL SELECT profile, category, 'Google Ads', site_has_google_ads FROM platform_flags
            UNION ALL SELECT profile, category, 'Criteo', site_has_criteo FROM platform_flags
            UNION ALL SELECT profile, category, 'TikTok', site_has_tiktok FROM platform_flags
        )
        SELECT profile, platform, category,
               SUM(CASE WHEN present THEN 1 ELSE 0 END) AS n_ads_with_platform,
               SUM(CASE WHEN NOT present THEN 1 ELSE 0 END) AS n_ads_without_platform
        FROM unpivoted
        GROUP BY profile, platform, category
        ORDER BY profile, platform, category
    """).df()


def targeting_accuracy_summary(
    con: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    """
    The core accuracy metric: does pixel presence increase the share
    of on-target ads?

    Requires ads_scored view (i.e. persona_affinity must be registered).

    Returns one row per (profile, site_has_pixel) with:
        n_ads, n_on_target, pct_on_target
    Plus a computed 'targeting_lift' column comparing pixel vs no-pixel.
    """
    df = con.execute("""
        SELECT
            profile,
            site_has_pixel,
            COUNT(*) AS n_ads,
            SUM(CASE WHEN is_on_target THEN 1 ELSE 0 END) AS n_on_target,
            100.0 * AVG(CASE WHEN is_on_target THEN 1.0 ELSE 0.0 END)
                AS pct_on_target
        FROM ads_scored
        GROUP BY profile, site_has_pixel
        ORDER BY profile, site_has_pixel
    """).df()

    # Compute lift = pct(pixel) / pct(no-pixel) per profile
    lift_rows = []
    for profile, sub in df.groupby("profile"):
        with_p = sub[sub["site_has_pixel"] == True]["pct_on_target"]
        no_p = sub[sub["site_has_pixel"] == False]["pct_on_target"]
        if len(with_p) and len(no_p) and no_p.iloc[0] > 0:
            lift = with_p.iloc[0] / no_p.iloc[0]
        else:
            lift = None
        lift_rows.append({"profile": profile, "targeting_lift": lift})
    lift_df = pd.DataFrame(lift_rows)
    return df.merge(lift_df, on="profile", how="left")


def targeting_accuracy_by_platform(
    con: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    """
    Per-platform accuracy: does the presence of Meta / DoubleClick /
    Criteo / etc. correlate with more on-target ads?

    Returns one row per (profile, platform, present_flag).
    """
    return con.execute("""
        WITH scored AS (
            SELECT profile, is_on_target,
                   site_has_meta, site_has_doubleclick,
                   site_has_google_ads, site_has_criteo, site_has_tiktok
            FROM ads_scored
        ),
        unpivoted AS (
            SELECT profile, is_on_target, 'Meta' AS platform, site_has_meta AS present FROM scored
            UNION ALL SELECT profile, is_on_target, 'DoubleClick', site_has_doubleclick FROM scored
            UNION ALL SELECT profile, is_on_target, 'Google Ads', site_has_google_ads FROM scored
            UNION ALL SELECT profile, is_on_target, 'Criteo', site_has_criteo FROM scored
            UNION ALL SELECT profile, is_on_target, 'TikTok', site_has_tiktok FROM scored
        )
        SELECT profile, platform, present,
               COUNT(*) AS n_ads,
               100.0 * AVG(CASE WHEN is_on_target THEN 1.0 ELSE 0.0 END)
                   AS pct_on_target
        FROM unpivoted
        GROUP BY profile, platform, present
        ORDER BY profile, platform, present
    """).df()


def seeded_site_impact(
    con: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    """
    A bonus analysis: for ads served on sites that were ALSO in the
    persona's seeding history with pixels, are they more on-target?

    This directly tests the "cross-site tracking works" hypothesis:
    if a persona was tracked by Meta Pixel on site X during seeding,
    and later sees an ad on site Y that also has Meta Pixel, is that
    ad more relevant?
    """
    return con.execute("""
        SELECT
            profile,
            site_was_seeded_with_pixel,
            site_has_pixel,
            COUNT(*) AS n_ads,
            SUM(CASE WHEN is_on_target THEN 1 ELSE 0 END) AS n_on_target,
            100.0 * AVG(CASE WHEN is_on_target THEN 1.0 ELSE 0.0 END)
                AS pct_on_target
        FROM ads_scored
        GROUP BY profile, site_was_seeded_with_pixel, site_has_pixel
        ORDER BY profile, site_was_seeded_with_pixel DESC, site_has_pixel DESC
    """).df()