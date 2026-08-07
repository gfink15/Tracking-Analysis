# src/analysis/ads_pixels_join.py
"""
Join layer: ads_enriched x pixel presence.

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
    AD_PIXEL_SIGNATURES,   # NEW: expose the signature keys for validation
)

logger = logging.getLogger(__name__)
parent_dir = str(Path(__file__).resolve().parent.parent.parent)


# NEW: default filter — only these count as "tracking pixels" for research
DEFAULT_TARGET_PIXELS = [
    "Meta Pixel",
    "DoubleClick",
    "Google Ads Conversion",
    "Criteo",
    "TikTok Pixel",
    "Pinterest Tag",
    "Snap Pixel",
    "Microsoft UET",
    "X/Twitter Pixel",
    "LinkedIn Insight",
]


def _validate_target_pixels(target_pixels: Optional[list[str]]) -> Optional[list[str]]:
    """NEW: Guard against typos in target_pixels — fail loud, not silent."""
    if target_pixels is None:
        return None
    valid = set(AD_PIXEL_SIGNATURES.keys())
    unknown = [p for p in target_pixels if p not in valid]
    if unknown:
        raise ValueError(
            f"Unknown pixel type(s): {unknown}. "
            f"Valid options: {sorted(valid)}"
        )
    return target_pixels


# ---------------------------------------------------------------------------
# Register pixel tables in DuckDB
# ---------------------------------------------------------------------------
def register_pixel_tables(
    con: duckdb.DuckDBPyConnection,
    http_requests_parquet_glob: str,
    profile_col: str = "profile",
    target_pixels: Optional[list[str]] = None,
    require_path_confirmed: bool = True,
    # NEW: entity-relationship filters (default to real 3P advertising)
    only_third_party: bool = True,
    exclude_technical_3p: bool = True,
    relationship_tiers: Optional[list[str]] = None,
) -> None:
    """
    Build measurement-side pixel tables from the ENRICHED http_requests
    parquet. Uses parent_entity for cross-property attribution.

    New defaults (only_third_party=True, exclude_technical_3p=True) mean
    that "has_pixel" now specifically flags sites carrying THIRD-PARTY
    ADVERTISING pixels, which is what your research question is actually
    asking about. Override to False for exposure-only analyses.
    """
    target_pixels = _validate_target_pixels(
        target_pixels if target_pixels is not None else DEFAULT_TARGET_PIXELS
    )

    logger.info("Loading enriched http_requests from %s",
                http_requests_parquet_glob)

    # NEW: pull the enriched columns
    req_df = con.execute(f"""
        SELECT {profile_col}      AS profile,
               visit_id,
               url,
               top_level_url,
               domain,
               subsidiary_entity,
               parent_entity,
               relationship_tier,
               is_technical_3p
        FROM read_parquet('{http_requests_parquet_glob}', union_by_name=true)
    """).df()

    logger.info("Classifying %d enriched requests...", len(req_df))
    hits_all = []
    for profile, group in req_df.groupby("profile"):
        hits = extract_pixels_from_parquet(group)
        if not hits.empty:
            hits.insert(0, "profile", profile)
            hits_all.append(hits)

    if not hits_all:
        logger.warning("No pixel hits found in measurement data.")
        pixel_hits = pd.DataFrame()
    else:
        pixel_hits = pd.concat(hits_all, ignore_index=True)

    con.register("pixel_hits_raw", pixel_hits)

    if not pixel_hits.empty:
        site_pixels = aggregate_pixels_by_site(
            pixel_hits,
            group_cols=["profile", "top_level_parent_entity"],   # CHANGED
            target_pixels=target_pixels,
            require_path_confirmed=require_path_confirmed,
            only_third_party=only_third_party,                   # NEW
            exclude_technical_3p=exclude_technical_3p,           # NEW
            relationship_tiers=relationship_tiers,               # NEW
        )
    else:
        site_pixels = pd.DataFrame(columns=[
            "profile",
            "top_level_parent_entity",   # was: top_level_etld1
            "has_pixel",
            "n_pixel_hits", "n_distinct_pixel_types", "pixel_types",
            "distinct_tracker_entities",
            "has_meta", "has_tiktok", "has_doubleclick",
            "has_google_ads", "has_criteo",
        ])

    con.register("site_pixels", site_pixels)

    n_sites = len(site_pixels)
    n_with = int(site_pixels["has_pixel"].sum()) if not site_pixels.empty else 0
    pct = 100.0 * n_with / max(n_sites, 1)
    logger.info(
        "site_pixels: %d entities | %d with pixel (%.1f%%) | "
        "filter=%s | path_confirmed=%s | 3P-only=%s | exclude-tech=%s",
        n_sites, n_with, pct,
        target_pixels or "ALL",
        require_path_confirmed,
        only_third_party,
        exclude_technical_3p,
    )
    if pct > 95 or pct < 5:
        logger.warning("⚠️  Pixel share (%.1f%%) is outside 5–95%%; check filters.", pct)


# ---------------------------------------------------------------------------
# Register seeding-side pixel exposure
# ---------------------------------------------------------------------------
def register_seeding_pixels(
    con: duckdb.DuckDBPyConnection,
    seeding_pixel_hits: pd.DataFrame,
    target_pixels: Optional[list[str]] = None,
    require_path_confirmed: bool = True,
) -> None:
    target_pixels = _validate_target_pixels(
        target_pixels if target_pixels is not None else DEFAULT_TARGET_PIXELS
    )

    if seeding_pixel_hits.empty:
        # FIX: use the new parent-entity schema so downstream JOIN works
        # even when no seeding data is available.
        seeded = pd.DataFrame(columns=[
            "persona",
            "top_level_parent_entity",   # was: top_level_etld1
            "has_pixel",
            "n_pixel_hits",
            "n_distinct_pixel_types",
            "pixel_types",
            "distinct_tracker_entities",
            "has_meta", "has_tiktok", "has_doubleclick",
            "has_google_ads", "has_criteo",
        ])
    else:
        seeded = aggregate_pixels_by_site(
            seeding_pixel_hits,
            group_cols=["persona", "top_level_parent_entity"],  # was: top_level_etld1
            target_pixels=target_pixels,
            require_path_confirmed=require_path_confirmed,
        )
    con.register("seeded_site_pixels", seeded)
    logger.info("Registered seeded_site_pixels with %d rows.", len(seeded))

# ---------------------------------------------------------------------------
# The main join view
# ---------------------------------------------------------------------------
def create_ads_with_pixel_context(
    con: duckdb.DuckDBPyConnection,
    ads_enriched_view: str = "ads_enriched",
    http_requests_enriched_view: str = "http_requests_enriched",  # NEW
    require_networks_agree: bool = False,
) -> None:
    """
    Build ads_with_pixel_context: every valid ad annotated with pixel
    presence on its serving site's PARENT ENTITY.
    """
    agree_filter = "AND ae.networks_agree = TRUE" if require_networks_agree else ""

    # NEW: look up parent_entity for each ad's page_url from the enriched
    # request log. We take the first non-null parent_entity per top_level_url.
    con.execute(f"""
        CREATE OR REPLACE VIEW page_url_to_entity AS
        SELECT top_level_url,
               ANY_VALUE(parent_entity) AS page_parent_entity
        FROM {http_requests_enriched_view}
        WHERE parent_entity IS NOT NULL
        GROUP BY top_level_url
    """)

    con.execute(f"""
        CREATE OR REPLACE VIEW ads_with_pixel_context AS
        WITH ads_base AS (
            SELECT
                ae.profile,
                ae.visit_id,
                ae.page_url,
                p2e.page_parent_entity,
                ae.ad_hash,
                ae.advertiser_network,
                ae.capture_network,
                ae.networks_agree,
                ae.confidence,
                ae.category,
                ae.product,
                ae.brand,
                ae.vlm_confidence,
                ae.same_company
            FROM {ads_enriched_view} ae
            LEFT JOIN page_url_to_entity p2e
                ON ae.page_url = p2e.top_level_url
            WHERE ae.is_valid_ad = TRUE
              AND ae.category IS NOT NULL
              AND ae.same_company IS NOT TRUE
              {agree_filter}
        )
        SELECT
            a.*,
            COALESCE(sp.has_pixel, FALSE)              AS site_has_pixel,
            COALESCE(sp.n_pixel_hits, 0)               AS site_pixel_hits,
            COALESCE(sp.n_distinct_pixel_types, 0)     AS site_distinct_pixels,
            COALESCE(sp.distinct_tracker_entities, 0)  AS site_tracker_entities,
            sp.pixel_types                             AS site_pixel_types,
            COALESCE(sp.has_meta, FALSE)               AS site_has_meta,
            COALESCE(sp.has_tiktok, FALSE)             AS site_has_tiktok,
            COALESCE(sp.has_doubleclick, FALSE)        AS site_has_doubleclick,
            COALESCE(sp.has_google_ads, FALSE)         AS site_has_google_ads,
            COALESCE(sp.has_criteo, FALSE)             AS site_has_criteo,
            COALESCE(ssp.has_pixel, FALSE)             AS site_was_seeded_with_pixel,
            ssp.pixel_types                            AS seeded_pixel_types
        FROM ads_base a
        LEFT JOIN site_pixels sp
            ON  a.profile            = sp.profile
            AND a.page_parent_entity = sp.top_level_parent_entity   -- CHANGED
        LEFT JOIN seeded_site_pixels ssp
            ON  a.profile            = ssp.persona
            AND a.page_parent_entity = ssp.top_level_parent_entity  -- CHANGED
    """)

    n_ads = con.execute("SELECT COUNT(*) FROM ads_with_pixel_context").fetchone()[0] # type: ignore
    n_with = con.execute(
        "SELECT COUNT(*) FROM ads_with_pixel_context WHERE site_has_pixel"
    ).fetchone()[0] # type: ignore
    n_no_entity = con.execute(
        "SELECT COUNT(*) FROM ads_with_pixel_context WHERE page_parent_entity IS NULL"
    ).fetchone()[0] # type: ignore

    logger.info(
        "ads_with_pixel_context: %d ads | %d on pixel entities (%.1f%%) | "
        "%d ads (%.1f%%) had no parent_entity lookup",
        n_ads, n_with, 100.0 * n_with / max(n_ads, 1),
        n_no_entity, 100.0 * n_no_entity / max(n_ads, 1),
    )
    if n_no_entity / max(n_ads, 1) > 0.1:
        logger.warning(
            "⚠️  >10%% of ads have no parent_entity match. Check that "
            "ads.page_url values match top_level_url values exactly."
        )


# ---------------------------------------------------------------------------
# Persona-affinity scoring (unchanged)
# ---------------------------------------------------------------------------
def register_persona_affinity(
    con: duckdb.DuckDBPyConnection,
    affinity_map: dict[str, Iterable[str]],
) -> None:
    rows = []
    for persona, cats in affinity_map.items():
        for cat in cats:
            rows.append({"profile": persona, "on_target_category": cat})
    df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["profile", "on_target_category"]
    )
    con.register("persona_affinity", df)
    logger.info("Registered persona_affinity with %d (persona, category) pairs.", len(df))


def create_ads_scored(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("""
        CREATE OR REPLACE VIEW ads_scored AS
        SELECT
            a.*,
            CASE WHEN pa.on_target_category IS NOT NULL THEN TRUE ELSE FALSE END
                AS is_on_target
        FROM ads_with_pixel_context a
        LEFT JOIN persona_affinity pa
            ON  a.profile  = pa.profile
            AND a.category = pa.on_target_category
    """)
    logger.info("Created ads_scored view.")


# ---------------------------------------------------------------------------
# Summary queries (unchanged — the fix upstream is what makes these correct)
# ---------------------------------------------------------------------------
def category_distribution_by_pixel(
    con: duckdb.DuckDBPyConnection,
    profile: Optional[str] = None,
    normalize: bool = True,
) -> pd.DataFrame:
    where = f"WHERE profile = '{profile}'" if profile else ""
    df = con.execute(f"""
        SELECT profile, category, site_has_pixel, COUNT(*) AS n_ads
        FROM ads_with_pixel_context
        {where}
        GROUP BY profile, category, site_has_pixel
        ORDER BY profile, category, site_has_pixel
    """).df()
    if normalize and not df.empty:
        totals = df.groupby(["profile", "site_has_pixel"])["n_ads"].transform("sum")
        df["pct_within_group"] = 100.0 * df["n_ads"] / totals
    else:
        df["pct_within_group"] = None
    return df


def category_distribution_by_platform(
    con: duckdb.DuckDBPyConnection,
    profile: Optional[str] = None,
) -> pd.DataFrame:
    where = f"WHERE profile = '{profile}'" if profile else ""
    return con.execute(f"""
        WITH platform_flags AS (
            SELECT profile, category,
                   site_has_meta, site_has_doubleclick,
                   site_has_google_ads, site_has_criteo, site_has_tiktok
            FROM ads_with_pixel_context
            {where}
        ),
        unpivoted AS (
            SELECT profile, category, 'Meta'        AS platform, site_has_meta        AS present FROM platform_flags
            UNION ALL
            SELECT profile, category, 'DoubleClick',              site_has_doubleclick           FROM platform_flags
            UNION ALL
            SELECT profile, category, 'Google Ads',               site_has_google_ads            FROM platform_flags
            UNION ALL
            SELECT profile, category, 'Criteo',                   site_has_criteo                FROM platform_flags
            UNION ALL
            SELECT profile, category, 'TikTok',                   site_has_tiktok                FROM platform_flags
        )
        SELECT profile, platform, category,
               SUM(CASE WHEN present     THEN 1 ELSE 0 END) AS n_ads_with_platform,
               SUM(CASE WHEN NOT present THEN 1 ELSE 0 END) AS n_ads_without_platform
        FROM unpivoted
        GROUP BY profile, platform, category
        ORDER BY profile, platform, category
    """).df()


def targeting_accuracy_summary(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    df = con.execute("""
        SELECT
            profile,
            site_has_pixel,
            COUNT(*) AS n_ads,
            SUM(CASE WHEN is_on_target THEN 1 ELSE 0 END) AS n_on_target,
            100.0 * AVG(CASE WHEN is_on_target THEN 1.0 ELSE 0.0 END) AS pct_on_target
        FROM ads_scored
        GROUP BY profile, site_has_pixel
        ORDER BY profile, site_has_pixel
    """).df()

    lift_rows = []
    for profile, sub in df.groupby("profile"):
        with_p = sub[sub["site_has_pixel"] == True]["pct_on_target"]
        no_p   = sub[sub["site_has_pixel"] == False]["pct_on_target"]
        lift = (with_p.iloc[0] / no_p.iloc[0]
                if len(with_p) and len(no_p) and no_p.iloc[0] > 0 else None)
        lift_rows.append({"profile": profile, "targeting_lift": lift})

    return df.merge(pd.DataFrame(lift_rows), on="profile", how="left")


def targeting_accuracy_by_platform(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return con.execute("""
        WITH scored AS (
            SELECT profile, is_on_target,
                   site_has_meta, site_has_doubleclick,
                   site_has_google_ads, site_has_criteo, site_has_tiktok
            FROM ads_scored
        ),
        unpivoted AS (
            SELECT profile, is_on_target, 'Meta'        AS platform, site_has_meta        AS present FROM scored
            UNION ALL
            SELECT profile, is_on_target, 'DoubleClick',              site_has_doubleclick           FROM scored
            UNION ALL
            SELECT profile, is_on_target, 'Google Ads',               site_has_google_ads            FROM scored
            UNION ALL
            SELECT profile, is_on_target, 'Criteo',                   site_has_criteo                FROM scored
            UNION ALL
            SELECT profile, is_on_target, 'TikTok',                   site_has_tiktok                FROM scored
        )
        SELECT profile, platform, present,
               COUNT(*) AS n_ads,
               100.0 * AVG(CASE WHEN is_on_target THEN 1.0 ELSE 0.0 END) AS pct_on_target
        FROM unpivoted
        GROUP BY profile, platform, present
        ORDER BY profile, platform, present
    """).df()


def seeded_site_impact(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Cross-site tracking test: for ads served on sites that were ALSO in
    the persona's seeding history with pixels, are they more on-target?

    If a persona was tracked by Meta Pixel on site X during seeding, and
    later sees an ad on site Y that also has Meta Pixel, is that ad more
    relevant than one served on a site that wasn't in their seeded history?
    """
    return con.execute("""
        SELECT
            profile,
            site_was_seeded_with_pixel,
            site_has_pixel,
            COUNT(*) AS n_ads,
            SUM(CASE WHEN is_on_target THEN 1 ELSE 0 END) AS n_on_target,
            100.0 * AVG(CASE WHEN is_on_target THEN 1.0 ELSE 0.0 END) AS pct_on_target
        FROM ads_scored
        GROUP BY profile, site_was_seeded_with_pixel, site_has_pixel
        ORDER BY profile, site_was_seeded_with_pixel DESC, site_has_pixel DESC
    """).df()


# ---------------------------------------------------------------------------
# NEW: Pixel intensity (dose-response) analysis
# ---------------------------------------------------------------------------
def get_pixel_intensity_stats(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Buckets sites by the number of distinct advertising pixels they carry,
    then measures on-target rate per bucket per profile.

    Tests the hypothesis: more trackers → higher targeting accuracy
    (a dose-response relationship). This is a stronger causal claim than
    the binary pixel/no-pixel comparison and strengthens the paper's
    contribution.

    Returns columns:
        profile, intensity_bucket, pct_on_target, n_ads
    """
    return con.execute("""
        SELECT
            profile,
            CASE
                WHEN site_distinct_pixels = 0 THEN '0 (None)'
                WHEN site_distinct_pixels = 1 THEN '1 (Single)'
                WHEN site_distinct_pixels BETWEEN 2 AND 3 THEN '2-3 (Moderate)'
                ELSE '4+ (High)'
            END AS intensity_bucket,
            100.0 * AVG(CASE WHEN is_on_target THEN 1.0 ELSE 0.0 END) AS pct_on_target,
            COUNT(*) AS n_ads
        FROM ads_scored
        GROUP BY profile, intensity_bucket
        ORDER BY profile,
            CASE intensity_bucket
                WHEN '0 (None)'         THEN 0
                WHEN '1 (Single)'       THEN 1
                WHEN '2-3 (Moderate)'   THEN 2
                ELSE 3
            END
    """).df()