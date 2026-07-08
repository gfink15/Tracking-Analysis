"""
src/analysis/ads.py — VLM-powered quantitative analysis of captured ads.

Confidence model: 'high' | 'medium' | 'low' (categorical, not numeric).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from config import PROFILES
from src.utils.db import db_session


# ─────────────────────────────────────────────────────────────────────
# CONFIDENCE FILTERING (categorical: 'high' | 'medium' | 'low')
# ─────────────────────────────────────────────────────────────────────
CONFIDENCE_LEVELS = {
    'high':   ("'high'",),
    'medium': ("'high'", "'medium'"),
    'low':    ("'high'", "'medium'", "'low'"),
}


def _confidence_clause(min_confidence: str = 'high',
                       table_alias: str = '') -> str:
    """Build SQL WHERE clause fragment for categorical confidence filtering.
    
    Args:
        min_confidence: 'high', 'medium', or 'low' (inclusive floor).
        table_alias: Optional table alias prefix (e.g., 'ae' -> 'ae.confidence').
    
    Returns:
        SQL fragment like: "ae.is_valid_ad = true AND ae.confidence IN ('high')"
    """
    prefix = f"{table_alias}." if table_alias else ""
    levels = CONFIDENCE_LEVELS.get(min_confidence, CONFIDENCE_LEVELS['high'])
    levels_sql = ", ".join(levels)
    return (f"{prefix}is_valid_ad = true "
            f"AND {prefix}confidence IN ({levels_sql})")


# ─────────────────────────────────────────────────────────────────────
# COVERAGE / SANITY CHECKS
# ─────────────────────────────────────────────────────────────────────
def vlm_coverage_report() -> pd.DataFrame:
    """Operational health check: how much of your ad corpus has been
    successfully analyzed by the VLM, broken down by profile and
    confidence tier.
    """
    with db_session(read_only=True) as con:
        return con.execute("""
            SELECT
                profile,
                COUNT(*) AS total_ads,
                SUM(CASE WHEN is_valid_ad IS NULL THEN 1 ELSE 0 END) AS unprocessed,
                SUM(CASE WHEN is_valid_ad = false THEN 1 ELSE 0 END) AS invalid_ads,
                SUM(CASE WHEN is_valid_ad = true AND confidence = 'high' 
                    THEN 1 ELSE 0 END) AS high_conf,
                SUM(CASE WHEN is_valid_ad = true AND confidence = 'medium' 
                    THEN 1 ELSE 0 END) AS medium_conf,
                SUM(CASE WHEN is_valid_ad = true AND confidence = 'low' 
                    THEN 1 ELSE 0 END) AS low_conf,
                ROUND(100.0 * SUM(CASE WHEN is_valid_ad = true 
                                       AND confidence = 'high' 
                                  THEN 1 ELSE 0 END) 
                            / NULLIF(COUNT(*), 0), 1) AS pct_high_conf
            FROM ads_enriched
            GROUP BY profile
            ORDER BY profile
        """).df()


def ad_counts_by_profile(min_confidence: str = 'high') -> pd.DataFrame:
    """Total valid ads per profile at the specified confidence floor."""
    with db_session(read_only=True) as con:
        return con.execute(f"""
            SELECT profile, COUNT(*) AS n_ads
            FROM ads_enriched
            WHERE {_confidence_clause(min_confidence)}
            GROUP BY profile
            ORDER BY n_ads DESC
        """).df()


# ─────────────────────────────────────────────────────────────────────
# 1. SOURCE ANALYSIS
# ─────────────────────────────────────────────────────────────────────
def network_distribution_by_profile(min_confidence: str = 'high') -> pd.DataFrame:
    """Long-format: (profile, advertiser_network, n_ads, pct)."""
    with db_session(read_only=True) as con:
        return con.execute(f"""
            WITH counts AS (
                SELECT profile, advertiser_network, COUNT(*) AS n_ads
                FROM ads_enriched
                WHERE {_confidence_clause(min_confidence)}
                  AND advertiser_network IS NOT NULL
                  AND advertiser_network != 'unknown'
                GROUP BY profile, advertiser_network
            ),
            totals AS (
                SELECT profile, SUM(n_ads) AS profile_total
                FROM counts GROUP BY profile
            )
            SELECT 
                c.profile,
                c.advertiser_network,
                c.n_ads,
                ROUND(100.0 * c.n_ads / t.profile_total, 2) AS pct
            FROM counts c
            JOIN totals t USING (profile)
            ORDER BY c.profile, c.n_ads DESC
        """).df()


def top_advertiser_networks(top_n: int = 10, 
                            min_confidence: str = 'high') -> pd.DataFrame:
    """Wide-format pivot: rows=networks, cols=profiles, values=% share."""
    long = network_distribution_by_profile(min_confidence=min_confidence)
    totals = long.groupby('advertiser_network')['n_ads'].sum()
    top_networks = totals.nlargest(top_n).index.tolist()

    pivot = (long[long['advertiser_network'].isin(top_networks)]
             .pivot(index='advertiser_network', columns='profile', values='pct')
             .reindex(columns=PROFILES)
             .reindex(index=top_networks)
             .fillna(0))
    return pivot


# ─────────────────────────────────────────────────────────────────────
# 2. CONTENT ANALYSIS (VLM-powered)
# ─────────────────────────────────────────────────────────────────────
def category_distribution_by_profile(min_confidence: str = 'high',
                                     top_n: int = 20) -> pd.DataFrame:
    """VLM-identified ad category breakdown per profile."""
    with db_session(read_only=True) as con:
        return con.execute(f"""
            WITH cat_counts AS (
                SELECT 
                    profile,
                    LOWER(TRIM(category)) AS category,
                    COUNT(*) AS n_ads
                FROM ads_enriched
                WHERE {_confidence_clause(min_confidence)}
                  AND category IS NOT NULL
                  AND TRIM(category) != ''
                GROUP BY profile, LOWER(TRIM(category))
            ),
            profile_totals AS (
                SELECT profile, SUM(n_ads) AS total 
                FROM cat_counts GROUP BY profile
            )
            SELECT 
                c.profile,
                c.category,
                c.n_ads,
                ROUND(100.0 * c.n_ads / p.total, 2) AS pct_of_profile
            FROM cat_counts c
            JOIN profile_totals p USING (profile)
            ORDER BY c.profile, c.n_ads DESC
            LIMIT {top_n * len(PROFILES)}
        """).df()


def top_brands_by_profile(top_n: int = 15,
                          min_confidence: str = 'high') -> pd.DataFrame:
    """Most-frequent brands per profile. Includes the modal confidence
    level (rather than an average) since confidence is categorical.
    """
    with db_session(read_only=True) as con:
        return con.execute(f"""
            SELECT 
                profile,
                LOWER(TRIM(brand)) AS brand,
                COUNT(*) AS n_ads,
                COUNT(DISTINCT advertiser_network) AS n_networks,
                MODE() WITHIN GROUP (ORDER BY confidence) AS modal_confidence
            FROM ads_enriched
            WHERE {_confidence_clause(min_confidence)}
              AND brand IS NOT NULL
              AND TRIM(brand) != ''
            GROUP BY profile, LOWER(TRIM(brand))
            HAVING n_ads >= 2
            ORDER BY profile, n_ads DESC
        """).df()


def top_products_by_profile(top_n: int = 15,
                            min_confidence: str = 'high') -> pd.DataFrame:
    """Product-level granularity per profile."""
    with db_session(read_only=True) as con:
        return con.execute(f"""
            SELECT 
                profile,
                LOWER(TRIM(product)) AS product,
                COUNT(*) AS n_ads
            FROM ads_enriched
            WHERE {_confidence_clause(min_confidence)}
              AND product IS NOT NULL
              AND TRIM(product) != ''
            GROUP BY profile, LOWER(TRIM(product))
            HAVING n_ads >= 2
            ORDER BY profile, n_ads DESC
        """).df()


def category_matrix(min_confidence: str = 'high',
                    top_n: int = 15) -> pd.DataFrame:
    """Wide pivot: rows=categories, cols=profiles, values=% share."""
    long = category_distribution_by_profile(min_confidence=min_confidence,
                                            top_n=top_n * 5)
    totals = long.groupby('category')['n_ads'].sum()
    top_cats = totals.nlargest(top_n).index.tolist()

    pivot = (long[long['category'].isin(top_cats)]
             .pivot(index='category', columns='profile', values='pct_of_profile')
             .reindex(columns=PROFILES)
             .reindex(index=top_cats)
             .fillna(0))
    return pivot


def _resolve_profile_pair(profile_a: str | None = None,
                          profile_b: str | None = None) -> tuple[str, str]:
    """Resolve a profile comparison pair from config defaults.

    If no explicit pair is provided, use the first two configured
    profiles so the analysis stays aligned with config.PROFILES.
    """
    if profile_a is None and profile_b is None:
        if len(PROFILES) < 2:
            raise ValueError("Need at least two profiles in config.PROFILES.")
        profile_a, profile_b = PROFILES[1], PROFILES[0]
    elif profile_a is None or profile_b is None:
        raise ValueError("Provide both profile_a and profile_b, or neither.")

    if profile_a not in PROFILES or profile_b not in PROFILES:
        raise ValueError(
            f"Unknown profile comparison: {profile_a!r} vs {profile_b!r}. "
            f"Configured profiles: {PROFILES}"
        )

    return profile_a, profile_b


def targeting_delta(min_confidence: str = 'high',
                    top_n: int = 15,
                    profile_a: str | None = None,
                    profile_b: str | None = None) -> pd.DataFrame:
    """Percentage-point delta per category for any two configured profiles."""
    profile_a, profile_b = _resolve_profile_pair(profile_a, profile_b)
    matrix = category_matrix(min_confidence=min_confidence, top_n=top_n)

    if profile_a not in matrix.columns or profile_b not in matrix.columns:
        raise ValueError(
            f"Comparison requires profiles present in the category matrix: "
            f"{profile_a!r}, {profile_b!r}"
        )

    delta = (matrix[profile_a] - matrix[profile_b]).sort_values(ascending=False)
    return delta.to_frame(name=f'{profile_a}_minus_{profile_b}_pct')


# ─────────────────────────────────────────────────────────────────────
# 3. LOCATION ANALYSIS
# ─────────────────────────────────────────────────────────────────────
def ad_placement_stats(min_confidence: str = 'high') -> pd.DataFrame:
    """Per-profile ad placement statistics."""
    with db_session(read_only=True) as con:
        return con.execute(f"""
            SELECT 
                profile,
                COUNT(*) AS n_ads,
                ROUND(AVG(ad_x), 1) AS avg_x,
                ROUND(AVG(ad_y), 1) AS avg_y,
                ROUND(AVG(ad_width), 1) AS avg_width,
                ROUND(AVG(ad_height), 1) AS avg_height,
                ROUND(AVG(ad_width * ad_height), 1) AS avg_area,
                SUM(CASE WHEN ad_y < 600 THEN 1 ELSE 0 END) AS above_fold_count,
                ROUND(100.0 * SUM(CASE WHEN ad_y < 600 THEN 1 ELSE 0 END) 
                      / COUNT(*), 2) AS pct_above_fold
            FROM ads_enriched
            WHERE {_confidence_clause(min_confidence)}
            GROUP BY profile
            ORDER BY profile
        """).df()


def iab_size_classification(min_confidence: str = 'high') -> pd.DataFrame:
    """Standard IAB ad size distribution per profile."""
    with db_session(read_only=True) as con:
        return con.execute(f"""
            SELECT
                profile,
                CASE
                    WHEN ad_width = 728 AND ad_height = 90    THEN '728x90 leaderboard'
                    WHEN ad_width = 300 AND ad_height = 250   THEN '300x250 medium rect'
                    WHEN ad_width = 160 AND ad_height = 600   THEN '160x600 skyscraper'
                    WHEN ad_width = 300 AND ad_height = 600   THEN '300x600 half-page'
                    WHEN ad_width = 970 AND ad_height = 250   THEN '970x250 billboard'
                    WHEN ad_width = 320 AND ad_height = 50    THEN '320x50 mobile banner'
                    WHEN ad_width * ad_height < 10000         THEN 'tiny (<100x100)'
                    WHEN ad_width * ad_height > 250000        THEN 'large (>500x500)'
                    ELSE 'non-standard'
                END AS iab_size,
                COUNT(*) AS n_ads
            FROM ads_enriched
            WHERE {_confidence_clause(min_confidence)}
              AND ad_width > 0 AND ad_height > 0
            GROUP BY profile, iab_size
            ORDER BY profile, n_ads DESC
        """).df()


# ─────────────────────────────────────────────────────────────────────
# 4. TRACKING NETWORK ANALYSIS
# ─────────────────────────────────────────────────────────────────────
def tracking_intensity_by_category(min_confidence: str = 'high') -> pd.DataFrame:
    """Tracker count correlated with VLM ad category."""
    with db_session(read_only=True) as con:
        return con.execute(f"""
            WITH tracker_counts AS (
                SELECT visit_id, COUNT(DISTINCT url) AS n_trackers
                FROM http_requests
                WHERE is_tracker = true
                GROUP BY visit_id
            )
            SELECT 
                ae.profile,
                LOWER(TRIM(ae.category)) AS category,
                COUNT(DISTINCT ae.ad_hash) AS n_ads,
                ROUND(AVG(tc.n_trackers), 1) AS avg_trackers_on_page,
                ROUND(STDDEV(tc.n_trackers), 1) AS stddev_trackers
            FROM ads_enriched ae
            LEFT JOIN tracker_counts tc USING (visit_id)
            WHERE {_confidence_clause(min_confidence, table_alias='ae')}
              AND ae.category IS NOT NULL
            GROUP BY ae.profile, LOWER(TRIM(ae.category))
            HAVING n_ads >= 3
            ORDER BY avg_trackers_on_page DESC
        """).df()


def tracker_network_cooccurrence(min_confidence: str = 'high',
                                 top_n: int = 20) -> pd.DataFrame:
    """Which tracker domains co-occur with which ad networks?"""
    with db_session(read_only=True) as con:
        return con.execute(f"""
            SELECT 
                ae.advertiser_network,
                REGEXP_EXTRACT(hr.url, '://([^/]+)', 1) AS tracker_domain,
                COUNT(DISTINCT ae.ad_hash) AS shared_ads,
                COUNT(DISTINCT ae.profile) AS n_profiles
            FROM ads_enriched ae
            JOIN http_requests hr USING (visit_id)
            WHERE {_confidence_clause(min_confidence, table_alias='ae')}
              AND hr.is_tracker = true
              AND ae.advertiser_network IS NOT NULL
              AND ae.advertiser_network != 'unknown'
            GROUP BY ae.advertiser_network, tracker_domain
            HAVING shared_ads >= 5
            ORDER BY shared_ads DESC
            LIMIT {top_n}
        """).df()


def network_category_tracking_matrix(min_confidence: str = 'high') -> pd.DataFrame:
    """Three-way analysis: (profile × network × category × avg trackers)."""
    with db_session(read_only=True) as con:
        return con.execute(f"""
            WITH tracker_counts AS (
                SELECT visit_id, COUNT(DISTINCT url) AS n_trackers
                FROM http_requests
                WHERE is_tracker = true
                GROUP BY visit_id
            )
            SELECT 
                ae.profile,
                ae.advertiser_network,
                LOWER(TRIM(ae.category)) AS category,
                COUNT(DISTINCT ae.ad_hash) AS n_ads,
                ROUND(AVG(tc.n_trackers), 1) AS avg_trackers
            FROM ads_enriched ae
            LEFT JOIN tracker_counts tc USING (visit_id)
            WHERE {_confidence_clause(min_confidence, table_alias='ae')}
              AND ae.advertiser_network IS NOT NULL
              AND ae.category IS NOT NULL
            GROUP BY ae.profile, ae.advertiser_network, LOWER(TRIM(ae.category))
            HAVING n_ads >= 2
            ORDER BY avg_trackers DESC
        """).df()


# ─────────────────────────────────────────────────────────────────────
# CLI ENTRYPOINT
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    profile_a, profile_b = _resolve_profile_pair()

    print("=" * 70)
    print("VLM COVERAGE REPORT")
    print("=" * 70)
    print(vlm_coverage_report().to_string(index=False))

    print("\n" + "=" * 70)
    print("AD COUNTS BY PROFILE (high-confidence VLM only)")
    print("=" * 70)
    print(ad_counts_by_profile(min_confidence='high').to_string(index=False))

    print("\n" + "=" * 70)
    print("TOP ADVERTISER NETWORKS (% share per profile)")
    print("=" * 70)
    print(top_advertiser_networks(top_n=10).round(1).to_string())

    print("\n" + "=" * 70)
    print("TOP VLM CATEGORIES BY PROFILE")
    print("=" * 70)
    print(category_matrix(top_n=15).round(1).to_string())

    print("\n" + "=" * 70)
    print(f"TARGETING DELTA ({profile_a} − {profile_b})")
    print("=" * 70)
    print(targeting_delta(top_n=15,
                         profile_a=profile_a,
                         profile_b=profile_b).round(2).to_string())

    print("\n" + "=" * 70)
    print("AD PLACEMENT STATS")
    print("=" * 70)
    print(ad_placement_stats().to_string(index=False))

    # print("\n" + "=" * 70)
    # print("TRACKING INTENSITY BY VLM CATEGORY")
    # print("=" * 70)
    # print(tracking_intensity_by_category().head(20).to_string(index=False))