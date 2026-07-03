"""
src/analysis/ads.py — Quantitative analysis of captured ad content.

While trackers and cookies tell us HOW users are profiled, ads tell us
what that profiling produces. This module answers questions like:
  - Do seeded profiles see more ads overall?
  - Which advertiser networks dominate each profile?
  - Are ads from certain networks (e.g., Criteo retargeting) more
    prevalent in seeded vs. control profiles?
  - Is ad TEXT content meaningfully different across profiles
    (a precursor to topic modeling in topic_modeling.py)?

All functions return pandas DataFrames composable with the statistical
test infrastructure in src/analysis/statistics.py.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
from src.utils.db import db_session

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from config import PROFILES
from src.utils.db import db_session


# Keyword lexicon — tune as you discover new patterns in your data
CATEGORY_KEYWORDS = {
    'Retail':  ['shop', 'sale', 'buy', 'discount', 'shipping', 'cart', 
                '% off', 'deal', 'order', 'checkout', 'free shipping'],
    'Finance': ['credit', 'loan', 'invest', 'bank', 'insurance', 
                'crypto', 'mortgage', 'refinance', 'apr'],
    'Health':  ['doctor', 'health', 'medication', 'symptom', 'treatment', 
                'wellness', 'therapy', 'prescription'],
    'Travel':  ['flight', 'hotel', 'vacation', 'trip', 'resort', 
                'book now', 'cruise', 'airline'],
    'Tech':    ['software', 'app', 'cloud', 'ai', 'data', 'platform', 
                'download', 'install'],
}

# ─────────────────────────────────────────────────────────────────────
# AD VOLUME — the most basic comparison
# ─────────────────────────────────────────────────────────────────────
def ad_counts_by_profile(
    min_confidence: Optional[str] = None,
) -> pd.DataFrame:
    """Per-profile ad counts and density.

    Args:
        min_confidence: Filter to 'high' confidence ads only, or None
            for all. For paper-quality numbers, use 'high'. For
            exploratory analysis, use None to see the full picture.

    Returns: profile, n_ads, n_visits_with_ads, n_total_visits,
             ads_per_visit, pct_visits_with_ads.
    """
    confidence_filter = ""
    if min_confidence == 'high':
        confidence_filter = "AND a.confidence = 'high'"
    elif min_confidence == 'medium':
        confidence_filter = "AND a.confidence IN ('high', 'medium')"

    with db_session(read_only=True) as con:
        df = con.execute(f"""
            WITH ad_stats AS (
                SELECT
                    profile,
                    COUNT(*)                       AS n_ads,
                    COUNT(DISTINCT visit_id)       AS n_visits_with_ads
                FROM ads a
                WHERE 1=1 {confidence_filter}
                GROUP BY profile
            ),
            visit_totals AS (
                SELECT profile, COUNT(*) AS n_total_visits
                FROM site_visits
                GROUP BY profile
            )
            SELECT
                v.profile,
                COALESCE(a.n_ads, 0)               AS n_ads,
                COALESCE(a.n_visits_with_ads, 0)   AS n_visits_with_ads,
                v.n_total_visits,
                ROUND(COALESCE(a.n_ads, 0) * 1.0 /
                      NULLIF(v.n_total_visits, 0), 2)  AS ads_per_visit,
                ROUND(COALESCE(a.n_visits_with_ads, 0) * 100.0 /
                      NULLIF(v.n_total_visits, 0), 1)  AS pct_visits_with_ads
            FROM visit_totals v
            LEFT JOIN ad_stats a USING (profile)
            ORDER BY v.profile
        """).df()
    return df


# ─────────────────────────────────────────────────────────────────────
# ADVERTISER NETWORK DISTRIBUTION
# ─────────────────────────────────────────────────────────────────────
def network_distribution_by_profile(
    min_confidence: Optional[str] = 'high',
) -> pd.DataFrame:
    """Distribution of advertiser networks per profile.

    The "who's serving the ads" question. Differences here are
    direct evidence of behavioral targeting: if 'shopping' sees
    35% Criteo ads vs. 'control' at 5%, retargeting is working.

    Returns long-format: profile, advertiser_network, n_ads, pct.
    """
    confidence_filter = ""
    if min_confidence == 'high':
        confidence_filter = "AND confidence = 'high'"

    with db_session(read_only=True) as con:
        df = con.execute(f"""
            WITH counts AS (
                SELECT
                    profile,
                    advertiser_network,
                    COUNT(*) AS n_ads
                FROM ads
                WHERE 1=1 {confidence_filter}
                GROUP BY profile, advertiser_network
            ),
            totals AS (
                SELECT profile, SUM(n_ads) AS total
                FROM counts
                GROUP BY profile
            )
            SELECT
                c.profile,
                c.advertiser_network,
                c.n_ads,
                ROUND(c.n_ads * 100.0 / NULLIF(t.total, 0), 2) AS pct
            FROM counts c
            JOIN totals t USING (profile)
            ORDER BY c.profile, c.n_ads DESC
        """).df()
    return df


def top_advertiser_networks(
    top_n: int = 10,
    min_confidence: Optional[str] = 'high',
) -> pd.DataFrame:
    """Top-N advertiser networks across all profiles, wide format.

    Returns a DataFrame with one row per network and one column
    per profile (% of that profile's ads served by this network).
    Ideal for heatmap visualization.
    """
    long = network_distribution_by_profile(min_confidence=min_confidence)
    # Top networks by total volume across all profiles
    totals = long.groupby('advertiser_network')['n_ads'].sum()
    top_networks = totals.nlargest(top_n).index.tolist()

    pivot = (long[long['advertiser_network'].isin(top_networks)]
             .pivot(index='advertiser_network',
                    columns='profile',
                    values='pct')
             .reindex(columns=PROFILES)
             .reindex(index=top_networks)
             .fillna(0))
    return pivot


# ─────────────────────────────────────────────────────────────────────
# AD SIZE & POSITION — secondary signals
# ─────────────────────────────────────────────────────────────────────
def ad_size_distribution() -> pd.DataFrame:
    """Distribution of ad dimensions per profile.

    Standard IAB ad sizes (728×90 leaderboard, 300×250 medium
    rectangle, 160×600 wide skyscraper, etc.) have different
    auction values. A profile receiving more "premium" sizes is
    being valued more highly by the ad ecosystem — a subtle but
    measurable form of profiling intensity.

    Returns: profile, size_bucket, n_ads.
    """
    with db_session(read_only=True) as con:
        return con.execute("""
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
                END AS size_bucket,
                COUNT(*) AS n_ads
            FROM ads
            WHERE confidence = 'high'
            GROUP BY profile, size_bucket
            ORDER BY profile, n_ads DESC
        """).df()


# ─────────────────────────────────────────────────────────────────────
# OCR TEXT ANALYSIS
# ─────────────────────────────────────────────────────────────────────
def ocr_text_stats() -> pd.DataFrame:
    """Summary statistics on OCR text extracted from ad screenshots.

    Per-profile: how many ads had extractable text, average text
    length, and whether profiles differ in textual ad density.

    A profile receiving more text-heavy ads (vs. pure-image ads)
    suggests different ad categories — text-heavy is often
    direct-response/retail; image-heavy is often brand advertising.
    """
    with db_session(read_only=True) as con:
        return con.execute("""
            SELECT
                profile,
                COUNT(*)                            AS n_ads_total,
                SUM(CASE WHEN ocr_char_count > 0
                    THEN 1 ELSE 0 END)              AS n_ads_with_text,
                ROUND(AVG(ocr_char_count), 1)       AS avg_ocr_chars,
                ROUND(stddev(ocr_char_count), 1)    AS stddev_ocr_chars,
                MAX(ocr_char_count)                 AS max_ocr_chars,
                ROUND(100.0 *
                    SUM(CASE WHEN ocr_char_count > 0
                        THEN 1 ELSE 0 END) /
                    NULLIF(COUNT(*), 0), 1)         AS pct_with_text
            FROM ads
            WHERE confidence = 'high'
            GROUP BY profile
            ORDER BY profile
        """).df()


# ─────────────────────────────────────────────────────────────────────
# KEYWORD SEARCH IN OCR TEXT
# ─────────────────────────────────────────────────────────────────────
# Hand-curated keyword categories. Used for quick smoke-testing of
# whether seeding produces detectable content shifts BEFORE running
# the full topic-modeling pipeline.
KEYWORD_CATEGORIES = {
    'retail':   ['shop', 'sale', 'buy now', 'discount', 'free shipping',
                 'cart', 'order', '% off', 'deal'],
    'finance':  ['credit', 'loan', 'mortgage', 'investment', 'bank',
                 'insurance', 'apr', 'savings', 'crypto'],
    'health':   ['doctor', 'medication', 'health', 'symptoms', 'treatment',
                 'prescription', 'pharmacy', 'wellness', 'therapy'],
    'travel':   ['flight', 'hotel', 'vacation', 'book now', 'trip',
                 'destination', 'cruise', 'resort'],
    'auto':     ['car', 'vehicle', 'lease', 'dealer', 'suv', 'truck',
                 'mpg', 'financing'],
    'tech':     ['software', 'app', 'download', 'subscribe', 'cloud',
                 'ai', 'free trial', 'platform'],
    'media':    ['stream', 'watch now', 'episode', 'season', 'movie',
                 'series', 'podcast', 'channel'],
    'food':     ['recipe', 'restaurant', 'delivery', 'meal', 'order food',
                 'menu', 'dining'],
}


def keyword_category_counts() -> pd.DataFrame:
    """Count ads matching each keyword category, per profile.

    A first-pass content analysis: for each (profile, category) pair,
    count how many ads contain ANY keyword from the category in their
    OCR text. Cheap, interpretable, and useful as a sanity check
    before investing in topic modeling.

    Returns long-format: profile, category, n_matching_ads, pct_of_ads.

    Note: An ad can match multiple categories — these are not mutually
    exclusive. The percentages will sum to >100% for that reason,
    which is expected and worth documenting in your methodology.
    """
    # Build a CASE expression per category. SQL has no native
    # "contains any of these substrings" operator, so we OR-chain
    # ILIKE patterns. ILIKE is case-insensitive in DuckDB.
    category_cases = []
    for category, keywords in KEYWORD_CATEGORIES.items():
        clauses = " OR ".join(
            f"ocr_text ILIKE '%{kw}%'" for kw in keywords
        )
        category_cases.append(
            f"SUM(CASE WHEN ({clauses}) THEN 1 ELSE 0 END) AS n_{category}"
        )

    case_sql = ",\n                ".join(category_cases)

    with db_session(read_only=True) as con:
        wide = con.execute(f"""
            SELECT
                profile,
                COUNT(*) AS n_ads_with_text,
                {case_sql}
            FROM ads
            WHERE confidence = 'high' AND ocr_char_count > 0
            GROUP BY profile
            ORDER BY profile
        """).df()

    # Melt to long format for easier plotting/analysis.
    category_cols = [f"n_{c}" for c in KEYWORD_CATEGORIES]
    long = wide.melt(
        id_vars=['profile', 'n_ads_with_text'],
        value_vars=category_cols,
        var_name='category',
        value_name='n_matching_ads',
    )
    long['category'] = long['category'].str.replace('n_', '', regex=False)
    long['pct_of_ads'] = (
        long['n_matching_ads'] * 100.0 / long['n_ads_with_text']
    ).round(2)
    return long[['profile', 'category', 'n_matching_ads', 'pct_of_ads']]


def differential_keyword_categories(
    profile_a: str,
    profile_b: str = 'control',
) -> pd.DataFrame:
    """Categories where profile_a's ads differ most from profile_b's.

    Returns a sorted DataFrame: category, pct_a, pct_b, pct_delta,
    pct_lift. The single most useful "is targeting working?" table
    you'll generate before topic modeling.
    """
    long = keyword_category_counts()
    pivot = long.pivot(
        index='category', columns='profile', values='pct_of_ads'
    ).fillna(0)

    if profile_a not in pivot.columns or profile_b not in pivot.columns:
        raise ValueError(
            f"Need both '{profile_a}' and '{profile_b}' in data"
        )

    df = pd.DataFrame({
        'category': pivot.index,
        'pct_a':    pivot[profile_a].values,
        'pct_b':    pivot[profile_b].values,
    })
    df['pct_delta'] = df['pct_a'] - df['pct_b']
    df['pct_lift']  = (df['pct_a'] + 0.1) / (df['pct_b'] + 0.1)
    return df.sort_values('pct_lift', ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────
# AD ↔ TRACKING CORRELATION
# ─────────────────────────────────────────────────────────────────────
def ads_per_visit_with_tracking() -> pd.DataFrame:
    """Joins ad counts and tracker counts at the visit level.

    Output suitable for scatter plots ("does heavier tracking lead to
    more ads?") and correlation analysis. One row per (profile, visit).

    Returns: profile, visit_id, n_trackers, n_ads, n_cookies.
    """
    with db_session(read_only=True) as con:
        return con.execute("""
            WITH tracker_counts AS (
                SELECT profile, visit_id,
                       COUNT(DISTINCT regexp_extract(url, '://([^/]+)', 1))
                           AS n_trackers
                FROM http_requests
                WHERE url LIKE 'http%'
                GROUP BY profile, visit_id
            ),
            ad_counts AS (
                SELECT profile, visit_id, COUNT(*) AS n_ads
                FROM ads
                WHERE confidence = 'high'
                GROUP BY profile, visit_id
            ),
            cookie_counts AS (
                SELECT profile, visit_id, COUNT(*) AS n_cookies
                FROM javascript_cookies
                GROUP BY profile, visit_id
            )
            SELECT
                v.profile,
                v.visit_id,
                COALESCE(t.n_trackers, 0) AS n_trackers,
                COALESCE(a.n_ads, 0)      AS n_ads,
                COALESCE(c.n_cookies, 0)  AS n_cookies
            FROM site_visits v
            LEFT JOIN tracker_counts t USING (profile, visit_id)
            LEFT JOIN ad_counts a      USING (profile, visit_id)
            LEFT JOIN cookie_counts c  USING (profile, visit_id)
            ORDER BY profile, visit_id
        """).df()
    


def _build_regex_case(col: str = 'lower(ocr_text)') -> str:
    """Build a SQL CASE expression from CATEGORY_KEYWORDS."""
    clauses = []
    for cat, kws in CATEGORY_KEYWORDS.items():
        pattern = '|'.join(kws)
        clauses.append(f"WHEN regexp_matches({col}, '{pattern}') THEN '{cat}'")
    return "CASE\n  " + "\n  ".join(clauses) + "\n  ELSE 'Unknown'\nEND"


def categorize_ads_by_keywords(min_ocr_chars: int = 10,
                               min_confidence: str = 'high') -> pd.DataFrame:
    """
    Returns one row per ad with columns:
      profile, advertiser_network, ad_category, ocr_text, page_url
    Ads with insufficient OCR text are tagged 'Unknown'.
    """
    case_sql = _build_regex_case()
    with db_session(read_only=True) as con:
        return con.execute(f"""
            SELECT
                profile,
                advertiser_network,
                page_url,
                ocr_text,
                ocr_char_count,
                CASE 
                    WHEN ocr_char_count < {min_ocr_chars} THEN 'Unknown'
                    ELSE {case_sql}
                END AS ad_category
            FROM ads
            WHERE confidence = '{min_confidence}'
              AND advertiser_network IS NOT NULL
              AND advertiser_network != 'unknown'
        """).df()


def network_category_matrix(cat_df: pd.DataFrame,
                            profile: str,
                            top_networks: list,
                            normalize: str = 'row') -> pd.DataFrame:
    """
    Build a network × category matrix for one profile.
    normalize: 'row' = % of each network's ads in each category
               'col' = % of each category supplied by each network
               None  = raw counts
    """
    sub = cat_df[(cat_df['profile'] == profile) &
                 (cat_df['advertiser_network'].isin(top_networks)) &
                 (cat_df['ad_category'] != 'Unknown')]
    mat = pd.crosstab(sub['advertiser_network'], sub['ad_category'])
    # Ensure all top networks appear even if 0 ads
    mat = mat.reindex(top_networks, fill_value=0)
    if normalize == 'row':
        mat = mat.div(mat.sum(axis=1).replace(0, np.nan), axis=0) * 100
    elif normalize == 'col':
        mat = mat.div(mat.sum(axis=0).replace(0, np.nan), axis=1) * 100
    return mat.fillna(0)


def network_category_differential(cat_df: pd.DataFrame,
                                  top_networks: list) -> pd.DataFrame:
    """
    For each (network, category): shopping_share% − control_share%.
    Positive values = network served MORE of that category to the shopping profile.
    This is the 'smoking gun' matrix for behavioral retargeting.
    """
    shopping = network_category_matrix(cat_df, 'shopping', top_networks, 'row')
    control = network_category_matrix(cat_df, 'control', top_networks, 'row')
    # Align indices/columns
    all_cats = sorted(set(shopping.columns) | set(control.columns))
    shopping = shopping.reindex(columns=all_cats, fill_value=0)
    control = control.reindex(columns=all_cats, fill_value=0)
    return shopping - control


if __name__ == "__main__":
    print("Ad counts by profile (high-confidence only):")
    print(ad_counts_by_profile(min_confidence='high').to_string(index=False))

    print("\nTop advertiser networks (% of each profile's ads):")
    print(top_advertiser_networks(top_n=10).round(1).to_string())

    print("\nOCR text statistics:")
    print(ocr_text_stats().to_string(index=False))

    print("\nKeyword category counts:")
    print(keyword_category_counts().to_string(index=False))

    for profile in PROFILES:
        if profile == 'control':
            continue
        print(f"\nDifferential categories: {profile} vs control")
        print(differential_keyword_categories(profile, 'control')
              .to_string(index=False))