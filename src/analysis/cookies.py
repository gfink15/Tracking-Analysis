"""
src/analysis/cookies.py — Cookie behavior comparisons across profiles.

Cookies are where behavioral targeting becomes visible. A seeded
profile should accumulate more third-party cookies, more long-lived
cookies, more cookie-sync events, and more cookies from retargeting
networks such as Criteo or AdRoll.

This module provides functions that quantify each of these.

REFACTOR LOG: Anya Barringer, Summer 2026
    - Source table: javascript_cookies -> javascript_cookies_enriched (per enrich_parquet.py)
    - Domain extraction + entity aggregation: precomputed domain, subsidiary_entity,
        parent_entity columns; most common replacement of host -> parent_entity
    - First-party filter: add "WHERE relationship_tier  IN ('inter-family third-party',
        'external third-party')" to remove first-party cookies and unknown (unresolvable
        domain) cookies (relationship_tier column from classify_relationships())
    - Cookie retargeting: replaced RETARGETING_HOSTS list of domains with
        RETARGETING_PARENTS list of parent entities; kept same domains in previous list
        but now mapped to parent_entity for more comprehensive analysis; also added several
        entities. Note that primary evidence for retargeting is known retargeting host name,
        often supported by the sheer number of cookie events
    - Cookie syncing: now depends on mapped parent_entity instead of simple host;
        chose highest level of granularity as syncing is by definition sharing
        cookies between two different corporate tracker entities
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from config import PROFILES
from src.utils.db import db_session


# Retargeting/behavioral-ad parent entities. These networks specifically
# build profiles from browsing history. Presence of their cookies in
# a seeded profile is strong evidence of behavioral profiling.
RETARGETING_PARENTS = (
    'criteo',               # criteo.com, criteo.net, bidswitch.net
    'nextroll',             # adroll.com, adrolls.com
    'liveramp',             # rlcdn.com
    'google',               # doubleclick.net, googleadservices.com
    'taboola',              # taboola.com
    'teads',                # outbrain.com
    'microsoft',            # adnxs.com
    'indexexchange',        # casalemedia.com
    'magnite',              # rubiconproject.com
    'pubmatic',             # pubmatic.com
    'meta',                 # fbcdn.net, fbsbx.com, facebook.com
    'the trade desk',       # adsrvr.org
    'rtbhouse',             # creativecdn.com, rtbhouse.com
    'quantcast',            # quantserve.com
    'amazon',               # amazon-adsystem.com
    'yahoo',                # yahoodsp.com, advertising.com - CHECK MAPPING W/ VERIZON
)


# ─────────────────────────────────────────────────────────────────────
# VOLUME METRICS
# ─────────────────────────────────────────────────────────────────────
def cookie_counts_by_profile() -> pd.DataFrame:
    """Total cookies set per profile, broken down by relationship tier.

    Classifies cookies by relationship tier based on pre-computed entity
    mappings in javascript_cookies_enriched: first-party (same parent and
    subsidiary), inter-family third-party (same parent, different subsidiary),
    and external third-party (different parent). Counts unique parent entities
    setting cookies per profile.

    Returns DataFrame: profile, n_total, n_first_party,
                       n_inter_family_third_party, n_external_third_party,
                       n_unique_parent_entities, pct_third_party,
                       pct_external_third_party.
    """
    with db_session(read_only=True) as con:
        df = con.execute("""
            SELECT
                profile,
                COUNT(*)                                          AS n_total,
                SUM(CASE WHEN relationship_tier = 'first-party'
                         THEN 1 ELSE 0 END)                       AS n_first_party,
                SUM(CASE WHEN relationship_tier = 'inter-family third-party'
                         THEN 1 ELSE 0 END)                       AS n_inter_family_third_party,
                SUM(CASE WHEN relationship_tier = 'external third-party'
                         THEN 1 ELSE 0 END)                       AS n_external_third_party,
                COUNT(DISTINCT parent_entity)                     AS n_unique_parent_entities,
                ROUND(100.0 * (SUM(CASE WHEN relationship_tier IN ('inter-family third-party', 'external third-party')
                                        THEN 1 ELSE 0 END) /
                              NULLIF(COUNT(*), 0)),
                      2)                                          AS pct_third_party,
                ROUND(100.0 * (SUM(CASE WHEN relationship_tier = 'external third-party'
                                        THEN 1 ELSE 0 END) /
                              NULLIF(COUNT(*), 0)),
                      2)                                          AS pct_external_third_party
            FROM javascript_cookies_enriched
            GROUP BY profile
            ORDER BY profile
        """).df()
    return df


# ─────────────────────────────────────────────────────────────────────
# LIFESPAN ANALYSIS
# ─────────────────────────────────────────────────────────────────────
def cookie_lifespan_distribution(
    third_party_only: bool = True,
) -> pd.DataFrame:
    """Distribution of cookie lifespans per profile.

    Retargeting cookies need to persist long enough to follow a user
    across sessions — typically 30+ days. A profile receiving more
    long-lived cookies is being more aggressively profiled.

    Returns long-format DataFrame: profile, lifespan_bucket, n_cookies.
    Buckets: 'session' (no expiry), '<1d', '1-7d', '7-30d', '30-365d',
             '1y+'.

    Args:
        third_party_only: If True, restrict to third-party cookies
            (where retargeting actually happens).
    """
    tier_filter = ""
    if third_party_only:
        tier_filter = """
            WHERE relationship_tier IN ('inter-family third-party', 'external third-party')
        """
    with db_session(read_only=True) as con:
        df = con.execute(f"""
            WITH cookies_with_lifespan AS (
                SELECT
                    profile,
                    expiry,
                    -- expiry is epoch seconds; visits are in same units.
                    -- Lifespan in days = (expiry - time_stamp) / 86400.
                    -- Session cookies have expiry IS NULL or = 0.
                    CASE
                        WHEN expiry IS NULL OR is_session = 1 THEN -1
                        ELSE (
                            EXTRACT(EPOCH FROM expiry)
                            - EXTRACT(EPOCH FROM time_stamp)
                        ) / 86400.0
                    END AS lifespan_days
                FROM javascript_cookies_enriched
                {tier_filter}
            )
            SELECT
                profile,
                CASE
                    WHEN lifespan_days < 0   THEN 'session'
                    WHEN lifespan_days < 1   THEN '<1d'
                    WHEN lifespan_days < 7   THEN '1-7d'
                    WHEN lifespan_days < 30  THEN '7-30d'
                    WHEN lifespan_days < 365 THEN '30-365d'
                    ELSE                          '1y+'
                END                                              AS lifespan_bucket,
                COUNT(*)                                         AS n_cookies
            FROM cookies_with_lifespan
            GROUP BY profile, lifespan_bucket
            ORDER BY profile,
                CASE lifespan_bucket
                    WHEN 'session' THEN 0 WHEN '<1d'     THEN 1
                    WHEN '1-7d'    THEN 2 WHEN '7-30d'   THEN 3
                    WHEN '30-365d' THEN 4 WHEN '1y+'     THEN 5
                END
        """).df()
    return df


# ─────────────────────────────────────────────────────────────────────
# RETARGETING NETWORK PRESENCE
# ─────────────────────────────────────────────────────────────────────
def retargeting_cookie_presence() -> pd.DataFrame:
    """Count cookies from known retargeting networks, per profile.

    This is one of the most direct measurements of behavioral
    targeting: each row is a (profile, retargeter) pair with the
    number of cookies recorded and unique visits affected.

    A profile-to-baseline delta here is the cleanest evidence you'll
    get that history seeding triggered behavioral profiling.
    """
    # Build the SQL IN list from RETARGETING_PARENTS for the CASE expression.
    # Single-quoted, comma-separated for injection into SQL IN (...).
    parents_sql = ", ".join(f"'{p}'" for p in RETARGETING_PARENTS)

    with db_session(read_only=True) as con:
        df = con.execute(f"""
            WITH normalized AS (
                SELECT
                    profile,
                    visit_id,
                    -- Label cookie with its parent entity if it is 
                    -- known retargeter, otherwise 'other'. Compares 
                    -- direct equality between column and list.
                    CASE
                        WHEN parent_entity IN ({parents_sql}) THEN parent_entity
                        ELSE 'other'
                    END AS retargeter
                FROM javascript_cookies_enriched
                WHERE relationship_tier IN (
                    'inter-family third-party',
                    'external third-party'
                )
            )
            SELECT
                profile,
                retargeter,
                COUNT(*)                       AS n_cookie_events,
                COUNT(DISTINCT visit_id)       AS n_visits_affected
            FROM normalized
            WHERE retargeter != 'other'
            GROUP BY profile, retargeter
            ORDER BY profile, n_cookie_events DESC
        """).df()
    return df


# ─────────────────────────────────────────────────────────────────────
# COOKIE SYNCING DETECTION
# ─────────────────────────────────────────────────────────────────────
def detect_cookie_syncs(min_id_length: int = 10) -> pd.DataFrame:
    """Identify probable cookie-sync events: same ID appearing in
    cookies from different parent entities within the same visit.

    Cookie syncing is the practice where two trackers exchange their
    user IDs so they can merge their profiles. It's the mechanism
    that lets a behavior on Site A inform an ad on Site B even when
    different tracking companies are involved.

    This function uses a heuristic: long alphanumeric cookie values
    that appear in cookies from ≥2 different parent entities during
    the same visit are probable sync events. False positives are
    possible (shared session tokens, common defaults) so always inspect
    results before drawing conclusions.

    Args:
        min_id_length: Minimum cookie-value length to consider an
            ID candidate. Below ~10 chars, false positive rate is
            unacceptable (boolean flags, version numbers, etc.).

    Returns: profile, visit_id, shared_value, n_parents, parents.
    """
    with db_session(read_only=True) as con:
        df = con.execute(f"""
            WITH long_values AS (
                SELECT
                    profile,
                    visit_id,
                    parent_entity,
                    value
                FROM javascript_cookies_enriched
                WHERE LENGTH(value) >= {min_id_length}
                  -- Heuristic: ID-like values are mostly alphanumeric
                  AND regexp_matches(value, '^[a-zA-Z0-9_.-]+$')
                  AND relationship_tier IN (
                      'inter-family third-party',
                      'external third-party'
                  )
            ),
            shared AS (
                SELECT
                    profile,
                    visit_id,
                    value                              AS shared_value,
                    COUNT(DISTINCT parent_entity)      AS n_parents,
                    string_agg(DISTINCT parent_entity, ', ') AS parents
                FROM long_values
                GROUP BY profile, visit_id, value
                HAVING COUNT(DISTINCT parent_entity) >= 2
            )
            SELECT *
            FROM shared
            ORDER BY profile, n_parents DESC
        """).df()
    return df


def cookie_sync_summary() -> pd.DataFrame:
    """Aggregate cookie-sync stats per profile."""
    syncs = detect_cookie_syncs()
    if syncs.empty:
        return pd.DataFrame({
            'profile': PROFILES,
            'n_sync_events': [0] * len(PROFILES),
            'n_visits_with_syncs': [0] * len(PROFILES),
            'avg_parents_per_sync': [0.0] * len(PROFILES),
        })
    return (syncs.groupby('profile')
                 .agg(n_sync_events=('shared_value', 'count'),
                      n_visits_with_syncs=('visit_id', 'nunique'),
                      avg_parents_per_sync=('n_parents', 'mean'))
                 .round(2)
                 .reset_index())


if __name__ == "__main__":
    print("Cookie counts by profile:")
    print(cookie_counts_by_profile().to_string(index=False))

    print("\nCookie lifespan distribution (3rd party):")
    print(cookie_lifespan_distribution().to_string(index=False))

    print("\nRetargeting cookie presence:")
    print(retargeting_cookie_presence().to_string(index=False))

    print("\nCookie sync summary:")
    print(cookie_sync_summary().to_string(index=False))