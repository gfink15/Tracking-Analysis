"""
src/analysis/cookies.py — Cookie behavior comparisons across profiles.

Cookies are where behavioral targeting becomes visible. A "shopping"
profile that's been seeded with e-commerce history should accumulate:
  • More third-party cookies overall
  • More long-lived cookies (retargeting requires persistence)
  • More cookie-sync events (trackers exchanging IDs)
  • Cookies from retargeting-specific networks (Criteo, AdRoll, etc.)

This module provides functions that quantify each of these.
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


# Retargeting/behavioral-ad cookie hosts. These networks specifically
# build profiles from browsing history. Presence of their cookies in
# a seeded profile is strong evidence of behavioral profiling.
RETARGETING_HOSTS = (
    'criteo.com', 'criteo.net',
    'adroll.com', 'adrolls.com',
    'rlcdn.com',            # Rakuten retargeting
    'doubleclick.net',      # Google retargeting
    'taboola.com', 'outbrain.com',
    'adnxs.com',            # AppNexus (now Xandr)
    'casalemedia.com',
    'rubiconproject.com',
    'pubmatic.com',
    'bidswitch.net',
)


# ─────────────────────────────────────────────────────────────────────
# VOLUME METRICS
# ─────────────────────────────────────────────────────────────────────
def cookie_counts_by_profile() -> pd.DataFrame:
    """Total cookies set per profile, broken down by first/third party.

    First-party vs third-party is determined by comparing the cookie's
    host to the visited site's host. A cookie set by `nytimes.com`
    during a visit to `nytimes.com` is first-party; a cookie set by
    `doubleclick.net` during the same visit is third-party.

    Returns DataFrame: profile, n_total, n_first_party, n_third_party,
                       n_unique_hosts, pct_third_party.
    """
    with db_session(read_only=True) as con:
        df = con.execute("""
            WITH joined AS (
                SELECT
                    c.profile,
                    c.host                                       AS cookie_host,
                    regexp_extract(v.site_url, '://([^/]+)', 1)  AS site_host,
                    c.name,
                    c.expiry
                FROM javascript_cookies c
                JOIN site_visits v USING (profile, visit_id)
            )
            SELECT
                profile,
                COUNT(*)                                          AS n_total,
                SUM(CASE WHEN cookie_host = site_host
                         OR cookie_host LIKE '%.' || site_host
                         THEN 1 ELSE 0 END)                       AS n_first_party,
                SUM(CASE WHEN cookie_host != site_host
                         AND cookie_host NOT LIKE '%.' || site_host
                         THEN 1 ELSE 0 END)                       AS n_third_party,
                COUNT(DISTINCT cookie_host)                       AS n_unique_hosts,
                ROUND(100.0 * SUM(CASE WHEN cookie_host != site_host
                                       AND cookie_host NOT LIKE '%.' || site_host
                                       THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0),
                      2)                                          AS pct_third_party
            FROM joined
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
    fp_filter = ""
    if third_party_only:
        fp_filter = """
            AND cookie_host != site_host
            AND cookie_host NOT LIKE '%.' || site_host
        """

    with db_session(read_only=True) as con:
        df = con.execute(f"""
            WITH cookies_with_site AS (
                SELECT
                    c.profile,
                    c.host                                       AS cookie_host,
                    regexp_extract(v.site_url, '://([^/]+)', 1)  AS site_host,
                    c.expiry,
                    -- expiry is epoch seconds; visits are in same units.
                    -- Lifespan in days = (expiry - time_stamp) / 86400.
                    -- Session cookies have expiry IS NULL or = 0.
                    CASE
                        WHEN c.expiry IS NULL OR c.expiry = 0 THEN -1
                        ELSE (c.expiry - EXTRACT(EPOCH FROM c.time_stamp)) / 86400.0
                    END AS lifespan_days
                FROM javascript_cookies c
                JOIN site_visits v USING (profile, visit_id)
                WHERE 1=1 {fp_filter}
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
            FROM cookies_with_site
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
    number of cookies set and unique visits affected.

    A profile-vs-control delta here is the cleanest evidence you'll
    get that history seeding triggered behavioral profiling.
    """
    # Build the SQL IN list once. We use LIKE rather than = because
    # retargeters often use multiple subdomains (ads.doubleclick.net,
    # stats.g.doubleclick.net, etc.) and we want to catch all of them.
    where_clauses = " OR ".join(
        f"c.host LIKE '%{h}%'" for h in RETARGETING_HOSTS
    )

    with db_session(read_only=True) as con:
        df = con.execute(f"""
            WITH normalized AS (
                SELECT
                    c.profile,
                    c.visit_id,
                    -- Map any matching subdomain back to the canonical retargeter
                    CASE
                        {' '.join(
                            f"WHEN c.host LIKE '%{h}%' THEN '{h}'"
                            for h in RETARGETING_HOSTS
                        )}
                        ELSE 'other'
                    END AS retargeter
                FROM javascript_cookies c
                WHERE {where_clauses}
            )
            SELECT
                profile,
                retargeter,
                COUNT(*)                       AS n_cookies,
                COUNT(DISTINCT visit_id)       AS n_visits_affected
            FROM normalized
            WHERE retargeter != 'other'
            GROUP BY profile, retargeter
            ORDER BY profile, n_cookies DESC
        """).df()
    return df


# ─────────────────────────────────────────────────────────────────────
# COOKIE SYNCING DETECTION
# ─────────────────────────────────────────────────────────────────────
def detect_cookie_syncs(min_id_length: int = 10) -> pd.DataFrame:
    """Identify probable cookie-sync events: same ID appearing in
    cookies from different hosts within the same visit.

    Cookie syncing is the practice where two trackers exchange their
    user IDs so they can merge their profiles. It's the mechanism
    that lets a behavior on Site A inform an ad on Site B even when
    different tracking companies are involved.

    This function uses a heuristic: long alphanumeric cookie values
    that appear in cookies from ≥2 different hosts during the same
    visit are probable sync events. False positives are possible
    (shared session tokens, common defaults) so always inspect
    results before drawing conclusions.

    Args:
        min_id_length: Minimum cookie-value length to consider an
            ID candidate. Below ~10 chars, false positive rate is
            unacceptable (boolean flags, version numbers, etc.).

    Returns: profile, visit_id, shared_value, n_hosts, hosts.
    """
    with db_session(read_only=True) as con:
        df = con.execute(f"""
            WITH long_values AS (
                SELECT
                    profile,
                    visit_id,
                    host,
                    value
                FROM javascript_cookies
                WHERE LENGTH(value) >= {min_id_length}
                  -- Heuristic: ID-like values are mostly alphanumeric
                  AND regexp_matches(value, '^[a-zA-Z0-9_.-]+$')
            ),
            shared AS (
                SELECT
                    profile,
                    visit_id,
                    value AS shared_value,
                    COUNT(DISTINCT host) AS n_hosts,
                    string_agg(DISTINCT host, ', ') AS hosts
                FROM long_values
                GROUP BY profile, visit_id, value
                HAVING COUNT(DISTINCT host) >= 2
            )
            SELECT *
            FROM shared
            ORDER BY profile, n_hosts DESC
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
            'avg_hosts_per_sync': [0.0] * len(PROFILES),
        })
    return (syncs.groupby('profile')
                 .agg(n_sync_events=('shared_value', 'count'),
                      n_visits_with_syncs=('visit_id', 'nunique'),
                      avg_hosts_per_sync=('n_hosts', 'mean'))
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