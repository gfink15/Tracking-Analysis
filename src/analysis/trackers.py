"""
src/analysis/trackers.py — Tracker prevalence and cross-profile comparison.

This is the first true analysis module: it answers research questions
like:
  - How many unique trackers does each profile encounter?
  - Which trackers appear only when history is seeded?
  - How similar are the tracker sets across profiles? (Jaccard)
  - Which trackers are most discriminating between profile pairs?

All functions return pandas DataFrames so they compose cleanly into
notebooks and plotting code. Heavy lifting happens in DuckDB SQL —
we use Python only for the final shaping.

Usage in a notebook:
    from src.analysis.trackers import tracker_prevalence_by_profile
    df = tracker_prevalence_by_profile()
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from config import PROFILES
from src.utils.db import db_session


# ─────────────────────────────────────────────────────────────────────
# HOST EXTRACTION — the foundational primitive
# ─────────────────────────────────────────────────────────────────────
# Many queries need "the hostname from a URL." Rather than repeating
# the regex in every function, we define it as a SQL fragment we can
# inject. DuckDB's regexp_extract is fast and handles the common cases.
#
# Pattern explanation:
#   ://     — match the protocol separator
#   ([^/]+) — capture group: everything up to the next /
#   1       — return capture group 1 (the hostname)
HOSTNAME_SQL = "regexp_extract(url, '://([^/]+)', 1)"

# eTLD+1 extraction (e.g., "ads.doubleclick.net" → "doubleclick.net").
# A real implementation should use the public suffix list, but for
# first-pass analysis this naive "last two labels" approach catches
# most cases. We note this as a TODO for paper-quality results.
ETLD1_SQL = """
    array_to_string(
        list_slice(string_split({host}, '.'), -2, -1),
        '.'
    )
"""


# ─────────────────────────────────────────────────────────────────────
# CORE METRICS
# ─────────────────────────────────────────────────────────────────────
def tracker_prevalence_by_profile(
    use_tracker_list: bool = False,
) -> pd.DataFrame:
    """Total unique third-party hosts (and trackers) contacted per profile.

    Args:
        use_tracker_list: If True, restrict counts to known trackers
            (requires src/utils/tracker_lists.py to be populated).
            If False, count all third-party hosts — useful as a
            sanity check and for the "unknown unknowns" question.

    Returns:
        DataFrame with columns: profile, n_visits, n_unique_hosts,
        n_unique_etld1, hosts_per_visit, etld1_per_visit.
    """
    with db_session(read_only=True) as con:
        df = con.execute(f"""
            WITH per_request AS (
                SELECT
                    profile,
                    visit_id,
                    {HOSTNAME_SQL} AS host
                FROM http_requests
                WHERE url LIKE 'http%'  -- exclude data:, blob:, etc.
            ),
            per_request_with_etld AS (
                SELECT
                    profile,
                    visit_id,
                    host,
                    {ETLD1_SQL.format(host='host')} AS etld1
                FROM per_request
            )
            SELECT
                profile,
                COUNT(DISTINCT visit_id)       AS n_visits,
                COUNT(DISTINCT host)           AS n_unique_hosts,
                COUNT(DISTINCT etld1)          AS n_unique_etld1,
                ROUND(
                    COUNT(DISTINCT host) * 1.0 /
                    NULLIF(COUNT(DISTINCT visit_id), 0), 2
                )                              AS hosts_per_visit,
                ROUND(
                    COUNT(DISTINCT etld1) * 1.0 /
                    NULLIF(COUNT(DISTINCT visit_id), 0), 2
                )                              AS etld1_per_visit
            FROM per_request_with_etld
            GROUP BY profile
            ORDER BY profile
        """).df()
    return df


def tracker_frequency_table(
    top_n: int = 50,
) -> pd.DataFrame:
    """Frequency of each tracker (eTLD+1) appearing per profile.

    Produces a long-format table you can pivot for heatmaps:
        profile | etld1 | n_visits_seen | pct_of_visits

    Args:
        top_n: Return the top-N trackers by total cross-profile
            visit count. Limiting prevents thousands of rare hosts
            from drowning out the signal in visualizations.

    Returns:
        Long-format DataFrame, sortable by total prevalence.
    """
    with db_session(read_only=True) as con:
        df = con.execute(f"""
            WITH per_visit_host AS (
                -- Deduplicate so each (profile, visit, etld1) is one row.
                -- A page making 50 requests to the same tracker should
                -- count as ONE tracker-presence, not 50.
                SELECT DISTINCT
                    profile,
                    visit_id,
                    {ETLD1_SQL.format(host=HOSTNAME_SQL)} AS etld1
                FROM http_requests
                WHERE url LIKE 'http%'
            ),
            per_profile_etld1 AS (
                SELECT
                    profile,
                    etld1,
                    COUNT(*) AS n_visits_seen
                FROM per_visit_host
                GROUP BY profile, etld1
            ),
            profile_visit_totals AS (
                SELECT profile, COUNT(DISTINCT visit_id) AS total_visits
                FROM site_visits
                GROUP BY profile
            ),
            top_overall AS (
                -- Identify the top-N most prevalent trackers across all
                -- profiles, so we return the same set for every profile
                -- (otherwise comparisons get apples-to-oranges).
                SELECT etld1
                FROM per_profile_etld1
                GROUP BY etld1
                ORDER BY SUM(n_visits_seen) DESC
                LIMIT {top_n}
            )
            SELECT
                p.profile,
                p.etld1,
                p.n_visits_seen,
                ROUND(p.n_visits_seen * 100.0 / t.total_visits, 2)
                    AS pct_of_visits
            FROM per_profile_etld1 p
            JOIN profile_visit_totals t USING (profile)
            WHERE p.etld1 IN (SELECT etld1 FROM top_overall)
            ORDER BY p.etld1, p.profile
        """).df()
    return df


def jaccard_similarity_matrix() -> pd.DataFrame:
    """Pairwise Jaccard similarity of tracker sets between profiles.

    Jaccard(A, B) = |A ∩ B| / |A ∪ B|

    Returns a square DataFrame where rows and columns are profiles
    and cells are Jaccard similarity (1.0 = identical tracker sets,
    0.0 = completely disjoint).

    This is the single most useful "at a glance" summary of how
    different the tracking landscapes are across profiles.
    """
    with db_session(read_only=True) as con:
        # Build the set of eTLD+1s per profile.
        sets_df = con.execute(f"""
            SELECT DISTINCT
                profile,
                {ETLD1_SQL.format(host=HOSTNAME_SQL)} AS etld1
            FROM http_requests
            WHERE url LIKE 'http%'
        """).df()

    # Convert to a dict of sets — much cleaner than SQL for this step.
    profile_sets = {
        profile: set(group['etld1'])
        for profile, group in sets_df.groupby('profile')
    }

    # Build the matrix. Python loops are fine here since we have at
    # most ~10 profiles (i.e., ~100 cells).
    # sorted() needs an actual iterable of comparable items; cast keys to list
    profiles = sorted(str(profile) for profile in profile_sets.keys())
    matrix = pd.DataFrame(index=profiles, columns=profiles, dtype=float)
    for a in profiles:
        for b in profiles:
            sa, sb = profile_sets[a], profile_sets[b]
            union = sa | sb
            matrix.loc[a, b] = len(sa & sb) / len(union) if union else 0.0
    return matrix


# ─────────────────────────────────────────────────────────────────────
# DIFFERENTIAL ANALYSIS — the core research question
# ─────────────────────────────────────────────────────────────────────
def differential_trackers(
    profile_a: str,
    profile_b: str = 'control',
    min_visits: int = 3,
) -> pd.DataFrame:
    """Trackers appearing significantly more in profile_a than profile_b.

    This is the central tool for answering "what does history seeding
    actually change?" If a tracker appears 50 times in 'shopping' but
    only 2 times in 'control', it's a strong candidate for behavioral
    targeting evidence.

    Args:
        profile_a: The "treatment" profile (with seeded history).
        profile_b: The "control" profile (default: 'control').
        min_visits: Minimum visits in profile_a for the tracker to
            be included. Filters out one-off appearances that aren't
            statistically meaningful.

    Returns:
        DataFrame sorted by lift (ratio of A frequency to B frequency).
        Columns: etld1, visits_a, visits_b, lift, delta.
    """
    if profile_a not in PROFILES or profile_b not in PROFILES:
        raise ValueError(
            f"Unknown profile(s). Valid: {PROFILES}"
        )

    with db_session(read_only=True) as con:
        df = con.execute(f"""
            WITH per_visit_etld AS (
                SELECT DISTINCT
                    profile,
                    visit_id,
                    {ETLD1_SQL.format(host=HOSTNAME_SQL)} AS etld1
                FROM http_requests
                WHERE url LIKE 'http%'
                  AND profile IN ('{profile_a}', '{profile_b}')
            ),
            counts AS (
                SELECT
                    etld1,
                    SUM(CASE WHEN profile = '{profile_a}' THEN 1 ELSE 0 END)
                        AS visits_a,
                    SUM(CASE WHEN profile = '{profile_b}' THEN 1 ELSE 0 END)
                        AS visits_b
                FROM per_visit_etld
                GROUP BY etld1
            )
            SELECT
                etld1,
                visits_a,
                visits_b,
                visits_a - visits_b AS delta,
                -- Add-one smoothing avoids division by zero and
                -- gives a meaningful lift for "appears in A, absent
                -- in B" cases. This is a standard technique.
                ROUND((visits_a + 1.0) / (visits_b + 1.0), 3) AS lift
            FROM counts
            WHERE visits_a >= {min_visits}
            ORDER BY lift DESC, delta DESC
        """).df()
    return df


def trackers_unique_to_profile(profile: str) -> pd.DataFrame:
    """Trackers that appear in ONLY this profile, in no others.

    The strongest possible evidence of profile-specific tracking:
    these hosts are summoned by something about this profile's
    seeded history that no other profile triggers.

    Returns:
        DataFrame with columns: etld1, n_visits_seen.
        Empty DataFrame if no profile-unique trackers exist.
    """
    other_profiles = [p for p in PROFILES if p != profile]
    if not other_profiles:
        raise ValueError("Need at least 2 profiles for this analysis.")

    # Build the "other profiles" list as a SQL IN clause.
    others_sql = ", ".join(f"'{p}'" for p in other_profiles)

    with db_session(read_only=True) as con:
        df = con.execute(f"""
            WITH etlds_in_target AS (
                SELECT DISTINCT
                    {ETLD1_SQL.format(host=HOSTNAME_SQL)} AS etld1
                FROM http_requests
                WHERE profile = '{profile}' AND url LIKE 'http%'
            ),
            etlds_in_others AS (
                SELECT DISTINCT
                    {ETLD1_SQL.format(host=HOSTNAME_SQL)} AS etld1
                FROM http_requests
                WHERE profile IN ({others_sql}) AND url LIKE 'http%'
            ),
            unique_etlds AS (
                SELECT etld1 FROM etlds_in_target
                EXCEPT
                SELECT etld1 FROM etlds_in_others
            )
            SELECT
                u.etld1,
                COUNT(DISTINCT r.visit_id) AS n_visits_seen
            FROM unique_etlds u
            JOIN http_requests r
                ON {ETLD1_SQL.format(host=HOSTNAME_SQL)} = u.etld1
            WHERE r.profile = '{profile}'
            GROUP BY u.etld1
            ORDER BY n_visits_seen DESC
        """).df()
    return df


if __name__ == "__main__":
    # Smoke test: print summary stats for every metric. Running this
    # file directly gives you a quick "is everything wired up right?"
    # check before opening a notebook.
    print("Tracker prevalence by profile:")
    print(tracker_prevalence_by_profile().to_string(index=False))

    print("\nJaccard similarity matrix:")
    print(jaccard_similarity_matrix().round(3).to_string())

    for profile in PROFILES:
        if profile == 'control':
            continue
        print(f"\nTop 10 differential trackers ({profile} vs control):")
        diff = differential_trackers(profile, 'control').head(10)
        print(diff.to_string(index=False))