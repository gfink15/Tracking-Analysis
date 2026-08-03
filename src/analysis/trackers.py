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

REFACTOR LOG: Anya Barringer, Summer 2026
    - Source table: http_requests -> http_requests_enriched (per enrich_parquet.py)
    - Domain extraction: HOSTNAME_SQL -> pre-computed "domain" column
    - Entity aggregation: ETLD1_SQL -> pre-computed "parent_entity" column
        (subsidiary_column also referenced occasionally)
    - First-party filter: add "WHERE relationship_tier NOT IN ('first_party', 'unknown')
        to remove first-party requests and unknown (unresolvable domain) requests
        (references relationship_tier column from classify_relationships())
    - Tier breakdown: added n_external_third_party, n_inter_family_third_party stats
    - Granularity paramter: added granularity flag with ValueError guard, allowing for
        analysis at multiple levels of domain-entity identity
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from config import PROFILES
from src.utils.db import db_session

from src.utils.domain_utils import load_tree, resolve_node, get_registered_domain


def _resolve_baseline_profile(baseline: str | None = None) -> str:
    """Resolve the baseline profile from config defaults."""
    if baseline is None:
        if not PROFILES:
            raise ValueError("config.PROFILES is empty.")
        baseline = PROFILES[0]
    if baseline not in PROFILES:
        raise ValueError(f"Unknown profile: {baseline!r}. Valid: {PROFILES}")
    return baseline

# Original constants for extracting tracker as etld+1 string from url

# HOSTNAME_SQL = "regexp_extract(url, '://([^/]+)', 1)"
# ETLD1_SQL = """
#     array_to_string(
#         list_slice(string_split({host}, '.'), -2, -1),
#         '.'
#     )
# """


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
        DataFrame with columns: profile, n_visits, n_unique_domains,
        n_unique_subsidiaries, n_unique_parents, n_external_third_party,
        n_inter_family_third_party, domains_per_visit, 
        subsidiaries_per_visit, parents_per_visit.
    """
    with db_session(read_only=True) as con:
        df = con.execute(f"""
            WITH tracker_requests AS (
                SELECT
                    profile,
                    visit_id,
                    domain,
                    subsidiary_entity,
                    parent_entity,
                    relationship_tier
                FROM http_requests_enriched
                WHERE url LIKE 'http%'
                  AND relationship_tier NOT IN ('first-party', 'unknown')
            )
            SELECT
                profile,
                COUNT(DISTINCT visit_id)           AS n_visits,
                COUNT(DISTINCT domain)             AS n_unique_domains,
                COUNT(DISTINCT subsidiary_entity)  AS n_unique_subsidiaries,
                COUNT(DISTINCT parent_entity)      AS n_unique_parents,
                -- Disaggregated entity counts
                COUNT(DISTINCT CASE 
                    WHEN relationship_tier = 'external third-party' 
                    THEN parent_entity END)        AS n_external_third_party,
                COUNT(DISTINCT CASE 
                    WHEN relationship_tier = 'inter-family third-party' 
                    THEN parent_entity END)        AS n_inter_family_third_party,
                -- Averages per visit
                ROUND(
                    COUNT(DISTINCT domain) * 1.0 /
                    NULLIF(COUNT(DISTINCT visit_id), 0), 2
                )                                  AS domains_per_visit,
                ROUND(
                    COUNT(DISTINCT subsidiary_entity) * 1.0 /
                    NULLIF(COUNT(DISTINCT visit_id), 0), 2
                )                                  AS subsidiaries_per_visit,
                ROUND(
                    COUNT(DISTINCT parent_entity) * 1.0 /
                    NULLIF(COUNT(DISTINCT visit_id), 0), 2
                )                                  AS parents_per_visit
            FROM tracker_requests
            GROUP BY profile
            ORDER BY profile
        """).df()
    return df


def tracker_frequency_table(
    top_n: int = 50,
) -> pd.DataFrame:
    """Frequency of each tracker (parent_entity) appearing per profile.
    Uses corporate parent entity for big-picture understanding of the
    wide reach of major tracking entities across websites.

    Produces a long-format table you can pivot for heatmaps:
        profile | parent_entity | n_visits_seen | pct_of_visits

    Args:
        top_n: Return the top-N trackers by total cross-profile
            visit count. Limiting prevents thousands of rare hosts
            from drowning out the signal in visualizations.

    Returns:
        Long-format DataFrame, sortable by total prevalence.
    """
    with db_session(read_only=True) as con:
        df = con.execute(f"""
            WITH per_visit_entity AS (
                -- Deduplicate so each (profile, visit, parent_entity) is one row.
                -- A page making 50 requests to the same tracker should
                -- count as ONE tracker-presence, not 50.
                SELECT DISTINCT
                    profile,
                    visit_id,
                    parent_entity
                FROM http_requests_enriched
                WHERE url LIKE 'http%'
                  AND relationship_tier NOT IN ('first-party', 'unknown')
            ),
            per_profile_entity AS (
                SELECT
                    profile,
                    parent_entity,
                    COUNT(*) AS n_visits_seen
                FROM per_visit_entity
                GROUP BY profile, parent_entity
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
                SELECT parent_entity
                FROM per_profile_entity
                GROUP BY parent_entity
                ORDER BY SUM(n_visits_seen) DESC
                LIMIT {top_n}
            )
            SELECT
                p.profile,
                p.parent_entity,
                p.n_visits_seen,
                ROUND(p.n_visits_seen * 100.0 / t.total_visits, 2)
                    AS pct_of_visits
            FROM per_profile_entity p
            JOIN profile_visit_totals t USING (profile)
            WHERE p.parent_entity IN (SELECT parent_entity FROM top_overall)
            ORDER BY p.parent_entity, p.profile
        """).df()
    return df


def jaccard_similarity_matrix(
    granularity: str = "subsidiary_entity",  # "domain", "subsidiary_entity", or "parent_entity"
) -> pd.DataFrame:
    """Pairwise Jaccard similarity of tracker sets between profiles.

    Jaccard(A, B) = |A ∩ B| / |A ∪ B|

    Returns a square DataFrame where rows and columns are profiles
    and cells are Jaccard similarity (1.0 = identical tracker sets,
    0.0 = completely disjoint).

    This is the single most useful "at a glance" summary of how
    different the tracking landscapes are across profiles.

    Args:
        granularity: The entity level at which to compare tracker sets.
            - "domain":            most granular; endpoint-level comparison.
            - "subsidiary_entity": mid-level; product/service-level comparison.
                                   Default — best balance of resolution and
                                   methodological accuracy.
            - "parent_entity":     least granular; corporate actor-level
                                   comparison. Will produce higher similarity
                                   scores by collapsing corporate families.
    """
    valid = {"domain", "subsidiary_entity", "parent_entity"}
    if granularity not in valid:
        raise ValueError(
            f"granularity must be one of {valid}, got '{granularity}'"
        )

    with db_session(read_only=True) as con:
        # Build the set of entities per profile at the requested granularity.
        sets_df = con.execute(f"""
            SELECT DISTINCT
                profile,
                {granularity} AS entity
            FROM http_requests_enriched
            WHERE url LIKE 'http%'
              AND relationship_tier NOT IN ('first-party', 'unknown')
        """).df()

    # Convert to a dict of sets — much cleaner than SQL for this step.
    profile_sets = {
        profile: set(group['entity'])
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
    profile_b: str | None = None,
    granularity: str = "parent_entity",  # "parent_entity" or "subsidiary_entity"
    min_visits: int = 3
) -> pd.DataFrame:
    """Trackers appearing significantly more in profile_a than profile_b.

    This is the central tool for answering "what does history seeding
    actually change?" If a tracker appears far more often in one
    profile than another, it's a strong candidate for behavioral
    targeting evidence.

    Args:
        profile_a: The "treatment" profile (with seeded history).
        profile_b: The comparison baseline (default: first configured profile).
        min_visits: Minimum visits in profile_a for the tracker to
            be included. Filters out one-off appearances that aren't
            statistically meaningful.
        granularity: Entity level at which to compare tracker presence.
            - "parent_entity":     corporate actor level (default). Best for
                                   headline findings — "which companies target
                                   seeded profiles more aggressively?"
            - "subsidiary_entity": product/service level. Useful for drill-down
                                   — "which specific products are driving lift?"

    Returns:
        DataFrame sorted by lift (ratio of A frequency to B frequency).
        Columns: parent_entity (or subsidiary_entity), visits_a, visits_b,
                 lift, delta.
    """
    valid = {"parent_entity", "subsidiary_entity"}
    if granularity not in valid:
        raise ValueError(
            f"granularity must be one of {valid}, got '{granularity}'"
        )
    
    profile_b = _resolve_baseline_profile(profile_b)
    if profile_a not in PROFILES or profile_b not in PROFILES:
        raise ValueError(
            f"Unknown profile(s). Valid: {PROFILES}"
        )

    with db_session(read_only=True) as con:
        df = con.execute(f"""
            WITH per_visit_entity AS (
                SELECT DISTINCT
                    profile,
                    visit_id,
                    {granularity} AS entity
                FROM http_requests_enriched
                WHERE url LIKE 'http%'
                  AND profile IN ('{profile_a}', '{profile_b}')
                  AND relationship_tier NOT IN ('first-party', 'unknown')
            ),
            counts AS (
                SELECT
                    entity,
                    SUM(CASE WHEN profile = '{profile_a}' THEN 1 ELSE 0 END)
                        AS visits_a,
                    SUM(CASE WHEN profile = '{profile_b}' THEN 1 ELSE 0 END)
                        AS visits_b
                FROM per_visit_entity
                GROUP BY entity
            )
            SELECT
                entity AS {granularity},
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


def trackers_unique_to_profile(
        profile: str,
        granularity: str = "parent_entity"  # "parent_entity" or "subsidiary_entity"
) -> pd.DataFrame:
    """Trackers that appear in ONLY this profile, in no others.

    The strongest possible evidence of profile-specific tracking:
    these hosts are summoned by something about this profile's
    seeded history that no other profile triggers.

    Args:
        profile: The profile to investigate.
        granularity: Entity level at which to test uniqueness.
            - "parent_entity":     corporate actor level (default). Strongest
                                    claim — this entire company appears nowhere
                                    else.
            - "subsidiary_entity": product/service level. A subsidiary product
                                    uniquely activated by this profile's history.

    Returns:
        DataFrame with columns: parent_entity or subsidiary_entity, n_visits_seen.
        Empty DataFrame if no profile-unique trackers exist.
    """
    valid = {"parent_entity", "subsidiary_entity"}
    if granularity not in valid:
        raise ValueError(
            f"granularity must be one of {valid}, got '{granularity}'"
        )
    
    other_profiles = [p for p in PROFILES if p != profile]
    if not other_profiles:
        raise ValueError("Need at least 2 profiles for this analysis.")

    # Build the "other profiles" list as a SQL IN clause.
    others_sql = ", ".join(f"'{p}'" for p in other_profiles)

    with db_session(read_only=True) as con:
        df = con.execute(f"""
            WITH entities_in_target AS (
                SELECT DISTINCT
                    {granularity} AS entity
                FROM http_requests_enriched
                WHERE profile = '{profile}'
                  AND url LIKE 'http%'
                  AND relationship_tier NOT IN ('first-party', 'unknown')
            ),
            entities_in_others AS (
                SELECT DISTINCT
                    {granularity} AS entity
                FROM http_requests_enriched
                WHERE profile IN ({others_sql})
                  AND url LIKE 'http%'
                  AND relationship_tier NOT IN ('first-party', 'unknown')
            ),
            unique_entities AS (
                SELECT entity FROM entities_in_target
                EXCEPT
                SELECT entity FROM entities_in_others
            )
            SELECT
                u.entity AS {granularity},
                COUNT(DISTINCT r.visit_id) AS n_visits_seen
            FROM unique_entities u
            JOIN http_requests_enriched r
                ON r.{granularity} = u.entity
            WHERE r.profile = '{profile}'
              AND r.relationship_tier NOT IN ('first-party', 'unknown')
            GROUP BY u.entity
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

    baseline = _resolve_baseline_profile()
    for profile in PROFILES:
        if profile == baseline:
            continue
        print(f"\nTop 10 differential trackers ({profile} vs {baseline}):")
        diff = differential_trackers(profile, baseline).head(10)
        print(diff.to_string(index=False))