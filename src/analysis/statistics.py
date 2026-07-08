"""
src/analysis/statistics.py — Hypothesis testing for profile comparisons.

This module is the statistical backbone of the project. Every claim
of the form "profile A differs from profile B" should ultimately be
backed by a function here. Centralizing these functions has three
benefits:

  1. Methodological consistency — the same test is used the same way
     wherever it's applied.
  2. Single point of correction — when you switch from Bonferroni to
     Benjamini-Hochberg, you change it in one place.
  3. Reproducibility — the paper's methods section can cite these
     functions by name.

Test selection guide (which function to use when):
  • Comparing PROPORTIONS or PRESENCE/ABSENCE → chi_square_*
  • Comparing COUNT or CONTINUOUS distributions → mann_whitney_*
  • Comparing MULTIPLE GROUPS at once → kruskal_wallis_*
  • Comparing TRACKER SETS across profiles → permutation_test_jaccard
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional, cast

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from config import ALPHA, BONFERRONI_CORRECT, PROFILES
from src.utils.db import db_session


def _resolve_baseline_profile(baseline: str | None = None) -> str:
    """Resolve the baseline profile from the configured profile list."""
    if baseline is None:
        if not PROFILES:
            raise ValueError("config.PROFILES is empty.")
        baseline = PROFILES[0]
    if baseline not in PROFILES:
        raise ValueError(f"Unknown profile: {baseline!r}. Valid: {PROFILES}")
    return baseline


# ─────────────────────────────────────────────────────────────────────
# RESULT STRUCTURE — every test returns one of these
# ─────────────────────────────────────────────────────────────────────
@dataclass
class TestResult:
    """A single hypothesis test result with all metadata for a paper.

    Fields here are the bare minimum a reviewer needs to evaluate
    the test. The to_string() method formats them for inclusion
    in notebooks; the asdict() inherited from dataclass works for
    DataFrame aggregation.
    """
    test_name: str              # e.g., "chi-square (2x2)"
    comparison: str             # e.g., "gaming vs control: doubleclick.net"
    statistic: float            # raw test statistic
    p_value: float              # uncorrected p
    p_value_corrected: float    # after multiple-testing correction
    correction_method: str      # "bonferroni", "benjamini-hochberg", "none"
    effect_size: Optional[float]  # Cramér's V, rank-biserial, etc.
    effect_size_name: Optional[str]
    n_observations: int         # total sample size
    reject_null: bool           # at α=ALPHA after correction
    notes: str = ""             # caveats, warnings

    def to_string(self) -> str:
        sig = "***" if self.p_value_corrected < 0.001 else \
              "**"  if self.p_value_corrected < 0.01  else \
              "*"   if self.p_value_corrected < ALPHA else "ns"
        es = (f", {self.effect_size_name}={self.effect_size:.3f}"
              if self.effect_size is not None else "")
        return (f"{self.test_name} | {self.comparison} | "
                f"stat={self.statistic:.3f}, p={self.p_value_corrected:.4g} "
                f"{sig}{es}, n={self.n_observations}")


# ─────────────────────────────────────────────────────────────────────
# MULTIPLE TESTING CORRECTION
# ─────────────────────────────────────────────────────────────────────
def correct_pvalues(
    pvalues: list[float],
    method: str = "bonferroni",
) -> tuple[list[float], str]:
    """Apply multiple-testing correction to a batch of p-values.

    Args:
        pvalues: Raw p-values from a family of tests.
        method: 'bonferroni' (conservative, controls FWER) or
            'benjamini-hochberg' (less conservative, controls FDR).
            Use Bonferroni for confirmatory tests (≤20 hypotheses);
            use BH for exploratory analyses (hundreds of tests).

    Returns:
        Tuple of (corrected_pvalues, method_used).
    """
    n = len(pvalues)
    if n == 0:
        return [], method
    arr = np.array(pvalues, dtype=float)

    if method == "bonferroni":
        corrected = np.minimum(arr * n, 1.0)
    elif method == "benjamini-hochberg":
        # Standard BH procedure: sort, scale by rank/n, enforce monotonicity
        order = np.argsort(arr)
        ranked = arr[order]
        scaled = ranked * n / (np.arange(n) + 1)
        # Monotone non-decreasing from the right
        scaled = np.minimum.accumulate(scaled[::-1])[::-1]
        corrected = np.empty(n)
        corrected[order] = np.minimum(scaled, 1.0)
    else:
        raise ValueError(f"Unknown correction method: {method}")

    return corrected.tolist(), method


# ─────────────────────────────────────────────────────────────────────
# CHI-SQUARE TESTS — for categorical comparisons
# ─────────────────────────────────────────────────────────────────────
def chi_square_tracker_presence(
    profile_a: str,
    profile_b: str,
    tracker_etld1: str,
) -> TestResult:
    """Does tracker X appear more often in profile A than profile B?

    Builds a 2x2 contingency table:
                    | tracker present | tracker absent |
        profile_a  |       a         |       b        |
        profile_b  |       c         |       d        |

    Then runs chi-square (or Fisher's exact for small expected
    frequencies). This is the workhorse test for "does seeded
    history make this specific tracker more likely to appear?"
    """
    with db_session(read_only=True) as con:
        df = con.execute(f"""
            WITH visits AS (
                SELECT
                    profile,
                    visit_id,
                    MAX(CASE WHEN url LIKE '%{tracker_etld1}%'
                        THEN 1 ELSE 0 END) AS has_tracker
                FROM http_requests
                WHERE profile IN ('{profile_a}', '{profile_b}')
                GROUP BY profile, visit_id
            )
            SELECT
                profile,
                SUM(has_tracker)              AS n_with,
                COUNT(*) - SUM(has_tracker)   AS n_without
            FROM visits
            GROUP BY profile
        """).df()

    if len(df) < 2:
        return TestResult(
            test_name="chi-square (2x2)",
            comparison=f"{profile_a} vs {profile_b}: {tracker_etld1}",
            statistic=float("nan"), p_value=1.0, p_value_corrected=1.0,
            correction_method="none", effect_size=None, effect_size_name=None,
            n_observations=0, reject_null=False,
            notes="Insufficient data — one or both profiles missing",
        )

    # Reshape to 2x2 table
    a_row = df[df['profile'] == profile_a].iloc[0]
    b_row = df[df['profile'] == profile_b].iloc[0]
    table = np.array([
        [a_row['n_with'], a_row['n_without']],
        [b_row['n_with'], b_row['n_without']],
    ])

    n = int(table.sum())

    # If any expected frequency is < 5, use Fisher's exact instead.
    # This is the standard guideline for small-sample chi-square.
    expected = (table.sum(axis=1, keepdims=True)
                * table.sum(axis=0, keepdims=True)) / n
    use_fisher = (expected < 5).any()

    if use_fisher:
        result = cast(tuple[float, float], stats.fisher_exact(table))
        statistic = float(result[0])
        p_value = float(result[1])
        test_name = "Fisher's exact (2x2)"
    else:
        result = cast(tuple[float, float, Any, Any], stats.chi2_contingency(table))
        statistic = float(result[0])
        p_value = float(result[1])
        test_name = "chi-square (2x2)"

    # Cramér's V — effect size for 2x2 tables, range [0, 1].
    cramers_v = float(np.sqrt(statistic / n)) if n > 0 and not use_fisher else None

    return TestResult(
        test_name=test_name,
        comparison=f"{profile_a} vs {profile_b}: {tracker_etld1}",
        statistic=statistic,
        p_value=float(p_value),
        p_value_corrected=float(p_value),   # caller batches & corrects later
        correction_method="none",
        effect_size=cramers_v,
        effect_size_name="Cramer's V" if cramers_v is not None else None,
        n_observations=n,
        reject_null=bool(p_value < ALPHA),
    )


def chi_square_batch(
    profile_a: str,
    profile_b: str,
    trackers: list[str],
    correction: str = "bonferroni",
) -> pd.DataFrame:
    """Run chi-square for many trackers, then correct for multiple testing.

    This is the function you actually call in practice. Running a
    single chi-square is rare; running 50–500 of them across all
    differential trackers is the norm. Batching ensures the
    correction is applied as a family, not per-test.

    Returns a DataFrame ready for the paper's "significant trackers"
    table — sorted by corrected p-value, with effect sizes included.
    """
    raw_results = [
        chi_square_tracker_presence(profile_a, profile_b, t)
        for t in trackers
    ]

    raw_ps = [r.p_value for r in raw_results]
    corrected_ps, method = correct_pvalues(raw_ps, method=correction)

    # Update each result with its corrected p-value and rejection.
    for r, p_corr in zip(raw_results, corrected_ps):
        r.p_value_corrected = p_corr
        r.correction_method = method
        r.reject_null = p_corr < ALPHA

    df = pd.DataFrame([asdict(r) for r in raw_results])
    return df.sort_values('p_value_corrected').reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────
# MANN-WHITNEY U — for continuous/count distributions
# ─────────────────────────────────────────────────────────────────────
def mann_whitney_metric(
    profile_a: str,
    profile_b: str,
    metric_sql: str,
    metric_name: str,
) -> TestResult:
    """Compare a per-visit metric distribution between two profiles.

    Args:
        profile_a, profile_b: profiles to compare
        metric_sql: a SQL expression that returns one row per visit
            with columns (profile, visit_id, metric_value). MUST
            include a {profiles} placeholder for the IN clause.
        metric_name: human-readable label for the result.

    Example:
        sql = '''
            SELECT profile, visit_id, COUNT(*) AS metric_value
            FROM http_requests
            WHERE profile IN {profiles}
            GROUP BY profile, visit_id
        '''
        result = mann_whitney_metric('gaming', 'control', sql,
                         'requests per visit')

    Why Mann-Whitney instead of t-test?
        Almost all OpenWPM metrics are heavily right-skewed (a few
        sites make hundreds of requests; most make ~30). t-tests
        assume normality; Mann-Whitney doesn't. Default to MW unless
        you've explicitly verified normality.
    """
    profiles_clause = f"('{profile_a}', '{profile_b}')"
    with db_session(read_only=True) as con:
        df = con.execute(
            metric_sql.format(profiles=profiles_clause)
        ).df()

    a_vals = df[df['profile'] == profile_a]['metric_value'].values
    b_vals = df[df['profile'] == profile_b]['metric_value'].values

    if len(a_vals) == 0 or len(b_vals) == 0:
        return TestResult(
            test_name="Mann-Whitney U",
            comparison=f"{profile_a} vs {profile_b}: {metric_name}",
            statistic=float("nan"), p_value=1.0, p_value_corrected=1.0,
            correction_method="none", effect_size=None, effect_size_name=None,
            n_observations=len(a_vals) + len(b_vals), reject_null=False,
            notes="Insufficient data in one or both profiles",
        )

    # two-sided test by default; switch to 'greater'/'less' if you
    # have a directional hypothesis you've pre-registered.
    result = cast(tuple[float, float], stats.mannwhitneyu(a_vals, b_vals, alternative='two-sided'))
    u_stat = float(result[0])
    p_value = float(result[1])

    # Rank-biserial correlation — standard effect size for MW.
    # Range [-1, 1]; |r| > 0.3 is moderate, > 0.5 is large.
    n1, n2 = len(a_vals), len(b_vals)
    rank_biserial = 1 - (2 * u_stat) / (n1 * n2)

    return TestResult(
        test_name="Mann-Whitney U",
        comparison=f"{profile_a} vs {profile_b}: {metric_name}",
        statistic=float(u_stat),
        p_value=float(p_value),
        p_value_corrected=float(p_value),
        correction_method="none",
        effect_size=float(rank_biserial),
        effect_size_name="rank-biserial r",
        n_observations=n1 + n2,
        reject_null=bool(p_value < ALPHA),
    )


# ─────────────────────────────────────────────────────────────────────
# KRUSKAL-WALLIS — for >2 group comparisons
# ─────────────────────────────────────────────────────────────────────
def kruskal_wallis_metric(
    metric_sql: str,
    metric_name: str,
    profiles: Optional[list[str]] = None,
) -> TestResult:
    """Test whether a metric differs across ALL profiles simultaneously.

    Use this BEFORE running pairwise Mann-Whitney tests, as an
    omnibus check. If Kruskal-Wallis is non-significant, pairwise
    tests are unlikely to be informative.

    metric_sql should produce (profile, visit_id, metric_value).
    """
    if profiles is None:
        profiles = PROFILES
    profiles_clause = "(" + ", ".join(f"'{p}'" for p in profiles) + ")"

    with db_session(read_only=True) as con:
        df = con.execute(
            metric_sql.format(profiles=profiles_clause)
        ).df()

    groups = [
        df[df['profile'] == p]['metric_value'].values
        for p in profiles if (df['profile'] == p).any()
    ]

    if len(groups) < 2:
        return TestResult(
            test_name="Kruskal-Wallis H",
            comparison=f"all profiles: {metric_name}",
            statistic=float("nan"), p_value=1.0, p_value_corrected=1.0,
            correction_method="none", effect_size=None, effect_size_name=None,
            n_observations=sum(len(g) for g in groups), reject_null=False,
            notes="Need at least 2 non-empty groups",
        )

    result = cast(tuple[float, float], stats.kruskal(*groups))
    h_stat = float(result[0])
    p_value = float(result[1])
    n = sum(len(g) for g in groups)

    # Epsilon-squared effect size for Kruskal-Wallis.
    # (h - k + 1) / (n - k) where k = number of groups.
    k = len(groups)
    epsilon_sq = max(0.0, (h_stat - k + 1) / (n - k)) if n > k else None

    return TestResult(
        test_name="Kruskal-Wallis H",
        comparison=f"{k} profiles: {metric_name}",
        statistic=float(h_stat),
        p_value=float(p_value),
        p_value_corrected=float(p_value),
        correction_method="none",
        effect_size=epsilon_sq,
        effect_size_name="epsilon²" if epsilon_sq is not None else None,
        n_observations=n,
        reject_null=bool(p_value < ALPHA),
    )


# ─────────────────────────────────────────────────────────────────────
# COMMON METRIC SQL TEMPLATES — reusable across pairwise comparisons
# ─────────────────────────────────────────────────────────────────────
# Defining these as constants means every comparison of "requests per
# visit" uses the same definition. Avoids the bug where one notebook
# counts http_requests and another counts http_responses, producing
# subtly different results.
METRIC_SQL = {
    'requests_per_visit': """
        SELECT profile, visit_id, COUNT(*) AS metric_value
        FROM http_requests
        WHERE profile IN {profiles}
        GROUP BY profile, visit_id
    """,
    'unique_hosts_per_visit': """
        SELECT profile, visit_id,
               COUNT(DISTINCT regexp_extract(url, '://([^/]+)', 1)) AS metric_value
        FROM http_requests
        WHERE profile IN {profiles} AND url LIKE 'http%'
        GROUP BY profile, visit_id
    """,
    'cookies_per_visit': """
        SELECT profile, visit_id, COUNT(*) AS metric_value
        FROM javascript_cookies
        WHERE profile IN {profiles}
        GROUP BY profile, visit_id
    """,
    'third_party_cookies_per_visit': """
        SELECT profile, visit_id, COUNT(*) AS metric_value
        FROM javascript_cookies c
        JOIN site_visits v USING (profile, visit_id)
        WHERE c.profile IN {profiles}
          AND regexp_extract(v.site_url, '://([^/]+)', 1) != c.host
        GROUP BY profile, visit_id
    """,
}


# ─────────────────────────────────────────────────────────────────────
# CONVENIENCE: run the full pairwise battery for one metric
# ─────────────────────────────────────────────────────────────────────
def pairwise_battery(
    metric_key: str,
    correction: str = "bonferroni",
    baseline: str | None = None,
) -> pd.DataFrame:
    """Run Mann-Whitney for every non-baseline profile vs baseline,
    plus Kruskal-Wallis omnibus, with correction applied across pairs.

    This is THE function you call to ask "does <metric> differ
    across my profiles?" — it produces a complete results table.
    """
    baseline = _resolve_baseline_profile(baseline)
    if metric_key not in METRIC_SQL:
        raise ValueError(f"Unknown metric: {metric_key}. "
                         f"Available: {list(METRIC_SQL)}")
    sql = METRIC_SQL[metric_key]

    # Omnibus first
    omnibus = kruskal_wallis_metric(sql, metric_key)
    results = [omnibus]

    # Pairwise vs baseline
    pair_results = [
        mann_whitney_metric(p, baseline, sql, metric_key)
        for p in PROFILES if p != baseline
    ]
    raw_ps = [r.p_value for r in pair_results]
    corrected_ps, method = correct_pvalues(raw_ps, method=correction)
    for r, pc in zip(pair_results, corrected_ps):
        r.p_value_corrected = pc
        r.correction_method = method
        r.reject_null = pc < ALPHA
    results.extend(pair_results)

    return pd.DataFrame([asdict(r) for r in results])


if __name__ == "__main__":
    # Smoke test — run the full battery for the headline metric.
    baseline = PROFILES[0]
    print("Pairwise battery: requests_per_visit")
    print(pairwise_battery('requests_per_visit', baseline=baseline).to_string(index=False))
    print("\nPairwise battery: unique_hosts_per_visit")
    print(pairwise_battery('unique_hosts_per_visit', baseline=baseline).to_string(index=False))