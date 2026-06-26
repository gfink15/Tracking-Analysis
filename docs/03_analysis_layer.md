# Analysis Layer

This document covers the SQL-and-statistics modules in `src/analysis/`.
These modules take the Parquet/DuckDB layer as input and produce
analytical DataFrames as output. They contain NO plotting code — that
lives in `src/viz/`.

## Table of Contents

1. [Design Principles](#design-principles)
2. [`trackers.py`](#trackerspy)
3. [`cookies.py`](#cookiespy)
4. [`fingerprinting.py`](#fingerprintingpy)
5. [`ads.py`](#adspy)
6. [`statistics.py`](#statisticspy)
7. [`topic_modeling.py`](#topic_modelingpy)
8. [Common SQL Patterns](#common-sql-patterns)

---

## Design Principles

Every function in this layer follows the same contract:

1. **Input:** None (queries DuckDB directly) or a few simple
   parameters (profile name, threshold, etc.)
2. **Database access:** Via `db_session(read_only=True)` — analysis
   never writes to the DB
3. **Heavy lifting:** SQL in DuckDB, not Python loops
4. **Output:** A pandas DataFrame ready for `src/viz/` or further
   statistical processing
5. **Side effects:** None (except optional print statements in
   `__main__` smoke tests)

**Critical:** Analysis modules return data, not figures. This
separation means the same analysis function can drive multiple
visualizations, and visualizations can be reworked without re-running
analysis.

---

## `trackers.py`

**Purpose:** Quantify third-party tracker activity per profile and
across profiles.

### SQL Fragments

| Constant | Purpose |
|---|---|
| `HOSTNAME_SQL` | DuckDB regex to extract hostname from a URL. Reused in every query that needs to identify request destinations. |
| `ETLD1_SQL` | Naive "last two labels" extraction for effective TLD+1 (e.g., `ads.doubleclick.net` → `doubleclick.net`). A proper implementation would use the public suffix list — noted as a TODO for paper-quality results. |

### Key Functions

#### `tracker_prevalence_by_profile(use_tracker_list=False) -> pd.DataFrame`

Per-profile counts of unique third-party hosts contacted.

**Returns columns:**
- `profile` — profile key
- `n_visits` — total visits for this profile (denominator)
- `n_unique_hosts` — distinct hostnames contacted
- `n_unique_etld1` — distinct eTLD+1s (deduplicates subdomains)
- `hosts_per_visit` — average distinct hosts per page
- `etld1_per_visit` — average distinct eTLD+1s per page

This is the headline metric for "how much tracking is each profile
exposed to?"

#### `tracker_frequency_table(top_n=50) -> pd.DataFrame`

Frequency of each top-N tracker across profiles. Long-format DataFrame
suitable for heatmap visualization.

**Returns columns:**
- `profile`, `etld1`, `n_visits_seen`, `pct_of_visits`

**Critical:** Counts at the **visit level**, not the request level.
A page making 50 requests to `doubleclick.net` counts as ONE
tracker-presence, not 50. Without this deduplication, popular sites
that hammer one tracker would dominate the rankings.

#### `jaccard_similarity_matrix() -> pd.DataFrame`

Pairwise Jaccard similarity of tracker sets between profiles. Returns
a square DataFrame where rows/columns are profiles and cells are
similarity values in [0, 1].

‹‹LB››
J(A, B) = \frac{|A \cap B|}{|A \cup B|}
‹‹/LB››

**Interpretation:**
- Values near 1.0 → profiles share most trackers (weak differentiation)
- Values 0.5–0.8 → healthy signal (most trackers universal, meaningful
  minority differ by profile)
- Values < 0.3 → very strong profile-specific tracking

#### `differential_trackers(profile_a, profile_b='control', min_visits=3)`

Trackers appearing significantly more in profile_a than profile_b.
Sorted by lift descending.

**Returns columns:**
- `etld1`, `visits_a`, `visits_b`, `delta`, `lift`

**Smoothing:** Lift uses `(visits_a + 1) / (visits_b + 1)` to handle
zero-division and give meaningful values for "appears in A, absent
in B" cases. This is standard Laplace smoothing.

#### `trackers_unique_to_profile(profile) -> pd.DataFrame`

Trackers appearing in ONLY this profile, in no others. The strongest
possible evidence of profile-specific tracking — uses SQL `EXCEPT`
operator.

---

## `cookies.py`

**Purpose:** Cookie behavior analysis — volume, lifespan, retargeting
network presence, and cookie syncing.

### Constants

#### `RETARGETING_HOSTS: tuple[str, ...]`

Hand-curated list of hostnames known to be behavioral-ad retargeters.
Document this list in your methodology section.

To add a retargeter:
1. Add the domain (e.g., `'newretargeter.com'`) to the tuple
2. The SQL `LIKE` patterns generated downstream pick it up automatically

### Key Functions

#### `cookie_counts_by_profile() -> pd.DataFrame`

First-party vs. third-party cookie counts per profile.

**First-party determination:** A cookie is first-party if its host
equals the visited site's host, OR if the cookie's host ends in
`.<site_host>` (subdomain match). This catches `ads.nytimes.com`
cookies on `nytimes.com` correctly.

**Returns columns:**
- `profile`, `n_total`, `n_first_party`, `n_third_party`,
  `n_unique_hosts`, `pct_third_party`

#### `cookie_lifespan_distribution(third_party_only=True)`

Distribution of cookie lifespans bucketed into:
`session`, `<1d`, `1-7d`, `7-30d`, `30-365d`, `1y+`.

**Lifespan calculation:**

lifespan_days = (expiry_epoch - capture_epoch) / 86400

With `expiry IS NULL OR expiry = 0` → `session`.

Retargeting cookies need 30+ days persistence; seeing more long-lived
cookies in seeded profiles is direct evidence of behavioral profiling.

#### `retargeting_cookie_presence() -> pd.DataFrame`

Counts cookies from known retargeting networks, per profile. Uses
`LIKE '%<host>%'` matching to catch subdomains.

**Returns columns:**
- `profile`, `retargeter`, `n_cookies`, `n_visits_affected`

#### `detect_cookie_syncs(min_id_length=10) -> pd.DataFrame`

Identifies probable cookie-sync events: the same long alphanumeric
ID-like value appearing in cookies from different hosts within the
same visit.

**Heuristic limitations:** Will produce false positives (shared
session tokens, common defaults). Always inspect before reporting.
For paper-quality results, cross-reference with the academic
literature on sync detection.

---

## `fingerprinting.py`

**Purpose:** Detect browser fingerprinting using the methodology from
Englehardt & Narayanan (2016).

### Constants

| Constant | Purpose |
|---|---|
| `CANVAS_SYMBOLS` | JS symbols for canvas fingerprinting (toDataURL, fillText, etc.) |
| `WEBGL_SYMBOLS` | WebGL parameter queries |
| `AUDIO_SYMBOLS` | AudioContext fingerprinting symbols |
| `NAVIGATOR_SYMBOLS` | navigator.* attribute reads |
| `SCREEN_SYMBOLS` | screen.* attribute reads |
| `FONT_SYMBOLS` | Canvas measureText (used for font enumeration) |
| `TECHNIQUE_SYMBOLS` | Dict mapping technique name → symbol tuple |

### Key Functions

#### `detect_canvas_fingerprinters(min_text_calls=1) -> pd.DataFrame`

Implements the canonical methodology: a SCRIPT exhibits canvas
fingerprinting if it (a) writes text to canvas AND (b) reads canvas
pixels in the same visit.

**Why script-level, not call-level:** Counting raw API calls gives
garbage because any page with analytics has `getImageData` calls. The
real signal is correlated behavior within a single script.

**Returns columns:**
- `profile`, `script_url`, `n_text_calls`, `n_read_calls`, `n_visits`

#### `detect_audio_fingerprinters() -> pd.DataFrame`

Scripts that create oscillator/compressor AND read channel data.
Audio fingerprinting has very low false-positive rate when both
behaviors co-occur.

#### `detect_navigator_probers(min_attributes=5)`

Scripts reading 5+ distinct navigator/screen attributes. The count
of *distinct* attributes (not total reads) is the diagnostic signal.

#### `fingerprinter_summary() -> pd.DataFrame`

One row per profile with counts of detected fingerprinters by
technique. The headline table for fingerprinting analysis.

---

## `ads.py`

**Purpose:** Quantitative analysis of captured ad content.

### Constants

#### `KEYWORD_CATEGORIES: dict[str, list[str]]`

Hand-curated topic keywords for first-pass content classification
without invoking the full topic-modeling pipeline.

Categories: `retail`, `finance`, `health`, `travel`, `auto`, `tech`,
`media`, `food`. Categories are NOT mutually exclusive — an ad can
match multiple. This is the right behavior for keyword classification
but worth documenting in methodology.

### Key Functions

#### `ad_counts_by_profile(min_confidence=None) -> pd.DataFrame`

Per-profile ad volume and density.

**`min_confidence` parameter:**
- `'high'` — only high-confidence detections (paper-quality numbers)
- `'medium'` — high + medium (exploratory)
- `None` — all detections (debugging only)

**Returns columns:**
- `profile`, `n_ads`, `n_visits_with_ads`, `n_total_visits`,
  `ads_per_visit`, `pct_visits_with_ads`

#### `network_distribution_by_profile(min_confidence='high')`

Distribution of advertiser networks per profile in long format.
Pivot for heatmaps via `top_advertiser_networks()`.

#### `keyword_category_counts() -> pd.DataFrame`

Counts ads matching each `KEYWORD_CATEGORIES` entry via case-insensitive
`ILIKE`. Returns long format suitable for grouped bar charts.

#### `differential_keyword_categories(profile_a, profile_b='control')`

Categories where `profile_a`'s ads differ most from `profile_b`'s.
Sorted by lift descending.

#### `ads_per_visit_with_tracking() -> pd.DataFrame`

Joins ad counts and tracker counts at the visit level for scatter
plots and correlation analysis. One row per (profile, visit).

---

## `statistics.py`

**Purpose:** Centralized hypothesis testing for cross-profile
comparisons.

### Dataclass: `TestResult`

One hypothesis test result with all metadata for inclusion in a paper.

| Field | Purpose |
|---|---|
| `test_name` | Test family (e.g., "chi-square (2x2)") |
| `comparison` | Human-readable comparison label |
| `statistic` | Raw test statistic |
| `p_value` | Uncorrected p-value |
| `p_value_corrected` | After multiple-testing correction |
| `correction_method` | `"bonferroni"`, `"benjamini-hochberg"`, or `"none"` |
| `effect_size` | Cramér's V, rank-biserial, ε², etc. |
| `effect_size_name` | Human-readable effect size label |
| `n_observations` | Total sample size |
| `reject_null` | Whether to reject at α=`ALPHA` after correction |
| `notes` | Caveats or warnings |

`.to_string()` formats results with significance markers (`***`, `**`,
`*`, `ns`) for easy reading.

### Key Functions

#### `correct_pvalues(pvalues, method='bonferroni')`

Apply multiple-testing correction to a batch of p-values.

| Method | When to use |
|---|---|
| `'bonferroni'` | Confirmatory tests (≤20 hypotheses). Controls family-wise error rate. |
| `'benjamini-hochberg'` | Exploratory analysis (hundreds of tests). Controls false discovery rate. |

#### `chi_square_tracker_presence(profile_a, profile_b, tracker_etld1)`

Tests whether a specific tracker appears differently across two
profiles. Builds a 2×2 contingency table and runs χ² — or Fisher's
exact when expected frequencies are below 5 (the textbook fallback).

#### `chi_square_batch(profile_a, profile_b, trackers, correction='bonferroni')`

Runs χ² for many trackers and applies family-wise correction. This is
the function you actually call — single tests are rare in practice.

#### `mann_whitney_metric(profile_a, profile_b, metric_sql, metric_name)`

Compares a per-visit metric distribution between two profiles. Uses
Mann-Whitney U (not t-test) because OpenWPM metrics are heavily
right-skewed.

**`metric_sql` parameter:** A SQL string with a `{profiles}` placeholder.
Must return `(profile, visit_id, metric_value)` rows.

#### `kruskal_wallis_metric(metric_sql, metric_name, profiles=None)`

Omnibus test across ALL profiles. **Run this BEFORE pairwise tests**
as a sanity check — if the omnibus is non-significant, pairwise tests
are statistically suspect.

#### `pairwise_battery(metric_key, correction='bonferroni', baseline='control')`

Convenience function: runs Kruskal-Wallis + pairwise Mann-Whitney for
a metric defined in `METRIC_SQL`, with correction applied across pairs.
The function you call to ask "does <metric> differ across profiles?"

### `METRIC_SQL` Dictionary

Pre-defined SQL templates for common metrics. Centralizing these
ensures every comparison of "requests per visit" uses the same
definition.

Available keys:
- `requests_per_visit`
- `unique_hosts_per_visit`
- `cookies_per_visit`
- `third_party_cookies_per_visit`

Adding a new metric is as simple as adding a key to this dict.

---

## `topic_modeling.py`

**Purpose:** Semantic clustering of ad OCR text using BERTopic.

### Pipeline

1. `load_ad_corpus()` → cleaned OCR text DataFrame
2. `fit_bertopic(corpus)` → fitted model saved to
   `artifacts/models/bertopic/<name>/`
3. `load_fitted_model(name)` → reload without refitting
4. `topic_distribution_by_profile()` → cross-profile topic shares
5. `differential_topics(profile_a, profile_b)` → topics
   overrepresented in profile_a

### Key Functions

#### `_clean_ocr_text(text) -> str`

Aggressive cleaning:
- Drops URLs and email-like patterns
- Drops non-letter characters
- Lowercases (OCR can't reliably distinguish case)
- Drops tokens shorter than 3 chars

#### `load_ad_corpus(min_chars=20, min_confidence='high')`

Loads ads with sufficient cleaned text. Filters out logos and tiny
banners that can't be meaningfully topic-modeled.

#### `fit_bertopic(corpus, embedding_model, min_topic_size, save_name)`

Fits BERTopic and saves to disk.

**Default `embedding_model='all-MiniLM-L6-v2'`:** Fast and decent.
For higher quality on a powerful machine, try `'all-mpnet-base-v2'`
(3× slower, noticeably better coherence).

**`min_topic_size=10`:** Smallest cluster to retain. Prevents
BERTopic from creating ultra-niche topics of 2-3 ads (usually noise).

#### `differential_topics(profile_a, profile_b='control', save_name='ads_v1')`

Topics overrepresented in `profile_a`. Returns DataFrame with topic
keywords, percentages per profile, delta, and lift.

---

## Common SQL Patterns

### Visit-Level Deduplication

When counting tracker presences, always count at the visit level:

```sql
WITH per_visit_host AS (
    SELECT DISTINCT profile, visit_id, etld1
    FROM ...
)

Without DISTINCT, a page making 50 requests to one tracker would count as 50 presences.
Smoothed Lift

When dividing counts to compute lift, always smooth:
sql

(visits_a + 1.0) / (visits_b + 1.0) AS lift

Or for percentages, (pct_a + 0.1) / (pct_b + 0.1).
eTLD+1 Extraction
sql

array_to_string(
    list_slice(string_split(host, '.'), -2, -1),
    '.'
)

This is the naive "last two labels" approach. Documented as intentionally simple; for paper quality, swap in a Public Suffix List lookup.
Cross-Profile Pivot

To get a wide DataFrame for heatmaps:
python

pivot = (long_df.pivot(index='etld1',
                       columns='profile',
                       values='pct_of_visits')
                .reindex(columns=PROFILES)
                .fillna(0))

Always .reindex(columns=PROFILES) to enforce the canonical order and to add zero-rows for profiles missing from the data.