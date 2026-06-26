# Ingestion Layer

This document covers the modules that transform raw OpenWPM output
into the unified Parquet store. If you're debugging "data isn't
showing up correctly in my analysis," start here.

## Table of Contents

1. [`src/ingestion/load_sqlite.py`](#load_sqlitepy)
2. [`src/ingestion/load_ad_artifacts.py`](#load_ad_artifactspy)
3. [`scripts/init_database.py`](#init_databasepy)
4. [`src/utils/db.py`](#dbpy)
5. [Common Issues & Debugging](#common-issues--debugging)

---

## `load_sqlite.py`

**Purpose:** Convert per-profile OpenWPM SQLite databases into
profile-tagged Parquet files.

### Inputs

- One `crawl-data.sqlite` per profile, located at
  `data/<profile>/crawl-data.sqlite`
- The list of profiles from `config.PROFILES`
- The list of tables from `config.OPENWPM_TABLES`

### Outputs

- One Parquet file per table at `artifacts/parquet/<table>.parquet`
- Each Parquet file contains rows from ALL profiles, with an added
  `profile` column identifying the source

### Key Functions

#### `_verify_sqlite_files_exist() -> dict[str, Path]`

Fails fast if any profile's SQLite file is missing. Returns a mapping
of profile name → SQLite path. **This check runs before any ingestion
work**, so you catch missing crawls in seconds instead of mid-ingest.

| Returns | When |
|---|---|
| `dict[str, Path]` | All profiles have SQLite files |
| Raises `FileNotFoundError` | Any profile is missing |

#### `_get_table_columns(con, profile, table) -> list[str]`

Returns the column names for a table in a specific attached SQLite
database. Used to detect schema drift across profiles (e.g., if you
upgraded OpenWPM between crawl batches).

#### `_convert_table(con, table, sqlite_paths) -> int`

Converts one OpenWPM table across all profiles into one Parquet file.

**Schema-drift handling:** Takes the *intersection* of columns across
all profiles. Columns unique to one profile are dropped with a warning.
This makes the pipeline graceful in the face of OpenWPM version
differences across crawl batches.

**Performance characteristics:**
- Uses `COPY (SELECT ... UNION ALL ...) TO file` to stream rows
  directly from SQLite to Parquet without buffering in Python memory
- ZSTD compression typically achieves ~5-10× size reduction vs raw
- Row group size 100,000 balances compression ratio against query
  selectivity

**Critical:** Uses `UNION ALL`, NOT `UNION`. `UNION` deduplicates rows
which would silently drop legitimately identical rows across profiles.

#### `sqlite_to_parquet() -> None`

Main entry point. Orchestrates verification, attachment, conversion,
and progress reporting. Invoke via:

```bash
python -m src.ingestion.load_sqlite

Internal Variables Reference
Variable	Type	Purpose
union_parts	list[str]	Per-profile SELECT statements. Joined with UNION ALL to build the final query.
common_cols	set[str]	Intersection of column sets across all profiles. Defines what gets written to Parquet.
column_sets	dict[str, set[str]]	Maps profile name → its column set for the table being processed. Used for the intersection calculation and for warning about dropped columns.
cols_sql	str	Sorted, comma-joined column list. Sorting ensures deterministic column order across runs (important for reproducibility).
n_rows	int	Row count of the output Parquet, used for progress reporting.
load_ad_artifacts.py

Purpose: Walk the ad screenshot artifact tree, extract OCR text, identify advertiser networks, and emit a single ads.parquet.
Inputs

    data/<profile>/ads/<visit_id>/<ad_hash>.json (ad metadata)
    data/<profile>/ads/<visit_id>/<ad_hash>.png (ad screenshot)
    Sidecar files (_visit_summary.json, _ad_content.json) — skipped

Outputs

    artifacts/parquet/ads.parquet — one row per detected ad

Key Constants
OCR_AVAILABLE: bool

Set at import time based on whether pytesseract + PIL can be imported. If False, ingestion still runs but ocr_text will be empty. Useful for environments without the system tesseract binary.
AD_NETWORK_PATTERNS: list[tuple[re.Pattern, str]]

Ordered list of (compiled_regex, network_id) pairs used to classify an ad's iframe src into a known network identifier.

Order matters — patterns are checked top-to-bottom and matching stops at first hit. Place more specific patterns first (doubleclick.net before a generic google.com/ads).

To add a new ad network:

    Add a (pattern_string, identifier) tuple to AD_NETWORK_PATTERNS
    The list is compiled to _COMPILED_PATTERNS at module load — no other changes needed

Dataclass: ProcessedAd

One row in the final ads.parquet table.
Field	Type	Source	Purpose
profile	str	walked dir	Which profile saw this ad
visit_id	int	JSON	Joins to site_visits.visit_id
page_url	str	JSON	URL where ad was detected
ad_hash	str	filename	Stable unique ID (md5 of url+marker)
ad_src	str | None	JSON ad_metadata.src	iframe src URL
ad_tag	str | None	JSON ad_metadata.tag	DOM tag (IFRAME, DIV, INS)
ad_width, ad_height	float	JSON rect.w/h	Pixel dimensions
ad_x, ad_y	float	JSON rect.x/y	Position on page
advertiser_network	str	ad_metadata.network OR derived via identify_ad_network()	Canonical network ID
ocr_text	str	tesseract on PNG	Extracted text content
ocr_char_count	int	derived	Convenience field for filtering
has_screenshot	bool	filesystem	Whether PNG exists
timestamp	str	JSON	Capture timestamp (epoch seconds, stringified)
confidence	str	ad_metadata.confidence	'high', 'medium', or 'unknown'
matched_selector	str	ad_metadata.matched_selector	CSS selector that captured this ad
schema_version	str	hardcoded	Detector version (e.g., "3.1") for reproducibility
Key Functions
identify_ad_network(src: str | None) -> str

Classifies an iframe src into a network ID using _COMPILED_PATTERNS.
Returns	When
'none'	src is empty/None (non-iframe ad slots)
'unknown'	src is set but matches no pattern
Network ID	src matches a known pattern
_extract_ocr_text(png_path: Path) -> str

Runs OCR with three layers of defensive handling:

    Returns "" if OCR libraries aren't available
    Returns "" if image is too small (< 40×40 px — usually tracking pixels)
    Returns "" if tesseract raises any exception

Normalizes whitespace with re.sub(r'\s+', ' ', text).strip() — OCR output is notoriously messy with stray newlines.
_process_one_ad(json_path, profile) -> list[ProcessedAd]

Reads one JSON file and produces zero or more ProcessedAd records.

v3.1 schema handling:

    Files starting with _ (e.g., _visit_summary.json) are skipped by filename prefix
    Ad fields live in record["ad_metadata"], not at top level
    Network ID prefers ad_metadata.network (set at capture time) over identify_ad_network(src) (derived from URL)
    Returns a list to handle the legacy case where one file contained multiple ad records — caller should .extend() not .append()

load_ad_artifacts() -> None

Main entry point. Walks every profile's ads/ directory, processes each JSON file, and writes the unified Parquet.
bash

python -m src.ingestion.load_ad_artifacts

Performance: ~0.5 seconds per ad with OCR enabled. For 10,000 ads, expect ~80 minutes. Progress logged every 500 ads.
Internal Variables Reference
Variable	Type	Purpose
all_ads	list[ProcessedAd]	Accumulator across all profiles. Converted to DataFrame at the end.
json_files	list[Path]	Pre-computed via rglob("*.json") so progress total is known upfront.
profile_count	int	Per-profile success count for the summary report.
rate	float	Ads/second processed, computed for progress reports.
init_database.py

Purpose: Create analysis.duckdb with persistent views over the Parquet files. Notebooks can then open it read-only.
Why This Step Exists

DuckDB's read-only mode forbids CREATE statements (including CREATE VIEW). If notebooks tried to create views on each connection, they'd fail. The solution: create views ONCE in write mode, save them into the .duckdb file, then notebooks see them automatically when opening read-only.
Flow

    Check that Parquet files exist in PARQUET_DIR. Exit with help message if not.
    Delete any existing analysis.duckdb for clean rebuild.
    Open a new DuckDB in write mode.
    For each table in OPENWPM_TABLES + ['ads'], create a view via CREATE OR REPLACE VIEW <table> AS SELECT * FROM read_parquet(...).
    Verify each view by querying its row count.
    Close the connection. View definitions persist in the file.

When to Re-Run
Trigger	Re-run init?
Ingestion (Parquet files changed)	Yes
Added a new table to OPENWPM_TABLES	Yes
Renamed/deleted a Parquet file	Yes
Just opened a new notebook	No
Made changes to analysis modules	No
Internal Variables Reference
Variable	Type	Purpose
parquet_files	list[Path]	All .parquet files found in PARQUET_DIR. Used for the sanity-check listing.
registered	int	Count of successfully registered views, for the final summary line.
db.py

Purpose: Centralized DuckDB connection management for all analysis modules.
Why a Central Connection Helper?

Every analysis function needs DB access. Centralizing it means:

    Consistent configuration (memory limit, threads) across the project
    Safe defaults (notebooks should open read-only)
    One place to add cross-cutting features (logging, profiling, etc.)

Key Functions
get_connection(read_only: bool = False) -> DuckDBPyConnection

Returns a configured DuckDB connection.

Behavior:

    If read_only=True and DUCKDB_PATH doesn't exist, auto-runs init_database() first (so first-time notebook users don't need to remember the init step)
    Applies project memory/thread settings via _configure()
    Calls _register_parquet_views(con, read_only=read_only)

Use in scripts — but db_session() is usually better.
db_session(read_only=False) -> Iterator[DuckDBPyConnection]

Context-manager version that guarantees connection cleanup:
python

with db_session(read_only=True) as con:
    df = con.execute("SELECT * FROM http_requests LIMIT 10").df()
# connection automatically closed here

Use in scripts; prefer get_connection() in notebooks where you want one persistent connection across many cells.
_register_parquet_views(con, read_only) -> None

Behavior depends on connection mode:

    Write mode: Runs CREATE OR REPLACE VIEW for each Parquet file. This is how init_database.py persists views.
    Read-only mode: Skips view creation. Just verifies that the expected views exist (warning if not, so user knows to run init).

This split is what resolves the "CREATE on read-only DB" error class.
table_row_counts() -> dict[str, int]

Quick sanity-check helper. Returns {table.profile: row_count} for every table × profile combination. Run this at the top of any analysis notebook to verify your data looks reasonable.
Internal Variables Reference
Variable	Type	Purpose
existing	set[str]	Set of view names actually present in the DB. Used in read-only mode to warn about missing views.
expected	set[str]	Set of view names that should exist (from OPENWPM_TABLES + ['ads']).
missing	set[str]	expected - existing. If non-empty, prints a hint to run init_database.py.
Common Issues & Debugging
"FileNotFoundError: crawl-data.sqlite"

Cause: A profile listed in config.PROFILES doesn't have a SQLite file. Either the crawl failed for that profile, or the directory name doesn't match the profile key.

Fix: Check data/<profile>/ exists and contains crawl-data.sqlite. Profile keys in config.PROFILES must match directory names exactly.
"CatalogException: Table with name X does not exist"

Cause: One profile's SQLite is missing a table that's listed in OPENWPM_TABLES. Often happens for optional tables like callstacks when callstack_instrument was False during the crawl.

Fix: Either enable the instrument and re-crawl, or remove the table from OPENWPM_TABLES. The ingestion script already skips missing tables with a warning — confirm it's not a hard failure.
"'list' object has no attribute 'get'" in _process_one_ad

Cause: Loader is being called on a sidecar file (_visit_summary.json or _ad_content.json) whose JSON shape is a list, not a dict.

Fix: Confirm the file-skipping logic at the top of _process_one_ad checks json_path.name.startswith("_"). Sidecar files should never reach the field-extraction code.
Read-only database errors after re-ingesting

Cause: Old views in analysis.duckdb may point to stale Parquet paths, OR your kernel cached a connection.

Fix:

    Run python scripts/init_database.py to refresh views
    Restart your Jupyter kernel
    Re-run notebook cells

Empty advertiser_network column (everything 'none' or 'unknown')

Cause: Either the ad_capture.py script isn't writing ad_metadata.network (older version) OR identify_ad_network() isn't matching the iframe src patterns.

Fix:

    Inspect a sample JSON: cat data/control/ads/*/*.json | head
    If ad_metadata.network is present, ensure the loader reads from ad_meta.get("network") before falling back to identify_ad_network
    If not present, check AD_NETWORK_PATTERNS against your actual src URLs and add patterns as needed
