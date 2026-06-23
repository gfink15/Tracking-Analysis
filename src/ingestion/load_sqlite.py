"""
src/ingestion/load_sqlite.py — Convert OpenWPM SQLite outputs to Parquet.

For each table in OPENWPM_TABLES, this script:
  1. Attaches every per-profile SQLite database
  2. Unions the table contents across profiles, tagging each row with
     its profile of origin
  3. Writes the combined result to a single Parquet file

After running this once, all downstream analysis queries the Parquet
files via DuckDB views (set up in src/utils/db.py) — never the raw
SQLite DBs again. This is the key performance win: Parquet is
columnar, compressed, and dramatically faster for analytical queries.

Run with:
    python -m src.ingestion.load_sqlite
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import duckdb

# Add project root to sys.path so `from config import ...` works
# regardless of where this script is invoked from. A simple
# alternative to packaging the project with pyproject.toml.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    DATA_DIR,
    PARQUET_DIR,
    PROFILES,
    OPENWPM_TABLES,
    DUCKDB_MEMORY_LIMIT,
    DUCKDB_THREADS,
)


def _verify_sqlite_files_exist() -> dict[str, Path]:
    """Confirm every profile has a crawl-data.sqlite file before we start.

    Failing fast here is much better than failing 20 minutes into
    ingestion because one profile's crawl never finished.

    Returns:
        Mapping of profile name → path to its SQLite file.

    Raises:
        FileNotFoundError if any profile's SQLite is missing.
    """
    paths = {}
    missing = []
    for profile in PROFILES:
        sqlite_path = DATA_DIR / profile / "crawl-data.sqlite"
        if sqlite_path.exists():
            paths[profile] = sqlite_path
        else:
            missing.append(str(sqlite_path))
    if missing:
        raise FileNotFoundError(
            "Missing crawl SQLite files for these profiles:\n  "
            + "\n  ".join(missing)
        )
    return paths


def _get_table_columns(
    con: duckdb.DuckDBPyConnection,
    profile: str,
    table: str,
) -> list[str]:
    """Get column names for a table in a specific attached SQLite DB.

    We need this because OpenWPM versions occasionally add or rename
    columns. If profile A's crawl ran on an older version than
    profile B's, the schemas can differ slightly. We use the column
    list to construct an explicit SELECT, which makes mismatches
    surface as clear errors instead of silently corrupting data.
    """
    result = con.execute(f"""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_catalog = 'db_{profile}' AND table_name = '{table}'
        ORDER BY ordinal_position
    """).fetchall()
    return [row[0] for row in result]


def _convert_table(
    con: duckdb.DuckDBPyConnection,
    table: str,
    sqlite_paths: dict[str, Path],
) -> int:
    """Convert one OpenWPM table across all profiles into one Parquet file.

    Strategy:
      1. For each profile, build a SELECT that pulls all columns and
         adds a literal 'profile' column.
      2. UNION ALL the per-profile SELECTs (UNION ALL preserves
         duplicates and is much faster than UNION).
      3. COPY the result to Parquet with ZSTD compression.

    Returns:
        Number of rows written.
    """
    # Find the intersection of columns across all profiles. This handles
    # the schema-drift case gracefully: we only export columns that
    # exist in every profile's table. Columns unique to one profile are
    # dropped with a warning — better than crashing the whole pipeline.
    column_sets = {
        profile: set(_get_table_columns(con, profile, table))
        for profile in sqlite_paths
    }
    common_cols = set.intersection(*column_sets.values())
    if not common_cols:
        print(f"  ⚠  {table}: no common columns across profiles, skipping")
        return 0

    # Warn about dropped columns so schema drift doesn't go unnoticed.
    for profile, cols in column_sets.items():
        dropped = cols - common_cols
        if dropped:
            print(f"  ⚠  {table}/{profile}: dropping columns {dropped}")

    # Sort for deterministic column order in the output Parquet —
    # important for reproducibility (diff-able outputs).
    cols_sql = ", ".join(sorted(common_cols))

    # Build the UNION ALL across profiles.
    union_parts = [
        f"SELECT {cols_sql}, '{profile}' AS profile "
        f"FROM db_{profile}.{table}"
        for profile in sqlite_paths
    ]
    union_sql = " UNION ALL ".join(union_parts)

    output_path = PARQUET_DIR / f"{table}.parquet"

    # COPY ... TO writes directly to disk in a streaming fashion,
    # so memory usage stays bounded even for huge tables.
    con.execute(f"""
        COPY ({union_sql}) TO '{output_path}'
        (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
    """)

    # Count rows for the progress report. This is a cheap query
    # against the Parquet we just wrote.
    result = con.execute(f"SELECT COUNT(*) FROM read_parquet('{output_path}')").fetchone()
    n_rows = result[0] if result else 0
    return n_rows


def sqlite_to_parquet() -> None:
    """Main entry point: convert all OpenWPM tables across all profiles."""
    print("─" * 60)
    print("OpenWPM SQLite → Parquet ingestion")
    print("─" * 60)

    # Step 1: verify inputs before doing any work.
    sqlite_paths = _verify_sqlite_files_exist()
    print(f"Found SQLite files for {len(sqlite_paths)} profiles: "
          f"{list(sqlite_paths.keys())}")

    # Step 2: open a fresh in-memory DuckDB (we don't need to persist
    # the ingestion connection — the output is the Parquet files).
    con = duckdb.connect(":memory:")
    con.execute(f"SET memory_limit = '{DUCKDB_MEMORY_LIMIT}'")
    con.execute(f"SET threads = {DUCKDB_THREADS}")

    # Step 3: attach all SQLite databases. ATTACH gives DuckDB query
    # access to SQLite as if it were a DuckDB schema.
    for profile, path in sqlite_paths.items():
        con.execute(
            f"ATTACH '{path}' AS db_{profile} (TYPE SQLITE, READ_ONLY)"
        )
        print(f"  ✓ Attached {profile}: {path}")

    # Step 4: convert each table, with timing for the progress report.
    print("\nConverting tables:")
    total_rows = 0
    for table in OPENWPM_TABLES:
        start = time.time()
        try:
            n_rows = _convert_table(con, table, sqlite_paths)
            elapsed = time.time() - start
            print(f"  ✓ {table:25s} {n_rows:>12,} rows  ({elapsed:5.1f}s)")
            total_rows += n_rows
        except duckdb.CatalogException as e:
            # A table is missing from at least one SQLite — log and continue.
            # This happens with optional OpenWPM features (e.g., callstacks
            # is only populated if instrumentation was enabled).
            print(f"  ⚠  {table}: skipped ({e})")

    print("─" * 60)
    print(f"Total: {total_rows:,} rows written to {PARQUET_DIR}")
    print("─" * 60)

    con.close()


if __name__ == "__main__":
    sqlite_to_parquet()