"""
src/utils/db.py — DuckDB connection helpers.

Provides a single function to get a properly-configured DuckDB
connection, plus utilities for registering Parquet files as views.

Why DuckDB?
  - Reads Parquet natively at near-disk speed
  - SQL is the most concise language for the joins and aggregations
    we'll need (tracker prevalence, cross-profile diffs, etc.)
  - Embeds in-process (no server to manage)
  - Handles tens of GB on a laptop without breaking a sweat
"""
from __future__ import annotations

import duckdb
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

# Import from the project root config. The relative-import pattern here
# assumes the project is run with the project root on PYTHONPATH (which
# we'll set up via pyproject.toml or a simple sys.path tweak in scripts).
from config import (
    DUCKDB_PATH,
    PARQUET_DIR,
    DUCKDB_MEMORY_LIMIT,
    DUCKDB_THREADS,
    OPENWPM_TABLES,
)


def _configure(con: duckdb.DuckDBPyConnection) -> None:
    """Apply project-wide DuckDB settings to a fresh connection.

    Pulled into a private function so both get_connection() and the
    context manager apply the exact same configuration. DRY matters
    here — silently inconsistent DB configs cause reproducibility bugs
    that are nearly impossible to track down later.
    """
    con.execute(f"SET memory_limit = '{DUCKDB_MEMORY_LIMIT}'")
    con.execute(f"SET threads = {DUCKDB_THREADS}")
    # Enable progress bar for long-running queries — invaluable during
    # interactive analysis when a query might take 30+ seconds.
    con.execute("SET enable_progress_bar = true")


def get_connection(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """Return a configured DuckDB connection to the project database.

    Args:
        read_only: If True, opens the DB in read-only mode. Use this
            in analysis notebooks to prevent accidental writes that
            could corrupt your analysis state.

    Returns:
        A DuckDB connection with project settings applied and all
        Parquet files in PARQUET_DIR registered as views.

    Why register Parquet as views?
        It lets you write `SELECT * FROM http_requests` instead of
        `SELECT * FROM 'artifacts/parquet/http_requests.parquet'`
        in every query. Big readability win.
    """
    con = duckdb.connect(str(DUCKDB_PATH), read_only=read_only)
    _configure(con)
    _register_parquet_views(con)
    return con


@contextmanager
def db_session(read_only: bool = False) -> Iterator[duckdb.DuckDBPyConnection]:
    """Context manager version of get_connection().

    Use this in scripts to guarantee the connection is closed even
    if an exception occurs:

        with db_session() as con:
            df = con.execute("SELECT * FROM http_requests LIMIT 10").df()

    In notebooks, get_connection() is often more convenient since you
    want the connection to persist across cells.
    """
    con = get_connection(read_only=read_only)
    try:
        yield con
    finally:
        con.close()


def _register_parquet_views(con: duckdb.DuckDBPyConnection) -> None:
    """Register each Parquet file in PARQUET_DIR as a queryable view.

    A 'view' in DuckDB is just a named SELECT statement — it doesn't
    materialize the data. So this is essentially free and lets queries
    reference table names directly.

    We use CREATE OR REPLACE so this is idempotent: running it multiple
    times in the same connection won't error.
    """
    for table in OPENWPM_TABLES + ['ads']:  # 'ads' added by enrich_ads.py
        parquet_path = PARQUET_DIR / f"{table}.parquet"
        if parquet_path.exists():
            con.execute(f"""
                CREATE OR REPLACE VIEW {table} AS
                SELECT * FROM read_parquet('{parquet_path}')
            """)


def table_row_counts() -> dict[str, int]:
    """Quick sanity-check helper: returns row counts per table per profile.

    Useful as the first cell of every analysis notebook to confirm
    you're working with the data you think you're working with.
    Catches the classic 'oops, I only ingested 1 of 4 profiles' bug.
    """
    with db_session(read_only=True) as con:
        counts: dict[str, int] = {}
        for table in OPENWPM_TABLES + ['ads']:
            try:
                result = con.execute(
                    f"SELECT profile, COUNT(*) AS n FROM {table} GROUP BY profile"
                ).fetchall()
                for profile, n in result:
                    counts[f"{table}.{profile}"] = n
            except duckdb.CatalogException:
                # Table doesn't exist yet (ingestion not run)
                counts[table] = 0
        return counts


if __name__ == "__main__":
    # Running this file directly prints a status report. Handy for
    # 'is my ingestion done?' checks from the command line.
    from pprint import pprint
    pprint(table_row_counts())