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
import sys
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

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
        read_only: If True, opens the DB in read-only mode (recommended
            for notebooks). Will auto-initialize the database file
            via scripts/init_database.py if it doesn't yet exist.
    """
    # Auto-initialize on first read-only access. This makes notebooks
    # "just work" without forcing users to remember the init step.
    if read_only and not DUCKDB_PATH.exists():
        print(f"⚠  {DUCKDB_PATH.name} not found — auto-initializing...")
        # Local import avoids circular dependency at module load.
        import importlib.util
        init_script = PROJECT_ROOT / "scripts" / "init_database.py"
        spec = importlib.util.spec_from_file_location(
            "init_database", init_script
        )
        module = importlib.util.module_from_spec(spec) # type: ignore
        spec.loader.exec_module(module) # type: ignore
        module.init_database()

    con = duckdb.connect(str(DUCKDB_PATH), read_only=read_only)
    _configure(con)
    _register_parquet_views(con, read_only=read_only)   # pass the flag!
    return con


@contextmanager
def db_session(read_only: bool = False) -> Iterator[duckdb.DuckDBPyConnection]:
    """Context-manager version of get_connection() — see that function
    for argument documentation."""
    con = get_connection(read_only=read_only)
    try:
        yield con
    finally:
        con.close()


def _register_parquet_views(
    con: duckdb.DuckDBPyConnection,
    read_only: bool = False,
) -> None:
    """Register each Parquet file in PARQUET_DIR as a queryable view.

    Behavior depends on connection mode:
      • Write mode: runs CREATE OR REPLACE VIEW to (re)register every
        Parquet file. This persists the view definitions into the
        .duckdb file for future read-only connections.
      • Read-only mode: skips registration entirely. The views are
        assumed to already exist (persisted by a prior write-mode
        connection, typically via scripts/init_database.py).

    This split is what resolves the 'CREATE statement on read-only
    database' error: the views are created ONCE in write mode and
    then simply queried in read-only mode.
    """
    if read_only:
        # Verify the expected views actually exist, fail fast if not.
        existing = {
            row[0] for row in con.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_type = 'VIEW'"
            ).fetchall()
        }
        expected = set(OPENWPM_TABLES + ['ads'])
        missing = expected - existing
        if missing:
            # Don't crash — some tables might legitimately be absent
            # (e.g., 'ads' before you've run ad ingestion). Just warn.
            print(f"⚠  Read-only connection: views not yet registered "
                  f"for {sorted(missing)}.")
            print(f"   Run: python scripts/init_database.py")
        return

    # Write mode: register/refresh all available views.
    for table in OPENWPM_TABLES + ['ads']:
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