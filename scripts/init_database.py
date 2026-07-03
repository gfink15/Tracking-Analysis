"""
scripts/init_database.py — One-time initialization of analysis.duckdb.

Run this AFTER ingestion completes and BEFORE opening any notebook.
Creates the DuckDB file and persists Parquet-backed views into it
so notebooks can connect read-only without errors.

Usage:
    python scripts/init_database.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import duckdb
from config import (
    DUCKDB_PATH,
    PARQUET_DIR,
    OPENWPM_TABLES,
    DUCKDB_MEMORY_LIMIT,
    DUCKDB_THREADS,
)


def init_database() -> None:
    """Create analysis.duckdb with persistent views over Parquet files."""
    print("─" * 60)
    print("Initializing analysis.duckdb")
    print("─" * 60)

    # Sanity check: do we even have Parquet files to view?
    parquet_files = list(PARQUET_DIR.glob("*.parquet"))
    if not parquet_files:
        print(f"⚠  No Parquet files found in {PARQUET_DIR}")
        print("   Run ingestion first:")
        print("     python -m src.ingestion.load_sqlite")
        print("     python -m src.ingestion.load_ad_artifacts")
        sys.exit(1)

    print(f"Found {len(parquet_files)} Parquet file(s):")
    for f in parquet_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  • {f.name:30s} ({size_mb:7.1f} MB)")

    # Delete any existing database so we start clean. This avoids
    # stale views pointing at Parquet files that have been replaced.
    # If you want to preserve existing custom tables in the DB,
    # comment this out and use CREATE OR REPLACE VIEW below.
    if DUCKDB_PATH.exists():
        print(f"\nRemoving existing {DUCKDB_PATH.name} for clean rebuild...")
        DUCKDB_PATH.unlink()

    # Connect WITHOUT read_only so the file gets created.
    con = duckdb.connect(str(DUCKDB_PATH))
    con.execute(f"SET memory_limit = '{DUCKDB_MEMORY_LIMIT}'")
    con.execute(f"SET threads = {DUCKDB_THREADS}")

    # Register each Parquet file as a persistent view.
    # The view definition is saved IN the .duckdb file, so notebooks
    # opening read-only will see them automatically.
    print("\nRegistering views:")
    registered = 0
    for table in OPENWPM_TABLES + ['ads']:
        parquet_path = PARQUET_DIR / f"{table}.parquet"
        if not parquet_path.exists():
            print(f"  ⚠  {table}: parquet file not found, skipping")
            continue
        con.execute(f"""
            CREATE OR REPLACE VIEW {table} AS
            SELECT * FROM read_parquet('{parquet_path}')
        """)
        # Verify the view works by querying row count.
        nn = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        n = nn[0] if nn else 0
        print(f"  ✓ {table:25s} ({n:>10,} rows)")
        registered += 1

    # ─────────────────────────────────────────────────────────────
    # VLM ad descriptions (from ad_desc.parquet)
    # ─────────────────────────────────────────────────────────────
    ad_desc_path = PARQUET_DIR / "ad_desc.parquet"
    if ad_desc_path.exists():
        con.execute(f"""
            CREATE OR REPLACE VIEW ads_desc AS 
            SELECT * FROM read_parquet('{ad_desc_path}')
        """)
        print(f"✓ Registered ads_desc view ({ad_desc_path.name})")
    else:
        print(f"⚠  {ad_desc_path.name} not found — skipping ads_desc registration")


    con.execute(f"""
        CREATE OR REPLACE VIEW ads_full AS
        SELECT 
            a.*,
            d.is_valid_ad,
            d.primary_product_or_service,
            d.advertiser_brand,
            d.visual_description,
            d.text_content AS vlm_text,
            d.confidence AS vlm_confidence
        FROM read_parquet('{PARQUET_DIR}/ads.parquet') a
        LEFT JOIN read_parquet('{PARQUET_DIR}/ad_desc.parquet') d
        USING (ad_hash)
    """)

    con.close()
    print("\n" + "─" * 60)
    print(f"✓ Initialized {DUCKDB_PATH}")
    print(f"  {registered} view(s) registered")
    print("─" * 60)


if __name__ == "__main__":
    init_database()