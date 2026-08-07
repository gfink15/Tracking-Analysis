# # scripts/run_pixel_analysis.py
# import duckdb
# import sys
# import pandas as pd
# from pathlib import Path


# PROJECT_ROOT = Path(__file__).resolve().parents[1]
# sys.path.insert(0, str(PROJECT_ROOT))

# from config import Categories as c

# from src.analysis.pixels import extract_pixels_from_sqlite
# from src.analysis.ads_pixels_join import (
#     register_pixel_tables,
#     register_seeding_pixels,
#     create_ads_with_pixel_context,
#     register_persona_affinity,
#     create_ads_scored,
#     category_distribution_by_pixel,
#     category_distribution_by_platform,
#     targeting_accuracy_summary,
#     targeting_accuracy_by_platform,
#     seeded_site_impact,
# )

# # 1. Connect to DuckDB with your ads_enriched view already defined
# con = duckdb.connect("artifacts/analysis.duckdb")

# # 2. Build measurement-side pixel tables from http_requests parquet
# register_pixel_tables(
#     con,
#     http_requests_parquet_glob="artifacts/parquet/http_requests.parquet",
# )

# # 3. Build seeding-side pixel table from profile-build SQLite DBs
# seeding_dfs = []
# for sqlite_path in Path("data/persona_profiles").glob("*/crawl-data.sqlite"):
#     persona = sqlite_path.parent.name  # e.g. "shopping", "news", "control"
#     hits = extract_pixels_from_sqlite(sqlite_path, persona=persona)
#     seeding_dfs.append(hits)
# seeding_all = pd.concat(seeding_dfs, ignore_index=True) if seeding_dfs else pd.DataFrame()
# register_seeding_pixels(con, seeding_all)

# # 4. Build the main join view
# create_ads_with_pixel_context(con, min_confidence=0.7)

# # 5. Register your persona-affinity map (from your VLM category list)

# PERSONA_AFFINITY = {
#     "gaming":           [str(c.Electronics.value), str(c.Entertainment.value), str(c.Gaming.value), str(c.Technology.value), str(c.Retail.value), str(c.Software.value), str(c.Crypto.value), str(c.Privacy.value), str(c.Stream.value), str(c.Hobbies.value), str(c.Events.value)],
#     "sports_car_fan":   [str(c.Auto.value), str(c.Construction.value), str(c.Transport.value), str(c.Hobbies.value)],
#     "investor":         [str(c.Finance.value), str(c.Business.value), str(c.Career.value), str(c.Charity.value), str(c.Crypto.value), str(c.Estate.value), str(c.Insurance.value), str(c.Legal.value)],
#     "retiree":          [str(c.Entertainment.value), str(c.Beauty.value), str(c.Govt.value), str(c.Legal.value), str(c.Estate.value), str(c.Travel.value), str(c.Family.value), str(c.Healthcare.value), str(c.Events.value)],
#     "control":  [],  # baseline — no expected affinity
#     # ... fill in from your VLM category list
# }
# register_persona_affinity(con, PERSONA_AFFINITY)
# create_ads_scored(con)

# # 6. Pull the outputs
# out = Path("artifacts/ad_tracker_analysis_outputs")
# out.mkdir(exist_ok=True)

# category_distribution_by_pixel(con).to_csv(out / "category_by_pixel.csv", index=False)
# category_distribution_by_platform(con).to_csv(out / "category_by_platform.csv", index=False)
# targeting_accuracy_summary(con).to_csv(out / "targeting_accuracy.csv", index=False)
# targeting_accuracy_by_platform(con).to_csv(out / "targeting_by_platform.csv", index=False)
# seeded_site_impact(con).to_csv(out / "seeded_site_impact.csv", index=False)

# print("Done. Outputs written to:", out)


# scripts/run_pixel_analysis.py (patched)
import duckdb, sys
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import Categories as c
from src.analysis.pixels import extract_pixels_from_sqlite
from src.analysis.ads_pixels_join import (
    register_pixel_tables, register_seeding_pixels,
    create_ads_with_pixel_context, register_persona_affinity, create_ads_scored,
    category_distribution_by_pixel, category_distribution_by_platform,
    targeting_accuracy_summary, targeting_accuracy_by_platform, seeded_site_impact,
)

con = duckdb.connect("artifacts/analysis.duckdb")
TARGET_PIXELS = [
        "Meta Pixel",
        "DoubleClick",
        "Google Ads Conversion",
        "Criteo", 
        "TikTok Pixel",
        "Pinterest Tag",
        "Snap Pixel",
        "Microsoft UET",
        "X/Twitter Pixel",
        "LinkedIn Insight",
    ]
# FIX #2: use a glob that actually matches per-profile parquets
register_pixel_tables(
    con,
    http_requests_parquet_glob="artifacts/parquet/http_requests_enriched.parquet",
    target_pixels=TARGET_PIXELS,
    require_path_confirmed=True,
    only_third_party=True,
    exclude_technical_3p=False,   # ← CHANGE: was True, now False
)

# --- DIAGNOSTIC BLOCK: figure out where the pixels are being lost ---
print("\n=== Pixel Detection Diagnostic ===\n")

# Step 1: How many raw requests are in the parquet?
n_reqs = con.execute("""
    SELECT COUNT(*) FROM read_parquet(
        'artifacts/parquet/http_requests_enriched.parquet',
        union_by_name=true
    )
""").fetchone()[0] # type: ignore
print(f"1. Total HTTP requests loaded: {n_reqs:,}")

# Step 2: How many pixel classifications occurred (BEFORE any filters)?
tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
if "pixel_hits_raw" not in tables:
    print("❌ pixel_hits_raw missing — register_pixel_tables didn't run properly")
else:
    n_hits = con.execute("SELECT COUNT(*) FROM pixel_hits_raw").fetchone()[0] # type: ignore
    print(f"2. Raw pixel classifications: {n_hits:,}")

    if n_hits > 0:
        # Break down by pixel_type
        print("\n3. Hits by pixel_type:")
        print(con.execute("""
            SELECT pixel_type, COUNT(*) AS n,
                   SUM(CASE WHEN path_confirmed THEN 1 ELSE 0 END) AS n_path_confirmed
            FROM pixel_hits_raw
            GROUP BY pixel_type
            ORDER BY n DESC
        """).df().to_string(index=False))

        # Impact of require_path_confirmed
        n_conf = con.execute("""
            SELECT COUNT(*) FROM pixel_hits_raw WHERE path_confirmed = TRUE
        """).fetchone()[0] # type: ignore
        print(f"\n4. Hits with path_confirmed=TRUE: {n_conf:,} "
              f"({100*n_conf/max(n_hits,1):.1f}%)")

        # Impact of only_third_party (are parent entities populated?)
        n_with_entity = con.execute("""
            SELECT COUNT(*) FROM pixel_hits_raw
            WHERE top_level_parent_entity IS NOT NULL
              AND request_parent_entity IS NOT NULL
        """).fetchone()[0] # type: ignore
        print(f"5. Hits with BOTH parent_entities populated: {n_with_entity:,} "
              f"({100*n_with_entity/max(n_hits,1):.1f}%)")

        n_cross = con.execute("""
            SELECT COUNT(*) FROM pixel_hits_raw
            WHERE top_level_parent_entity IS NOT NULL
              AND request_parent_entity IS NOT NULL
              AND top_level_parent_entity != request_parent_entity
        """).fetchone()[0] # type: ignore
        print(f"6. Hits that are cross-entity (3P): {n_cross:,} "
              f"({100*n_cross/max(n_hits,1):.1f}%)")

        # Impact of exclude_technical_3p
        if "is_technical_3p" in con.execute(
            "DESCRIBE pixel_hits_raw"
        ).df()["column_name"].values:
            n_nontech = con.execute("""
                SELECT COUNT(*) FROM pixel_hits_raw
                WHERE COALESCE(is_technical_3p, 0) < 0.5
            """).fetchone()[0] # type: ignore
            print(f"7. Non-technical 3P hits: {n_nontech:,} "
                  f"({100*n_nontech/max(n_hits,1):.1f}%)")

print("\n=== End Diagnostic ===\n")

# FIX #3: warn if seeding data is missing
seeding_dfs = []
for sqlite_path in Path("data/persona_profiles").glob("*/crawl-data.sqlite"):
    persona = sqlite_path.parent.name
    seeding_dfs.append(extract_pixels_from_sqlite(sqlite_path, persona=persona))

if not seeding_dfs:
    print("⚠️  No seeding SQLite DBs found — seeded-site analysis will be empty.")
seeding_all = pd.concat(seeding_dfs, ignore_index=True) if seeding_dfs else pd.DataFrame()
register_seeding_pixels(
    con, seeding_all,
    target_pixels=TARGET_PIXELS,   # same list as above
    require_path_confirmed=True,
    # (no exclude_technical_3p here — the function doesn't take it,
    # but if you add it later, keep it False)
)
# Which pixel platforms is is_technical_3p flagging as "technical"?
# If Meta/DoubleClick/etc. all show >95% technical, the flag is broken
# for our purposes and dropping it is the right call.
print("\n=== Is is_technical_3p reliable for ad-pixel filtering? ===")
print(con.execute("""
    SELECT
        pixel_type,
        COUNT(*) AS total_hits,
        SUM(CASE WHEN COALESCE(is_technical_3p, 0) >= 0.5 THEN 1 ELSE 0 END)
            AS flagged_technical,
        ROUND(100.0 * AVG(CASE WHEN COALESCE(is_technical_3p, 0) >= 0.5
                              THEN 1.0 ELSE 0.0 END), 1)
            AS pct_technical
    FROM pixel_hits_raw
    GROUP BY pixel_type
    ORDER BY total_hits DESC
""").df().to_string(index=False))
# --- Confirm what site_pixels actually contains ---
print("\n=== What site_pixels actually contains ===")
sp_summary = con.execute("""
    SELECT
        profile,
        COUNT(*) AS n_entities,
        SUM(CASE WHEN has_pixel THEN 1 ELSE 0 END) AS n_with_pixel,
        ROUND(100.0 * AVG(CASE WHEN has_pixel THEN 1.0 ELSE 0.0 END), 1)
            AS pct_with_pixel,
        SUM(n_pixel_hits) AS total_hits_aggregated
    FROM site_pixels
    GROUP BY profile
    ORDER BY profile
""").df()
print(sp_summary.to_string(index=False))

print("\nTop entities by pixel hits (sanity check):")
print(con.execute("""
    SELECT top_level_parent_entity, n_pixel_hits, n_distinct_pixel_types,
           pixel_types
    FROM site_pixels
    ORDER BY n_pixel_hits DESC
    LIMIT 10
""").df().to_string(index=False))
# Are non-technical requests concentrated in NON-pixel domains?
print(con.execute("""
    WITH sample AS (
        SELECT domain, parent_entity, is_technical_3p, COUNT(*) AS n_reqs
        FROM read_parquet(
            'artifacts/parquet/http_requests_enriched.parquet',
            union_by_name=true
        )
        WHERE is_technical_3p < 0.5
        GROUP BY domain, parent_entity, is_technical_3p
        ORDER BY n_reqs DESC
        LIMIT 20
    )
    SELECT * FROM sample
""").df().to_string(index=False))
# FIX #4: categorical confidence (adjust if create_ads_with_pixel_context expects float)
create_ads_with_pixel_context(con)

PERSONA_AFFINITY = {
    "gaming":         
[c.Electronics.value, c.Entertainment.value, c.Gaming.value,
                       c.Technology.value, c.Retail.value, c.Software.value,
                       c.Crypto.value, c.Privacy.value, c.Stream.value,
                       c.Hobbies.value, c.Events.value],
    "sports_car_fan": 
[c.Auto.value, c.Construction.value, c.Transport.value, c.Hobbies.value],
    "investor":       
[c.Finance.value, c.Business.value, c.Career.value, c.Charity.value,
                       c.Crypto.value, c.Estate.value, c.Insurance.value, c.Legal.value],
    "retiree":        
[c.Entertainment.value, c.Beauty.value, c.Govt.value, c.Legal.value,
                       c.Estate.value, c.Travel.value, c.Family.value,
                       c.Healthcare.value, c.Events.value],
    "control":        
[],
}
register_persona_affinity(con, PERSONA_AFFINITY)
create_ads_scored(con)

# FIX #1: parents=True
out = Path("artifacts/ad_tracker_analysis_outputs")
out.mkdir(parents=True, exist_ok=True)

category_distribution_by_pixel(con).to_csv(out / "category_by_pixel.csv", index=False)
category_distribution_by_platform(con).to_csv(out / "category_by_platform.csv", index=False)
targeting_accuracy_summary(con).to_csv(out / "targeting_accuracy.csv", index=False)
targeting_accuracy_by_platform(con).to_csv(out / "targeting_by_platform.csv", index=False)
seeded_site_impact(con).to_csv(out / "seeded_site_impact.csv", index=False)

print("Done. Outputs written to:", out)