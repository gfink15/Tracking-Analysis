# scripts/run_pixel_analysis.py
import duckdb
import sys
import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import Categories as c

from src.analysis.pixels import extract_pixels_from_sqlite
from src.analysis.ads_pixels_join import (
    register_pixel_tables,
    register_seeding_pixels,
    create_ads_with_pixel_context,
    register_persona_affinity,
    create_ads_scored,
    category_distribution_by_pixel,
    category_distribution_by_platform,
    targeting_accuracy_summary,
    targeting_accuracy_by_platform,
    seeded_site_impact,
)

# 1. Connect to DuckDB with your ads_enriched view already defined
con = duckdb.connect("artifacts/analysis.duckdb")

# 2. Build measurement-side pixel tables from http_requests parquet
register_pixel_tables(
    con,
    http_requests_parquet_glob="artifacts/parquet/http_requests.parquet",
)

# 3. Build seeding-side pixel table from profile-build SQLite DBs
seeding_dfs = []
for sqlite_path in Path("data/persona_profiles").glob("*/crawl-data.sqlite"):
    persona = sqlite_path.parent.name  # e.g. "shopping", "news", "control"
    hits = extract_pixels_from_sqlite(sqlite_path, persona=persona)
    seeding_dfs.append(hits)
seeding_all = pd.concat(seeding_dfs, ignore_index=True) if seeding_dfs else pd.DataFrame()
register_seeding_pixels(con, seeding_all)

# 4. Build the main join view
create_ads_with_pixel_context(con, min_confidence=0.7)

# 5. Register your persona-affinity map (from your VLM category list)

PERSONA_AFFINITY = {
    "gaming":           [str(c.Electronics.value), str(c.Entertainment.value), str(c.Gaming.value), str(c.Technology.value), str(c.Retail.value), str(c.Software.value), str(c.Crypto.value), str(c.Privacy.value), str(c.Stream.value), str(c.Hobbies.value), str(c.Events.value)],
    "sports_car_fan":   [str(c.Auto.value), str(c.Construction.value), str(c.Transport.value), str(c.Hobbies.value)],
    "investor":         [str(c.Finance.value), str(c.Business.value), str(c.Career.value), str(c.Charity.value), str(c.Crypto.value), str(c.Estate.value), str(c.Insurance.value), str(c.Legal.value)],
    "retiree":          [str(c.Entertainment.value), str(c.Beauty.value), str(c.Govt.value), str(c.Legal.value), str(c.Estate.value), str(c.Travel.value), str(c.Family.value), str(c.Healthcare.value), str(c.Events.value)],
    "control":  [],  # baseline — no expected affinity
    # ... fill in from your VLM category list
}
register_persona_affinity(con, PERSONA_AFFINITY)
create_ads_scored(con)

# 6. Pull the outputs
out = Path("artifacts/ad_tracker_analysis_outputs")
out.mkdir(exist_ok=True)

category_distribution_by_pixel(con).to_csv(out / "category_by_pixel.csv", index=False)
category_distribution_by_platform(con).to_csv(out / "category_by_platform.csv", index=False)
targeting_accuracy_summary(con).to_csv(out / "targeting_accuracy.csv", index=False)
targeting_accuracy_by_platform(con).to_csv(out / "targeting_by_platform.csv", index=False)
seeded_site_impact(con).to_csv(out / "seeded_site_impact.csv", index=False)

print("Done. Outputs written to:", out)