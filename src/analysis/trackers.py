import duckdb
import pandas as pd
from config import PARQUET_DIR, DUCKDB_PATH
from src.utils.tracker_lists import load_disconnect_list

def tracker_prevalence_by_profile() -> pd.DataFrame:
    """Per-profile count of unique tracker domains contacted."""
    con = duckdb.connect(str(DUCKDB_PATH))
    trackers = load_disconnect_list()
    
    return con.execute(f"""
        WITH req AS (
            SELECT profile, visit_id,
                   regexp_extract(url, '://([^/]+)', 1) AS host
            FROM '{PARQUET_DIR}/http_requests.parquet'
        )
        SELECT profile,
               COUNT(DISTINCT host) AS unique_hosts,
               COUNT(DISTINCT CASE WHEN host IN {tuple(trackers)} 
                                   THEN host END) AS unique_trackers
        FROM req
        GROUP BY profile
        ORDER BY profile
    """).df()

def differential_trackers(profile_a: str, profile_b: str) -> pd.DataFrame:
    """Trackers appearing in profile_a but not profile_b, per site."""
    # ...