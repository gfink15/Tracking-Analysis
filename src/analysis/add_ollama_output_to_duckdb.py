"""
ad_clip_ollama.py does not automatically add its processed data to duckdb so that happens here now
So that the data can be used with the default connection and duckdb

"""


import duckdb
import pandas as pd
from pathlib import Path

from config import PARQUET_DIR
df = pd.read_parquet(Path(f"{PARQUET_DIR}/ad_desc.parquet"))

con = duckdb.connect(":memory:")
con.register('ads_desc_df', df)

con.close()