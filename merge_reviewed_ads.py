# merge_reviewed_ads.py
# Merges two lab partners' reviewed parquet files into a single gold-standard dataset.
# Run with: python merge_reviewed_ads.py

import pandas as pd
from pathlib import Path
import sys

# ---- CONFIG ----
# Update these paths to match wherever you and your partner saved your files
PARTNER_A_FILE = Path("artifacts/parquet/ads_desc_reviewed_graham.parquet")
PARTNER_B_FILE = Path("artifacts/parquet/ads_desc_reviewed_anya.parquet")
OUTPUT_FILE = Path("artifacts/parquet/ads_desc.parquet")

def merge_reviewed_ads():
    # ---- 1. Validate inputs ----
    for f in [PARTNER_A_FILE, PARTNER_B_FILE]:
        if not f.exists():
            print(f"❌ File not found: {f}")
            sys.exit(1)
    
    print(f"Loading {PARTNER_A_FILE.name}...")
    df_a = pd.read_parquet(PARTNER_A_FILE)
    print(f"  → {len(df_a)} total rows")
    
    print(f"Loading {PARTNER_B_FILE.name}...")
    df_b = pd.read_parquet(PARTNER_B_FILE)
    print(f"  → {len(df_b)} total rows")

    # ---- 2. Coerce the 'viewed' column to real bools ----
    # Your reviewer app had some early runs where 'viewed' got saved as strings.
    # This defensive coercion handles both cases cleanly.
    for df in [df_a, df_b]:
        df["viewed"] = (
            df["viewed"]
            .replace({"True": True, "False": False, "": False})
            .fillna(False)
            .astype(bool)
        )

    # ---- 3. Filter to only viewed ads BEFORE concatenation ----
    # Doing this per-partner first gives clearer progress reporting
    df_a_viewed = df_a[df_a["viewed"]].copy()
    df_b_viewed = df_b[df_b["viewed"]].copy()
    
    print(f"\nPartner A reviewed: {len(df_a_viewed)} / {len(df_a)} ads")
    print(f"Partner B reviewed: {len(df_b_viewed)} / {len(df_b)} ads")

    # ---- 5. Concatenate ----
    combined = pd.concat([df_a_viewed, df_b_viewed], ignore_index=True)
    print(f"\nCombined viewed rows: {len(combined)}")

    # ---- 6. Handle overlap: prefer the 'modified' row if the same index appears twice ----
    # If you split by profile there shouldn't be overlap, but this handles it safely.
    if "modified" in combined.columns:
        combined["modified"] = combined["modified"].fillna(False).astype(bool)
        # Sort so that modified=True rows come first; drop_duplicates keeps the first occurrence
        combined = combined.sort_values(by="modified", ascending=False)
    
    duplicates_found = combined.duplicated(subset=["index"]).sum()
    if duplicates_found > 0:
        print(f"⚠️  Found {duplicates_found} overlapping ads (both partners reviewed). Keeping the modified version.")
    
    final_df = combined.drop_duplicates(subset=["index"], keep="first")

    # ---- 7. Reset index for a clean, unique integer index ----
    final_df = final_df.reset_index(drop=True)
    final_df.sort_values(by="index", inplace=True)

    # ---- 8. Save ----
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_parquet(OUTPUT_FILE, index=False)
    
    print(f"\n✅ Merged {len(final_df)} unique reviewed ads")
    print(f"   → Saved to {OUTPUT_FILE}")
    
    # ---- 9. Quick summary stats ----
    if "category" in final_df.columns:
        print(f"\nCategory distribution:")
        print(final_df["category"].value_counts().to_string())
    
    if "modified" in final_df.columns:
        n_modified = final_df["modified"].sum()
        print(f"\nAds with human corrections: {n_modified} ({100*n_modified/len(final_df):.1f}%)")

if __name__ == "__main__":
    merge_reviewed_ads()