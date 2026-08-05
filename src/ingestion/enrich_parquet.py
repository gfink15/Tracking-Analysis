"""
Script: /src/ingestion/enrich_parquet.py

Author: Anya Barringer, aided by Claude Sonnet 4.6 and
        Codestral through Furman University BoodleBox

Container:  Part of CSC Summer Research 2026 Project
            "Pervasive Online Third-Party Tracking: A Measurement Study"
            with Graham Fink, under Dr. Rebecca Drucker

Goal:   Enriches crawl data in TARGET parquet files to add domain, entity,
        and relationship classification columns, allowing for more accurate
        analysis of third-party tracking. Uses domain-entity mapping tree
        and helper utility functions from openwpm-tracker-analysis module.
        - Output files TARGET_enriched.parquet will be called by later scripts
        eg trackers.py. 
        - If other OpenWPM tables need enriching, simply add to ENRICHED_TABLES in config.py
        and to ENRICHMENT_TARGETS in enrich_parquet.py before running.
        - Including this step in separate script allows for clean separation of classification
        and analysis steps. Run once after load_sqlite.py and before init_database.py
        as part of data cleaning step (Silver layer).
"""



import pandas as pd
from pathlib import Path
from types import SimpleNamespace
from config import TREE_CSV_PATH, PARQUET_DIR
from src.utils.domain_utils import load_tree, get_node_info, get_registered_domain, classify_relationship
import pyarrow.parquet as pq
import pyarrow as pa


BATCH_SIZE = 50_000

# Configuration for tables to be enriched
# Update corresponding constant ENRICHED_TABLES in config.py
ENRICHMENT_TARGETS = {
    "http_requests": {
        "input_file": "http_requests.parquet",
        "output_file": "http_requests_enriched.parquet",
        "domain_column": "url",
        "context_column": "top_level_url"
    },
    "javascript_cookies": {
        "input_file": "javascript_cookies.parquet",
        "output_file": "javascript_cookies_enriched.parquet",
        "domain_column": "host",
        "context_column": None
    },
    # "javascript": {
    #     "input_file": "javascript.parquet",
    #     "output_file": "javascript_enriched.parquet",
    #     "domain_column": "script_url",
    #     "context_column": "top_level_url"
    # }
}


def build_visit_map(site_visits_df: pd.DataFrame) -> dict[int, str]:
    """
    Creates lookup dictionary mapping visit_id to registered domain
    of site_url. Used as ground-truth fallback for classification.
    
    Args:
        site_visits_df: DataFrame loaded from site_visits.parquet
        
    Returns:
        dict: {visit_id: registered_domain_string}
    """
    print("[build_visit_map] Building visit-level domain fallback map...")
    visit_map = {}
    for _, row in site_visits_df.iterrows():
        v_id = row['visit_id']
        site_url = str(row['site_url'])
        domain = get_registered_domain(site_url)
        visit_map[v_id] = domain
    
    print(f"[build_visit_map] Map built for {len(visit_map)} unique visits.")
    return visit_map


def enrich_row(
    row: pd.Series, 
    domain_to_node: dict, 
    visit_map: dict, 
    domain_column: str, 
    context_column: str
) -> pd.Series:
    """
    Enriches a single row with entity and relationship data.
    
    Logic:
    1. Extracts registered domain from specified domain_column.
    2. Extracts top-level domain from context_column with visit_map fallback.
    3. Resolves both to entity nodes via mapping tree.
    4. Classifies relationship tier (First-party, Inter-family, or External).

    Args:
        row: individual DataFrame row
        domain_to_node: mapping dictionary from load_tree() for easy lookup
        visit_map: site_visit dictionary for unknown domain fallback
        domain_column: column name to use for request domain (e.g., 'url' or 'host')
        context_column: column name to use for top-level context (e.g., 'top_level_url'),
                        can be None if no context column exists for the table

    Returns:
        pd.Series: Enriched row with domain, subsidiary_entity,
                   parent_entity, and relationship_tier classification.
    """
    # --- 1. Resolve Request Domain ---
    # Use dynamic domain_column (e.g., 'url' for requests, 'host' for cookies)
    req_domain = get_registered_domain(row[domain_column])
    
    # --- 2. Resolve Top-Level Domain (with fallback) ---
    req_top_domain = None
    
    # Only attempt context_column resolution if it is provided and not null in row
    if context_column and context_column in row and pd.notna(row[context_column]):
        req_top_domain = get_registered_domain(row[context_column])
    
    # Fallback to visit_map if context_column is None (cookies) or extraction failed
    if not req_top_domain:
        req_top_domain = visit_map.get(row['visit_id'])
        
    # --- 3. Guard for Unknown/Unresolvable Context ---
    if not req_domain or not req_top_domain:
        return pd.Series({
            'domain': req_domain if req_domain else "",
            'subsidiary_entity': "",
            'parent_entity': "",
            'relationship_tier': "unknown",
            'is_technical_3p': None
        })
        
    # --- 4. Resolve Entity Nodes ---
    req_node = get_node_info(req_domain, domain_to_node)
    top_node = get_node_info(req_top_domain, domain_to_node)
    
    # --- 5. Classify Relationship ---
    # Returns dict: {is_technical_3p, is_subsidiary_3p, is_parent_3p}
    flags = classify_relationship(req_node, top_node)
    
    # --- 6. Determine Tier (Priority: Parent > Subsidiary > First-party) ---
    if flags['is_parent_third_party']:
        tier = "external third-party"
    elif flags['is_subsidiary_third_party']:
        tier = "inter-family third-party"
    else:
        tier = "first-party"
        
    return pd.Series({
        'domain': req_domain,
        'subsidiary_entity': req_node.subsidiary_entity,
        'parent_entity': req_node.parent_entity,
        'relationship_tier': tier,
        'is_technical_3p': int(flags['is_technical_third_party'])
    })


def enrich_table(target: dict, domain_to_node: dict, visit_map: dict) -> None:
    """
    Orchestrates the enrichment pipeline for a single target table.

    Loads input parquet file, applies row-level enrichment via
    enrich_row(), concatenates enriched columns to original DataFrame,
    writes enriched output file, and prints per-table tier summary.

    Args:
        target: dict entry from ENRICHMENT_TARGETS containing input/output
                paths and column mappings
        domain_to_node: mapping dictionary from load_tree() for entity lookup
        visit_map: site_visit dictionary for unknown domain fallback
    """
    input_path = PARQUET_DIR / target['input_file']
    output_path = PARQUET_DIR / target['output_file']
    
    parquet_file = pq.ParquetFile(input_path)
    total_rows = parquet_file.metadata.num_rows
    print(f"Total rows to enrich: {total_rows:,}")
    
    # Build the target schema from the input schema + enrichment columns
    input_schema = parquet_file.schema_arrow
    enrichment_fields = [
        pa.field('domain', pa.large_string()),
        pa.field('subsidiary_entity', pa.large_string()),
        pa.field('parent_entity', pa.large_string()),
        pa.field('relationship_tier', pa.large_string()),
        pa.field('is_technical_3p', pa.float64()),  # explicit float64
    ]
    target_schema = pa.schema(list(input_schema) + enrichment_fields)
    
    writer = pq.ParquetWriter(output_path, target_schema)
    rows_processed = 0
    tier_counter = {}
    
    try:
        for batch in parquet_file.iter_batches(batch_size=50_000):
            df_chunk = batch.to_pandas()
            
            enrichment_results = df_chunk.apply(
                lambda row: enrich_row(
                    row, domain_to_node, visit_map,
                    target['domain_column'], target['context_column']
                ),
                axis=1
            )
            enriched_chunk = pd.concat([df_chunk, enrichment_results], axis=1)
            
            # Convert to Arrow AND cast to the fixed schema
            table_chunk = pa.Table.from_pandas(
                enriched_chunk,
                schema=target_schema,      # <-- enforce schema
                preserve_index=False
            )
            
            writer.write_table(table_chunk)
            
            for tier, count in enriched_chunk['relationship_tier'].value_counts().items():
                tier_counter[tier] = tier_counter.get(tier, 0) + count
            
            rows_processed += len(df_chunk)
            print(f"  Processed {rows_processed:,} / {total_rows:,} rows "
                  f"({100*rows_processed/total_rows:.1f}%)")
            
            del df_chunk, enrichment_results, enriched_chunk, table_chunk
    finally:
        writer.close()
    
    print(f"\nEnrichment Summary — {target['input_file']}:")
    for tier, count in sorted(tier_counter.items(), key=lambda x: -x[1]):
        print(f" - {tier:25}: {count:>8,}")


def main():
    """
    Main orchestration function for the enrichment pipeline.
    """
    print("─" * 60)
    print("STARTING ENRICHMENT PIPELINE")
    print("─" * 60)
    
    # Step 1: Load the Entity Tree
    print(f"Loading entity tree from {TREE_CSV_PATH.name}...")
    # load_tree returns (root, domain_to_node)
    _, domain_to_node = load_tree(TREE_CSV_PATH)

    # Step 2: Load site visits and build fallback visit map
    visit_path = PARQUET_DIR / "site_visits.parquet"
    print(f"Loading site visits: {visit_path.name}")
    visit_map = build_visit_map(pd.read_parquet(visit_path))

    # Step 3: Enrich each registered target table
    for target_name, target_config in ENRICHMENT_TARGETS.items():
        print("─" * 60)
        print(f"STARTING ENRICHMENT: {target_name}")
        enrich_table(target_config, domain_to_node, visit_map)
        
    print("─" * 60)
    print("ENRICHMENT COMPLETE")
    print("─" * 60)

if __name__ == "__main__":
    main()