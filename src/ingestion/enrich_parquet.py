"""
Script: /src/ingestion/enrich_parquet.py

Author: Anya Barringer, aided by Claude Sonnet 4.6 and
        Codestral through Furman University BoodleBox

Container:  Part of CSC Summer Research 2026 Project
            "Pervasive Online Third-Party Tracking: A Measurement Study"
            with Graham Fink, under Dr. Rebecca Drucker

Goal:   Enriches crawl data in http_requests.parquet to add domain, entity, and
        relationship classification columns, allowing for more accurate analysis
        of third-party tracking. Output file http_requests_enriched.parquet will
        be called by later scripts eg trackers.py. Uses domain-entity mapping tree
        and helper utility functions from openwpm-tracker-analysis module. Including
        this step in separate script allows for clean separation of classification
        and analysis steps. Run once after load_sqlite.py and before init_database.py
        as part of data cleaning step (Silver layer).
"""



import pandas as pd
from pathlib import Path
from types import SimpleNamespace
from config import TREE_CSV_PATH, PARQUET_DIR
from src.utils.domain_utils import load_tree, get_node_info, get_registered_domain, classify_relationship



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


def enrich_row(row: pd.Series, domain_to_node: dict, visit_map: dict) -> pd.Series:
    """
    Enriches a single HTTP request row with entity and relationship data.
    
    Logic:
    1. Extracts registered domain from request URL.
    2. Extracts top-level domain from top_level_url with visit_map fallback.
    3. Resolves both to entity nodes via the mapping tree.
    4. Classifies relationship tier (First-party, Inter-family, or External).

    Args:
        row: individual HTTP request DataFrame row
        domain_to_node: mapping dictionary from load_tree() for easy lookup
        visit_map: site_visit dictionary for unknown domain fallback

    Returns:
        pd.Series: enriched requests row with domain, subsidiary_entity,
                parent_entity, and relationship_tier classification
    """
    # --- 1. Resolve Request Domain ---
    req_domain = get_registered_domain(row['url'])
    
    # --- 2. Resolve Top-Level Domain (with fallback) ---
    # Check top_level_url column first
    req_top_domain = get_registered_domain(row['top_level_url'])
    
    # Fallback to visit_map if top_level_url is null or unresolvable
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


def main():
    """
    Main orchestration function for the enrichment pipeline.
    """
    print("─" * 60)
    print("STARTING ENRICHMENT: http_requests_enriched.parquet")
    print("─" * 60)
    
    # Step 1: Load the Entity Tree
    print(f"Loading entity tree from {TREE_CSV_PATH.name}...")
    # load_tree returns (root, domain_to_node)
    _, domain_to_node = load_tree(str(TREE_CSV_PATH))
    
    # Step 2: Load Parquet Data
    req_path = PARQUET_DIR / "http_requests.parquet"
    visit_path = PARQUET_DIR / "site_visits.parquet"
    
    print(f"Loading http requests: {req_path.name}")
    http_df = pd.read_parquet(req_path)
    
    print(f"Loading site visits: {visit_path.name}")
    visits_df = pd.read_parquet(visit_path)
    
    # Step 3: Build Fallback Map
    visit_map = build_visit_map(visits_df)
    
    # Step 4: Run Enrichment
    print(f"Enriching {len(http_df):,} rows. This may take a moment...")
    
    # Use lambda to pass the dictionaries into the row-level function
    enrichment_results = http_df.apply(
        lambda row: enrich_row(row, domain_to_node, visit_map), 
        axis=1
    )
    
    # Concatenate the new columns to the original DataFrame
    enriched_df = pd.concat([http_df, enrichment_results], axis=1)
    
    # Step 5: Export Enriched Parquet
    print(f"Writing enriched file to {PARQUET_DIR / 'http_requests_enriched.parquet'}...")
    enriched_df.to_parquet(PARQUET_DIR / 'http_requests_enriched.parquet', index=False)
    
    # Final Summary
    tier_counts = enriched_df['relationship_tier'].value_counts()
    print("\nEnrichment Summary:")
    for tier, count in tier_counts.items():
        print(f" - {tier:25}: {count:>8,}")
        
    print("─" * 60)
    print("ENRICHMENT COMPLETE")
    print("─" * 60)

if __name__ == "__main__":
    main()