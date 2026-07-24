# suggested order for file:
#
# pd.read_parquet('http_requests.parquet') — read his output
# Apply get_registered_domain() to the url column → reg_domain
# Apply get_node_info() to reg_domain → flatten to subsidiary_entity, parent_entity
# Apply classify_relationship() for cookie party classification → relationship_tier
# df.to_parquet('http_requests_enriched.parquet') — write back out
# Graham adds one line to his db.py: register http_requests_enriched.parquet as a view