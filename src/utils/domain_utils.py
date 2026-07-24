"""
Script: /src/utils/domain_utils.py

Author: Anya Barringer, aided by Claude Sonnet 4.6 and
        Codestral through Furman University BoodleBox

Container:  Part of CSC Summer Research 2026 Project
            "Pervasive Online Third-Party Tracking: A Measurement Study"
            with Graham Fink, under Dr. Rebecca Drucker

Goal:   Contains shared utility functions that are called
        by other analysis scripts, as well as related constants.
        Does not contain pipeline-specific dependencies, file
        paths, database connections or loading, or output logic.
        
        Functions include:
            - URL and domain extraction - get_registered_domain()
            - Entity tree loading and node / domain lookup -
                load_tree(), get_node_info()
            - Hierarchial relationship classification - classify_relationship()
            - Summary statistic helpers - get_number_children()
"""



import logging
import tldextract
import pandas as pd
from pathlib import Path
from types import SimpleNamespace
from bigtree import Node, Tree, dataframe_to_tree
from config import TREE_CSV_PATH

logger = logging.getLogger(__name__)



def get_registered_domain(url: str) -> str:
    """
    Extracts the eTLD+1 (registrable domain) from a URL using tldextract.
    Handles None, NaN, empty strings, bare IPs, and malformed URLs.
    Adds branded gTLD handling for domains like .google, .fox.
    Returns a lowercase domain string, or None if unresolvable.
    """
    if not url or pd.isna(url):
        return None
    
    try:
        ext = tldextract.extract(str(url))
        if not ext.domain or not ext.suffix:
            # Branded gTLD fallback — tldextract may put these entirely in suffix
            if ext.suffix and not ext.domain:
                return ext.suffix.lower()
            return None
        return f"{ext.domain}.{ext.suffix}".lower()
    except Exception:
        return None


def load_tree(
    tree_csv_path: Path=TREE_CSV_PATH
) -> tuple[Node, dict]:
    """
    Loads pre-built entity tree from CSV and reconstructs bigtree object.
    Also builds domain_to_node index for O(1) classification lookup.

    Args:
        tree_csv_path: Path to output_tree.csv produced by build_mapping_tree.py

    Returns:
        root:           Root node of reconstructed bigtree structure.
        domain_to_node: Dict mapping domain strings to leaf node objects.
    """
    try:
        tree_df = pd.read_csv(tree_csv_path)
    except FileNotFoundError:
        logger.error(f"Entity tree not found at '{tree_csv_path}'. Run build_mapping_tree.py first.")
        raise

    root = dataframe_to_tree(tree_df, path_col="path")
    domain_to_node = {node.name: node for node in root.leaves}

    logger.info(f"Entity tree loaded: {len(domain_to_node)} domains mapped.")
    return root, domain_to_node


def get_node_info(domain: str, domain_to_node: dict) -> SimpleNamespace:
    """
    Retrieves node from the index. If domain is unknown and
    dictionary returns None, function returns Virtual Node.
    """
    node = domain_to_node.get(domain)
    if node:
        return node
    
    return SimpleNamespace(
        name=domain,
        subsidiary_entity=domain,
        parent_entity=domain,
        source="none",
        priority=0
    )


def resolve_node(domain: str, domain_to_node: dict) -> tuple[str, str]:
    """
    Resolves subsidiary and parent entities for given domain string.
    Wraps around get_node_info() for cleaner implementation in analysis
    scripts. Falls back to raw domain string for both fields if unmapped.

    Args:
        domain:         Registered domain string to be mapped
        domain_to_node: Domain-to-node mapping dict produced by build_mapping_tree()

    Returns:
        Tuple of (subsidiary_entity, parent_entity)
    """
    if not domain:
        return (None, None)
    node = get_node_info(domain, domain_to_node)
    return (node.subsidiary_entity, node.parent_entity)


def classify_relationship(req_node: Node, top_node: Node) -> dict:
    """
    Compares two entity tree nodes to determine their relationship.
    Uses nested checks so each tier only fires if the previous tier confirmed
    a difference — avoids redundant comparisons.

    Args:
        req_node:       request URL domain node
        top_node:       top-level URL domain node

    Returns dict of boolean flags:
        is_technical_third_party   → eTLD+1 differs
        is_subsidiary_third_party  → subsidiary entity differs
        is_parent_third_party      → parent entity differs
    """
    result = {
        "is_technical_third_party":  False,
        "is_subsidiary_third_party": False,
        "is_parent_third_party":     False,
    }

    # Technical check — node names are the domain strings
    if req_node.name != top_node.name:
        result["is_technical_third_party"] = True

        # Subsidiary check — attribute, or immediate parent node
        if req_node.subsidiary_entity != top_node.subsidiary_entity:
            result["is_subsidiary_third_party"] = True

            # Parent check — attribute, or grandparent node
            if req_node.parent_entity != top_node.parent_entity:
                result["is_parent_third_party"] = True
    return result


def get_number_children(node: Node) -> int:
    """
    Returns number of nodes in node.children list as number of
    subsidiaries per parent, or number of domains per subsidiary. 
    """
    return len(node.children)