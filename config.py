"""
config.py — Central configuration for the OpenWPM analysis project.

All paths, profile definitions, and analysis constants live here.
Other modules import from this file so that changing a path or adding
a profile only requires editing one location.
"""
import os
from pathlib import Path
from enum import Enum

# ─────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────
# PROJECT_ROOT resolves to the directory containing this config.py file.
# Using Path(__file__).resolve().parent makes the project portable —
# it works regardless of where you run scripts from (cwd-independent).
PROJECT_ROOT = Path(__file__).resolve().parent

# Raw crawl outputs: one subdirectory per profile, each containing
# crawl-data.sqlite and an ads/ subdirectory with JSON+PNG artifacts.
DATA_DIR = PROJECT_ROOT / "data"

# Everything derived from raw data lives here. This directory is
# regeneratable from data/ + code, so it can be safely git-ignored.
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
PARQUET_DIR   = ARTIFACTS_DIR / "parquet"
FIGURES_DIR   = ARTIFACTS_DIR / "figures"
DUCKDB_PATH   = ARTIFACTS_DIR / "analysis.duckdb"

# External reference data (tracker lists, etc.) — checked into git.
REFERENCE_DIR = PROJECT_ROOT / "reference"

# --- Cross-repo dependency: tree CSV from openwpm-tracker-analysis ---
# Default assumes sibling-repo layout. Override with TREE_CSV_PATH if needed.
TREE_CSV_PATH = Path(os.environ.get(
    "TREE_CSV_PATH",
    PROJECT_ROOT.parent / "openwpm-tracker-analysis" / "data" / "output_tree.csv"
))

# Also allow using a local snapshot copy if present (for archival/release)
_LOCAL_SNAPSHOT = DATA_DIR / "output_tree.csv"
if _LOCAL_SNAPSHOT.exists() and "TREE_CSV_PATH" not in os.environ:
    TREE_CSV_PATH = _LOCAL_SNAPSHOT

# Ensure derived directories exist at import time. This is a small
# convenience so downstream code can write files without each module
# needing its own mkdir() calls.
for d in (ARTIFACTS_DIR, PARQUET_DIR, FIGURES_DIR, REFERENCE_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────
# EXPERIMENTAL DESIGN
# ─────────────────────────────────────────────────────────────────────
# The list of profiles is the backbone of every comparison we'll do.
# Order matters for plot consistency — 'control' first as the baseline.
PROFILES = [
    'control', 
    'gaming', 
    'sports_car_fan', 
    'investor',
    'retiree'
    ]

# Human-readable labels for plots and tables. Kept separate from the
# internal profile keys so we can change presentation without breaking
# code that references profile identifiers.
PROFILE_LABELS = {
    'control':     'Control (no history)',
    'gaming':      'Gaming history',
    'sports_car_fan':     'Automotive history',
    'investor':     'Investment history',
    'retiree': 'Retiree history'
}

# Consistent colors across every figure in the project. A single source
# for colors is critical when you have 10+ plots in a paper — reviewers
# notice if "shopping" is red in Figure 2 and blue in Figure 5.
PROFILE_COLORS = {
    'control':  '#888888',  # neutral gray for baseline
    'gaming': "#3B9ADA",  # blue
    'sports_car_fan':     "#FF4A4A",  # red
    'investor':   "#14CF0D",  # green
    'retiree': "#e96df5" # pink
}

# ─────────────────────────────────────────────────────────────────────
# OPENWPM TABLE NAMES
# ─────────────────────────────────────────────────────────────────────
# These are the SQLite tables OpenWPM produces that we care about.
# Listing them here means we can iterate cleanly during ingestion
# instead of hardcoding strings in multiple places.
OPENWPM_TABLES = [
    'site_visits',          # one row per page visit (the join key)
    'http_requests',        # outgoing network requests
    'http_responses',       # responses received
    'http_redirects',       # redirect chains (useful for tracker hops)
    'javascript',           # instrumented JS API calls (fingerprinting)
    'javascript_cookies',   # cookies set/read via document.cookie
    'callstacks',           # JS call stacks (helps attribute behavior)
]

# ─────────────────────────────────────────────────────────────────────
# ENRICHED TABLE NAMES
# ─────────────────────────────────────────────────────────────────────
# These are the tables that are generated via the enrichment pipeline
# in enrich_parquet.py. Access these enriched tables in analysis scripts
# for information like domain-entity mapping and relationship classification.
# Update corresponding constant ENRICHMENT_TARGETS in enrich_parquet.py.
ENRICHED_TABLES = [
    'http_requests_enriched',       # outgoing network requests
    'javascript_cookies_enriched',  # cookies set/read via document.cookie
    'javascript_enriched',          # instrumented JS API calls (fingerprinting)
]

# ─────────────────────────────────────────────────────────────────────
# STATISTICAL SETTINGS
# ─────────────────────────────────────────────────────────────────────
# Significance threshold for hypothesis tests. Defined here so every
# analysis script uses the same value — and so you can change it in
# one place if a reviewer asks for α = 0.01.
ALPHA = 0.05

# Whether to apply Bonferroni correction when running multiple tests.
# At this stage of a project I default to True; with N profile pairs
# and M metrics, you're running N*M tests and uncorrected p-values
# will produce false positives.
BONFERRONI_CORRECT = True

# ─────────────────────────────────────────────────────────────────────
# RUNTIME / PERFORMANCE
# ─────────────────────────────────────────────────────────────────────
# DuckDB memory limit — adjust based on your machine. Setting this
# prevents DuckDB from OOM-killing other processes on smaller systems.
DUCKDB_MEMORY_LIMIT = '16GB'
DUCKDB_THREADS = 4

class Categories(Enum):
    Auto = "Automotive",
    Beauty = "Beauty & Personal Care",
    Business = "Business Services",
    Construction = "Construction & Home Improvement",
    Electronics = "Consumer Electronics",
    Education = "Education",
    Energy = "Energy & Utilities",
    Entertainment = "Entertainment",
    Fashion = "Fashion & Apparel",
    Finance = "Finance",
    Food = "Food & Beverage",
    Gaming = "Gaming",
    Govt = "Government & Public Services",
    Wellness = "Health & Wellness",
    Healthcare = "Healthcare",
    Garden = "Home & Garden",
    Manufacturing = "Industrial & Manufacturing",
    Insurance = "Insurance",
    Luxury = "Jewelry & Luxury Goods",
    Legal = "Legal Services",
    Marketplace = "Marketplace & Classifieds",
    Media = "Media & Publishing",
    Charity = "Nonprofit & Charity",
    Pets = "Pets",
    Estate = "Real Estate",
    Career = "Recruitment & Careers",
    Dining = "Restaurants & Dining",
    Retail = "Retail",
    Software = "Software & SaaS",
    Fitness = "Sports & Fitness",
    Technology = "Technology",
    Telecom = "Telecommunications",
    Travel = "Travel & Hospitality",
    Transport = "Transportation & Logistics",
    Goods = "Consumer Packaged Goods",
    Crypto = "Cryptocurrency & Web3",
    Dating = "Dating",
    Events = "Events & Conferences",
    Family = "Parenting & Family",
    Photos = "Photography & Creative Services",
    Faith = "Religion & Faith",
    Privacy = "Security & Privacy",
    IoT = "Smart Home & IoT",
    Stream = "Streaming Services",
    Subscription = "Subscription Services",
    Hobbies = "Toys & Hobbies",
    Adult = "Adult",
    Political = "Political",
    Safety = "Public Safety",
    Scam = "Likely Scam",
    Other = "Other"