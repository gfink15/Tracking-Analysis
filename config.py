"""
config.py — Central configuration for the OpenWPM analysis project.

All paths, profile definitions, and analysis constants live here.
Other modules import from this file so that changing a path or adding
a profile only requires editing one location.
"""
from pathlib import Path

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
    'shopping', 
    # 'news', 
    # 'health'
    ]

# Human-readable labels for plots and tables. Kept separate from the
# internal profile keys so we can change presentation without breaking
# code that references profile identifiers.
PROFILE_LABELS = {
    'control':  'Control (no history)',
    'shopping': 'Shopping history',
    'news':     'News history',
    'health':   'Health history',
}

# Consistent colors across every figure in the project. A single source
# for colors is critical when you have 10+ plots in a paper — reviewers
# notice if "shopping" is red in Figure 2 and blue in Figure 5.
PROFILE_COLORS = {
    'control':  '#888888',  # neutral gray for baseline
    'shopping': '#E74C3C',  # red
    'news':     '#3498DB',  # blue
    'health':   '#2ECC71',  # green
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
DUCKDB_MEMORY_LIMIT = '8GB'
DUCKDB_THREADS = 4