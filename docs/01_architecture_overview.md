# OpenWPM Analysis Pipeline — Architecture Overview

This document describes the **analysis pipeline** that ingests OpenWPM
crawl data, transforms it into a queryable analytical store, and
produces research figures. It does NOT cover the data collection
crawler (see the OpenWPM crawl scripts for that).

If you're new to the project, read this document first. Then proceed to:
- `02_ingestion_layer.md` — how raw data becomes Parquet
- `03_analysis_layer.md` — the SQL/statistics modules
- `04_visualization_layer.md` — how data becomes figures

---

## Table of Contents

1. [What This Pipeline Does](#what-this-pipeline-does)
2. [The Medallion Architecture](#the-medallion-architecture)
3. [Data Flow End-to-End](#data-flow-end-to-end)
4. [Directory Layout](#directory-layout)
5. [Core Technologies](#core-technologies)
6. [Configuration: `config.py`](#configuration-configpy)
7. [Running the Pipeline](#running-the-pipeline)

---

## What This Pipeline Does

Given output from an OpenWPM crawl across multiple browser profiles
(typically a control profile plus several "seeded" profiles with
pre-loaded browsing history), this pipeline:

1. **Converts** raw SQLite databases and ad screenshot artifacts into
   compressed Parquet files
2. **Unifies** all profiles' data into a single DuckDB database that
   can be queried with SQL
3. **Analyzes** the data with reusable Python modules covering
   trackers, cookies, fingerprinting, and ad content
4. **Visualizes** results as publication-quality figures saved to PDF

The end product is the raw material for a research paper on
behavioral tracking and ad targeting.

---

## The Medallion Architecture

The pipeline follows a **bronze → silver → gold** data layering pattern
adapted from data engineering practice. Each layer has clear inputs,
outputs, and responsibilities, and downstream layers never reach back
to upstream raw data.

┌────────────────────────────────────────────────────────────────┐ │ 🥉 BRONZE — Raw Data (data/) │ │ ────────────────────────────────────────────── │ │ • crawl-data.sqlite (one per profile, from OpenWPM) │ │ • ads/<visit_id>/.json (ad metadata, from ad_capture) │ │ • ads/<visit_id>/.png (ad screenshots) │ │ │ │ Precious & unreproducible. NEVER modified by the pipeline. │ └────────────────────────────────────────────────────────────────┘ ↓ (src/ingestion/load_sqlite.py) (src/ingestion/load_ad_artifacts.py) ↓ ┌────────────────────────────────────────────────────────────────┐ │ 🥈 SILVER — Cleaned & Unified (artifacts/parquet/) │ │ ────────────────────────────────────────────── │ │ • http_requests.parquet (all profiles unioned) │ │ • http_responses.parquet │ │ • javascript.parquet │ │ • javascript_cookies.parquet │ │ • site_visits.parquet │ │ • ads.parquet (enriched with OCR + network) │ │ │ │ Single source of truth for analysis. Regeneratable from │ │ Bronze + code, so safely git-ignored. │ └────────────────────────────────────────────────────────────────┘ ↓ (scripts/init_database.py registers views) ↓ ┌────────────────────────────────────────────────────────────────┐ │ 🪙 GOLD — Analytical Outputs (artifacts/) │ │ ────────────────────────────────────────────── │ │ • analysis.duckdb (Parquet-backed view layer) │ │ • figures/*.pdf (publication figures) │ │ • models/bertopic/ (fitted topic models) │ │ │ │ Cheap to recreate. Driven by analysis modules + notebooks. │ └────────────────────────────────────────────────────────────────┘


### Why This Structure?

- **Expensive operations cache their output.** Ingestion (Bronze→Silver)
  takes 5-30 minutes; analysis (Silver→Gold) runs many times. Caching
  at the boundary means iterating on analysis is fast.
- **Each layer is independently rebuildable.** You can delete all of
  `artifacts/` and reproduce it from `data/` + code. You can never
  destroy raw data through normal operation.
- **The DuckDB layer is a façade, not a copy.** Views in
  `analysis.duckdb` point at the Parquet files; they don't duplicate
  data. Updates to Parquet are picked up by simply re-registering
  views (via `scripts/init_database.py`).

---

## Data Flow End-to-End

Multiple per-profile Single set of SQLite databases unioned Parquet files ───────────────── ────────────────────── data/control/ artifacts/parquet/ crawl-data.sqlite http_requests.parquet data/shopping/ http_responses.parquet crawl-data.sqlite → javascript.parquet data/news/ javascript_cookies.parquet crawl-data.sqlite site_visits.parquet data/health/ ads.parquet crawl-data.sqlite

     ↓                                ↓
                              artifacts/analysis.duckdb
                              (views pointing at Parquet)
                                      ↓
                              Notebook + analysis modules
                                      ↓
                              artifacts/figures/*.pdf


A row in the Bronze SQLite has a `visit_id`. After ingestion, that
same row in Silver Parquet has both `visit_id` AND a new `profile`
column tagging which profile it came from. This is the join key that
makes cross-profile comparisons possible.

---

## Directory Layout

openwpm_analysis/ ├── README.md ├── pyproject.toml # dependency pinning ├── config.py # ★ central configuration (paths, profiles) │ ├── data/ # 🥉 Bronze — raw crawl outputs (git-ignored) │ ├── control/ │ │ ├── crawl-data.sqlite │ │ └── ads/ │ ├── shopping/ │ ├── news/ │ └── health/ │ ├── artifacts/ # 🥈🪙 Silver+Gold — derived (git-ignored) │ ├── parquet/ │ │ ├── http_requests.parquet │ │ ├── http_responses.parquet │ │ ├── javascript.parquet │ │ ├── javascript_cookies.parquet │ │ ├── site_visits.parquet │ │ └── ads.parquet │ ├── analysis.duckdb # DuckDB file with views over Parquet │ ├── figures/ # PDF outputs from notebooks │ └── models/ # fitted models (BERTopic, etc.) │ ├── reference/ # external reference data (git-tracked) │ └── disconnect_blocklist.json │ ├── src/ # importable Python modules │ ├── init.py │ ├── ingestion/ # Bronze → Silver │ │ ├── load_sqlite.py │ │ └── load_ad_artifacts.py │ ├── analysis/ # Silver → analytical DataFrames │ │ ├── trackers.py │ │ ├── cookies.py │ │ ├── fingerprinting.py │ │ ├── ads.py │ │ ├── statistics.py │ │ └── topic_modeling.py │ ├── viz/ # DataFrame → Figure │ │ ├── tracker_plots.py │ │ ├── cookie_plots.py │ │ └── ad_plots.py │ └── utils/ │ └── db.py # DuckDB connection helpers │ ├── scripts/ # CLI entry points │ └── init_database.py # register Parquet views in analysis.duckdb │ └── notebooks/ # Jupyter lab notebooks ├── 01_data_overview.ipynb ├── 02_tracking_comparison.ipynb └── 03_ad_content_analysis.ipynb


---

## Core Technologies

| Tool | Why we use it |
|---|---|
| **Python 3.10+** | Modern type hints (`dict[str, int]`, `list[Path]`) without `typing.` prefixes. |
| **DuckDB** | In-process analytical SQL. Reads Parquet at near-disk speed. Single embedded file, no server. Handles tens of GB on a laptop. |
| **Parquet (ZSTD)** | Columnar, compressed file format. ~10× smaller than SQLite for the same data; ~100× faster for aggregations. ZSTD compression typically gives 3× better ratios than the default Snappy with similar read speed. |
| **pandas** | Standard interface for analytical DataFrames in/out of plotting code. Used at the edges (post-SQL, pre-plot). |
| **matplotlib** | Figure generation. Chosen over seaborn/plotly because we need fine control over publication-quality output. |
| **scipy.stats** | Hypothesis testing (chi-square, Mann-Whitney, Kruskal-Wallis). |
| **pytesseract** | OCR for extracting text from ad screenshots. Requires the `tesseract` system binary. |
| **BERTopic** *(optional)* | Semantic topic modeling on ad OCR text. Heavy dependency; loaded lazily. |

---

## Configuration: `config.py`

`config.py` is the **single source of truth** for paths, profile
definitions, and analysis constants. **Every other module imports
from here.** Changing a path or adding a profile only requires
editing one file.

### Path Variables

| Variable | Purpose |
|---|---|
| `PROJECT_ROOT` | Absolute path to project root. Computed from `__file__` so it works regardless of working directory. |
| `DATA_DIR` | Where raw crawl data lives. Subdirectories named per profile. |
| `ARTIFACTS_DIR` | Root for all derived data. Git-ignored. |
| `PARQUET_DIR` | Where ingestion writes Parquet files. |
| `FIGURES_DIR` | Where notebooks save figure PDFs. |
| `DUCKDB_PATH` | Path to the unified DuckDB file. |
| `REFERENCE_DIR` | External reference data (tracker blocklists, etc.) — git-tracked. |

**Important:** All paths are resolved relative to `PROJECT_ROOT`, NOT
the current working directory. This means scripts and notebooks work
identically whether invoked from the project root or from any
subdirectory.

### Profile Definitions

| Variable | Purpose |
|---|---|
| `PROFILES` | Ordered list of profile keys. Order is preserved in every plot. **Control must be first** for cross-profile comparisons. |
| `PROFILE_LABELS` | Dict mapping profile keys → human-readable strings for figure legends. Decoupling labels from keys means you can rename profiles in plots without touching code. |
| `PROFILE_COLORS` | Dict mapping profile keys → hex colors. **The single source of color** for every visualization. |

### OpenWPM Schema

| Variable | Purpose |
|---|---|
| `OPENWPM_TABLES` | List of SQLite tables to ingest. Adding a table here automatically includes it in ingestion and view registration. |

### Statistical Settings

| Variable | Purpose |
|---|---|
| `ALPHA` | Significance threshold for hypothesis tests (default: 0.05). |
| `BONFERRONI_CORRECT` | Whether to apply Bonferroni correction by default in multiple-testing scenarios. |

### Performance Tuning

| Variable | Purpose |
|---|---|
| `DUCKDB_MEMORY_LIMIT` | Caps DuckDB's memory usage. Prevents OOM-killing other processes on smaller machines. Default `'8GB'`. |
| `DUCKDB_THREADS` | DuckDB parallelism. Default 4. Increase on multi-core servers. |

---

## Running the Pipeline

A typical end-to-end run after a fresh crawl:

```bash
# 1. Ingest OpenWPM SQLite into Parquet (one Parquet file per table,
#    unioned across all profiles, tagged with a 'profile' column)
python -m src.ingestion.load_sqlite

# 2. Ingest ad artifacts (JSON metadata + PNG screenshots → ads.parquet
#    with OCR text and advertiser network identification)
python -m src.ingestion.load_ad_artifacts

# 3. Register Parquet files as views in analysis.duckdb
python scripts/init_database.py

# 4. Open notebooks and run cells
jupyter lab notebooks/01_data_overview.ipynb

Subsequent re-runs only require steps you've actually invalidated:

    New crawl data → start from step 1
    Re-ran ingestion → run step 3
    Just iterating on analysis → open notebooks directly

Smoke Test (No Crawl Required)

Each module has a if __name__ == "__main__": block that runs a small self-test:
bash

python -m src.utils.db                  # show row counts per table per profile
python -m src.analysis.trackers         # tracker prevalence + Jaccard + differentials
python -m src.analysis.cookies          # cookie counts, lifespans, retargeters
python -m src.analysis.fingerprinting   # fingerprinter detection
python -m src.analysis.ads              # ad volumes and networks

If any of these fail, you have a data or environment issue to fix before opening notebooks.