# Tracking Analysis

## Data Dependency

This project consumes `output_tree.csv`, produced by the sibling repo
[openwpm-tracker-analysis](https://github.com/anyab12/openwpm-tracker-analysis).
This tree is the basis for domain-entity hierarchy mapping, which dictates
the first-party vs third-party classification system used in analysis.

**Option A (recommended):** Clone both repos as siblings:
    cd C:\Users\<you>\
    git clone https://github.com/anyab12/openwpm-tracker-analysis.git
    git clone https://github.com/gfink15/Tracking-Analysis.git

Then run `build_mapping_tree.py` in openwpm-tracker-analysis. The tree
CSV now exists and will be found automatically.

**Option B (custom layout):** Set the `TREE_CSV_PATH` environment variable (from config.py):
    export TREE_CSV_PATH=/your/path/to/output_tree.csv

**Option C (archived snapshot):** If `Tracker-Analysis/data/output_tree.csv`
already exists, likely as an artifact from the research work or from previous
instantiation, it will be used automatically — no need to clone the sibling repo.