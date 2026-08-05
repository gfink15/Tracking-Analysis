# ad_reviewer.py
# Run with: streamlit run ad_reviewer.py

import streamlit as st
import pandas as pd
import shutil
from pathlib import Path
from streamlit_shortcuts import shortcut_button
import config

# ---- CONFIG ----
# Pristine originals — never written to after initialization
ORIGINAL_DESC = Path("artifacts/parquet/ads_desc.parquet")
ORIGINAL_DATA = Path("artifacts/parquet/ads.parquet")

# Working files — the app reads AND writes exclusively to these
WORKING_DESC = Path("artifacts/parquet/ads_desc_reviewed.parquet")
WORKING_DATA = Path("artifacts/parquet/ads_reviewed.parquet")

SCREENSHOT_DIR = Path("data")
CATEGORY_OPTIONS = [str(c.value) for c in config.Categories]

# ---- INITIALIZATION: create working copies once ----
def initialize_working_files():
    """
    On first run, clone the pristine originals into working files and add
    review-tracking columns. On subsequent runs, do nothing — we read the
    working files as-is so all prior edits persist.
    """
    if not WORKING_DESC.exists():
        if not ORIGINAL_DESC.exists():
            st.error(f"Original file not found: `{ORIGINAL_DESC}`")
            st.stop()
        st.info(f"First run — creating working copy: `{WORKING_DESC.name}`")
        df = pd.read_parquet(ORIGINAL_DESC)
        # Add review-tracking columns with correct dtypes
        if "viewed" not in df.columns:
            df["viewed"] = False
        if "modified" not in df.columns:
            df["modified"] = False
        if "same_company" not in df.columns:
            df["same_company"] = False
        for col in ["old_category", "old_description", "notes"]:
            if col not in df.columns:
                df[col] = ""
        df.to_parquet(WORKING_DESC, index=False)

    if not WORKING_DATA.exists():
        if not ORIGINAL_DATA.exists():
            st.error(f"Original file not found: `{ORIGINAL_DATA}`")
            st.stop()
        shutil.copy(ORIGINAL_DATA, WORKING_DATA)

initialize_working_files()

# ---- LOAD DATA (from working files only) ----
@st.cache_data
def load_desc():
    df = pd.read_parquet(WORKING_DESC)
    # Belt-and-suspenders dtype coercion for booleans
    for col in ["viewed", "modified"]:
        df[col] = (
            df[col]
            .replace({"True": True, "False": False, "": False})
            .fillna(False)
            .astype(bool)
        )
    for col in ["old_category", "old_description", "notes"]:
        df[col] = df[col].fillna("").astype(str)
    return df

@st.cache_data
def load_data():
    return pd.read_parquet(WORKING_DATA)

if "df_desc" not in st.session_state or "df_data" not in st.session_state:
    st.session_state.df_desc = load_desc()
    st.session_state.df_data = load_data()

if "idx" not in st.session_state:
    st.session_state.idx = 0

df_desc = st.session_state.df_desc
df_data = st.session_state.df_data
total = len(df_desc)

# ---- SIDEBAR: NAVIGATION & PROGRESS ----
st.sidebar.title("Ad Review Tool")
viewed_count = int(df_desc["viewed"].sum()) if df_desc["viewed"].sum() else 0
st.sidebar.metric("Progress", f"{viewed_count} / {total}")
st.sidebar.progress(viewed_count / total if total else 0)

jump_to = st.sidebar.number_input(
    "Jump to index", min_value=0, max_value=total - 1,
    value=st.session_state.idx, step=1
)
if jump_to != st.session_state.idx:
    st.session_state.idx = jump_to

show_unviewed_only = st.sidebar.checkbox("Skip already-viewed", value=False)

# ---- MAIN LAYOUT ----
row = df_desc.iloc[st.session_state.idx]
row_data = df_data.iloc[st.session_state.idx]
st.title(f"Ad {st.session_state.idx + 1} of {total}")

col_img, col_edit = st.columns([1, 1])

with col_img:
    st.subheader("Screenshot")
    # Adjust this to match how your screenshot paths are stored
    # Common patterns: row['ad_hash'] + '.png', or row['screenshot_path']
    img_name = f"{row.get('ad_hash')}.png"
    img_path = SCREENSHOT_DIR / str(row.get('profile')) / "ads" / str(row.get('profile')) / str(row.get('visit_id')) / img_name
    if img_path.exists():
        st.image(str(img_path), use_container_width=True)
    else:
        st.warning(f"Screenshot not found: {img_path}")
    
    # Show metadata for context
    with st.expander("Ad metadata"):
        st.write({
            "profile": row.get("profile", "N/A"),
            "page_url": row_data.get("page_url", "N/A"),
            "ad_network": row_data.get("advertiser_network", "N/A"),
            "size": f"{row_data.get('ad_width', 0)}x{row_data.get('ad_height', 0)}",
        })

with col_edit:
    st.subheader("AI-Generated Description")
    st.info(row.get("description", "(no description)"))
    
    st.subheader("Your Corrections")
    # Pre-fill with any existing corrections
    current_cat = row.get("category", "Other")
    if current_cat not in CATEGORY_OPTIONS:
        current_cat = f"Other"
    
    new_category = st.selectbox(
        "Category", CATEGORY_OPTIONS,
        index=CATEGORY_OPTIONS.index(current_cat)
    )
    new_description = st.text_area(
        "Edit description (optional)",
        value=row.get("description", ""),
        height=150
    )
    new_same_company = st.checkbox(
        "Ad for same company as website?",
        value=row.get("same_company", False),
        key="same_company"
    )
    notes = st.text_input("Notes (optional)", value=row.get("notes", ""))

# ---- ACTION BUTTONS ----
c1, c2, c3, c4 = st.columns(4)

def save_current(mark_modified=True):
    i = st.session_state.idx
    # Preserve the "before" values only the first time we edit this row
    if not df_desc.at[i, "modified"]:
        df_desc.at[i, "old_category"] = df_desc.at[i, "category"]
        df_desc.at[i, "old_description"] = df_desc.at[i, "description"]
    df_desc.at[i, "category"] = new_category
    df_desc.at[i, "description"] = new_description
    df_desc.at[i, "same_company"] = new_same_company
    df_desc.at[i, "notes"] = notes
    if mark_modified:
        df_desc.at[i, "modified"] = True
    df_desc.to_parquet(WORKING_DESC, index=False)

def advance():
    df_desc.at[st.session_state.idx, "viewed"] = True
    df_desc.to_parquet(WORKING_DESC, index=False)
    if show_unviewed_only:
        forward = df_desc.index[
            (df_desc.index > st.session_state.idx) & (~df_desc["viewed"].astype(bool))
        ]
        if len(forward):
            st.session_state.idx = int(forward[0])
            return
        backward = df_desc.index[
            (df_desc.index < st.session_state.idx) & (~df_desc["viewed"].astype(bool))
        ]
        if len(backward):
            st.session_state.idx = int(backward[0])
            st.toast("Wrapped to earlier unviewed ads.")
            return
        st.toast("🎉 All ads have been viewed!")
    else:
        st.session_state.idx = min(st.session_state.idx + 1, total - 1)
    

with c1:
    if shortcut_button("⬅ Previous", shortcut="a"):
        st.session_state.idx = max(0, st.session_state.idx - 1)
        st.rerun()
with c2:
    if shortcut_button("💾 Save", shortcut="w"):
        save_current()
        st.success("Saved!")
with c3:
    if shortcut_button("✅ Save & Next", type="primary", shortcut="s"):
        save_current()
        advance()
        st.rerun()
with c4:
    if shortcut_button("⏭ Skip", shortcut="d"):
        advance()
        st.rerun()

st.caption(f"Auto-saves to `{WORKING_DESC}` on every Save action. Original file untouched.")