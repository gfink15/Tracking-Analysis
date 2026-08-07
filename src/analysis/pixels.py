# src/analysis/pixels.py
"""
Pixel detection and analysis for OpenWPM crawl data.

Detects advertising pixels (Meta, TikTok, DoubleClick, etc.) across two data sources:
  1. Profile-build SQLite databases (what the persona was "exposed to" during seeding)
  2. Measurement Parquet tables (what fired during the ad-capture crawls)

Matches sites across sources using eTLD+1 (registrable domain) for robustness.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import tldextract

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Advertising pixel definitions
# ---------------------------------------------------------------------------
# Focused on ADVERTISING pixels only (not pure analytics like GA/GTM).
# Each entry: canonical name -> (domain patterns, url path patterns)

AD_PIXEL_SIGNATURES: dict[str, dict] = {
    "Meta Pixel": {
        "domains": {"facebook.com", "connect.facebook.net", "facebook.net"},
        "paths": [r"/tr/?(\?|$)", r"/en_US/fbevents\.js", r"/signals/"],
    },
    "TikTok Pixel": {
        "domains": {"analytics.tiktok.com", "analytics-sg.tiktok.com"},
        "paths": [r"/i18n/pixel", r"/api/v2/pixel", r"/pixel/"],
    },
    "DoubleClick": {
        "domains": {"doubleclick.net", "g.doubleclick.net", "stats.g.doubleclick.net"},
        "paths": [r"/activity", r"/adview", r"/pagead/viewthroughconversion",
                  r"/pagead/1p-user-list"],
    },
    "Google Ads Conversion": {
        "domains": {"googleadservices.com", "googlesyndication.com"},
        "paths": [r"/pagead/conversion", r"/pagead/landing", r"/pagead/viewthrough"],
    },
    "LinkedIn Insight": {
        "domains": {"px.ads.linkedin.com", "snap.licdn.com"},
        "paths": [r"/collect", r"/li\.lms-analytics/insight\.min\.js"],
    },
    "Microsoft UET": {
        "domains": {"bat.bing.com"},
        "paths": [r"/action/", r"/bat\.js"],
    },
    "X/Twitter Pixel": {
        "domains": {"analytics.twitter.com", "ads-twitter.com", "t.co"},
        "paths": [r"/i/adsct", r"/uwt\.js"],
    },
    "Pinterest Tag": {
        "domains": {"ct.pinterest.com", "s.pinimg.com"},
        "paths": [r"/ct\.gif", r"/ct/", r"/v3/"],
    },
    "Snap Pixel": {
        "domains": {"tr.snapchat.com", "sc-static.net"},
        "paths": [r"/p", r"/scevent\.min\.js"],
    },
    "Criteo": {
        "domains": {"criteo.com", "criteo.net", "static.criteo.net"},
        "paths": [r"/delivery/", r"/load/", r"/tag/"],
    },
    "Taboola": {
        "domains": {"taboola.com", "trc.taboola.com"},
        "paths": [r"/trc/", r"/pixel"],
    },
    "Outbrain": {
        "domains": {"outbrain.com", "amplify.outbrain.com"},
        "paths": [r"/pixel", r"/OutbrainRatingsEngine"],
    },
}

# Compile once for speed
_COMPILED_SIGNATURES = {
    name: {
        "domains": sig["domains"],
        "paths": [re.compile(p, re.IGNORECASE) for p in sig["paths"]],
    }
    for name, sig in AD_PIXEL_SIGNATURES.items()
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_tld_extractor = tldextract.TLDExtract(suffix_list_urls=())  # offline, cached

def etld_plus_one(url_or_host: str) -> Optional[str]:
    """Return the registrable domain (eTLD+1) for a URL or hostname."""
    if not url_or_host or not isinstance(url_or_host, str):
        return None
    ext = _tld_extractor(url_or_host)
    if not ext.domain or not ext.suffix:
        return None
    return f"{ext.domain}.{ext.suffix}".lower()


@dataclass
class PixelHit:
    pixel_type: str
    matched_domain: str
    matched_path_pattern: Optional[str]


def classify_pixel(url: str) -> Optional[PixelHit]:
    """
    Classify a request URL as an advertising pixel, or None if not a pixel.

    Uses a two-signal approach:
      1. Domain match against known ad-pixel hosts (necessary condition)
      2. Path pattern match (confirms it's the tracking endpoint, not just
         a script fetch from the same CDN)
    """
    if not url or not isinstance(url, str):
        return None

    host_e1 = etld_plus_one(url)
    if host_e1 is None:
        return None

    # Also grab full host for subdomain-specific matches
    ext = _tld_extractor(url)
    full_host = ".".join(p for p in [ext.subdomain, ext.domain, ext.suffix] if p).lower()

    for pixel_name, sig in _COMPILED_SIGNATURES.items():
        # Domain check: either eTLD+1 or full host matches a known domain
        domain_match = None
        for d in sig["domains"]:
            if host_e1 == d or full_host == d or full_host.endswith("." + d):
                domain_match = d
                break
        if not domain_match:
            continue

        # Path check
        for path_re in sig["paths"]:
            if path_re.search(url):
                return PixelHit(
                    pixel_type=pixel_name,
                    matched_domain=domain_match,
                    matched_path_pattern=path_re.pattern,
                )

        # Domain matched but no path pattern — still likely a pixel host,
        # tag it as domain-only (lower confidence)
        # return PixelHit(
        #     pixel_type=pixel_name,
        #     matched_domain=domain_match,
        #     matched_path_pattern=None,
        # )

    return None


# ---------------------------------------------------------------------------
# Extract pixel hits from measurement Parquet (http_requests table)
# ---------------------------------------------------------------------------
# src/analysis/pixels.py
# (only the function that reads the enriched parquet changes)

def extract_pixels_from_parquet(
    http_requests_df: pd.DataFrame,
    url_col: str = "url",
    visit_id_col: str = "visit_id",
    top_url_col: str = "top_level_url",
    # NEW: enriched-schema columns
    parent_entity_col: str = "parent_entity",
    subsidiary_entity_col: str = "subsidiary_entity",
    relationship_tier_col: str = "relationship_tier",
    is_technical_3p_col: str = "is_technical_3p",
) -> pd.DataFrame:
    """
    Scan an http_requests_enriched DataFrame and return a long-format
    DataFrame of pixel hits, now with entity attribution.

    Returns columns:
        visit_id, top_level_url, top_level_parent_entity,
        request_url, request_parent_entity, request_subsidiary_entity,
        relationship_tier, is_technical_3p,
        pixel_type, matched_domain, path_confirmed
    """
    logger.info("Scanning %d requests for ad pixels...", len(http_requests_df))

    # NEW: Build a lookup of top_level_url -> parent_entity from the
    # dataframe itself, since top-level URLs also appear as requests.
    # This lets us attribute the *site* to its parent entity too.
    top_entity_map = (
        http_requests_df.dropna(subset=[top_url_col])
        .drop_duplicates(subset=[top_url_col])
        .set_index(top_url_col)[parent_entity_col]
        .to_dict()
    )

    hits = []
    has_enriched = parent_entity_col in http_requests_df.columns

    for row in http_requests_df.itertuples(index=False):
        url = str(getattr(row, url_col, None))
        hit = classify_pixel(url)
        if hit is None:
            continue

        top_url = getattr(row, top_url_col, None)
        top_url = str(top_url) if top_url else None

        record = {
            "visit_id": getattr(row, visit_id_col, None),
            "top_level_url": top_url,
            "top_level_parent_entity": (
                top_entity_map.get(top_url) if top_url else None
            ),
            "request_url": url,
            "pixel_type": hit.pixel_type,
            "matched_domain": hit.matched_domain,
            "path_confirmed": hit.matched_path_pattern is not None,
        }

        # NEW: enriched columns for the pixel-serving request
        if has_enriched:
            record["request_parent_entity"]      = getattr(row, parent_entity_col, None)
            record["request_subsidiary_entity"]  = getattr(row, subsidiary_entity_col, None)
            record["relationship_tier"]          = getattr(row, relationship_tier_col, None)
            record["is_technical_3p"]            = getattr(row, is_technical_3p_col, None)

        hits.append(record)

    df = pd.DataFrame(hits)
    if not df.empty:
        n_sites = df["top_level_parent_entity"].nunique()
        logger.info("Found %d ad-pixel hits across %d distinct parent entities.",
                    len(df), n_sites)
    return df


# ---------------------------------------------------------------------------
# Extract pixel hits from profile-build SQLite (OpenWPM standard schema)
# ---------------------------------------------------------------------------
def extract_pixels_from_sqlite(
    sqlite_path: str | Path,
    persona: str,
) -> pd.DataFrame:
    """
    Query an OpenWPM crawl-data.sqlite from a profile-build run and return
    ad-pixel hits observed during seeding.

    Returns the same long-format schema as extract_pixels_from_parquet,
    with an added `persona` column.
    """
    sqlite_path = Path(sqlite_path)
    if not sqlite_path.exists():
        raise FileNotFoundError(f"SQLite DB not found: {sqlite_path}")

    logger.info("Loading http_requests from %s", sqlite_path)
    query = """
        SELECT visit_id, url, top_level_url
        FROM http_requests
    """
    with sqlite3.connect(str(sqlite_path)) as conn:
        req_df = pd.read_sql_query(query, conn)

    hits_df = extract_pixels_from_parquet(req_df)
    hits_df.insert(0, "persona", persona)
    return hits_df


# ---------------------------------------------------------------------------
# Site-level aggregation
# ---------------------------------------------------------------------------
# src/analysis/pixels.py

def aggregate_pixels_by_site(
    pixel_hits: pd.DataFrame,
    group_cols: Iterable[str] = ("top_level_parent_entity",),
    target_pixels: Optional[list[str]] = None,
    require_path_confirmed: bool = False,
    only_third_party: bool = False,
    exclude_technical_3p: bool = False,
    relationship_tiers: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Collapse pixel hits to one row per site (grouped by parent_entity).
    """
    if pixel_hits.empty:
        return pd.DataFrame()

    hits = pixel_hits.copy()

    # --- Filters ---
    if target_pixels:
        hits = hits[hits["pixel_type"].isin(target_pixels)]

    if require_path_confirmed and "path_confirmed" in hits.columns:
        hits = hits[hits["path_confirmed"] == True]

    if only_third_party and {"top_level_parent_entity", "request_parent_entity"}.issubset(hits.columns):
        hits = hits[
            hits["top_level_parent_entity"].notna()
            & hits["request_parent_entity"].notna()
            & (hits["top_level_parent_entity"] != hits["request_parent_entity"])
        ]

    if exclude_technical_3p and "is_technical_3p" in hits.columns:
        hits = hits[hits["is_technical_3p"].fillna(0) < 0.5]

    if relationship_tiers and "relationship_tier" in hits.columns:
        hits = hits[hits["relationship_tier"].isin(relationship_tiers)]

    if hits.empty:
        logger.warning("No pixel hits after filtering.")
        return pd.DataFrame(columns=list(group_cols) + [
            "has_pixel", "n_pixel_hits", "n_distinct_pixel_types",
            "distinct_tracker_entities", "pixel_types",
            "has_meta", "has_tiktok", "has_doubleclick",
            "has_google_ads", "has_criteo",
        ])

    # --- Aggregation ---
    group_cols = list(group_cols)
    grouped = hits.groupby(group_cols)

    agg_kwargs = {
        "n_pixel_hits":            ("pixel_type", "size"),
        "n_distinct_pixel_types":  ("pixel_type", "nunique"),
        "pixel_types":             ("pixel_type", lambda s: ", ".join(sorted(set(s)))),
    }

    # CRITICAL: this is the column the SQL view is looking for
    if "request_parent_entity" in hits.columns:
        agg_kwargs["distinct_tracker_entities"] = ("request_parent_entity", "nunique")

    agg = grouped.agg(**agg_kwargs).reset_index()
    agg["has_pixel"] = True

    # Fallback: if request_parent_entity wasn't available, backfill with a zero column
    # so the SQL COALESCE(sp.distinct_tracker_entities, 0) never crashes
    if "distinct_tracker_entities" not in agg.columns:
        agg["distinct_tracker_entities"] = 0
        logger.warning(
            "request_parent_entity missing from pixel_hits; "
            "distinct_tracker_entities defaulted to 0."
        )

    # Per-platform flags from the UNFILTERED hits
    for flag_name, pixel_name in [
        ("has_meta",        "Meta Pixel"),
        ("has_tiktok",      "TikTok Pixel"),
        ("has_doubleclick", "DoubleClick"),
        ("has_google_ads",  "Google Ads Conversion"),
        ("has_criteo",      "Criteo"),
    ]:
        flags = (
            pixel_hits[pixel_hits["pixel_type"] == pixel_name]
            .groupby(group_cols).size().rename(flag_name)
        )
        agg = agg.merge(flags, on=group_cols, how="left")
        agg[flag_name] = agg[flag_name].fillna(0).astype(int).gt(0)

    return agg
# ---------------------------------------------------------------------------
# Convenience: build the full site-pixel table for a crawl
# ---------------------------------------------------------------------------
def build_site_pixel_table(
    http_requests_df: pd.DataFrame,
    crawl_name: str,
) -> pd.DataFrame:
    """
    End-to-end: from raw http_requests -> site-level pixel presence table
    ready to join against ads_enriched.
    """
    hits = extract_pixels_from_parquet(http_requests_df)
    if hits.empty:
        logger.warning("No pixel hits found for crawl %s", crawl_name)
        return pd.DataFrame(columns=["crawl_name", "top_level_parent_entity", "has_pixel"])
    site_table = aggregate_pixels_by_site(hits, group_cols=["top_level_parent_entity"])
    site_table.insert(0, "crawl_name", crawl_name)
    return site_table