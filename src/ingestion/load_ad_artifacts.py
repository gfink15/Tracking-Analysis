"""
src/ingestion/load_ad_artifacts.py — Process ad scraping outputs into Parquet.

Version 3.1 — aligned with ad_capture.py v3.1 output schema.

Key features:
    - Tiered detection metadata (high/medium confidence)
    - Three-way network cross-verification (src regex × capture metadata × outerHTML)
    - Disagreement logging for methodology audit trail
    - Skips capture sidecars such as _visit_summary.json and _ad_content.json
    - Schema version tracking for reproducibility
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Any

import duckdb
import pandas as pd

# Path setup
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from config import DATA_DIR, PARQUET_DIR, PROFILES

try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════
# ADVERTISER NETWORK DETECTION
# ═══════════════════════════════════════════════════════════════════════
# Patterns ordered by specificity — more specific patterns first so we
# don't classify a doubleclick URL as "google_other" by accident.
AD_NETWORK_PATTERNS = [
    # Google family (specific → general)
    (r'safeframe\.googlesyndication\.com', 'google_adsense'),
    (r'googlesyndication\.com',            'google_adsense'),
    (r'doubleclick\.net',                  'google_doubleclick'),
    (r'2mdn\.net',                         'google_doubleclick'),
    (r'googleadservices\.com',             'google_adservices'),
    (r'googletagservices\.com',            'google_tagmanager'),
    (r'google\.com/ads',                   'google_other'),
    # Other major networks
    (r'amazon-adsystem\.com',              'amazon'),
    (r'facebook\.com.*plugins',            'facebook'),
    (r'criteo\.(com|net)',                 'criteo'),
    (r'taboola\.com',                      'taboola'),
    (r'outbrain\.com',                     'outbrain'),
    (r'adnxs\.com',                        'appnexus'),
    (r'pubmatic\.com',                     'pubmatic'),
    (r'rubiconproject\.com',               'rubicon'),
    (r'openx\.net',                        'openx'),
    (r'adsrvr\.org',                       'thetradedesk'),
    (r'media\.net',                        'medianet'),
    (r'yahoo\.com.*ad',                    'yahoo'),
    (r'bing\.com.*ad',                     'microsoft_bing'),
]

_COMPILED_PATTERNS = [
    (re.compile(pattern, re.IGNORECASE), network)
    for pattern, network in AD_NETWORK_PATTERNS
]


def _match_network_in_text(text: Optional[str]) -> Optional[str]:
    """Return the first matching network name found in `text`, or None."""
    if not text:
        return None
    for pattern, network in _COMPILED_PATTERNS:
        if pattern.search(text):
            return network
    return None


def identify_ad_network(
    src: Optional[str],
    outer_html: Optional[str] = None,
    meta_network: Optional[str] = None,
) -> tuple[str, str, bool]:
    """
    Classify an ad network using three signals:
      1. Regex on the iframe `src` attribute (highest priority — actual URL)
      2. Regex on the captured `outerHTML` (catches nested iframe srcs)
      3. The capture-time `network` label from ad_capture.py (CSS-selector based)

    Returns (final_network, verification_source, networks_agree).

    `verification_source` is one of:
      - 'src_regex'        : matched from iframe.src
      - 'outerhtml_regex'  : matched from outerHTML (often nested iframes)
      - 'capture_metadata' : fell back to capture-time label
      - 'unknown'          : no signal matched
    """
    src_network = _match_network_in_text(src)
    html_network = _match_network_in_text(outer_html)

    # Priority 1: src regex wins if present
    if src_network:
        # Check agreement with capture metadata (for audit logging)
        agrees = (meta_network is None
                  or _networks_compatible(src_network, meta_network))
        return src_network, 'src_regex', agrees

    # Priority 2: outerHTML regex (catches nested ad-network iframes)
    if html_network:
        agrees = (meta_network is None
                  or _networks_compatible(html_network, meta_network))
        return html_network, 'outerhtml_regex', agrees

    # Priority 3: trust the capture-time label
    if meta_network and meta_network not in ('unknown', 'none', ''):
        return meta_network, 'capture_metadata', True

    return 'unknown', 'unknown', False


# Networks that are part of the same family — used to determine whether
# disagreement between sources is meaningful or just a labeling variation.
_NETWORK_FAMILIES = {
    'google_adsense': 'google',
    'google_doubleclick': 'google',
    'google_adservices': 'google',
    'google_tagmanager': 'google',
    'google_other': 'google',
}


def _networks_compatible(a: str, b: str) -> bool:
    """Two network labels are compatible if they're identical or in the
    same family (e.g., google_adsense vs google_doubleclick)."""
    if a == b:
        return True
    return _NETWORK_FAMILIES.get(a) == _NETWORK_FAMILIES.get(b) is not None


# ═══════════════════════════════════════════════════════════════════════
# DATA STRUCTURE
# ═══════════════════════════════════════════════════════════════════════
# Sidecar JSON filenames written by ad_capture.py v3.1 that should NOT
# be processed as individual ads.
SIDECAR_FILES = {
    "_visit_summary.json",
    "_iframe_content.json",
}


@dataclass
class ProcessedAd:
    """One row in the final ads.parquet table (schema v3.0)."""
    profile: str
    visit_id: int
    page_url: str
    ad_hash: str
    ad_src: Optional[str]
    ad_tag: Optional[str]
    ad_id: Optional[str]
    ad_width: float
    ad_height: float
    ad_x: float
    ad_y: float
    # Network classification
    advertiser_network: str           # Final verified network
    capture_network: str              # What capture tool labeled it
    verification_source: str          # How final network was determined
    networks_agree: bool              # True if all sources agree
    # Detection provenance
    confidence: str                   # 'high' or 'medium'
    matched_selector: str             # CSS selector that triggered capture
    schema_version: str
    # Content
    ocr_text: str
    ocr_char_count: int
    has_screenshot: bool
    timestamp: float


# ═══════════════════════════════════════════════════════════════════════
# PROCESSING
# ═══════════════════════════════════════════════════════════════════════
def _extract_ocr_text(png_path: Path) -> str:
    """Extract OCR text from an ad screenshot, normalized to single spaces."""
    if not OCR_AVAILABLE or not png_path.exists():
        return ""
    try:
        with Image.open(png_path) as img: # type: ignore
            if img.width < 40 or img.height < 40:
                return ""
            text = pytesseract.image_to_string(img) # type: ignore
            return re.sub(r'\s+', ' ', text).strip()
    except Exception:
        return ""


def _process_one_ad(
    json_path: Path,
    profile: str,
    disagreements: Optional[list[dict]] = None,
) -> list[ProcessedAd]:
    """Process one ad's JSON + PNG into ProcessedAd record(s).

    Compatible with ad_capture v3.1 schema. Handles three file types
    that v3.1 writes into each visit directory:

      • <ad_hash>.json        — actual ad record (dict). Processed.
      • _visit_summary.json   — capture stats (dict). Skipped.
      • _ad_content.json      — iframe content payload (often list). Skipped.

    Also defensively handles legacy list-shaped payloads and other
    unexpected structures without crashing.
    """
    # ── Skip non-ad sidecar files by name ──
    # These are v3.1 metadata/content files, not individual ad records.
    if json_path.name.startswith("_"):
        return []

    # ── Read + parse JSON ──
    try:
        raw = json_path.read_text()
        meta = json.loads(raw)
    except (json.JSONDecodeError, OSError) as e:
        print(f"  ⚠  Skipping malformed JSON {json_path}: {e}")
        return []

    # ── Handle double-encoded JSON (string-of-JSON) ──
    # Defensive guard for any future schema quirks.
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except json.JSONDecodeError:
            return []

    # ── Normalize to a list of dicts for uniform processing ──
    # v3.1 writes single dicts, but we accept lists too for robustness
    # against legacy data or future batch-write scenarios.
    if isinstance(meta, dict):
        records = [meta]
    elif isinstance(meta, list):
        records = [m for m in meta if isinstance(m, dict)]
    else:
        # Unknown shape — skip silently rather than crash.
        return []

    # ── Process each record ──
    png_path = json_path.with_suffix(".png")
    ocr_text = _extract_ocr_text(png_path)
    has_png = png_path.exists()

    processed: list[ProcessedAd] = []
    for i, rec in enumerate(records):
        # Defense in depth: even after the list filter above, double-check.
        if not isinstance(rec, dict):
            continue

        # Skip records that explicitly mark themselves as non-ad.
        # v3.1 doesn't write this field, but future versions might.
        if rec.get("record_type") not in (None, "ad"):
            continue

        # ── Extract nested ad_metadata ──
        # In v3.1, ad fields live inside rec["ad_metadata"], NOT at top level.
        ad_meta = rec.get("ad_metadata", {})
        if not isinstance(ad_meta, dict):
            ad_meta = {}

        rect = ad_meta.get("rect") or rec.get("rect") or {}
        if not isinstance(rect, dict):
            rect = {}
        src = ad_meta.get("src") or rec.get("src")
        outer_html = (
            ad_meta.get("outerHTML")
            or ad_meta.get("outer_html")
            or rec.get("outerHTML")
            or rec.get("outer_html")
        )

        # ── Generate ad_hash from stem ──
        # v3.1 derives ad_hash from md5(url + marker) at capture time
        # and uses it as the FILENAME. So json_path.stem IS the ad_hash.
        # For batch files (legacy), append index to disambiguate.
        ad_hash = json_path.stem if len(records) == 1 \
            else f"{json_path.stem}_{i}"

        # ── Timestamp normalization ──
        # v3.1 writes time.time() (float epoch seconds).
        try:
            timestamp = float(rec.get("timestamp") or 0.0)
        except (TypeError, ValueError):
            timestamp = 0.0

        # ── Advertiser network ──
        # Keep the capture-time label for provenance, but re-verify it
        # against the src / outerHTML signals when they are available.
        capture_network = (
            ad_meta.get("network")
            or rec.get("network")
            or "unknown"
        )
        final_network, verification_source, networks_agree = identify_ad_network(
            src,
            outer_html=outer_html,
            meta_network=capture_network,
        )

        if disagreements is not None and not networks_agree:
            disagreements.append({
                "profile": profile,
                "visit_id": int(rec.get("visit_id", -1)),
                "page_url": rec.get("page_url", ""),
                "ad_hash": ad_hash,
                "capture_network": capture_network,
                "regex_network": final_network,
                "verification_source": verification_source,
                "matched_selector": ad_meta.get("matched_selector", ""),
                "src": src,
                "outer_html": outer_html,
            })

        ad_id = ad_meta.get("id") or rec.get("ad_id") or rec.get("id")

        processed.append(ProcessedAd(
            profile=profile,
            visit_id=int(rec.get("visit_id", -1)),
            page_url=rec.get("page_url", ""),
            ad_hash=ad_hash,
            ad_src=src,
            ad_tag=ad_meta.get("tag"),
            ad_id=ad_id,
            ad_width=float(rect.get("w", 0) or 0),
            ad_height=float(rect.get("h", 0) or 0),
            ad_x=float(rect.get("x", 0) or 0),
            ad_y=float(rect.get("y", 0) or 0),
            advertiser_network=final_network,
            capture_network=capture_network,
            verification_source=verification_source,
            networks_agree=networks_agree,
            ocr_text=ocr_text,
            ocr_char_count=len(ocr_text),
            has_screenshot=has_png,
            timestamp=timestamp,
            # v3.1 fields — extract from ad_metadata
            confidence=ad_meta.get("confidence", "unknown"),
            matched_selector=ad_meta.get("matched_selector", ""),
            schema_version="3.1",   # hardcoded — v3.1 doesn't write this itself
        ))

    return processed


def load_ad_artifacts() -> None:
    """Walk every profile's ads/ directory and emit ads.parquet."""
    print("─" * 70)
    print("Ad Artifact Ingestion — Schema v3.0 (compatible with capture v3.1)")
    print("─" * 70)

    all_ads: list[ProcessedAd] = []
    all_disagreements: list[dict] = []
    skipped_sidecars = 0
    malformed = 0

    for profile in PROFILES:
        ads_dir = DATA_DIR / profile / "ads"
        if not ads_dir.exists():
            print(f"  {profile}: no ads/ directory, skipping")
            continue

        json_files = list(ads_dir.rglob("*.json"))
        print(f"  {profile}: processing {len(json_files):,} JSON files...")

        before_count = len(all_ads)
        for json_path in json_files:
            if json_path.name in SIDECAR_FILES:
                skipped_sidecars += 1
                continue
            ads = _process_one_ad(json_path, profile, all_disagreements)
            if not ads:
                malformed += 1
            else:
                all_ads.extend(ads)
        added = len(all_ads) - before_count
        print(f"           → {added:,} ads ingested")

    if not all_ads:
        print("\n⚠  No ads found. Exiting.")
        return

    df = pd.DataFrame([asdict(ad) for ad in all_ads])
    output_path = PARQUET_DIR / "ads.parquet"

    con = duckdb.connect(":memory:")
    con.register('ads_df', df)
    con.execute(
        f"COPY (SELECT * FROM ads_df) TO '{output_path}' "
        f"(FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    con.close()

    # ── Summary report ──
    print(f"\n✓ Wrote {len(all_ads):,} ads to {output_path}")
    print(f"  Skipped sidecar files: {skipped_sidecars:,}")
    print(f"  Malformed/unprocessable: {malformed:,}")

    print("\n📊 Network classification by confidence tier:")
    print(
        df.groupby(['confidence', 'advertiser_network'])
          .size()
          .unstack(fill_value=0)
          .to_string()
    )

    print("\n🔍 Verification source breakdown:")
    print(df['verification_source'].value_counts().to_string())

    print("\n🤝 Source agreement:")
    agree_count = df['networks_agree'].sum()
    total = len(df)
    pct = 100 * agree_count / total if total else 0
    print(f"  Sources agreed: {agree_count:,} / {total:,} ({pct:.1f}%)")

    # ── Disagreement audit ──
    if all_disagreements:
        disagreement_path = PARQUET_DIR / "ads_network_disagreements.json"
        disagreement_path.write_text(
            json.dumps(all_disagreements, indent=2)
        )
        print(f"\n⚠  {len(all_disagreements):,} network disagreements logged "
              f"to {disagreement_path}")
        # Show a sample of what disagreed
        print("\n  Sample disagreements (first 5):")
        for d in all_disagreements[:5]:
            print(f"    • {d['page_url'][:50]}")
            print(f"        capture={d['capture_network']:25s} "
                  f"→ regex={d['regex_network']}")
            print(f"        selector={d['matched_selector']}")

    print("─" * 70)


if __name__ == "__main__":
    load_ad_artifacts()
