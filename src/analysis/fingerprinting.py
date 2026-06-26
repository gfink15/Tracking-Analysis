"""
src/analysis/fingerprinting.py — Browser fingerprinting detection.

Fingerprinting is "stateless" tracking: instead of storing an ID in
a cookie, the tracker queries browser APIs that return values stable
enough to identify the user across sessions. The signals are:

  • Canvas fingerprinting    — render text/shapes to an offscreen
                                canvas, then read pixel data
  • WebGL fingerprinting     — query GPU details (renderer, vendor)
  • AudioContext fingerprinting — generate audio, analyze the output
  • Font enumeration         — measure rendered text dimensions
  • Navigator/Screen probing — read userAgent, platform, screen size

This module queries OpenWPM's `javascript` table for calls to APIs
associated with each of these signals. Detection is based on the
canonical methodology from Englehardt & Narayanan (2016).

References:
  Englehardt, S., & Narayanan, A. (2016). Online tracking: A 1-
  million-site measurement and analysis. CCS '16.
  https://webtransparency.cs.princeton.edu/webcensus/
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from config import PROFILES
from src.utils.db import db_session


# ─────────────────────────────────────────────────────────────────────
# FINGERPRINTING API SIGNATURES
# ─────────────────────────────────────────────────────────────────────
# Each technique is identified by JavaScript symbols (API calls).
# The 'symbol' column in OpenWPM's javascript table contains values
# like 'HTMLCanvasElement.toDataURL' or 'window.navigator.userAgent'.
#
# Sources: Englehardt & Narayanan 2016, FP-Inspector (Iqbal et al.
# 2021), and the EFF's Cover Your Tracks methodology.

CANVAS_SYMBOLS = (
    'HTMLCanvasElement.toDataURL',
    'HTMLCanvasElement.toBlob',
    'CanvasRenderingContext2D.getImageData',
    'CanvasRenderingContext2D.fillText',
    'CanvasRenderingContext2D.strokeText',
    'CanvasRenderingContext2D.measureText',
)

WEBGL_SYMBOLS = (
    'WebGLRenderingContext.getParameter',
    'WebGLRenderingContext.getSupportedExtensions',
    'WebGLRenderingContext.getExtension',
    'WebGL2RenderingContext.getParameter',
)

AUDIO_SYMBOLS = (
    'AudioContext.createOscillator',
    'AudioContext.createDynamicsCompressor',
    'AudioContext.createAnalyser',
    'OfflineAudioContext.startRendering',
    'AudioBuffer.getChannelData',
)

# Navigator/screen probes — individually low signal, but counting
# how many of them a single script reads is highly diagnostic of
# fingerprinting (legitimate code reads 1-2; fingerprinters read 10+).
NAVIGATOR_SYMBOLS = (
    'window.navigator.userAgent',
    'window.navigator.platform',
    'window.navigator.language',
    'window.navigator.languages',
    'window.navigator.hardwareConcurrency',
    'window.navigator.deviceMemory',
    'window.navigator.maxTouchPoints',
    'window.navigator.vendor',
    'window.navigator.appVersion',
    'window.navigator.doNotTrack',
    'window.navigator.cookieEnabled',
    'window.navigator.plugins',
    'window.navigator.mimeTypes',
)

SCREEN_SYMBOLS = (
    'window.screen.width',
    'window.screen.height',
    'window.screen.availWidth',
    'window.screen.availHeight',
    'window.screen.colorDepth',
    'window.screen.pixelDepth',
)

# Font enumeration: measureText is the canonical trick. A script
# calling it 50+ times with different font-family strings is almost
# certainly enumerating installed fonts.
FONT_SYMBOLS = (
    'CanvasRenderingContext2D.measureText',
    'CanvasRenderingContext2D.font',
)

# Maps technique name → its symbol tuple. Used for batch queries.
TECHNIQUE_SYMBOLS = {
    'canvas':     CANVAS_SYMBOLS,
    'webgl':      WEBGL_SYMBOLS,
    'audio':      AUDIO_SYMBOLS,
    'navigator':  NAVIGATOR_SYMBOLS,
    'screen':     SCREEN_SYMBOLS,
    'font':       FONT_SYMBOLS,
}


def _symbols_in_clause(symbols: tuple[str, ...]) -> str:
    """Convert a tuple of symbols into a SQL IN clause."""
    quoted = ", ".join(f"'{s}'" for s in symbols)
    return f"({quoted})"


# ─────────────────────────────────────────────────────────────────────
# RAW API CALL COUNTS
# ─────────────────────────────────────────────────────────────────────
def fingerprinting_api_calls() -> pd.DataFrame:
    """Total calls per fingerprinting symbol, broken down by profile.

    This is the raw signal: which fingerprinting APIs were called,
    how often, on which profiles. Useful as a sanity check before
    moving to script-level detection.

    Returns: profile, technique, symbol, n_calls.
    """
    # Build one big UNION ALL across techniques so we can return
    # everything in one query. Faster than N round-trips.
    union_parts = [
        f"""
        SELECT
            profile,
            '{technique}' AS technique,
            symbol,
            COUNT(*) AS n_calls
        FROM javascript
        WHERE symbol IN {_symbols_in_clause(symbols)}
        GROUP BY profile, symbol
        """
        for technique, symbols in TECHNIQUE_SYMBOLS.items()
    ]
    full_sql = " UNION ALL ".join(union_parts) + " ORDER BY profile, technique, n_calls DESC"

    with db_session(read_only=True) as con:
        return con.execute(full_sql).df()


# ─────────────────────────────────────────────────────────────────────
# SCRIPT-LEVEL DETECTION (the methodologically correct approach)
# ─────────────────────────────────────────────────────────────────────
# A single API call doesn't make a fingerprinter — legitimate code
# uses canvas, audio, etc. for many reasons. The Englehardt-Narayanan
# methodology requires that a SCRIPT exhibits multiple behaviors
# characteristic of fingerprinting before being flagged.

def detect_canvas_fingerprinters(min_text_calls: int = 1) -> pd.DataFrame:
    """Detect scripts performing canvas fingerprinting.

    Englehardt & Narayanan criteria for canvas fingerprinting:
      1. Script writes text to canvas (fillText OR strokeText)
      2. Script reads canvas pixels (toDataURL OR getImageData)
      3. (Optional) Script does NOT call save/restore (legitimate
         drawing usually saves state; fingerprinters skip this)

    Returns: profile, script_url, n_text_calls, n_read_calls, n_visits.
    """
    with db_session(read_only=True) as con:
        return con.execute(f"""
            WITH per_script AS (
                SELECT
                    profile,
                    script_url,
                    visit_id,
                    SUM(CASE WHEN symbol IN
                        ('CanvasRenderingContext2D.fillText',
                         'CanvasRenderingContext2D.strokeText')
                        THEN 1 ELSE 0 END) AS n_text_calls,
                    SUM(CASE WHEN symbol IN
                        ('HTMLCanvasElement.toDataURL',
                         'CanvasRenderingContext2D.getImageData')
                        THEN 1 ELSE 0 END) AS n_read_calls
                FROM javascript
                WHERE symbol IN
                    ('CanvasRenderingContext2D.fillText',
                     'CanvasRenderingContext2D.strokeText',
                     'HTMLCanvasElement.toDataURL',
                     'CanvasRenderingContext2D.getImageData')
                GROUP BY profile, script_url, visit_id
            )
            SELECT
                profile,
                script_url,
                SUM(n_text_calls)         AS n_text_calls,
                SUM(n_read_calls)         AS n_read_calls,
                COUNT(DISTINCT visit_id)  AS n_visits
            FROM per_script
            WHERE n_text_calls >= {min_text_calls}
              AND n_read_calls >= 1
            GROUP BY profile, script_url
            ORDER BY n_visits DESC, n_read_calls DESC
        """).df()


def detect_audio_fingerprinters() -> pd.DataFrame:
    """Detect scripts performing audio fingerprinting.

    Criteria: script creates an oscillator/compressor AND reads
    channel data. Audio fingerprinting is rarer than canvas but
    when present is almost never a false positive — legitimate
    audio code is very rarely AnalyserNode+createOscillator+
    getChannelData in one script.

    Returns: profile, script_url, n_visits.
    """
    with db_session(read_only=True) as con:
        return con.execute("""
            WITH per_script AS (
                SELECT
                    profile,
                    script_url,
                    visit_id,
                    SUM(CASE WHEN symbol IN
                        ('AudioContext.createOscillator',
                         'AudioContext.createDynamicsCompressor',
                         'OfflineAudioContext.startRendering')
                        THEN 1 ELSE 0 END) AS n_setup,
                    SUM(CASE WHEN symbol = 'AudioBuffer.getChannelData'
                        THEN 1 ELSE 0 END) AS n_read
                FROM javascript
                WHERE symbol IN
                    ('AudioContext.createOscillator',
                     'AudioContext.createDynamicsCompressor',
                     'OfflineAudioContext.startRendering',
                     'AudioBuffer.getChannelData')
                GROUP BY profile, script_url, visit_id
            )
            SELECT
                profile,
                script_url,
                COUNT(DISTINCT visit_id) AS n_visits
            FROM per_script
            WHERE n_setup >= 1 AND n_read >= 1
            GROUP BY profile, script_url
            ORDER BY n_visits DESC
        """).df()


def detect_navigator_probers(min_attributes: int = 5) -> pd.DataFrame:
    """Detect scripts that read many navigator/screen attributes.

    A page's analytics code might read 1-2 navigator properties.
    A fingerprinter reads 8+ in quick succession. This function
    counts the number of DISTINCT navigator/screen attributes each
    script accesses per visit, flagging those above a threshold.

    Args:
        min_attributes: Minimum distinct attributes for the script
            to be flagged. Default 5 catches most fingerprinters
            with low false-positive rate.

    Returns: profile, script_url, n_attributes_read, attributes_list.
    """
    all_symbols = NAVIGATOR_SYMBOLS + SCREEN_SYMBOLS
    with db_session(read_only=True) as con:
        return con.execute(f"""
            SELECT
                profile,
                script_url,
                COUNT(DISTINCT symbol)        AS n_attributes_read,
                string_agg(DISTINCT symbol, ', ') AS attributes_list,
                COUNT(DISTINCT visit_id)      AS n_visits
            FROM javascript
            WHERE symbol IN {_symbols_in_clause(all_symbols)}
            GROUP BY profile, script_url
            HAVING COUNT(DISTINCT symbol) >= {min_attributes}
            ORDER BY n_attributes_read DESC, n_visits DESC
        """).df()


# ─────────────────────────────────────────────────────────────────────
# SUMMARY: how many fingerprinters per profile?
# ─────────────────────────────────────────────────────────────────────
def fingerprinter_summary() -> pd.DataFrame:
    """Per-profile summary of detected fingerprinting activity.

    Combines results from the three detectors into one row-per-profile
    summary. This is the headline figure for fingerprinting analysis:
    "the shopping profile encountered N canvas fingerprinters across
    M visits, etc."
    """
    canvas = detect_canvas_fingerprinters()
    audio = detect_audio_fingerprinters()
    navigator = detect_navigator_probers()

    rows = []
    for profile in PROFILES:
        rows.append({
            'profile': profile,
            'n_canvas_fp_scripts':
                int(canvas[canvas['profile'] == profile]['script_url'].nunique()),
            'n_audio_fp_scripts':
                int(audio[audio['profile'] == profile]['script_url'].nunique()),
            'n_navigator_fp_scripts':
                int(navigator[navigator['profile'] == profile]['script_url'].nunique()),
            'canvas_fp_visits':
                int(canvas[canvas['profile'] == profile]['n_visits'].sum()),
            'audio_fp_visits':
                int(audio[audio['profile'] == profile]['n_visits'].sum()),
            'navigator_fp_visits':
                int(navigator[navigator['profile'] == profile]['n_visits'].sum()),
        })
    return pd.DataFrame(rows)


def fingerprinter_top_scripts(top_n: int = 20) -> pd.DataFrame:
    """Most prevalent fingerprinting scripts across all profiles.

    Useful for the "who's doing this?" question. The same handful
    of scripts (FingerprintJS, Imperva, PerimeterX, etc.) tend to
    dominate. Knowing the top offenders lets you contextualize
    your findings with the broader literature.
    """
    canvas = detect_canvas_fingerprinters()
    canvas['technique'] = 'canvas'
    audio = detect_audio_fingerprinters()
    audio['technique'] = 'audio'
    navigator = detect_navigator_probers()
    navigator['technique'] = 'navigator'

    combined = pd.concat([
        canvas[['profile', 'script_url', 'n_visits', 'technique']],
        audio[['profile', 'script_url', 'n_visits', 'technique']],
        navigator[['profile', 'script_url', 'n_visits', 'technique']],
    ], ignore_index=True)

    return (combined.groupby(['script_url', 'technique'])
                    .agg(total_visits=('n_visits', 'sum'),
                         profiles_seen=('profile', 'nunique'))
                    .reset_index()
                    .sort_values('total_visits', ascending=False)
                    .head(top_n))


if __name__ == "__main__":
    print("Fingerprinting API call counts:")
    print(fingerprinting_api_calls().head(30).to_string(index=False))

    print("\nCanvas fingerprinter detection:")
    print(detect_canvas_fingerprinters().head(15).to_string(index=False))

    print("\nFingerprinter summary by profile:")
    print(fingerprinter_summary().to_string(index=False))

    print("\nTop 20 fingerprinting scripts:")
    print(fingerprinter_top_scripts().to_string(index=False))