"""
src/analysis/ad_clip.py
────────────────────────
CLIP-based semantic classification of captured ad images.

Fits into the Tracking-Analysis Medallion architecture:
  - Reads:   artifacts/parquet/ads.parquet         (Silver)
  - Writes:  artifacts/parquet/ads_clip.parquet    (Gold-adjacent)
  - Joins:   on 'ad_hash' back to ads.parquet

The output Parquet is deliberately narrow — only ad_hash + CLIP columns —
so it can be joined via DuckDB rather than duplicating ad metadata.

Usage (from repo root):
    python -m src.analysis.ad_clip --persona-labels shopper
    python -m src.analysis.ad_clip --persona-labels shopper --limit 50   # smoke test

Or from a notebook:
    from src.analysis.ad_clip import run_clip_pipeline
    df = run_clip_pipeline(persona_labels="shopper")
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError


try:
    import clip  # openai/CLIP
except ImportError as e:
    raise ImportError(
        "CLIP not installed. From your (openwpm) env, run:\n"
        "  pip install ftfy regex tqdm\n"
        "  pip install git+https://github.com/openai/CLIP.git"
    ) from e

# Project-local imports — matches your existing pattern in src/analysis/*
import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("ad_clip")


# ─────────────────────────────────────────────────────────────────────
# PATHS (derived from config.py — adjust names if yours differ)
# ─────────────────────────────────────────────────────────────────────
PARQUET_DIR = Path(getattr(config, "PARQUET_DIR", "artifacts/parquet"))
ADS_PARQUET = PARQUET_DIR / "ads.parquet"
ADS_CLIP_PARQUET = PARQUET_DIR / "ads_clip.parquet"

# Root containing per-profile ad PNGs. Adjust if config.py names it differently.
IMAGE_ROOT = Path(getattr(config, "DATA_DIR", "data"))


# ─────────────────────────────────────────────────────────────────────
# PERSONA-AWARE LABEL SETS
# Tuned to your shopper HistoryGenerator seed. Iterate on wording after
# your first run — CLIP is very sensitive to prompt phrasing.
# ─────────────────────────────────────────────────────────────────────
LABEL_SETS: Dict[str, List[str]] = {
    "shopper": [
        "a retail advertisement for clothing or shoes",
        "an advertisement for a specific product with a price or discount",
        "an electronics or gadget advertisement",
        "an advertisement for household or home goods",
        "a beauty or personal care advertisement",
        "a travel or hotel booking advertisement",
        "a food delivery or grocery advertisement",
        "a financial service, insurance, or credit card advertisement",
        "a generic brand logo with no product",
        "a public service announcement or news headline",
        "a blank, broken, or placeholder image",
    ],
    "control": [
        "a retail or e-commerce advertisement",
        "a financial service advertisement",
        "a travel advertisement",
        "a technology or software advertisement",
        "a healthcare or pharmaceutical advertisement",
        "an entertainment or media advertisement",
        "a generic brand logo with no product",
        "a public service announcement or news headline",
        "a blank, broken, or placeholder image",
    ],
}

NOISE_LABELS = {"a blank, broken, or placeholder image"}


# ─────────────────────────────────────────────────────────────────────
# CLIP CLASSIFIER
# ─────────────────────────────────────────────────────────────────────
@dataclass
class ClipResult:
    top_label: str
    top_confidence: float
    second_label: Optional[str]
    second_confidence: Optional[float]
    is_noise: bool
    is_low_confidence: bool


class ClipClassifier:
    def __init__(
        self,
        labels: List[str],
        model_name: str = "ViT-B/32",
        confidence_threshold: float = 0.5,
        device: Optional[str] = None,
    ):
        self.labels = labels
        self.confidence_threshold = confidence_threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        log.info("Loading CLIP %s on %s", model_name, self.device)

        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()

        self.text_inputs = clip.tokenize(labels).to(self.device)
        with torch.no_grad():
            self.text_features = self.model.encode_text(self.text_inputs)
            self.text_features /= self.text_features.norm(dim=-1, keepdim=True)

    def classify(self, image_path: Path) -> Optional[ClipResult]:
        try:
            img = Image.open(image_path).convert("RGB")
        except (UnidentifiedImageError, FileNotFoundError, OSError) as e:
            log.debug("Could not open %s: %s", image_path.name, e)
            return None

        if img.size[0] < 10 or img.size[1] < 10:
            return ClipResult(
                top_label="a blank, broken, or placeholder image",
                top_confidence=1.0,
                second_label=None,
                second_confidence=None,
                is_noise=True,
                is_low_confidence=False,
            )

        # Ensure preprocess result is a torch.Tensor before unsqueezing.
        prepped = self.preprocess(img)
        if isinstance(prepped, torch.Tensor):
            image_input = prepped.unsqueeze(0).to(self.device)
        else:
            # Fallback: convert PIL-like output to tensor
            import numpy as _np

            arr = _np.array(prepped)
            # If grayscale, expand dims to HWC
            if arr.ndim == 2:
                arr = _np.stack([arr] * 3, axis=-1)
            tensor = torch.from_numpy(arr).permute(2, 0, 1).float()
            image_input = tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            logits = 100.0 * image_features @ self.text_features.T
            probs = logits.softmax(dim=-1).cpu().numpy()[0]

        ranked = sorted(zip(self.labels, probs.tolist()),
                        key=lambda x: x[1], reverse=True)
        top_label, top_conf = ranked[0]
        second_label, second_conf = ranked[1] if len(ranked) > 1 else (None, None)

        return ClipResult(
            top_label=top_label,
            top_confidence=float(top_conf),
            second_label=second_label,
            second_confidence=float(second_conf) if second_conf is not None else None,
            is_noise=top_label in NOISE_LABELS,
            is_low_confidence=top_conf < self.confidence_threshold,
        )


# ─────────────────────────────────────────────────────────────────────
# IMAGE PATH RESOLUTION
# ─────────────────────────────────────────────────────────────────────
def resolve_image_path(row: pd.Series, image_root: Path) -> Optional[Path]:
    if pd.isna(row.get("png_path")) or not row["png_path"]:
        return None
    candidate = image_root / row["png_path"]
    return candidate if candidate.exists() else None


# ─────────────────────────────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────────────────────────────
def run_clip_pipeline(
    persona_labels: str = "shopper",
    confidence_threshold: float = 0.5,
    limit: Optional[int] = None,
    input_parquet: Path = ADS_PARQUET,
    output_parquet: Path = ADS_CLIP_PARQUET,
    image_root: Path = IMAGE_ROOT,
) -> pd.DataFrame:
    """
    Main entry point. Loads ads.parquet, classifies each image, writes
    ads_clip.parquet, returns the classification DataFrame.
    """
    if persona_labels not in LABEL_SETS:
        raise ValueError(f"Unknown label set '{persona_labels}'. "
                         f"Available: {list(LABEL_SETS.keys())}")

    log.info("Loading %s", input_parquet)
    ads = pd.read_parquet(input_parquet)
    log.info("Loaded %d ad records", len(ads))

    if "ad_hash" not in ads.columns:
        raise KeyError(
            "ads.parquet must contain an 'ad_hash' column for join-back. "
            "Update load_ad_artifacts.py to emit one if missing."
        )

    if limit:
        ads = ads.head(limit)
        log.info("SMOKE TEST MODE: classifying first %d ads only", limit)

    classifier = ClipClassifier(
        labels=LABEL_SETS[persona_labels],
        confidence_threshold=confidence_threshold,
    )

    rows = []
    missing = 0
    for i, (_, row) in enumerate(ads.iterrows(), start=1):
        img_path = resolve_image_path(row, image_root)
        if img_path is None:
            missing += 1
            rows.append({
                "profile": row["profile"],
                "visit_id": row["visit_id"],
                "ad_hash": row["ad_hash"],
                "clip_top_label": None,
                "clip_top_confidence": None,
                "clip_second_label": None,
                "clip_second_confidence": None,
                "clip_is_noise": None,
                "clip_is_low_confidence": None,
                "clip_status": "image_not_found",
            })
            continue

        r = classifier.classify(img_path)
        if r is None:
            rows.append({
                "profile": row["profile"],
                "visit_id": row["visit_id"],
                "ad_hash": row["ad_hash"],
                "clip_top_label": None,
                "clip_top_confidence": 0.0,
                "clip_second_label": None,
                "clip_second_confidence": None,
                "clip_is_noise": True,
                "clip_is_low_confidence": True,
                "clip_status": "unreadable",
            })
            continue

        rows.append({
            "profile": row["profile"],
            "visit_id": row["visit_id"],
            "ad_hash": row["ad_hash"],
            "clip_top_label": r.top_label,
            "clip_top_confidence": r.top_confidence,
            "clip_second_label": r.second_label,
            "clip_second_confidence": r.second_confidence,
            "clip_is_noise": r.is_noise,
            "clip_is_low_confidence": r.is_low_confidence,
            "clip_status": "ok",
        })

        if i % 50 == 0:
            log.info("Classified %d/%d", i, len(ads))

    if missing:
        log.warning("Could not resolve %d image paths — check IMAGE_ROOT in config.py",
                    missing)

    clip_df = pd.DataFrame(rows)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    clip_df.to_parquet(output_parquet, index=False)
    log.info("Wrote %d rows → %s", len(clip_df), output_parquet)

    _print_summary(clip_df, ads)
    return clip_df


def _print_summary(clip_df: pd.DataFrame, ads: pd.DataFrame) -> None:
    log.info("─" * 60)
    log.info("CLIP CLASSIFICATION SUMMARY")
    log.info("─" * 60)
    total = len(clip_df)
    noise = clip_df["clip_is_noise"].fillna(False).sum()
    low_conf = clip_df["clip_is_low_confidence"].fillna(False).sum()
    log.info("Total ads:              %d", total)
    log.info("Noise/blank:            %d (%.1f%%)", noise, 100 * noise / total)
    log.info("Low confidence:         %d (%.1f%%)", low_conf, 100 * low_conf / total)

    # Join back to ads to show per-profile breakdown
    if "profile" in ads.columns:
        merged = ads[["ad_hash", "profile"]].merge(clip_df, on="ad_hash", how="inner")
        usable = merged[
            ~merged["clip_is_noise"].fillna(True)
            & ~merged["clip_is_low_confidence"].fillna(True)
        ]
        log.info("Usable classifications: %d", len(usable))
        log.info("\nTop labels by profile:")
        for profile, sub in usable.groupby("profile"):
            log.info("  [%s]", profile)
            for label, n in sub["clip_top_label"].value_counts().head(5).items():
                log.info("    %4d  %s", n, label)


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="CLIP-classify captured ads")
    parser.add_argument("--persona-labels", default="shopper",
                        choices=list(LABEL_SETS.keys()))
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=None,
                        help="Classify only first N ads (smoke test)")
    args = parser.parse_args()

    run_clip_pipeline(
        persona_labels=args.persona_labels,
        confidence_threshold=args.confidence_threshold,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
