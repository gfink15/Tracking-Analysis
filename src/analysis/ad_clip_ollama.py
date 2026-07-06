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

from wakepy import keep
from json_repair import repair_json

import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
import numpy as np
import json

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

import sys
from pathlib import Path

# Calculate the parent directory path
parent_dir = str(Path(__file__).resolve().parent.parent.parent)

# Insert it into sys.path so Python can see it
sys.path.insert(0, parent_dir)


import ollama

from datetime import timedelta
import time
from pydantic import BaseModel, Field, field_validator

# Project-local imports — matches your existing pattern in src/analysis/*
import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("ad_desc")


# ─────────────────────────────────────────────────────────────────────
# PATHS (derived from config.py — adjust names if yours differ)
# ─────────────────────────────────────────────────────────────────────
PARQUET_DIR = Path(getattr(config, "PARQUET_DIR", "artifacts/parquet"))
ADS_PARQUET = PARQUET_DIR / "ads.parquet"
ADS_CLIP_PARQUET = PARQUET_DIR / "ads_desc.parquet"
BATCH_SIZE = 25


# Root containing per-profile ad PNGs. Adjust if config.py names it differently.
IMAGE_ROOT = Path(getattr(config, "DATA_DIR", "data"))
CATEGORIES = [
    "Automotive",
    "Beauty & Personal Care",
    "Business Services",
    "Construction & Home Improvement",
    "Consumer Electronics",
    "Education",
    "Energy & Utilities",
    "Entertainment",
    "Fashion & Apparel",
    "Finance",
    "Food & Beverage",
    "Gaming",
    "Government & Public Services",
    "Health & Wellness",
    "Healthcare",
    "Home & Garden",
    "Industrial & Manufacturing",
    "Insurance",
    "Jewelry & Luxury Goods",
    "Legal Services",
    "Marketplace & Classifieds",
    "Media & Publishing",
    "Nonprofit & Charity",
    "Pets",
    "Real Estate",
    "Recruitment & Careers",
    "Restaurants & Dining",
    "Retail",
    "Software & SaaS",
    "Sports & Fitness",
    "Technology",
    "Telecommunications",
    "Travel & Hospitality",
    "Transportation & Logistics",
    "Consumer Packaged Goods",
    "Cryptocurrency & Web3",
    "Dating",
    "Events & Conferences",
    "Parenting & Family",
    "Photography & Creative Services",
    "Religion & Faith",
    "Security & Privacy",
    "Smart Home & IoT",
    "Streaming Services",
    "Subscription Services",
    "Toys & Hobbies",
    "Adult",
    "Political",
    "Public Safety",
    "Likely Scam",
    "Other"
]

VLM_PROMPT = f"""You are analyzing an image that may or may not contain an 
advertisement, captured during a web privacy study.

FIRST, evaluate the image:
- If the image is mostly blank, contains only a fragment of an ad, 
  is unreadable, is only a play button, or does not show a coherent advertisement, respond with 
  this JSON and nothing else:
    {{
        "is_valid_ad": false, 
        "category": "None",
        "product": "None",
        "brand": "None",
        "description": "None",
        "content": "None",
        "confidence": "None",
        "reason": "<brief explanation>"
    }}

- If the image DOES contain a coherent, readable advertisement, respond with:
    {{
        "is_valid_ad": true,
        "category": "<Specify a category from list: {CATEGORIES}>",
        "product": "<what is being sold>",
        "brand": "<brand name if visible, otherwise 'unknown'>",
        "description": "<one sentence describing the imagery>",
        "content": "<any headline/CTA text visible>",
        "confidence": "<'high', 'medium', or 'low' — how sure are you>"
        "reason": "None"
    }}

- CRITICAL: If the image has a large button with a label such as 'Continue' or 'Download' it is likely a scam.
  Please keep these ads marked as valid, but CATEGORIZE THEM AS SCAM.


Return ONLY the JSON. Do not include any other text or reasoning.
You MUST populate all fields, including 'is_valid_ad', 'confidence', and 'reason'. Do not leave fields out.
"""

# VLM_PROMPT = f"""You are analyzing an image that may or may not contain an 
# advertisement, captured during a web privacy study.

# FIRST, evaluate the image:
# - If the image is mostly blank, contains only a fragment of an ad, 
#   is unreadable, is only a play button, or does not show a coherent advertisement, it is
#   not a valid ad. Please respond with a reason.

# - If the image DOES contain a coherent, readable advertisement, respond with
#   is_valid_ad: true, the category from this list {CATEGORIES}, the primary product
#   or service, the brand being advertised, a visual description of the ad, the actual text content visible
#   in the ad, and your confidence level.

# - If the image has a large button with a label such as 'Continue' or 'Download' it is likely a scam.
#   Please keep these ads marked as valid, but categorize them as a potential scam.

# - Return all output as JSON with the following format:
# {{
#    "is_valid_ad": bool,
#    "category": str,
#    "product": str,
#    "brand": "<brand name if visible, otherwise 'unknown'>",
#    "description": "<one sentence describing the imagery>",
#    "content": "<any headline/CTA text visible>",
#    "confidence": "<'high', 'medium', or 'low' — how sure are you>"
# }}
# """


# ─────────────────────────────────────────────────────────────────────
# CLIP CLASSIFIER
# ─────────────────────────────────────────────────────────────────────
#@dataclass
class VisionResult(BaseModel):
    is_valid_ad: bool = Field(
        default=False,
        description="Ad is valid"
    )
    category: str = Field(
        default="None",
        description="Category ad fits into"
    )
    product: str = Field(
        default="None",
        description="Primary product or service in ad"
    )
    brand: str = Field(
        default="None",
        description="Brand being advertised"
    )
    description: str = Field(
        default="None",
        description="Description of ad"
    )
    content: str = Field(
        default="None",
        description="Text visible in ad"
    )
    confidence: str = Field(
        default="None",
        description="Confidence level of accuracy"
    )
    reason: str = Field(
        default="None",
        description="Reasoning for marking ad as invalid"
    )


class VisionDescriber:
    def __init__(
        self,
        #model_name: str = "hf.co/vinimuchulski/gemma-3-12b-it-qat-q4_0-gguf:latest",
        #model_name: str = "llama3.2-vision:11b",
        model_name: str = "hf.co/vinimuchulski/gemma-3-12b-it-qat-q4_0-gguf:latest",
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        log.info("Loading Vision Model %s on %s", model_name, self.device)

    #@field_validator("category", "content", "product", "brand", "description", "confidence", "reason", mode="before")
    def describe(self, image_path: Path) -> tuple[int, VisionResult]:
        try:
            img = Image.open(image_path).convert("RGB")
        except (UnidentifiedImageError, FileNotFoundError, OSError) as e:
            log.debug("Could not open %s: %s", image_path.name, e)
            return 0, VisionResult(
                is_valid_ad=False,
                category="None",
                product="None",
                brand="None",
                description="None",
                content="None",
                confidence="None",
                reason="Ad Missing"
            )

        if img.size[0] < 20 or img.size[1] < 20:
            return 0, VisionResult(
                is_valid_ad=False,
                category="None",
                product="None",
                brand="None",
                description="None",
                content="None",
                confidence="None",
                reason="Image too small"
            )

        response = ollama.chat(
            model="hf.co/vinimuchulski/gemma-3-12b-it-qat-q4_0-gguf:latest", #qwen3.5:4b, llama3.2-vision:11b, hf.co/vinimuchulski/gemma-3-12b-it-qat-q4_0-gguf:latest
            messages=[{
                'role': 'user',
                'content': VLM_PROMPT,
                'images': [str(image_path)]
            }],
            think=False,
            format=VisionResult.model_json_schema(),
            options={
                "temperature": 0.0,
                "num_ctx": 2048,
                "num_predict": 2048,
                "top_k": 20,
                "top_p": 0.9
            }
        )
        return self.validate_response(response)
        
        
    def validate_response(self, response: ollama.ChatResponse) -> tuple[int, VisionResult]:
        if response.message.content and response.total_duration:
            try:
                parsed_response = VisionResult.model_validate_json(response.message.content, by_name=True)
                #parsed_response = json.loads(str(response.message.content))
                # if not parsed_response.is_valid_ad:
                #     return response['total_duration'] / 1000, VisionResult(
                #         is_valid_ad=False,
                #         category="None",
                #         product="None",
                #         brand="None",
                #         description="None",
                #         content="None",
                #         confidence="None",
                #         reason=parsed_response.reason
                #     )

                # return response['total_duration'] / 1000, VisionResult(
                #     is_valid_ad=parsed_response['is_valid_ad'],
                #     category=parsed_response['category'],
                #     product=parsed_response['product'],
                #     brand=parsed_response['brand'],
                #     description=parsed_response['description'],
                #     content=parsed_response['content'],
                #     confidence=parsed_response['confidence'],
                #     reason=parsed_response['reason']
                # )
                return int(response.total_duration / 1000), parsed_response
            except Exception as e:
                log.error(f"Strict parse failed, attempting repair: {e}")
                # log.error(f"Output: {str(response)}")
                # return 0, VisionResult(
                #         is_valid_ad=False,
                #         category="None",
                #         product="None",
                #         brand="None",
                #         description="None",
                #         content="None",
                #         confidence="None",
                #         reason="JSON output unreadable"
                #     )
            try:
                repaired = repair_json(response.message.content, schema=VisionResult.model_json_schema(), return_objects=True)
                if isinstance(repaired, list) and repaired:
                    repaired = repaired[0] if isinstance(repaired[0], dict) else {}
                if isinstance(repaired, dict):
                    return int(response.total_duration / 1000), VisionResult.model_validate(repaired)
            except Exception as e:
                log.error(f"Repair failed: {e}")
                log.error(f"Output: {str(response)}")
                return 0, VisionResult(
                    is_valid_ad=False,
                    category="None",
                    product="None",
                    brand="None",
                    description="None",
                    content="None",
                    confidence="None",
                    reason="JSON output unreadable"
                )
        elif response.total_duration:
            log.error("Empty response message content")
            return int(response.total_duration / 1000), VisionResult(
                is_valid_ad=False,
                category="None",
                product="None",
                brand="None",
                description="None",
                content="None",
                confidence="None",
                reason="Empty response message content"
            )
        else:
            log.error("Empty response")
            return 0, VisionResult(
                is_valid_ad=False,
                category="None",
                product="None",
                brand="None",
                description="None",
                content="None",
                confidence="None",
                reason="Empty response"
            )
        
        log.error(f"JSON Unreadable: {response}")
        return 0, VisionResult(
            is_valid_ad=False,
            category="None",
            product="None",
            brand="None",
            description="None",
            content="None",
            confidence="None",
            reason="JSON Unreadable"
        )


# ─────────────────────────────────────────────────────────────────────
# IMAGE PATH RESOLUTION
# ─────────────────────────────────────────────────────────────────────
def resolve_image_path(row: pd.Series, image_root: Path) -> Optional[Path]:
    if pd.isna(row.get("png_path")) or not row["png_path"]:
        return None
    candidate = image_root / row["png_path"]
    return candidate if candidate.exists() else None

# Save dataframe to parquet file every so often so fatal error doesn't erase work
def flush_batch(batch: list[dict], output_path: Path) -> None:
    """Append the current batch to the output Parquet."""
    if not batch:
        return
    batch_df = pd.DataFrame(batch)
    if output_path.exists():
        # Read existing, concat, write back (safe pattern for small-to-medium data)
        existing = pd.read_parquet(output_path)
        combined = pd.concat([existing, batch_df], ignore_index=True)
    else:
        combined = batch_df
    # Write atomically via temp file to avoid corruption on crash mid-write
    tmp_path = output_path.with_suffix('.parquet.tmp')
    combined.to_parquet(tmp_path, compression='zstd', index=False)
    tmp_path.replace(output_path)  # atomic rename on POSIX and Windows

# ─────────────────────────────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────────────────────────────
def run_clip_pipeline(
    limit: Optional[int] = None,
    single_ad: Optional[list[str]] = None,
    input_parquet: Path = ADS_PARQUET,
    output_parquet: Path = ADS_CLIP_PARQUET,
    image_root: Path = IMAGE_ROOT,
    resume: bool = True,
) -> pd.DataFrame:
    """
    Main entry point. Loads ads.parquet, classifies each image, writes
    ads_clip.parquet, returns the classification DataFrame.
    """

    start_time = time.perf_counter()
    
    processed_indices: set[int] = set()
    if resume and output_parquet.exists():
        existing = pd.read_parquet(output_parquet, columns=['index'])
        processed_indices = set(existing['index'].tolist())
        print(f"↻ Resuming: {len(processed_indices):,} ads already processed, skipping those.")

    log.info("Loading %s", input_parquet)
    ads = pd.read_parquet(input_parquet)
    to_process = ads[~ads['index'].isin(processed_indices)]
    log.info("Loaded %d ad records", len(to_process))

    if (len(to_process) == 0):
        print("All ads already processed. Nothing to do.")
        return to_process

    if "ad_hash" not in to_process.columns:
        raise KeyError(
            "ads.parquet must contain an 'ad_hash' column for join-back. "
            "Update load_ad_artifacts.py to emit one if missing."
        )

    if not single_ad == None:
        to_process = to_process.loc[to_process['ad_hash'].isin(single_ad)]
        log.info("Running in sample list mode")

    if limit:
        to_process = to_process.head(limit)
        log.info("SMOKE TEST MODE: classifying first %d ads only", limit)

    describer = VisionDescriber()

    #rows = []
    batch_results: list[dict] = []
    missing = 0
    skipped = 0
    succeeded = 0
    time_total = 0 # in ms
    with logging_redirect_tqdm():
        for i, (_, row) in enumerate(tqdm(to_process.iterrows(), desc="Processing Ads"), start=1):
            img_path = resolve_image_path(row, image_root)
            if img_path is None:
                missing += 1
                batch_results.append({
                    "index": row["index"],
                    "profile": row["profile"],
                    "visit_id": row["visit_id"],
                    "ad_hash": row["ad_hash"],
                    "is_valid_ad": False,
                    "category": "None",
                    "brand": "None",
                    "product": "None",
                    "description": "None",
                    "content": "None",
                    "confidence": "None",
                    "status": "image_not_found",
                    "reason": "None",
                })
                if len(batch_results) >= BATCH_SIZE:
                    flush_batch(batch_results, output_parquet)
                    batch_results = []
                continue
            skip, reason = is_low_content_image(img_path)
            if skip:
                skipped += 1
                batch_results.append({
                    "index": row["index"],
                    "profile": row["profile"],
                    "visit_id": row["visit_id"],
                    "ad_hash": row["ad_hash"],
                    "is_valid_ad": False,
                    "category": "None",
                    "brand": "None",
                    "product": "None",
                    "description": "None",
                    "content": "None",
                    "confidence": "None",
                    "status": "image_skipped",
                    "reason": reason,
                })
                if len(batch_results) >= BATCH_SIZE:
                    flush_batch(batch_results, output_parquet)
                    batch_results = []
                continue

            r = describer.describe(img_path)
            if r is None:
                batch_results.append({
                    "index": row["index"],
                    "profile": row["profile"],
                    "visit_id": row["visit_id"],
                    "ad_hash": row["ad_hash"],
                    "is_valid_ad": False,
                    "category": "None",
                    "brand": "None",
                    "product": "None",
                    "description": "None",
                    "content": "None",
                    "confidence": "None",
                    "status": "vlm_error_empty",
                    "reason": "None",
                })
                if len(batch_results) >= BATCH_SIZE:
                    flush_batch(batch_results, output_parquet)
                    batch_results = []
                continue

            batch_results.append({
                "index": row["index"],
                "profile": row["profile"],
                "visit_id": row["visit_id"],
                "ad_hash": row["ad_hash"],
                "is_valid_ad": r[1].is_valid_ad,
                "category": r[1].category,
                "brand": r[1].brand,
                "product": r[1].product,
                "description": r[1].description,
                "content": r[1].content,
                "confidence": r[1].confidence,
                "status": "vlm_success",
                "reason": r[1].reason,
            })
            if len(batch_results) >= BATCH_SIZE:
                    flush_batch(batch_results, output_parquet)
                    batch_results = []
            succeeded += 1
            time_total += (r[0] / 1000000)
            if i % 10 == 0:
                log.info("Described %d/%d", i, len(to_process))

    if missing:
        log.warning("Could not resolve %d image paths — check IMAGE_ROOT in config.py",
                    missing)

    log.info(f"Analysis complete. Writing {len(batch_results)} remaining rows to parquet table...")
    flush_batch(batch_results, output_parquet)
    # desc_df = pd.DataFrame(rows)
    # output_parquet.parent.mkdir(parents=True, exist_ok=True)
    # desc_df.to_parquet(output_parquet, index=False)
    # log.info("Wrote %d rows → %s", len(desc_df), output_parquet)
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    hours, remainder = divmod(elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    log.info(f"Elapsed time: {int(hours):02}:{int(minutes):02}:{int(seconds):05.2f}")
    if succeeded > 0:
        log.info(f"Average prompt time: {str(timedelta(seconds=int(time_total/succeeded)))}")

    # _print_summary(clip_df, ads)
    return to_process #desc_df


def is_low_content_image(png_path: Path,
                         min_dim: int = 40,
                         blank_std_threshold: float = 15.0,
                         edge_content_ratio: float = 0.05) -> tuple[bool, str]:
    """
    Detects blank/broken images
    Returns tuple with t/f and reason
    """
    try:
        with Image.open(png_path) as img:
            # Guard 1: matches your existing load_ad_artifacts.py filter
            if img.width < min_dim or img.height < min_dim:
                return True, f"too_small_{img.width}x{img.height}"
            
            gray = np.array(img.convert('L'))
            
            # Guard 2: nearly uniform color (blank/solid images)
            if gray.std() < blank_std_threshold:
                return True, f"low_variance_std={gray.std():.1f}"
            
            # Guard 3: most content concentrated in <5% of the image
            # (catches "corner fragment" case you described)
            row_content = (gray.std(axis=1) > 10).sum() / gray.shape[0]
            col_content = (gray.std(axis=0) > 10).sum() / gray.shape[1]
            content_area = row_content * col_content
            if content_area < edge_content_ratio:
                return True, f"fragmented_content_area={content_area:.2%}"
            
        return False, "ok"
    except Exception as e:
        return True, f"unreadable_{type(e).__name__}"


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Llama Describe captured ads")
    # parser.add_argument("--persona-labels", default="shopper",
    #                     choices=list(LABEL_SETS.keys()))
    # parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--s", type=list, default=None, help="Sample list mode; enter list of ad hashes here")
    parser.add_argument("--limit", type=int, default=None,
                        help="Classify only first N ads (smoke test)")
    args = parser.parse_args()

    run_clip_pipeline(
        limit=args.limit,
        single_ad=args.s
    )


if __name__ == "__main__":
    with keep.running():
        main()
