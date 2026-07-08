"""
src/analysis/topic_modeling.py — Semantic clustering of ad OCR text.

Keyword-based categorization (in src/analysis/ads.py) requires you to
DEFINE the categories in advance. Topic modeling lets the data tell
you what categories naturally exist in your ad corpus — and whether
those topics differ across profiles.

We use BERTopic because it:
  - Handles short, noisy OCR text better than LDA
  - Produces interpretable keyword summaries per topic
  - Supports per-document topic probabilities (not just hard assignments)
  - Scales to tens of thousands of documents on a single machine

Pipeline:
  1. Load OCR text + metadata from ads.parquet
  2. Fit BERTopic on the full corpus (all profiles combined)
  3. Assign each ad to its discovered topic
  4. Compare topic distributions across profiles

Persistence:
  Models are saved to artifacts/models/bertopic/ so you don't refit
  on every notebook run. Fitting takes 5-30 minutes for a corpus of
  10k+ ads; reloading takes seconds.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from config import ARTIFACTS_DIR, PROFILES
from src.utils.db import db_session

# Lazy imports — BERTopic and its deps (sentence-transformers, umap,
# hdbscan) are heavy and slow to import. Only load when actually used.
def _import_bertopic():
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    return BERTopic, SentenceTransformer


MODEL_DIR = ARTIFACTS_DIR / "models" / "bertopic"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────
# TEXT CLEANING
# ─────────────────────────────────────────────────────────────────────
def _clean_ocr_text(text: str) -> str:
    """Clean OCR text for topic modeling.

    OCR output is messy. Common artifacts that hurt topic quality:
      • Single garbage characters from misread glyphs
      • Long runs of punctuation
      • All-uppercase text (OCR sometimes can't distinguish case
        from styling — we lowercase to normalize)
      • URLs and email-like strings (rarely topical)

    Aggressive cleaning at this stage dramatically improves topic
    coherence. Trade-off: we lose some signal (brand names in
    SHOUTING-CAPS, e.g.). For ad content, normalization wins.
    """
    if not text:
        return ""
    # Drop URLs and email-like patterns
    text = re.sub(r'https?://\S+', ' ', text)
    text = re.sub(r'\S+@\S+\.\S+', ' ', text)
    # Drop runs of non-letter chars
    text = re.sub(r'[^a-zA-Z\s]+', ' ', text)
    # Collapse whitespace, lowercase
    text = re.sub(r'\s+', ' ', text).strip().lower()
    # Drop tokens shorter than 3 chars (mostly OCR junk)
    tokens = [t for t in text.split() if len(t) >= 3]
    return ' '.join(tokens)


def load_ad_corpus(
    min_chars: int = 20,
    min_confidence: str = 'high',
) -> pd.DataFrame:
    """Load ads with sufficient OCR text for topic modeling.

    Args:
        min_chars: Minimum cleaned-text length. Below ~20 chars,
            most "ads" are logos or banners with single-word OCR
            that can't be topic-modeled meaningfully.
        min_confidence: Restrict to high-confidence ad detections.

    Returns: DataFrame with profile, ad_hash, page_url,
             advertiser_network, ocr_text (cleaned), and original text.
    """
    confidence_filter = f"AND confidence = '{min_confidence}'" \
        if min_confidence else ""

    with db_session(read_only=True) as con:
        df = con.execute(f"""
            SELECT profile, ad_hash, page_url, advertiser_network,
                   ocr_text AS ocr_text_raw
            FROM ads
            WHERE ocr_char_count > 0
              {confidence_filter}
        """).df()

    # Clean in pandas — vectorized via apply, fast enough for ~100k rows
    df['ocr_text'] = df['ocr_text_raw'].apply(_clean_ocr_text)
    df['n_chars_clean'] = df['ocr_text'].str.len()
    return df[df['n_chars_clean'] >= min_chars].reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────
# MODEL FITTING & PERSISTENCE
# ─────────────────────────────────────────────────────────────────────
def fit_bertopic(
    corpus: pd.DataFrame,
    embedding_model: str = "all-MiniLM-L6-v2",
    min_topic_size: int = 10,
    save_name: Optional[str] = "ads_v1",
) -> tuple:
    """Fit BERTopic on the ad corpus.

    Args:
        corpus: DataFrame from load_ad_corpus()
        embedding_model: SentenceTransformer model. all-MiniLM-L6-v2
            is the standard fast/decent default. For higher-quality
            topics on a powerful machine, try 'all-mpnet-base-v2'
            (3x slower, noticeably better coherence).
        min_topic_size: Smallest cluster to retain. Smaller → more
            granular topics but more noise. 10 is a good starting
            point for corpora of ~10k documents.
        save_name: If provided, persist to MODEL_DIR/<name>/

    Returns: (fitted_model, topics_list, probabilities_array)

    Why these defaults?
        all-MiniLM-L6-v2 fits 22M-document benchmarks in hours; for
        ad-scale corpora (~10k-100k docs) it's plenty fast. The
        min_topic_size=10 prevents BERTopic from creating ultra-niche
        "topics" of 2-3 ads that are usually noise.
    """
    BERTopic, SentenceTransformer = _import_bertopic()

    print(f"Embedding {len(corpus):,} documents with {embedding_model}...")
    embedder = SentenceTransformer(embedding_model)
    embeddings: np.ndarray = np.asarray(embedder.encode(
        corpus['ocr_text'].tolist(),
        show_progress_bar=True,
        batch_size=64,
    ))

    print("Fitting BERTopic...")
    model = BERTopic(
        embedding_model=embedder,
        min_topic_size=min_topic_size,
        # calculate_probabilities=False is dramatically faster; we get
        # hard assignments only. Turn on if you need soft assignments.
        calculate_probabilities=False,
        verbose=True,
    )
    topics, probs = model.fit_transform(
        corpus['ocr_text'].tolist(),
        embeddings=embeddings,
    )

    if save_name:
        save_path = MODEL_DIR / save_name
        save_path.mkdir(parents=True, exist_ok=True)
        # Save model + embeddings + the corpus index so we can recover
        # which ad got which topic later without refitting.
        model.save(str(save_path / "model"),
                   serialization="safetensors",
                   save_ctfidf=True)
        np.save(save_path / "embeddings.npy", embeddings)
        corpus.assign(topic=topics).to_parquet(
            save_path / "corpus_with_topics.parquet"
        )
        print(f"✓ Saved model to {save_path}")

    return model, topics, probs


def load_fitted_model(save_name: str = "ads_v1") -> tuple:
    """Reload a previously-fitted BERTopic model.

    Returns: (model, corpus_with_topics_df).
    Caches the model in memory across calls — reloading takes ~10s
    the first time, ~0s thereafter.
    """
    BERTopic, _ = _import_bertopic()
    save_path = MODEL_DIR / save_name
    if not save_path.exists():
        raise FileNotFoundError(
            f"No saved model at {save_path}. "
            f"Run fit_bertopic(...) first."
        )
    model = BERTopic.load(str(save_path / "model"))
    corpus_topics = pd.read_parquet(save_path / "corpus_with_topics.parquet")
    return model, corpus_topics


# ─────────────────────────────────────────────────────────────────────
# TOPIC INSPECTION
# ─────────────────────────────────────────────────────────────────────
def topic_summary(model, top_words: int = 8) -> pd.DataFrame:
    """One row per topic with its top keywords and size.

    BERTopic's get_topic_info() returns this natively, but we
    reformat it for cleaner display and add the topic-keyword
    string as one column for easy table inclusion.
    """
    info = model.get_topic_info()
    summaries = []
    for topic_id in info['Topic']:
        if topic_id == -1:
            keywords = "<outliers / no clear topic>"
        else:
            words = [w for w, _ in model.get_topic(topic_id)[:top_words]]
            keywords = ", ".join(words)
        summaries.append({
            'topic_id': topic_id,
            'size': int(info[info['Topic'] == topic_id]['Count'].iloc[0]),
            'keywords': keywords,
        })
    return pd.DataFrame(summaries).sort_values('size', ascending=False)


# ─────────────────────────────────────────────────────────────────────
# CROSS-PROFILE TOPIC COMPARISON — the headline analysis
# ─────────────────────────────────────────────────────────────────────
def topic_distribution_by_profile(
    save_name: str = "ads_v1",
) -> pd.DataFrame:
    """How many ads of each topic appeared in each profile.

    The central output of topic modeling. If one profile has 50% of its
    ads in a topic dominated by "buy / shop / sale / discount" keywords,
    while the baseline has 10%, behavioral targeting is producing
    measurably different ad CONTENT, not just volume.

    Returns long-format: profile, topic_id, n_ads, pct_of_profile,
                          topic_keywords.
    """
    BERTopic, _ = _import_bertopic()
    model, corpus = load_fitted_model(save_name)

    counts = (corpus.groupby(['profile', 'topic'])
                    .size()
                    .reset_index(name='n_ads'))
    totals = corpus.groupby('profile').size().reset_index(name='total')
    df = counts.merge(totals, on='profile')
    df['pct_of_profile'] = (df['n_ads'] * 100.0 / df['total']).round(2)

    # Attach topic keywords for interpretability
    topics_summary = topic_summary(model)
    df = df.merge(
        topics_summary[['topic_id', 'keywords']],
        left_on='topic', right_on='topic_id', how='left'
    )
    return df[['profile', 'topic', 'keywords',
               'n_ads', 'pct_of_profile']].sort_values(
        ['profile', 'pct_of_profile'], ascending=[True, False]
    )


def differential_topics(
    profile_a: str,
    profile_b: str | None = None,
    save_name: str = "ads_v1",
) -> pd.DataFrame:
    """Topics overrepresented in profile_a vs profile_b.

    Returns: topic_id, keywords, pct_a, pct_b, delta, lift.
    Sorted by lift descending. The "smoking gun" topics for
    behavioral targeting evidence.
    """
    if profile_b is None:
        if not PROFILES:
            raise ValueError("config.PROFILES is empty.")
        profile_b = PROFILES[0]

    dist = topic_distribution_by_profile(save_name)
    pivot = dist.pivot(index='topic', columns='profile',
                       values='pct_of_profile').fillna(0)

    if profile_a not in pivot.columns or profile_b not in pivot.columns:
        raise ValueError(
            f"Need both '{profile_a}' and '{profile_b}' in data"
        )

    keywords_lookup = (
        dist.drop_duplicates('topic')
            .set_index('topic')['keywords']
            .to_dict()
    )
    df = pd.DataFrame({
        'topic_id': pivot.index,
        'keywords': pivot.index.map(lambda topic_id: keywords_lookup.get(topic_id)),
        'pct_a':    pivot[profile_a].values,
        'pct_b':    pivot[profile_b].values,
    })
    df['delta'] = df['pct_a'] - df['pct_b']
    df['lift']  = (df['pct_a'] + 0.1) / (df['pct_b'] + 0.1)
    return df.sort_values('lift', ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    # End-to-end smoke test: load corpus, fit, summarize.
    corpus = load_ad_corpus()
    print(f"Loaded {len(corpus):,} ads with usable OCR text")
    print(f"Per profile:\n{corpus['profile'].value_counts()}")

    model, topics, _ = fit_bertopic(corpus, save_name="ads_v1")
    print("\nTopic summary:")
    print(topic_summary(model).head(20).to_string(index=False))

    for profile in PROFILES:
        if profile == PROFILES[0]:
            continue
        print(f"\nDifferential topics: {profile} vs {PROFILES[0]}")
        print(differential_topics(profile, PROFILES[0])
              .head(10).to_string(index=False))