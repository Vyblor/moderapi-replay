#!/usr/bin/env python3
"""Generate the ~6K stratified evaluation dataset for moderapi-replay.

Data sources:
    - Jigsaw Toxic Comment Classification (CC0 license)
    - Civil Comments (public domain)

Process:
    1. Load texts from Jigsaw + Civil Comments
    2. Call live Perspective API to get continuous scores (0.0-1.0)
    3. Run texts through Detoxify locally
    4. Pair the scores and save as JSONL
    5. Stratify by score band (10 bins) and domain

Usage:
    # Full dataset generation (requires PERSPECTIVE_API_KEY env var):
    python scripts/generate_dataset.py --output data/evaluation_dataset.jsonl

    # Dry run (no API calls, shows plan):
    python scripts/generate_dataset.py --dry-run

    # Resume from checkpoint (if API calls were interrupted):
    python scripts/generate_dataset.py --resume data/checkpoint.jsonl

Environment:
    PERSPECTIVE_API_KEY: Your Google Perspective API key (required)

Rate limits:
    Perspective API allows ~1 QPS on free tier. For ~6K texts:
    - Estimated time: ~2 hours at 1 QPS
    - With quota increase: ~10 minutes at 10 QPS
    Script auto-detects rate limits and backs off.

Reading the signals:
    After generation, the JSONL output contains paired scores. Key things to check:
    - Score distribution: run `python scripts/generate_dataset.py --stats data/evaluation_dataset.jsonl`
    - Coverage: each attribute should have ~1K texts across 10 score bands
    - Domain balance: texts from both Jigsaw and Civil Comments in each band
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# --- Constants ---

PERSPECTIVE_ATTRIBUTES = [
    "TOXICITY",
    "SEVERE_TOXICITY",
    "IDENTITY_ATTACK",
    "INSULT",
    "PROFANITY",
    "THREAT",
]

SCORE_BINS = 10  # 10 bins: [0.0-0.1), [0.1-0.2), ..., [0.9-1.0]
TARGET_PER_BIN_PER_ATTR = 100  # ~100 per bin × 10 bins × 6 attrs ≈ 6K
TARGET_TOTAL = TARGET_PER_BIN_PER_ATTR * SCORE_BINS  # ~1K per attribute

# Rate limiting
DEFAULT_QPS = 1.0
BACKOFF_FACTOR = 2.0
MAX_BACKOFF = 60.0
MAX_RETRIES = 5


@dataclass
class TextSample:
    """A text sample with metadata."""

    text: str
    source: str  # "jigsaw" or "civil_comments"
    source_id: str = ""


@dataclass
class ScoredSample:
    """A text with both Perspective and Detoxify scores."""

    text: str
    source: str
    perspective_scores: dict[str, float] = field(default_factory=dict)
    detoxify_scores: dict[str, float] = field(default_factory=dict)


# --- Data Loading ---


def load_jigsaw_texts(path: Path, max_texts: int = 10000) -> list[TextSample]:
    """Load texts from Jigsaw Toxic Comment Classification dataset.

    Expected format: CSV with 'comment_text' column.
    Download from: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge

    Args:
        path: Path to train.csv (or test.csv).
        max_texts: Maximum texts to load.

    Returns:
        List of TextSample objects.
    """
    import csv

    samples = []
    with open(path, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= max_texts:
                break
            text = row.get("comment_text", "").strip()
            if len(text) < 10 or len(text) > 5000:
                continue
            samples.append(
                TextSample(
                    text=text,
                    source="jigsaw",
                    source_id=row.get("id", str(i)),
                )
            )

    logger.info("Loaded %d texts from Jigsaw dataset", len(samples))
    return samples


def load_civil_comments_texts(path: Path, max_texts: int = 10000) -> list[TextSample]:
    """Load texts from Civil Comments dataset.

    Expected format: CSV with 'comment_text' column.
    Download from: https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification

    Args:
        path: Path to train.csv.
        max_texts: Maximum texts to load.

    Returns:
        List of TextSample objects.
    """
    import csv

    samples = []
    with open(path, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= max_texts:
                break
            text = row.get("comment_text", "").strip()
            if len(text) < 10 or len(text) > 5000:
                continue
            samples.append(
                TextSample(
                    text=text,
                    source="civil_comments",
                    source_id=row.get("id", str(i)),
                )
            )

    logger.info("Loaded %d texts from Civil Comments dataset", len(samples))
    return samples


# --- Perspective API Client ---


def call_perspective_api(
    text: str,
    api_key: str,
    attributes: list[str] | None = None,
    qps: float = DEFAULT_QPS,
) -> dict[str, float]:
    """Call Perspective API for a single text.

    Args:
        text: Input text to score.
        api_key: Perspective API key.
        attributes: List of attributes to request (default: all 6).
        qps: Queries per second limit.

    Returns:
        Dict mapping attribute name to score (0.0-1.0).

    Raises:
        RuntimeError: If API call fails after retries.
    """
    import urllib.request
    import urllib.error

    if attributes is None:
        attributes = PERSPECTIVE_ATTRIBUTES

    url = (
        "https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze"
        f"?key={api_key}"
    )

    request_body = {
        "comment": {"text": text},
        "requestedAttributes": {attr: {} for attr in attributes},
        "languages": ["en"],
    }

    data = json.dumps(request_body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
    )

    backoff = 1.0 / qps
    for attempt in range(MAX_RETRIES):
        try:
            time.sleep(backoff)
            with urllib.request.urlopen(req, timeout=30) as response:
                result = json.loads(response.read().decode("utf-8"))

            scores = {}
            for attr in attributes:
                score_obj = result.get("attributeScores", {}).get(attr, {})
                summary = score_obj.get("summaryScore", {})
                scores[attr] = summary.get("value", 0.0)
            return scores

        except urllib.error.HTTPError as e:
            if e.code == 429:  # Rate limited
                backoff = min(backoff * BACKOFF_FACTOR, MAX_BACKOFF)
                logger.warning(
                    "Rate limited (attempt %d/%d), backing off %.1fs",
                    attempt + 1,
                    MAX_RETRIES,
                    backoff,
                )
                time.sleep(backoff)
            elif e.code >= 500:  # Server error
                backoff = min(backoff * BACKOFF_FACTOR, MAX_BACKOFF)
                logger.warning(
                    "Server error %d (attempt %d/%d), backing off %.1fs",
                    e.code,
                    attempt + 1,
                    MAX_RETRIES,
                    backoff,
                )
                time.sleep(backoff)
            else:
                raise RuntimeError(
                    f"Perspective API error {e.code}: {e.read().decode()}"
                ) from e
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                raise RuntimeError(f"Perspective API failed after {MAX_RETRIES} retries: {e}") from e
            backoff = min(backoff * BACKOFF_FACTOR, MAX_BACKOFF)
            time.sleep(backoff)

    raise RuntimeError("Perspective API failed after all retries")


# --- Detoxify Scoring ---


def score_with_detoxify(texts: list[str], batch_size: int = 32) -> list[dict[str, float]]:
    """Score texts with Detoxify locally.

    Uses the moderapi inference module for consistency with the main tool.

    Returns:
        List of dicts mapping Perspective-compatible attribute names to scores.
    """
    from moderapi.inference import predict_batch

    return predict_batch(texts, batch_size=batch_size)


# --- Stratification ---


def stratify_by_score_band(
    samples: list[ScoredSample],
    attribute: str,
    target_per_bin: int = TARGET_PER_BIN_PER_ATTR,
) -> list[ScoredSample]:
    """Select samples stratified across score bands for a given attribute.

    Ensures representation across the full 0.0-1.0 score range.

    Returns:
        Stratified subset of samples.
    """
    bins: dict[int, list[ScoredSample]] = {i: [] for i in range(SCORE_BINS)}

    for sample in samples:
        score = sample.perspective_scores.get(attribute, 0.0)
        bin_idx = min(int(score * SCORE_BINS), SCORE_BINS - 1)
        bins[bin_idx].append(sample)

    selected = []
    for bin_idx in range(SCORE_BINS):
        bin_samples = bins[bin_idx]
        if len(bin_samples) <= target_per_bin:
            selected.extend(bin_samples)
        else:
            rng = np.random.default_rng(seed=42 + bin_idx)
            indices = rng.choice(len(bin_samples), size=target_per_bin, replace=False)
            selected.extend(bin_samples[i] for i in indices)

    # Also balance by domain (source)
    selected = _balance_domains(selected, target_per_bin * SCORE_BINS)

    logger.info(
        "Stratified %d samples for %s across %d bins",
        len(selected),
        attribute,
        SCORE_BINS,
    )
    return selected


def _balance_domains(
    samples: list[ScoredSample], target_total: int
) -> list[ScoredSample]:
    """Balance samples across data sources (domains)."""
    by_source: dict[str, list[ScoredSample]] = {}
    for s in samples:
        by_source.setdefault(s.source, []).append(s)

    per_source = target_total // max(len(by_source), 1)
    balanced = []
    rng = np.random.default_rng(seed=42)

    for source, source_samples in by_source.items():
        if len(source_samples) <= per_source:
            balanced.extend(source_samples)
        else:
            indices = rng.choice(len(source_samples), size=per_source, replace=False)
            balanced.extend(source_samples[i] for i in indices)

    return balanced


# --- Checkpoint / Resume ---


def save_checkpoint(samples: list[ScoredSample], path: Path) -> None:
    """Save scored samples to JSONL checkpoint."""
    with open(path, "w") as f:
        for s in samples:
            record = {
                "text": s.text,
                "source": s.source,
                "perspective_scores": s.perspective_scores,
                "detoxify_scores": s.detoxify_scores,
            }
            f.write(json.dumps(record) + "\n")
    logger.info("Saved checkpoint with %d samples to %s", len(samples), path)


def load_checkpoint(path: Path) -> list[ScoredSample]:
    """Load scored samples from JSONL checkpoint."""
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            samples.append(
                ScoredSample(
                    text=data["text"],
                    source=data["source"],
                    perspective_scores=data.get("perspective_scores", {}),
                    detoxify_scores=data.get("detoxify_scores", {}),
                )
            )
    logger.info("Loaded checkpoint with %d samples from %s", len(samples), path)
    return samples


# --- Stats ---


def print_dataset_stats(samples: list[ScoredSample]) -> None:
    """Print distribution statistics for the dataset."""
    print(f"\n{'='*60}")
    print(f"Dataset Statistics: {len(samples)} samples")
    print(f"{'='*60}")

    # Source distribution
    sources = {}
    for s in samples:
        sources[s.source] = sources.get(s.source, 0) + 1
    print(f"\nSources: {sources}")

    # Per-attribute score distribution
    for attr in PERSPECTIVE_ATTRIBUTES:
        persp_scores = [s.perspective_scores.get(attr, 0.0) for s in samples if attr in s.perspective_scores]
        if not persp_scores:
            print(f"\n{attr}: no scores")
            continue

        arr = np.array(persp_scores)
        print(f"\n{attr}:")
        print(f"  Perspective: mean={arr.mean():.3f}, std={arr.std():.3f}, "
              f"min={arr.min():.3f}, max={arr.max():.3f}")

        # Bin distribution
        bins = np.histogram(arr, bins=SCORE_BINS, range=(0.0, 1.0))[0]
        print(f"  Bins: {bins.tolist()}")

        detox_scores = [s.detoxify_scores.get(attr, 0.0) for s in samples if attr in s.detoxify_scores]
        if detox_scores:
            darr = np.array(detox_scores)
            print(f"  Detoxify:    mean={darr.mean():.3f}, std={darr.std():.3f}, "
                  f"min={darr.min():.3f}, max={darr.max():.3f}")


# --- Main Pipeline ---


def generate_dataset(
    jigsaw_path: Path | None,
    civil_comments_path: Path | None,
    output_path: Path,
    api_key: str,
    checkpoint_path: Path | None = None,
    resume_path: Path | None = None,
    max_texts_per_source: int = 5000,
    qps: float = DEFAULT_QPS,
    dry_run: bool = False,
) -> None:
    """Full dataset generation pipeline.

    Steps:
        1. Load texts from Jigsaw + Civil Comments
        2. Call Perspective API for continuous scores
        3. Score with Detoxify locally
        4. Stratify by score band + domain
        5. Save as JSONL
    """
    # Step 0: Resume from checkpoint if available
    already_scored: list[ScoredSample] = []
    scored_texts: set[str] = set()
    if resume_path and resume_path.exists():
        already_scored = load_checkpoint(resume_path)
        scored_texts = {s.text for s in already_scored}
        logger.info("Resuming: %d texts already scored", len(already_scored))

    # Step 1: Load texts
    all_texts: list[TextSample] = []
    if jigsaw_path and jigsaw_path.exists():
        all_texts.extend(load_jigsaw_texts(jigsaw_path, max_texts_per_source))
    if civil_comments_path and civil_comments_path.exists():
        all_texts.extend(
            load_civil_comments_texts(civil_comments_path, max_texts_per_source)
        )

    if not all_texts:
        logger.error("No texts loaded. Check your data paths.")
        sys.exit(1)

    # Filter already-scored texts
    new_texts = [t for t in all_texts if t.text not in scored_texts]
    logger.info(
        "Total texts: %d, New (not yet scored): %d", len(all_texts), len(new_texts)
    )

    if dry_run:
        print(f"\n[DRY RUN] Would score {len(new_texts)} texts via Perspective API")
        print(f"  Estimated time at {qps} QPS: {len(new_texts) / qps / 60:.1f} minutes")
        print(f"  Already scored (from checkpoint): {len(already_scored)}")
        print(f"  Sources: {set(t.source for t in new_texts)}")
        return

    # Step 2: Call Perspective API
    checkpoint_interval = 100
    ckpt_path = checkpoint_path or output_path.with_suffix(".checkpoint.jsonl")
    scored = list(already_scored)

    for i, text_sample in enumerate(new_texts):
        try:
            persp_scores = call_perspective_api(text_sample.text, api_key, qps=qps)
            sample = ScoredSample(
                text=text_sample.text,
                source=text_sample.source,
                perspective_scores=persp_scores,
            )
            scored.append(sample)

            if (i + 1) % checkpoint_interval == 0:
                save_checkpoint(scored, ckpt_path)
                logger.info("Progress: %d/%d texts scored", i + 1, len(new_texts))

        except RuntimeError as e:
            logger.error("Failed on text %d: %s", i, e)
            save_checkpoint(scored, ckpt_path)
            logger.info("Checkpoint saved. Resume with --resume %s", ckpt_path)
            sys.exit(1)
        except KeyboardInterrupt:
            logger.info("Interrupted. Saving checkpoint...")
            save_checkpoint(scored, ckpt_path)
            logger.info("Resume with --resume %s", ckpt_path)
            sys.exit(0)

    logger.info("Perspective API scoring complete: %d texts", len(scored))

    # Step 3: Score with Detoxify
    logger.info("Running Detoxify inference...")
    texts_for_detox = [s.text for s in scored]
    detox_results = score_with_detoxify(texts_for_detox)
    for sample, detox_scores in zip(scored, detox_results):
        sample.detoxify_scores = detox_scores

    # Step 4: Stratify — use TOXICITY as primary stratification attribute
    # (ensures broad coverage; other attributes get natural distribution)
    stratified = stratify_by_score_band(scored, "TOXICITY")

    # Step 5: Save final dataset
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(stratified, output_path)

    # Print stats
    print_dataset_stats(stratified)

    # Cleanup checkpoint
    if ckpt_path.exists() and ckpt_path != output_path:
        ckpt_path.unlink()
        logger.info("Cleaned up checkpoint file")

    print(f"\nDataset saved to {output_path}")
    print(f"Total samples: {len(stratified)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate evaluation dataset for moderapi-replay",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/evaluation_dataset.jsonl"),
        help="Output JSONL path (default: data/evaluation_dataset.jsonl)",
    )
    parser.add_argument(
        "--jigsaw",
        type=Path,
        default=None,
        help="Path to Jigsaw Toxic Comment train.csv",
    )
    parser.add_argument(
        "--civil-comments",
        type=Path,
        default=None,
        help="Path to Civil Comments train.csv",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume from checkpoint JSONL",
    )
    parser.add_argument(
        "--qps",
        type=float,
        default=DEFAULT_QPS,
        help=f"Perspective API queries per second (default: {DEFAULT_QPS})",
    )
    parser.add_argument(
        "--max-texts",
        type=int,
        default=5000,
        help="Max texts per source dataset (default: 5000)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show plan without making API calls",
    )
    parser.add_argument(
        "--stats",
        type=Path,
        default=None,
        help="Print stats for an existing dataset JSONL",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    # Stats mode
    if args.stats:
        samples = load_checkpoint(args.stats)
        print_dataset_stats(samples)
        return

    # Check API key
    api_key = os.environ.get("PERSPECTIVE_API_KEY")
    if not api_key and not args.dry_run:
        print("ERROR: Set PERSPECTIVE_API_KEY environment variable")
        print("  Get one at: https://developers.perspectiveapi.com/")
        sys.exit(1)

    # Check data sources
    if not args.jigsaw and not args.civil_comments and not args.resume:
        print("ERROR: Provide at least one data source:")
        print("  --jigsaw path/to/jigsaw/train.csv")
        print("  --civil-comments path/to/civil_comments/train.csv")
        print("  --resume path/to/checkpoint.jsonl")
        print()
        print("Download datasets from:")
        print("  Jigsaw: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge")
        print("  Civil Comments: https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification")
        sys.exit(1)

    generate_dataset(
        jigsaw_path=args.jigsaw,
        civil_comments_path=args.civil_comments,
        output_path=args.output,
        api_key=api_key or "",
        resume_path=args.resume,
        max_texts_per_source=args.max_texts,
        qps=args.qps,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
