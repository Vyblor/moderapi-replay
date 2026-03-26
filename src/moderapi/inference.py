"""Detoxify inference engine.

Phase 1: Native PyTorch inference via Detoxify library.
Phase 2: ONNX Runtime for production server (deferred).

First-run UX: Downloads ~400MB model with Rich progress bar.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from moderapi.exceptions import InferenceError, ModelDownloadError, ModelLoadError
from moderapi.models import Attribute
from moderapi.parser import DETOXIFY_TO_PERSPECTIVE

logger = logging.getLogger(__name__)
console = Console(stderr=True)

# Singleton model instance
_model: Any | None = None


def _ensure_model() -> Any:
    """Load the Detoxify model, downloading on first run with progress UX."""
    global _model
    if _model is not None:
        return _model

    try:
        import torch  # noqa: F401
        from detoxify import Detoxify
    except ImportError as e:
        raise InferenceError(
            "Required packages not installed. Run: pip install moderapi-replay"
        ) from e

    # Check if model is cached
    cache_dir = os.path.expanduser("~/.cache/moderapi/models")
    os.makedirs(cache_dir, exist_ok=True)

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task("Loading Detoxify toxic-bert model...", total=None)
            _model = Detoxify("original", device="cpu")
    except Exception as e:
        if "download" in str(e).lower() or "connection" in str(e).lower():
            raise ModelDownloadError(
                "Failed to download Detoxify model (~400MB). "
                "Check your internet connection and try again."
            ) from e
        raise ModelLoadError(f"Failed to load Detoxify model: {e}") from e

    logger.info("Detoxify model loaded successfully")
    return _model


def predict_batch(texts: list[str], batch_size: int = 32) -> list[dict[str, float]]:
    """Run Detoxify inference on a batch of texts.

    Args:
        texts: List of text strings to analyze.
        batch_size: Number of texts per inference batch.

    Returns:
        List of dicts mapping Perspective attribute names to scores (0.0-1.0).
    """
    model = _ensure_model()
    all_results: list[dict[str, float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        try:
            raw = model.predict(batch)
        except Exception as e:
            raise InferenceError(f"Inference failed on batch {i // batch_size}: {e}") from e

        # raw is dict[str, list[float]] — Detoxify keys to per-text scores
        for j in range(len(batch)):
            scores: dict[str, float] = {}
            for detoxify_key, perspective_attr in DETOXIFY_TO_PERSPECTIVE.items():
                val = raw.get(detoxify_key, [])
                if j < len(val):
                    score = float(val[j])
                    # NaN filter
                    if score != score:  # NaN check
                        logger.warning(
                            "NaN score for text %d, attribute %s", i + j, perspective_attr
                        )
                        continue
                    scores[perspective_attr] = score
            all_results.append(scores)

    return all_results


def predict_single(text: str) -> dict[str, float]:
    """Run Detoxify inference on a single text.

    Returns:
        Dict mapping Perspective attribute names to scores (0.0-1.0).
    """
    results = predict_batch([text], batch_size=1)
    if not results:
        return {}
    return results[0]
