"""Streaming JSONL parser for Perspective API replay logs.

Handles large files via line-by-line streaming to avoid memory issues.
Invalid lines are skipped with warnings (not fatal).
"""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from moderapi.exceptions import StreamingParseError, UnicodeParseError
from moderapi.models import Attribute, ReplayRecord

logger = logging.getLogger(__name__)

# Detoxify attribute name mapping to Perspective API attribute names
#
#   Detoxify output key     →  Perspective attribute
#   ─────────────────────────────────────────────────
#   toxicity                →  TOXICITY
#   severe_toxicity         →  SEVERE_TOXICITY
#   identity_attack         →  IDENTITY_ATTACK
#   insult                  →  INSULT
#   obscene                 →  PROFANITY (closest semantic match)
#   threat                  →  THREAT
#
DETOXIFY_TO_PERSPECTIVE: dict[str, str] = {
    "toxicity": Attribute.TOXICITY.value,
    "severe_toxicity": Attribute.SEVERE_TOXICITY.value,
    "identity_attack": Attribute.IDENTITY_ATTACK.value,
    "insult": Attribute.INSULT.value,
    "obscene": Attribute.PROFANITY.value,
    "threat": Attribute.THREAT.value,
}

PERSPECTIVE_TO_DETOXIFY: dict[str, str] = {v: k for k, v in DETOXIFY_TO_PERSPECTIVE.items()}


def parse_jsonl(path: Path, max_line_bytes: int = 10 * 1024 * 1024) -> Iterator[ReplayRecord]:
    """Stream-parse a JSONL file of Perspective API responses.

    Each line should be a JSON object with at minimum:
        {"text": "...", "scores": {"TOXICITY": 0.8, ...}}

    Args:
        path: Path to the JSONL file.
        max_line_bytes: Maximum bytes per line (default 10MB). Lines exceeding
            this are skipped to prevent MemoryError.

    Yields:
        ReplayRecord for each valid line.

    Raises:
        StreamingParseError: If the file cannot be opened.
        UnicodeParseError: If the file has unrecoverable encoding issues.
    """
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line_num, line in enumerate(f, 1):
                if sys.getsizeof(line) > max_line_bytes:
                    logger.warning("Line %d exceeds %d bytes, skipping", line_num, max_line_bytes)
                    continue

                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning("Line %d: invalid JSON: %s", line_num, e)
                    continue

                record = _parse_record(data, line_num)
                if record is not None:
                    yield record

    except MemoryError as e:
        raise StreamingParseError(f"MemoryError reading {path}: {e}") from e
    except UnicodeDecodeError as e:
        raise UnicodeParseError(f"Encoding error in {path}: {e}") from e
    except OSError as e:
        raise StreamingParseError(f"Cannot read {path}: {e}") from e


def _parse_record(data: dict[str, Any], line_num: int) -> ReplayRecord | None:
    """Parse a single JSON object into a ReplayRecord."""
    text = data.get("text")
    if not isinstance(text, str) or not text.strip():
        logger.warning("Line %d: missing or empty 'text' field", line_num)
        return None

    scores = data.get("scores", {})
    if not isinstance(scores, dict):
        logger.warning("Line %d: 'scores' is not a dict", line_num)
        return None

    perspective_scores: dict[str, float] = {}
    for attr in Attribute:
        val = scores.get(attr.value)
        if val is not None:
            try:
                perspective_scores[attr.value] = float(val)
            except (ValueError, TypeError):
                logger.warning("Line %d: invalid score for %s: %r", line_num, attr.value, val)

    if not perspective_scores:
        logger.warning("Line %d: no valid attribute scores found", line_num)
        return None

    return ReplayRecord(
        text=text,
        perspective_scores=perspective_scores,
        detoxify_scores={},
    )
