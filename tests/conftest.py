"""Shared test fixtures for moderapi-replay."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from moderapi.models import Attribute

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SMALL_FIXTURE = FIXTURES_DIR / "small_sample.jsonl"


@pytest.fixture
def small_fixture_path() -> Path:
    """Path to the committed small fixture (100 texts)."""
    return SMALL_FIXTURE


@pytest.fixture
def sample_scores() -> dict[str, list[float]]:
    """Sample paired scores for testing calibration and comparison."""
    import numpy as np

    rng = np.random.default_rng(42)
    n = 100

    # Simulated Perspective scores: uniform 0-1
    perspective = rng.uniform(0, 1, n)
    # Simulated Detoxify scores: correlated but noisy
    detoxify = np.clip(perspective * 0.9 + rng.normal(0, 0.1, n), 0, 1)

    return {
        "perspective": perspective.tolist(),
        "detoxify": detoxify.tolist(),
    }


@pytest.fixture
def sample_jsonl(tmp_path: Path) -> Path:
    """Create a temporary JSONL file with sample data."""
    p = tmp_path / "sample.jsonl"
    lines = []
    for i in range(10):
        record = {
            "text": f"Sample text number {i}",
            "scores": {
                attr.value: round(0.1 * i + 0.05, 2)
                for attr in Attribute
            },
        }
        lines.append(json.dumps(record))
    p.write_text("\n".join(lines))
    return p
