"""Per-attribute calibration: OLS regression + isotonic fallback.

Eng review decisions:
- 70/30 train/test split (calibrate on 70%, gate on 30%)
- Switch criterion: use held-out threshold agreement, NOT R²
- Both OLS and isotonic are tried; keep whichever has better threshold agreement
- Scores clamped to [0, 1] after affine transform
- Pre-calibration semantic viability: raw Spearman < 0.5 = incompatible
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.isotonic import IsotonicRegression

from moderapi.exceptions import (
    CalibrationError,
    CalibrationFileError,
    ConstantDataError,
    InsufficientDataError,
)
from moderapi.models import CalibrationCoefficients, CalibrationConfig

logger = logging.getLogger(__name__)

MIN_SAMPLES = 30  # Minimum samples for meaningful calibration


def _ols_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Ordinary least squares: y = slope * x + intercept.

    Returns:
        (slope, intercept, r_squared)
    """
    if len(x) < MIN_SAMPLES:
        raise InsufficientDataError(f"Need at least {MIN_SAMPLES} samples, got {len(x)}")

    x_mean = np.mean(x)
    y_mean = np.mean(y)
    ss_xx = np.sum((x - x_mean) ** 2)
    ss_xy = np.sum((x - x_mean) * (y - y_mean))
    ss_yy = np.sum((y - y_mean) ** 2)

    if ss_xx == 0:
        raise ConstantDataError("All Detoxify scores are identical — cannot fit OLS")

    slope = float(ss_xy / ss_xx)
    intercept = float(y_mean - slope * x_mean)

    ss_res = np.sum((y - (slope * x + intercept)) ** 2)
    r_squared = float(1.0 - ss_res / ss_yy) if ss_yy > 0 else 0.0

    return slope, intercept, r_squared


def _isotonic_fit(x: np.ndarray, y: np.ndarray) -> IsotonicRegression:
    """Fit isotonic regression model."""
    if len(x) < MIN_SAMPLES:
        raise InsufficientDataError(f"Need at least {MIN_SAMPLES} samples, got {len(x)}")
    ir = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    ir.fit(x, y)
    return ir


def clamp(value: float) -> float:
    """Clamp a score to [0.0, 1.0]."""
    return max(0.0, min(1.0, value))


def apply_ols(score: float, slope: float, intercept: float) -> float:
    """Apply OLS calibration with clamping."""
    return clamp(slope * score + intercept)


def threshold_agreement(
    scores_a: np.ndarray, scores_b: np.ndarray, threshold: float = 0.8
) -> float:
    """Compute threshold agreement: % of texts where both scores agree on above/below T.

    Args:
        scores_a: Reference scores (Perspective).
        scores_b: Predicted scores (calibrated Detoxify).
        threshold: Decision threshold (default 0.8).

    Returns:
        Agreement ratio 0.0-1.0.
    """
    decisions_a = scores_a >= threshold
    decisions_b = scores_b >= threshold
    return float(np.mean(decisions_a == decisions_b))


def calibrate_attribute(
    detoxify_scores: np.ndarray,
    perspective_scores: np.ndarray,
) -> tuple[CalibrationCoefficients, Any | None]:
    """Calibrate a single attribute using train split.

    Tries both OLS and isotonic. Keeps whichever achieves better
    threshold agreement on the provided data.

    Args:
        detoxify_scores: Raw Detoxify scores (train split).
        perspective_scores: Perspective ground truth scores (train split).

    Returns:
        (coefficients, isotonic_model_or_None)
    """
    # Fit OLS
    slope, intercept, r_squared = _ols_fit(detoxify_scores, perspective_scores)
    ols_calibrated = np.array([apply_ols(s, slope, intercept) for s in detoxify_scores])
    ols_agreement = threshold_agreement(perspective_scores, ols_calibrated)

    # Fit isotonic
    try:
        iso_model = _isotonic_fit(detoxify_scores, perspective_scores)
        iso_calibrated = iso_model.predict(detoxify_scores)
        iso_agreement = threshold_agreement(perspective_scores, iso_calibrated)
    except CalibrationError:
        iso_model = None
        iso_agreement = 0.0

    # Pick whichever has better threshold agreement on train data
    if iso_model is not None and iso_agreement > ols_agreement:
        logger.info(
            "Isotonic selected (agreement: %.3f vs OLS %.3f)", iso_agreement, ols_agreement
        )
        return (
            CalibrationCoefficients(slope=0.0, intercept=0.0, r_squared=r_squared, method="isotonic"),
            iso_model,
        )
    else:
        logger.info("OLS selected (agreement: %.3f vs isotonic %.3f)", ols_agreement, iso_agreement)
        return (
            CalibrationCoefficients(slope=slope, intercept=intercept, r_squared=r_squared, method="ols"),
            None,
        )


def save_calibration(config: CalibrationConfig, path: Path) -> None:
    """Write calibration.json to disk."""
    try:
        # Pre-check disk space (rough estimate: calibration.json < 10KB)
        import shutil
        free_bytes = shutil.disk_usage(path.parent).free
        if free_bytes < 1024 * 1024:  # 1MB minimum
            from moderapi.exceptions import DiskSpaceError
            raise DiskSpaceError(f"Less than 1MB free on {path.parent}")

        path.write_text(json.dumps(config.model_dump(), indent=2))
        logger.info("Calibration saved to %s", path)
    except OSError as e:
        raise CalibrationFileError(f"Cannot write {path}: {e}") from e


def load_calibration(path: Path) -> CalibrationConfig:
    """Read calibration.json from disk."""
    try:
        data = json.loads(path.read_text())
        return CalibrationConfig(**data)
    except (OSError, json.JSONDecodeError, Exception) as e:
        raise CalibrationFileError(f"Cannot read {path}: {e}") from e
