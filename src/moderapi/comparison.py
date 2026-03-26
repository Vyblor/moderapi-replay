"""Score comparison: Spearman correlation, threshold agreement, bootstrap CI.

Eng review decisions:
- Pre-calibration semantic viability check (raw Spearman < 0.5 = incompatible)
- Per-attribute gate (not all-or-nothing)
- Bootstrap 95% CI on all gate metrics
- Ablation: report raw, OLS, and isotonic scores side-by-side
"""

from __future__ import annotations

import logging
import math

import numpy as np
from scipy import stats

from moderapi.calibration import threshold_agreement
from moderapi.exceptions import ConstantDataError, InsufficientDataError
from moderapi.models import AttributeGateResult, GateResult

logger = logging.getLogger(__name__)

SPEARMAN_GATE = 0.85
THRESHOLD_AGREEMENT_GATE = 0.90
SEMANTIC_VIABILITY_THRESHOLD = 0.50
DEFAULT_THRESHOLD = 0.8
BOOTSTRAP_ITERATIONS = 1000
BOOTSTRAP_CI = 0.95


def _safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Spearman correlation with pre-checks for constant data."""
    if len(a) < 3:
        raise InsufficientDataError(f"Need at least 3 samples, got {len(a)}")
    if np.std(a) == 0:
        raise ConstantDataError("All values in first array are constant")
    if np.std(b) == 0:
        raise ConstantDataError("All values in second array are constant")

    # Filter out NaN pairs
    mask = ~(np.isnan(a) | np.isnan(b))
    a_clean = a[mask]
    b_clean = b[mask]

    if len(a_clean) < 3:
        raise InsufficientDataError("Fewer than 3 non-NaN pairs")

    corr, _ = stats.spearmanr(a_clean, b_clean)
    if math.isnan(corr):
        raise ConstantDataError("Spearman returned NaN (likely constant data)")
    return float(corr)


def _bootstrap_ci(
    a: np.ndarray,
    b: np.ndarray,
    metric_fn: callable,
    n_iterations: int = BOOTSTRAP_ITERATIONS,
    ci: float = BOOTSTRAP_CI,
) -> tuple[float, float]:
    """Compute bootstrap confidence interval for a metric.

    Args:
        a: First array.
        b: Second array.
        metric_fn: Function(a, b) -> float.
        n_iterations: Number of bootstrap samples.
        ci: Confidence level (0.95 = 95% CI).

    Returns:
        (lower_bound, upper_bound)
    """
    rng = np.random.default_rng(42)  # Reproducible
    n = len(a)
    estimates = []

    for _ in range(n_iterations):
        idx = rng.integers(0, n, size=n)
        try:
            val = metric_fn(a[idx], b[idx])
            if not math.isnan(val):
                estimates.append(val)
        except (ConstantDataError, InsufficientDataError):
            continue

    if len(estimates) < n_iterations * 0.5:
        logger.warning("Bootstrap: only %d/%d iterations succeeded", len(estimates), n_iterations)
        return 0.0, 0.0

    alpha = (1 - ci) / 2
    lower = float(np.quantile(estimates, alpha))
    upper = float(np.quantile(estimates, 1 - alpha))
    return lower, upper


def evaluate_attribute(
    attribute: str,
    perspective_scores: np.ndarray,
    raw_scores: np.ndarray,
    calibrated_scores: np.ndarray,
) -> AttributeGateResult:
    """Evaluate a single attribute against the hard gate.

    Steps:
    1. Check semantic viability (raw Spearman >= 0.5)
    2. Compute calibrated Spearman + bootstrap CI
    3. Compute threshold agreement + bootstrap CI
    4. Gate: Spearman >= 0.85 AND agreement >= 90%
    """
    # Step 1: Semantic viability
    try:
        spearman_raw = _safe_spearman(perspective_scores, raw_scores)
    except (ConstantDataError, InsufficientDataError) as e:
        logger.warning("Attribute %s: raw Spearman failed: %s", attribute, e)
        return AttributeGateResult(
            attribute=attribute,
            viable=False,
            spearman_raw=0.0,
            spearman_calibrated=0.0,
            threshold_agreement=0.0,
            gate_passed=False,
        )

    viable = spearman_raw >= SEMANTIC_VIABILITY_THRESHOLD
    if not viable:
        logger.info(
            "Attribute %s: semantically incompatible (raw Spearman %.3f < %.1f)",
            attribute,
            spearman_raw,
            SEMANTIC_VIABILITY_THRESHOLD,
        )
        return AttributeGateResult(
            attribute=attribute,
            viable=False,
            spearman_raw=spearman_raw,
            spearman_calibrated=0.0,
            threshold_agreement=0.0,
            gate_passed=False,
        )

    # Step 2: Calibrated Spearman + CI
    try:
        spearman_cal = _safe_spearman(perspective_scores, calibrated_scores)
    except (ConstantDataError, InsufficientDataError):
        spearman_cal = 0.0

    sp_lower, sp_upper = _bootstrap_ci(perspective_scores, calibrated_scores, _safe_spearman)

    # Step 3: Threshold agreement + CI
    agreement = threshold_agreement(perspective_scores, calibrated_scores, DEFAULT_THRESHOLD)

    def _agreement_fn(a: np.ndarray, b: np.ndarray) -> float:
        return threshold_agreement(a, b, DEFAULT_THRESHOLD)

    ta_lower, ta_upper = _bootstrap_ci(perspective_scores, calibrated_scores, _agreement_fn)

    # Step 4: Gate check
    gate_passed = spearman_cal >= SPEARMAN_GATE and agreement >= THRESHOLD_AGREEMENT_GATE

    return AttributeGateResult(
        attribute=attribute,
        viable=True,
        spearman_raw=spearman_raw,
        spearman_calibrated=spearman_cal,
        spearman_ci_lower=sp_lower,
        spearman_ci_upper=sp_upper,
        threshold_agreement=agreement,
        threshold_agreement_ci_lower=ta_lower,
        threshold_agreement_ci_upper=ta_upper,
        gate_passed=gate_passed,
    )


def evaluate_gate(attribute_results: list[AttributeGateResult]) -> GateResult:
    """Compute overall gate result (per-attribute, not all-or-nothing)."""
    passed = sum(1 for r in attribute_results if r.gate_passed)
    total = len(attribute_results)
    _viable_count = sum(1 for r in attribute_results if r.viable)

    return GateResult(
        attributes=attribute_results,
        passed_count=passed,
        total_count=total,
        overall_viable=passed > 0,
    )
