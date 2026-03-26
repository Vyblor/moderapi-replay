"""Tests for calibration module."""

import numpy as np
import pytest

from moderapi.calibration import (
    _ols_fit,
    apply_ols,
    calibrate_attribute,
    clamp,
    threshold_agreement,
)
from moderapi.exceptions import ConstantDataError, InsufficientDataError


def test_clamp():
    assert clamp(0.5) == 0.5
    assert clamp(-0.1) == 0.0
    assert clamp(1.5) == 1.0


def test_apply_ols():
    assert apply_ols(0.5, slope=1.0, intercept=0.0) == 0.5
    assert apply_ols(0.5, slope=2.0, intercept=-0.5) == 0.5
    # Clamping
    assert apply_ols(0.9, slope=2.0, intercept=0.0) == 1.0
    assert apply_ols(0.1, slope=1.0, intercept=-0.5) == 0.0


def test_ols_fit():
    x = np.array([0.1, 0.2, 0.3, 0.4, 0.5] * 10)
    y = np.array([0.12, 0.22, 0.28, 0.42, 0.48] * 10)
    slope, intercept, r_sq = _ols_fit(x, y)
    assert 0.8 < slope < 1.2
    assert r_sq > 0.9


def test_ols_fit_insufficient_data():
    with pytest.raises(InsufficientDataError):
        _ols_fit(np.array([1.0]), np.array([1.0]))


def test_ols_fit_constant_data():
    with pytest.raises(ConstantDataError):
        _ols_fit(np.array([0.5] * 50), np.array([0.6] * 50))


def test_threshold_agreement():
    a = np.array([0.9, 0.7, 0.85, 0.3])
    b = np.array([0.85, 0.6, 0.75, 0.2])
    # At T=0.8: a=[above, below, above, below], b=[above, below, below, below]
    # Agreement: positions 0,1,3 agree (3/4 = 0.75)
    assert threshold_agreement(a, b, threshold=0.8) == 0.75


def test_calibrate_attribute(sample_scores):
    perspective = np.array(sample_scores["perspective"])
    detoxify = np.array(sample_scores["detoxify"])
    coeffs, iso_model = calibrate_attribute(detoxify, perspective)
    assert coeffs.method in ("ols", "isotonic")
