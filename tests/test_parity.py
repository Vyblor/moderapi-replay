"""Parity tests against frozen baseline.

These tests validate that calibration + inference produce consistent
results against committed fixtures. Used by the GitHub Action.
"""

import pytest


@pytest.mark.skip(reason="Requires fixture generation — see TODOS.md data acquisition plan")
def test_frozen_baseline_parity():
    """Validate calibrated scores match frozen baseline within tolerance."""
    pass
