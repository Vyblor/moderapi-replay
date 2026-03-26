"""Tests for exception hierarchy."""

from moderapi.exceptions import (
    CalibrationError,
    ComparisonError,
    InferenceError,
    ModerAPIError,
    ParseError,
    ReportError,
    ServerError,
)


def test_exception_hierarchy():
    assert issubclass(ParseError, ModerAPIError)
    assert issubclass(InferenceError, ModerAPIError)
    assert issubclass(CalibrationError, ModerAPIError)
    assert issubclass(ComparisonError, ModerAPIError)
    assert issubclass(ReportError, ModerAPIError)
    assert issubclass(ServerError, ModerAPIError)


def test_exception_catches():
    try:
        raise CalibrationError("test")
    except ModerAPIError as e:
        assert str(e) == "test"
