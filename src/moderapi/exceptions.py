"""Custom exception hierarchy for ModerAPI.

All exceptions inherit from ModerAPIError for catch-all handling.
Each exception carries context for structured logging.
"""

from __future__ import annotations


class ModerAPIError(Exception):
    """Base exception for all ModerAPI errors."""


# --- Parser Errors ---


class ParseError(ModerAPIError):
    """Failed to parse input data."""


class StreamingParseError(ParseError):
    """Error during streaming JSONL parse (e.g., MemoryError on huge lines)."""


class UnicodeParseError(ParseError):
    """Input contains invalid UTF-8 sequences."""


# --- Inference Errors ---


class InferenceError(ModerAPIError):
    """ML model inference failed."""


class ModelNotFoundError(InferenceError):
    """Model files not found locally."""


class ModelDownloadError(InferenceError):
    """Failed to download model on first run."""


class ModelLoadError(InferenceError):
    """Model loaded but failed to initialize."""


# --- Calibration Errors ---


class CalibrationError(ModerAPIError):
    """Calibration computation failed."""


class InsufficientDataError(CalibrationError):
    """Not enough data points for statistical computation."""


class ConstantDataError(CalibrationError):
    """All values are identical — Spearman correlation undefined."""


class CalibrationFileError(CalibrationError):
    """Failed to read/write calibration.json."""


# --- Comparison Errors ---


class ComparisonError(ModerAPIError):
    """Score comparison failed."""


class AttributeMismatchError(ComparisonError):
    """Perspective and Detoxify attribute sets don't match."""


# --- Report Errors ---


class ReportError(ModerAPIError):
    """Report generation failed."""


class DiskSpaceError(ReportError):
    """Insufficient disk space for report output."""


# --- Server Errors (Phase 2) ---


class ServerError(ModerAPIError):
    """API server error."""
