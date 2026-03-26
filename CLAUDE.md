# ModerAPI Replay

## Project Overview
CLI tool to validate Perspective API moderation thresholds work with Detoxify.
Drop-in replacement validation for teams migrating from Google's sunsetting Perspective API.

## Tech Stack
- Python >=3.10
- Detoxify (toxic-bert, native PyTorch — NOT ONNX in Phase 1)
- Typer (CLI), Rich (output formatting)
- Pydantic (models), FastAPI (Phase 2 server)
- Textual (TUI, optional)
- pytest + pytest-cov (testing)
- ruff (linting), mypy (type checking)

## Project Structure
```
src/moderapi/       # Main package
  models.py         # Pydantic models (Perspective API format)
  parser.py         # Streaming JSONL parser
  inference.py      # Detoxify inference engine
  calibration.py    # OLS + isotonic calibration
  comparison.py     # Spearman, threshold agreement, bootstrap CI
  report.py         # HTML report generation
  estimator.py      # Migration effort estimator
  tui.py            # Textual TUI (3 screens)
  cli.py            # Typer CLI entry point
  server.py         # FastAPI server (Phase 2)
  exceptions.py     # Custom exception hierarchy
tests/              # pytest tests
  fixtures/         # Test data (small committed, large generated)
```

## Commands
```bash
# Run tests
pytest

# Run with coverage
pytest --cov=moderapi --cov-report=term-missing

# Lint
ruff check src/ tests/

# Type check
mypy src/moderapi/

# Format
ruff format src/ tests/
```

## Key Design Decisions
- 70/30 train/test split for calibration vs gate validation
- Per-attribute gate (not all-or-nothing) — reports X of 6 pass
- Pre-calibration semantic viability: raw Spearman < 0.5 = incompatible
- Calibration method selection: try both OLS and isotonic, keep better threshold agreement
- Bootstrap 95% CI on all gate metrics
- Score band + domain stratified sampling (~6K total)
- --redact flag (default ON) strips user text from reports
- GH Action validates against frozen baseline (no live Perspective API calls)

## Testing
pytest with fixtures. Small fixture committed (10 texts), large fixture to be generated.
