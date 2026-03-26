# moderapi-replay

> Validate your Perspective API moderation thresholds still work with Detoxify.

Google's [Perspective API](https://perspectiveapi.com/) is sunsetting on December 31, 2026. If your platform uses Perspective for content moderation, your carefully tuned thresholds (`if TOXICITY > 0.8, flag for review`) need to survive the migration.

**moderapi-replay** runs your historical Perspective API outputs through [Detoxify](https://github.com/unitaryai/detoxify) and tells you — with statistical confidence — whether your thresholds still work.

## Quick Start

```bash
pip install moderapi-replay

# Run parity check against your Perspective API logs
moderapi-replay replay your_logs.jsonl --output report.html

# Estimate migration effort
moderapi-replay estimate "We use Perspective API for TOXICITY and INSULT with threshold 0.8"
```

## What It Does

1. **Parses** your Perspective API JSONL logs
2. **Runs** the same texts through Detoxify (toxic-bert)
3. **Calibrates** scores per-attribute (OLS + isotonic regression)
4. **Reports** per-attribute parity with bootstrap 95% confidence intervals

### Hard Gate Criteria

Per attribute:
- Spearman rank correlation ≥ 0.85
- Threshold agreement ≥ 90% at T=0.8

Attributes that fail pre-calibration viability (raw Spearman < 0.5) are flagged as semantically incompatible rather than "failed."

## Attribute Mapping

| Perspective API | Detoxify | Semantic Match |
|----------------|----------|----------------|
| TOXICITY | toxicity | ✅ Strong |
| SEVERE_TOXICITY | severe_toxicity | ✅ Strong |
| IDENTITY_ATTACK | identity_attack | ⚠️ Moderate |
| INSULT | insult | ✅ Strong |
| PROFANITY | obscene | ⚠️ Moderate |
| THREAT | threat | ⚠️ Moderate |

## Commands

| Command | Description |
|---------|-------------|
| `replay` | Run parity check against JSONL file |
| `calibrate` | Fit calibration coefficients |
| `estimate` | Estimate migration effort |
| `report` | Generate HTML parity report |
| `serve` | Start drop-in API server (Phase 2) |

## Installation

```bash
# Core CLI
pip install moderapi-replay

# With interactive TUI
pip install moderapi-replay[tui]

# With API server (Phase 2)
pip install moderapi-replay[server]

# Everything
pip install moderapi-replay[all]
```

## Development

```bash
git clone https://github.com/Vyblor/moderapi-replay.git
cd moderapi-replay
pip install -e ".[dev]"
pytest
```

## License

MIT
