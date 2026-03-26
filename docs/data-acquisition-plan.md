# Data Acquisition Plan

## Goal

Generate a ~6,000 text evaluation dataset with **paired** Perspective API and Detoxify scores for calibration and gate validation.

## Why Paired Scores Matter

Jigsaw and Civil Comments datasets have **human-annotated binary labels** (toxic/not toxic), not Perspective API's continuous 0.0–1.0 scores. Since moderapi-replay's calibration pipeline maps Detoxify scores to Perspective-equivalent scores, we need **actual Perspective API outputs** as ground truth.

This means: we must call the live Perspective API on each text to get the continuous scores, then pair them with Detoxify's local inference results.

## Data Sources

### 1. Jigsaw Toxic Comment Classification (CC0 License)
- **URL:** https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
- **License:** CC0 (public domain) — no restrictions on use
- **Format:** CSV with `comment_text` column + binary labels
- **Size:** ~160K comments from Wikipedia talk pages
- **Why:** Broad toxicity coverage, well-studied dataset, freely usable

### 2. Civil Comments (Public)
- **URL:** https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification
- **License:** Public dataset, CC0
- **Format:** CSV with `comment_text` column + continuous toxicity score (crowd-sourced)
- **Size:** ~1.8M comments from news sites
- **Why:** Different domain than Jigsaw (news vs. Wikipedia), ensures domain diversity

## Perspective API Constraints

### Terms of Service
- **Rate limits:** Free tier ~1 QPS, can request quota increase
- **Usage policy:** Scores may be stored for research/evaluation purposes
- **Sunset:** API accepting requests until December 31, 2026
- **Key requirement:** Apply at https://developers.perspectiveapi.com/

### Cost Estimate
- Free tier (1 QPS): ~6,000 texts ÷ 1 QPS = ~100 minutes
- With quota increase (10 QPS): ~10 minutes
- No per-request cost — Perspective API is free

### ⚠️ Timing Risk
The dataset **must** be generated before API shutdown. If the API goes down early or rate limits change, we lose the ability to generate ground truth. Generate the dataset in Week 0–1 and freeze it.

## Dataset Design

### Stratification Strategy

**Score band stratification** (primary):
- 10 bins across [0.0, 1.0): each bin 0.1 wide
- Target: ~100 texts per bin per attribute
- Why: Ensures calibration covers the full score range, not just the dense low-toxicity cluster

**Domain stratification** (secondary):
- Balance texts from Jigsaw (Wikipedia) and Civil Comments (news)
- Why: Calibration should work across content types, not overfit to one domain

### Target Composition

| Stratum | Count | Source |
|---------|-------|--------|
| Per score bin (×10) | ~100 texts | Mixed Jigsaw + Civil |
| Per attribute (×6) | ~1,000 texts | Score band stratified |
| **Total** | **~6,000 texts** | **Deduplicated** |

### 70/30 Train/Test Split

- **70% (4,200 texts):** Calibration fitting (OLS + isotonic regression)
- **30% (1,800 texts):** Gate validation (Spearman + threshold agreement)
- Split is stratified to maintain score band balance in both sets
- **Critical:** Test set is never seen during calibration fitting

## Generation Pipeline

```
┌─────────────────┐     ┌─────────────────┐
│  Jigsaw CSV      │     │ Civil Comments   │
│  (~5K texts)     │     │ (~5K texts)      │
└────────┬────────┘     └────────┬─────────┘
         │                        │
         └──────────┬─────────────┘
                    │
                    ▼
         ┌─────────────────────┐
         │ Filter: 10-5000 chars│
         │ Deduplicate          │
         └──────────┬──────────┘
                    │
                    ▼
         ┌─────────────────────┐
         │ Perspective API      │──→ Checkpoint every 100 texts
         │ (live, rate-limited) │
         └──────────┬──────────┘
                    │
                    ▼
         ┌─────────────────────┐
         │ Detoxify inference   │
         │ (local, batched)     │
         └──────────┬──────────┘
                    │
                    ▼
         ┌─────────────────────┐
         │ Stratify by          │
         │ score band + domain  │
         └──────────┬──────────┘
                    │
                    ▼
         ┌─────────────────────┐
         │ evaluation_dataset   │
         │ .jsonl (~6K records) │
         └─────────────────────┘
```

## JSONL Output Format

Each line:
```json
{
  "text": "the comment text",
  "source": "jigsaw",
  "perspective_scores": {
    "TOXICITY": 0.823,
    "SEVERE_TOXICITY": 0.102,
    "IDENTITY_ATTACK": 0.045,
    "INSULT": 0.671,
    "PROFANITY": 0.534,
    "THREAT": 0.012
  },
  "detoxify_scores": {
    "TOXICITY": 0.791,
    "SEVERE_TOXICITY": 0.089,
    "IDENTITY_ATTACK": 0.038,
    "INSULT": 0.645,
    "PROFANITY": 0.498,
    "THREAT": 0.009
  }
}
```

## Reading the Signals

After generating the dataset, here's how to interpret it:

### 1. Score Distribution Check
```bash
python scripts/generate_dataset.py --stats data/evaluation_dataset.jsonl
```
- **Healthy:** Each attribute has texts across all 10 bins (no empty bins)
- **Warning:** If high-score bins (0.8–1.0) are sparse for some attributes, those attributes may have less reliable calibration at the tail

### 2. Perspective vs. Detoxify Raw Correlation
Before calibration, check raw Spearman correlations:
```bash
moderapi-replay replay data/evaluation_dataset.jsonl --no-calibration
```
- **Spearman > 0.7:** Good raw alignment, calibration will likely pass the gate
- **Spearman 0.5–0.7:** Moderate alignment, calibration needed but feasible
- **Spearman < 0.5:** Semantic incompatibility — that attribute may not pass the gate even after calibration

### 3. Domain Balance
- Both Jigsaw and Civil Comments should be present in each score bin
- If one source dominates high-toxicity bins, the calibration may overfit to that domain's writing style

### 4. Red Flags
- **All zeros for an attribute:** Perspective API didn't return scores for that attribute — check API response
- **NaN values:** Bad text encoding or API error — filtered by the script but check counts
- **Identical Perspective scores:** API may have returned cached/default scores — investigate

## Usage

```bash
# 1. Set up API key
export PERSPECTIVE_API_KEY="your-key-here"

# 2. Download datasets
# Visit the Kaggle URLs above and download train.csv for each

# 3. Dry run (no API calls)
python scripts/generate_dataset.py \
  --jigsaw data/raw/jigsaw/train.csv \
  --civil-comments data/raw/civil_comments/train.csv \
  --dry-run

# 4. Generate dataset (takes ~2 hours at 1 QPS)
python scripts/generate_dataset.py \
  --jigsaw data/raw/jigsaw/train.csv \
  --civil-comments data/raw/civil_comments/train.csv \
  --output data/evaluation_dataset.jsonl \
  --verbose

# 5. If interrupted, resume from checkpoint
python scripts/generate_dataset.py \
  --resume data/evaluation_dataset.checkpoint.jsonl \
  --jigsaw data/raw/jigsaw/train.csv \
  --civil-comments data/raw/civil_comments/train.csv \
  --output data/evaluation_dataset.jsonl

# 6. Check stats
python scripts/generate_dataset.py --stats data/evaluation_dataset.jsonl
```

## Legal Notes

- **Jigsaw dataset:** CC0 license — no restrictions
- **Civil Comments:** CC0 — no restrictions
- **Perspective API scores:** ToS allows storing for evaluation/research
- **Detoxify scores:** Generated locally, no license constraints
- **Dataset distribution:** The generated JSONL contains original comment texts (CC0) paired with API scores — distributable under CC0

## Dependencies on Plan

- **Blocks:** Calibration pipeline, gate validation, CI parity checks
- **Blocked by:** Perspective API ToS review (completed — ToS allows this use)
- **Timeline:** Must complete before Phase 1 calibration work begins
- **Risk:** If Perspective API changes rate limits or shuts down early, generation becomes impossible

## Checklist

- [ ] Obtain Perspective API key
- [ ] Download Jigsaw train.csv from Kaggle
- [ ] Download Civil Comments train.csv from Kaggle
- [ ] Run dry-run to verify script
- [ ] Generate full dataset (~2 hours)
- [ ] Verify stats (score distribution, domain balance)
- [ ] Run calibration pipeline on generated dataset
- [ ] Freeze dataset for reproducible CI
