# TODOS

## Data Acquisition Plan
- **What:** Specify exact data sources for the ~6K stratified evaluation dataset
- **Why:** Need legally reusable text + Perspective labels for calibration and gate validation
- **Context:** Candidates: Jigsaw Toxic Comment dataset (CC0), Civil Comments (public), or fresh Perspective API calls against public corpora. Score band + domain stratification. Resolve before implementation.
- **Priority:** P1 (blocks calibration work)

## Per-Customer Calibration Profiles
- **What:** Support per-customer calibration.json files tuned to their specific threshold/attribute usage
- **Why:** Different customers use different thresholds and attribute subsets
- **Context:** Phase 2 feature. Requires per-attribute gate (implemented) as prerequisite.
- **Priority:** P2
