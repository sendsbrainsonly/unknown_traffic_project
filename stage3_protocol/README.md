# Stage 3 Protocol Freeze

Status: **FROZEN BEFORE UNKNOWN TEST** on 2026-09-12.

This directory freezes the next experiment's class scenarios, fixed sample split, leakage constraints, representation, support models, threshold, metrics, and decision gates. It contains no Unknown Detection result and did not trigger training.

## What is official and what is adapted

- Official/recovered: USTC scenarios A-1/A-2/A-3, their 19/1, 17/3, 15/5 class counts, and exact held-out class names.
- Official/recovered: 95% Known Validation acceptance rule and five-fold reporting claim.
- Not recovered: author sample-level five-fold assignments, fold seeds/indices, and standalone validation artifact.
- Adapted/frozen here: the existing project `compatible_min1` 80/10/10 flow split (`seed1=41`, `seed2=42`) is used within each official class scenario. This is not author-exact five-fold reproduction.

## Files

- `opendetect_protocol_source.md`: paper/source evidence and deviations.
- `stage3_protocol.json`: machine-readable normative protocol.
- `stage3_protocol.md`: human-readable normative protocol.
- `class_inventory.csv`: class names, official/current IDs, type, source counts.
- `split_manifest.csv`: all three official class scenarios translated by class name.
- `fold_A1.json`, `fold_A2.json`, `fold_A3.json`: frozen class-held-out scenarios; these are not author sample folds.
- `split_hashes.sha256`: integrity hashes.

## Execution boundary

Each scenario requires a new Known-only Open-Detect encoder. The existing 20-class Open-Detect checkpoint is diagnosis-only and must never be used as the formal Stage 3 detector. Unknown rows from source train/validation are excluded and must not be loaded; only their source test rows become Unknown Final Test.
