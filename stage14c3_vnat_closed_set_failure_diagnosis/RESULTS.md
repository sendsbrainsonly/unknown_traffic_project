# Experiment results: stage14c3-vnat-closed-set-failure-diagnosis-20260918-v1

- Status: `success`
- Experiment type: `failure-diagnosis`
- Claim scope: `diagnostic`
- Created (UTC): `2026-09-18T02:02:28Z`
- Completed (UTC): `2026-09-18T02:47:50Z`
- Objective: diagnose the verified `0.18–0.22` Macro-F1 gap between native Open-Detect and F0/F2 using frozen VNAT Known Train/Validation only.

## Data and split

- Stage 14B frozen protocol hash: `c904daef04d2c8e79469191121325c0a6ceeaeb3e595ef9c4c7d7a62c980ae2f`.
- All 30 OD-vs-F0/F2 class/sample/label/count/hash comparisons: `PASS`.
- Known Test / Unknown Test samples read: `0 / 0`; DES executed: `false`.

## Configuration and execution

- Static audit covers all 15 frozen protocols and 2,026 epoch records.
- Controlled ablation uses `Medium-2025` (weak) and `Medium-2026` (normal).
- Ten new Known-only runs completed on physical GPUs 4/5, one serial worker per GPU.
- The first four-way launch was stopped on request, preserved under `interrupted_attempts/`, and excluded from formal results.

## Core results

- Overall Val Macro-F1: F0 `0.614834`, F2 `0.652969`, native OD `0.830904`.
- D1 batch-only weak/normal: `0.487150 / 0.930687`.
- D6 seed-only weak/normal: `0.666589 / 0.816054`.
- D7 no-early-stop weak/normal: `0.553769 / 0.976402`.
- Final diagnosis: `ROOT_CAUSE_IDENTIFIED`.
- Dominant cause: interaction of batch-512 update deficit, patience-5 premature stopping, and seed-sensitive optimization; not data mismatch or F0 preprocessing.

## Preserved evidence

- `stage14c3_failure_diagnosis.md`
- `training_config_diff.md`
- `data_alignment_audit.csv`
- `feature_distribution_audit.csv`
- `training_dynamics.csv`
- `per_class_comparison.csv`
- `component_ablation.csv`
- `input_gradient_audit.csv`
- `confusion_pair_analysis.csv`
- `batch_imbalance_audit.csv`
- `ablation_runs/`, `interrupted_attempts/`, `figures/`, `scripts/`
- project-local `.tmux-task/stage14c3_*` logs and exit-status records

## Limitations

- The controlled component ablation uses two representative protocols, not all 15.
- One-factor effects are non-additive; D6 proves seed sensitivity but is not a universally better seed.
- F2's independent benefit must be reevaluated under the corrected native training pipeline before any formal feature freeze.

## Conclusion and next step

Use native Open-Detect training as the authoritative closed-set baseline. Preserve F0 as a diagnosed legacy baseline and F2 as a separate feature ablation. Do not proceed to Unknown Test or DES until a new formal encoder-training stage is explicitly authorized.
