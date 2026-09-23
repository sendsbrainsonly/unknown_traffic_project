# Experiment results: stage14c4-our-method-training-cleanup-20260918-v1

- Status: `partial`
- Experiment type: `training-mechanism-cleanup`
- Claim scope: `two-protocol Known-Train/Validation diagnostic`
- Created (UTC): `2026-09-18T03:02:38Z`
- Completed (UTC): `2026-09-18T04:04:09Z`
- Final conclusion: `NEEDS_FURTHER_DIAGNOSIS`
- Objective: Audit and clean the F2 own-method training mechanism on frozen VNAT Known Train/Validation, without converting it to native Open-Detect or using Test/Unknown data.

## Data and split

- Frozen Stage 14B protocol hash: `c904daef04d2c8e79469191121325c0a6ceeaeb3e595ef9c4c7d7a62c980ae2f`.
- Representative protocols: `medium_seed2025` (weak) and `medium_seed2026` (normal).
- Only frozen Known Train and Known Validation were opened.
- Known Test samples used: `0`; Unknown Test samples used: `0`; DES executed: `false`.
- Original F2 checkpoints and arrays remain immutable inputs; no Stage 14B/14C/14C.5 result was overwritten.

## Configuration and execution

- Own-method identity retained: F2 feature representation, `Stage3OpenDetectNet`, existing composite loss, prototype/representation mechanism, MultiStepLR `[50,80]` with `gamma=0.1`, released initialization, logvar clamp, and composite Known-Val checkpoint rule.
- Cleanup: disable `patience=5`, train exactly `100` epochs, reduce batch `512 -> 128`, and make prototype resets at epochs `51/81` optimizer-safe by copying in place and clearing the prototype Adam state.
- Four runs were executed serially on physical GPU `4`: two protocol-seed primary runs and two fixed-seed-2022 diagnostic controls.
- All `4/4` runs completed `100/100` epochs; no NaN, crash, or numerical-guard activation occurred.
- Open-Detect checkpoints were read only as closed-set references; their training configuration was not copied into the method.

## Core results

| Protocol | Version | Epochs | Best epoch | Val loss | Accuracy | Macro-F1 | Weighted-F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| Medium-2025 | Original F2 | 30 | 25 | 0.230403 | 0.893367 | 0.653313 | 0.864173 |
| Medium-2025 | Cleaned primary | 100 | 93 | 0.186996 | 0.892857 | 0.727247 | 0.864414 |
| Medium-2025 | Cleaned fixed-2022 control | 100 | 83 | 0.181006 | 0.897959 | 0.766133 | 0.873975 |
| Medium-2026 | Original F2 | 10 | 5 | 0.160869 | 0.975406 | 0.732555 | 0.972495 |
| Medium-2026 | Cleaned primary | 100 | 53 | 0.045150 | 0.994767 | 0.980668 | 0.994803 |
| Medium-2026 | Cleaned fixed-2022 control | 100 | 76 | 0.046100 | 0.995814 | 0.980992 | 0.995856 |

- Weak-run primary Macro-F1 recovery: `+0.073934`; this is meaningful but below the preregistered `+0.10` cleanup-pass threshold.
- Normal-run primary Macro-F1 change: `+0.248113`; no degradation occurred.
- Cleaned absolute seed deltas: Medium-2025 `0.038887`, Medium-2026 `0.000324`; both are smaller than their Stage 14C-3 controls.
- Under-training is resolved: both primary best epochs (`93`, `53`) occur after the old early-stopping window, and all runs reach epoch 100.
- Residual issue: individual validation epochs still show severe transient collapses, so optimization stability is improved but not fully explained.

## Preserved evidence

- `stage14c4_report.md`: complete scientific report and required answers.
- `cleanup_plan.json`: preregistered cleanup decisions and gate.
- `training_mechanism_audit.csv`: mechanism-by-mechanism decision and evidence.
- `cleanup_comparison.csv`: original, cleaned, seed-control, and reference metrics.
- `deterministic_split_metrics.csv`: deterministic Known Train/Validation re-evaluation.
- `training_dynamics.csv`: all epoch-level training/validation metrics.
- `seed_sensitivity.csv`: protocol-seed versus fixed-2022 comparison.
- `prototype_optimizer_link_audit.json`: direct proof of the old Parameter replacement defect and in-place fix.
- `runs/`, `checkpoints/`, `per_class_metrics/`, and `confusion_matrices/`: complete run evidence.
- Tmux logs: `.tmux-task/stage14c4_gpu4_serial/` and `.tmux-task/stage14c4_deterministic_eval/`.

## Limitations

- This is a two-protocol diagnostic, not the formal 15-run result.
- Fixed seed 2022 is a sensitivity control, not a production seed selection.
- Large transient validation collapses remain visible; the cleanup does not yet establish fully stable optimization.
- No Known Test, Unknown Test, open-set detector, or DES result was observed.

## Conclusion and next step

`NEEDS_FURTHER_DIAGNOSIS`. The cleanup fixes three concrete defects—premature early stopping, batch-512 update starvation, and optimizer-disconnecting prototype reset—and materially improves both representative runs. However, the weak-run primary gain does not meet the preregistered `+0.10` gate and transient validation collapses remain. Do not freeze or launch the formal 15-run pipeline yet; first isolate the remaining validation/optimization instability with a bounded Known-only diagnostic.
