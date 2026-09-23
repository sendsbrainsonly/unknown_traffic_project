# Experiment results: stage14c6-cleaned-pipeline-full15-closed-set-20260918-v1

- Status: `success`
- Experiment type: `full-grid-training`
- Claim scope: `formal frozen-protocol Known-Validation comparison`
- Created (UTC): `2026-09-18T08:15:29Z`
- Completed (UTC): `2026-09-18T12:25:57Z`
- Objective: run the frozen Stage 14C-4/14C-5 cleaned F2 method on all 15 Stage 14B Known-only protocols and compare it with the existing Native Open-Detect results.

## Data and split

- Stage 14B freeze hash: `c904daef04d2c8e79469191121325c0a6ceeaeb3e595ef9c4c7d7a62c980ae2f`.
- Protocols: Low/Medium/High × seeds 2022–2026 (`15` runs).
- Model-visible inputs: Known Train and Known Validation only.
- Known Test used: `0`; Unknown Test used: `0`; DES executed: `false`.

## Configuration and execution

- Own-method architecture and F2 representation; composite loss `0.005*(rec+kld+ent)+0.995*dis`.
- Batch size 128; 100 epochs; no early stopping; Adam LR 0.001.
- MultiStepLR milestones 50/80; in-place prototype reset at epochs 51/81 with optimizer link retained and prototype optimizer state cleared.
- Checkpoint selection: highest Known-Validation Accuracy/Macro-F1 harmonic mean.
- Four-way parallel execution on physical GPUs 0–3; each GPU ran one protocol at a time.
- Training wall time: `6869.68` seconds; all 15 runs finished successfully.

## Core results

| Setting | Accuracy mean ± std | Macro-F1 mean ± std | Weighted-F1 mean ± std |
|---|---:|---:|---:|
| Low | 0.876386 ± 0.046067 | 0.791160 ± 0.122757 | 0.867275 ± 0.042774 |
| Medium | 0.840436 ± 0.147691 | 0.802217 ± 0.125775 | 0.821985 ± 0.163654 |
| High | 0.888714 ± 0.083975 | 0.869709 ± 0.082148 | 0.885014 ± 0.087458 |
| Overall | 0.868512 ± 0.096451 | 0.821029 ± 0.109748 | 0.858092 ± 0.105427 |

- Runs/epochs/checkpoints completed and verified: `15/15`, `1500/1500`, `15/15`.
- Frozen weak-run count: `0`.
- Our-Cleaned − Native mean Δ Accuracy/Macro-F1/Weighted-F1: `+0.000451/−0.009875/−0.001386`.
- Our-Cleaned wins: Accuracy `7/15` with one tie; Macro-F1 `8/15`; Weighted-F1 `8/15`.
- Medium-2025/2026 reproduce Stage 14C-4 selected metrics exactly to `1e-12`.
- Final conclusion: `CLOSED_SET_PASS_WITH_HARD_PROTOCOLS`.

## Hard-protocol evidence

- Low-2025 Macro-F1 `0.581333`; simultaneously includes rsync/scp/sftp.
- Medium-2024 Macro-F1 `0.677427`; largest paired Macro-F1 deficit versus Native (`−0.175685`).
- Medium-2025 Macro-F1 `0.727247`; rsync Recall `0.031414` with rsync/scp simultaneously Known, reproducing the previously diagnosed hard composition.
- Mean difficult-class Recall: rdp `0.818182`, rsync `0.542599`, scp `0.778603`, sftp `0.451807`.

## Preserved evidence

- `stage14c6_report.md`: complete formal report and required answers.
- `run_level_results.csv`, `setting_summary.csv`: single-run and aggregate metrics.
- `per_class_results.csv`, `class_summary.csv`, `confusion_matrices/`: class-level evidence.
- `paired_vs_native.csv`, `paired_vs_native_summary.csv`, `per_class_vs_native.csv`: paired Native comparison.
- `training_dynamics.csv`: all 1,500 epoch records.
- `runs/*`: configurations, training histories, per-class metrics, predictions, latest checkpoints and completion markers.
- `checkpoints/*`: 15 selected best checkpoints with verified SHA256 values.
- Training tmux log/status: `../.tmux-task/stage14c6_gpu4_parallel_20260918_0834/`.

## Limitations

- This stage evaluates Known Validation only; it makes no Known Test or Unknown-detection claim.
- Native retains its previously frozen fixed-2022 seed policy, whereas Our-Cleaned uses each protocol seed; pairing is exact by protocol/data but not a single-seed architecture ablation.
- Known-class pair associations are descriptive because multiple classes change jointly across protocols.
- rdp validation support is only four samples in several protocols, so its per-class estimates are high variance.

## Conclusion and next step

The cleaned training pipeline can be formally frozen. It is operationally stable and comparable with Native Open-Detect overall, while retaining repeatable composition-specific weaknesses. It may proceed to a separately frozen open-set evaluation; this experiment did not start that evaluation.
