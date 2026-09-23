# Stage 14C — VNAT Open-Detect Encoder Training

## Scope and integrity

- Completed runs: **15/15**, all exit 0 and finite.
- Unknown samples used in training/validation: **0/0** for every run.
- Known Test samples used: **0**; Unknown inference executed: **false**.
- Encoder initialization: independent released Open-Detect initialization for every protocol; no new hyperparameter search.
- Checkpoint selection: Known Validation harmonic mean of Accuracy and Macro-F1 only; patience 5, maximum 100 epochs.
- Stage 14B freeze hash before/after: `c904daef04d2c8e79469191121325c0a6ceeaeb3e595ef9c4c7d7a62c980ae2f` / `c904daef04d2c8e79469191121325c0a6ceeaeb3e595ef9c4c7d7a62c980ae2f` (**PASS**).
- Checkpoints saved/hash-verified/unique: **15/15/15**.

## Known Validation results (five seeds)

| Setting | Accuracy mean ± population SD | Macro-F1 mean ± population SD | Accuracy range | Macro-F1 range |
|---|---:|---:|---:|---:|
| Low | 0.817794 ± 0.085747 | 0.594534 ± 0.133691 | 0.658260–0.904673 | 0.445140–0.804239 |
| Medium | 0.759880 ± 0.165323 | 0.559701 ± 0.240777 | 0.542959–0.986918 | 0.284444–0.923316 |
| High | 0.821455 ± 0.140071 | 0.690267 ± 0.143272 | 0.590674–0.966964 | 0.429056–0.849572 |

## Run-level results

| Setting | Seed | Best/stop epoch | Train loss | Validation loss | Validation Accuracy | Validation Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| High | 2022 | 13/18 | 0.263062 | 0.205892 | 0.926058 | 0.849572 |
| High | 2023 | 14/19 | 0.797393 | 0.654247 | 0.732612 | 0.717431 |
| High | 2024 | 10/15 | 1.361767 | 1.181231 | 0.590674 | 0.429056 |
| High | 2025 | 16/21 | 0.300989 | 0.259779 | 0.890966 | 0.675224 |
| High | 2026 | 4/9 | 0.276452 | 0.253112 | 0.966964 | 0.780054 |
| Low | 2022 | 8/13 | 0.529857 | 0.424114 | 0.873754 | 0.597320 |
| Low | 2023 | 18/23 | 0.431116 | 0.372390 | 0.841841 | 0.665631 |
| Low | 2024 | 10/15 | 0.905681 | 1.166414 | 0.658260 | 0.460337 |
| Low | 2025 | 4/9 | 0.728433 | 0.629853 | 0.810442 | 0.445140 |
| Low | 2026 | 18/23 | 0.252113 | 0.229309 | 0.904673 | 0.804239 |
| Medium | 2022 | 15/20 | 0.311947 | 0.253384 | 0.901565 | 0.732211 |
| Medium | 2023 | 8/13 | 1.417595 | 1.149912 | 0.542959 | 0.524399 |
| Medium | 2024 | 11/16 | 1.613423 | 1.584176 | 0.627143 | 0.334134 |
| Medium | 2025 | 5/10 | 1.367820 | 1.153479 | 0.740816 | 0.284444 |
| Medium | 2026 | 13/18 | 0.118483 | 0.112458 | 0.986918 | 0.923316 |

## Quality diagnosis

- Training crashes, NaN, missing checkpoints, or hash mismatches: **none**.
- Numerically abnormal runs: **none**.
- Weak Known Validation Macro-F1 flag (`<0.35`): **2/15** — Medium-2024 (0.334134), Medium-2025 (0.284444).
- These flags indicate uneven minority-class representation quality under the frozen, highly imbalanced VNAT class counts; they are not hidden or retuned and do not indicate data leakage.

## Completion judgment

- Stage 14C status: **PASS_WITH_QUALITY_FLAGS**.
- Stage 14D technical readiness: **YES**, using the 15 frozen checkpoint hashes below and preserving run-level quality stratification.
- Stage 14D was **not** started.
