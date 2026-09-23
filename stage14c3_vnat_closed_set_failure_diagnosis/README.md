# Stage 14C-3 — VNAT Closed-Set Failure Diagnosis

This is a Known-Train/Validation-only diagnostic. It does not load Known Test or
Unknown Test samples, run DES, or modify the frozen Stage 14B protocol.

## Frozen evidence boundary

- Stage 14B freeze hash:
  `c904daef04d2c8e79469191121325c0a6ceeaeb3e595ef9c4c7d7a62c980ae2f`
- Existing OD, F0 and F2 artifacts are read-only inputs.
- All newly trained ablations use exactly the existing F0 Known Train/Val IDs,
  labels and byte images.
- Unknown application names may be read from the frozen protocol document only
  to prove disjointness; Unknown sample content and results are forbidden.

## Pre-registered representative protocols

- Normal: `medium_seed2026` — frozen F0 Known-Val Macro-F1 0.923316.
- Weak: `medium_seed2025` — frozen F0 Known-Val Macro-F1 0.284444.

Both are Medium protocols with seven Known classes and similar train sizes,
while differing strongly in frozen F0 quality.

## Component ablation plan

| ID | Change relative to frozen F0 | Execution |
|---|---|---|
| D0 | None | Reuse immutable F0 result |
| D1 | Batch size 512 -> 128 only | New Known-only run |
| D2 | OD optimizer/LR | Not rerun: already exactly Adam/0.001/betas/weight decay equal |
| D3 | OD scheduler | Not rerun: already exactly MultiStepLR [50,80], gamma 0.1 |
| D4 | Checkpoint selection composite -> Validation Accuracy only | New Known-only run; early-stop monitor remains composite |
| D5 | Remove F0 raw-logvar upper clamp only | New Known-only run |
| D6 | Protocol seed -> fixed training seed 2022 only | New Known-only run |
| D7 | Disable patience-5 early stopping only; complete 100 epochs | New Known-only run; batch remains 512 and checkpoint remains composite |
| D8 | Full released native OD training strategy | Reuse immutable native OD result; multi-factor reference, not a one-factor causal estimate |

F2 is also included as an immutable feature-reference row, not as a training
component ablation.

No additional hyperparameter values or protocols will be searched after
results are observed.

