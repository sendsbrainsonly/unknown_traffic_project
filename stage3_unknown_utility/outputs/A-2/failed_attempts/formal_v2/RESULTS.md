# A-2 Formal Attempt v2 — Preserved Failure

- Experiment ID: `stage3-a2-formal-failed-v2-20260912`
- Status: **FAILED, diagnosed**
- Claim scope: diagnostic only
- Leakage audit: PASS; Unknown samples loaded: 0

## Configuration and execution

- Setting: `A-2`
- Seed: `2022`
- Batch size: `512`
- Environment: `/home/birkenwald/data/SEU-WXY/conda_envs/2025-10-8-WXY-dgl_py310`
- Formal session: `stage3_a2_formal_v2`
- Diagnostic session: `stage3_a2_nan_guard_diag_v2`

## Data and split

- Frozen source split: `data/splits/compatible_min1`
- Known train / validation: `346651 / 43331`
- Known classes: `17`
- Unknown classes excluded: `Geodo`, `Htbot`, `Tinba`
- Unknown samples loaded during training/selection: `0`

## Core results

The first guard prevented direct `exp(logvar)` infinity, but the exact replay
showed a non-finite encoder gradient at zero-based batch 44 while all forward
losses were still finite. The batch had `raw_logvar.max=87.541748` and
`KLD=1.0490577230064987e35`; 76 entries of `encoder.conv1.weight` gradient were
non-finite. This proves that waiting for a non-finite forward variance term is
too late to prevent float32 backward overflow.

The replacement guard bounds only the upper log-variance tail at 20. A full
first Known-train epoch (678 batches) then completed with finite losses,
gradients, and parameters (`upper_guard_epoch_recheck.json`).

## Preserved evidence

- `training_config.json`
- `nonfinite_diagnostics.json`
- `upper_guard_epoch_recheck.json`
- `.tmux-task/stage3_a2_formal_v2/output.log`
- `.tmux-task/stage3_a2_nan_guard_diag_v2/output.log`
- `.tmux-task/stage3_a2_guard_epoch_v3/output.log`

## Limitations

This failed run produced no formal epoch metric or checkpoint and never reached
Unknown evaluation. It is evidence for a Known-train numerical failure only.

## Conclusion and next step

Rerun A-2 from the same seed and from-scratch initialization with the documented
upper-logvar bound; do not overwrite this attempt.
