# A-2 Formal Attempt v1 — Preserved Failure

- Experiment ID: `stage3-a2-formal-failed-v1-20260912`
- Status: **FAILED, diagnosed**
- Claim scope: diagnostic only
- Objective: reproduce and preserve the A-2 first-epoch non-finite Open-Detect loss
- Leakage audit: PASS; Unknown samples loaded: 0
- Failure: released variance-KL became non-finite at zero-based batch 5

## Configuration and execution

- Setting: `A-2`
- Seed: `2022`
- Batch size: `512`
- Environment: `/home/birkenwald/data/SEU-WXY/conda_envs/2025-10-8-WXY-dgl_py310`
- Original execution session: `stage3_primary_continue_v1`

## Data and split

- Frozen source split: `data/splits/compatible_min1`
- Known-only training audit: PASS
- Unknown samples loaded during training/selection: `0`

## Core results

`raw_logvar.max()` reached `102.106575`; `exp(logvar)` contained two non-finite
values, prototype distances contained 17 non-finite values, and KLD/entropy/
discriminative/total loss became non-finite. Inputs, labels, `mu`, and
prototypes were finite. The exact diagnostics are in
`nonfinite_diagnostics.json`.

The overflow-triggered guard was then tested on the identical first six
batches: 6/6 remained finite (`guarded_recheck.json`).

## Preserved evidence

- `training_config.json`
- `failure_trace.md`
- `nonfinite_diagnostics.json`
- `guarded_recheck.json`
- `.tmux-task/stage3_primary_continue_v1/output.log`

## Limitations

This failed run produced no formal epoch metric or checkpoint and never reached
Unknown evaluation. It is evidence for a Known-train numerical failure only.

## Conclusion and next step

Rerun A-2 from the same seed and initialization with the documented
overflow-triggered guard; do not overwrite this attempt.
