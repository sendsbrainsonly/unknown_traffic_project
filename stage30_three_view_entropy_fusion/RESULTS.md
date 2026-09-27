# Stage 30 — True three-view entropy fusion

- Status: success; 20/20 Known-only development units complete.

## Data and split

- Frozen Stage20 Known Train/Validation, 3 independent Stage22/27 feature views. Known Test/Unknown usage 0.

## Configuration and execution

- Shared 30-epoch three-view Gaussian adapters; equal T0 versus per-flow entropy T1; at most three physical GPUs.

## Core results

- Preregistered gate: **FAIL**.
- T1−forced-equal mean Macro-F1: -0.003166; positive 8/20.
- See `stage30_report.md` for both dataset tables and paired results.

## Preserved evidence

- 20 validated run bundles, 40 independent CPU checkpoint replays, curves, predictions, weights, counterfactuals and input hashes.

## Limitations

- Frozen encoders, not end-to-end retraining; reused validation flows; no Test/open-set claim.

## Conclusion and next step

- Stop at the preregistered Known-Val gate; do not tune on Test/Unknown.
