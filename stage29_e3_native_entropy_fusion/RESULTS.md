# Stage 29 native E3-only entropy fusion

- Status: success; 20/20 Known-only development units complete.

## Data and split

- Frozen Stage20 Known Train/Validation; E3 TrafficFormer and FIG/TAGCN only. YaTC/Test/Unknown usage 0.

## Configuration and execution

- Same 30-epoch Gaussian adapters for N0/N1; shared-λ dynamic versus equal-weight control; at most three physical GPUs.

## Core results

- Preregistered gate: **FAIL**.
- N1−forced-equal average Macro-F1: +0.000441.
- See `stage29_report.md` for both dataset tables and paired counts.

## Preserved evidence

- 20 validated unit bundles, 40 CPU checkpoint replays, curves, predictions, weights, counterfactuals and source hashes.

## Limitations

- Frozen encoders, not end-to-end retraining; repeated validation flows; no untouched Test or Unknown claim.

## Conclusion and next step

- Stop at the preregistered Known-Val gate; no next stage automatically launched.
