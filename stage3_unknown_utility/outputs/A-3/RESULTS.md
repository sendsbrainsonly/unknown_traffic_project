# A-3 Strict Unknown-Free Utility Results

- Status: **SUCCESS**
- Claim scope: local frozen-split approximate experiment; not author-fold equivalent

## Data and split

- Known / Unknown classes: 15 / 5
- Known train / validation / test: 308836 / 38604 / 38605
- Unknown final-test: 10306
- Leakage audit: PASS; Unknown usage before final evaluation: 0

## Configuration and execution

- Checkpoint selection: Known Validation Accuracy/Macro-F1 harmonic mean only
- Threshold calibration: 95% Known Validation acceptance only
- Density comparison: PCA-64 + full covariance, K=1 versus fixed K=2
- Bootstrap: 1,000 paired class-stratified resamples, seed 0

## Core results

| Detector | Known test FRR | Unknown FAR | AUROC | AUPRC-Unknown |
|---|---:|---:|---:|---:|
| Native | 0.049838 | 0.383369 | 0.928339 | 0.808421 |
| Single-Full | 0.047585 | 0.116437 | 0.976963 | 0.933100 |
| Multi-Full-K2 | 0.048595 | 0.093538 | 0.967501 | 0.930243 |


- Delta Unknown FAR (Multi - Single): `-0.022899282`; negative favors Multi.
- Paired-bootstrap 95% CI: `[-0.025715603, -0.020085387]`.

## Preserved evidence

- Final predictions, detector metrics, per-class results, decision transitions,
  absorption counts, K2 stability diagnostics, and all 1,000 bootstrap samples
  are retained in this setting bundle.
- Checkpoints, latent exports, scaler, PCA, GMMs, and tmux logs are retained as
  hash-verified external artifacts.

## Limitations

- Official Open-Detect five sample-fold identities are unavailable; this uses
  the frozen local 80/10/10 flow split and official class-holdout scenario.
- Fixed K2 is a controlled utility diagnostic, not evidence that every class
  has two semantic modes.

## Conclusion and next step

The frozen cross-setting result is **Gate D (MIXED)**;
multi-component utility is not established across all three primary settings.
Per the task boundary, do not proceed to a later stage.
