# A-1 Strict Unknown-Free Utility Results

- Status: **SUCCESS**
- Claim scope: local frozen-split approximate experiment; not author-fold equivalent

## Data and split

- Known / Unknown classes: 19 / 1
- Known train / validation / test: 384478 / 48059 / 48061
- Unknown final-test: 850
- Leakage audit: PASS; Unknown usage before final evaluation: 0

## Configuration and execution

- Checkpoint selection: Known Validation Accuracy/Macro-F1 harmonic mean only
- Threshold calibration: 95% Known Validation acceptance only
- Density comparison: PCA-64 + full covariance, K=1 versus fixed K=2
- Bootstrap: 1,000 paired class-stratified resamples, seed 0

## Core results

| Detector | Known test FRR | Unknown FAR | AUROC | AUPRC-Unknown |
|---|---:|---:|---:|---:|
| Native | 0.049853 | 0.004706 | 0.993904 | 0.555943 |
| Single-Full | 0.047648 | 0.522353 | 0.955767 | 0.185912 |
| Multi-Full-K2 | 0.047419 | 0.623529 | 0.940232 | 0.171242 |


- Delta Unknown FAR (Multi - Single): `0.101176471`; negative favors Multi.
- Paired-bootstrap 95% CI: `[0.082352941, 0.122352941]`.

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
