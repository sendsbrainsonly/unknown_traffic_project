# Experiment results: stage16s-service-open-set-benchmark-20260920-v1

- Status: `success_with_preserved_recovery`
- Claim scope: `diagnostic`

## Data and split

- Six LOSO Service protocols over the frozen 3,065-flow development pool.
- Historical Known Train retained; historical Known Validation split deterministically into new Known Validation/Test.
- All labels are weak capture labels; P2P has one capture.

## Configuration and execution

- Seeds: 2022, 2023, 2024; 18 corrected Open-Detect five-class trainings.
- One run (Streaming-2023) retained a post-training parity failure; its frozen checkpoint was evaluated canonically without retraining.
- Four frozen scores share each checkpoint; Known-Val P95 only; no Unknown/Test fitting.

## Core results

- OD-Native mean AUROC: `0.623635`.
- DES-v0 mean ΔAUROC: `0.028376` (11/18 positive).
- DES-v1 mean ΔAUROC: `0.062949` (13/18 positive).
- H1 mean ΔAUROC: `0.018835` (8/18 positive).

## Preserved evidence

- All required CSVs, checkpoints, logs, score arrays, predictions, configs, hashes, and reports are stored in this bundle.

## Limitations

- Flow-level weak-label study; captures overlap; P2P has one capture; three seeds are not independent datasets.

## Conclusion and next step

See `stage16s_report.md` for method- and Service-conditional conclusions. Do not start a new detector or encoder automatically.
