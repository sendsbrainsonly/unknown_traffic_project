# Stage 7 — CipherSpectrum Known-only Encoder Training

## Goal

Train three independent Open-Detect encoders for the Stage 6 Low, Medium, and High settings. This stage consumes only frozen `KNOWN_TRAIN` and `KNOWN_VALIDATION` rows.

## Frozen boundary

- Stage 6 protocol hashes must verify before input construction and before training.
- Unknown and Known Test PCAPs are not opened by the input builder.
- Unknown samples are not loaded, scored, predicted, calibrated, or used for model selection.
- Low, Medium, and High start independently from `released_weight_init`; checkpoint warm-start across settings is forbidden.
- Best checkpoint and early stopping use the harmonic mean of Known Validation Accuracy and Macro-F1 with patience 5.

## Status

Completed on 2026-09-13.

- Unit/integration tests: `3 passed`.
- Balanced-subset smoke gates: Low, Medium, and High all passed and were archived.
- Formal independent training: Low, Medium, and High all completed with patience-5 early stopping.
- All six run bundles passed artifact/hash validation.
- Aggregate Known Validation metrics and immutable checkpoint hashes are recorded in [`FORMAL_RESULTS.md`](FORMAL_RESULTS.md).
- `scripts/verify_stage7.py` provides a PCAP-free final audit of the frozen protocol, inputs, metrics, and checkpoints.
- Known Test/Unknown PCAPs opened or used: `0`; Unknown inference executed: `false`.

This stage does not report external open-set performance and does not authorize Unknown evaluation.
