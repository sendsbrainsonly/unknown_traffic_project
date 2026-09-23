# Boundary Leakage Audit

- Rule G threshold source: frozen Stage 3 Known Validation threshold.
- Rule C thresholds: true-class K2 mixture-score P05 on Known Validation only.
- Rule L thresholds: true-class posterior-assigned component joint-score P05 on Known Validation only.
- Rule L deterministic fallback: validation count `< 30` uses the class P05 threshold.
- Unknown samples used for threshold, quantile, fallback, K, model, or formula selection: `0`.
- Known Test was evaluated only after all thresholds were frozen.
- USTC Unknown Test was read only after Known Validation and Known Test evaluation; its results are `POST_HOC_DIAGNOSTIC_ONLY`.
- Quantile search performed: `NO`; only P05 was executed.
- Adaptive K, refitting, and Stage 3 threshold modification: `NO`.
