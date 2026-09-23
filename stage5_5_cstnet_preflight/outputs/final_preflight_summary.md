# Stage 5.5 Final Preflight Summary

Final Preflight Gate: **NOT_READY**

## Frozen Integrity

- SHA-256 checks: 26/26 PASS.
- Frozen Stage 5 rule/protocol files were not modified.

## Q1: DGSB Global Gate Contribution

- Nonzero Global-exclusive rejection in every A-1/A-2/A-3 Known Validation/Test cell: **NO**.

## Q2: Global-vs-Local Class Consistency

- Maximum mismatch rate: **0.0000%**; structural mismatch (>5%): **NO**.

## Q3: CSTNET Actual Input Leakage

- Sample: 600 PCAPs, 600 valid actual 32x32 inputs.
- Full class-domain visibility: 0.0000%.
- Parsed-SNI visibility: 0.0000%.
- Direct domain/hostname/SNI visibility: 0.0000% (LOW_DIRECT_DOMAIN_LEAKAGE).
- Domain/token union visibility: 0.0000%.
- IP: MASKED; Port: VISIBLE.

## Q4: Calibration Sufficiency

- minimum_total_samples_per_class=100: **REVISE_BEFORE_TRAINING**.
- Protocol v2 recommended before training: **YES**.

## Blocking Issues

- Global Gate has zero independent effect in at least one setting/split

## Warnings

- some class-level calibration counts are weak
- component n<30 fallback risk is high in stress scenarios
- transport ports remain visible in the model input

## Boundary

No CSTNET training, formal mu_x extraction, scaler/PCA/GMM fit, boundary calibration, Unknown inference, UFAR, AUROC, or AUPRC was performed. USTC Unknown artifacts were not read.
