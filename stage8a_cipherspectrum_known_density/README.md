# Stage 8A — CipherSpectrum Known-Only Representation, Density and Boundary Freeze

## Goal

Freeze deterministic Open-Detect representations, train-only StandardScaler/PCA64 and K1/K2 full-covariance density models, then calibrate five fixed detector boundaries on Known Validation only.

## Frozen Stage6 Protocol

The 40-class canonical 120k protocol, Low/Medium/High folds, and all 16 Stage 6 protocol hashes are immutable inputs.

## Frozen Stage7 Encoders

Only the formal best checkpoints are permitted: Low epoch 26, Medium epoch 22, and High epoch 22. Retraining, resume, fine-tuning, warm-start, latest checkpoints, and architecture changes are forbidden.

## Representation Extraction

The formal representation is deterministic latent mean `mu_x` with dimension 128. Only Stage 7 Known Train and Known Validation tensors are opened.

## Train-Only Scaler/PCA

StandardScaler and randomized PCA64 (`random_state=0`) are fit on Known Train only and transform Known Train/Validation.

## K1

Each Known class receives one full-covariance Gaussian with `reg_covar=1e-3` fit on train-only PCA64 values.

## K2

Each Known class receives a full-covariance two-component GMM with `n_init=3`, `max_iter=300`, `random_state=0`, and `reg_covar=1e-3` fit on train only. No Adaptive K or K>2 is allowed.

## Component Diagnostics

Convergence, weights, train/validation assignments, total variation, covariance log determinants/condition numbers, Euclidean separation, and pooled Mahalanobis separation are recorded without tuning.

## Native Boundary

Native score strictly reuses the verified Open-Detect definition: negative minimum KL divergence to frozen prototypes. Its threshold is the Stage 3 linear P05 Known-Validation quantile.

## Global Boundary

Single-K1 and Multi-K2 global thresholds target 95% Known-Validation acceptance. The closest empirical acceptance is selected; an exact tie uses the higher, more conservative threshold.

## Class-P05

Each class threshold is the linear P05 of its true-class Known-Validation K2 class score.

## Component-P05

Each component threshold is the linear P05 of weighted local scores after posterior assignment only within the sample's true-class train-fitted K2 model.

## Calibration Capacity

Component validation counts below 10/20/30/50/100 are reported. A fixed `val_n < 30` component falls back to its class P05 threshold; the cutoff is not tuned here.

## Known Validation Diagnostics

Classification, confusion, density-fit, overall/per-class/per-component coverage diagnostics use Known Validation only and cannot change any frozen choice.

## Strict Test Isolation

Known Test and Unknown Test PCAPs, tensors, representations, scores, and predictions must remain unopened and ungenerated. Formal scripts maintain an explicit file-access ledger and fail before a forbidden role can be opened.

## Frozen Evaluation Configuration

`configs/evaluation_config.json` will freeze Native, Single-Full-K1, Multi-Global-K2, Class-P05-K2, and Component-P05-K2 for the next stage. DGSB-v1/v2 are excluded.

## Final Gate

**READY_FOR_FINAL_TEST.** Stage 6/7 provenance, all three deterministic representation exports, train-only transforms/models, validation-only thresholds, access ledgers, evaluation configuration, and Low/Medium/High bundle hashes independently passed verification.

- Low/Medium/High frozen files verified: `107/99/91`.
- Known Test opened / representation generated: `0 / false`.
- Unknown Test opened / representation generated / inference executed: `0 / false / false`.

## Next Step

Stop here. Test access requires a separate next-stage instruction and must consume the frozen `configs/evaluation_config.json` and bundle hashes without refitting or recalibration.
