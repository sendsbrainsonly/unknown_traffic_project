# Open-Detect USTC Encoder Robustness Audit

## Research Background

The completed USTC-TFC2016 TrafficFormer analyses found that a single diagonal Gaussian was insufficient, while covariance misspecification explained a substantial part of the apparent multimodality. Under train-only PCA64 plus full covariance, K>1 retained held-out validation improvement for the representative classes. Stage 2.6 also found that some GMM assignments were strongly associated with packet count, directional packet counts, total bytes, and other simple flow statistics.

Those results do not yet distinguish between:

- stable representation-level class structure in USTC traffic; and
- a phenomenon specific to the TrafficFormer encoder or its training objective.

## Experimental Question

Keep the USTC-TFC2016 dataset, actual 20-class label mapping, sample universe, train/validation/test split, and analysis classes fixed. Change only the representation pipeline from TrafficFormer to the author-code Open-Detect representation, then compare full-covariance Gaussian/GMM validation NLL for K=1, K=2, and K=3.

Core question:

> Does the observed class-conditional Gaussian complexity persist when the encoder is changed from TrafficFormer to Open-Detect?

The controlled comparison is:

- TrafficFormer representation;
- Open-Detect CVAE mean representation (`mu_x`).

## Scope Boundary

This audit does not determine:

- semantic attack subtypes;
- the true number of modes;
- Unknown Detection or unknown discovery;
- Adaptive K;
- prototype-specific decision boundaries.

It is an encoder robustness audit, not the final Unknown Detection experiment.

## Representative Classes

- FTP
- Cridex
- Miuref
- Outlook

These classes already have TrafficFormer Stage 2.5 and Stage 2.6 diagnostics and therefore support a controlled encoder comparison.

## Experimental Principles

- Dataset fixed.
- Split fixed.
- Labels fixed.
- Classes fixed.
- Evaluation fixed.
- Only the representation pipeline changes.
- Open-Detect upstream remains read-only.
- Train data fit the representation and Gaussian models; validation is evaluation-only for Gaussian diagnosis.
- Test data are excluded from Gaussian fitting and the main diagnosis.

## Current Status

Open-Detect formal training, latent Gaussian diagnosis, and the K=2 cross-encoder component correspondence/shortcut audit are COMPLETE.

## Completed Experiments

- Historical USTC protocol recovery: PASS. The executed split contains 391,280 train, 48,910 validation, and 48,911 test flows across the fixed 20 classes.
- SMB-1 and SMB-2 were verified as one logical `SMB` class (ID 12).
- Existing TrafficFormer Stage 2.5/2.6 results remain immutable comparison inputs.
- Open-Detect source audit: PASS. The actual released model is a 128-dimensional custom ResNet18 CVAE with one learnable unit-covariance Gaussian prototype per class and the released generative/discriminative objective.
- Flow alignment: PASS, 489,101/489,101 inputs reconstructed with no class/split failure.
- Smoke: PASS. The full 20-class graph produced finite `mu`, `logvar`, sampled `z`, reconstruction and all loss terms; training loss decreased.

## Main Results

Formal training is incomplete. Current validation-best checkpoint (epoch 6): accuracy 97.3993%, macro-F1 97.1561%. These are progress metrics, not the final encoder-robustness result.

## Interpretation

No encoder-robustness conclusion is available until exact flow alignment, a successful smoke test, formal 20-class training, `mu_x` extraction, and held-out Gaussian evaluation are complete.

## Known Limitations

- The exact historical 20-class mapping and split must be recovered from executed artifacts rather than reconstructed from current scripts.
- Some current split scripts may represent SMB-1 and SMB-2 as separate classes; that definition must not replace the historical 20-class experiment mapping.
- Open-Detect's released USTC NPZ files cannot be aligned one-to-one with the historical Stage 1 flow IDs. Formal inputs must be reconstructed from existing Stage-0 PKLs; the alignment success rate is not yet known.
- The released loss/decoder/prototype-reset implementation differs from the paper in several important details documented in `outputs/opendetect_source_audit.md`.

## Next Step

Continue the fixed-split 100-epoch, 20-class training on physical GPU 3, then extract train/validation `mu_x` and run the predeclared Gaussian audit.

## Latent Gaussian Audit

- Status: COMPLETE
- Frozen best checkpoint: epoch 32, SHA-256 `f8e53d36bd6dc8333451b2f159300bc5a2cd875fccaff01a2c6dc43356951b36`
- Training stopped at epoch 37: validation combined score did not improve for 5 consecutive epochs
- Deterministic train/validation `mu_x`: complete; 391,280/48,910 rows, 128 dimensions
- StandardScaler/PCA64/GMM fit: train only; validation evaluation only; test unused

| Class | TrafficFormer | Open-Detect | Case |
|---|---|---|---|
| FTP | Multi | Multi | Case A |
| Cridex | Multi | Multi | Case A |
| Miuref | Multi | Multi | Case A |
| Outlook | Multi | Multi | Case A |

Overall Gate: **A. CROSS-ENCODER PERSISTENCE**

Detailed NLL, DeltaNLL, seed stability, component degeneracy and covariance diagnostics are in `outputs/latent_gaussian_audit/`.

Known limitations: epoch-5 GPU migration lacked optimizer/RNG state; batch 1024 caused log-variance overflow and formal training used batch 512; released Open-Detect implementation differs from the paper. These results do not imply an Unknown Detection outcome.

## Cross-Encoder Component Correspondence Audit

- Status: COMPLETE
- Main protocol: K=2, full covariance, PCA64, seed=0; train fit and validation-primary correspondence
- Exact flow-id assertions: 8/8 class-split sets equal

| Class | NMI | ARI | Hungarian | OD strongest continuous | Case |
|---|---:|---:|---:|---|---|
| FTP | 0.2168 | 0.2623 | 0.7561 | total_bytes (0.2511) | Case 4 |
| Cridex | 0.9876 | 0.9951 | 0.9988 | backward_packet_count (0.8952) | Case 4 |
| Miuref | 0.4722 | 0.4316 | 0.8286 | packet_count (0.9044) | Case 4 |
| Outlook | 0.2245 | 0.3952 | 0.9469 | total_bytes (0.3109) | Case 3 |

Overall Gate: **C. SIMPLE-STATISTICS PERSISTENCE**

Interpretation boundary: these results concern persistent representation-level local structure and simple-statistics associations. They do not establish semantic modes and do not contain an Unknown Detection experiment.

## K=3 Correspondence Sensitivity

This is an appendix-only robustness check. K=2 remains the fixed main analysis and the main Gate is not reselected.

| class | NMI | AMI | ARI | Hungarian accuracy | label |
|---|---:|---:|---:|---:|---|
| FTP | 0.336017 | 0.335871 | 0.329931 | 0.715431 | MODERATE_CORRESPONDENCE |
| Cridex | 0.790774 | 0.790468 | 0.777831 | 0.832112 | MODERATE_CORRESPONDENCE |
| Miuref | 0.668105 | 0.667576 | 0.641185 | 0.811573 | MODERATE_CORRESPONDENCE |
| Outlook | 0.220940 | 0.213484 | 0.338466 | 0.895086 | MODERATE_CORRESPONDENCE |

K=3 does not establish semantic modes and is not used to alter the K=2 SIMPLE-STATISTICS PERSISTENCE Gate.
