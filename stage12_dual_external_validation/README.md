# Stage 12 — Dual Independent External Validation

## Goal

Evaluate whether frozen DES-v0 outperforms Open-Detect Native on two method-untouched external datasets: ISCX-VPN and ISCXTor2016.

## Why ISCX-VPN

ISCX-VPN contains application captures collected under VPN and non-VPN conditions, providing a method-untouched external domain with raw packets and filename-traceable application labels.

## Why ISCXTor2016

ISCXTor2016 contains application captures collected under Tor and non-Tor conditions, providing a second method-untouched external domain with different tunnel behavior. Both datasets are independent of the USTC development experiments and the already-opened CipherSpectrum test.

## Independent Validation Status

This directory is an independent external confirmation stage, not method development. DES-v0, the corrected Open-Detect training protocol, thresholds, metrics, seeds, and final gate are fixed before any test opening.

## Dataset Audit

`protocol/iscx_vpn/` and `protocol/iscx_tor/` retain the source inventory, parse/readability evidence, preprocessing audit, eligibility decision, and split evidence. Source datasets are read-only.

Eligibility is frozen before training from data integrity, flow count, and group count only: a class needs at least one readable source capture and at least 200 successfully selected flows (the deterministic per-class cap is 2,000). Three duplicate-linked source groups are required for the preferred source-PCAP group-aware 80/10/10 split. If an otherwise eligible class has fewer than three such groups, the required explicit fallback is `FLOW_DISJOINT_ONLY`: flow IDs and exact 1,024-byte image hashes are kept disjoint, the class is recorded as `GROUP_AWARE_NOT_FEASIBLE`, and the external claim is reduced. This fallback is never silent.

## Canonical Labels

The canonical unknown unit is the finest application or service identity that can be traced to an official capture filename and the official dataset application list. The coarser official traffic category is retained separately. VPN/non-VPN and Tor/non-Tor are metadata, never canonical unknown classes.

## Unknown-Free Protocol

Unknown classes are absent from encoder training, early stopping, checkpoint selection, prototype estimation, empirical-centroid estimation, threshold calibration, normalization, and all other fitted statistics. Unknown samples are used only after the one-shot test opening.

## Low / Medium / High Openness

Eligible canonical classes are sorted and permuted once with `numpy.random.default_rng(42)`. Low, Medium, and High use nested prefixes of that frozen permutation and the pre-registered 5%, 15%, and 25% class-count rules.

## Open-Detect Native

M0 uses the corrected Open-Detect encoder and its learned class prototypes. Its anomaly score is the minimum Gaussian KL divergence from the deterministic posterior to `N(prototype_y, I)`.

## DES-v0

M1 shares the exact M0 checkpoint and deterministic `mu_x`. It replaces only the detection readout with the minimum squared Euclidean distance to Known-Train empirical class centroids.

## Shared Encoder Fairness

Exactly one encoder is trained for each dataset, setting, and seed. M0 and M1 always share that checkpoint; no M1 encoder is trained.

## Validation-Only Threshold Calibration

Each method uses its own Known-Validation P95 threshold via `numpy.quantile(method="higher")`. Unknown and test samples are never used for threshold fitting.

## One-Shot Test Opening

Every run must first contain a complete `pretest_freeze.json` with status `READY_FOR_ONE_SHOT_TEST`. Test evaluation is prohibited until all expected runs pass this gate. The opening is then recorded once in `outputs/summary/FINAL_TEST_OPENING.json`.

## Metrics

Primary metrics are AUROC, AUPRC, UFAR, and Known FRR. Known Accuracy, Known Macro-F1, Unknown Recall, natural Binary F1, deterministic balanced Binary F1, and Balanced Accuracy are also retained.

## Paired Bootstrap

M1-minus-M0 deltas use 1,000 paired stratified bootstrap resamples with seed 0 and 95% percentile intervals on identical samples.

## Per-Unknown-Class Results

Each unknown class reports sample count, UFAR, mean and median anomaly score, and AUROC against all Known Test samples for both methods.

## Unknown Absorption

Accepted unknown samples are tabulated by true unknown class and predicted known class for both M0 and M1.

## Five-Seed Summary

Each dataset and openness setting reports mean, median, standard deviation, minimum, maximum, and the number of seeds for which DES AUROC exceeds Open-Detect AUROC.

## Cross-Dataset Results

Datasets are summarized separately before any cross-dataset decision. A pooled mean cannot override a dataset-level failure.

## Final Gate

The only possible decisions are `EXTERNAL_CONFIRMED`, `PARTIAL_CONFIRMATION`, and `NOT_CONFIRMED`, using the pre-registered six-condition gate in `configs/stage12_config.json`.

## Reproducibility

All generated artifacts, source references, hashes, commands, warnings, failures, checkpoints, thresholds, centroids, predictions, and summaries remain under this directory. Stage 6–11C and the sibling Open-Detect project are read-only.

## Limitations

Application labels inherit the controlled-capture semantics of source filenames; individual aggregate captures can contain incidental background flows. Group-aware splits prevent a source capture from crossing Known Train/Validation/Test. Any required fallback is explicit and lowers the claim.

## Frozen Boundaries

No K1/K2, GMM, DAP, DGSB, adaptive selector, new loss, new encoder, per-class threshold, PCA, scaler, or covariance model is part of Stage 12. No method may be changed after test opening.

<!-- STAGE12_RESULTS_START -->
## ISCX-VPN Results

- Low: M0 AUROC `0.534856`, DES AUROC `0.554219`, Delta `+0.019363`, DES wins `5/5` seeds.
- Medium: M0 AUROC `0.556404`, DES AUROC `0.574673`, Delta `+0.018269`, DES wins `4/5` seeds.
- High: M0 AUROC `0.567813`, DES AUROC `0.600537`, Delta `+0.032723`, DES wins `5/5` seeds.

## ISCXTor2016 Results

- Low: M0 AUROC `0.503141`, DES AUROC `0.528147`, Delta `+0.025006`, DES wins `4/5` seeds.
- Medium: M0 AUROC `0.524548`, DES AUROC `0.635209`, Delta `+0.110661`, DES wins `5/5` seeds.
- High: M0 AUROC `0.546153`, DES AUROC `0.619687`, Delta `+0.073534`, DES wins `5/5` seeds.

## Cross-Dataset Comparison

- iscx_vpn: mean Delta AUROC `+0.023452`.
- iscx_tor: mean Delta AUROC `+0.069734`.

## Known Classification

Known Accuracy and Macro-F1 are reported per method, setting, and seed in each dataset's `setting_summary.csv`.

## Unknown Detection

AUROC, AUPRC, Unknown Recall, and both natural and balanced binary metrics are preserved in the run-level and setting summaries.

## UFAR / FRR Tradeoff

UFAR and Known FRR deltas are reported jointly; the final Gate rejects improvements purchased through a registered large FRR increase.

## Failure Cases

Per-unknown degradation and absorption hubs are retained in `per_unknown_class_results.csv` and `absorption_analysis.csv` for each dataset.

## What Can Be Claimed

The registered cross-dataset claim is supported under this frozen protocol.

## What Cannot Be Claimed

The result does not establish that DES universally dominates Open-Detect or generalizes to every encrypted-traffic dataset.
<!-- STAGE12_RESULTS_END -->
