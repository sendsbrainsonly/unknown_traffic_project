# Stage 16S — Service-Level Open-Set Benchmark

## Scope

Six flow-level Leave-One-Service-Out protocols, three training seeds, one corrected Open-Detect checkpoint shared by OD-Native, DES-v0, DES-v1, and H1 in each protocol/seed. All labels are `WEAK_CAPTURE_LABEL`; P2P has one capture.

## Overall six-metric means across 18 protocol-seed runs

| Method | AUROC | AUPRC | UFAR | Known FRR | Open Macro-F1 | Open Weighted-F1 |
|---|---:|---:|---:|---:|---:|---:|
| OD-Native | 0.623635 | 0.776025 | 0.927674 | 0.035700 | 0.305398 | 0.219430 |
| DES-v0 | 0.652011 | 0.789242 | 0.927872 | 0.039963 | 0.303409 | 0.220200 |
| DES-v1 | 0.686584 | 0.806845 | 0.914446 | 0.038198 | 0.310614 | 0.235060 |
| H1 | 0.642470 | 0.787600 | 0.915263 | 0.042734 | 0.308450 | 0.235742 |

## Paired difference versus OD-Native

| Method | ΔAUROC | wins/18 | worst | ΔAUPRC | ΔUFAR | ΔKnown FRR |
|---|---:|---:|---:|---:|---:|---:|
| DES-v0 | 0.028376 | 11/18 | -0.091862 | 0.013217 | 0.000197 | 0.004263 |
| DES-v1 | 0.062949 | 13/18 | -0.080947 | 0.030820 | -0.013229 | 0.002498 |
| H1 | 0.018835 | 8/18 | -0.028563 | 0.011575 | -0.012411 | 0.007034 |

## Unknown-Service means

| Unknown | Method | AUROC | AUPRC | UFAR | Known FRR |
|---|---|---:|---:|---:|---:|
| Chat | OD-Native | 0.661339 | 0.618575 | 0.930649 | 0.037500 |
| Chat | DES-v0 | 0.675867 | 0.655227 | 0.883669 | 0.025000 |
| Chat | DES-v1 | 0.738814 | 0.688982 | 0.859060 | 0.031250 |
| Chat | H1 | 0.684214 | 0.645654 | 0.901566 | 0.029167 |
| Email | OD-Native | 0.602416 | 0.605770 | 0.920635 | 0.040340 |
| Email | DES-v0 | 0.700346 | 0.643534 | 0.966270 | 0.036093 |
| Email | DES-v1 | 0.718684 | 0.653641 | 0.948413 | 0.031847 |
| Email | H1 | 0.693882 | 0.649419 | 0.938492 | 0.044586 |
| File-Transfer | OD-Native | 0.693203 | 0.836837 | 0.914667 | 0.022676 |
| File-Transfer | DES-v0 | 0.718404 | 0.836852 | 0.916444 | 0.047619 |
| File-Transfer | DES-v1 | 0.691023 | 0.835631 | 0.872889 | 0.047619 |
| File-Transfer | H1 | 0.699710 | 0.835253 | 0.893333 | 0.043084 |
| P2P | OD-Native | 0.529694 | 0.876250 | 0.980457 | 0.039216 |
| P2P | DES-v0 | 0.597810 | 0.892683 | 0.983776 | 0.028011 |
| P2P | DES-v1 | 0.626525 | 0.905147 | 0.973451 | 0.028011 |
| P2P | H1 | 0.564715 | 0.886615 | 0.981563 | 0.030812 |
| Streaming | OD-Native | 0.699653 | 0.942699 | 0.864649 | 0.028490 |
| Streaming | DES-v0 | 0.679011 | 0.935672 | 0.854906 | 0.045584 |
| Streaming | DES-v1 | 0.674952 | 0.936397 | 0.883090 | 0.039886 |
| Streaming | H1 | 0.676290 | 0.937373 | 0.813152 | 0.051282 |
| VoIP | OD-Native | 0.555503 | 0.776020 | 0.954990 | 0.045977 |
| VoIP | DES-v0 | 0.540626 | 0.771481 | 0.962166 | 0.057471 |
| VoIP | DES-v1 | 0.669505 | 0.821274 | 0.949772 | 0.050575 |
| VoIP | H1 | 0.536010 | 0.771285 | 0.963470 | 0.057471 |

## Known-classifier quality by held-out Service

| Unknown Service | Mean Accuracy | Mean Macro-F1 | Mean Weighted-F1 | Macro-F1 by seed 2022/2023/2024 |
|---|---:|---:|---:|---|
| Chat | 0.797917 | 0.752871 | 0.783804 | 0.956907 / 0.377309 / 0.924397 |
| Email | 0.764331 | 0.698090 | 0.742383 | 0.861278 / 0.365948 / 0.867044 |
| File-Transfer | 0.798186 | 0.709573 | 0.781194 | 0.897691 / 0.340783 / 0.890246 |
| P2P | 0.756303 | 0.647973 | 0.728444 | 0.806470 / 0.339048 / 0.798401 |
| Streaming | 0.797721 | 0.694651 | 0.762221 | 0.910702 / 0.255009 / 0.918242 |
| VoIP | 0.790805 | 0.700496 | 0.773020 | 0.864767 / 0.351027 / 0.885693 |

## Service-conditional paired AUROC

| Unknown Service | Method | Mean ΔAUROC | AUROC wins/3 | Mean ΔUFAR | Mean ΔKnown FRR |
|---|---|---:|---:|---:|---:|
| Chat | DES-v0 | 0.014527 | 2/3 | -0.046980 | -0.012500 |
| Chat | DES-v1 | 0.077475 | 2/3 | -0.071588 | -0.006250 |
| Chat | H1 | 0.022875 | 2/3 | -0.029083 | -0.008333 |
| Email | DES-v0 | 0.097930 | 3/3 | 0.045635 | -0.004246 |
| Email | DES-v1 | 0.116267 | 3/3 | 0.027778 | -0.008493 |
| Email | H1 | 0.091466 | 3/3 | 0.017857 | 0.004246 |
| File-Transfer | DES-v0 | 0.025200 | 1/3 | 0.001778 | 0.024943 |
| File-Transfer | DES-v1 | -0.002180 | 1/3 | -0.041778 | 0.024943 |
| File-Transfer | H1 | 0.006506 | 1/3 | -0.021333 | 0.020408 |
| P2P | DES-v0 | 0.068116 | 3/3 | 0.003319 | -0.011204 |
| P2P | DES-v1 | 0.096831 | 3/3 | -0.007006 | -0.011204 |
| P2P | H1 | 0.035022 | 2/3 | 0.001106 | -0.008403 |
| Streaming | DES-v0 | -0.020642 | 1/3 | -0.009743 | 0.017094 |
| Streaming | DES-v1 | -0.024701 | 1/3 | 0.018441 | 0.011396 |
| Streaming | H1 | -0.023363 | 0/3 | -0.051496 | 0.022792 |
| VoIP | DES-v0 | -0.014877 | 1/3 | 0.007175 | 0.011494 |
| VoIP | DES-v1 | 0.114002 | 3/3 | -0.005219 | 0.004598 |
| VoIP | H1 | -0.019493 | 0/3 | 0.008480 | 0.011494 |

## Interpretation

- DES-v0 mean ΔAUROC=0.028376; negative service means=['Streaming', 'VoIP']; all-seed negative=[].
- DES-v1 mean ΔAUROC=0.062949; negative service means=['File-Transfer', 'Streaming']; all-seed negative=[].
- H1 mean ΔAUROC=0.018835; negative service means=['Streaming', 'VoIP']; all-seed negative=['Streaming', 'VoIP'].
- DES-v1 is the strongest ranking method (mean AUROC 0.686584, +0.062949 vs OD, 13/18 wins), but it is not uniformly better: File-Transfer and Streaming have negative mean deltas.
- H1 is not the most stable AUROC method here: +0.018835 mean, only 8/18 wins, and all three Streaming and VoIP seeds are negative.
- At Known-Val P95, UFAR remains extremely high: OD 0.927674, DES-v0 0.927872, DES-v1 0.914446, H1 0.915263. Ranking improved, but practical unknown rejection is not solved.
- Mean Known FRR remains low (0.035700--0.042734); DES-v1/H1 improve UFAR slightly at a +0.002498/+0.007034 FRR cost versus OD.
- Seed 2023 is a systematic optimization-collapse regime in all six protocols (Known Test Macro-F1 0.255009--0.377309); it is retained in every mean and must not be interpreted as an independent dataset replicate.
- AUROC and P95 operating-point behavior must be read together; UFAR and Known FRR are reported without Test tuning.
- Service-level results are not comparable as algorithm deltas to historical Fine-Application metrics.
- The task is useful as a development diagnostic, but weak labels, shared captures, and P2P's single capture prevent a dataset-generalization claim.

## Required conclusions

1. Historical method sources: corrected OD in `../Open-Detect/reproduction/corrected_model.py`; DES-v0/v1 primitives in `../stage14d_vnat_frozen_open_set_evaluation/scripts/stage14d_common.py`; H1 percentile-max in `../stage15b_known_only_hybrid_detector/scripts/run_stage15b.py`. Source hashes are preserved in `artifact_hashes.csv` and protected-input hashes.
2. Open-Detect was reused, not reimplemented: each LOSO run imports the existing corrected model and reproduction training loop.
3. Strict Service-level Unknown-Free LOSO passed: Unknown fitting/calibration/support/selection counts are all zero.
4. Chat, Email, File-Transfer, P2P, Streaming, and VoIP all completed three-seed evaluation; P2P is protocol-limited.
5. P2P has one capture, so its result is same-capture flow-level diagnosis only, not cross-capture generalization.
6. Known classification is strong for seeds 2022/2024 and collapses for seed 2023; the table above reports every run.
7. OD-Native overall AUROC/AUPRC/UFAR/Known-FRR are 0.623635/0.776025/0.927674/0.035700.
8. DES-v0 improves mean AUROC by 0.028376 (11/18 wins), but has essentially no aggregate UFAR gain (+0.000197) and is negative on Streaming and VoIP means.
9. DES-v1 improves mean AUROC by 0.062949 (13/18 wins) and UFAR by 0.013229 absolute, but regresses on File-Transfer and Streaming means.
10. H1 improves mean AUROC by 0.018835, but only wins 8/18 and is consistently negative for Streaming and VoIP; it is not a stable global winner.
11. Negative-gain Services and per-seed win counts are reported in the Service-conditional table; no run was removed.
12. P95 UFAR improves slightly for DES-v1/H1, not DES-v0, but remains above 0.91 overall.
13. Known FRR changes are small in absolute terms, with the largest mean cost for H1 (+0.007034).
14. Yes: AUROC ranking gains coexist with unacceptable false acceptance, especially P2P (UFAR 0.973--0.984 across DES methods).
15. Coarse Service is useful as a controlled development task, but not yet a defensible sole main task because labels are weak and capture-disjoint generalization is unavailable.
16. Current evidence supports a Service-conditional DES-v1 ranking benefit inside this 3,065-flow pool. It does not support universal superiority or external/cross-capture generalization; independent Service data are required.

## Integrity

- Strict Unknown-Free checks: PASS for 18/18 runs.
- Shared encoder/checkpoint within each four-method comparison: PASS.
- Unknown/Test calibration samples: 0/0.
- Test threshold tuning: NOT_RUN.
- DES/H1 formulas and historical assets: unchanged.
