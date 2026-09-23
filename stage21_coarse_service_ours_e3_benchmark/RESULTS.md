# Experiment results: stage21-coarse-service-ours-e3-t8-20260922-v1

- Status: `success / CLOSED_SET_DIAGNOSTIC_COMPLETE`
- Experiment type: `benchmark`
- Claim scope: `diagnostic`
- Created (UTC): `2026-09-22T02:07:27Z`
- Objective: train and test the independent project E3 fusion method on the frozen Stage20 VPN-6 and TOR-7 coarse-Service closed-set tasks.

## Method identity

`OURS-E3-T8` is the recovered TrafficFormer branch (first 5 packets, strict random initialization) plus the project FIG/TAGCN branch (first 8 packets), Known-Train-only z-score per branch, 768+128 concatenation, and a linear fusion classifier. It does not use OpenDetectNet, Open-Detect prototype loss, or the Open-Detect classifier.

This is a transparent Stage20 input adaptation. It must not be described as the Stage17 30-packet E3 implementation or as an author-equivalent external method.

## Data and split

| Dataset | Services | Total | Train | Validation | Test |
|---|---:|---:|---:|---:|---:|
| ISCX-VPN | 6 | 10,955 | 8,764 | 1,098 | 1,093 |
| ISCXTor2016 | 7 | 11,181 | 8,946 | 1,118 | 1,117 |

- Membership is exactly the frozen Stage20 `closed_service_manifest.csv`.
- VPN cache reused all 10,955 Stage19 packet views.
- TOR reused 11,044 Stage19 packet views and recovered the remaining 137 flows from the exact Stage12 packet references; membership missing=`0`.
- All IP addresses are zeroed. Every cached flow has finite graph values and at least one packet.

## Execution

- User-final scope: seeds `2022, 2023`, four formal runs.
- Training used two independent one-GPU processes, not DataParallel: physical GPU 1 for seed2022 and GPU 2 for seed2023.
- The initially launched seed2024 process on GPU4 was stopped on user request and excluded. Its log and `runs/iscx_vpn/seed2024/INTERRUPTED.md` are retained.
- TrafficFormer: 3 epochs, batch 16, LR 6e-5, random initialization.
- TAGCN: 50 epochs, batch 64, LR 1e-3, hidden 128, K=2.
- Fusion: 30 epochs, batch 256, LR 1e-3.
- Every checkpoint was selected only by Known Validation Macro-F1; Test samples used for checkpoint selection=`0`.
- Per-run wall time was 367.18--378.62 s; peak allocated GPU memory was 8.802 GiB per process.

## Primary E3 results

Mean ± population standard deviation over two seeds:

| Dataset | Accuracy | Balanced Acc. | Macro P | Macro R | Macro-F1 | Weighted-F1 | MCC | Kappa |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ISCX-VPN | 0.5201±0.0389 | 0.5328±0.0447 | 0.5613±0.0395 | 0.5328±0.0447 | 0.5249±0.0479 | 0.5129±0.0451 | 0.4294±0.0466 | 0.4196±0.0475 |
| ISCXTor2016 | 0.6222±0.0134 | 0.5881±0.0165 | 0.6075±0.0058 | 0.5881±0.0165 | 0.5911±0.0146 | 0.6176±0.0127 | 0.5460±0.0162 | 0.5449±0.0166 |

Per-seed evidence:

| Dataset | Seed | Accuracy | Macro-F1 | Weighted-F1 | Best fusion epoch |
|---|---:|---:|---:|---:|---:|
| ISCX-VPN | 2022 | 0.5590 | 0.5729 | 0.5580 | 29 |
| ISCX-VPN | 2023 | 0.4812 | 0.4770 | 0.4678 | 24 |
| ISCXTor2016 | 2022 | 0.6356 | 0.6057 | 0.6303 | 23 |
| ISCXTor2016 | 2023 | 0.6088 | 0.5766 | 0.6049 | 28 |

VPN has materially larger seed sensitivity: Macro-F1 standard deviation is 0.0479 versus 0.0146 on TOR. With only two seeds these are stability descriptors, not confidence intervals.

## Branch ablation

| Dataset | Branch | Accuracy | Macro-F1 | Worst-class F1 | MCC |
|---|---|---:|---:|---:|---:|
| ISCX-VPN | E1 TrafficFormer | 0.2640 | 0.1266 | 0.0000 | 0.1259 |
| ISCX-VPN | E2 TAGCN | 0.4346 | 0.4101 | 0.2981 | 0.3243 |
| ISCX-VPN | E3 fusion | 0.5201 | 0.5249 | 0.3917 | 0.4294 |
| ISCXTor2016 | E1 TrafficFormer | 0.1786 | 0.0433 | 0.0000 | 0.0000 |
| ISCXTor2016 | E2 TAGCN | 0.4526 | 0.3784 | 0.0290 | 0.3696 |
| ISCXTor2016 | E3 fusion | 0.6222 | 0.5911 | 0.1747 | 0.5460 |

E3 beats E2 on both seeds and both datasets. Mean E3−E2 Macro-F1 is `+0.1149` on VPN and `+0.2128` on TOR. The fusion gain is real within this experiment, but E1 itself is severely underfit/collapsed, especially on TOR, so the result does not establish that TrafficFormer alone is strong.

## Per-class E3 results

Mean over two seeds:

| Dataset | Service | Precision | Recall | F1 | Test support/seed |
|---|---|---:|---:|---:|---:|
| ISCX-VPN | Chat | 0.5908 | 0.4795 | 0.5293 | 195 |
| ISCX-VPN | Email | 0.5527 | 0.3725 | 0.4435 | 200 |
| ISCX-VPN | File-Transfer | 0.6024 | 0.2925 | 0.3917 | 200 |
| ISCX-VPN | P2P | 0.6399 | 0.6700 | 0.6534 | 100 |
| ISCX-VPN | Streaming | 0.6261 | 0.7449 | 0.6747 | 198 |
| ISCX-VPN | VoIP | 0.3561 | 0.6375 | 0.4570 | 200 |
| ISCXTor2016 | Browsing | 0.6132 | 0.6375 | 0.6251 | 200 |
| ISCXTor2016 | Chat | 0.2995 | 0.1250 | 0.1747 | 64 |
| ISCXTor2016 | Email | 0.7956 | 0.7593 | 0.7770 | 54 |
| ISCXTor2016 | File-Transfer | 0.8022 | 0.7875 | 0.7946 | 200 |
| ISCXTor2016 | P2P | 0.7653 | 0.7100 | 0.7357 | 200 |
| ISCXTor2016 | Streaming | 0.4707 | 0.4724 | 0.4714 | 199 |
| ISCXTor2016 | VoIP | 0.5062 | 0.6250 | 0.5593 | 200 |

The main failure modes are VPN File-Transfer recall (`0.2925`) and TOR Chat recall (`0.1250`). Consequently, TOR's good overall Accuracy (`0.6222`) coexists with a weak worst-class F1 (`0.1747`). The full count and row-normalized confusion matrices are retained in `multiview_confusion_matrices.csv`.

## Validation-to-test behavior

- E3 mean Test−Validation Macro-F1: VPN `+0.0090`; TOR `−0.0270`.
- The gaps are modest compared with the model's absolute error, but TOR shows a consistent validation-to-test decrease on both seeds.
- Detailed branch-level gaps are in `multiview_validation_test_gap.csv`.

## Preserved evidence

- Aggregate: `aggregate_summary.json`, `all_run_results.csv`, `summary_results.csv`.
- Multi-view: `multiview_summary.json`, `multiview_run_metrics.csv`, `multiview_aggregate_metrics.csv`, `multiview_per_class_metrics.csv`, `multiview_per_class_aggregate.csv`, `multiview_confusion_matrices.csv`, `multiview_validation_test_gap.csv`, `multiview_e3_deltas.csv`.
- Per run: three checkpoints, training history, predictions, E3 embeddings, metrics and run manifest.
- Verification: `completion_verification.json` is `PASS` with 98 checks and no failures.
- Interrupted seed2024 evidence is preserved but excluded from every formal aggregate.

## Limitations

- Only two seeds were requested; variance estimation is weak.
- This is closed-set classification only. No AUROC, AUPRC, UFAR, Known-FRR or unknown rejection claim is made.
- Service labels inherit Stage20's `WEAK_CAPTURE_LABEL` boundary; splits are not uniformly capture-disjoint and VPN P2P has only one capture.
- The FIG branch is limited to 8 packets rather than Stage17's 30 packets.
- Strict random-init TrafficFormer training for three epochs underfits badly; stronger pretraining or Known-Validation-only tuning would change the method configuration and must be reported as a separate experiment.
- Historical TrafficFormer and DQ-3F/Open-Detect scores use different populations/protocols and are context only, not direct paired comparisons.

## Conclusion

Within the frozen Stage20 closed-set tasks, E3 fusion is consistently better than either recovered branch alone, but absolute performance remains moderate: Macro-F1 `0.5249` on VPN and `0.5911` on TOR. The multi-dimensional evidence does not support calling the present configuration competitive with the earlier high closed-set baselines. The most actionable next step is Known-Validation-only optimization of the underfit TrafficFormer branch and a T8/T16/T30 graph-window ablation, while keeping the current four runs as the untouched baseline.
