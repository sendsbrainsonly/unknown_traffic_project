# Experiment results: stage42s-cic-slowloris-first-candidate-20260928

- Status: `success`
- Experiment type: `open-set-evaluation`
- Claim scope: `diagnostic`
- Created (UTC): `2026-09-28T01:47:14Z`
- Objective: Evaluate frozen BENIGN+PortScan three-view encoder with DoS slowloris as Unknown using Known-Val-only calibration

## Data and split

- Fixed Known classes: `BENIGN`, `PortScan`; exact Stage40 Known Train/Validation/Test and frozen model are reused.
- Candidate Unknown class: `DoS slowloris`; all 5,709 flows whose official mapping status begins with `MATCHED` are frozen as Unknown Test. They span eight five-minute groups in the Wednesday PCAP.
- The candidate manifest SHA256 is `908208f25c3e88a2b4153e538090ec222e24b8d21ce91cc6a08f447df4eca3f4`. It was frozen before candidate packet-feature access. Unknown fitting and threshold-calibration counts are zero.

## Configuration and execution

- Reuse the completed Stage40 `BENIGN + PortScan` three-view encoder, Known-Train empirical support, and Known-Validation P95 thresholds without retraining or recalibration.
- Packet views use the unchanged Stage34 TrafficFormer (first 5 packets), FIG (first 30 packets), and YaTC (first 5 packets) formulas. Only the frozen slowloris Unknown Test flows are newly extracted.
- Methods: MSP, Energy, empirical centroid, DES-v1. Report both all 5,709 Unknown flows and a deterministic score-blind 1:1 view with 2,272 Unknown and 2,272 Known Test flows.
- Closed-set Known Test performance copied from the frozen Stage40 result: Accuracy/Macro-F1/Weighted-F1=`0.993838/0.993838/0.993838`.
- Candidate Unknown packet cache built from the Wednesday PCAP: 5,709/5,709 flows emitted; builder audit PASS.
- Frozen inference used physical GPU 0. Stage40 checkpoint SHA256 values matched before and after inference. Independent verification replayed all eight method-by-view metric rows; status PASS; Unknown/Test fitting counts both zero.
- The first inference launch stopped on a Python module-name collision before inference. That failure log is preserved; the isolated import was renamed and the resumed inference plus replay succeeded. No Stage40 files or checkpoints were changed.

## Core results

All metrics use the Stage40 Known-Val P95 thresholds without recalibration. `UFAR` is the fraction of Unknown flows accepted as Known; `Known FRR` is the fraction of Known Test flows rejected.

### Natural Test prevalence (2,272 Known + 5,709 Unknown; Unknown prevalence 71.53%)

| Method | AUROC | AUPRC | UFAR | Known FRR | Binary F1 |
|---|---:|---:|---:|---:|---:|
| MSP | 0.814714 | 0.931797 | 0.307935 | 0.049296 | 0.808637 |
| Energy | 0.803361 | 0.926911 | 0.329305 | 0.048856 | 0.793657 |
| Centroid | 0.800734 | 0.929836 | 0.294272 | 0.054577 | 0.817076 |
| DES-v1 | **0.993050** | **0.996724** | **0.000350** | 0.071303 | **0.985835** |

### Score-blind balanced view (2,272 Known + 2,272 Unknown)

| Method | AUROC | AUPRC | UFAR | Known FRR | Binary F1 |
|---|---:|---:|---:|---:|---:|
| MSP | 0.815112 | 0.869939 | 0.310299 | 0.049296 | 0.793217 |
| Energy | 0.803802 | 0.860882 | 0.333627 | 0.048856 | 0.777008 |
| Centroid | 0.801048 | 0.869670 | 0.294454 | 0.054577 | 0.801700 |
| DES-v1 | **0.992970** | **0.991799** | **0.000440** | 0.071303 | **0.965356** |

Under the prelisted favorable-candidate screen (`AUROC >= 0.95`, `AUPRC >= 0.80`, `UFAR <= 0.10`, `Known FRR <= 0.10`), the frozen DES-v1 score passes on both prevalence views. MSP, Energy, and centroid do not pass because their AUROC is about 0.80 and their UFAR is about 0.29–0.33. This records one favorable slowloris role; it does not establish the same result for other Unknown classes.

## Preserved evidence

- `candidate_protocol.json`, `candidate_unknown_manifest.csv`, `candidate_results.csv`, scripts, the complete sample-score file, both prevalence-view metrics, Known Test closed-set result, inference audit, independent replay, and `manifest.json`. The first failed import attempt and successful resumed run logs remain under project-local `.tmux-task/`.

## Limitations

- This is an explicitly post-hoc favorable-setting development diagnostic, not a general CIC-IDS-2017 benchmark.
- Slowloris occurs in one source PCAP and is strongly tied to endpoints/time; high performance may reflect capture artifacts. Eight five-minute groups are not independent captures.

## Conclusion and next step

- First candidate complete: slowloris is favorable for the frozen DES-v1 score under both natural and balanced views. The next prelisted Unknown candidate has not been started.
