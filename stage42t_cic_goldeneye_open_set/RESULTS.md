# Experiment results: stage42t-cic-goldeneye-second-candidate-20260928

- Status: `success`
- Experiment type: `open-set-evaluation`
- Claim scope: `diagnostic`
- Created (UTC): `2026-09-28T02:10:25Z`
- Objective: Evaluate frozen BENIGN+PortScan three-view encoder with DoS GoldenEye as Unknown using Known-Val-only calibration

## Data and split

- Known classes: `BENIGN`, `PortScan`, with the exact Stage40 Known Train/Validation/Test roles; Unknown candidate: all 7,441 `MATCHED` `DoS GoldenEye` flows from the Wednesday mapping. They occupy only two five-minute groups in one source PCAP.
- Candidate manifest SHA256: `630fd0b2ddf41b92670447f7a330023ce7585f2155b908feef9cb5521c05e0bf`. IDs were frozen before candidate packet-feature access; no overlap with Stage40 role IDs.

## Configuration and execution

- Reuse the frozen Stage40 three-view encoder, Known-Train empirical support, and Known-Validation P95 thresholds. Reuse the Stage42-S packet-view, score, and independent replay code with only output/manifest paths rebound to this separate bundle; source code SHA256 values are recorded in `candidate_protocol.json`.
- Candidate selection was based on label and flow count only. Full Unknown Test and score-blind 1:1 balanced views are planned. No new encoder training or Unknown fitting.
- Packet cache job `stage42t-cache-20260928` completed with exit 0: 10,159,565 packets scanned, 7,441/7,441 flows emitted, both feature-cache audits PASS.
- Frozen inference used physical GPU 0 (selected from a live 16 GB free-memory floor) and completed with exit 0. Stage40 checkpoint hashes matched before and after inference. `stage42t-replay-20260928` independently recomputed all eight method-by-view metric rows and per-sample decisions; exit 0, status PASS. Unknown/Test fitting counts both remained zero.
- Frozen Known Test Accuracy/Macro-F1/Weighted-F1 = `0.993838/0.993838/0.993838`.

## Core results

All scores use the unchanged Stage40 Known-Validation P95 thresholds. Unknown is the positive class. `UFAR` is the fraction of Unknown accepted as Known; `Known FRR` is the fraction of Known Test rejected.

### Natural prevalence: 2,272 Known + 7,441 Unknown (Unknown prevalence 76.61%)

| Method | AUROC | AUPRC | UFAR | Known FRR | Binary F1 |
|---|---:|---:|---:|---:|---:|
| MSP | 0.331679 | 0.706984 | 0.910630 | 0.049296 | 0.161840 |
| Energy | 0.295777 | 0.667694 | 0.958742 | 0.048856 | 0.078127 |
| Centroid | 0.469988 | 0.779624 | 0.812525 | 0.054577 | 0.311384 |
| DES-v1 | **0.969958** | **0.977195** | **0.002419** | 0.071303 | **0.988021** |

### Score-blind balanced view: 2,272 Known + 2,272 Unknown

| Method | AUROC | AUPRC | UFAR | Known FRR | Binary F1 |
|---|---:|---:|---:|---:|---:|
| MSP | 0.330420 | 0.441410 | 0.906250 | 0.049296 | 0.164035 |
| Energy | 0.294427 | 0.397574 | 0.958627 | 0.048856 | 0.075898 |
| Centroid | 0.470550 | 0.538696 | 0.808539 | 0.054577 | 0.307312 |
| DES-v1 | **0.970014** | **0.930666** | **0.002201** | 0.071303 | **0.964476** |

Under the prelisted favorable-candidate screen (`AUROC >= 0.95`, `AUPRC >= 0.80`, `UFAR <= 0.10`, `Known FRR <= 0.10`), DES-v1 passes on both views; MSP, Energy, and centroid fail. In the natural view DES-v1 accepts 18/7,441 Unknown flows as Known and rejects 162/2,272 Known Test flows. Relative to the first slowloris candidate, DES-v1 natural AUROC is lower by 0.023091 but remains above the screen. AUPRC should be read with the Unknown prevalence shown above.

## Preserved evidence

- `candidate_protocol.json`, `candidate_unknown_manifest.csv`, `candidate_results.csv`, scripts, full per-sample `unknown_goldeneye/detection/sample_scores.csv`, both-view metrics, frozen Known Test metric copy, extraction audits, inference audit, independent replay, and `manifest.json`. `verify_summary.py` independently checked all eight exported table rows against the frozen metric CSV and passed. Execution logs and exit statuses are under project-local `.tmux-task/`.
- During code reuse, Python regenerated two Stage42-S `__pycache__` files. Their previous/current hashes and the unchanged source hashes are recorded in `stage42s_cache_drift_audit.md`; the runner now disables bytecode writes before loading those modules. No Stage42-S protocol, model, or scored result file was changed.

## Limitations

- The Unknown class comes from a single Wednesday PCAP and just two five-minute groups. A favorable score could reflect capture/time/endpoint shortcuts rather than broad attack-family separation.

## Conclusion and next step

- The second candidate is complete and satisfies the stated DES-v1 diagnostic screen. It supplies another favorable CIC role under the same frozen Known model, with severe single-capture/two-group limitations. No later candidate has been started.
