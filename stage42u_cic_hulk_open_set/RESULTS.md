# Experiment results: stage42u-cic-hulk-third-candidate-20260928

- Status: `success` (independent metric replay PASS, 2026-09-28 03:54 UTC)
- Experiment type: `open-set-evaluation`
- Claim scope: `diagnostic`
- Created (UTC): `2026-09-28T02:26:47Z`
- Objective: Evaluate frozen BENIGN+PortScan three-view encoder with all matched DoS Hulk flows as Unknown using Known-Val-only calibration

## Data and split

- Frozen Stage40 Known classes: `BENIGN`, `PortScan`, with the exact prior Known Train/Validation/Test roles. Unknown candidate: all 155,168 `MATCHED` `DoS Hulk` flows in the Wednesday mapped-flow table; six five-minute groups from one source PCAP. No downsampling or score-based selection.
- Candidate manifest SHA256: `b158b2bb751fa5fd61c18f34149411583b8ba3515d4ad0dc5d9d822a573cb37f`. It was frozen before Hulk packet-feature access and has no Stage40 role-ID overlap.

## Configuration and execution

- Reuse the frozen Stage40 three-view encoder, Known-Train empirical support, and Known-Validation P95 thresholds. Stage42-S extraction, scoring, and independent metric replay are reused through a path adapter whose SHA256 is frozen in `candidate_protocol.json`; bytecode writes to older bundles are disabled.
- Methods: MSP, Energy, centroid, DES-v1; report the complete 155,168-flow Unknown Test population and a deterministic score-blind 1:1 view. No encoder training or Unknown fitting. Packet-cache session: `stage42u-cache-20260928`. `stage42u-finish-queue-20260928` waits for its cache audit, then selects a live GPU for frozen inference and runs independent metric replay.

## Core results

Known Test: 2,272 flows; full Hulk Unknown Test: 155,168 flows. The frozen encoder and support were reused; thresholds came from Known Validation P95. AUROC and UFAR below are from the full natural-prevalence evaluation; balanced AUPRC uses the predeclared score-blind 1:1 Test view.

| Method | AUROC | Natural AUPRC | Balanced AUPRC | UFAR | Known FRR |
|---|---:|---:|---:|---:|---:|
| MSP | 0.093804 | 0.959585 | 0.327381 | 0.981021 | 0.049296 |
| Energy | 0.088121 | 0.957379 | 0.322790 | 0.987472 | 0.048856 |
| Centroid | 0.658286 | 0.992366 | 0.693603 | 0.588994 | 0.054577 |
| DES-v1 | **0.973767** | **0.998806** | **0.928132** | **0.001134** | 0.071303 |

The balanced 1:1 AUROC for DES-v1 is 0.973580 and its UFAR is 0.001761. All eight method × view metric rows independently replayed PASS. The frozen candidate manifest was unchanged; Unknown-fit and Test-fit counts were both zero. See `unknown_hulk/detection/open_set_results.csv`, `sample_scores.csv`, and `independent_verification.json` for exact values and sample-level evidence.

## Preserved evidence

- Frozen candidate protocol and manifest, extraction cache, exact scores/metrics, independent verification, scripts, progress, `manifest.json`, and project-local tmux logs are retained. The artifact inventory and hashes are refreshed after this update.

## Limitations

- Hulk is large but still originates from a single Wednesday PCAP and only six five-minute groups. High AUPRC under natural prevalence may be driven by the class ratio; the balanced view is required for interpretation.

## Conclusion and next step

- This third CIC setting is a successful **diagnostic** result for Hulk rejection under the frozen `BENIGN + PortScan` Known model. It is not untouched external validation or evidence that post-hoc favorable-setting selection is unbiased. No next experiment is launched by this result.
