# Experiment results: stage42v-cic-ddos-fourth-candidate-20260928

- Status: `success` (independent metric replay PASS, 2026-09-28 07:24 UTC)
- Experiment type: `open-set-evaluation`
- Claim scope: `diagnostic`
- Created (UTC): `2026-09-28T07:09:41Z`
- Objective: Evaluate all matched Friday DDoS flows as Unknown with frozen Stage40 Known-only encoder/support/calibration

## Data and split

- Known classes `BENIGN + PortScan` and their exact Stage40 Train/Validation/Test roles are unchanged. Unknown candidate is all 76,613 `MATCHED` `DDoS` flows in the Friday mapped-flow table, from one PCAP and five five-minute groups. No downsampling or score-based filtering.
- Candidate manifest SHA256: `9300363ccdfc786e7524705446455b2a2ac117deced5fc94367c37f5a6f2359a`; frozen before opening DDoS packet features, with no Stage40 role-ID overlap.
- This is the largest untested matched Friday attack class sharing the Friday PCAP with Known `PortScan`; selection used label/count/source metadata, not DDoS model scores.

## Configuration and execution

- Reuse frozen Stage40 three-view encoder, Known-Train support, and Known-Validation P95 thresholds. The new packet-cache adapter changes the source day from Wednesday to Friday without changing packet or feature formulas. Stage42-S frozen scoring/replay code is hash-pinned in `candidate_protocol.json`.
- Methods: MSP, Energy, centroid, DES-v1; report full natural-prevalence Unknown Test and deterministic score-blind 1:1 Test view. No training, Unknown fitting, or Test threshold tuning. `stage42v-cache-20260928` builds the full packet cache; `stage42v-finish-queue-20260928` waits for cache audit, selects a live GPU, then evaluates and independently replays metrics.

## Core results

- Known Test: 2,272 flows; all matched DDoS Unknown Test: 76,613 flows. Unknown prevalence in the natural Test view is 97.12%. Thresholds are unchanged Stage40 Known-Validation P95 values. AUROC, natural AUPRC, UFAR and Known FRR are from the full view; balanced AUPRC is from the predeclared score-blind 1:1 view.

| Method | AUROC | Natural AUPRC | Balanced AUPRC | UFAR | Known FRR |
|---|---:|---:|---:|---:|---:|
| MSP | 0.058170 | 0.905376 | 0.312868 | 0.999595 | 0.049296 |
| Energy | 0.056350 | 0.904987 | 0.312665 | 0.999961 | 0.048856 |
| Centroid | 0.635802 | 0.981756 | 0.660620 | 0.662812 | 0.054577 |
| DES-v1 | **0.948990** | **0.994485** | **0.882042** | **0.166134** | 0.071303 |

- Balanced 1:1 DES-v1 AUROC/AUPRC/UFAR = `0.948690/0.882042/0.169014`. In the full view, DES-v1 accepts 12,728/76,613 Unknown DDoS flows as Known and rejects 162/2,272 Known Test flows. Exact per-sample scores and decisions are in `unknown_ddos/detection/sample_scores.csv`.
- Cache extraction emitted 76,613/76,613 target flows after scanning 9,628,920 Friday packets. Frozen model hashes were unchanged. Independent replay verified all eight method-by-view metric rows and every saved threshold decision; candidate manifest unchanged, Unknown-fit = 0, Test-fit = 0.
- Under the earlier favorable-candidate screen (`AUROC >= 0.95`, `AUPRC >= 0.80`, `UFAR <= 0.10`, `Known FRR <= 0.10`), DDoS **does not pass**: DES-v1 AUROC is slightly below 0.95 and UFAR exceeds 0.10. The very high natural AUPRC is prevalence-sensitive and does not rescue this operating-point failure.

## Preserved evidence

- Frozen candidate protocol and manifest, Friday packet cache, exact score/metric CSVs, evaluation audit, independent verification, scripts, `manifest.json`, and project-local tmux logs are retained. The artifact inventory and hashes are refreshed after this update.

## Limitations

- This fourth candidate follows three previously viewed favorable settings, so it is explicitly post-hoc development diagnosis, not independent external validation. Sharing Friday PCAP with Known PortScan does not remove time-window, endpoint, or attack-schedule confounding. Natural AUPRC must be read alongside a balanced view.

## Conclusion and next step

- This fourth CIC setting is a completed **diagnostic** run, not a successful favorable-screen result. It shows that the frozen detector does not reject DDoS as reliably as the prior Hulk candidate at the same Known-Val threshold. Do not relabel or tune the threshold based on this Test result, and do not promote the three favorable candidates as unbiased external validation. No further experiment is launched by this result.
