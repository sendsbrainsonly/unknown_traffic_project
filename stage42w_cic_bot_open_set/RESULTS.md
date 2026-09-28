# Experiment results: stage42w-cic-bot-fifth-candidate-20260928

- Status: `success` (independent metric replay PASS, 2026-09-28 07:59 UTC)
- Experiment type: `open-set-evaluation`
- Claim scope: `diagnostic`
- Created (UTC): `2026-09-28T07:53:32Z`
- Objective: Evaluate all matched Friday Bot flows as Unknown with frozen Stage40 Known-only encoder/support/calibration

## Data and split

- Known `BENIGN + PortScan` Train/Validation/Test roles are reused exactly from Stage40. Unknown candidate is every matched Friday `Bot` flow: 1,228 flow IDs in one PCAP and 37 five-minute groups, without downsampling or score filtering.
- Candidate manifest SHA256: `b304556ae0de5043ef58cb1207f234bbb7fe08c7d784c51af51b611695aba24b`. The exact 1,228 score-blind Known Test IDs for the 1:1 view were separately frozen before opening Bot packet features: `balanced_known_manifest.csv`, SHA256 `6f00684e6d5565c10ce04d9e5c8727c7b49fbefc57bd2d0f635602d715e1124d`.

## Configuration and execution

- Reuse the frozen Stage40 three-view encoder, Known-Train support, Known-Validation P95 thresholds, and hash-pinned Stage42-S scoring/replay. A path adapter uses the Friday packet source without changing feature formulas. Four methods: MSP, Energy, centroid, DES-v1. No encoder training, Unknown fitting, or Test threshold tuning.
- Implementation audit: the earlier evaluator retains all 2,272 Known Test rows when the Unknown pool is smaller, so its nominal 1:1 view would be unequal here. The isolated `fix_balanced_view.py` corrects only score-blind Known membership and recomputes balanced-view metrics after frozen inference; full natural scores, thresholds, and model remain untouched. Its code SHA256 is frozen in `balanced_view_implementation_audit.json` before Bot packet-feature access. Independent replay must validate the corrected 1,228:1,228 view.

## Core results

- Known Test: 2,272 flows; all matched Bot Unknown Test: 1,228 flows. The natural Test prevalence is 35.09% Unknown. The pre-frozen 1:1 view contains exactly 1,228 Known and 1,228 Unknown flows. All thresholds remain Stage40 Known-Validation P95 values.

| Method | Natural AUROC | Natural AUPRC | 1:1 AUROC | 1:1 AUPRC | UFAR | Natural Known FRR |
|---|---:|---:|---:|---:|---:|---:|
| MSP | 0.629074 | 0.621579 | 0.631343 | 0.717281 | 0.596906 | 0.049296 |
| Energy | 0.627147 | 0.619656 | 0.629135 | 0.714958 | 0.600163 | 0.048856 |
| Centroid | 0.570930 | 0.618974 | 0.575418 | 0.709223 | 0.561889 | 0.054577 |
| DES-v1 | **0.946433** | **0.875758** | **0.950623** | **0.936257** | 0.508958 | 0.071303 |

- DES-v1 accepts **625/1,228** Bot Unknown flows as Known and rejects **162/2,272** Known Test flows in the natural view. The 1:1 Known FRR is 0.062704; the difference is due solely to the score-blind Known subset, not a different threshold.
- Friday packet extraction emitted 1,228/1,228 flows after scanning 5,819,709 packets. Frozen model/checkpoint hashes were unchanged. After the isolated 1:1 correction, independent replay verified all eight method-by-view metric rows and every saved decision. Candidate and balanced-Known manifest hashes were unchanged; Unknown-fit = 0 and Test-fit = 0.
- Under the earlier favorable-candidate screen (`AUROC >= 0.95`, `AUPRC >= 0.80`, `UFAR <= 0.10`, `Known FRR <= 0.10`), Bot **fails**: natural AUROC is below 0.95 and UFAR is 50.90%, despite 1:1 AUROC marginally exceeding 0.95.

## Preserved evidence

- Candidate protocol/manifest, balanced Known membership and implementation audit, Friday packet cache, exact score/metric CSVs, independent verification, scripts, `manifest.json`, and project-local tmux logs are retained. The artifact inventory and hashes are refreshed after this update.

## Limitations

- This is a fifth post-hoc candidate after viewing prior results; it is development diagnosis, not untouched validation. Friday PCAP membership does not remove time-window, endpoint, or attack-schedule confounding. AUPRC must be reported with its Unknown prevalence.

## Conclusion and next step

- This fifth CIC setting is a completed **diagnostic** run, not a successful favorable-screen result. DES-v1 has good ranking relative to MSP/Energy but misses roughly half of Bot flows at the frozen P95 operating point. No Test-driven threshold change or next candidate is launched by this result.
