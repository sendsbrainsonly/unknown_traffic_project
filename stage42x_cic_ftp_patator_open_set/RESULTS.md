# Experiment results: stage42x-cic-ftp-patator-sixth-candidate-20260928

- Status: `success` (independent metric replay PASS, 2026-09-28 08:24 UTC)
- Experiment type: `open-set-evaluation`
- Claim scope: `diagnostic`
- Created (UTC): `2026-09-28T08:20:21Z`
- Objective: Evaluate all matched Tuesday FTP-Patator flows as Unknown with frozen Stage40 Known-only encoder/support/calibration

## Data and split

- Known `BENIGN + PortScan` Train/Validation/Test roles are reused exactly from Stage40. Unknown candidate is all 3,985 matched Tuesday `FTP-Patator` flows from one PCAP and 15 five-minute groups; no downsampling or score filtering.
- Candidate manifest SHA256: `f031cbdf6ce8721854ca4f4c55834f60fd703ae664ca9d16b4b0a0c60c9be1b7`. It was frozen before FTP-Patator packet-feature access and has no Stage40 role-ID overlap.

## Configuration and execution

- Reuse frozen Stage40 three-view encoder, Known-Train empirical support, and Known-Validation P95 thresholds. Stage42-S scoring and independent replay are hash-pinned; the isolated packet adapter changes only the source day to Tuesday, not the feature formulas. Methods: MSP, Energy, centroid, DES-v1. Full natural Unknown population and score-blind 1:1 Test view. No training, Unknown fitting, or Test threshold tuning.
- `stage42x-cache-20260928` builds the packet views. `stage42x-finish-queue-20260928` waits for cache audit, then live-selects one GPU for frozen inference and runs independent replay.

## Core results

- Known Test: 2,272 flows; all matched FTP-Patator Unknown Test: 3,985 flows. Natural Unknown prevalence is 63.69%. Thresholds are unchanged Stage40 Known-Validation P95 values. The 1:1 view contains 2,272 Known and 2,272 score-blind selected Unknown flows.

| Method | Natural AUROC | Natural AUPRC | 1:1 AUPRC | UFAR | Known FRR |
|---|---:|---:|---:|---:|---:|
| MSP | 0.987460 | 0.975271 | 0.960392 | 0.005270 | 0.049296 |
| Energy | 0.984026 | 0.971663 | 0.955900 | 0.015056 | 0.048856 |
| Centroid | 0.988482 | 0.978711 | 0.965204 | 0.003513 | 0.054577 |
| DES-v1 | **0.998342** | **0.998596** | **0.997596** | **0.000000** | 0.071303 |

- DES-v1 accepts **0/3,985** FTP-Patator Unknown flows as Known and rejects **162/2,272** Known Test flows. Balanced 1:1 DES-v1 AUROC/AUPRC/UFAR = `0.998381/0.997596/0.000000`.
- Cache extraction emitted 3,985/3,985 target flows after scanning 6,913,051 Tuesday packets. Frozen checkpoint hashes were unchanged. Independent replay verified all eight method-by-view metric rows and per-sample decisions; candidate manifest unchanged, Unknown-fit = 0, Test-fit = 0.
- Under the earlier favorable-candidate screen (`AUROC >= 0.95`, `AUPRC >= 0.80`, `UFAR <= 0.10`, `Known FRR <= 0.10`), DES-v1 **passes**. However, MSP, Energy and Centroid also separate this Unknown class unusually well: their AUROC is 0.984–0.988 and UFAR is 0.35–1.51%. This weakens any claim that the high FTP-Patator score uniquely validates DES-v1; attack schedule, capture or endpoint shortcuts are plausible alternatives, not yet proven causes.

## Preserved evidence

- Frozen candidate protocol/manifest, Tuesday packet cache, exact score/metric CSVs, independent verification, scripts, `manifest.json`, and project-local tmux logs are retained. The artifact inventory and hashes are refreshed after this update.

## Limitations

- This is the sixth post-hoc candidate after viewing previous results; it is development diagnosis, not untouched validation. The Tuesday attack schedule, capture and endpoint differences may confound classification. Interpret AUPRC alongside Unknown prevalence.

## Conclusion and next step

- This sixth CIC setting is a completed **diagnostic** run with a passing favorable screen. Because nearly every detector is excellent on the same data, it must not be treated as isolated evidence of a DES-specific mechanism or untouched external validation. Do not adjust thresholds from these Test scores or launch another candidate automatically.
