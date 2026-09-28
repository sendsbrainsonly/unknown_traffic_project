# Experiment results: stage43-cic-mixed-prevalence-20260928

- Status: `success / independent replay PASS`
- Experiment type: `open-set-evaluation`
- Claim scope: `diagnostic`
- Created (UTC): `2026-09-28T09:32:38Z`
- Objective: Audit mixed Unknown composition and Known/Unknown prevalence robustness using frozen CIC score pools

## Data and split

- Reused 10 frozen Stage42-S–Y score files: 2,272 identical Known Test flows (`BENIGN + PortScan`) and 260,220 unique Unknown flows from 10 attack classes.
- Alignment audit found zero Known/Unknown overlaps and zero cross-Unknown duplicate flow IDs. All four method thresholds and every saved threshold decision replayed exactly.
- Each mixture contains 2,000 samples. Five Known:Unknown ratios, seven Unknown-composition settings and three Known-composition settings were evaluated with deterministic seeds 2022–2041.

## Configuration and execution

- Commands: `python run_stage43.py freeze`, `python run_stage43.py run`, then `python verify_stage43.py`.
- No encoder training, score fitting, threshold calibration, GPU inference or PCAP read occurred. The unchanged Stage40 Known-Validation P95 thresholds were reused.
- Generated 300 mixtures, 600,000 membership rows, 1,200 run-level method rows, 10,320 per-Unknown-class rows and 2,400 Known-class rows.

## Core results

- Final Gate: **`COMPOSITION_SENSITIVE`**.
- DES-v1 remained stable across class-balanced prevalence shifts: AUROC means `0.980384–0.981316`, UFAR means `0.067250–0.069500`, Known FRR means `0.068250–0.072300`.
- AUPRC changed mechanically from `0.852588` at Known:Unknown 9:1 to `0.997304` at 1:9. This is prevalence sensitivity of the metric, not a detector improvement.
- All-balanced DES-v1: AUROC `0.980976`, AUPRC `0.976932`, UFAR `0.067100`, Known FRR `0.070650`.
- Authentication/Web mixture: AUROC `0.995600`, UFAR `0`. Previously-hard Bot+DDoS mixture: AUROC `0.947600`, UFAR `0.338350`.
- In the all-balanced setting, Bot UFAR was `0.495000` and DDoS UFAR `0.169500`; the remaining eight class means were at or below `0.002000`.
- Known composition matters: overall Known FRR was `0.091250` at BENIGN:PortScan 3:1 and `0.050250` at 1:3. At 1:1, BENIGN FRR was `0.1075` versus PortScan `0.0335`.
- Binary Accuracy / Binary F1 are now included in every aggregate table. Across prevalence settings, Accuracy stayed near `0.9293–0.9306`, while F1 changed from `0.725275` at Known:Unknown 9:1 to `0.960223` at 1:9 because Unknown is the positive class.
- All-balanced Binary Accuracy/F1 were `0.931125/0.931247`; Authentication/Web `0.964575/0.965790`; Bot+DDoS `0.795325/0.763697`.

## Preserved evidence

- Frozen protocol and source hashes; score-pool/alignment audit; full mixture membership; run-level, class-level and method-comparison CSVs; main report; independent verification; tmux logs; `manifest.json`.

## Limitations

- This is a post-hoc diagnostic over previously exposed attack classes and scores, not untouched validation.
- CIC attack day, endpoint and capture confounding remains. Score-level mixing cannot remove it.
- Resampling quantiles quantify mixture sampling variability, not model-training or cross-capture uncertainty.
- The aggregate mixed result can conceal severe class-specific failure, especially Bot and DDoS.

## Conclusion and next step

- Ratio changes alone did not destabilize rank discrimination or conditional error rates. Unknown composition did.
- Do not report one mixed aggregate without per-class UFAR. No new detector, threshold or candidate experiment is started automatically.
