# Stage 25 — Semantic-coarse matched-flow closed-set development

This is a development comparison; Stage20 Known Test has been exposed before Stage25. No Unknown/open-set claim.
The hard-remap and independently trained coarse-head experiments are different estimands and are not mixed in one ranking.

## Frozen label map

Chat + Email + VoIP → Communication; File-Transfer, P2P, Streaming remain separate; Tor Browsing remains separate.
Original six/seven-class checkpoints and predictions are unchanged.

## Full-flow Known Test Macro-F1, mean of two seeds

| Dataset | Method | Original fine | Coarse hard remap | Frozen-encoder coarse head |
| --- | --- | ---: | ---: | ---: |
| iscx_vpn | E1 | 0.860607 | 0.901497 | 0.896161 |
| iscx_vpn | E3 | 0.859493 | 0.896004 | 0.897301 |
| iscx_vpn | OD | 0.744265 | 0.792638 | — |
| iscx_vpn | RoNeTC | 0.715732 | 0.772034 | — |
| iscx_vpn | YaTC | 0.887085 | 0.923689 | — |
| iscx_vpn | TFE | 0.658549 | 0.735528 | — |
| iscx_vpn | Trident | 0.632207 | 0.757029 | — |
| iscx_tor | E1 | 0.821259 | 0.852637 | 0.853024 |
| iscx_tor | E3 | 0.822178 | 0.852783 | 0.854317 |
| iscx_tor | OD | 0.576563 | 0.651818 | — |
| iscx_tor | RoNeTC | 0.534011 | 0.646579 | — |
| iscx_tor | YaTC | 0.827393 | 0.876259 | — |
| iscx_tor | TFE | 0.553611 | 0.624333 | — |
| iscx_tor | Trident | 0.605258 | 0.675505 | — |

## ≥2-packet conditional Known Test Macro-F1 and coverage

| Dataset | Method | Fine | Coarse hard remap | Coarse head |
| --- | --- | ---: | ---: | ---: |
| iscx_vpn | E1 | 0.850765 | 0.885403 | 0.879747 |
| iscx_vpn | E3 | 0.847921 | 0.878272 | 0.880616 |
| iscx_vpn | OD | 0.768688 | 0.820250 | — |
| iscx_vpn | RoNeTC | 0.730602 | 0.790393 | — |
| iscx_vpn | YaTC | 0.879758 | 0.906105 | — |
| iscx_vpn | TFE | 0.682273 | 0.778427 | — |
| iscx_vpn | Trident | 0.607762 | 0.741262 | — |
| iscx_tor | E1 | 0.825056 | 0.878324 | 0.881900 |
| iscx_tor | E3 | 0.830292 | 0.880108 | 0.882709 |
| iscx_tor | OD | 0.693889 | 0.780469 | — |
| iscx_tor | RoNeTC | 0.646152 | 0.767982 | — |
| iscx_tor | YaTC | 0.834466 | 0.901986 | — |
| iscx_tor | TFE | 0.707469 | 0.805219 | — |
| iscx_tor | Trident | 0.695401 | 0.774955 | — |
- iscx_vpn known_train: 6571/8764 retained (74.98%); exact per-original-Service counts in `scope_coverage.csv`.
- iscx_vpn known_validation: 830/1098 retained (75.59%); exact per-original-Service counts in `scope_coverage.csv`.
- iscx_vpn known_test: 807/1093 retained (73.83%); exact per-original-Service counts in `scope_coverage.csv`.
- iscx_tor known_train: 4946/8946 retained (55.29%); exact per-original-Service counts in `scope_coverage.csv`.
- iscx_tor known_validation: 629/1118 retained (56.26%); exact per-original-Service counts in `scope_coverage.csv`.
- iscx_tor known_test: 595/1117 retained (53.27%); exact per-original-Service counts in `scope_coverage.csv`.

The ≥2-packet restriction does **not** uniformly improve scores: our E3 coarse hard-remap Macro-F1 falls from 0.896004 to 0.878272 on VPN, while it rises from 0.852783 to 0.880108 on Tor. Tor's rise applies to only 53.27% of its Known Test flows; the discarded one-packet population cannot be ignored. Full per-seed values and population standard deviations are in `remap_run_results.csv` and `stage25_summary.csv`.

## Integrity and interpretation

- 28/28 historical fine Known Test metrics independently replayed. 48/48 protected input SHA256 checks unchanged.
- Independent numeric replay: 61,964 hard-remap predictions, 224 metric rows, all 8 saved head parameter sets, and 17,704 coarse-head decisions PASS (`independent_verification.json`).
- Seven methods were compared on identical flow IDs, roles, coarse mapping and subset rule. The new E1/E3 heads used only Known Train representations for scaler and fit; original encoders/fine heads were not modified.
- Hard remap maps an existing argmax label; it does not sum class probabilities. A new coarse head is additional training and must be compared separately.
- Higher coarse Macro-F1 means a different, easier label task, not improved six/seven-class skill. ≥2-packet scores condition on a **different** population; it is not uniformly easier and must be read with coverage.
- Capture-derived Service labels are weak labels and Stage20 is not uniformly capture-disjoint; this experiment does not repair those limits.

## Files

`remap_run_results.csv`, `remap_per_class.csv`, `remap_predictions.csv`, `coarse_head_fit.csv`, `coarse_head_results.csv`, `coarse_head_per_class.csv`, `coarse_head_predictions.csv`, `scope_coverage.csv`, `stage25_summary.csv`, source hashes and verification JSON.
