# Experiment results: stage26-yatc-e3-frozen-score-fusion-20260924-v1

- Status: `success / FUSION_NOT_CONFIRMED`
- Experiment type: `benchmark`
- Claim scope: `diagnostic`
- Created (UTC): `2026-09-24T15:58:15Z`
- Objective: Test whether a fixed equal-weight frozen YaTC+E3 probability ensemble improves matched Stage20 fine/coarse closed-set performance beyond both single models

## Data and split

- Same frozen Stage20 ISCX-VPN and ISCXTor2016 Known Validation/Test flow IDs, original 6/7 Service labels and Stage25 fixed 4/5 coarse mapping; seeds 2022/2023. Full-flow is primary; ≥2-packet is conditional. No Unknown examples were accessed.

## Configuration and execution

- Frozen Stage22 E3 representation/checkpoint plus frozen Stage23 YaTC checkpoint and exact MFR cache. Replayed each model's fine-class logits using its original inference path; historical Validation/Test argmax parity was exact in 4/4 pairs. Single fixed rule `0.5·softmax(E3)+0.5·softmax(YaTC)`; coarse Communication probability sums Chat/Email/VoIP before argmax. No training, normalization fit, temperature/weight search or checkpoint replacement.
- Four sequential GPU inference runs used live-selected physical GPU availability, with project-local tmux logs. Aggregation and independent numerical replay ran in named tmux sessions. All source SHA256 values were unchanged before/after each run.

## Core results

- Full-flow Known Test Macro-F1, mean of 2022/2023:

| Dataset | Task | E3 | YaTC | Fusion | Fusion−YaTC |
| --- | --- | ---: | ---: | ---: | ---: |
| VPN | original fine | 0.859493 | 0.887085 | 0.878475 | −0.008610 |
| VPN | semantic coarse | 0.895838 | 0.923689 | 0.906884 | −0.016805 |
| Tor | original fine | 0.822178 | 0.827393 | 0.848813 | +0.021420 |
| Tor | semantic coarse | 0.852381 | 0.875886 | 0.877392 | +0.001506 |

- Primary coarse gate: `FUSION_NOT_CONFIRMED`; only 1/4 paired dataset-seed cells beat YaTC. VPN coarse loss is concentrated in Communication and File-Transfer. Fusion does improve on E3, but it fails to consistently beat the stronger YaTC component.
- Independent replay from saved logits passed 26,556 predictions, all 96 metric rows and 60 source-hash entries. Four per-run bundles and this aggregate bundle passed artifact validation.

## Preserved evidence

- `EXPERIMENT_PLAN.md`, frozen logits and complete predictions in `runs/`, each run's `RESULTS.md`/`manifest.json`, `run_metrics.csv`, `setting_summary.csv`, `paired_comparison.csv`, `per_class.csv`, `sample_rescue.csv`, `stage26_report.md`, completion and independent verification JSON, source hashes and project-local tmux logs/status files.

## Limitations

- Stage20 Test was previously exposed, so this is development evidence. Only two seeds and one fixed weight were tested; no post-Test tuning is justified. Probability aggregation is not equivalent to Stage25 hard-prediction remapping, so their coarse baseline values differ slightly. The ensemble keeps two encoders and adds inference/storage cost; cached-embedding run times exclude E3 feature extraction. Method/representation/pretraining differences are not causally isolated.

## Conclusion and next step

- A naive equal-weight YaTC+E3 fusion does not deliver a stable improvement over YaTC on the current coarse task. Keep YaTC as the stronger closed-set baseline; preserve the Tor fine-class rescue as diagnostic complementarity only. Stop without selecting a new weight from exposed Test results.
