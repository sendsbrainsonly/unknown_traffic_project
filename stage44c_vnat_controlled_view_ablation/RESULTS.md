# Stage44C — VNAT controlled frozen-view ablation

- Status: `success / independent replay PASS`.
- Owner: codex. Single seed 2022; four previously developed VNAT Service LOSO folds.

## Data and split

Frozen Stage44 flow IDs/labels/roles; no protocol regeneration. Four folds, each 23,449 source flows.

## Configuration and execution

16 freshly initialized fusion heads; identical 192→128→128→C architecture, seed, 30 epochs, batch256, Adam1e-3 and Known-Val Macro-F1 selection. Encoders/adapters/scalers frozen. Missing 64D slots zeroed; scaling remains1/3. All heads frozen before Test inference. One live-selected GPU, serial queue.

## Core results

DES-v1, natural Test prevalence, Known-Val P95. Values below are four-fold descriptive means, not independent-seed confidence estimates.

| Variant | Closed Macro-F1 | Closed Weighted-F1 | AUROC | AUPRC | Unknown F1 | UFAR | Known FRR |
|---|---:|---:|---:|---:|---:|---:|---:|
| full | 0.998711 | 0.998874 | 0.992224 | 0.994998 | 0.983976 | 0.010907 | 0.044550 |
| without_fig | 0.998905 | 0.999217 | 0.994513 | 0.997660 | 0.986366 | 0.014051 | 0.024944 |
| without_yatc | 0.998731 | 0.998853 | 0.969187 | 0.987453 | 0.898269 | 0.148026 | 0.049133 |
| without_trafficformer | 0.997419 | 0.997482 | 0.983478 | 0.988058 | 0.941445 | 0.085304 | 0.048112 |

## Preserved evidence

PLAN.md/config.json; source_lock.json; HEADS_FROZEN.json; latent caches; 16 best heads; training dynamics, role-level embeddings/logits, sample scores, per-class/confusion/calibration files; run_results.csv, closed_results.csv, paired_comparison.csv, variant_summary.csv, historical_full_comparison.csv; independent replay and source hash verification.

## Limitations

Existing Known application coverage gaps and VPN/non-VPN imbalance remain. This isolates view information under frozen jointly selected independent adapters; it does not establish encoder-training or adapter-design causality. Effective informative input dimensions change despite equal nominal architecture. Previously exposed Test; single seed; four folds share source data. Full closed-set scores can have a ceiling effect. No Test-driven architecture or threshold choice.

## Conclusion and next step

Inspect each fold and both closed/open metrics before attributing a benefit to any branch. An ablation improving a metric is evidence against assuming that branch always helps in this protocol. All variants retained; no automatic branch selection, protocol repair, hyperparameter search or subsequent training is triggered.

Observed: removing YaTC or TrafficFormer reduces AUROC in all four folds. Removing FIG improves average AUROC but slightly worsens average UFAR, while reducing Known FRR; the complete three-view model is not uniformly better than TF+YaTC. See the [Chinese analysis and per-fold tables](stage44c_ablation_report.md). Seven source input cache arrays were separately checked unchanged; all historical Full operating-point/ranking metrics reproduce exactly.
