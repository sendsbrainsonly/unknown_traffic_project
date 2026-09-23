# Experiment results: stage15a-failure-regime-complementarity-20260918-v1

- Status: `success`
- Experiment type: `diagnostic_analysis`
- Claim scope: `diagnostic`
- Created (UTC): `2026-09-18T14:08:03Z`
- Objective: diagnose frozen Open-Detect versus DES failure regimes and sample-level complementarity without training, refitting, or detector design.

## Data and split

- Primary: VNAT Stage 14D, all 15 frozen Low/Medium/High × seeds 2022–2026 protocols.
- Supplementary: 15 frozen USTC development protocols with M0/M1/M2; 15 ISCX-VPN and 15 ISCXTor2016 frozen protocols with M0/M1 only.
- Existing evaluation composition was preserved: VNAT/ISCX natural test composition; USTC frozen balanced 1:1 composition.
- 60 protocols, 160 protocol × Unknown-class units, and 481,190 sample/comparison rows were analyzed.
- All 255 referenced frozen source files were SHA256-checked before and after; no hash changed.

## Configuration and execution

- M0: existing Native learned-prototype forward Gaussian KL, decomposed exactly as `D_proto + V_post`.
- M1: existing empirical Known-Train centroid squared distance.
- M2: existing frozen 0.5/0.5 robust global + kNN-10 local score; no M2 was invented for ISCX.
- Operating point: each method's existing Known-Validation P95 (`numpy method=higher`).
- Ranking complementarity: exact unknown-known pair ordering; no threshold involved.
- Correlations: exploratory protocol × Unknown-class Spearman tests.
- Execution: CPU only in tmux session `s15a_analysis_enriched_20260918_1641`; no encoder inference/training, detector fitting, parameter search, or GPU.
- Unit tests: 4/4 passed.

## Core results

| Dataset | M0 AUROC | M1 AUROC | M2 AUROC | M1−M0 | M2−M0 |
|---|---:|---:|---:|---:|---:|
| VNAT | 0.837535 | 0.851706 | 0.853085 | +0.014171 | +0.015549 |
| USTC-TFC2016 | 0.939528 | 0.970982 | 0.979954 | +0.031454 | +0.040426 |
| ISCX-VPN | 0.553024 | 0.576476 | unavailable | +0.023452 | unavailable |
| ISCXTor2016 | 0.524614 | 0.594347 | unavailable | +0.069734 | unavailable |

- VNAT M1−M0 Unknown-class/protocol units: 24 positive and 21 negative.
- VNAT P95 complementarity for M1 vs M0: OD-only rescue 15,129; DES-only rescue 11,585.
- VNAT exact ranking complementarity: OD-only correct 6,203,271 pairs; DES-only correct 7,741,572 pairs.
- H1 prototype mismatch was not established globally: rho=0.200659, p=0.186278, n=45.
- H2 empirical-support overlap was supported exploratorily: overlap rho=−0.343555, p=0.020854; nearest-support distance rho=0.300659, p=0.044768.
- Margin/ratio ambiguity and aggregate closed-set confusion were not statistically established.
- Native uncertainty rescued 19/21 VNAT DES-loss units relative to learned-prototype distance alone.
- rsync and scp were the clearest DES-v0 failures (mean ΔAUROC −0.161182/−0.045337), with overlap 0.981433/0.958865 and mutual nearest-support absorption.
- sftp was not a mean failure: ΔAUROC +0.202935 and improved in 4/5 units, consistent with its unusually large normalized prototype mismatch (2.460952).
- Final gate: `COMPLEMENTARITY_CONFIRMED`.

## Preserved evidence

- `stage15a_protocol_diagnosis.csv`
- `stage15a_unknown_class_diagnosis.csv`
- `stage15a_sample_complementarity.csv`
- `stage15a_signal_correlation.csv`
- `stage15a_failure_regime_report.md`
- `source_hashes_before.json` / `source_hashes_after.json`
- `completion_verification.json`
- `config.json`, analysis script, and unit tests
- tmux log: `.tmux-task/s15a_analysis_enriched_20260918_1641/output.log`

## Limitations

- Correlation rows are not independent across seeds, protocols, and repeated class identities; p-values are exploratory.
- USTC is development evidence, not independent external validation.
- VNAT is not described as untouched because it has already been used in prior stages.
- ISCX-VPN/ISCXTor2016 contain no frozen M2 scores, so they support only the M0-vs-M1 decoupling analysis.
- Mean improvement does not imply uniform improvement: VNAT Stage 14D already showed protocol-level instability.

## Conclusion and next step

`COMPLEMENTARITY_CONFIRMED`: Open-Detect and DES make substantial errors in both directions. The most reproducible failure-regime signal is empirical support overlap, while global prototype mismatch alone does not explain gains. Posterior uncertainty specifically retains useful signal in the rsync/scp DES failure regime. This is sufficient evidence to permit—but does not itself implement—a later Hybrid-detector design stage.
