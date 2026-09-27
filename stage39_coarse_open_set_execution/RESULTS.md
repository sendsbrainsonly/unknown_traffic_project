# Experiment results: stage39-coarse-service-open-set-execution-20260926

- Status: `failed / 8 of 12 new-training settings trained; 1 of 12 evaluated` (queue snapshot 2026-09-26 16:16 UTC)
- Experiment type: `open-set-evaluation`
- Claim scope: `diagnostic`
- Created (UTC): `2026-09-26T08:27:04Z`
- Objective: Execute predeclared whole-service holdouts with the frozen three-view recipe and Known-only detection calibration

The later `training_queue_progress.json` records `FAILED` at `VNAT/communication` fusion (worker exit code 2). The dependent evaluation queue also records `FAILED` and did not evaluate the incomplete settings. The worker failure has not been diagnosed in this publication task; the earlier running-state narrative below is historical, not a live queue claim.

## Data and split

- Frozen Stage39 whole-service holdouts: 12 new-training settings (VPN 4, Tor 4, VNAT 4) plus one retrospective Tor P2P-only control. Seed 2022; VNAT retains Stage14B `medium_seed2025` flow membership. Role manifests and original source hashes are recorded in `protocol_index.csv` and `freeze_verification.json` (PASS, no Unknown in Known Train/Val, no historical Test promoted to training).
- The first completed setting holds out all ISCX-VPN Communication flows: Known Train/Val/Test 6,823/851/858, Unknown evaluation 10,528, and separately evaluated known-service/unseen-application flows 3,082.

## Configuration and execution

- Stage32/38 staged three-view TrafficFormer + FIG/TAGCN + YaTC, 64-D adapters, fixed equal feature fusion. Stage37 end-to-end joint training is **not** used: its previous VPN/Tor closed-set result was lower under a different training budget.
- DES-v1 uses Known Train class centroids and class-conditional kNN-10, Known Validation median/MAD, fixed global/local 0.5/0.5; MSP is the reference score. Known-Val P90/P95/P99 thresholds use `method="higher"` and `score > threshold` for rejection. No Test/Unknown normalization or threshold fitting.
- Training queue: named tmux `stage39_known_training_queue_v2_0926`, progress `training_queue_progress.json`, physical GPU lanes 0/1/7, maximum three owned GPUs. Dependent evaluation queue: `stage39_evaluation_queue_v2_0926`, progress `evaluation_queue_progress.json`; it waits for Known-only training completion, then runs calibrate → frozen evaluation-cache assembly → one-shot score evaluation → replay → deployment analysis per setting. `stage39_aggregate_queue_v2_0926` waits for every setting and will write aggregate tables/report only after complete replay.
- The first queue attempt completed all four VPN and all four Tor Known-only settings, then stopped at VNAT Communication cache creation because the builder used `input_caches/VNAT` instead of the existing lowercase `input_caches/vnat`. This was a path error before any VNAT Test/Unknown feature read. Original failure files and tmux logs remain intact. The corrected builder passed the VNAT Communication Known Train/Val cache audit at 14,687/1,834 flows; attempt 2 resumed without retraining the eight successful settings.
- First VPN Communication evaluation: `.tmux-task/stage39_vpn_comm_evaluate_0926/` exit 0; independent replay `.tmux-task/stage39_vpn_comm_verify_0926/` exit 0; saved 14,468 unique evaluation scores. Its calibration was completed before the Test cache was assembled.
- While VNAT Known-only training continues, all eight ISCX settings completed independent Known Validation-only calibration (`stage39_iscx_known_calibration_0926`, exit 0). Their frozen evaluation-input caches were then assembled and ID-audited (`stage39_vpn_file_eval_cache_0926` and `stage39_iscx_eval_caches_0926`, both exit 0). Cache construction started only after each setting's calibration PASS and did not fit any model or threshold. These are preparation milestones, **not** eight detection results.

## Core results

The following are **single-setting development results**, not a dataset-level conclusion. P95 is the predeclared operating point; Unknown is positive.

| VPN held-out Communication | Known Test Macro-F1 before rejection | AUROC | AUPRC | UFAR P95 | Known FRR P95 |
|---|---:|---:|---:|---:|---:|
| MSP | 0.995504 | 0.351050 | 0.899017 | 0.918788 | 0.051282 |
| DES-v1 | 0.995504 | 0.405230 | 0.923279 | 0.849354 | 0.058275 |

The Unknown prevalence in that primary comparison is 0.924644, so its high AUPRC must not be read as strong practical rejection. With 3,082 same-service new-application samples included as deployment Known, DES-v1 AUROC/AUPRC is 0.444371/0.714638, Known FRR is 0.136548, and UFAR remains 0.849354. The new model's high closed-set score does **not** satisfy the Stage39 detection goals in this setting. The Tor P2P-only retrospective control reuses Stage38 D1: AUROC 0.913526, UFAR 0.706000; it is not a newly trained Stage39 setting.

## Preserved evidence

- `manifest.json`, `freeze_verification.json`, `protocol_index.csv`, each setting's `protocol.json`/`role_manifest.csv`, three branch and fusion checkpoints, training histories and tmux logs, calibration and per-flow scores, primary/deployment/domain metrics, independent replay markers.
- Completed pilot: `settings/iscx_vpn/communication/detection/` including `completion_verification.json` and `deployment_verification.json`. Tor control: `settings/iscx_tor/p2p_only/results.csv`.

## Limitations

- Eleven new-training settings have not yet produced final detection results; **do not** generalize the first failure or the retrospective Tor control to all settings. VNAT training is ongoing and four VNAT settings remain before the gated evaluation phase. ISCX calibration/cache readiness is not final open-set evaluation.
- ISCX service labels are capture-derived weak labels; Stage39 is exposed development data, not untouched external validation. Capture overlap and domain imbalance remain threats. VPN Communication's Unknown prevalence is high, making AUPRC prevalence-sensitive.
- Training/evaluation queues remain live; intermediate branch Validation scores are not final open-set evidence. Any worker failure must retain its log and stop dependent evaluation.

## Conclusion and next step

- `in_progress`: retain the failed VPN Communication result without changing score, threshold, label, or split. Let the frozen Known-only and gated evaluation queues process all 12 settings, then aggregate every setting, verify protected hashes, and finalize this bundle. Read-only monitoring: `python stage39_coarse_open_set_execution/progress.py` through the workspace tmux helper.
