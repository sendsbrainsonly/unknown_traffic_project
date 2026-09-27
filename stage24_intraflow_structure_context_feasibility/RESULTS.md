# Experiment results: stage24-intraflow-structure-context-feasibility-20260924-v1

- Status: `success` (diagnostic result: `PILOT_NO_GLOBAL_GAIN`)
- Experiment type: `benchmark`
- Claim scope: `diagnostic`
- Created (UTC): `2026-09-24T12:28:54Z`
- Objective: Test whether added within-flow packet and burst structure improves the frozen TrafficFormer dual-branch representation, and audit cross-flow context feasibility without leakage

## Data and split

- Stage 20 frozen ISCX-VPN Service-6 and ISCXTor Service-7. The pilot used Known Train and Known Validation only, with Stage 22 E1 representations from seeds 2022/2023 and aligned Stage 21 first-eight-packet inputs. Known Test and Unknown Test feature use: 0.
- VPN: 8,764 Known Train, 1,098 Known Validation; Tor: 8,946 Known Train, 1,118 Known Validation. All Stage 20/21/22 protected assets were checked before and after (19/19 SHA256 unchanged).

## Configuration and execution

- 24A: 4 variants (S1 statistics/burst MLP, G1 augmented packet FIG, G2 packet→burst→flow, G2-shuffle negative control) × 2 datasets × 2 seeds = 16 completed runs. Frozen E1 and E3 baselines were reused in every paired cell. New branch: 50 epochs, batch 64, Adam 1e-3; fusion: 30 epochs, batch 256, Adam 1e-3; best epoch by Known Validation Macro-F1. All new normalization was Known-Train-only. At most 3 physical GPUs (0–2) concurrently.
- 24B: read-only capture/group, timestamp and neighbor-count feasibility audit. No cross-flow GNN trained.
- Queue session: `codex_stage24_pilot_queue_20260924`; aggregate session: `codex_stage24_aggregate_20260924`. Per-run commands, histories, scores, checkpoints, logs and status are in `runs/pilot/`.

## Core results

- 16/16 run bundles completed and independently validated. Mean Known Validation Macro-F1 (2022/2023):

| Dataset | E1 | E3 | S1 | G1 | G2 | G2-shuffle |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ISCX-VPN | 0.849134 | 0.851877 | 0.850469 | 0.850401 | 0.849936 | 0.850997 |
| ISCXTor | 0.850669 | 0.844107 | 0.846332 | 0.846216 | 0.846903 | 0.846500 |

- No S1/G1/G2 candidate beat both E1 and E3 in both datasets. `PILOT_NO_GLOBAL_GAIN`; no five-seed confirmation was launched.
- 24B gate: `INTERFLOW_NOT_FEASIBLE_FOR_FULL_CLASS_CAPTURE_DISJOINT_CLAIM`. VPN P2P has one capture; the frozen flow-random protocol shares captures across roles. Cross-flow modeling was not launched.

## Preserved evidence

- `EXPERIMENT_PLAN.md`, `DECISION_RULE_AMENDMENT.md`, `stage24_report.md`, `preflight.json`, protected-file hash records, five pilot CSVs, `pilot_candidate_decision.json`, `completion_verification.json`, all 16 per-run bundles, failed first preflight evidence, queue logs, and this `manifest.json`.

## Limitations

- Only two seeds were run; this is a development pilot, not evidence of stable superiority. The pilot continuation rule was amended after the first negative S1 result, as disclosed in `DECISION_RULE_AMENDMENT.md`.
- The first eight packets leave 79.14% of VPN and 58.50% of Tor Known Train/Validation flows at just one or two packets; structure is sparse. G2 has more parameters than S1/G1, so individual differences cannot be assigned solely to hierarchy. Stage 20 Test was previously exposed, and official pretraining Unknown exposure is unresolved; no open-set or untouched-external claim is made.

## Conclusion and next step

- Neither global within-flow structure improvement nor safe full-class cross-flow context was established. Stop Stage 24 at its stated gates; preserve these diagnostics and design any future context protocol separately, without relabeling this pilot as a positive result.
