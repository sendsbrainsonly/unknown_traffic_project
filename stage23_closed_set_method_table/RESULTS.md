# Experiment results: stage23-matched-closed-set-method-table-20260923-v1

- Status: COMPLETE / FIVE_METHOD_MATCHED_TABLE_VERIFIED (2026-09-23 terminal)
- Experiment type: `benchmark`
- Claim scope: `diagnostic`
- Created (UTC): `2026-09-23T15:42:39Z`
- Objective: Build a flow-ID matched closed-set comparison table across eligible methods on the frozen Stage20 Service tasks

## Data and split

- Reuse the untouched Stage 20 Service-level closed-set manifest, SHA256
  `6404d496f50a6f7f6f397d125ad0cd653c9df152d8db1b9ed2ad2310967950bb`.
- ISCX-VPN: 6 Services, 10,955 flows, Train/Validation/Test
  `8,764/1,098/1,093`. ISCXTor2016: 7 Services, 11,181 flows,
  `8,946/1,118/1,117`.
- Exact image groups do not cross roles. Labels are capture-derived weak labels;
  the split is not uniformly capture-disjoint.

## Configuration and execution

- `scripts/build_preflight.py` completed 120/120 checks: exact Stage 20
  membership, source-array index bounds, label/role parity, Stage 21 cache
  coverage and Stage 22 per-sample E1/E3 prediction replay.
- Existing Stage 22 E1/E3 results are reused, not retrained. Eight
  dataset/seed/method rows were imported only after flow-ID and label parity.
- New method pilot: Open-Detect corrected-paper (not released-code Native),
  with the frozen Stage 16S 100-epoch recipe, batch 128, Adam LR 0.001,
  prototype reset at epochs 51/81 and checkpoint selection on Known Validation
  Accuracy. Stage 20 images are reloaded by exact source index and checked
  against every frozen image hash.
- CPU-only input preflight passed for both datasets with Known Test and Unknown
  Test feature values loaded equal to zero. Three one-GPU runs are in progress:
  VPN seed2022 on physical GPU0, TOR seed2022 on GPU1, VPN seed2023 on GPU2.
  This respects the three-GPU ceiling. TOR seed2023 has not started.
- Live progress is available with
  `python -B stage23_closed_set_method_table/scripts/show_progress.py` inside
  the mandated tmux helper. The helper logs and per-epoch JSONL histories are
  retained. No Test metric from a new method is available yet.

## Core results

Verified Stage 22 reused baselines, mean over seeds 2022/2023:

| Dataset | Pretrained TrafficFormer E1 Macro-F1 | OURS-E3-T8 Macro-F1 |
|---|---:|---:|
| ISCX-VPN Service-6 | 0.860607 ± 0.003961 | 0.859493 ± 0.007015 |
| ISCXTor Service-7 | 0.821259 ± 0.006376 | 0.822178 ± 0.007643 |

These are the first two method rows of the matched closed-set table. They
must not be interpreted as a completed cross-method ranking.

## Preserved evidence

- `PROTOCOL.md`, `method_readiness.csv`, `sample_inventory.csv`,
  `sample_parity.csv`, `closed_set_run_results.csv`,
  `preflight_verification.json` and the read-only preflight scripts.
- Input-preflight JSONs, three unique tmux logs/status files, per-run
  configurations, evolving training histories and best checkpoints.
- `manifest.json`; its artifact inventory will be refreshed after active
  files stop changing.

## Limitations

- New Open-Detect training is not complete; no new Test score can be claimed.
- YaTC, ET-BERT, TFE-GNN, RoNeTC and Trident need exact Stage 20 input
  reconstruction audits before formal rows. UnDiff is a one-class detector
  and has no native 6/7-way closed-set row.
- The public TrafficFormer pretraining corpus overlap is not established.

## Conclusion and next step

The matched benchmark has started and its sample-parity gate passed. After one
GPU slot is released, launch TOR seed2023 using the same adapter and frozen
configuration; then independently replay all Open-Detect Test predictions,
refresh the bundle inventory and continue only with input-feasible methods.

The paragraph above is preserved as the original launch-state record. The
terminal result and current authoritative interpretation follow below.

## Terminal result — 2026-09-23

Five methods on the exact 22,136 frozen Stage20 flows, two datasets and two
seeds, yielded 20/20 complete verified result rows. Independent final replay
passed 22,441 checks; every new method's checkpoint hash, full flow-ID/label
membership and reported metrics passed. Stage20 manifest hash remained
6404d496f50a6f7f6f397d125ad0cd653c9df152d8db1b9ed2ad2310967950bb.

Known Test Macro-F1 mean ± population std:

| Dataset | TrafficFormer E1 | OURS-E3-T8 | Open-Detect corrected-paper | RoNeTC reconstruction | YaTC pretrained |
|---|---:|---:|---:|---:|---:|
| ISCX-VPN | 0.860607 ± 0.003961 | 0.859493 ± 0.007015 | 0.744265 ± 0.014241 | 0.715732 ± 0.008051 | **0.887085 ± 0.011593** |
| ISCXTor | 0.821259 ± 0.006376 | 0.822178 ± 0.007643 | 0.576563 ± 0.099701 | 0.534011 ± 0.019844 | **0.827393 ± 0.000774** |

YaTC − TrafficFormer paired Macro-F1 is +0.026478 on VPN and +0.006134 on
Tor (both seeds positive in each dataset). E3 − E1 is −0.001115 on VPN and
+0.000918 on Tor (one of two seeds positive in each). Thus E3 is essentially
on par with, not convincingly above, the TrafficFormer baseline here. YaTC
is highest, but its Tor margin is small. Open-Detect Tor-2023 weak run is
preserved; no best-run cherry-picking.

The formal report with Accuracy, Weighted-F1, per-service errors, resource
costs, protocol caveats and method exclusions is STAGE23_FINAL_REPORT.md.
Machine-readable terminal outputs are stage23_full_run_results.csv,
stage23_full_summary.csv, stage23_paired_vs_e1_e3.csv,
stage23_all_method_per_class.csv, stage23_all_method_confusion.csv and
completion_verification.json. Checkpoints, training histories, prediction
files and input caches remain in the bundle. ET-BERT strict flow, TFE-GNN
native TCP graph, Trident native and UnDiff are not mixed into the exact-flow
multiclass table; METHOD_ADMISSION_AUDIT.md records why.

The earlier CPU YaTC diagnostic and seed2023 parity failure are retained.
The formal YaTC results use CUDA mixed-precision post-selection replay for
both seeds, with exact Known Validation parity before Test acceptance. All
GPU training/evaluation jobs finished; physical GPUs 0–3 were released
well before the user's 08:00 Asia/Shanghai deadline. No Unknown detection
was run. This is a development diagnostic on weak capture-derived labels
and non-uniformly capture-disjoint splits, not independent external evidence.

## Progress update — 2026-09-23 17:00 UTC

The preceding launch description is historical, not current live state. All
four Open-Detect corrected-paper runs are complete and independently replayed:
`phase1_completion_verification.json` reports 40,025 checks, 4/4 new runs,
12 verified rows including eight reused E1/E3 baseline rows. Open-Detect
Known Test Macro-F1 by dataset/seed: VPN 2022 `0.758506`, VPN 2023
`0.730024`; Tor 2022 `0.676264`, Tor 2023 `0.476862`. The weak
Tor-2023 run is retained.

RoNeTC exact-flow 8-packet three-view cache now covers all 22,136 frozen
flows; 137 Tor flows missing in Stage19 were recovered from the original
PCAP by exact frozen packet refs. Four formal 100-epoch RoNeTC runs are live
on physical GPUs 0–3; smoke/parity tests passed. YaTC author-rule MFR Known
Train/Val caches are complete on both datasets. Two Tor formal 200-epoch
YaTC runs are live on GPUs 4–5; the VPN pair is queued after these finish.
YaTC Known Test MFR is intentionally not built before checkpoint selection.
`scripts/evaluate_yatc_closed.py` and `scripts/finalize_stage23.py` prepare
post-selection evaluation and independent table replay.

Native-input admission: strict ET-BERT would discard at least 15,195/22,136
one/two-packet flows; TFE-GNN's TCP graph input already excludes 8,316 UDP
flows in Known Train/Val. Trident's current 86-D cache is for different full
flows, and UnDiff has no native multiclass head. See
`METHOD_ADMISSION_AUDIT.md` and `native_method_coverage_audit_v2.json`.
These methods' historical scores must not be merged into the exact-flow table.

User-authorized six-GPU ceiling is currently respected. A project-local
deadline guard will release the four RoNeTC sessions on GPUs 0–3 by
2026-09-24 07:55 Asia/Shanghai if they are still live. No Unknown results
or Test-guided tuning have been used.
