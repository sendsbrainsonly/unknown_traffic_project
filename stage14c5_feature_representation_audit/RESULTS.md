# Experiment results: stage14c5-vnat-feature-representation-audit-20260917-v1

- Status: `success`
- Experiment type: `feature-ablation`
- Claim scope: `diagnostic`
- Created (UTC): `2026-09-17T15:02:54Z`
- Objective: Compare F0/F1/F2/F3 traffic input representations using only frozen VNAT Known Train and Known Validation across all 15 protocols without Unknown or Known Test access.

## Data and split

- Frozen Stage 14B hash: `c904daef04d2c8e79469191121325c0a6ceeaeb3e595ef9c4c7d7a62c980ae2f`.
- Exactly 15 frozen protocols: Low/Medium/High x seeds 2022--2026.
- Model-visible roles: Known Train and Known Validation only.
- Known Test rows loaded: `0`; Unknown Test rows loaded: `0`.
- Raw feature cache: 23,447 unique flows that occur as Known Train/Validation in at least one protocol; 214,440 protocol-specific row uses; 162 captures; 0 packet-count mismatches.

## Configuration and execution

- F0 reuses the 15 frozen Stage 14C checkpoints and receives Known-Val-only diagnostic evaluation.
- F1/F2/F3 keep the exact Stage 14C 1x32x32 encoder, Adam optimizer, batch 512, max 100 epochs, scheduler, augmentation, seed, and Known-Val composite early stopping with patience 5.
- New numeric scalers are fitted independently on each protocol's Known Train only.
- F1/F2/F3 inject features only into F0's already-zero IPv4 source/destination address slots; non-slot F0 bytes are unchanged.
- Physical GPUs were selected live. GPUs 1 and 2 were excluded by the user's standing constraint; GPU 3 was released on request and was not used for the replacement runs.
- Three pre-scan launch failures were preserved in project-local tmux evidence. They stopped before PCAP scanning or result generation and were repaired before the successful cache run.
- `F3 Low-2026` on physical GPU 3 was interrupted after 11 epochs at the user's request. Its partial artifacts remain isolated under `interrupted_attempts/`, were excluded from formal aggregation, and the formal run restarted from scratch on GPU 6.

## Core results

All 60 formal runs completed and passed the completion audit. Values below are means over all 15 setting-seed protocols.

| Feature | Accuracy | Macro-F1 | Weighted-F1 | Minority recall | Weak runs | Mean within-setting Macro-F1 std |
|---|---:|---:|---:|---:|---:|---:|
| F0 | 0.799710 | 0.614834 | 0.770825 | 0.570958 | 2 | 0.192951 |
| F1 | 0.781039 | 0.612508 | 0.750486 | 0.592474 | 1 | 0.206921 |
| F2 | 0.820070 | 0.652969 | 0.796212 | 0.612431 | 0 | 0.154531 |
| F3 | 0.812372 | 0.638205 | 0.786493 | 0.593201 | 1 | 0.166071 |

- F1 versus F0: `-0.002326` Macro-F1 and `-0.020339` Weighted-F1; no overall improvement.
- F2 versus F0: `+0.038136` Macro-F1, `+0.025387` Weighted-F1, `+0.041473` minority recall, and Macro-F1 wins in `9/15` paired runs.
- F3 versus F0: `+0.023371` Macro-F1 and `+0.015668` Weighted-F1; smaller overall gain than F2.
- F2 setting-level Macro-F1 deltas were Low `+0.052920`, Medium `+0.061742`, and High `-0.000255`.
- The two weak F0 Medium runs changed from `0.334134/0.284444` Macro-F1 to F2 `0.373376/0.653313`. F2 had no run below `0.35`, but Medium-2024 remained weak.
- F2 was both the best non-F0 representation and the most stable representation by the pre-registered stability statistic.

## Preserved evidence

- `manifest.json`
- `README.md` (pre-registered feature encoding and decision rules)
- `raw_feature_cache/cache_audit.json`
- `input_verification.json`
- `runs/`, `checkpoints/`, and `confusion_matrices/`
- project-local `.tmux-task/` logs for successful and failed attempts

## Limitations

- This audit measures Known Validation representation quality only. It cannot establish Unknown Detection quality.
- Flow-level statistics use the current VNAT flow/session definitions; they do not change the frozen Stage 14B split.
- Validation performance remained seed-dependent, and the best feature won fewer than two thirds of paired protocols.

## Conclusion and next step

- Added features are not wholly ineffective: F2 gives a moderate, measurable improvement and reduces weak-run frequency and seed variance. However, its `+0.038136` Macro-F1 gain, `9/15` paired wins, and `+0.041473` minority-recall gain do not meet the pre-registered primary-bottleneck or new-feature-freeze gates.
- Final decision: feature representation is **not established as the primary bottleneck**, and **do not yet freeze a replacement feature configuration**. F2 is the preferred candidate for a later targeted stability study, not an automatically accepted formal replacement.
- Do not run Unknown Detection or Stage 14D as part of this audit.
