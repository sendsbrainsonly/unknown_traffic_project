# Experiment results: stage23b-tfe-trident-stage20-matched-20260924

- Status: `complete`
- Experiment type: `benchmark`
- Claim scope: `approximate`
- Created (UTC): `2026-09-24T02:16:00Z`
- Objective: Train adapted TFE-GNN and Trident on exact frozen Stage20 Service flows

## Data and split

- Exact frozen Stage20 Service-6/7 flow IDs, labels and Train/Validation/Test membership.
  Manifest SHA256: 6404d496f50a6f7f6f397d125ad0cd653c9df152d8db1b9ed2ad2310967950bb.
- Known Train/Val input extraction passed: ISCX-VPN 9,862/9,862 flows and
  26,301 packet refs; ISCXTor 10,064/10,064 flows and 34,897 packet refs.
  Known Test is not materialized yet. Unknown data is unused.

## Configuration and execution

- Pre-registration: PROTOCOL.md. All-flow TFE-GNN uses original dual graph,
  GraphSAGE, gated BiLSTM and head, but a frozen first-8-packet aligned
  UDP/empty-payload/PAD adapter. Trident uses the existing reconstructed 86-D
  FlowStats and per-class AEs, computed only from the same first eight packets;
  fixed known-only loss-space classifier calibration.
- Scripts: scripts/build_inputs.py, scripts/run_tfe.py,
  scripts/run_trident.py and scripts/finalize.py. The two-flow TFE CPU
  model-forward smoke test passed.
- Four Trident Known-Validation fits completed: VPN seed2022/2023 Macro-F1
  0.633974/0.626922; Tor seed2022/2023 0.619967/0.609786.
  These are validation numbers, not Test results. Trident EVT emitted
  numerical warnings on Tor-2022 and fell back to documented q85 thresholds
  where required; the classification calibrator is separate.
- VPN TFE seed2022/2023 completed their pre-registered 20 epochs and selected
  checkpoints on Known Validation. Tor seed2022/2023 are running their
  pre-registered 100-epoch schedules in named tmux sessions. The unattended
  queue `codex_stage23b_completion_queue_20260924` enforces at most three
  concurrent GPU training runs and gates all Known Test work on selection.

## Core results

- Final Known Test results follow below; validation values above were used only for selection.

## Preserved evidence

- PROTOCOL.md, exact-flow input arrays and audits, per-run Trident checkpoints
  and validation predictions, TFE smoke log, active TFE logs/history,
  and manifest.json. Failed or interrupted artifacts will remain in place.

## Limitations

- These are adapted methods, not author-native TFE or official Trident USTC
  results. Stage20 labels are weak capture-derived Service labels; the split
  is not uniformly capture-disjoint. TFE graph construction is CPU-heavy.

## Conclusion and next step

- Status: complete. Four TFE Known-Val checkpoint selections, separate Known
  Test materialization, all eight method/dataset/seed evaluations, and the
  independent flow-ID/label/metric replay have completed.


## Terminal same-flow adapted results

Status: COMPLETE / 8 of 8 adapted runs independently replayed. These are
Stage20 same-flow **adaptations**, not native paper implementations.
No Unknown Test was used; Known Test was materialized after all Known-Val selections.

| Dataset | Method | Accuracy mean±std | Macro-F1 mean±std | Weighted-F1 mean±std |
|---|---|---:|---:|---:|
| iscx_vpn | TFE-GNN-8-UDP-short | 0.648673 ± 0.019213 | 0.658549 ± 0.014647 | 0.644530 ± 0.016252 |
| iscx_vpn | Trident-early8-86D | 0.613449 ± 0.002287 | 0.632207 ± 0.000615 | 0.606484 ± 0.000466 |
| iscx_tor | TFE-GNN-8-UDP-short | 0.600716 ± 0.000895 | 0.553611 ± 0.013833 | 0.588108 ± 0.008733 |
| iscx_tor | Trident-early8-86D | 0.632945 ± 0.002686 | 0.605258 ± 0.001875 | 0.632355 ± 0.003314 |

Paired differences to the pre-existing TrafficFormer and E3 rows are in
stage23b_paired_comparison.csv; all eight individual scores are retained
in stage23b_run_results.csv, with per-class, TCP/UDP, packet-count
subgroup results and predictions. Subgroup macro-F1 retains all frozen
dataset services as the label universe.
The original Stage23 five-method table was not modified.
Stage20 frozen manifest SHA256 after evaluation: 6404d496f50a6f7f6f397d125ad0cd653c9df152d8db1b9ed2ad2310967950bb.
Independent checks: 8904.

## Expanded comparison across all Stage 23 methods

To answer the follow-up request, the five frozen Stage23 baselines and both
Stage23B adaptations were joined into an auditable 7-method comparison using
the same dataset, seed, split counts and Known-Test service labels. The
original Stage23 source CSV was read-only.

- `stage23b_all_methods_comparison.md`: all-method mean/std tables, per-seed
  Accuracy and Macro-F1, service-level F1 matrix, and paired mean differences.
- `stage23b_all_methods_run_level.csv`: all 28 method × dataset × seed rows.
- `stage23b_all_methods_summary.csv`: 14 method/dataset mean ± population SD rows.
- `stage23b_all_methods_paired.csv` and
  `stage23b_all_methods_paired_summary.csv`: seed-level and mean differences
  of both adaptations against all five baselines.
- `stage23b_all_methods_per_class.csv`: all service-level precision, recall,
  F1 and support for the seven methods on Known Test.

The complete comparison confirms that both Stage23B adaptations trail the
pretrained TrafficFormer/E3/YaTC baselines on these closed-set splits; details
are in the comparison report and CSVs above.
