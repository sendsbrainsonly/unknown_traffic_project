# Experiment results: stage14c5-protocol-difficulty-validation-dip-audit-20260918-v1

- Status: `success`
- Experiment type: `diagnostic-audit`
- Claim scope: `two-protocol Known-Train/Validation diagnosis`
- Created (UTC): `2026-09-18T07:16:14Z`
- Completed (UTC): `2026-09-18T07:56:22Z`
- Objective: Explain Medium-2025 versus Medium-2026 closed-set difficulty and Stage 14C-4 validation dips without changing training or using Known Test/Unknown Test.

## Data and split

- Stage 14B frozen protocol hash: `c904daef04d2c8e79469191121325c0a6ceeaeb3e595ef9c4c7d7a62c980ae2f`.
- Medium-2025 Known: `netflix, rdp, rsync, scp, skype, ssh, youtube`; Unknown: `zoiper, vimeo, sftp`; Known Train/Val: `15,704/1,960`.
- Medium-2026 Known: `netflix, rdp, rsync, skype, ssh, vimeo, zoiper`; Unknown: `sftp, youtube, scp`; Known Train/Val: `15,328/1,911`.
- Known Test used: `0`; Unknown Test used: `0`; DES executed: `false`.

## Configuration and execution

- Reused immutable Stage 14C-4 F2 training: 100 epochs, batch 128, Adam LR 0.001, MultiStepLR milestones 50/80 with gamma 0.1, prototype resets at epochs 51/81.
- Two primary runs were replayed serially on physical GPU 4 solely to add per-epoch per-class recall and gradient/prototype norm telemetry.
- The replay imported the original training step and reset implementation, restored RNG state around extra validation observation, and was accepted only after exact metric parity.
- Maximum absolute replay difference versus immutable Stage 14C-4 dynamics: `1.1102230246251565e-16`.

## Core results

- Medium-2025 selected Known-Val Accuracy/Macro-F1/Weighted-F1: `0.892857/0.727247/0.864414`.
- Medium-2026 selected Known-Val Accuracy/Macro-F1/Weighted-F1: `0.994767/0.980668/0.994803`.
- Primary Macro-F1 gap (2026 minus 2025): `+0.253421`.
- With both protocols trained at seed 2022, the gap remains `+0.214858`, retaining `84.78%` of the primary gap.
- Medium-2025: `185/191` rsync validation samples were predicted as scp; Medium-2026 holds scp out and classifies rsync `191/191` correctly.
- Across all four Stage 14C-4 trajectories, 24 significant validation drops were found; `0` occurred at reset epochs and `0` occurred after epoch 50.
- Prototype resets produced norm jumps, but reset-epoch Macro-F1 changes were small/mixed and recovered; no NaN, crash, or gradient explosion was observed.
- Final conclusion: `PIPELINE_READY_WITH_KNOWN_TRANSIENT`.

## Preserved evidence

- `stage14c5_report.md`: full interpretation and required answers.
- `protocol_class_composition.csv`, `per_class_comparison.csv`, `nonzero_confusion_pairs.csv`: protocol and class-level evidence.
- `validation_drop_events.csv`, `replay_drop_telemetry.csv`, `replay_reset_window.csv`, `per_class_drop_recall.csv`: historical and replay dynamics.
- `telemetry_replay/*/epoch_telemetry.csv`, `per_class_validation.csv`, `parity.csv`, `status.json`: raw observational replay evidence.
- Tmux replay log/status: `../.tmux-task/stage14c5_gpu4_replay/output.log` and `exit.status` relative to the project root.

## Limitations

- Scheduler LR reductions and prototype resets co-occur at epochs 51/81, so their small immediate effects cannot be experimentally separated without changing configuration, which this audit forbids.
- Per-epoch norm and per-class telemetry did not exist in the historical artifacts; exact-config replay supplies observational evidence but does not change official checkpoint selection.
- This is a two-protocol diagnosis, not evidence for all 15 frozen protocols.

## Conclusion and next step

The cleaned training pipeline is ready to freeze with a documented high-LR validation transient. The Medium-2025/2026 gap is predominantly protocol-composition difficulty, especially the rsync/scp boundary, not random seed. This task did not start the formal 15-run rerun.
