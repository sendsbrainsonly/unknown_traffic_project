# Experiment results: stage34_ustc_cic_strict_three_view_20260925

- Status: `interrupted_by_user / USTC complete, CIC incomplete`
- Experiment type: `benchmark`
- Claim scope: `diagnostic`
- Created (UTC): `2026-09-25T11:57:05Z`
- Objective: USTC A-2 and CIC-IDS-2017 strict three-class closed-set three-view equal fusion tests

## Data and split

- USTC-TFC2016: frozen A-2 Known 17-class Train/Validation/Test membership, then deterministic seed-2022 SHA256 10% per class and split: 34,665 / 4,333 / 4,333 flows. Unknown Geodo/Htbot/Tinba are excluded.
- CIC-IDS-2017: `BENIGN`, `DoS Slowhttptest`, `PortScan` from the prior strict matched-flow audit. Fixed five-minute `(source_pcap, floor(flow_start_epoch_utc/300))` groups are disjoint across Train/Validation/Test. The first metadata-only 10%-all-classes draft would leave 13 Slowhttptest Test flows; it is preserved but rejected before packet extraction or training. Final v2 keeps all 5,096 Slowhttptest flows and 10% of each large class. v2 Train/Validation/Test counts: BENIGN 133,189/16,849/16,793; Slowhttptest 3,909/1,055/132; PortScan 13,849/901/1,136.

## Configuration and execution

- Reuse Stage31's three independent feature branches and T0 equal fusion: TrafficFormer 20 epochs, FIG/TAGCN 50 epochs, YaTC 200 epochs; adapter and head 30+30 epochs. Seed 2022, Known Train-only normalization; checkpoint selection uses Known Validation only. No Unknown or Test fitting.
- USTC Train/Val input caches were subset from the exact Stage31 A-2 caches. CIC Train/Val packet inputs are being recovered from original read-only PCAPs via frozen mapped flow boundaries. Test packet values are locked until all branch and fusion checkpoint hashes pass.
- Named project-local tmux queue `stage34_bounded_queue` is active and owns at most three physical GPUs (0/1/2 selected live per job); current independent GPU3 process is untouched. Aggregate state: `queue_progress.json`.

## Core results

- Pending final Known Test evaluations. Branch validation metrics are progress diagnostics, not final closed-set results.

## Preserved evidence

- `manifest.json`, USTC/CIC frozen sample manifests and audits, both CIC protocol versions, group assignments, named tmux logs/statuses, branch caches/checkpoints/training histories as they complete, `queue_progress.json`.

## Limitations

- Different tasks: USTC 17 classes vs CIC 3 classes; scores must not be treated as equal-difficulty comparisons.
- CIC's two attack labels each occur in one day and PCAP. Group-disjoint five-minute windows reduce direct capture-block leakage but do not eliminate class/day or endpoint shortcuts. Slowhttptest Test has only one group and 132 flows; class-level uncertainty remains high.
- The reused TrafficFormer raw-byte input retains network-header content, and the graph retains packet lengths/timing; high CIC scores could exploit endpoint/capture artifacts rather than transferable attack behavior.
- These datasets and task choices are development diagnostics, not untouched external validation.

## Conclusion and next step

- In progress. The bounded queue will train/freeze USTC, evaluate its Known Test once, then train/freeze CIC and evaluate its Known Test once. A final report is not valid until `completion_verification.json` passes.

## GPU 2 release and two-GPU continuation (2026-09-25 15:20 UTC)

- User requested physical GPU 2 be released and all subsequent work use at most GPUs 0/1. The CIC YaTC attempt on GPU 2 was stopped at epoch 2/200; its checkpoint, training history, config, project-local tmux log and exit code 143 are preserved in `cicids2017/runs/ustc/A-2/yatc_interrupted_gpu2_20260925/` and `.tmux-task/stage34_cic_yatc/`. It was not promoted as a finished branch.
- Original queue/finalizer controllers were deliberately stopped with exit code 143 before their three-GPU dependency could mark the experiment complete or launch further work. USTC Test had independently completed with 4,333 flows and Accuracy/Macro-F1/Weighted-F1 `0.984306/0.987649/0.984315`; the Stage34 two-dataset bundle is still in progress.
- `run_queue_two_gpu_resume.py` is the continuation. It keeps the original CIC TrafficFormer/graph processes on GPUs 0/1, restarts unchanged YaTC training from official initialization on the first free card, then runs fusion and CIC Test on GPUs 0/1 only. No data split, model, optimizer, epoch budget, or Test selection rule changed. The old partial YaTC run cannot be resumed exactly because it did not save optimizer state.
- Live queue: `stage34_two_gpu_resume`; finalizer: `stage34_two_gpu_finalizer`; aggregate state: `queue_progress.json`. Stage35 now waits for those sessions. Final CIC results and full bundle validation remain pending.

## User-directed stop (2026-09-26 03:27 UTC)

- The user requested stopping the old CIC attempt and immediately testing a new benign/malicious 1:1 sample. Exact sessions `stage34_two_gpu_resume`, `stage34_two_gpu_finalizer`, and `stage34_cic_trafficformer` were closed; the last CIC TrafficFormer log completed epoch 18/20 with Validation Macro-F1 `0.315046` after an epoch-1 best `0.998628`.
- CIC graph (50/50) and YaTC (200/200) completed earlier and remain preserved. No CIC fusion checkpoint, CIC Test cache, CIC Test predictions or final CIC metric was generated. The completed USTC Known Test result remains `0.984306/0.987649/0.984315` for Accuracy/Macro-F1/Weighted-F1.
- The tmux helper's `close` operation removed the sessions without creating `exit.status`; closure and GPU0/1 release are independently recorded in `.tmux-task/cic_balance_stop_verify_0926/output.log`. This is an interruption, not an experiment failure or a valid CIC final score. Replacement experiment: `../stage34b_cic_balanced_retest/`.
