# Experiment results: stage31-four-dataset-three-view-equal-20260925-v1

- Status: `aborted / superseded_by_stage32_scope_change`
- Experiment type: `benchmark`
- Claim scope: `diagnostic`
- Created (UTC): `2026-09-25T07:17:17Z`
- Objective: Retest equal-weight three-view closed-set classifier on four frozen dataset protocols

## Data and split

- Five previously frozen Stage15R representative protocols cover four datasets: ISCX-VPN medium-2022, ISCXTor medium-2022, VNAT Medium-2025/2026, USTC A-2. No new split or Unknown-class selection.
- Train/Val/Test metadata counts are in `input_readiness.csv`; Stage31 has not opened Test feature arrays.

## Configuration and execution

- Frozen method and execution rules: `EXPERIMENT_PLAN.md`. Three independent TrafficFormer, FIG/TAGCN, and YaTC features; Stage30 T0 fixed 1/3 weights after Gaussian adapters.
- Metadata preflight: `INPUT_GENERATION_REQUIRED`. The prior Stage30 features belong to different Stage20 ISCX tasks and cannot be silently reused.
- First-five-packet YaTC MFR Known Train/Val input reconstruction passed full coverage for VPN (14,521), Tor (9,983), VNAT Medium-2025 (17,664), VNAT Medium-2026 (17,239), and USTC A-2 (389,982). VPN/Tor byte-for-byte parity with existing Stage23 MFR on 7,040/6,081 overlapping flows: 0 mismatches.
- VNAT Medium-2025 recovery attempt 1 failed on one Stage14A `other:` IP flow because the initial extractor handled TCP/UDP streams only. Attempt 2 found that flow but failed on IPv6 MFR. Both failed caches and tmux logs are preserved. Attempt 3 passed all 17,664 flows with an explicit IPv6 input adaptation on 2 flows / 10 selected packets; Medium-2026 needed no IPv6 adaptation. This is not strict official YaTC input parity for those two flows.
- VNAT Medium-2025 TrafficFormer/FIG reconstruction passed exact same-ID coverage (17,664/17,664) against the YaTC MFR cache. One-batch TrafficFormer/FIG/YaTC smokes passed; the initial VNAT YaTC smoke failed only on `train` versus `known_train` cache filename mapping and remains preserved before a successful second smoke.
- Formal VNAT Medium-2025 TrafficFormer, FIG/TAGCN, and YaTC branch training started on physical GPUs 0, 1, and 2 after the input gate. The FIG/TAGCN 50-epoch branch has completed; the other two are running. All formal metrics remain Known Validation only. No T0 adapter/head or Test evaluation has run yet.
- USTC A-2 and VNAT Medium-2026 TrafficFormer/FIG Known Train/Val reconstruction now also pass exact same-flow coverage; ISCX-VPN/Tor remain in progress. USTC A-2 graph branch passed a bounded smoke and entered formal 50-epoch training on GPU 1 after the VNAT graph branch completed, maintaining a maximum of three concurrent GPUs.
- Stage31's T0 model copy passed direct numerical parity against immutable Stage30: initialized adapter/head state, adapter training loss, logits, and equal weights all match exactly. `architecture_parity.json` records the check.
- `progress.py` writes one-shot `progress.json` and prints input/branch/head/Test completion; current status must be refreshed before interpretation.
- A bounded queue is running in `codex_stage31_bounded_queue_20260925`: fixed physical GPU lanes 0/1/2, one Stage31 job per lane, fail-closed on partial/failing input or training. It will finish all 15 branch models and five T0 heads, then stop before Test. Live state is `queue_progress.json`.
- Known Test builders are implemented but have an enforced lock: they refuse to materialize any Known Test feature until all five T0 checkpoints exist and their recorded hashes match. Lock rejection was tested while training is incomplete.
- ISCX-VPN and ISCXTor TrafficFormer/FIG Known Train/Val input reconstruction completed with exact flow-ID coverage. All five protocol cells now have all three Known Train/Val views aligned (5/5). Live check at 2026-09-25 10:37 UTC: 11/15 branches completed. Active: ISCXTor TrafficFormer epoch 2/20 (Known-Val Macro-F1 0.868794) and USTC A-2 YaTC epoch 8/200. USTC YaTC has 5,416 batches per epoch; at the observed ~0.045 s/batch, it is the long pole. These are training progress observations, not final results.
- A separate named finalizer `codex_stage31_finalize_v2_20260925` is waiting for the queue's all-heads-frozen state. It will verify frozen source hashes, build only Known Test inputs, evaluate all five T0 heads without Test-based selection, replay saved predictions/logits, and validate the final bundle. Its fail-closed state is `finalization_progress.json`. Static command/cache mapping, syntax, and source-hash preflight passed before launch. The first waiting-only finalizer was closed and replaced by v2 after adding queue-liveness protection; neither version had opened Test input.

## Core results

- No Stage31 closed-set model metric yet. Input preparation is not model testing.

## Preserved evidence

- `manifest.json`, `EXPERIMENT_PLAN.md`, `input_readiness.csv`, `preflight_summary.json`, source hashes, all successful/failed Known-only MFR attempts, the YaTC smoke result, and named tmux logs.

## Limitations

- No same-protocol aligned three-view features were available at start; all three feature sources must be generated/retrained for these frozen protocols. USTC A-2 alone contains 346,651 Known Train flows, so full training is materially larger than Stage30.
- This is a five-protocol pilot, not a full VNAT/USTC protocol sweep. No Test or open-set claim may be made until all requested cells finish and are replayed.

## Conclusion and next step

- Finish exact input alignment for all five protocols, then perform Known-only branch and T0 training; after checkpoint freeze, run Known Test evaluation and independent replay.

## Terminal scope-change correction — 2026-09-25

- The preceding progress paragraphs are historical snapshots, **not current running state**. User replaced the four-dataset fine-application pilot with Stage 32's three-dataset coarse single-seed plan. The bounded queue, waiting finalizer, and two active branch jobs were stopped; a subsequent process check found no remaining Stage31 queue/finalizer/branch-training workers.
- At cancellation: five input cells were ready; 11/15 branch trainings had completed, while T0 heads were 0/5 and Known Test evaluations were 0/5. Completed branch artifacts, interrupted attempts and named tmux logs were retained. No Stage31 four-dataset model result exists and Stage31 must not be resumed automatically.
- New, separate work is tracked in `../stage32_three_dataset_coarse_single_seed/`; its coarse labels and Stage20 ISCX flows are different from Stage31's fine-application pilot.
