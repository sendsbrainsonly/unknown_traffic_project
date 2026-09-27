# Experiment results: stage35-joint-vs-staged-ustc-20260925-v1

- Status: `paused / baseline-dependent queue stopped before training`
- Experiment type: `ablation`
- Claim scope: `diagnostic`
- Created (UTC): `2026-09-25T13:40:17Z`
- Objective: Compare feature-level three-view joint encoder training with the current separately trained encoder fusion on identical frozen USTC flows.

## Data and split

- USTC A-2 17-class seed-2022 Stage 34 deterministic 10% sample: 34,665 Known Train, 4,333 Known Validation, 4,333 Known Test. Known Train/Val ID, cache, source and official-weight hashes passed preflight; sample-manifest SHA256 `73f7eecef25ee91015d557f7f9f8e7eecd7560cd995c5cc7ec21c9390d45246d`. No Test feature values were loaded.

## Configuration and execution

- See `EXPERIMENT_PLAN.md`. Stage 34 staged reference continues independently. Stage 35 joint training has not started. A microbatch-32 three-view forward/backward smoke passed at 15.57 GiB allocated peak GPU memory; all four optimizer groups (TrafficFormer, graph, YaTC, fusion) received finite nonzero gradients and stepped. These are implementation checks, not classification results. The named Stage35 queue waits for the Stage34 baseline to finish before training/evaluation, respecting the project three-GPU ceiling.

## Core results

- Pending. No Stage 35 Known Test values have been loaded and no joint checkpoint has been selected.

## Preserved evidence

- `manifest.json`, `EXPERIMENT_PLAN.md`, `preflight.json`, `smoke_batch32/result.json`, `optimizer_smoke_batch32/result.json`, and project-local named tmux logs. The initial preflight failure due to checking each role against the combined Train/Val cache was preserved in `stage35_preflight` logs; the role-subset correction passed in `stage35_preflight_v2`. An initial smoke launcher path typo was preserved in `stage35_smoke_batch32` logs; the corrected smoke passed in `stage35_smoke_batch32_v2`.

## Limitations

- One USTC seed and a practical 20-epoch joint versus 20/50/200+30+30 staged recipe; training cost and optimization schedule differ.

## Conclusion and next step

- Wait for the frozen Stage 34 baseline, then run the fixed 20-epoch joint training and one-shot matched Known Test evaluation. Do not infer a winner from smoke checks or branch-only validation metrics.

## Queue dependency update (2026-09-25 15:20 UTC)

- At the user's request, Stage34 released GPU 2 and moved to a two-GPU continuation. The original waiting `stage35_joint_queue` controller was stopped with exit code 143 before joint training began; its log remains preserved.
- Replacement `stage35_joint_queue_two_gpu` waits for `stage34_two_gpu_resume` and `stage34_two_gpu_finalizer`, then selects only physical GPU 0 or 1 using live capacity. The frozen USTC samples, Stage35 model, training budget, and checkpoint rule are unchanged. Stage35 still has no joint checkpoint or Test result.

## Dependency stop (2026-09-26 03:27 UTC)

- The user stopped the old CIC Stage34 continuation to run an isolated balanced CIC experiment. The Stage35 controller `stage35_joint_queue_two_gpu` was closed before its dependency could fail or it could launch any training. Its preflight and smoke evidence remain intact; no Stage35 joint training/Test result exists. Restarting Stage35 would require a new explicit baseline dependency decision.
