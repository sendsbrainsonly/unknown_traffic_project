# Stage 19 — RoNeTC Four-Dataset Fair Comparison

- Experiment ID: `stage19-ronetc-four-dataset-comparison-20260921-v1`
- Status: `ABORTED_BY_USER_WITH_FOUR_COMPLETED_PROTOCOLS`
- Claim scope: approximate runnable reconstruction on frozen project protocols
- Scientific result status: partial comparison; four matched protocols are formal, USTC is excluded

## Resume record

Training was explicitly resumed after the interruption with a strict maximum of three physical GPUs. The four interrupted run directories were preserved under `interrupted_attempts/user_stop_20260921/` and excluded from formal results. A three-GPU USTC smoke test on physical GPUs `0,4,5` passed exact dense-adapter parity and finite-loss checks while preserving the global batch size of 128; peak allocated memory was approximately 11.75/11.74/11.46 GB per GPU. The formal queue starts three independent one-GPU protocols, starts the fourth only after a slot is released, and starts USTC on three GPUs only after all four earlier protocols succeed. Automatic retries are disabled.

## Parallel takeover record

On 2026-09-21 the user authorized use of idle GPUs. The original queue supervisor was replaced because it hard-coded a three-GPU concurrency ceiling and a barrier before USTC. Terminating that supervisor also terminated its active VNAT child at 56/100 epochs; all partial artifacts were preserved under `interrupted_attempts/parallel_takeover_20260921/vnat/medium_seed2026/` with an `INTERRUPTED.json` record. No partial checkpoint is eligible as a formal result.

A new project-local supervisor now cleanly restarts VNAT `medium_seed2026` on physical GPU 0 and concurrently runs USTC `A-2` with DataParallel on physical GPUs 1/2/3. Live steady-state evidence showed approximately 36.95 GB on GPU 0 and 12.92/12.90/12.59 GB on GPUs 1/2/3. The durable status is `queue_status.json`; takeover metadata is `takeover_parallel.json`; the tmux session is `rontec_stage19_parallel_takeover`.

## Independent single-GPU correction

On 2026-09-21 the user clarified that each training process should use one GPU and that the three-card limit is a ceiling, not a requirement to split one batch across all available cards. The four-card attempt was stopped cleanly. Its restarted VNAT run reached 7/100 epochs, while USTC had not completed epoch 1; both partial directories are preserved under `interrupted_attempts/three_gpu_reconfigure_20260921/` and are excluded from formal results.

Before restarting, USTC `A-2` passed a full-batch single-GPU smoke test with batch size 128, no DataParallel, finite forward/backward/optimizer values, exact dense-adapter parity, and peak allocated memory `34,887,070,720` bytes (about 32.49 GiB). The smoke evidence is `smoke_runs/ustc/A-2/smoke_result.json`.

The current formal layout uses two independent processes: VNAT `medium_seed2026` on physical GPU 0 and USTC `A-2` on physical GPU 1. GPUs 2 and 3 remain free; the configured maximum remains three physical GPUs. Both processes showed approximately 36.95 GB live device memory shortly after launch. USTC is not passed `--data-parallel`, so its full global batch of 128 stays on GPU 1. The tmux session is `rontec_stage19_independent_single_gpu`, and unique takeover metadata is `takeover_independent_single_gpu_20260921.json`.

## Three-GPU USTC restart

On 2026-09-21 the user subsequently requested three-GPU execution for the remaining USTC run. The single-GPU USTC attempt was stopped cleanly after 7/100 epochs and preserved under `interrupted_attempts/single_gpu_to_three_gpu_20260921/ustc/A-2/`; it is excluded from formal results. Because its checkpoint did not include optimizer and scheduler state, the three-GPU formal run restarts from epoch 1 rather than presenting an approximate continuation as equivalent training.

The completed VNAT `medium_seed2026` PASS result is reused without retraining. USTC now runs with the same global batch size 128 through `nn.DataParallel` on physical GPUs 0/1/2. Live startup evidence showed approximately 12.92/12.90/12.59 GB and active compute on all three cards. A first three-GPU launch was stopped before epoch 1 solely because monitoring metadata incorrectly counted the completed VNAT GPU as active; its adapter parity artifact and interruption record are preserved under `interrupted_attempts/three_gpu_status_metadata_fix_20260921/`. After correction, `queue_status.json` reports exactly three active physical GPUs and `ustc_data_parallel=true`.

The active tmux session is `rontec_stage19_ustc_three_gpu_final`; takeover metadata is `takeover_ustc_three_gpu_final_20260921.json`; the current log is `queue_logs/ustc_A2_ustc_three_gpu_final_20260921.log`.

## User-requested terminal stop

On 2026-09-21 the user requested release of the current training process and comparison of only the already completed matched protocols. The project-local `STOP_QUEUE` sentinel stopped the USTC child with SIGTERM after epoch 5. The supervisor ended with exit code 1 because the child return code was `-15`; this is an intentional user stop, not an OOM or model failure.

USTC partial validation evidence and its best checkpoint are preserved under `runs/ustc/A-2/`, with the terminal metadata in `runs/ustc/A-2/INTERRUPTED.json`. Known Test and Unknown Test evaluation did not run, so USTC is excluded from every formal comparison. Post-stop live verification found no Stage 19 process, and physical GPUs 0/1/2 were all at 0 MiB. The unrelated process on physical GPU7 remained alive and was not touched. `queue_status.json` retains `active_physical_gpus=3` from the final pre-exit update; this field is stale and must not be used as live GPU evidence.

## Data and split

Run the recovered RoNeTC implementation on the same five representative Stage 15R protocols: ISCX-VPN `medium_seed2022`, ISCXTor `medium_seed2022`, VNAT `medium_seed2025/2026`, and USTC `A-2`. The generated `protocol_manifest.csv` contains 523,029 role-expanded rows and has SHA256 `4947d66c44a04d7b52e155f98c5f71fcb5b1239174d87ddc41ebad41584d76cf`.

All four packet-view caches passed before training. They recover every frozen flow, use the first eight packets and 64 bytes for each of the IP-header, transport-header, and payload views, and contain zero missing flows. Unknown Train/Validation rows are both zero.

## Configuration and execution

- Recovered RoNeTC three-view MobileViT/evidential architecture.
- Fixed input: 8 packets, 64 bytes per view; IPv4/IPv6 endpoint addresses zeroed.
- Adam, learning rate `1e-4`, weight decay `1e-5`, batch size 128, 100 epochs.
- MultiStepLR milestones `[50,80]`, gamma `0.1`; no early stopping.
- Checkpoint selection: Known Validation accuracy only.
- Threshold planned from Known Validation P95 only; no Test data were used for tuning.

## Historical initial interruption record

The user requested termination of all owned GPU processes on 2026-09-21. All four active Stage 19 runs exited with code 143. Their histories and best-so-far checkpoints are preserved in place and explicitly marked `INTERRUPTED.json`.

| Dataset / protocol | Epochs completed | Best epoch so far | Best Val Accuracy | Val Macro-F1 at best-accuracy epoch | Checkpoint SHA256 |
|---|---:|---:|---:|---:|---|
| ISCXTor `medium_seed2022` | 35/100 | 33 | 0.642276 | 0.565634 | `e05491176cf7f8a7e577fae8d763194d07cccef81e32e1cd948a0031e0ed79a2` |
| ISCX-VPN `medium_seed2022` | 24/100 | 24 | 0.601490 | 0.443159 | `38ce5dbee8a969f0b18f76836ac4facf4c8c4a430bccef64ebd2b75f5e7c38c2` |
| VNAT `medium_seed2025` | 2/100 | 2 | 0.846429 | 0.324889 | `9574916f240937d13b5d0b3506b109c899d75a6d5176467eafb275f715e99654` |
| VNAT `medium_seed2026` | 2/100 | 2 | 0.869702 | 0.484210 | `fb627413986dab58677a1f4e13ba83ec68c742de7d66fa1ea18612f6b4a2d698` |
| USTC `A-2` | 0/100 formal | — | — | — | formal run not started |

These values are interim optimization traces only. Known Test and Unknown Test evaluation did not run, so no RoNeTC open-set AUROC/AUPRC/UFAR or final closed-set comparison is available.

## Core results

The current project method is represented by Stage 15B H1, `max(A_OD, A_DES0)`, using Known-Validation-only calibration. Protocol IDs, Known/Unknown classes, split membership, and sample counts were checked before comparison.

| Protocol | RoNeTC AUROC | H1 AUROC | RoNeTC AUPRC | H1 AUPRC | RoNeTC UFAR | H1 UFAR | RoNeTC FRR | H1 FRR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ISCXTor medium-2022 | **0.609545** | 0.604153 | **0.813783** | 0.796046 | **0.949000** | 0.981500 | **0.052111** | 0.055705 |
| ISCX-VPN medium-2022 | 0.583144 | **0.587177** | **0.812776** | 0.810527 | **0.947833** | 0.959000 | 0.058606 | **0.057372** |
| VNAT medium-2025 | **0.864428** | 0.806226 | **0.885988** | 0.871320 | 0.767059 | **0.662222** | 0.062755 | **0.038265** |
| VNAT medium-2026 | 0.639538 | **0.759437** | 0.754789 | **0.809419** | **0.881833** | 0.936264 | 0.047619 | **0.046049** |
| Unweighted mean | 0.674164 | **0.689248** | 0.816834 | **0.821828** | 0.886431 | **0.884747** | 0.055273 | **0.049348** |

AUROC/AUPRC are higher-is-better; UFAR/FRR are lower-is-better. H1 leads the four-protocol unweighted mean by `0.015084` AUROC and `0.004994` AUPRC, while reducing mean UFAR by `0.001685` and Known FRR by `0.005925`. AUROC wins are tied 2–2, and RoNeTC varies sharply between the two VNAT protocols; this is therefore a partial matched-protocol conclusion, not a complete four-dataset result.

Machine-readable and human-readable comparison artifacts are `completed_protocol_comparison.json` and `completed_protocol_comparison.md`.

## Preserved evidence

- Stop command log: `.tmux-task/stop_all_user_gpu_20260921/output.log`
- Post-stop verification: `.tmux-task/verify_user_gpu_stop_20260921/output.log`
- Interrupted training logs: `.tmux-task/rontec_train_iscx_tor/`, `.tmux-task/rontec_train_iscx_vpn/`, `.tmux-task/rontec_train_vnat_2025/`, `.tmux-task/rontec_train_vnat_2026/`
- Historical first-stop verification recorded current-user GPU process count `0` at that earlier time; it is not the current live state.
- Final USTC stop evidence: `runs/ustc/A-2/INTERRUPTED.json`
- Final USTC tmux log/status: `.tmux-task/rontec_stage19_ustc_three_gpu_final/output.log` and `exit.status`
- Completed-protocol comparison: `completed_protocol_comparison.md` and `completed_protocol_comparison.json`
- Final release verification: `.tmux-task/rontec_stop_verify1/output.log`

## Limitations

Stage 19 is not a complete four-dataset benchmark because USTC was stopped at 5/100 epochs and never reached Test evaluation. The comparison is valid only for the four listed protocol runs that completed 100/100 epochs. H1 remains a `HYBRID_PARTIAL` project candidate rather than a universally validated final detector.

## Conclusion and next step

The requested training release is complete. Four finished matched protocols support a partial conclusion: the project H1 method has better unweighted mean AUROC, AUPRC, UFAR, and Known FRR than RoNeTC, while protocol-level AUROC wins are tied 2–2. USTC remains intentionally incomplete and excluded; any complete four-dataset conclusion requires a new clean USTC run from epoch 1 under an explicitly approved continuation.
