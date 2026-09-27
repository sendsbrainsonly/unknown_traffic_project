# Stage 37 — VPN/Tor single-seed three-view joint training

- Status: `success / diagnostic`, completed 2026-09-26 02:22 UTC; named GPU2 tmux session exited 0. Both datasets completed 20/20 training epochs and one-shot Known Test evaluation. Physical GPU2 was idle at the final progress check.
- Objective: on the exact Stage20 coarse-Service flows, test end-to-end joint training of TrafficFormer, FIG-TAGCN and YaTC features against the previous separately trained Stage32 three-view equal-feature fusion.

## Data and split

- Data: ISCX-VPN four classes, Known Train/Validation/Test 8,764/1,098/1,093; ISCXTor five classes, 8,946/1,118/1,117. Frozen Stage20 manifest SHA256 `6404d496f50a6f7f6f397d125ad0cd653c9df152d8db1b9ed2ad2310967950bb`. Raw three-view input IDs and labels passed exact alignment checks.

## Configuration and execution

- Configuration: seed 2022, 20 epochs, effective batch 64 / microbatch 32, official TrafficFormer and YaTC initial weights, FIG-TAGCN, three 64-dimensional adapters, fixed equal feature fusion and fused cross-entropy. Only Known Validation Macro-F1 selected checkpoints; Unknown data was not used. Test was evaluated after each checkpoint was frozen.

## Core results

| Dataset | Best epoch | Val Macro-F1 | Test Accuracy | Test Macro-F1 | Test Weighted-F1 | Stage32 staged Test Macro-F1 | Joint − staged |
|---|---:|---:|---:|---:|---:|---:|---:|
| ISCX-VPN | 15 | 0.882000 | 0.878317 | 0.882180 | 0.875811 | 0.919350 | −0.037171 |
| ISCXTor | 12 | 0.852184 | 0.847807 | 0.853184 | 0.850408 | 0.899375 | −0.046192 |

The Stage32 comparison uses exactly the same Known Test flow IDs, coarse labels and seed; independent replay verified all 2,210 sample identities, both confusion matrices, all headline metrics and both joint checkpoint SHA256 values. Joint-only versus staged-only correct Test decisions: VPN 21 versus 62; Tor 55 versus 109. On VPN, joint File-Transfer recall was 0.635 versus staged 0.750; on Tor, joint Browsing recall was 0.795 versus 0.910 and Communication recall 0.799 versus 0.890.

## Preserved evidence

- `runs/<dataset>/seed2022_joint_e2e20/`: config, Known-Train scalers, full epoch history, best checkpoint, validation metrics, Test logits, confusion matrix, per-flow predictions, SHA256-bearing summaries and PASS markers.
- `paired_comparison.csv`, `completion_verification.json`, `queue_progress.json`, `manifest.json`.
- Named execution log/status: `../.tmux-task/stage37_gpu2_joint_0926/output.log` and `exit.status`.

## Limitations

The tested **20-epoch joint recipe is worse on both development Test tasks**; this does not prove that joint training is inherently inferior. Stage32 separately trained its branches and fusion head with different, generally longer schedules, so optimization budget is confounded. Both tasks use weak capture-derived Service labels, one seed, and previously exposed Test sets. Very low final train loss with lower validation Macro-F1 is consistent with overfitting, but this run alone cannot isolate the cause. This is closed-set evidence only; it says nothing about Unknown detection.

## Conclusion and next step

Retain Stage32 staged fusion as the current stronger closed-set result on these two tasks. No follow-up tuning or open-set experiment was started by Stage37.
