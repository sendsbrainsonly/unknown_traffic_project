# Stage46 Resume Attempt 1 — Results

- Status: `complete / TRAINING_AND_INDEPENDENT_REPLAY_PASS`
- Scope: single-seed closed-set known-service budget comparison; 100-epoch Stage46 trajectory continued from complete epoch 75.
- Original interrupted attempt remains preserved. The mid-epoch `failure_state.pt` was forensic only; recovery resumed from `recovery_epoch075.pt` in a separate output directory.

## Recovery and integrity

- CPU recovery preflight: 4/4 checks passed: frozen-source lock, snapshot/config/step match, best Known-Val checkpoint parity against the first 75 rows, and exact epoch-76 DataLoader order replay.
- GPU restore/update smoke: PASS for VPN on physical GPU 6 and Tor on physical GPU 7. Each performed one atomic Known-Train update with finite loss/gradients.
- Both continuation workers completed epochs 76–100 and exited 0. Epoch-85 and epoch-95 recovery snapshots and the epoch-100 selected checkpoints were saved.
- The original Stage46 source lock remained intact. Known Test was evaluated only after all six 20/50/100 checkpoints were frozen. Unknown usage = 0; DES usage = 0.

## Closed-set results

| Dataset | Budget | Best epoch | Validation Accuracy / Macro-F1 / Weighted-F1 | Known Test Accuracy / Macro-F1 / Weighted-F1 |
|---|---:|---:|---:|---:|
| ISCX-VPN | 20 | 6 | 0.877049 / 0.880098 / 0.875882 | 0.878317 / 0.884922 / 0.877149 |
| ISCX-VPN | 50 | 6 | 0.877049 / 0.880098 / 0.875882 | 0.878317 / 0.884922 / 0.877149 |
| ISCX-VPN | 100 | 6 | 0.877049 / 0.880098 / 0.875882 | 0.878317 / 0.884922 / 0.877149 |
| ISCXTor2016 | 20 | 18 | 0.858676 / 0.862831 / 0.859516 | 0.847807 / 0.851394 / 0.848633 |
| ISCXTor2016 | 50 | 36 | 0.865832 / 0.867322 / 0.867253 | 0.847807 / 0.851864 / 0.849235 |
| ISCXTor2016 | 100 | 36 | 0.865832 / 0.867322 / 0.867253 | 0.847807 / 0.851864 / 0.849235 |

The 100-epoch run did not improve over 20 epochs on VPN; on Tor, the gain was already reached at 50 epochs, with no further gain at 100. Longer training is therefore **not consistently better** for this fixed recipe. The selected 100-budget checkpoints are early Known-Val best states (VPN epoch 6; Tor epoch 36), not necessarily final-epoch weights.

Historical same-Test references, with different training budgets/schedules:

- Stage37 original joint-20: VPN 0.878317 / 0.882180 / 0.875811; Tor 0.847807 / 0.853184 / 0.850408.
- Stage32 staged: VPN 0.915828 / 0.919350 / 0.914487; Tor 0.896150 / 0.899375 / 0.896717.

The 100-epoch schedule lengthens warmup/decay compared with historical 20-epoch runs and uses the repaired atomic GradScaler handling; differences from historical runs cannot be attributed to epoch count alone. This is one seed, and historical Test data are development-exposed.

## Training curves

Use `combined/iscx_vpn_training_history.jsonl`, `combined/iscx_tor_training_history.jsonl` and the two plots under `figures/`. These canonical curves combine original epochs 1–75 with resumed epochs 76–100 after exact epoch-order preflight. The parent finalizer also produced curves from the first interrupted attempt's incomplete histories; those root-level curves are not authoritative.

## Artifacts

- Parent Stage46 report: `../stage46_report.md`
- Parent run-level results, paired comparisons, per-class metrics, and completion verification are stored in `../`.
- Recovery implementation, preflight, smoke, continuation histories/checkpoints, combined complete history, and logs are preserved in this attempt directory and the project-local `.tmux-task/` tree.

Final conclusion: `LONGER_BUDGET_NOT_CONSISTENTLY_BETTER`. No follow-up training or open-set evaluation was started.
