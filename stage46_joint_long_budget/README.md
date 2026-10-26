# Stage46: original-initialization joint100epoch budget diagnosis

See PLAN.md for the preregistered configuration and scheduler caveat. Historical Stage32/37/45 assets remain read-only. Single seed2022; VPN4 and Tor5 services; no Unknown or detector work. A maximum of two live-selected GPUs, one dataset each.

## Read-only total progress

```bash
/home/birkenwald/data/SEU-WXY/WXY-SEU/TrafficClassifier/.agents/skills/tmux-task-execution/scripts/tmux_task.sh run "stage46_progress_$(date +%s)" /home/birkenwald/data/SEU-WXY/WXY-SEU/TrafficClassifier/Projects/unknown_traffic_project -- python stage46_joint_long_budget/progress.py
```

Controller (does not launch another task):

```bash
tmux attach -t stage46_train_20261023
```

Detach using Ctrl-b then d; this does not stop training. The detached queue trains both100epoch runs, freezes all six Known-Val-selected checkpoints, evaluates KnownTest, independently replays saved predictions, and validates the bundle. It does not start open-set evaluation or further hyperparameter experiments.

Artifacts: runs/<dataset>/training_history.jsonl, per-epoch validation predictions, three prefix checkpoints, initial_model.pt, four recovery checkpoints, numerical events, source/code hashes, run_results.csv, paired_comparison.csv, per_class_results.csv, historical_references.csv, figures/, stage46_report.md, RESULTS.md, completion_verification.json, manifest.json. Partial/failed evidence is retained; existing run paths refuse overwrite.
