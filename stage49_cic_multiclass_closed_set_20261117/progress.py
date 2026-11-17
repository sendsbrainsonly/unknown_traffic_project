#!/usr/bin/env python3
"""Read-only total task/epoch progress; safe inside watch through tmux."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
path = ROOT / "queue_progress.json"
if not path.exists():
    print("Stage49: queue not started")
    raise SystemExit(0)
state = json.loads(path.read_text())
print(f"Stage49 CIC11 | {state['status']} | {state['phase']} | {state['updated_at_utc']}")
protocol_path = ROOT / "protocol.json"
if protocol_path.exists():
    config = json.loads(protocol_path.read_text())
    print(f"seed={config['seed']} classes={len(config['classes'])} splits={config['split_counts']}")
print("job                 state       gpu    last epoch / Validation Macro-F1")
for name, job in state.get("jobs", {}).items():
    log = PROJECT / ".tmux-task" / job["session"] / "output.log"
    status = PROJECT / ".tmux-task" / job["session"] / "exit.status"
    label = f"exit{status.read_text().strip()}" if status.exists() else job["state"]
    last = "-"
    if log.exists():
        # Bounded tail avoids reloading large training logs on every refresh.
        with log.open("rb") as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - 262144))
            lines = stream.read().decode(errors="replace").splitlines()
        for line in lines:
            if not line.startswith("{"):
                continue
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if "epoch" in event:
                score = event.get("val_macro_f1", event.get("val_mean_macro_f1", "-"))
                last = f"{event.get('phase', '')} {event['epoch']} / {score}"
            elif name in ("protocol", "trainval_inputs", "test_inputs") and any(k in event for k in ("phase", "flows_emitted", "status")):
                last = json.dumps(event, ensure_ascii=False)[:180]
    print(f"{name:<19} {label:<11} {str(job.get('physical_gpu_id', '-')):<6} {last}")
complete = ROOT / "completion_verification.json"
if complete.exists():
    print("Final:", complete.read_text().strip())
