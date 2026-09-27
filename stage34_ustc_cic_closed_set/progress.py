#!/usr/bin/env python3
"""One-shot read-only Stage34 aggregate progress for the user's terminal."""
import json
import re
from pathlib import Path

root = Path(__file__).resolve().parent
path = root / "queue_progress.json"
if not path.is_file():
    print("Stage34 queue not started")
else:
    status = json.loads(path.read_text())
    print(f"Stage34: {status['status']} / {status.get('stage', 'unknown')}")
    print(f"Updated UTC: {status.get('updated_at_utc', 'unknown')}")
    sessions = ["stage34_ustc_tf", "stage34_ustc_graph", "stage34_ustc_yatc",
                "stage34_cic_inputs_trainval", "stage34_bounded_queue",
                *status.get("sessions", [])]
    for session in dict.fromkeys(sessions):
        file = root.parent / ".tmux-task" / session / "exit.status"
        log = root.parent / ".tmux-task" / session / "output.log"
        code = file.read_text().strip() if file.is_file() else "running"
        progress = ""
        if log.is_file():
            with log.open("rb") as handle:
                handle.seek(max(0, log.stat().st_size - 16000))
                tail = handle.read().decode("utf-8", errors="replace")
            epochs = re.findall(r'"epoch"\s*:\s*(\d+)', tail)
            if epochs:
                progress = f"; latest epoch={epochs[-1]}"
            day = re.findall(r'"day"\s*:\s*"([A-Za-z]+)"', tail)
            if day:
                progress += f"; latest capture day={day[-1]}"
        print(f"  {session}: {code}{progress}; log={log}")
    for label, file in (("USTC A-2", root / "runs/ustc/A-2/known_test_evaluation/results.json"),
                        ("CIC strict 3", root / "cicids2017/runs/ustc/A-2/known_test_evaluation/results.json")):
        if file.is_file():
            value = json.loads(file.read_text())
            print(f"  {label}: n={value['known_test_samples']}, Acc={value['accuracy']:.6f}, "
                  f"Macro-F1={value['macro_f1']:.6f}, Weighted-F1={value['weighted_f1']:.6f}")
        else:
            print(f"  {label}: Test result pending")
