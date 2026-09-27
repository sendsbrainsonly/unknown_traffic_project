#!/usr/bin/env python3
"""Low-overhead periodic monitor for Stage34's owned sessions only."""
from __future__ import annotations

import json
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
SESSIONS = [
    "stage34_ustc_tf", "stage34_ustc_graph", "stage34_ustc_yatc",
    "stage34_cic_inputs_trainval", "stage34_ustc_fusion", "stage34_ustc_test_inputs",
    "stage34_ustc_test_eval", "stage34_cic_trafficformer", "stage34_cic_graph",
    "stage34_cic_yatc", "stage34_cic_fusion", "stage34_cic_test_inputs",
    "stage34_cic_test_eval", "stage34_bounded_queue", "stage34_finalizer",
]
OUT = ROOT / "live_monitor_status.json"


def capture(session):
    path = PROJECT / ".tmux-task" / session
    status_file, log = path / "exit.status", path / "output.log"
    code = status_file.read_text().strip() if status_file.is_file() else None
    tail = ""
    if log.is_file():
        with log.open("rb") as handle:
            handle.seek(max(0, log.stat().st_size - 12000))
            tail = handle.read().decode("utf-8", errors="replace")
    epochs = re.findall(r'"epoch"\s*:\s*(\d+)', tail)
    days = re.findall(r'"day"\s*:\s*"([A-Za-z]+)"', tail)
    created = path.is_dir()
    return {"session": session, "state": ("not_started" if not created else
            "running" if code is None else "finished" if code == "0" else "failed"),
            "exit_code": code, "latest_epoch": int(epochs[-1]) if epochs else None,
            "latest_day": days[-1] if days else None, "log_bytes": log.stat().st_size if log.is_file() else 0}


def main():
    started = time.time()
    while True:
        items = [capture(name) for name in SESSIONS]
        try:
            result = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                                     "--format=csv,noheader,nounits"], text=True, capture_output=True,
                                    timeout=8, check=True)
            gpu = [line.strip() for line in result.stdout.splitlines()]
        except Exception as exc:
            gpu = [f"unavailable:{type(exc).__name__}"]
        snapshot = {"updated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "elapsed_seconds": int(time.time()-started), "sessions": items, "gpu": gpu}
        OUT.write_text(json.dumps(snapshot, indent=2) + "\n")
        active = [x for x in items if x["state"] == "running"]
        print(json.dumps({"updated_at_utc": snapshot["updated_at_utc"],
                          "active_sessions": [x["session"] for x in active],
                          "latest": {x["session"]: (x["latest_epoch"] if x["latest_epoch"] is not None else x["latest_day"])
                                     for x in active}, "gpu": gpu}), flush=True)
        finalizer = next(x for x in items if x["session"] == "stage34_finalizer")
        if finalizer["state"] in ("finished", "failed"):
            break
        time.sleep(60)


if __name__ == "__main__":
    main()
