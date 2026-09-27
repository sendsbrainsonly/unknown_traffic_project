#!/usr/bin/env python3
"""Refresh final artifact hashes after the already-running queue exits."""
from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime, timezone

from prepare_balanced import ROOT

PROJECT = ROOT.parent
WORKSPACE = PROJECT.parents[1]
SKILL = WORKSPACE / ".agents/skills/experiment-data-preservation/scripts"
PYTHON = "/home/birkenwald/data/SEU-WXY/conda_envs/2025-10-8-WXY-dgl_py310/bin/python"
STATUS = PROJECT / ".tmux-task/stage34b_balanced_queue_0926/exit.status"


def main() -> None:
    while not STATUS.is_file():
        time.sleep(30)
    code = int(STATUS.read_text().strip())
    path = ROOT / "postqueue_verification.json"
    if code != 0:
        path.write_text(json.dumps({"status": "INCOMPLETE", "queue_exit_code": code,
                                    "reason": "inspect preserved queue_failure.json and worker logs"}, indent=2) + "\n")
        print(json.dumps({"status": "INCOMPLETE", "queue_exit_code": code}))
        return
    completion = json.loads((ROOT / "completion_verification.json").read_text())
    if completion["status"] != "PASS":
        raise RuntimeError("queue exited zero but completion gate not PASS")
    payload = {"status": "CHECKING", "queue_exit_code": code,
               "checked_at_utc": datetime.now(timezone.utc).isoformat(),
               "completion_status": completion["status"]}
    path.write_text(json.dumps(payload, indent=2) + "\n")
    def check() -> None:
        subprocess.run([PYTHON, str(SKILL / "refresh_artifact_manifest.py"), str(ROOT)], check=True, cwd=PROJECT)
        subprocess.run([PYTHON, str(SKILL / "validate_experiment_bundle.py"), str(ROOT), "--verify-hashes"], check=True, cwd=PROJECT)
    check()
    payload["status"] = "PASS"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    check()
    print(json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()
