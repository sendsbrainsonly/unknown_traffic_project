#!/usr/bin/env python3
"""User-authorized immediate Stage36 pilot, bounded to physical GPU2.

This changes scheduling only. The frozen split, checkpoints, scores and
Known-Validation-only calibration in EXPERIMENT_PLAN.md stay unchanged.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
WORKSPACE = PROJECT.parents[1]
HELPER = WORKSPACE / "skills/tmux-task-execution/scripts/tmux_task.sh"
SELECTOR = WORKSPACE / "skills/using-superpowers/scripts/select_gpu.py"
PYTHON = "/home/birkenwald/data/SEU-WXY/conda_envs/2025-10-8-WXY-dgl_py310/bin/python"
DEADLINE = datetime(2026, 9, 26, 7, 45, tzinfo=timezone.utc)
START = time.monotonic()
PROGRESS = ROOT / "queue_progress.json"


def update(status: str, **details: object) -> None:
    value = {"status": status, "mode": "immediate_gpu2", "updated_at_utc":
             datetime.now(timezone.utc).isoformat(), "elapsed_seconds":
             round(time.monotonic() - START, 1), "gpu2_release_deadline_utc":
             DEADLINE.isoformat(), **details}
    temporary = ROOT / "queue_progress.json.tmp"
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, PROGRESS)
    print(json.dumps(value), flush=True)


def exit_code(session: str) -> int | None:
    path = PROJECT / ".tmux-task" / session / "exit.status"
    return int(path.read_text().strip()) if path.is_file() else None


def wait_for(session: str, phase: str) -> None:
    update("RUNNING", phase=phase, session=session)
    while (code := exit_code(session)) is None:
        time.sleep(15)
    if code != 0:
        raise RuntimeError(f"{phase} failed: session {session} exit={code}; preserve log")
    update("PASS", phase=phase, session=session, exit_code=code)


def launch(session: str, phase: str, *, gpu2: bool = False) -> None:
    if (PROJECT / ".tmux-task" / session).exists():
        raise FileExistsError(f"refusing to reuse tmux evidence: {session}")
    command = [PYTHON, str(ROOT / "run_pilot.py"), phase]
    if gpu2:
        remaining = int((DEADLINE - datetime.now(timezone.utc)).total_seconds())
        if remaining < 3600:
            raise RuntimeError("less than 1h before GPU2 release deadline; inference not started")
        command = ["timeout", "--signal=TERM", "--kill-after=30s", str(remaining),
                   PYTHON, str(SELECTOR), "--min-free-gb", "15", "--allowed", "2",
                   "--", *command]
    subprocess.run([str(HELPER), "start", session, str(PROJECT), "--", *command],
                   cwd=PROJECT, check=True)
    wait_for(session, phase)


def main() -> None:
    try:
        for name in ("preflight.json", "parity.json", "known_smoke.json"):
            if json.loads((ROOT / name).read_text())["status"] != "PASS":
                raise RuntimeError(f"Known-only gate not PASS: {name}")
        if (ROOT / "input_caches/unknown_test").exists():
            raise RuntimeError("Unknown cache already exists; no duplicate extraction")
        if (PROJECT / ".tmux-task/stage36_conditional_queue/exit.status").exists():
            raise RuntimeError("old conditional queue has completed; refuse duplicate run")
        if any("run_conditional.py" in line for line in subprocess.check_output(
                ["ps", "-eo", "args"], text=True).splitlines() if "python " in line):
            raise RuntimeError("old conditional queue process is still alive")
        source = json.loads((ROOT / "queue_progress.json").read_text())
        if source.get("status") != "WAITING":
            raise RuntimeError("unexpected old queue state")
        (ROOT / "queue_progress_waiting_before_gpu2_override.json").write_text(
            json.dumps(source, indent=2, ensure_ascii=False) + "\n")
        (ROOT / "launch_override.json").write_text(json.dumps({
            "status": "USER_AUTHORIZED", "change": "scheduling_only",
            "previous": "wait for Stage34/35; use GPUs0/1",
            "current": "start Stage36 now; use only physical GPU2",
            "science_protocol_changed": False, "physical_gpu_ids": [2],
            "gpu2_release_deadline_utc": DEADLINE.isoformat(),
            "known_only_gates": ["preflight", "packet_parity", "frozen_head_parity"],
        }, indent=2, ensure_ascii=False) + "\n")
        update("READY", phase="KNOWN_ONLY_GATES_PASS")
        launch("stage36_unknown_extract_gpu2_now", "extract")
        launch("stage36_open_set_evaluate_gpu2_now", "evaluate", gpu2=True)
        launch("stage36_independent_replay_gpu2_now", "verify")
        if json.loads((ROOT / "completion_verification.json").read_text())["status"] != "PASS":
            raise RuntimeError("independent score replay not PASS")
        subprocess.run([PYTHON, str(ROOT / "finalize_pilot.py")], cwd=PROJECT, check=True)
        update("COMPLETE", phase="BUNDLE_VALIDATED", physical_gpu_ids=[2])
    except BaseException as exc:
        (ROOT / "queue_failure_gpu2_now.json").write_text(json.dumps({
            "status": "FAILED", "error": repr(exc), "traceback": traceback.format_exc(),
            "gpu2_release_deadline_utc": DEADLINE.isoformat(),
        }, indent=2, ensure_ascii=False) + "\n")
        update("FAILED", error=repr(exc))
        raise


if __name__ == "__main__":
    main()
