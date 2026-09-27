#!/usr/bin/env python3
"""Fail-closed Stage36 continuation after Stage34 and Stage35 finish."""
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
PROGRESS = ROOT / "queue_progress.json"
START = time.monotonic()


def update(status: str, **items: object) -> None:
    payload = {"status": status, "updated_at_utc": datetime.now(timezone.utc).isoformat(),
               "elapsed_seconds": round(time.monotonic()-START, 1), **items}
    temporary = ROOT / "queue_progress.json.tmp"
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, PROGRESS)
    print(json.dumps(payload), flush=True)


def exit_code(session: str) -> int | None:
    path = PROJECT / ".tmux-task" / session / "exit.status"
    return int(path.read_text().strip()) if path.is_file() else None


def wait_for(session: str) -> None:
    update("WAITING", session=session)
    while (code := exit_code(session)) is None:
        time.sleep(30)
    if code != 0:
        raise RuntimeError(f"dependency {session} failed with exit {code}")


def launch(session: str, phase: str, gpu: bool = False) -> None:
    if (PROJECT / ".tmux-task" / session).exists():
        raise FileExistsError(f"will not overwrite previous tmux evidence: {session}")
    command = [PYTHON, str(ROOT / "run_pilot.py"), phase]
    if gpu:
        # GPU2 is deliberately excluded; it remains released before either
        # possible interpretation of the user's 08:00 deadline.
        command = [PYTHON, str(SELECTOR), "--min-free-gb", "15", "--allowed", "0,1",
                   "--", *command]
    subprocess.run([str(HELPER), "start", session, str(PROJECT), "--", *command],
                   cwd=PROJECT, check=True)
    update("RUNNING", session=session, phase=phase, gpu_selector="live 0,1" if gpu else None)
    wait_for(session)


def main() -> None:
    try:
        wait_for("stage34_two_gpu_resume")
        wait_for("stage34_two_gpu_finalizer")
        wait_for("stage35_joint_queue_two_gpu")
        for subdir in ("stage34_ustc_cic_closed_set", "stage35_joint_vs_staged_training"):
            result = json.loads((PROJECT / subdir / "completion_verification.json").read_text())
            if result["status"] != "PASS":
                raise RuntimeError(f"{subdir} verification is not PASS")
        if json.loads((ROOT / "preflight.json").read_text())["status"] != "PASS":
            raise RuntimeError("Stage36 preflight is not PASS")
        if json.loads((ROOT / "parity.json").read_text())["status"] != "PASS":
            raise RuntimeError("Stage36 Known-Val parity is not PASS")
        if json.loads((ROOT / "known_smoke.json").read_text())["status"] != "PASS":
            raise RuntimeError("Stage36 frozen-head Known-Val smoke is not PASS")
        launch("stage36_unknown_extract", "extract")
        launch("stage36_open_set_evaluate", "evaluate", gpu=True)
        launch("stage36_independent_replay", "verify")
        if json.loads((ROOT / "completion_verification.json").read_text())["status"] != "PASS":
            raise RuntimeError("Stage36 metric replay is not PASS")
        subprocess.run([PYTHON, str(ROOT / "finalize_pilot.py")], cwd=PROJECT, check=True)
        update("COMPLETE", phase="BUNDLE_VALIDATED", gpu2_used=False)
    except BaseException as exc:
        (ROOT / "queue_failure.json").write_text(json.dumps({"status": "FAILED",
            "error": repr(exc), "traceback": traceback.format_exc()}, indent=2) + "\n")
        update("FAILED", error=repr(exc))
        raise


if __name__ == "__main__":
    main()
