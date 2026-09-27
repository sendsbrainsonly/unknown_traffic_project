#!/usr/bin/env python3
"""Fail-closed Stage35 continuation after the independent Stage34 queue finishes."""
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
STAGE34 = PROJECT / "stage34_ustc_cic_closed_set"
HELPER = WORKSPACE / "skills/tmux-task-execution/scripts/tmux_task.sh"
SELECTOR = WORKSPACE / "skills/using-superpowers/scripts/select_gpu.py"
PYTHON = "/home/birkenwald/data/SEU-WXY/conda_envs/2025-10-8-WXY-dgl_py310/bin/python"
PROGRESS = ROOT / "queue_progress.json"
START = time.monotonic()
STAGE34_QUEUE_SESSION = os.environ.get("STAGE34_QUEUE_SESSION", "stage34_bounded_queue")
STAGE34_FINALIZER_SESSION = os.environ.get("STAGE34_FINALIZER_SESSION", "stage34_finalizer")


def update(state: str, **details) -> None:
    value = {"status": state, "updated_at_utc": datetime.now(timezone.utc).isoformat(),
             "elapsed_seconds": round(time.monotonic() - START, 1), **details}
    temporary = ROOT / "queue_progress.json.tmp"
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, PROGRESS)
    print(json.dumps(value), flush=True)


def exit_status(session: str) -> int | None:
    path = PROJECT / ".tmux-task" / session / "exit.status"
    return int(path.read_text().strip()) if path.is_file() else None


def wait_for(session: str, phase: str) -> None:
    update("WAITING", phase=phase, session=session)
    while (code := exit_status(session)) is None:
        time.sleep(30)
    if code != 0:
        raise RuntimeError(f"{phase}: {session} exited {code}; inspect preserved tmux log")
    update("PASS", phase=phase, session=session, exit_code=code)


def launch(session: str, arguments: list[str], gpu: bool) -> None:
    if (PROJECT / ".tmux-task" / session).exists():
        raise FileExistsError(f"refusing to reuse prior evidence: {session}")
    command = [PYTHON, str(ROOT / "joint_ustc.py"), *arguments]
    if gpu:
        command = [PYTHON, str(SELECTOR), "--min-free-gb", "30", "--allowed", "0,1",
                   "--", *command]
    subprocess.run([str(HELPER), "start", session, str(PROJECT), "--", *command],
                   check=True, cwd=PROJECT)
    update("LAUNCHED", phase=arguments[0], session=session, gpu_selector="live 0,1" if gpu else None)


def main() -> None:
    try:
        if not (ROOT / "optimizer_smoke_batch32/SUCCESS").is_file():
            raise RuntimeError("all-branch optimizer smoke not PASS")
        # Do not add a fourth research GPU while Stage34 is preparing or training CIC.
        wait_for(STAGE34_QUEUE_SESSION, "STAGE34_FROZEN_BASELINE")
        wait_for(STAGE34_FINALIZER_SESSION, "STAGE34_BUNDLE_VERIFICATION")
        completion = json.loads((STAGE34 / "completion_verification.json").read_text())
        if completion["status"] != "PASS":
            raise RuntimeError("Stage34 completion verification not PASS")
        launch("stage35_joint_train", ["train", "--microbatch", "32"], gpu=True)
        wait_for("stage35_joint_train", "JOINT_TRAIN")
        launch("stage35_joint_test", ["evaluate"], gpu=True)
        wait_for("stage35_joint_test", "MATCHED_TEST")
        update("COMPLETE", phase="FINALIZING", comparison_path=str(ROOT / "comparison_summary.csv"))
        subprocess.run([PYTHON, str(ROOT / "finalize.py")], check=True, cwd=PROJECT)
    except BaseException as exc:
        failure = {"status": "FAILED", "error": repr(exc), "traceback": traceback.format_exc(),
                   "elapsed_seconds": round(time.monotonic() - START, 1)}
        (ROOT / "queue_failure.json").write_text(json.dumps(failure, indent=2) + "\n")
        update("FAILED", error=repr(exc))
        raise


if __name__ == "__main__":
    main()
