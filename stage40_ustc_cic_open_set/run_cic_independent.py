#!/usr/bin/env python3
"""Resume only the frozen CIC pilots, independently of the Stage39 queue.

The original queue and its failure evidence are deliberately left untouched.
All model/data/calibration/evaluation logic is reused from run_cic_queue.py.
Worker waits poll tmux status so runs longer than 60 seconds are not mistaken for failures.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import run_cic_queue as original

ROOT = original.ROOT
PROJECT = original.PROJECT
ATTEMPT = ROOT / "cic_resume_20260927"
START = time.monotonic()
FOLDS = original.FOLDS


def save(status: str, fold: str = "", phase: str = "", **extra: object) -> None:
    record = {
        "status": status,
        "fold": fold,
        "phase": phase,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.monotonic() - START, 1),
        "independent_of_stage39": True,
        "maximum_concurrent_gpu_workloads": 1,
        **extra,
    }
    target = ATTEMPT / "progress.json"
    temporary = ATTEMPT / "progress.json.tmp"
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, target)
    print(json.dumps(record, ensure_ascii=False), flush=True)


def run(fold: str, phase: str, args: list[str], *, gpu: bool = False) -> None:
    command = [original.PYTHON, str(ROOT / args[0]), *args[1:]]
    if gpu:
        command = [
            original.PYTHON, str(original.SELECTOR),
            "--allowed", "0,1,3,4,6,7",
            "--min-free-gb", "12", "--max-utilization", "30",
            "--", *command,
        ]
    for attempt in range(1, 121 if gpu else 2):
        name = f"stage40cic_{fold}_{phase}_{attempt}_0927"
        if (PROJECT / ".tmux-task" / name).exists():
            raise FileExistsError(f"Refusing to overwrite prior execution evidence: {name}")
        save("RUNNING", fold, phase, session=name, capacity_attempt=attempt)
        result = subprocess.run(
            [str(original.HELPER), "start", name, str(PROJECT), "--", *command],
            cwd=PROJECT, check=False, capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"could not start CIC worker {name}: {result.stderr or result.stdout}")
        log_path = PROJECT / ".tmux-task" / name / "output.log"
        while True:
            status = subprocess.run([str(original.HELPER), "status", name],
                                    cwd=PROJECT, check=False, capture_output=True, text=True)
            if status.returncode != 0:
                raise RuntimeError(f"cannot read CIC worker status {name}: {status.stdout} {status.stderr}")
            status_line = status.stdout.strip().splitlines()[-1]
            if "state=running" in status_line:
                save("RUNNING", fold, phase, session=name, capacity_attempt=attempt)
                time.sleep(30)
                continue
            if "exit_code=0" in status_line:
                return
            log = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
            if gpu and "No GPU satisfies the declared constraints." in log:
                save("WAITING_FOR_GPU_CAPACITY", fold, phase, failed_session=name,
                     capacity_attempt=attempt)
                time.sleep(60)
                break
            raise RuntimeError(f"CIC worker failed; inspect {name}: {status_line}")
    raise RuntimeError(f"GPU capacity unavailable after 120 attempts: {fold}/{phase}")


def preflight() -> None:
    for fold in FOLDS:
        original.freeze_check(fold)
        for view in ("tf_fig", "yatc_mfr"):
            audit = ROOT / fold / "input_caches/ustc/A-2" / view / "cache_audit.json"
            if not original.marker_pass(audit):
                raise RuntimeError(f"Known-only CIC input cache missing: {audit}")
        for component in ("graph", "trafficformer", "yatc"):
            original.completed_branch(fold, component)
    print(json.dumps({"status": "PASS", "frozen_cic_folds": list(FOLDS),
                      "stage39_gate_used": False}), flush=True)


def wait_for_existing_worker(session: str, fold: str, phase: str) -> None:
    if not (PROJECT / ".tmux-task" / session).is_dir():
        raise FileNotFoundError(f"expected live worker evidence is missing: {session}")
    while True:
        status = subprocess.run([str(original.HELPER), "status", session],
                                cwd=PROJECT, check=False, capture_output=True, text=True)
        if status.returncode != 0:
            raise RuntimeError(f"cannot inspect existing worker {session}: {status.stdout} {status.stderr}")
        status_line = status.stdout.strip().splitlines()[-1]
        if "state=running" in status_line:
            save("WAITING_FOR_EXISTING_WORKER", fold, phase, session=session)
            time.sleep(30)
            continue
        if "exit_code=0" not in status_line:
            raise RuntimeError(f"existing worker failed: {session}: {status_line}")
        if not original.completed_branch(fold, "trafficformer"):
            raise RuntimeError(f"worker exited successfully but branch artifacts are incomplete: {session}")
        return


def execute(attempt_dir: Path, wait_session: str | None) -> None:
    global ATTEMPT
    ATTEMPT = attempt_dir.resolve()
    if not (ATTEMPT / "manifest.json").is_file():
        raise RuntimeError("Initialize the experiment evidence bundle before launching")
    preflight()
    original.save = save
    original.run = run
    completed: list[str] = []
    try:
        save("RUNNING", phase="preflight_complete", completed=completed)
        if wait_session:
            wait_for_existing_worker(wait_session, "unknown_portscan", "trafficformer")
        for fold in FOLDS:
            original.one(fold)
            completed.append(fold)
            save("RUNNING", fold, "fold_complete", completed=completed)
        save("CIC_PILOTS_COMPLETE", phase="all_verified", completed=completed)
    except BaseException as exc:
        failure = {"status": "FAILED", "error": repr(exc),
                   "traceback": traceback.format_exc(),
                   "updated_at_utc": datetime.now(timezone.utc).isoformat()}
        (ATTEMPT / "failure.json").write_text(
            json.dumps(failure, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        save("FAILED", phase="stopped", completed=completed, error=repr(exc))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--attempt-dir", type=Path, default=ATTEMPT)
    parser.add_argument("--wait-session")
    args = parser.parse_args()
    preflight() if args.preflight else execute(args.attempt_dir, args.wait_session)
