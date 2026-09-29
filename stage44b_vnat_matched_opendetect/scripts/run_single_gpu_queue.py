#!/usr/bin/env python3
"""Train and evaluate the four frozen Native baselines serially on one GPU."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

from matched_common import FOLDS, PROJECT, ROOT, role_rows, sha256, verify_freeze


PROGRESS = ROOT / "queue_progress.json"


def save_progress(value: dict[str, object]) -> None:
    temporary = PROGRESS.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(PROGRESS)


def run(script: str, *args: str) -> None:
    command = [sys.executable, str(Path(__file__).with_name(script)), *args]
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"{script} {args} exited {completed.returncode}")


def main() -> None:
    before = verify_freeze()
    source_paths = (
        PROJECT.parent / "Open-Detect/code/model.py",
        PROJECT.parent / "Open-Detect/code/utils.py",
        PROJECT.parent / "Open-Detect/reproduction/run_reproduction.py",
        PROJECT / "stage14c_native_opendetect_vnat/scripts/train_one.py",
        Path(__file__).with_name("matched_common.py"),
        Path(__file__).with_name("prepare_inputs.py"),
        Path(__file__).with_name("train_fold.py"),
        Path(__file__).with_name("evaluate_fold.py"),
        Path(__file__).with_name("aggregate_results.py"),
        Path(__file__),
    )
    source_lock = {str(path): sha256(path) for path in source_paths}
    (ROOT / "source_lock.json").write_text(json.dumps(source_lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for fold in FOLDS:
        role_rows(fold)
        audit = json.loads((ROOT / "inputs" / fold / "input_audit.json").read_text(encoding="utf-8"))
        if audit["status"] != "PASS" or audit["frozen_inputs"] != before:
            raise RuntimeError(f"{fold}: Known-only input audit failed")
    progress: dict[str, object] = {
        "status": "RUNNING", "folds": 4, "total_components": 8, "completed_components": [],
        "current": None, "visible_gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "UNSET"),
        "started_at_unix": time.time(), "frozen_source": before,
    }
    save_progress(progress)
    try:
        for fold in FOLDS:
            for component, script, arguments in (
                ("training", "train_fold.py", ("--protocol-id", fold, "--device", "cuda:0")),
                ("evaluation", "evaluate_fold.py", ("--fold", fold)),
            ):
                progress["current"] = {"fold": fold, "component": component}
                progress["updated_at_unix"] = time.time()
                save_progress(progress)
                print(json.dumps({"status": "START", "fold": fold, "component": component}), flush=True)
                started = time.time()
                run(script, *arguments)
                if {str(path): sha256(path) for path in source_paths} != source_lock:
                    raise RuntimeError("Native or Stage 44B source code changed during execution")
                progress["completed_components"].append({
                    "fold": fold, "component": component, "status": "PASS",
                    "elapsed_seconds": time.time() - started,
                })
                progress["updated_at_unix"] = time.time()
                save_progress(progress)
                print(json.dumps({"status": "PASS", "fold": fold, "component": component}), flush=True)
        progress["current"] = {"component": "aggregation"}
        save_progress(progress)
        run("aggregate_results.py")
        if verify_freeze() != before or {str(path): sha256(path) for path in source_paths} != source_lock:
            raise RuntimeError("frozen input or code changed during matched Open-Detect queue")
        progress["status"] = "COMPLETE"
        progress["current"] = None
        progress["finished_at_unix"] = time.time()
        save_progress(progress)
        print(json.dumps({"status": "COMPLETE", "components": len(progress["completed_components"])}), flush=True)
    except Exception as exc:
        failure = {"status": "FAILED", "exception": repr(exc), "traceback": traceback.format_exc(),
                   "current": progress["current"], "completed_components": progress["completed_components"]}
        (ROOT / "queue_failure.json").write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        progress["status"] = "FAILED"
        progress["updated_at_unix"] = time.time()
        save_progress(progress)
        raise


if __name__ == "__main__":
    main()
