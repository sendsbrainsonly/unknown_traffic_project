#!/usr/bin/env python3
"""Continue Stage44 from cache recovery through single-GPU training/evaluation."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
PYTHON = Path(sys.executable)
PROGRESS = ROOT / "pipeline_progress.json"
CACHE_EXIT = PROJECT / ".tmux-task/stage44_missing_cache_v2_20260928/exit.status"
SELECTOR = PROJECT.parent.parent / ".agents/skills/using-superpowers/scripts/select_gpu.py"


def atomic_json(value: object) -> None:
    temporary = PROGRESS.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, PROGRESS)


def update(status: str, **extra: object) -> None:
    atomic_json({"status": status, "updated_at_utc": datetime.now(timezone.utc).isoformat(), **extra})


def main() -> None:
    try:
        update("WAITING_FOR_MISSING_KNOWN_TEST_CACHE")
        while not CACHE_EXIT.is_file():
            time.sleep(30)
        cache_code = int(CACHE_EXIT.read_text().strip())
        if cache_code != 0:
            raise RuntimeError(f"missing-cache extractor failed: exit={cache_code}")
        cache_audit = ROOT / "input_caches/missing_known_test/cache_audit.json"
        if json.loads(cache_audit.read_text())["status"] != "PASS":
            raise RuntimeError("missing-cache audit is not PASS")
        update("ASSEMBLING_FULL_CACHE")
        subprocess.run([str(PYTHON), str(ROOT / "scripts/assemble_full_cache.py")], cwd=PROJECT, check=True)
        if json.loads((ROOT / "input_caches/full/cache_audit.json").read_text())["status"] != "PASS":
            raise RuntimeError("full cache audit is not PASS")

        attempt = 0
        while True:
            attempt += 1
            update("WAITING_FOR_SINGLE_GPU", gpu_selection_attempt=attempt)
            # Clear the queue state for this attempt so a prior preserved
            # failure record cannot be mistaken for a new selector failure.
            atomic_queue = ROOT / "training_queue_progress.json"
            temporary_queue = atomic_queue.with_suffix(".json.tmp")
            temporary_queue.write_text(json.dumps({
                "status": "WAITING_FOR_GPU",
                "gpu_selection_attempt": attempt,
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            os.replace(temporary_queue, atomic_queue)
            command = [
                str(PYTHON), str(SELECTOR), "--min-free-gb", "25", "--count", "1",
                "--max-utilization", "30", "--", str(PYTHON),
                str(ROOT / "scripts/run_single_gpu_training_queue.py"),
            ]
            result = subprocess.run(command, cwd=PROJECT)
            if result.returncode == 0:
                break
            training = ROOT / "training_queue_progress.json"
            if training.is_file() and json.loads(training.read_text()).get("status") == "FAILED":
                raise RuntimeError("single-GPU training/evaluation queue failed")
            time.sleep(60)
        update("AGGREGATING_RESULTS")
        subprocess.run([str(PYTHON), str(ROOT / "scripts/aggregate_results.py")], cwd=PROJECT, check=True)
        update("COMPLETE", folds=4, training_jobs=16, evaluation_jobs=4)
        print(json.dumps({"status": "COMPLETE", "folds": 4}), flush=True)
    except BaseException as exc:
        failure = {
            "status": "FAILED", "error": repr(exc), "traceback": traceback.format_exc(),
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        atomic_json(failure)
        (ROOT / "pipeline_failure.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        raise


if __name__ == "__main__":
    main()
