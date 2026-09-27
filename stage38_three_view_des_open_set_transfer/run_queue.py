#!/usr/bin/env python3
"""Sequential Stage 38A pipeline with live-selected physical GPU 2."""
from __future__ import annotations

import json
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
WORKSPACE = PROJECT.parents[1]
PYTHON = "/home/birkenwald/data/SEU-WXY/conda_envs/2025-10-8-WXY-dgl_py310/bin/python"
SELECTOR = WORKSPACE / "skills/using-superpowers/scripts/select_gpu.py"
PRESERVE = WORKSPACE / "skills/experiment-data-preservation/scripts"
DEADLINE = datetime(2026, 9, 26, 7, 45, tzinfo=timezone.utc)
START = time.monotonic()


def update(status: str, phase: str, **details: object) -> None:
    payload = {"status": status, "phase": phase,
               "updated_at_utc": datetime.now(timezone.utc).isoformat(),
               "elapsed_seconds": round(time.monotonic() - START, 1),
               "physical_gpu_ids": [2],
               "gpu2_release_deadline_utc": DEADLINE.isoformat(), **details}
    temporary = ROOT / "queue_progress.json.tmp"
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(ROOT / "queue_progress.json")
    print(json.dumps(payload), flush=True)


def phase(name: str, *, gpu: bool = False) -> None:
    command = [PYTHON, str(ROOT / "run_vnat.py"), name]
    if gpu:
        remaining = int((DEADLINE - datetime.now(timezone.utc)).total_seconds())
        if remaining < 2700:
            raise RuntimeError("GPU2 release deadline is too close for frozen inference")
        command = ["timeout", "--signal=TERM", "--kill-after=30s", str(remaining),
                   PYTHON, str(SELECTOR), "--min-free-gb", "20", "--allowed", "2",
                   "--max-utilization", "30", "--", *command]
    update("RUNNING", name, gpu_required=gpu)
    result = subprocess.run(command, cwd=PROJECT, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Stage38A {name} failed with exit {result.returncode}")
    update("PASS", name, exit_code=result.returncode)


def main() -> None:
    try:
        for name in ("preflight", "parity", "known_smoke", "extract", "evaluate",
                     "verify", "finalize"):
            phase(name, gpu=name in ("known_smoke", "evaluate"))
        update("VNAT_COMPLETE", "STAGE38A", stage38b_status="PENDING")
        subprocess.run([PYTHON, str(PRESERVE / "refresh_artifact_manifest.py"), str(ROOT)],
                       cwd=PROJECT, check=True)
        subprocess.run([PYTHON, str(PRESERVE / "validate_experiment_bundle.py"),
                        str(ROOT), "--verify-hashes"], cwd=PROJECT, check=True)
    except BaseException as exc:
        (ROOT / "queue_failure.json").write_text(json.dumps({
            "status": "FAILED", "error": repr(exc), "traceback": traceback.format_exc(),
            "updated_at_utc": datetime.now(timezone.utc).isoformat()},
            indent=2, ensure_ascii=False) + "\n")
        update("FAILED", "STOPPED", error=repr(exc))
        subprocess.run([PYTHON, str(PRESERVE / "refresh_artifact_manifest.py"), str(ROOT)],
                       cwd=PROJECT, check=False)
        raise


if __name__ == "__main__":
    main()
