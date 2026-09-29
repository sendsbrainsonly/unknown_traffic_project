#!/usr/bin/env python3
"""Preserve first-pass evaluations and finish with exact Native preprocessing."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

from matched_common import FOLDS, PROJECT, ROOT, role_rows, sha256, verify_freeze


def save_progress(value: dict[str, object]) -> None:
    path = ROOT / "resume_progress.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def run(script: str, *args: str) -> None:
    command = [sys.executable, str(Path(__file__).with_name(script)), *args]
    completed = subprocess.run(command, check=False)
    if completed.returncode:
        raise RuntimeError(f"{script} {args} exited {completed.returncode}")


def main() -> None:
    before = verify_freeze()
    source_paths = (
        PROJECT.parent / "Open-Detect/code/model.py",
        PROJECT.parent / "Open-Detect/code/utils.py",
        PROJECT.parent / "Open-Detect/reproduction/run_reproduction.py",
        PROJECT / "stage14c_native_opendetect_vnat/scripts/train_one.py",
        *(Path(__file__).with_name(name) for name in (
            "matched_common.py", "prepare_inputs.py", "train_fold.py",
            "evaluate_fold.py", "aggregate_results.py", "resume_native_exact.py",
        )),
    )
    source_lock = {str(path): sha256(path) for path in source_paths}
    (ROOT / "resume_source_lock.json").write_text(json.dumps(source_lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for fold in FOLDS:
        role_rows(fold)
        audit = json.loads((ROOT / "inputs" / fold / "input_audit.json").read_text(encoding="utf-8"))
        if audit["status"] != "PASS" or audit["frozen_inputs"] != before:
            raise RuntimeError(f"{fold}: Known-only input audit mismatch")
    for fold in FOLDS[:3]:
        result = json.loads((ROOT / "runs" / fold / "result.json").read_text(encoding="utf-8"))
        checkpoint = ROOT / "runs" / fold / "best_checkpoint.pt"
        if result["status"] != "success" or sha256(checkpoint) != result["checkpoint_sha256"]:
            raise RuntimeError(f"{fold}: completed Native checkpoint mismatch")
        if not (ROOT / "runs" / fold / "COMPLETED").is_file():
            raise RuntimeError(f"{fold}: Native training incomplete")
    if (ROOT / "runs" / "streaming").exists():
        raise RuntimeError("streaming training target already exists")

    archive = ROOT / "evaluation_attempt1"
    archive.mkdir(exist_ok=False)
    for fold in FOLDS[:3]:
        original = ROOT / "evaluation" / fold
        if not original.is_dir():
            raise RuntimeError(f"{fold}: first-pass evaluation target missing")
        original.rename(archive / fold)
    (ROOT / "evaluation_attempt1" / "README.md").write_text(
        "# Preserved first-pass evaluation\n\n"
        "The initial direct uint8-to-Tensor adapter differed from Native PIL/ToTensor "
        "on one remote_access validation prediction. The original output is retained "
        "unaltered for diagnosis; formal metrics are recomputed under `evaluation/`.\n",
        encoding="utf-8",
    )
    progress: dict[str, object] = {
        "status": "RUNNING", "completed_components": [], "current": None,
        "frozen_source": before, "visible_gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "UNSET"),
        "started_at_unix": time.time(), "reason": "Native PIL/ToTensor validation parity repair",
    }
    save_progress(progress)
    try:
        for fold in FOLDS:
            components = (("training", "train_fold.py", ("--protocol-id", fold, "--device", "cuda:0")),
                          ("evaluation", "evaluate_fold.py", ("--fold", fold))) if fold == "streaming" else (
                              ("evaluation", "evaluate_fold.py", ("--fold", fold)),)
            for component, script, args in components:
                progress["current"] = {"fold": fold, "component": component}
                progress["updated_at_unix"] = time.time()
                save_progress(progress)
                print(json.dumps({"status": "START", "fold": fold, "component": component}), flush=True)
                start = time.time()
                run(script, *args)
                if verify_freeze() != before or {str(path): sha256(path) for path in source_paths} != source_lock:
                    raise RuntimeError("frozen input or Native code changed during resumed queue")
                progress["completed_components"].append({
                    "fold": fold, "component": component, "status": "PASS",
                    "elapsed_seconds": time.time() - start,
                })
                save_progress(progress)
                print(json.dumps({"status": "PASS", "fold": fold, "component": component}), flush=True)
        progress["current"] = {"component": "aggregation"}
        save_progress(progress)
        run("aggregate_results.py")
        if verify_freeze() != before or {str(path): sha256(path) for path in source_paths} != source_lock:
            raise RuntimeError("frozen source changed at resumed-queue completion")
        progress["status"] = "COMPLETE"
        progress["current"] = None
        progress["finished_at_unix"] = time.time()
        save_progress(progress)
        print(json.dumps({"status": "COMPLETE", "components": len(progress["completed_components"])}), flush=True)
    except Exception as exc:
        failure = {"status": "FAILED", "exception": repr(exc), "traceback": traceback.format_exc(),
                   "current": progress["current"], "completed_components": progress["completed_components"]}
        (ROOT / "resume_failure.json").write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        progress["status"] = "FAILED"
        save_progress(progress)
        raise


if __name__ == "__main__":
    main()
