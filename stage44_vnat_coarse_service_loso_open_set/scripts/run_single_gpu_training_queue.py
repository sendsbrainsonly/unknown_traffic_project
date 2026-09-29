#!/usr/bin/env python3
"""Run all Stage44 Known-only training jobs sequentially on one visible GPU."""
from __future__ import annotations

import hashlib
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
FOLDS = ("communication", "file_transfer", "remote_access", "streaming")
COMPONENTS = ("trafficformer", "graph", "yatc", "fusion", "evaluation")
PROTOCOL_ID = "medium_seed2025"
PROGRESS = ROOT / "training_queue_progress.json"
FAILURE = ROOT / "training_queue_failure.json"
LOCK_PREFIX = "training_source_lock"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def source_hashes() -> dict[str, str]:
    paths = [
        ROOT / "config.json",
        ROOT / "vnat_loso_protocol.json",
        ROOT / "vnat_loso_role_manifest.csv",
        ROOT / "scripts/build_fold_cache.py",
        ROOT / "scripts/train_fold_component.py",
        ROOT / "scripts/evaluate_fold.py",
        ROOT / "scripts/aggregate_results.py",
        ROOT / "scripts/run_single_gpu_training_queue.py",
        ROOT / "scripts/run_end_to_end_controller.py",
        PROJECT / "stage31_four_dataset_three_view_equal/train_tf_fig_branch.py",
        PROJECT / "stage31_four_dataset_three_view_equal/train_yatc_branch.py",
        PROJECT / "stage31_four_dataset_three_view_equal/train_equal_fusion.py",
        PROJECT / "stage22_pretrained_trafficformer_e3_closed_set_comparison/scripts/train_closed_run.py",
        PROJECT / "stage23_closed_set_method_table/scripts/train_yatc_closed.py",
    ]
    return {str(path.relative_to(PROJECT)): sha256(path) for path in paths}


def output_dir(fold: str, component: str) -> Path:
    if component == "evaluation":
        return ROOT / "protocols" / fold / "evaluation"
    name = "T0_equal" if component == "fusion" else component
    return (ROOT / "protocols" / fold / "stage44_native_runs_attempt3" / "runs" /
            "vnat" / "medium_seed2025" / name)


def main() -> None:
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("queue must be launched through the live single-GPU selector")
    hashes = source_hashes()
    existing_locks = sorted(ROOT.glob(f"{LOCK_PREFIX}*.json"))
    matching_lock = next((path for path in reversed(existing_locks)
                          if json.loads(path.read_text(encoding="utf-8")).get("source_hashes") == hashes), None)
    if matching_lock is None:
        attempt = len(existing_locks) + 1
        lock_path = ROOT / (f"{LOCK_PREFIX}.json" if attempt == 1 else f"{LOCK_PREFIX}_attempt{attempt}.json")
        atomic_json(lock_path, {
            "status": "LOCKED_PRETRAIN",
            "attempt": attempt,
            "locked_at_utc": datetime.now(timezone.utc).isoformat(),
            "visible_gpu": os.environ["CUDA_VISIBLE_DEVICES"],
            "source_hashes": hashes,
            "unknown_training_samples": 0,
            "unknown_validation_samples": 0,
            "test_selection_samples": 0,
        })
    completed: list[dict[str, object]] = []
    started = time.time()
    try:
        for fold in FOLDS:
            cache_marker = ROOT / "protocols" / fold / "cache_subset_verification.json"
            fold_root = ROOT / "protocols" / fold / "stage44_native_runs_attempt3" / "input_caches" / "vnat" / PROTOCOL_ID
            if not cache_marker.is_file() or not (fold_root / "tf_fig" / "cache_audit.json").is_file():
                subprocess.run([
                    str(PYTHON), str(ROOT / "scripts/build_fold_cache.py"), "--fold", fold,
                ], cwd=PROJECT, check=True)
            if json.loads(cache_marker.read_text(encoding="utf-8"))["status"] != "PASS":
                raise RuntimeError(f"cache subset audit failed: {fold}")
            for component in COMPONENTS:
                marker = (output_dir(fold, component) / "evaluation_verification.json"
                          if component == "evaluation" else output_dir(fold, component) / "SUCCESS")
                if marker.is_file():
                    completed.append({"fold": fold, "component": component, "status": "SKIPPED_EXISTING_SUCCESS"})
                    continue
                current = {
                    "status": "RUNNING",
                    "current_fold": fold,
                    "current_component": component,
                    "completed_jobs": len(completed),
                    "total_jobs": len(FOLDS) * len(COMPONENTS),
                    "elapsed_seconds": time.time() - started,
                    "visible_gpu": os.environ["CUDA_VISIBLE_DEVICES"],
                    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "completed": completed,
                }
                atomic_json(PROGRESS, current)
                job_started = time.time()
                command = ([str(PYTHON), str(ROOT / "scripts/evaluate_fold.py"), "--fold", fold]
                           if component == "evaluation" else [
                               str(PYTHON), str(ROOT / "scripts/train_fold_component.py"),
                               "--fold", fold, "--component", component,
                           ])
                subprocess.run(command, cwd=PROJECT, check=True)
                if not marker.is_file():
                    raise RuntimeError(f"missing SUCCESS after training: {fold}/{component}")
                completed.append({
                    "fold": fold,
                    "component": component,
                    "status": "PASS",
                    "elapsed_seconds": time.time() - job_started,
                })
        atomic_json(PROGRESS, {
            "status": "TRAINING_COMPLETE",
            "completed_jobs": len(completed),
            "total_jobs": len(FOLDS) * len(COMPONENTS),
            "elapsed_seconds": time.time() - started,
            "visible_gpu": os.environ["CUDA_VISIBLE_DEVICES"],
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "completed": completed,
        })
        print(json.dumps({"status": "TRAINING_COMPLETE", "jobs": len(completed)}), flush=True)
    except BaseException as exc:
        failure = {
            "status": "FAILED",
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "completed": completed,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        atomic_json(FAILURE, failure)
        atomic_json(PROGRESS, failure)
        raise


if __name__ == "__main__":
    main()
