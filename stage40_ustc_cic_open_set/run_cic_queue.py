#!/usr/bin/env python3
"""Finish the two frozen CIC open-set pilots without competing with Stage39 training."""
from __future__ import annotations

import json
import os
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from freeze_protocols import PROJECT, ROOT, digest

WORKSPACE = PROJECT.parents[1]
HELPER = WORKSPACE / "skills/tmux-task-execution/scripts/tmux_task.sh"
SELECTOR = WORKSPACE / "skills/using-superpowers/scripts/select_gpu.py"
PYTHON = "/home/birkenwald/data/SEU-WXY/conda_envs/2025-10-8-WXY-dgl_py310/bin/python"
STAGE39 = PROJECT / "stage39_coarse_open_set_execution/training_queue_progress.json"
FOLDS = ("unknown_portscan", "unknown_slowhttptest")
START = time.monotonic()


def save(status: str, fold: str = "", phase: str = "", **extra: object) -> None:
    record = {"status": status, "fold": fold, "phase": phase,
              "updated_at_utc": datetime.now(timezone.utc).isoformat(),
              "elapsed_seconds": round(time.monotonic() - START, 1),
              "stage39_training_gpu_cap": 3, "stage40_physical_gpu_limit": 1, **extra}
    target = ROOT / "cic_queue_progress.json"
    temp = ROOT / "cic_queue_progress.json.tmp"
    temp.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, target)
    print(json.dumps(record, ensure_ascii=False), flush=True)


def freeze_check(fold: str) -> None:
    protocol = json.loads((ROOT / "protocols.json").read_text(encoding="utf-8"))
    if protocol["status"] != "PASS":
        raise RuntimeError("Stage40 protocol freeze not PASS")
    unit = next(x for x in protocol["units"] if x.get("key") == fold)
    if digest(Path(unit["role_manifest"])) != unit["role_manifest_sha256"]:
        raise RuntimeError(f"Frozen CIC role manifest drift: {fold}")
    for path, expected in unit["source_hashes"].items():
        if digest(Path(path)) != expected:
            raise RuntimeError(f"Frozen CIC source manifest drift: {path}")


def run(fold: str, phase: str, args: list[str], *, gpu: bool = False) -> None:
    command = [PYTHON, str(ROOT / args[0]), *args[1:]]
    if gpu:
        command = [PYTHON, str(SELECTOR), "--allowed", "3", "--min-free-gb", "12",
                   "--max-utilization", "30", "--", *command]
    for attempt in range(1, 121 if gpu else 2):
        name = f"stage40q_{fold}_{phase}_attempt{attempt}_0926"
        if (PROJECT / ".tmux-task" / name).exists():
            raise FileExistsError(f"Refusing to overwrite task evidence: {name}")
        save("RUNNING", fold, phase, session=name, capacity_attempt=attempt)
        result = subprocess.run([str(HELPER), "run", name, str(PROJECT), "--", *command],
                                cwd=PROJECT, check=False)
        if result.returncode == 0:
            return
        log_path = PROJECT / ".tmux-task" / name / "output.log"
        log = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
        if gpu and "No GPU satisfies the declared constraints." in log:
            save("WAITING_FOR_GPU3_CAPACITY", fold, phase, failed_session=name,
                 capacity_attempt=attempt)
            time.sleep(60)
            continue
        raise RuntimeError(f"Stage40 worker failed, inspect {name}: exit={result.returncode}")
    raise RuntimeError(f"GPU3 capacity unavailable after 120 minutes: {fold}/{phase}")


def completed_branch(fold: str, component: str) -> bool:
    branch = ROOT / fold / "runs/ustc/A-2" / component
    if not (branch / "SUCCESS").is_file():
        return False
    meta = json.loads((branch / "metrics.json").read_text())
    if digest(branch / "model_best.pt") != meta["checkpoint_sha256"]:
        raise RuntimeError(f"Completed branch checkpoint drift: {fold}/{component}")
    return True


def completed_fusion(fold: str) -> bool:
    fusion = ROOT / fold / "runs/ustc/A-2/T0_equal"
    if not (fusion / "SUCCESS").is_file():
        return False
    meta = json.loads((fusion / "known_validation_metrics.json").read_text())
    for name, expected in meta["checkpoint_hashes"].items():
        if digest(fusion / name) != expected:
            raise RuntimeError(f"Completed fusion checkpoint drift: {fold}/{name}")
    return True


def marker_pass(path: Path) -> bool:
    if not path.is_file():
        return False
    marker = json.loads(path.read_text(encoding="utf-8"))
    if marker.get("status") != "PASS":
        raise RuntimeError(f"Existing marker not PASS: {path}")
    return True


def one(fold: str) -> None:
    freeze_check(fold)
    for component in ("graph", "trafficformer", "yatc"):
        if not completed_branch(fold, component):
            run(fold, component, ["train_cic.py", "--key", fold,
                                  "--component", component], gpu=True)
            if not completed_branch(fold, component):
                raise RuntimeError(f"Branch did not finish: {fold}/{component}")
    if not completed_fusion(fold):
        run(fold, "fusion", ["train_cic.py", "--key", fold,
                             "--component", "fusion"], gpu=True)
        if not completed_fusion(fold):
            raise RuntimeError(f"Fusion did not finish: {fold}")
    detection = ROOT / fold / "detection"
    if not marker_pass(detection / "calibration.json"):
        run(fold, "calibrate", ["score_frozen.py", "calibrate", "--key", fold])
        if not marker_pass(detection / "calibration.json"):
            raise RuntimeError(f"Known-only calibration did not finish: {fold}")
    cache = ROOT / fold / "input_caches/ustc/A-2/tf_fig_eval/cache_audit.json"
    if not marker_pass(cache):
        run(fold, "eval_cache", ["prepare_cic.py", "--key", fold,
                                  "--phase", "eval"])
        if not marker_pass(cache):
            raise RuntimeError(f"Test cache did not finish: {fold}")
    if not (detection / "SUCCESS").is_file():
        run(fold, "evaluate", ["score_frozen.py", "evaluate", "--key", fold], gpu=True)
        if not (detection / "SUCCESS").is_file():
            raise RuntimeError(f"Frozen evaluation did not finish: {fold}")
    if not marker_pass(detection / "independent_verification.json"):
        run(fold, "verify", ["verify_unit.py", "--key", fold])
        if not marker_pass(detection / "independent_verification.json"):
            raise RuntimeError(f"Independent metric replay did not finish: {fold}")
    freeze_check(fold)


def main() -> None:
    try:
        while True:
            state = json.loads(STAGE39.read_text()) if STAGE39.exists() else {}
            if state.get("status") == "FAILED":
                raise RuntimeError("Stage39 training queue failed; inspect GPU ownership before Stage40 launch")
            if state.get("status") == "KNOWN_TRAINING_COMPLETE":
                break
            save("WAITING_FOR_STAGE39_TRAINING", phase="gpu_capacity",
                 stage39_status=state.get("status"),
                 stage39_setting=f"{state.get('dataset', '')}/{state.get('fold', '')}")
            time.sleep(30)
        completed = []
        for fold in FOLDS:
            one(fold)
            completed.append(fold)
            save("RUNNING", fold, "fold_complete", completed=completed)
        save("CIC_PILOTS_COMPLETE", phase="all_verified", completed=completed)
    except BaseException as exc:
        failure = {"status": "FAILED", "error": repr(exc),
                   "traceback": traceback.format_exc(),
                   "updated_at_utc": datetime.now(timezone.utc).isoformat()}
        (ROOT / "cic_queue_failure.json").write_text(
            json.dumps(failure, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        save("FAILED", phase="stopped", error=repr(exc))
        raise


if __name__ == "__main__":
    main()
