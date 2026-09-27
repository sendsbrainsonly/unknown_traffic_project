#!/usr/bin/env python3
"""Auto-advance one Stage38B dataset from frozen fusion through score replay."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import traceback
from datetime import datetime, timezone

from run_iscx_known import ROOT, PROJECT, lock

WORKSPACE = PROJECT.parents[1]
HELPER = WORKSPACE / "skills/tmux-task-execution/scripts/tmux_task.sh"
SELECTOR = WORKSPACE / "skills/using-superpowers/scripts/select_gpu.py"
PYTHON = "/home/birkenwald/data/SEU-WXY/conda_envs/2025-10-8-WXY-dgl_py310/bin/python"


def main(dataset: str, resume_after_calibrate: bool = False, attempt_suffix: str = "") -> None:
    short = dataset.removeprefix("iscx_")
    progress = ROOT / f"{dataset}_detection_queue_progress.json"
    start = time.monotonic()

    def update(status: str, phase: str, **details: object) -> None:
        value = {"status": status, "phase": phase, "dataset": dataset,
                 "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                 "elapsed_seconds": round(time.monotonic() - start, 1),
                 "gpu_allowed_physical_ids": [3, 4, 7],
                 "gpu2_used_after_training": False, **details}
        tmp = progress.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
        os.replace(tmp, progress)
        print(json.dumps(value), flush=True)

    def exit_code(session: str) -> int | None:
        path = PROJECT / ".tmux-task" / session / "exit.status"
        return int(path.read_text().strip()) if path.is_file() else None

    def wait_fusion() -> None:
        name = f"stage38b_{short}_fusion_auto_0926"
        update("WAITING", "KNOWN_ONLY_FUSION", session=name)
        while True:
            code = exit_code(name)
            if code is not None:
                if code != 0:
                    raise RuntimeError(f"frozen Known-only fusion failed with exit {code}")
                break
            training = ROOT / "training_queue_progress.json"
            if training.is_file() and json.loads(training.read_text())["status"] == "FAILED":
                raise RuntimeError("Stage38B training controller failed before fusion")
            time.sleep(30)
        lock(dataset)
        checkpoint = ROOT / "runs" / dataset / "medium_seed2022/T0_equal/SUCCESS"
        if not checkpoint.is_file():
            raise RuntimeError("frozen Known-only fusion marker missing")

    def run(phase: str, script: str, args: list[str], gpu: bool) -> None:
        suffix = f"_{attempt_suffix}" if attempt_suffix else ""
        name = f"stage38b_{short}_{phase}{suffix}_auto_0926"
        if (PROJECT / ".tmux-task" / name).exists():
            raise FileExistsError(f"refusing to overwrite named task: {name}")
        command = [PYTHON, str(ROOT / script), *args]
        if gpu:
            command = [PYTHON, str(SELECTOR), "--min-free-gb", "25", "--allowed", "3,4,7",
                       "--max-utilization", "30", "--", *command]
        update("STARTING", phase, session=name, gpu_required=gpu)
        subprocess.run([str(HELPER), "start", name, str(PROJECT), "--", *command],
                       cwd=PROJECT, check=True)
        update("RUNNING", phase, session=name, gpu_required=gpu)
        while (code := exit_code(name)) is None:
            time.sleep(30)
        if code != 0:
            raise RuntimeError(f"Stage38B {dataset}/{phase} failed with exit {code}")
        update("PASS", phase, session=name, exit_code=code)

    try:
        lock(dataset)
        wait_fusion()
        if resume_after_calibrate:
            calibration = ROOT / "runs" / dataset / "medium_seed2022/detection/calibration.json"
            audit = ROOT / "runs" / dataset / "medium_seed2022/detection/calibration_verification.json"
            if json.loads(calibration.read_text())["status"] != "PASS" or json.loads(audit.read_text())["status"] != "PASS":
                raise RuntimeError("cannot resume without the verified frozen Known-only calibration")
            update("PASS", "CALIBRATION_REUSED", source=str(calibration))
        else:
            run("calibrate", "run_iscx_detection.py", ["calibrate", "--dataset", dataset], gpu=True)
        run("tf_fig_test", "build_iscx_eval_inputs.py", ["--dataset", dataset, "--phase", "tf_fig"], gpu=False)
        run("mfr_test", "build_iscx_eval_inputs.py", ["--dataset", dataset, "--phase", "mfr"], gpu=False)
        run("evaluate", "run_iscx_detection.py", ["evaluate", "--dataset", dataset], gpu=True)
        run("verify", "run_iscx_detection.py", ["verify", "--dataset", dataset], gpu=False)
        check = ROOT / "runs" / dataset / "medium_seed2022/detection/completion_verification.json"
        if json.loads(check.read_text())["status"] != "PASS":
            raise RuntimeError("independent frozen score verification did not pass")
        update("COMPLETE", "STAGE38B_DATASET", verification=str(check))
    except BaseException as exc:
        suffix = f"_{attempt_suffix}" if attempt_suffix else ""
        failure = ROOT / f"{dataset}_detection_queue_failure{suffix}.json"
        failure.write_text(json.dumps({"status": "FAIL", "error": repr(exc),
            "traceback": traceback.format_exc(),
            "updated_at_utc": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n")
        update("FAILED", "STOPPED", error=repr(exc))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("iscx_vpn", "iscx_tor"), required=True)
    parser.add_argument("--resume-after-calibrate", action="store_true")
    parser.add_argument("--attempt-suffix", default="")
    args = parser.parse_args()
    if args.resume_after_calibrate and not args.attempt_suffix:
        parser.error("resumed attempt needs a unique suffix")
    main(args.dataset, args.resume_after_calibrate, args.attempt_suffix)
