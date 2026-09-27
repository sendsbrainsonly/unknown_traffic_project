#!/usr/bin/env python3
"""Continue Stage41 on at most physical GPUs 6 and 7 after user GPU-cap change.

The interrupted A-1 GPU0 output is retained. A-1 is retrained from scratch in
an isolated directory after the already-running A-2/A-3 Open-Detect jobs exit.
"""
from __future__ import annotations

import json
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from freeze_matched_protocol import PROJECT, ROOT, digest

WORKSPACE = PROJECT.parents[1]
HELPER = WORKSPACE / "skills/tmux-task-execution/scripts/tmux_task.sh"
SELECTOR = WORKSPACE / "skills/using-superpowers/scripts/select_gpu.py"
PROGRESS = ROOT / "queue_progress.json"
START = time.monotonic()
GPU_IDS = (6, 7)
EXISTING_OD = ("stage41_od_a2_gpu6_0927", "stage41_od_a3_gpu7_0927")
READY_INPUTS = ("stage41_inputs_a1_0927", "stage41_inputs_a3_0927",
                "stage41_input_parity_0927")
A1_OD_RETRY = "stage41_od_a1_retry_gpu7_0927"


def status(session: str) -> int | None:
    path = PROJECT / ".tmux-task" / session / "exit.status"
    return int(path.read_text().strip()) if path.is_file() else None


def record(phase: str, **extra) -> None:
    payload = {"status": "RUNNING", "phase": phase,
               "updated_at_utc": datetime.now(timezone.utc).isoformat(),
               "elapsed_seconds": round(time.monotonic() - START, 1),
               "max_parallel_stage41_gpus": 2, "physical_gpu_ids": list(GPU_IDS), **extra}
    PROGRESS.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload), flush=True)


def wait(sessions: list[str], phase: str) -> None:
    record(phase, waiting_for=sessions)
    while True:
        codes = {name: status(name) for name in sessions}
        if any(code is not None and code != 0 for code in codes.values()):
            raise RuntimeError(f"{phase} failed: {codes}")
        if all(code == 0 for code in codes.values()):
            record(phase + "_PASS", exit_codes=codes)
            return
        time.sleep(20)


def launch_raw(session: str, command: list[str], *, gpu: int | None = None,
               min_free: float = 12) -> None:
    if (PROJECT / ".tmux-task" / session).exists():
        raise FileExistsError(f"refusing to reuse tmux evidence: {session}")
    if gpu is not None:
        if gpu not in GPU_IDS:
            raise RuntimeError(f"physical GPU {gpu} is outside the two-GPU cap")
        command = ["python", str(SELECTOR), "--min-free-gb", str(min_free),
                   "--allowed", str(gpu), "--", *command]
    subprocess.run([str(HELPER), "start", session, str(PROJECT), "--", *command],
                   check=True, cwd=PROJECT)
    record("LAUNCHED", last_session=session, last_gpu=gpu, last_command=command)


def launch(session: str, script: str, args: list[str], *, gpu: int | None = None,
           min_free: float = 12) -> None:
    launch_raw(session, ["python", str(ROOT / script), *args], gpu=gpu, min_free=min_free)


def main() -> None:
    try:
        protocol = json.loads((ROOT / "matched_protocol.json").read_text())
        if protocol["status"] != "PASS" or digest(Path(protocol["source_manifest"])) != protocol["source_sha256"]:
            raise RuntimeError("frozen protocol source hash changed")
        wait([*EXISTING_OD, *READY_INPUTS], "EXISTING_OD_AND_KNOWN_INPUTS")
        if json.loads((ROOT / "input_parity_audit.json").read_text())["status"] != "PASS":
            raise RuntimeError("three-view input parity failed")
        # GPU0 A-1 attempt was interrupted at epoch 52; never overwrite it.
        sibling = PROJECT.parent / "Open-Detect/reproduction/run_ustc_reproduction.py"
        launch_raw(A1_OD_RETRY, ["python", str(sibling),
            "--data-dir", str(ROOT / "A-1/od_inputs"),
            "--output-dir", str(ROOT / "A-1/od_run_2gpu"),
            "--split", "0", "--epochs", "100", "--batch-size", "128",
            "--workers", "0", "--seed", "2022", "--eval-seed", "2021",
            "--training-protocol", "released-code",
            "--result-label", "Stage41 matched 10-percent USTC flow pool, two-GPU retry",
            "--reproduction-boundary", "local frozen 10-percent flow subset; not authors exact five folds"],
            gpu=7, min_free=16)
        tf_a1 = "stage41_a1_trafficformer_0927"
        launch(tf_a1, "run_threeview.py", ["--setting", "A-1", "--action", "trafficformer"],
               gpu=6, min_free=20)
        wait([A1_OD_RETRY], "A1_OD_RETRY")
        # A-1 TF may still occupy GPU6; extract the three frozen OD scores on GPU7.
        for setting in ("A-1", "A-2", "A-3"):
            session = f"stage41_od_extract_{setting.replace('-', '').lower()}_0927"
            launch(session, "extract_od_scores.py", ["--setting", setting], gpu=7, min_free=8)
            wait([session], "OD_SCORE_EXTRACTION_" + setting)
        pending = [("A-3", "trafficformer"), ("A-1", "yatc"),
                   ("A-3", "yatc"), ("A-1", "graph"), ("A-3", "graph")]
        active: dict[str, int] = {tf_a1: 6}
        completed: list[str] = []
        free = [7]
        while pending or active:
            for session, gpu in list(active.items()):
                code = status(session)
                if code is None:
                    continue
                if code != 0:
                    raise RuntimeError(f"three-view branch failed: {session}, exit {code}")
                del active[session]
                free.append(gpu)
                completed.append(session)
                record("BRANCH_PASS", completed_branches=completed, running_branches=active)
            while free and pending:
                gpu = free.pop(0)
                setting, branch = pending.pop(0)
                session = f"stage41_{setting.replace('-', '').lower()}_{branch}_0927"
                launch(session, "run_threeview.py", ["--setting", setting, "--action", branch],
                       gpu=gpu, min_free=20 if branch in ("yatc", "trafficformer") else 8)
                active[session] = gpu
            if active:
                record("BRANCHES_RUNNING", completed_branches=completed,
                       running_branches=active, pending_branches=pending)
                time.sleep(20)
        for setting, gpu in (("A-1", 6), ("A-3", 7)):
            launch(f"stage41_{setting.replace('-', '').lower()}_fusion_0927",
                   "run_threeview.py", ["--setting", setting, "--action", "fusion"], gpu=gpu, min_free=8)
        wait(["stage41_a1_fusion_0927", "stage41_a3_fusion_0927"], "FUSION")
        for setting in ("A-1", "A-3"):
            launch(f"stage41_{setting.replace('-', '').lower()}_calibrate_0927",
                   "run_threeview_scores.py", ["--setting", setting, "--phase", "calibrate"])
        wait(["stage41_a1_calibrate_0927", "stage41_a3_calibrate_0927"], "KNOWN_VAL_CALIBRATION")
        for setting in ("A-1", "A-3"):
            session = f"stage41_{setting.replace('-', '').lower()}_test_inputs_0927"
            launch(session, "build_threeview_inputs.py", ["--setting", setting, "--phase", "eval"])
            wait([session], "FROZEN_TEST_INPUTS_" + setting)
        for setting, gpu in (("A-1", 6), ("A-3", 7)):
            launch(f"stage41_{setting.replace('-', '').lower()}_eval_0927",
                   "run_threeview_scores.py", ["--setting", setting, "--phase", "evaluate"],
                   gpu=gpu, min_free=18)
        wait(["stage41_a1_eval_0927", "stage41_a3_eval_0927"], "FROZEN_EVALUATION")
        launch("stage41_comparison_0927", "compare_scores.py", [])
        wait(["stage41_comparison_0927"], "PAIRED_COMPARISON")
        launch("stage41_finalize_0927", "finalize.py", [])
        # Finalizer hashes queue_progress.json; do not mutate it afterward.
        while status("stage41_finalize_0927") is None:
            time.sleep(20)
        if status("stage41_finalize_0927") != 0:
            raise RuntimeError("Stage41 finalizer failed; see named tmux log")
    except BaseException as exc:
        failure = {"status": "FAIL", "error": repr(exc), "traceback": traceback.format_exc()}
        (ROOT / "queue_failure_2gpu.json").write_text(json.dumps(failure, indent=2) + "\n")
        record("FAIL", error=repr(exc))
        raise


if __name__ == "__main__":
    main()
