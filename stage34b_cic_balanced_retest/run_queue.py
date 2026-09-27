#!/usr/bin/env python3
"""Run the isolated balanced CIC experiment on at most physical GPUs 0 and 1."""
from __future__ import annotations

import json
import os
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from prepare_balanced import ROOT, verified_rows

PROJECT = ROOT.parent
WORKSPACE = PROJECT.parents[1]
HELPER = WORKSPACE / "skills/tmux-task-execution/scripts/tmux_task.sh"
SELECTOR = WORKSPACE / "skills/using-superpowers/scripts/select_gpu.py"
PYTHON = "/home/birkenwald/data/SEU-WXY/conda_envs/2025-10-8-WXY-dgl_py310/bin/python"
START = time.monotonic()
SESSIONS: list[str] = []


def update(status: str, phase: str, **details) -> None:
    payload = {"status": status, "phase": phase,
               "updated_at_utc": datetime.now(timezone.utc).isoformat(),
               "elapsed_seconds": round(time.monotonic() - START, 1),
               "max_parallel_owned_gpus": 2, "allowed_physical_gpus": [0, 1],
               "sessions": SESSIONS, **details}
    temporary = ROOT / "queue_progress.json.tmp"
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, ROOT / "queue_progress.json")
    print(json.dumps(payload), flush=True)


def exit_code(session: str) -> int | None:
    path = PROJECT / ".tmux-task" / session / "exit.status"
    return int(path.read_text().strip()) if path.is_file() else None


def wait(sessions: list[str], phase: str) -> None:
    update("WAITING", phase, waiting_for=sessions)
    while True:
        codes = {name: exit_code(name) for name in sessions}
        if any(code is not None and code != 0 for code in codes.values()):
            raise RuntimeError(f"{phase} failed: {codes}")
        if all(code == 0 for code in codes.values()):
            update("PASS", phase, exit_codes=codes)
            return
        time.sleep(30)


def wait_capacity(ids: tuple[int, ...], min_free_mib: int = 30720) -> None:
    update("WAITING", "GPU_CAPACITY", required_ids=ids, min_free_mib=min_free_mib)
    while True:
        result = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu",
                                 "--format=csv,noheader,nounits"], text=True, capture_output=True, check=True)
        gpu = {}
        for line in result.stdout.splitlines():
            number, free, utilization = (int(x.strip()) for x in line.split(","))
            gpu[number] = (free, utilization)
        if all(gpu.get(index, (0, 100))[0] >= min_free_mib and
               gpu.get(index, (0, 100))[1] <= 30 for index in ids):
            update("PASS", "GPU_CAPACITY", physical_gpu_ids=ids,
                   live_snapshot={str(index): gpu[index] for index in ids})
            return
        time.sleep(30)


def launch(session: str, script: str, args: tuple[str, ...], gpu: int | None) -> None:
    if (PROJECT / ".tmux-task" / session).exists():
        raise FileExistsError(f"refusing to reuse named task evidence: {session}")
    command = [PYTHON, str(ROOT / script), *args]
    if gpu is not None:
        command = [PYTHON, str(SELECTOR), "--min-free-gb", "30", "--allowed", str(gpu),
                   "--max-utilization", "30", "--", *command]
    subprocess.run([str(HELPER), "start", session, str(PROJECT), "--", *command],
                   check=True, cwd=PROJECT)
    SESSIONS.append(session)
    update("RUNNING", args[0] if args else script, last_session=session, physical_gpu_id=gpu)


def job(session: str, script: str, args: tuple[str, ...], gpu: int | None = None) -> None:
    if gpu is not None:
        wait_capacity((gpu,))
    launch(session, script, args, gpu)
    wait([session], session)


def main() -> None:
    try:
        verified_rows()
        cache = ROOT / "cicids2017/input_caches/ustc/A-2/tf_fig/cache_audit.json"
        if json.loads(cache.read_text())["status"] != "PASS":
            raise RuntimeError("balanced Train/Val cache not PASS")
        update("PASS", "PREFLIGHT", frozen_manifest=True, balanced_trainval_flows=39428)
        job("stage34b_graph_0926", "run_balanced.py", ("branch", "--branch", "graph"), gpu=1)
        wait_capacity((0, 1))
        launch("stage34b_trafficformer_0926", "run_balanced.py", ("branch", "--branch", "trafficformer"), gpu=0)
        launch("stage34b_yatc_0926", "run_balanced.py", ("branch", "--branch", "yatc"), gpu=1)
        wait(["stage34b_trafficformer_0926", "stage34b_yatc_0926"], "BALANCED_BRANCHES")
        job("stage34b_fusion_0926", "run_balanced.py", ("fusion",), gpu=1)
        job("stage34b_test_inputs_0926", "build_test_inputs.py", ())
        job("stage34b_test_0926", "run_balanced.py", ("evaluate",), gpu=1)
        launch("stage34b_finalize_0926", "finalize_balanced.py", (), gpu=None)
        # The finalizer writes the terminal queue state before inventory hashing.
        # Do not mutate queue_progress.json after its bundle validation passes.
        while (code := exit_code("stage34b_finalize_0926")) is None:
            time.sleep(30)
        if code != 0:
            raise RuntimeError(f"Stage34B final verification failed with exit code {code}")
    except BaseException as exc:
        (ROOT / "queue_failure.json").write_text(json.dumps({
            "status": "FAIL", "error": repr(exc), "traceback": traceback.format_exc(),
            "sessions": SESSIONS, "elapsed_seconds": round(time.monotonic() - START, 1)}, indent=2) + "\n")
        update("FAIL", "STOPPED", error=repr(exc))
        raise


if __name__ == "__main__":
    main()
