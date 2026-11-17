#!/usr/bin/env python3
"""Persistent CPU preparation and capacity-gated two-GPU task queue."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
WORKSPACE = PROJECT.parents[1]
HELPER = WORKSPACE / "skills/tmux-task-execution/scripts/tmux_task.sh"
SELECTOR = WORKSPACE / "skills/using-superpowers/scripts/select_gpu.py"
PYTHON = "/home/birkenwald/data/SEU-WXY/conda_envs/2025-10-8-WXY-dgl_py310/bin/python"
SESSION_PREFIX = "stage49_cic_"
STARTED = time.monotonic()
SESSIONS = []
JOBS = {}
CONSERVATIVE_MIN_FREE_MIB = 30720


def state(status, phase, **extra):
    value = {"status": status, "phase": phase, "updated_at_utc": datetime.now(timezone.utc).isoformat(),
             "elapsed_seconds": round(time.monotonic() - STARTED, 1), "sessions": SESSIONS,
             "jobs": JOBS, "max_parallel_owned_gpus": 2, **extra}
    temp = ROOT / "queue_progress.json.tmp"
    temp.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temp, ROOT / "queue_progress.json")
    print(json.dumps(value), flush=True)


def code(session):
    p = PROJECT / ".tmux-task" / session / "exit.status"
    return int(p.read_text().strip()) if p.exists() else None


def launch(label, script, args=(), gpu=None):
    session = f"{SESSION_PREFIX}{label}_20261117"
    if (PROJECT / ".tmux-task" / session).exists():
        raise FileExistsError(f"refuse duplicate child: {session}")
    command = [PYTHON, str(ROOT / script), *args]
    if gpu is not None:
        command = [PYTHON, str(SELECTOR), "--min-free-gb", "30", "--max-utilization", "30",
                   "--allowed", str(gpu), "--", *command]
    subprocess.run([str(HELPER), "start", session, str(PROJECT), "--", *command],
                   check=True, cwd=PROJECT)
    SESSIONS.append(session)
    JOBS[label] = {"session": session, "state": "running", "physical_gpu_id": gpu}
    state("RUNNING", label)
    return session


def wait(label, session):
    while code(session) is None:
        time.sleep(15)
    value = code(session)
    JOBS[label]["state"] = "success" if value == 0 else "failed"
    JOBS[label]["exit_code"] = value
    if value != 0:
        raise RuntimeError(f"child failed {label}: exit={value}; session={session}")
    state("RUNNING", f"{label}_complete")


def candidates(reserved=()):
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
    snapshot = []
    eligible = []
    for line in out.stdout.splitlines():
        index, free, util = map(lambda x: int(x.strip()), line.split(","))
        snapshot.append({"physical_id": index, "free_mib": free, "utilization": util})
        if index not in reserved and free >= CONSERVATIVE_MIN_FREE_MIB and util <= 30:
            eligible.append((util, -free, index))
    return [i for _, _, i in sorted(eligible)], snapshot


def gpu_job(label, args):
    last_print = 0
    while True:
        ids, snapshot = candidates()
        if ids:
            session = launch(label, "worker.py", args, gpu=ids[0])
            wait(label, session)
            return
        if time.monotonic() - last_print >= 60:
            state("WAITING", f"GPU_CAPACITY_{label}", min_free_gib=30,
                  max_utilization=30, live_gpu_snapshot=snapshot)
            last_print = time.monotonic()
        time.sleep(30)


def branch_queue():
    pending = ["trafficformer", "yatc", "graph"]
    active = {}
    last_print = 0
    while pending or active:
        for name in list(active):
            session, gpu = active[name]
            result = code(session)
            if result is None:
                continue
            JOBS[name].update(state="success" if result == 0 else "failed", exit_code=result)
            if result:
                raise RuntimeError(f"branch{ name } failed with exit{result}: {session}")
            del active[name]
            state("RUNNING", "BRANCH_COMPLETE", completed_branch=name)
        ids, snapshot = candidates([gpu for _, gpu in active.values()])
        while pending and ids and len(active) < 2:
            name, gpu = pending.pop(0), ids.pop(0)
            session = launch(name, "worker.py", ("branch", "--branch", name), gpu)
            active[name] = (session, gpu)
        if time.monotonic() - last_print >= 60:
            state("RUNNING" if active else "WAITING", "BRANCHES" if active else "GPU_CAPACITY_BRANCHES",
                  pending_branches=pending, live_gpu_snapshot=snapshot, min_free_gib=30, max_utilization=30)
            last_print = time.monotonic()
        if pending or active:
            time.sleep(20)


def main():
    if (ROOT / "queue_progress.json").exists():
        raise FileExistsError("a queue record already exists; inspect before any continuation")
    os.environ.update(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
    try:
        state("RUNNING", "STARTING")
        for label, script, args in (("protocol", "protocol.py", ()),
                                    ("preflight", "worker.py", ("preflight",)),
                                    ("trainval_inputs", "worker.py", ("inputs", "--phase", "trainval"))):
            session = launch(label, script, args)
            wait(label, session)
        branch_queue()
        gpu_job("fusion", ("fusion",))
        session = launch("test_inputs", "worker.py", ("inputs", "--phase", "test"))
        wait("test_inputs", session)
        gpu_job("evaluate", ("evaluate",))
        state("RUNNING", "FINAL_REPLAY")
        # Finalizer runs as a separate child and owns final queue metadata from
        # this point, so the completed archive will not be mutated afterward.
        session = launch("finalize", "finalize.py")
        while code(session) is None:
            time.sleep(15)
        if code(session):
            raise RuntimeError(f"final verification failed: {session}")
    except BaseException as exc:
        failure = {"status": "FAIL", "error": repr(exc), "traceback": traceback.format_exc(),
                   "sessions": SESSIONS}
        (ROOT / "queue_failure.json").write_text(json.dumps(failure, indent=2) + "\n")
        state("FAIL", "STOPPED", error=repr(exc))
        raise


if __name__ == "__main__":
    main()
