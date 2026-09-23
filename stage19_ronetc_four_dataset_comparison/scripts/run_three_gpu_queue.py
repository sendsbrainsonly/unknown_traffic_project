#!/usr/bin/env python3
"""Run the frozen Stage 19 grid with at most three physical GPUs.

The four one-GPU protocols are queued first.  USTC starts only after all four
finish successfully and uses the same global batch of 128 split over three GPUs.
No failed task is retried automatically.
"""
from __future__ import annotations

import csv
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from common import PROJECT, ROOT


WORKSPACE = PROJECT.parents[1]
SELECTOR = WORKSPACE / ".agents" / "skills" / "using-superpowers" / "scripts" / "select_gpu.py"
TRAINER = ROOT / "scripts" / "train_eval.py"
STATUS_PATH = ROOT / "queue_status.json"
PROGRESS_PATH = ROOT / "progress.txt"
LOG_ROOT = ROOT / "queue_logs"
STOP_PATH = ROOT / "STOP_QUEUE"

TASKS = [
    {"name": "iscx_tor_medium2022", "dataset": "iscx_tor", "protocol": "medium_seed2022", "gpu": "0", "gpu_count": 1},
    {"name": "iscx_vpn_medium2022", "dataset": "iscx_vpn", "protocol": "medium_seed2022", "gpu": "4", "gpu_count": 1},
    {"name": "vnat_medium2025", "dataset": "vnat", "protocol": "medium_seed2025", "gpu": "5", "gpu_count": 1},
    {"name": "vnat_medium2026", "dataset": "vnat", "protocol": "medium_seed2026", "gpu": None, "gpu_count": 1},
    {"name": "ustc_A2", "dataset": "ustc", "protocol": "A-2", "gpu": "0,4,5", "gpu_count": 3, "data_parallel": True},
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def current_epoch(task: dict) -> int:
    path = ROOT / "runs" / task["dataset"] / task["protocol"] / "history.csv"
    if not path.is_file():
        return 0
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return int(rows[-1]["epoch"]) if rows else 0


def write_status(records: dict[str, dict], queue_state: str) -> None:
    rows = []
    total_epochs = 0
    for task in TASKS:
        record = records[task["name"]]
        epoch = current_epoch(task)
        total_epochs += min(epoch, 100)
        rows.append({
            "name": task["name"], "dataset": task["dataset"], "protocol": task["protocol"],
            "state": record["state"], "epoch": epoch, "planned_epochs": 100,
            "physical_gpu": record.get("physical_gpu") or task.get("gpu") or "waiting_for_slot",
            "pid": record.get("pid"), "returncode": record.get("returncode"),
            "started_at_utc": record.get("started_at_utc"), "finished_at_utc": record.get("finished_at_utc"),
            "log": record.get("log"),
        })
    payload = {
        "updated_at_utc": utc_now(), "queue_state": queue_state,
        "max_concurrent_physical_gpus": 3, "completed_epochs": total_epochs,
        "planned_epochs": 500, "progress_fraction": total_epochs / 500.0,
        "tasks": rows, "automatic_retry": False,
    }
    atomic_write(STATUS_PATH, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    lines = [
        "Stage 19 RoNeTC three-GPU queue",
        f"updated: {payload['updated_at_utc']}",
        f"queue: {queue_state}",
        f"total epoch progress: {total_epochs}/500 ({100 * total_epochs / 500:.2f}%)",
        "",
    ]
    for row in rows:
        lines.append(
            f"{row['name']:<24} {row['state']:<12} epoch {row['epoch']:>3}/100 "
            f"GPU {str(row['physical_gpu']):<5} rc={row['returncode']}"
        )
    lines.extend(["", f"stop sentinel: {STOP_PATH}", "Test is evaluated only after each run completes 100 epochs."])
    atomic_write(PROGRESS_PATH, "\n".join(lines) + "\n")


def command(task: dict, gpu: str) -> list[str]:
    cmd = [
        sys.executable, str(SELECTOR), "--min-free-gb", "38", "--max-utilization", "10",
        "--allowed", gpu,
    ]
    if task["gpu_count"] > 1:
        cmd.extend(["--count", str(task["gpu_count"])])
    cmd.extend([
        "--", sys.executable, str(TRAINER), "--dataset", task["dataset"],
        "--protocol-id", task["protocol"], "--device", "cuda:0",
    ])
    if task.get("data_parallel"):
        cmd.append("--data-parallel")
    return cmd


def main() -> int:
    if STATUS_PATH.exists() or STOP_PATH.exists():
        raise RuntimeError("queue status or STOP_QUEUE already exists; refusing ambiguous restart")
    LOG_ROOT.mkdir(parents=True, exist_ok=False)
    records = {task["name"]: {"state": "PENDING"} for task in TASKS}
    running: dict[str, tuple[subprocess.Popen, object, str]] = {}
    stopped = False

    def terminate_children(_signum=None, _frame=None):
        nonlocal stopped
        stopped = True
        for process, _handle, _gpu in running.values():
            if process.poll() is None:
                process.terminate()

    signal.signal(signal.SIGTERM, terminate_children)
    signal.signal(signal.SIGINT, terminate_children)

    def launch(task: dict, gpu: str) -> None:
        log_path = LOG_ROOT / f"{task['name']}.log"
        handle = log_path.open("w", encoding="utf-8")
        env = os.environ.copy()
        env.update({"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4"})
        process = subprocess.Popen(command(task, gpu), cwd=PROJECT, stdout=handle, stderr=subprocess.STDOUT, text=True, env=env)
        running[task["name"]] = (process, handle, gpu)
        records[task["name"]].update({
            "state": "RUNNING", "pid": process.pid, "physical_gpu": gpu,
            "started_at_utc": utc_now(), "log": str(log_path.relative_to(ROOT)),
        })

    initial = TASKS[:3]
    for task in initial:
        launch(task, task["gpu"])
        time.sleep(8)
    pending_fourth = TASKS[3]
    fourth_launched = False
    failure = False
    write_status(records, "RUNNING_FIRST_WAVE")

    while running or not fourth_launched:
        if STOP_PATH.exists():
            terminate_children()
        finished = []
        freed_gpu = None
        for name, (process, handle, gpu) in list(running.items()):
            code = process.poll()
            if code is None:
                continue
            handle.close()
            finished.append(name)
            freed_gpu = freed_gpu or gpu
            records[name].update({"state": "SUCCESS" if code == 0 else "FAILED", "returncode": code, "finished_at_utc": utc_now()})
            failure = failure or code != 0
        for name in finished:
            running.pop(name)
        if not fourth_launched and freed_gpu is not None and not failure and not stopped:
            launch(pending_fourth, freed_gpu)
            fourth_launched = True
        elif not fourth_launched and (failure or stopped):
            records[pending_fourth["name"]]["state"] = "NOT_RUN_DUE_TO_FAILURE"
            fourth_launched = True
        write_status(records, "STOPPING" if stopped else "RUNNING_FIRST_WAVE")
        if stopped and not running:
            break
        time.sleep(15)

    first_wave_ok = all(records[task["name"]]["state"] == "SUCCESS" for task in TASKS[:4])
    ustc = TASKS[4]
    if first_wave_ok and not stopped and not STOP_PATH.exists():
        launch(ustc, ustc["gpu"])
        write_status(records, "RUNNING_USTC_THREE_GPU")
        process, handle, _gpu = running[ustc["name"]]
        while process.poll() is None:
            if STOP_PATH.exists():
                terminate_children()
            write_status(records, "STOPPING" if stopped else "RUNNING_USTC_THREE_GPU")
            time.sleep(15)
        code = process.wait(); handle.close(); running.pop(ustc["name"])
        records[ustc["name"]].update({"state": "SUCCESS" if code == 0 else "FAILED", "returncode": code, "finished_at_utc": utc_now()})
    else:
        records[ustc["name"]]["state"] = "NOT_RUN_DUE_TO_FAILURE"

    success = all(records[task["name"]]["state"] == "SUCCESS" for task in TASKS)
    final_state = "SUCCESS" if success else "ABORTED" if stopped or STOP_PATH.exists() else "FAILED"
    write_status(records, final_state)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
