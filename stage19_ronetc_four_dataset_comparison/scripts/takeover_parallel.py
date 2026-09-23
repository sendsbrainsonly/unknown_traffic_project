#!/usr/bin/env python3
"""Run the remaining Stage 19 tasks concurrently on explicit physical GPUs.

Each task can use one independent GPU.  Multi-GPU USTC remains available only
when more than one USTC GPU is explicitly requested.  Every takeover gets
unique metadata and logs so interrupted attempts are never overwritten.
"""
from __future__ import annotations

import argparse
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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_text(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def pid_cmdline(pid: int) -> str:
    path = Path(f"/proc/{pid}/cmdline")
    return path.read_bytes().replace(b"\0", b" ").decode("utf-8", errors="replace") if path.exists() else ""


def current_epoch(dataset: str, protocol: str) -> int:
    path = ROOT / "runs" / dataset / protocol / "history.csv"
    if not path.is_file():
        return 0
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return int(rows[-1]["epoch"]) if rows else 0


def has_success(dataset: str, protocol: str) -> bool:
    path = ROOT / "runs" / dataset / protocol / "result.json"
    if not path.is_file():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("status") == "PASS"
    except (OSError, json.JSONDecodeError):
        return False


def write_progress(
    tasks: list[dict],
    state: str,
    gpus: str,
    vnat_adopted: bool,
    vnat_reused: bool,
    max_physical_gpus: int,
    data_parallel: bool,
) -> None:
    completed = sum(min(int(task["epoch"]), 100) for task in tasks)
    payload = {
        "updated_at_utc": utc_now(),
        "queue_state": state,
        "scheduler": "takeover_parallel",
        "max_concurrent_physical_gpus": max_physical_gpus,
        "active_physical_gpus": len(gpus.split(",")) + (0 if vnat_reused else 1),
        "completed_epochs": completed,
        "planned_epochs": 500,
        "progress_fraction": completed / 500.0,
        "tasks": tasks,
        "automatic_retry": False,
        "takeover": {
            "vnat_adopted": vnat_adopted,
            "vnat_reused_completed": vnat_reused,
            "ustc_physical_gpus": gpus,
            "ustc_data_parallel": data_parallel,
        },
    }
    atomic_json(STATUS_PATH, payload)
    lines = [
        "Stage 19 RoNeTC parallel takeover",
        f"updated: {payload['updated_at_utc']}",
        f"queue: {state}",
        f"total epoch progress: {completed}/500 ({100 * completed / 500:.2f}%)",
        "",
    ]
    for task in tasks:
        lines.append(
            f"{task['name']:<24} {task['state']:<12} epoch {int(task['epoch']):>3}/100 "
            f"GPU {str(task.get('physical_gpu', '')):<5} rc={task.get('returncode')}"
        )
    lines.extend([
        "",
        ("takeover: completed VNAT result reused; only USTC restarted"
         if vnat_reused else
         "takeover: existing VNAT process adopted; USTC started concurrently on separate GPUs"
         if vnat_adopted else
         "takeover: interrupted evidence preserved; VNAT and USTC restarted as independent concurrent tasks"),
        f"USTC mode: {'DataParallel' if data_parallel else 'single GPU, full batch'}",
        f"stop sentinel: {STOP_PATH}",
        "Test is evaluated only after each run completes 100 epochs.",
    ])
    atomic_text(PROGRESS_PATH, "\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vnat-pid", type=int)
    parser.add_argument("--vnat-gpu", default="0")
    parser.add_argument("--gpus", default="1")
    parser.add_argument("--max-physical-gpus", type=int, default=3)
    parser.add_argument("--takeover-id", required=True)
    parser.add_argument("--old-supervisor-pid", type=int, required=True)
    parser.add_argument("--reuse-completed-vnat", action="store_true")
    args = parser.parse_args()

    if not args.takeover_id.replace("_", "").replace("-", "").isalnum():
        raise RuntimeError("takeover id must contain only letters, numbers, hyphens, or underscores")
    ustc_gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not ustc_gpus or len(set(ustc_gpus)) != len(ustc_gpus):
        raise RuntimeError(f"invalid USTC GPU list: {args.gpus}")
    if args.reuse_completed_vnat and args.vnat_pid is not None:
        raise RuntimeError("cannot both reuse completed VNAT and adopt a live VNAT PID")
    if not args.reuse_completed_vnat and args.vnat_gpu in ustc_gpus:
        raise RuntimeError("VNAT and USTC must use disjoint physical GPUs")
    active_physical_gpus = len(ustc_gpus) + (0 if args.reuse_completed_vnat else 1)
    if active_physical_gpus > args.max_physical_gpus:
        raise RuntimeError(
            f"requested {active_physical_gpus} active GPUs exceeds maximum {args.max_physical_gpus}"
        )
    data_parallel = len(ustc_gpus) > 1
    takeover_path = ROOT / f"takeover_{args.takeover_id}.json"
    pre_status_path = ROOT / f"takeover_{args.takeover_id}_pre_status.json"

    expected = "train_eval.py --dataset vnat --protocol-id medium_seed2026"
    if args.vnat_pid is not None:
        actual = pid_cmdline(args.vnat_pid)
        if expected not in actual:
            raise RuntimeError(f"refusing to adopt unexpected PID {args.vnat_pid}: {actual}")
    vnat_output = ROOT / "runs" / "vnat" / "medium_seed2026"
    if args.reuse_completed_vnat and not has_success("vnat", "medium_seed2026"):
        raise RuntimeError("--reuse-completed-vnat requires a PASS result.json")
    if not args.reuse_completed_vnat and args.vnat_pid is None and vnat_output.exists():
        raise RuntimeError(f"refusing to overwrite existing VNAT output: {vnat_output}")
    ustc_output = ROOT / "runs" / "ustc" / "A-2"
    if ustc_output.exists():
        raise RuntimeError(f"refusing to overwrite existing USTC output: {ustc_output}")
    if takeover_path.exists() or pre_status_path.exists():
        raise RuntimeError("takeover evidence already exists; refusing ambiguous restart")

    pre_status = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    atomic_json(pre_status_path, pre_status)
    started_at = utc_now()
    takeover = {
        "status": "RUNNING",
        "started_at_utc": started_at,
        "reason": "user-authorized Stage 19 GPU scheduling change",
        "old_supervisor_pid": args.old_supervisor_pid,
        "vnat_mode": (
            "completed_result_reused" if args.reuse_completed_vnat else
            "adopted" if args.vnat_pid is not None else
            "clean_restart_after_preserved_interruption"
        ),
        "adopted_vnat_pid": args.vnat_pid,
        "vnat_physical_gpu": args.vnat_gpu,
        "ustc_physical_gpus": args.gpus,
        "ustc_mode": "data_parallel" if data_parallel else "single_gpu_full_batch",
        "maximum_physical_gpus": args.max_physical_gpus,
        "active_physical_gpus": active_physical_gpus,
        "pre_takeover_status": str(pre_status_path.relative_to(ROOT)),
    }
    atomic_json(takeover_path, takeover)

    records = {task["name"]: dict(task) for task in pre_status["tasks"]}
    for record in records.values():
        if record["state"] == "SUCCESS":
            record["epoch"] = 100
    vnat_log_path = LOG_ROOT / f"vnat_medium2026_{args.takeover_id}.log"
    vnat_log_handle = None
    vnat_process = None
    if args.reuse_completed_vnat:
        vnat_pid = records["vnat_medium2026"].get("pid")
        vnat_command = None
        vnat_log = records["vnat_medium2026"].get("log")
    elif args.vnat_pid is None:
        vnat_log_handle = vnat_log_path.open("x", encoding="utf-8")
        vnat_command = [
            sys.executable, str(SELECTOR), "--min-free-gb", "38", "--max-utilization", "10",
            "--allowed", args.vnat_gpu, "--", sys.executable, str(TRAINER),
            "--dataset", "vnat", "--protocol-id", "medium_seed2026", "--device", "cuda:0",
        ]
        vnat_env = os.environ.copy()
        vnat_env.update({"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4"})
        vnat_process = subprocess.Popen(
            vnat_command, cwd=PROJECT, stdout=vnat_log_handle, stderr=subprocess.STDOUT,
            text=True, env=vnat_env,
        )
        vnat_pid = vnat_process.pid
        vnat_log = str(vnat_log_path.relative_to(ROOT))
    else:
        vnat_pid = args.vnat_pid
        vnat_command = None
        vnat_log = records["vnat_medium2026"].get("log")
    vnat = records["vnat_medium2026"]
    if args.reuse_completed_vnat:
        vnat.update({"state": "SUCCESS", "epoch": 100, "returncode": 0, "log": vnat_log})
    else:
        vnat.update({
            "state": "RUNNING", "epoch": 0, "pid": vnat_pid, "physical_gpu": args.vnat_gpu,
            "returncode": None, "started_at_utc": utc_now(), "finished_at_utc": None, "log": vnat_log,
        })

    log_path = LOG_ROOT / f"ustc_A2_{args.takeover_id}.log"
    log_handle = log_path.open("x", encoding="utf-8")
    command = [
        sys.executable, str(SELECTOR), "--min-free-gb", "38", "--max-utilization", "10",
        "--allowed", args.gpus, "--count", str(len(ustc_gpus)), "--", sys.executable, str(TRAINER),
        "--dataset", "ustc", "--protocol-id", "A-2", "--device", "cuda:0",
    ]
    if data_parallel:
        command.append("--data-parallel")
    env = os.environ.copy()
    env.update({"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4"})
    process = subprocess.Popen(command, cwd=PROJECT, stdout=log_handle, stderr=subprocess.STDOUT, text=True, env=env)
    ustc = records["ustc_A2"]
    ustc.update({
        "state": "RUNNING", "epoch": 0, "physical_gpu": args.gpus, "pid": process.pid,
        "returncode": None, "started_at_utc": utc_now(), "finished_at_utc": None,
        "log": str(log_path.relative_to(ROOT)),
    })
    takeover.update({
        "vnat_launcher_pid": vnat_pid, "vnat_command": vnat_command, "vnat_log": vnat_log,
        "ustc_launcher_pid": process.pid, "ustc_command": command, "ustc_log": str(log_path.relative_to(ROOT)),
    })
    atomic_json(takeover_path, takeover)

    stopping = False

    def request_stop(_signum=None, _frame=None):
        nonlocal stopping
        stopping = True
        if vnat_process is not None and vnat_process.poll() is None:
            vnat_process.terminate()
        if process.poll() is None:
            process.terminate()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    while True:
        if STOP_PATH.exists():
            request_stop()
            if args.vnat_pid is not None and pid_alive(args.vnat_pid):
                os.kill(args.vnat_pid, signal.SIGTERM)

        if args.reuse_completed_vnat:
            vnat["epoch"] = 100
            vnat_code = 0
        else:
            vnat["epoch"] = current_epoch("vnat", "medium_seed2026")
            vnat_code = vnat_process.poll() if vnat_process is not None else (None if pid_alive(vnat_pid) else -1)
            if vnat_code is not None:
                vnat["state"] = "SUCCESS" if vnat_code == 0 and has_success("vnat", "medium_seed2026") else ("ABORTED" if stopping else "FAILED")
                vnat["returncode"] = vnat_code
                vnat["finished_at_utc"] = vnat.get("finished_at_utc") or utc_now()

        ustc["epoch"] = current_epoch("ustc", "A-2")
        code = process.poll()
        if code is not None:
            ustc["returncode"] = code
            ustc["state"] = "SUCCESS" if code == 0 and has_success("ustc", "A-2") else ("ABORTED" if stopping else "FAILED")
            ustc["finished_at_utc"] = ustc.get("finished_at_utc") or utc_now()

        tasks = [records[task["name"]] for task in pre_status["tasks"]]
        terminal = vnat["state"] != "RUNNING" and ustc["state"] != "RUNNING"
        all_success = all(task["state"] == "SUCCESS" for task in tasks)
        state = "SUCCESS" if terminal and all_success else "ABORTED" if terminal and stopping else "FAILED" if terminal else "TAKEOVER_PARALLEL"
        write_progress(
            tasks,
            state,
            args.gpus,
            args.vnat_pid is not None,
            args.reuse_completed_vnat,
            args.max_physical_gpus,
            data_parallel,
        )
        print(json.dumps({"time": utc_now(), "state": state, "vnat_epoch": vnat["epoch"], "ustc_epoch": ustc["epoch"]}), flush=True)
        if terminal:
            takeover.update({"status": state, "finished_at_utc": utc_now(), "ustc_returncode": code})
            atomic_json(takeover_path, takeover)
            if vnat_log_handle is not None:
                vnat_log_handle.close()
            log_handle.close()
            return 0 if all_success else 1
        time.sleep(15)


if __name__ == "__main__":
    raise SystemExit(main())
