#!/usr/bin/env python3
"""Run the frozen Stage 14C-6 15-protocol grid on at most four GPUs."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "scripts/train_one.py"
PROTOCOLS = [
    f"{setting}_seed{seed}"
    for setting in ("low", "medium", "high")
    for seed in range(2022, 2027)
]


def write_status(status: dict[str, object]) -> None:
    (ROOT / "run_queue_status.json").write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def run_one(protocol_id: str, physical_gpu: str) -> dict[str, object]:
    log_path = ROOT / "run_logs" / f"{protocol_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = physical_gpu
    environment["OMP_NUM_THREADS"] = "1"
    environment["MKL_NUM_THREADS"] = "1"
    started = time.time()
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.run(
            [sys.executable, str(TRAIN), "--protocol-id", protocol_id],
            cwd=ROOT.parent,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return {
        "protocol_id": protocol_id,
        "physical_gpu": physical_gpu,
        "exit_code": process.returncode,
        "duration_seconds": time.time() - started,
        "log_path": str(log_path),
        "status": "success" if process.returncode == 0 else "failed",
    }


def main() -> None:
    selected = [value.strip() for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if value.strip()]
    if not selected:
        raise RuntimeError("GPU selector did not provide CUDA_VISIBLE_DEVICES")
    selected = selected[:4]
    if len(selected) > 4:
        raise RuntimeError("refusing to use more than four GPUs")
    if any((ROOT / "runs" / protocol).exists() or (ROOT / "checkpoints" / f"{protocol}_best.pt").exists() for protocol in PROTOCOLS):
        raise RuntimeError("fresh formal grid required; refusing to overwrite existing run evidence")

    pending = list(PROTOCOLS)
    completed: list[dict[str, object]] = []
    active: dict[Future[dict[str, object]], tuple[str, str]] = {}
    started = time.time()
    status: dict[str, object] = {
        "status": "running",
        "protocols_total": len(PROTOCOLS),
        "selected_physical_gpus": selected,
        "max_parallel_runs": len(selected),
        "pending": pending.copy(),
        "active": [],
        "completed": [],
        "started_at_unix": started,
    }
    write_status(status)

    with ThreadPoolExecutor(max_workers=len(selected)) as executor:
        for gpu in selected:
            if not pending:
                break
            protocol = pending.pop(0)
            future = executor.submit(run_one, protocol, gpu)
            active[future] = (protocol, gpu)
        while active:
            status["pending"] = pending.copy()
            status["active"] = [
                {"protocol_id": protocol, "physical_gpu": gpu}
                for protocol, gpu in active.values()
            ]
            status["completed"] = completed
            write_status(status)
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                protocol, gpu = active.pop(future)
                try:
                    result = future.result()
                except Exception as exc:  # preserve an orchestration failure explicitly
                    result = {
                        "protocol_id": protocol,
                        "physical_gpu": gpu,
                        "exit_code": -1,
                        "duration_seconds": 0.0,
                        "log_path": str(ROOT / "run_logs" / f"{protocol}.log"),
                        "status": "failed",
                        "orchestration_error": repr(exc),
                    }
                completed.append(result)
                print(json.dumps({"event": "run_finished", **result}, sort_keys=True), flush=True)
                if pending:
                    next_protocol = pending.pop(0)
                    next_future = executor.submit(run_one, next_protocol, gpu)
                    active[next_future] = (next_protocol, gpu)

    failures = [row for row in completed if row["status"] != "success"]
    status.update({
        "status": "success" if not failures and len(completed) == len(PROTOCOLS) else "failed",
        "pending": [],
        "active": [],
        "completed": completed,
        "failures": failures,
        "duration_seconds": time.time() - started,
        "finished_at_unix": time.time(),
    })
    write_status(status)
    print(json.dumps({
        "event": "grid_complete",
        "status": status["status"],
        "runs_completed": len(completed),
        "failures": len(failures),
        "selected_physical_gpus": selected,
    }, sort_keys=True), flush=True)
    if failures or len(completed) != len(PROTOCOLS):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
