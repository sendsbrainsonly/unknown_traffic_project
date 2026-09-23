#!/usr/bin/env python3
"""Run the remaining frozen Stage 16S grid on selector-assigned physical GPUs."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from stage16s_common import OUT, RUNS, SEEDS, SERVICES, protocol_id, write_json


def worker(gpu: str, tasks: list[tuple[str, int]]) -> list[dict[str, object]]:
    records = []
    log_root = OUT / "launcher_logs"
    log_root.mkdir(parents=True, exist_ok=True)
    script = OUT / "scripts" / "train_evaluate_run.py"
    for pid, seed in tasks:
        if (RUNS / pid / f"seed{seed}" / "SUCCESS").is_file():
            records.append({"protocol_id": pid, "seed": seed, "gpu": gpu, "status": "SKIP_SUCCESS", "exit_code": 0})
            continue
        log_path = log_root / f"{pid}_seed{seed}.log"
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = gpu
        with log_path.open("x", encoding="utf-8") as handle:
            process = subprocess.run(
                [sys.executable, str(script), "--protocol-id", pid, "--seed", str(seed)],
                cwd=OUT.parent, env=env, stdout=handle, stderr=subprocess.STDOUT, check=False,
            )
        records.append({
            "protocol_id": pid, "seed": seed, "gpu": gpu,
            "status": "SUCCESS" if process.returncode == 0 else "FAILED",
            "exit_code": process.returncode, "log_path": str(log_path),
        })
    return records


def main() -> int:
    visible = [value.strip() for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if value.strip()]
    if not visible:
        raise RuntimeError("GPU selector did not provide CUDA_VISIBLE_DEVICES")
    tasks = [(protocol_id(service), seed) for service in SERVICES for seed in SEEDS]
    buckets = [[] for _ in visible]
    for index, task in enumerate(tasks):
        buckets[index % len(visible)].append(task)
    records = []
    with ThreadPoolExecutor(max_workers=len(visible)) as executor:
        futures = [executor.submit(worker, gpu, bucket) for gpu, bucket in zip(visible, buckets)]
        for future in as_completed(futures):
            records.extend(future.result())
            completed = sum(record["exit_code"] == 0 for record in records)
            print(f"worker_complete completed_tasks={completed}/{len(tasks)}", flush=True)
    records.sort(key=lambda row: (row["protocol_id"], row["seed"]))
    summary = {
        "selected_physical_gpu_ids": visible,
        "tasks": records,
        "successful_or_skipped": sum(row["exit_code"] == 0 for row in records),
        "failed": sum(row["exit_code"] != 0 for row in records),
    }
    write_json(OUT / "training_grid_execution.json", summary)
    print(json.dumps(summary, indent=2))
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
