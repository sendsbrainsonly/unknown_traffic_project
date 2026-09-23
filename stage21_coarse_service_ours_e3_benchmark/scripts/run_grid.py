#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from stage21_common import OUT, RUN_ROOT, write_json


TASKS = [(dataset, seed) for dataset in ("iscx_vpn", "iscx_tor") for seed in (2022, 2023)]


def worker(gpu: str, tasks: list[tuple[str, int]]) -> list[dict]:
    records = []
    log_root = OUT / "launcher_logs"
    log_root.mkdir(parents=True, exist_ok=True)
    script = OUT / "scripts" / "train_closed_run.py"
    for dataset, seed in tasks:
        run = RUN_ROOT / dataset / f"seed{seed}"
        if (run / "SUCCESS").is_file():
            records.append({"dataset": dataset, "seed": seed, "gpu": gpu, "status": "SKIP_SUCCESS", "exit_code": 0})
            continue
        log_path = log_root / f"{dataset}_seed{seed}.log"
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = gpu
        with log_path.open("x", encoding="utf-8") as handle:
            process = subprocess.run(
                [sys.executable, str(script), "--dataset", dataset, "--seed", str(seed)],
                cwd=OUT.parent, env=env, stdout=handle, stderr=subprocess.STDOUT, check=False,
            )
        records.append({"dataset": dataset, "seed": seed, "gpu": gpu, "status": "SUCCESS" if process.returncode == 0 else "FAILED", "exit_code": process.returncode, "log_path": str(log_path)})
        write_json(OUT / "grid_progress.json", {"tasks": records, "completed": len(records), "total": len(TASKS)})
        if process.returncode != 0:
            break
    return records


def main() -> int:
    visible = [value.strip() for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if value.strip()]
    if not visible:
        raise RuntimeError("GPU selector did not provide CUDA_VISIBLE_DEVICES")
    buckets = [[] for _ in visible]
    for index, task in enumerate(TASKS):
        buckets[index % len(visible)].append(task)
    records = []
    with ThreadPoolExecutor(max_workers=len(visible)) as executor:
        futures = [executor.submit(worker, gpu, bucket) for gpu, bucket in zip(visible, buckets)]
        for future in as_completed(futures):
            records.extend(future.result())
            print(json.dumps({"workers_completed": len(records), "total": len(TASKS)}), flush=True)
    records.sort(key=lambda row: (row["dataset"], row["seed"]))
    summary = {"selected_physical_gpu_ids": visible, "tasks": records, "successful_or_skipped": sum(row["exit_code"] == 0 for row in records), "failed": sum(row["exit_code"] != 0 for row in records)}
    write_json(OUT / "training_grid_execution.json", summary)
    print(json.dumps(summary, indent=2))
    return 1 if summary["failed"] or len(records) != len(TASKS) else 0


if __name__ == "__main__":
    raise SystemExit(main())
