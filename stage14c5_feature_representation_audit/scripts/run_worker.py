#!/usr/bin/env python3
"""Run a fixed queue of Stage 14C.5 training tasks sequentially on one GPU."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

from common import NEW_FEATURES, ROOT, RUNS_ROOT, STAGE14C_ROOT, json_dump, protocol_ids


def balanced_queue(worker_index: int, worker_count: int) -> list[str]:
    with (STAGE14C_ROOT / "stage14c_training_summary.csv").open(encoding="utf-8", newline="") as handle:
        weights = {
            f"{row['setting'].lower()}_seed{row['seed']}": int(row["train_samples"]) * int(row["stop_epoch"])
            for row in csv.DictReader(handle)
        }
    tasks = [(f"{feature}:{protocol_id}", weights[protocol_id]) for feature in NEW_FEATURES for protocol_id in protocol_ids()]
    bins: list[list[str]] = [[] for _ in range(worker_count)]
    loads = [0 for _ in range(worker_count)]
    for task, weight in sorted(tasks, key=lambda item: (-item[1], item[0])):
        target = min(range(worker_count), key=lambda index: (loads[index], index))
        bins[target].append(task)
        loads[target] += weight
    return bins[worker_index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", help="feature:protocol_id")
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--worker-index", type=int)
    parser.add_argument("--worker-count", type=int)
    args = parser.parse_args()
    if args.tasks is None:
        if args.worker_index is None or args.worker_count is None or not 0 <= args.worker_index < args.worker_count:
            parser.error("provide --tasks or a valid --worker-index/--worker-count")
        args.tasks = balanced_queue(args.worker_index, args.worker_count)
    allowed = {f"{feature}:{protocol_id}" for feature in NEW_FEATURES for protocol_id in protocol_ids()}
    if set(args.tasks) - allowed:
        raise RuntimeError(f"unknown tasks: {sorted(set(args.tasks) - allowed)}")
    status_path = ROOT / "worker_status" / f"{args.worker_id}.json"
    status_path.parent.mkdir(exist_ok=True)
    records = []
    for task_index, task in enumerate(args.tasks, start=1):
        feature, protocol_id = task.split(":", 1)
        result_path = RUNS_ROOT / feature / protocol_id / "result.json"
        if result_path.is_file():
            records.append({"task": task, "status": "already_complete"})
            continue
        command = [sys.executable, str(ROOT / "scripts" / "train_one.py"), "--feature", feature, "--protocol-id", protocol_id]
        print(json.dumps({"event": "task_start", "worker_id": args.worker_id, "task_index": task_index, "task_total": len(args.tasks), "task": task}), flush=True)
        completed = subprocess.run(command, check=False)
        status = "success" if completed.returncode == 0 and result_path.is_file() else "failed"
        records.append({"task": task, "status": status, "returncode": completed.returncode})
        json_dump(status_path, {"worker_id": args.worker_id, "records": records, "completed_tasks": sum(r["status"] in {"success", "already_complete"} for r in records), "failed_tasks": sum(r["status"] == "failed" for r in records), "total_tasks": len(args.tasks)})
        if status == "failed":
            print(json.dumps({"event": "task_failed", "task": task, "returncode": completed.returncode}), flush=True)
        else:
            print(json.dumps({"event": "task_complete", "task": task}), flush=True)
    json_dump(status_path, {"worker_id": args.worker_id, "records": records, "completed_tasks": sum(r["status"] in {"success", "already_complete"} for r in records), "failed_tasks": sum(r["status"] == "failed" for r in records), "total_tasks": len(args.tasks), "worker_complete": True})


if __name__ == "__main__":
    main()
