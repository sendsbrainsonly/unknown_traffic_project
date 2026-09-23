#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

from common import ROOT, pilot_specs, write_json

SELECTOR = Path("/home/birkenwald/data/SEU-WXY/WXY-SEU/TrafficClassifier/.agents/skills/using-superpowers/scripts/select_gpu.py")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-ids", nargs="+", required=True)
    args = parser.parse_args()
    jobs = deque((window, spec["dataset"], spec["protocol_id"]) for window in (8, 16, 32) for spec in pilot_specs())
    active = {}
    completed, failed = [], []
    status_path = ROOT / "grid_status.json"

    while jobs or active:
        for gpu in args.gpu_ids:
            if gpu in active or not jobs:
                continue
            window, dataset, protocol_id = jobs.popleft()
            output = ROOT / "runs" / f"T{window}" / dataset / protocol_id
            result = output / "result.json"
            if result.exists():
                completed.append({"window": window, "dataset": dataset, "protocol_id": protocol_id, "status": "SKIPPED_COMPLETE"})
                continue
            if output.exists():
                failed.append({"window": window, "dataset": dataset, "protocol_id": protocol_id, "status": "REFUSED_PARTIAL_OUTPUT"})
                continue
            output.parent.mkdir(parents=True, exist_ok=True)
            log = output.parent / f"{protocol_id}.T{window}.launcher.log"
            handle = log.open("w", encoding="utf-8")
            command = [
                sys.executable, str(SELECTOR), "--allowed", str(gpu), "--count", "1", "--min-free-gb", "1",
                "--", sys.executable, str(ROOT / "scripts" / "run_window_pilot.py"),
                "--dataset", dataset, "--protocol-id", protocol_id, "--window", str(window), "--device", "cuda:0",
            ]
            env = os.environ.copy()
            env.update({"OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "NUMEXPR_NUM_THREADS": "2"})
            process = subprocess.Popen(command, cwd=ROOT.parent, stdout=handle, stderr=subprocess.STDOUT, env=env)
            active[gpu] = {"process": process, "handle": handle, "job": (window, dataset, protocol_id), "log": str(log), "started": time.time()}
            print(json.dumps({"event": "started", "gpu": gpu, "window": window, "dataset": dataset, "protocol_id": protocol_id, "pid": process.pid}), flush=True)
        for gpu, item in list(active.items()):
            code = item["process"].poll()
            if code is None:
                continue
            item["handle"].close()
            window, dataset, protocol_id = item["job"]
            row = {"window": window, "dataset": dataset, "protocol_id": protocol_id, "gpu": gpu, "exit_code": code, "elapsed_seconds": time.time() - item["started"], "log": item["log"]}
            (completed if code == 0 else failed).append(row)
            del active[gpu]
            print(json.dumps({"event": "finished", **row}), flush=True)
        write_json(status_path, {"queued": len(jobs), "active": [{"gpu": gpu, "window": item["job"][0], "dataset": item["job"][1], "protocol_id": item["job"][2], "pid": item["process"].pid, "elapsed_seconds": time.time() - item["started"]} for gpu, item in active.items()], "completed": completed, "failed": failed})
        if active:
            time.sleep(5)
    if failed:
        raise SystemExit(f"{len(failed)} jobs failed")
    print(json.dumps({"status": "PASS", "completed": len(completed)}, sort_keys=True))


if __name__ == "__main__":
    main()
