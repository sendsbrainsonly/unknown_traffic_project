#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from stage22_common import OUT, RUN_ROOT, write_json


JOBS = [(dataset, seed) for dataset in ("iscx_vpn", "iscx_tor") for seed in (2022, 2023)]


def write_progress(state: dict) -> None:
    lines = [
        "Stage 22 pretrained TrafficFormer + E3 closed-set comparison",
        f"updated_utc={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
        f"completed={sum(value['status'] == 'SUCCESS' for value in state.values())}/{len(JOBS)}",
        f"failed={sum(value['status'] == 'FAILED' for value in state.values())}",
        f"running={sum(value['status'] == 'RUNNING' for value in state.values())}",
        "",
    ]
    for dataset, seed in JOBS:
        item = state[f"{dataset}:{seed}"]
        lines.append(f"{dataset} seed{seed}: {item['status']} device={item.get('device', '-')} exit={item.get('exit_code', '-')}")
    tmp = OUT / "progress.txt.tmp"
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    tmp.replace(OUT / "progress.txt")
    write_json(OUT / "queue_status.json", state)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-indices", default="0,1,2")
    args = parser.parse_args()
    devices = [int(value) for value in args.device_indices.split(",") if value.strip()]
    if not 1 <= len(devices) <= 3:
        raise ValueError("queue requires one to three selected logical GPU indices")
    visible = [value.strip() for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if value.strip()]
    logs = OUT / "launcher_logs"
    logs.mkdir(parents=True, exist_ok=True)
    state = {f"{dataset}:{seed}": {"status": "PENDING", "dataset": dataset, "seed": seed} for dataset, seed in JOBS}
    active: dict[int, tuple[subprocess.Popen, object, str]] = {}
    pending = list(JOBS)
    write_progress(state)
    while pending or active:
        for device in devices:
            if device in active or not pending:
                continue
            dataset, seed = pending.pop(0)
            key = f"{dataset}:{seed}"
            log_path = logs / f"{dataset}_seed{seed}.log"
            handle = log_path.open("w", encoding="utf-8")
            command = [
                sys.executable,
                str(OUT / "scripts" / "train_closed_run.py"),
                "--dataset", dataset,
                "--seed", str(seed),
                "--device", f"cuda:{device}",
            ]
            process = subprocess.Popen(command, cwd=OUT.parent, stdout=handle, stderr=subprocess.STDOUT, text=True)
            active[device] = (process, handle, key)
            physical_gpu = visible[device] if device < len(visible) else None
            state[key].update({"status": "RUNNING", "device": f"cuda:{device}", "physical_gpu": physical_gpu, "pid": process.pid, "log": str(log_path)})
            write_progress(state)
        time.sleep(5)
        for device, (process, handle, key) in list(active.items()):
            return_code = process.poll()
            if return_code is None:
                continue
            handle.close()
            success = return_code == 0 and (RUN_ROOT / state[key]["dataset"] / f"seed{state[key]['seed']}" / "SUCCESS").is_file()
            state[key].update({"status": "SUCCESS" if success else "FAILED", "exit_code": return_code})
            del active[device]
            write_progress(state)

    if any(item["status"] != "SUCCESS" for item in state.values()):
        raise SystemExit(1)
    for script in ("aggregate.py", "verify_completion.py", "finalize.py"):
        subprocess.run([sys.executable, str(OUT / "scripts" / script)], cwd=OUT.parent, check=True)


if __name__ == "__main__":
    main()
