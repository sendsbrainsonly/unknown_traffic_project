#!/usr/bin/env python3
"""At most two live-capacity-selected GPU workers, then all-model Test barrier."""
from __future__ import annotations

import concurrent.futures
import json
import os
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
SKILLS = PROJECT.parent.parent / ".agents/skills"
SELECTOR = SKILLS / "using-superpowers/scripts/select_gpu.py"
PRESERVE = SKILLS / "experiment-data-preservation/scripts"
DATASETS = ("iscx_vpn", "iscx_tor")
LOCK = threading.Lock()
STATE = {"status": "starting", "completed_components": 0, "total_components": 6,
         "max_gpu_workers": 2, "minimum_free_gib": 26, "jobs": []}


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def state_write():
    STATE["updated_at_utc"] = now()
    save(ROOT / "queue_state.json", STATE)


def inventory():
    result = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
                            check=True, capture_output=True, text=True)
    rows = []
    for line in result.stdout.splitlines():
        gpu, free, util = [int(x.strip()) for x in line.split(",")]
        rows.append({"gpu": gpu, "free_mib": free, "utilization": util})
    return rows


def eligible(rows):
    return sorted([r for r in rows if r["free_mib"] >= 26 * 1024 and r["utilization"] <= 95],
                  key=lambda r: r["free_mib"] - 128 * r["utilization"], reverse=True)


def command(action, dataset=None, gpu=None):
    cmd = [sys.executable, str(ROOT / "experiment.py"), action]
    if dataset:
        cmd += ["--dataset", dataset]
    if gpu is not None:
        # Capacity can change between stages; do not bypass a stale-utilization check.
        snapshots = []
        for attempt in range(16):
            rows = inventory()
            snapshots.append({"at_utc": now(), "inventory": rows})
            if any(r["gpu"] == gpu for r in eligible(rows)):
                break
            if attempt == 15:
                raise RuntimeError(f"GPU{gpu} no longer meets26GiB/util<=95: {rows}")
            time.sleep(2)
        save(ROOT / "logs" / f"{action}_{dataset}_gpu_capacity.json", snapshots)
        cmd = [sys.executable, str(SELECTOR), "--min-free-gb", "26", "--allowed", str(gpu), "--", *cmd]
    label = action + ("_" + dataset if dataset else "")
    entry = {"label": label, "command": cmd, "gpu": gpu, "status": "running", "started_at_utc": now()}
    with LOCK:
        STATE["jobs"].append(entry)
        STATE["status"] = "running"
        state_write()
    print(f"START {label} gpu={gpu}", flush=True)
    with (ROOT / "logs" / f"{label}.log").open("x") as log:
        proc = subprocess.Popen(cmd, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT)
        with LOCK:
            entry["pid"] = proc.pid
            state_write()
        code = proc.wait()
    with LOCK:
        entry.update(exit_code=code, finished_at_utc=now(), status="complete" if code == 0 else "failed")
        if code == 0:
            STATE["completed_components"] += 1
        state_write()
    save(ROOT / "logs" / f"{label}_exit.json", entry)
    print(f"EXIT {label} code={code}", flush=True)
    if code:
        raise RuntimeError(f"{label} failed, exit={code}; see logs/{label}.log")


def stage(action, gpu_ids):
    if len(gpu_ids) == 1:
        for dataset in DATASETS:
            command(action, dataset, gpu_ids[0])
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            jobs = [pool.submit(command, action, ds, gpu) for ds, gpu in zip(DATASETS, gpu_ids, strict=True)]
            errors = []
            for future in concurrent.futures.as_completed(jobs):
                try:
                    future.result()
                except BaseException as exc:
                    errors.append(repr(exc))
            if errors:
                raise RuntimeError("; ".join(errors))


def preserve():
    subprocess.run([sys.executable, str(PRESERVE / "refresh_artifact_manifest.py"), str(ROOT), "--hash-max-bytes", "20000000000"],
                   cwd=PROJECT, check=True)
    subprocess.run([sys.executable, str(PRESERVE / "validate_experiment_bundle.py"), str(ROOT), "--verify-hashes"],
                   cwd=PROJECT, check=True)


def main():
    (ROOT / "logs").mkdir(exist_ok=False)
    rows = inventory()
    available = eligible(rows)
    if not available:
        raise RuntimeError(f"no GPU meets26GiB andutil<=95: {rows}")
    gpu_ids = [r["gpu"] for r in available[:2]]
    save(ROOT / "launch_gpu_inventory.json", {"at_utc": now(), "inventory": rows, "selected": gpu_ids,
                                               "min_free_gib": 26, "max_workers": 2})
    STATE.update(started_at_utc=now(), physical_gpu_ids=gpu_ids)
    state_write()
    manifest = json.loads((ROOT / "manifest.json").read_text())
    manifest["execution"]["physical_gpu_ids"] = gpu_ids
    manifest["artifacts"].extend({"path": f"../.tmux-task/stage46_train_20261023/{name}", "scope": "external",
                                "role": "controller-execution-evidence", "sha256": None, "hash_status": "live-log"}
                               for name in ("output.log",))
    save(ROOT / "manifest.json", manifest)
    try:
        stage("train", gpu_ids)
        command("freeze")
        time.sleep(2)
        stage("evaluate", gpu_ids)
        command("finalize")
        STATE.update(status="complete", finished_at_utc=now())
        state_write()
        manifest = json.loads((ROOT / "manifest.json").read_text())
        manifest["execution"]["exit_code"] = 0
        save(ROOT / "manifest.json", manifest)
        preserve()
        print("STAGE46_COMPLETE_INDEPENDENT_REPLAY_PASS", flush=True)
    except BaseException as exc:
        STATE.update(status="failed", error=repr(exc), traceback=traceback.format_exc(), finished_at_utc=now())
        state_write()
        manifest = json.loads((ROOT / "manifest.json").read_text())
        manifest.update(status="failed", next_step="Inspect preserved queue/worker failures; do not rerun into existing output paths.")
        manifest["execution"]["exit_code"] = 1
        save(ROOT / "manifest.json", manifest)
        with (ROOT / "RESULTS.md").open("a") as handle:
            handle.write(f"\n## Execution failure ({now()})\n\n{exc!r}\n\nSee queue_state.json and logs; partial artifacts retained. No test barrier bypass.\n")
        try:
            preserve()
        except BaseException:
            traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
