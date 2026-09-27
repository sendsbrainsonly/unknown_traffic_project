#!/usr/bin/env python3
"""Fixed Stage24 16-run Known-only pilot with at most three physical GPUs."""
from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path


OUT = Path(__file__).resolve().parents[1]
PROJECT = OUT.parent
WORKSPACE = PROJECT.parents[1]
SKILLS = WORKSPACE / "skills"
INIT = SKILLS / "experiment-data-preservation" / "scripts" / "init_experiment_bundle.py"
REFRESH = SKILLS / "experiment-data-preservation" / "scripts" / "refresh_artifact_manifest.py"
VALIDATE = SKILLS / "experiment-data-preservation" / "scripts" / "validate_experiment_bundle.py"
GPU_SELECTOR = SKILLS / "using-superpowers" / "scripts" / "select_gpu.py"
TRAINER = OUT / "scripts" / "train_pilot.py"
SESSION = "codex_stage24_pilot_queue_20260924"
METHODS = ("S1", "G1", "G2", "G2-shuffle")
DATASETS = ("iscx_vpn", "iscx_tor")
SEEDS = (2022, 2023)
GPUS = (0, 1, 2)


def run_dir(dataset: str, seed: int, method: str) -> Path:
    return OUT / "runs" / "pilot" / dataset / f"seed{seed}" / method


def record_status(states: dict[str, dict], lock: threading.Lock) -> None:
    with lock:
        counts: dict[str, int] = {}
        for item in states.values():
            status = item["status"]
            counts[status] = counts.get(status, 0) + 1
        progress = {"total": len(states), "counts": counts,
                    "runs": states, "updated_unix_time": time.time(),
                    "max_physical_gpus": len(GPUS), "physical_gpus": list(GPUS)}
        tmp = OUT / "progress.json.tmp"
        tmp.write_text(json.dumps(progress, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(OUT / "progress.json")


def invoke(command: list[str], log_path: Path | None = None) -> int:
    env = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS"):
        env[name] = "2"
    if log_path:
        with log_path.open("w", encoding="utf-8") as handle:
            proc = subprocess.run(command, cwd=PROJECT, env=env, stdout=handle,
                                  stderr=subprocess.STDOUT, check=False)
    else:
        proc = subprocess.run(command, cwd=PROJECT, env=env, check=False)
    return proc.returncode


def worker(gpu: int, jobs: queue.Queue, states: dict[str, dict],
           lock: threading.Lock, stop: threading.Event) -> None:
    while not stop.is_set():
        try:
            dataset, seed, method = jobs.get_nowait()
        except queue.Empty:
            return
        key = f"{dataset}/{seed}/{method}"
        output = run_dir(dataset, seed, method)
        try:
            with lock:
                states[key] = {"status": "initializing", "physical_gpu": gpu,
                               "output": str(output.relative_to(PROJECT))}
            record_status(states, lock)
            init_cmd = [sys.executable, str(INIT), str(output),
                        "--id", f"stage24-pilot-{dataset}-{seed}-{method}",
                        "--objective", "Known-only frozen-E1 structural branch ablation",
                        "--experiment-type", "ablation", "--claim-scope", "diagnostic",
                        "--status", "running", "--tmux-session", SESSION]
            if invoke(init_cmd) != 0:
                raise RuntimeError("experiment bundle initialization failed")
            with lock:
                states[key]["status"] = "training"
            record_status(states, lock)
            train_cmd = [sys.executable, str(GPU_SELECTOR), "--min-free-gb", "4",
                         "--allowed", str(gpu), "--", sys.executable, str(TRAINER),
                         "--dataset", dataset, "--seed", str(seed), "--method", method,
                         "--output", str(output), "--device", "cuda:0"]
            train_exit = invoke(train_cmd, output / "train.log")
            if train_exit != 0 or not (output / "SUCCESS").is_file():
                raise RuntimeError(f"training failed, exit={train_exit}; see {output / 'train.log'}")
            if invoke([sys.executable, str(REFRESH), str(output)]) != 0:
                raise RuntimeError("artifact inventory refresh failed")
            if invoke([sys.executable, str(VALIDATE), str(output), "--verify-hashes"]) != 0:
                raise RuntimeError("experiment bundle validation failed")
            result = json.loads((output / "result.json").read_text())
            with lock:
                states[key].update({"status": "success", "val_macro_f1":
                                    result["validation"][method]["macro_f1"],
                                    "wall_seconds": result["wall_seconds"]})
            print(json.dumps({"event": "run_verified", "key": key, "gpu": gpu,
                              "val_macro_f1": result["validation"][method]["macro_f1"]}),
                  flush=True)
        except Exception as exc:
            with lock:
                states[key].update({"status": "failed", "error": str(exc)})
            print(json.dumps({"event": "run_failed", "key": key,
                              "gpu": gpu, "error": str(exc)}), flush=True)
            stop.set()
        finally:
            record_status(states, lock)
            jobs.task_done()


def main() -> None:
    if not (OUT / "preflight.json").is_file():
        raise RuntimeError("preflight result missing")
    if json.loads((OUT / "preflight.json").read_text())["status"] != "PASS":
        raise RuntimeError("preflight did not pass")
    if (OUT / "QUEUE_COMPLETE").exists() or (OUT / "QUEUE_FAILED").exists():
        raise RuntimeError("queue terminal marker already exists; do not overwrite")
    jobs = queue.Queue()
    states: dict[str, dict] = {}
    for dataset in DATASETS:
        for seed in SEEDS:
            for method in METHODS:
                key = f"{dataset}/{seed}/{method}"
                output = run_dir(dataset, seed, method)
                if (output / "SUCCESS").is_file():
                    if not (output / "result.json").is_file():
                        raise RuntimeError(f"success marker without result: {output}")
                    result = json.loads((output / "result.json").read_text())
                    states[key] = {"status": "success", "physical_gpu": 0,
                                   "output": str(output.relative_to(PROJECT)),
                                   "val_macro_f1": result["validation"][method]["macro_f1"],
                                   "wall_seconds": result["wall_seconds"]}
                else:
                    if output.exists():
                        raise RuntimeError(f"partial directory exists: {output}; preserve and inspect")
                    states[key] = {"status": "queued", "physical_gpu": None,
                                   "output": str(output.relative_to(PROJECT))}
                    jobs.put((dataset, seed, method))
    lock = threading.Lock()
    stop = threading.Event()
    record_status(states, lock)
    threads = [threading.Thread(target=worker, args=(gpu, jobs, states, lock, stop),
                                name=f"gpu{gpu}", daemon=False) for gpu in GPUS]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if stop.is_set() or any(item["status"] != "success" for item in states.values()):
        (OUT / "QUEUE_FAILED").write_text("At least one run failed or remained queued; see progress.json.\n",
                                           encoding="utf-8")
        raise RuntimeError("Stage24 pilot queue incomplete; evidence retained")
    (OUT / "QUEUE_COMPLETE").write_text("16/16 preregistered pilot bundles verified.\n",
                                         encoding="utf-8")
    print(json.dumps({"event": "queue_complete", "verified_runs": len(states)}), flush=True)


if __name__ == "__main__":
    main()
