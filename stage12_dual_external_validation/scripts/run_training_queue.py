#!/usr/bin/env python3
"""Run a deterministic shard of Stage 12 pre-test training tasks."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from stage12_common import STAGE_ROOT, load_config


def plan(selected_datasets: set[str] | None = None) -> list[tuple[str, str, int]]:
    config = load_config()
    tasks = []
    for dataset in config["datasets"]:
        if selected_datasets is not None and dataset not in selected_datasets:
            continue
        protocol_path = STAGE_ROOT / "protocol" / dataset / "unknown_class_protocol.json"
        if not protocol_path.is_file():
            raise FileNotFoundError(protocol_path)
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        for setting in protocol["settings"]:
            for seed in config["training_seeds"]:
                tasks.append((dataset, setting, int(seed)))
    return tasks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue-index", type=int, required=True)
    parser.add_argument("--queue-count", type=int, required=True)
    parser.add_argument("--print-plan", action="store_true")
    parser.add_argument("--dataset", action="append", choices=("iscx_vpn", "iscx_tor"))
    args = parser.parse_args()
    if not 0 <= args.queue_index < args.queue_count:
        raise ValueError("queue index outside queue count")
    tasks = plan(set(args.dataset) if args.dataset else None)
    queue = [task for index, task in enumerate(tasks) if index % args.queue_count == args.queue_index]
    print(json.dumps({
        "queue_index": args.queue_index,
        "queue_count": args.queue_count,
        "tasks_total": len(tasks),
        "queue_tasks": queue,
    }, indent=2), flush=True)
    if args.print_plan:
        return 0
    trainer = Path(__file__).with_name("train_pretest.py")
    finalizer = Path(__file__).with_name("finalize_run_evidence.py")
    workspace_root = STAGE_ROOT.parents[2]
    preservation_scripts = workspace_root / ".agents" / "skills" / "experiment-data-preservation" / "scripts"
    refresher = preservation_scripts / "refresh_artifact_manifest.py"
    validator = preservation_scripts / "validate_experiment_bundle.py"
    for position, (dataset, setting, seed) in enumerate(queue, start=1):
        run_dir = STAGE_ROOT / "runs" / dataset / setting / f"seed{seed}"
        ready = run_dir / "READY_FOR_ONE_SHOT_TEST"
        if ready.is_file():
            print(f"queue_skip={position}/{len(queue)} task={dataset}/{setting}/{seed}", flush=True)
        else:
            print(f"queue_start={position}/{len(queue)} task={dataset}/{setting}/{seed}", flush=True)
            subprocess.run([
                sys.executable, "-B", str(trainer), "--dataset", dataset,
                "--setting", setting, "--seed", str(seed),
            ], cwd=STAGE_ROOT, check=True)
        if not ready.is_file():
            raise RuntimeError(f"training returned without ready marker: {ready}")
        subprocess.run([sys.executable, "-B", str(finalizer), str(run_dir), "--phase", "pretest"], cwd=STAGE_ROOT, check=True)
        subprocess.run([sys.executable, "-B", str(refresher), str(run_dir)], cwd=STAGE_ROOT, check=True)
        subprocess.run([sys.executable, "-B", str(validator), str(run_dir), "--verify-hashes"], cwd=STAGE_ROOT, check=True)
        print(f"queue_complete={position}/{len(queue)} task={dataset}/{setting}/{seed}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
