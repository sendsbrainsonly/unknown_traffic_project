#!/usr/bin/env python3
"""Run a deterministic subset of Stage 11A jobs on one physical GPU."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from stage11a_common import CONFIG_PATH, STAGE_ROOT, read_json


WORKSPACE_ROOT = STAGE_ROOT.parents[2]
GPU_SELECTOR = WORKSPACE_ROOT / "skills" / "using-superpowers" / "scripts" / "select_gpu.py"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("dap", "b0"), required=True)
    parser.add_argument("--queue-index", type=int, required=True)
    parser.add_argument("--queue-count", type=int, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--print-plan", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.queue_index < args.queue_count:
        raise ValueError("queue-index must be in [0, queue-count)")
    config = read_json(CONFIG_PATH)
    jobs: list[dict[str, object]] = []
    for scenario in config["scenarios"]:
        for fold, seed in enumerate(config["seeds"]):
            jobs.append({"scenario": scenario, "fold": fold, "seed": seed})
    jobs = [job for index, job in enumerate(jobs) if index % args.queue_count == args.queue_index]
    print(json.dumps({
        "mode": args.mode,
        "queue_index": args.queue_index,
        "queue_count": args.queue_count,
        "physical_gpu": args.physical_gpu,
        "jobs": jobs,
    }, indent=2), flush=True)
    if args.print_plan:
        return
    script = STAGE_ROOT / "scripts" / ("run_dap.py" if args.mode == "dap" else "evaluate_b0.py")
    for index, job in enumerate(jobs, start=1):
        command = [
            sys.executable,
            "-B",
            str(script),
            "--scenario",
            str(job["scenario"]),
            "--fold",
            str(job["fold"]),
            "--seed",
            str(job["seed"]),
        ]
        print(f"queue_start={index}/{len(jobs)} job={job}", flush=True)
        subprocess.run(
            [
                sys.executable,
                str(GPU_SELECTOR),
                "--min-free-gb",
                "8",
                "--allowed",
                str(args.physical_gpu),
                "--",
                *command,
            ],
            cwd=STAGE_ROOT,
            check=True,
        )
        print(f"queue_complete={index}/{len(jobs)} job={job}", flush=True)
    print(f"queue_finished={args.queue_index}", flush=True)


if __name__ == "__main__":
    main()
