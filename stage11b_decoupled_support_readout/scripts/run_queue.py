#!/usr/bin/env python3
"""Run a deterministic shard of the 15 frozen Stage 11B evaluations."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from stage11b_common import CONFIG_PATH, STAGE_ROOT, read_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue-index", type=int, required=True)
    parser.add_argument("--queue-count", type=int, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    args = parser.parse_args()
    if not 0 <= args.queue_index < args.queue_count:
        raise ValueError("queue-index must be in [0, queue-count)")
    config = read_json(CONFIG_PATH)
    jobs = [
        {"scenario": scenario, "fold": fold, "seed": int(seed)}
        for scenario in ("a1", "a2", "a3")
        for fold, seed in enumerate(config["seeds"])
    ]
    jobs = [job for index, job in enumerate(jobs) if index % args.queue_count == args.queue_index]
    print(
        json.dumps(
            {
                "queue_index": args.queue_index,
                "queue_count": args.queue_count,
                "physical_gpu": args.physical_gpu,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "jobs": jobs,
                "new_encoder_training": False,
            },
            indent=2,
        ),
        flush=True,
    )
    script = STAGE_ROOT / "scripts" / "evaluate_frozen.py"
    for index, job in enumerate(jobs, start=1):
        print(f"queue_start={index}/{len(jobs)} job={job}", flush=True)
        command = [
            sys.executable,
            "-B",
            str(script),
            "--scenario",
            job["scenario"],
            "--fold",
            str(job["fold"]),
            "--seed",
            str(job["seed"]),
        ]
        subprocess.run(command, cwd=STAGE_ROOT, check=True)
        print(f"queue_complete={index}/{len(jobs)} job={job}", flush=True)
    print(f"queue_finished={args.queue_index}", flush=True)


if __name__ == "__main__":
    main()
