#!/usr/bin/env python3
"""Run a deterministic shard of the 15 CPU-only Stage 11C diagnoses."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

from stage11c_common import SCENARIOS, SEEDS, STAGE_ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue-index", type=int, required=True)
    parser.add_argument("--queue-count", type=int, required=True)
    args = parser.parse_args()
    if not 0 <= args.queue_index < args.queue_count:
        raise ValueError("queue-index must be in [0, queue-count)")
    jobs = [
        {"scenario": scenario, "fold": fold, "seed": seed}
        for scenario in SCENARIOS
        for fold, seed in enumerate(SEEDS)
    ]
    jobs = [job for index, job in enumerate(jobs) if index % args.queue_count == args.queue_index]
    print(json.dumps({"queue_index": args.queue_index, "queue_count": args.queue_count, "jobs": jobs, "gpu_used": False}, indent=2), flush=True)
    script = STAGE_ROOT / "scripts" / "analyze_run.py"
    for index, job in enumerate(jobs, 1):
        print(f"queue_start={index}/{len(jobs)} {job}", flush=True)
        subprocess.run(
            [sys.executable, "-B", str(script), "--scenario", job["scenario"], "--fold", str(job["fold"]), "--seed", str(job["seed"])],
            cwd=STAGE_ROOT,
            check=True,
        )
        print(f"queue_complete={index}/{len(jobs)} {job}", flush=True)


if __name__ == "__main__":
    main()
