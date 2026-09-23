#!/usr/bin/env python3
"""Run a deterministic shard of the 15 CPU-only Stage 13A-2 evaluations."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

from stage13a2_common import CONFIG_PATH, STAGE_ROOT, read_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue-index", type=int, required=True)
    parser.add_argument("--queue-count", type=int, required=True)
    parser.add_argument("--print-plan", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.queue_index < args.queue_count:
        raise ValueError("queue-index must be in [0, queue-count)")
    config = read_json(CONFIG_PATH)
    tasks = [
        (scenario, fold, int(seed))
        for scenario in ("a1", "a2", "a3")
        for fold, seed in enumerate(config["seeds"])
    ]
    selected = [task for index, task in enumerate(tasks) if index % args.queue_count == args.queue_index]
    print(json.dumps({"queue_index": args.queue_index, "queue_count": args.queue_count, "tasks": selected}, indent=2))
    if args.print_plan:
        return
    for number, (scenario, fold, seed) in enumerate(selected, start=1):
        print(f"stage13a2_progress={number}/{len(selected)} scenario={scenario} fold={fold} seed={seed}", flush=True)
        subprocess.run(
            [
                sys.executable,
                "-B",
                str(STAGE_ROOT / "scripts" / "evaluate_run.py"),
                "--scenario",
                scenario,
                "--fold",
                str(fold),
                "--seed",
                str(seed),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()

