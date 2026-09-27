#!/usr/bin/env python3
"""Replay all completed Stage28 checkpoints and refresh run evidence."""
from __future__ import annotations

import json
import subprocess
import sys

from preflight import OUT, PROJECT

PRESERVE = PROJECT / ".." / ".." / "skills" / "experiment-data-preservation" / "scripts"


def call(args: list[str]) -> None:
    result = subprocess.run(args, cwd=PROJECT, text=True, check=False,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(result.stdout[-1200:], flush=True)
    if result.returncode:
        raise RuntimeError(f"command failed: {args}; exit {result.returncode}")


def main() -> None:
    state = json.loads((OUT / "queue_progress.json").read_text())
    if state["status"] != "success":
        raise RuntimeError("queue must finish successfully before checkpoint replay")
    completed = 0
    for dataset in ("iscx_vpn", "iscx_tor"):
        for encoder in (2022, 2023):
            for train in range(2022, 2027):
                run = OUT / "runs" / dataset / f"encoder{encoder}_train{train}"
                if not (run / "independent_checkpoint_replay.json").exists():
                    call([sys.executable, str(OUT / "replay_checkpoints.py"),
                          "--dataset", dataset, "--encoder-seed", str(encoder),
                          "--training-seed", str(train)])
                replay = json.loads((run / "independent_checkpoint_replay.json").read_text())
                if replay["status"] != "PASS" or len(replay["methods"]) != 6:
                    raise RuntimeError(f"checkpoint replay incomplete: {run}")
                call([sys.executable, str(PRESERVE / "refresh_artifact_manifest.py"), str(run)])
                call([sys.executable, str(PRESERVE / "validate_experiment_bundle.py"), str(run),
                      "--verify-hashes"])
                completed += 1
                print(json.dumps({"checkpoint_replayed": completed, "total": 20}), flush=True)
    (OUT / "checkpoint_replay_summary.json").write_text(json.dumps({
        "status": "PASS", "units": completed, "methods_per_unit": 6,
        "test_usage": 0, "unknown_usage": 0,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
