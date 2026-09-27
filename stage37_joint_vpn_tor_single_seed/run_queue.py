#!/usr/bin/env python3
"""Run the two frozen single-seed joint experiments sequentially on one GPU."""
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNNER = ROOT / "run_joint.py"
PROGRESS = ROOT / "queue_progress.json"
DATASETS = ("iscx_vpn", "iscx_tor")


def update(**fields):
    payload = json.loads(PROGRESS.read_text()) if PROGRESS.is_file() else {}
    payload.update(fields)
    payload["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    temporary = PROGRESS.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(PROGRESS)
    print(json.dumps(payload, sort_keys=True), flush=True)


def main():
    if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
        raise RuntimeError("GPU selection wrapper required")
    update(status="RUNNING", selected_visible_gpu=os.environ["CUDA_VISIBLE_DEVICES"],
           datasets=list(DATASETS), seed=2022, complete=[])
    done = []
    for dataset in DATASETS:
        for action in ("train", "evaluate"):
            update(status="RUNNING", current_dataset=dataset, current_action=action, complete=done)
            subprocess.run([sys.executable, str(RUNNER), action, "--dataset", dataset],
                           cwd=ROOT.parent, check=True)
        done.append(dataset)
        update(status="RUNNING", current_dataset=None, current_action=None, complete=done)
    update(status="PASS", complete=done)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        update(status="FAIL", error=repr(exc), traceback=traceback.format_exc())
        raise
