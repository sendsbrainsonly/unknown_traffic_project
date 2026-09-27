#!/usr/bin/env python3
"""Sequential Stage32 single-seed training queue; fail closed before Test."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from common import DATASETS, ROOT

SELECTOR = ROOT.parent.parent.parent / "skills" / "using-superpowers" / "scripts" / "select_gpu.py"


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_progress(value: dict) -> None:
    path = ROOT / "progress.json"
    temporary = ROOT / "progress.json.tmp"
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    if (ROOT / "progress.json").exists():
        raise FileExistsError("Stage32 progress already exists; refusing to restart queue")
    audit = json.loads((ROOT / "preflight.json").read_text())
    if audit["status"] != "PASS" or audit["test_feature_values_loaded"] != 0:
        raise RuntimeError("Stage32 Known-only audit missing or failed")
    state = {"status": "TRAINING", "updated_at_utc": utc(), "datasets": {
        dataset: {"status": "PENDING", "train_log": None, "val_macro_f1": None}
        for dataset in DATASETS}, "test_features_opened": 0,
        "completed_heads": 0, "total_heads": len(DATASETS),
        "next_step": "Train one coarse T0 head per frozen dataset, then stop before Test"}
    save_progress(state)
    for dataset in DATASETS:
        run = ROOT / "runs" / dataset
        state["datasets"][dataset]["status"] = "RUNNING"
        state["datasets"][dataset]["started_at_utc"] = utc()
        state["updated_at_utc"] = utc()
        save_progress(state)
        command = [sys.executable, str(SELECTOR), "--min-free-gb", "8",
                   "--", sys.executable, str(ROOT / "train_one.py"),
                   "--dataset", dataset]
        state["datasets"][dataset]["command"] = "live-select-GPU then train_one.py --dataset " + dataset
        save_progress(state)
        print(json.dumps({"event": "START", "dataset": dataset, "at": utc()}), flush=True)
        result = subprocess.run(command, cwd=ROOT.parent, check=False)
        state["datasets"][dataset]["exit_code"] = result.returncode
        state["datasets"][dataset]["finished_at_utc"] = utc()
        if result.returncode != 0 or not (run / "SUCCESS").is_file():
            state["datasets"][dataset]["status"] = "FAILED"
            state["status"] = "FAILED_BEFORE_TEST"
            state["next_step"] = "Read the queue tmux log and preserved run/FAILURE.json; do not open Test"
            save_progress(state)
            raise RuntimeError(f"{dataset} training failed with exit {result.returncode}")
        report = json.loads((run / "known_validation_metrics.json").read_text())
        state["datasets"][dataset]["status"] = "SUCCESS"
        state["datasets"][dataset]["val_macro_f1"] = report["metrics"]["macro_f1"]
        state["datasets"][dataset]["adapter_best_epoch"] = report["adapter_best_epoch"]
        state["datasets"][dataset]["head_best_epoch"] = report["head_best_epoch"]
        state["completed_heads"] += 1
        state["updated_at_utc"] = utc()
        save_progress(state)
        print(json.dumps({"event": "SUCCESS", "dataset": dataset,
                          "val_macro_f1": report["metrics"]["macro_f1"]}), flush=True)
    state["status"] = "THREE_HEADS_FROZEN_TEST_NOT_OPENED"
    state["updated_at_utc"] = utc()
    state["next_step"] = "Verify three checkpoints and frozen sources before independent Known Test evaluation"
    save_progress(state)
    print(json.dumps({"status": state["status"], "completed_heads": 3,
                      "test_features_opened": 0}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
