#!/usr/bin/env python3
"""Gated one-shot Stage32 Known Test recovery, evaluation and replay."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import traceback
from datetime import datetime, timezone

from common import DATASETS, ROOT
from test_inputs import ensure_three_heads_frozen, prepare_vnat_test

SELECTOR = ROOT.parent.parent.parent / "skills" / "using-superpowers" / "scripts" / "select_gpu.py"


def utc():
    return datetime.now(timezone.utc).isoformat()


def save(value):
    value["updated_at_utc"] = utc()
    temp = ROOT / "progress.json.tmp"
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    os.replace(temp, ROOT / "progress.json")


def main():
    ensure_three_heads_frozen()
    state = json.loads((ROOT / "progress.json").read_text())
    if state["status"] != "THREE_HEADS_FROZEN_TEST_NOT_OPENED" or state["test_features_opened"]:
        raise RuntimeError("Stage32 finalizer can run only once before Test opens")
    state["status"] = "TEST_INPUT_RECOVERY"
    state["next_step"] = "Recover only frozen VNAT Known Test packet views"
    state["test_evaluation"] = {dataset: {"status": "PENDING"} for dataset in DATASETS}
    save(state)
    try:
        recovered = prepare_vnat_test()
        state["vn_test_cache"] = recovered
        state["test_features_opened"] = recovered["vn_test_flows"]
        state["status"] = "TEST_INPUTS_READY"
        state["next_step"] = "One-shot Known Test inference; no parameter changes"
        save(state)
        for dataset in DATASETS:
            state["status"] = "TEST_EVALUATING"
            state["test_evaluation"][dataset]["status"] = "RUNNING"
            state["test_evaluation"][dataset]["started_at_utc"] = utc()
            save(state)
            print(json.dumps({"event": "TEST_START", "dataset": dataset}), flush=True)
            command = [sys.executable, str(SELECTOR), "--min-free-gb", "8", "--",
                       sys.executable, str(ROOT / "evaluate_one.py"), "--dataset", dataset]
            result = subprocess.run(command, cwd=ROOT.parent, check=False)
            state["test_evaluation"][dataset]["exit_code"] = result.returncode
            if result.returncode != 0 or not (ROOT / "runs" / dataset / "known_test_evaluation" / "SUCCESS").is_file():
                state["test_evaluation"][dataset]["status"] = "FAILED"
                raise RuntimeError(f"{dataset} one-shot Test failed: exit {result.returncode}")
            report = json.loads((ROOT / "runs" / dataset / "known_test_evaluation" / "results.json").read_text())
            state["test_evaluation"][dataset].update({"status": "SUCCESS", "finished_at_utc": utc(),
                                                       "macro_f1": report["metrics"]["macro_f1"],
                                                       "samples": report["known_test_samples"]})
            state["test_features_opened"] += 0 if dataset == "vnat" else report["known_test_samples"]
            save(state)
            print(json.dumps({"event": "TEST_SUCCESS", "dataset": dataset,
                              "macro_f1": report["metrics"]["macro_f1"]}), flush=True)
        print(json.dumps({"event": "INDEPENDENT_REPLAY_START"}), flush=True)
        result = subprocess.run([sys.executable, str(ROOT / "summarize.py")], cwd=ROOT.parent, check=False)
        if result.returncode != 0 or not (ROOT / "completion_verification.json").is_file():
            raise RuntimeError(f"Stage32 independent replay failed: exit {result.returncode}")
        state["status"] = "TEST_COMPLETE"
        state["next_step"] = "Inspect report, validate experiment bundle and update durable project handoff"
        state["independent_replay"] = "PASS"
        save(state)
        print(json.dumps({"status": "TEST_COMPLETE", "test_evaluation": state["test_evaluation"]}), flush=True)
    except BaseException as exc:
        state["status"] = "TEST_FAILED"
        state["next_step"] = "Inspect preserved Test cache/result failure artifacts; do not restart or tune on Test"
        state["error"] = repr(exc)
        save(state)
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
