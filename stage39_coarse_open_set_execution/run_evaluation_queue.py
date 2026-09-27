"""Automatically calibrate, evaluate and verify all frozen Stage39 settings.

Waits for the separate Known-only training queue to finish. Every payload runs
in its own named tmux session, and every GPU inference selects live capacity.
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import time
import traceback
from datetime import datetime, timezone

from freeze_protocols import PROJECT, ROOT, sha

WORKSPACE = PROJECT.parents[1]
HELPER = WORKSPACE / "skills/tmux-task-execution/scripts/tmux_task.sh"
SELECTOR = WORKSPACE / "skills/using-superpowers/scripts/select_gpu.py"
PYTHON = "/home/birkenwald/data/SEU-WXY/conda_envs/2025-10-8-WXY-dgl_py310/bin/python"
START = time.monotonic()


def save(status, dataset="", fold="", phase="", **extra):
    record = {"status": status, "dataset": dataset, "fold": fold, "phase": phase,
              "updated_at_utc": datetime.now(timezone.utc).isoformat(),
              "elapsed_seconds": round(time.monotonic() - START, 1),
              "max_parallel_owned_gpus": 3, **extra}
    temp = ROOT / "evaluation_queue_progress.json.tmp"
    temp.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temp, ROOT / "evaluation_queue_progress.json")
    print(json.dumps(record, ensure_ascii=False), flush=True)


def session_name(dataset, slug, phase):
    return f"stage39e_{dataset.lower()}_{slug}_{phase}_0926"


def execute(dataset, slug, phase, args, gpu=False):
    name = session_name(dataset, slug, phase)
    if (PROJECT / ".tmux-task" / name).exists():
        raise FileExistsError(f"refusing to overwrite existing session: {name}")
    command = [PYTHON, str(ROOT / args[0]), *args[1:]]
    if gpu:
        command = [PYTHON, str(SELECTOR), "--allowed", "0,1,7", "--min-free-gb", "25",
                   "--max-utilization", "30", "--", *command]
    subprocess.run([str(HELPER), "run", name, str(PROJECT), "--", *command],
                   cwd=PROJECT, check=True)


def ready(dataset, slug):
    fold = ROOT / "settings" / dataset / slug
    protocol = json.loads((fold / "protocol.json").read_text(encoding="utf-8"))
    if protocol["status"] != "FROZEN_PRETRAIN" or protocol["checkpoint_reuse"] or \
       sha(fold / "role_manifest.csv") != protocol["role_manifest_sha256"]:
        raise RuntimeError(f"frozen protocol changed: {dataset}/{slug}")
    run_name = "vnat" if dataset == "VNAT" else dataset
    run = fold / "runs" / run_name / protocol["protocol_id"]
    for component in ("trafficformer", "graph", "yatc", "T0_equal"):
        if not (run / component / "SUCCESS").is_file():
            raise RuntimeError(f"Known-only training incomplete: {dataset}/{slug}/{component}")
    return fold


def one(dataset, slug):
    fold = ready(dataset, slug)
    detection = fold / "detection"
    if (detection / "deployment_verification.json").is_file():
        evidence = json.loads((detection / "deployment_verification.json").read_text())
        if evidence["status"] != "PASS":
            raise RuntimeError(f"existing deployment evidence invalid: {dataset}/{slug}")
        return "already_complete"
    phases = (
        ("calibrate", "calibration_verification.json",
         ["detect_iscx_fold.py", "calibrate", "--dataset", dataset, "--fold", slug], False),
        ("cache", "eval_cache_verification.json",
         ["build_vnat_eval_cache.py", "--fold", slug] if dataset == "VNAT" else
         ["build_iscx_eval_cache.py", "--dataset", dataset, "--fold", slug], False),
        ("evaluate", "evaluation_audit.json",
         ["detect_iscx_fold.py", "evaluate", "--dataset", dataset, "--fold", slug], True),
        ("verify", "completion_verification.json",
         ["detect_iscx_fold.py", "verify", "--dataset", dataset, "--fold", slug], False),
        ("deployment", "deployment_verification.json",
         ["analyze_fold_results.py", "--dataset", dataset, "--fold", slug], False),
    )
    for phase, marker, args, gpu in phases:
        path = detection / marker
        if path.is_file():
            if json.loads(path.read_text(encoding="utf-8"))["status"] != "PASS":
                raise RuntimeError(f"existing {phase} marker is not PASS: {path}")
            continue
        save("RUNNING", dataset, slug, phase)
        execute(dataset, slug, phase, args, gpu)
        if not path.is_file() or json.loads(path.read_text(encoding="utf-8"))["status"] != "PASS":
            raise RuntimeError(f"{phase} finished without verified evidence: {dataset}/{slug}")
    return "complete"


def main():
    try:
        with (ROOT / "protocol_index.csv").open(newline="", encoding="utf-8") as stream:
            planned = [r for r in csv.DictReader(stream) if r["new_training_required"] == "True"]
        if len(planned) != 12:
            raise RuntimeError("expected exactly 12 frozen new-training settings")
        while True:
            status_path = ROOT / "training_queue_progress.json"
            state = json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else {}
            if state.get("status") == "FAILED":
                raise RuntimeError("Known-only training queue failed; do not evaluate incomplete models")
            if state.get("status") == "KNOWN_TRAINING_COMPLETE":
                break
            save("WAITING_FOR_KNOWN_TRAINING", phase="waiting",
                 training_status=state.get("status"), training_setting=
                 f"{state.get('dataset', '')}/{state.get('fold', '')}")
            time.sleep(30)
        completed = []
        for row in planned:
            dataset, slug = row["dataset"], row["slug"]
            result = one(dataset, slug)
            completed.append(f"{dataset}/{slug}")
            save("RUNNING", dataset, slug, result, completed=completed)
        save("AGGREGATING", phase="all_frozen_settings", completed=completed)
        execute("all", "all", "aggregate", ["aggregate_results.py"])
        aggregate_path = ROOT / "aggregate_verification.json"
        if not aggregate_path.is_file() or json.loads(aggregate_path.read_text())["status"] != "PASS":
            raise RuntimeError("Stage39 aggregate did not verify")
        save("RESULTS_AGGREGATED", phase="complete", completed=completed,
             report="stage39_report.md", next_phase="refresh_and_validate_experiment_bundle")
    except BaseException as exc:
        failure = {"status": "FAILED", "error": repr(exc), "traceback": traceback.format_exc(),
                   "updated_at_utc": datetime.now(timezone.utc).isoformat()}
        failure_path = ROOT / "evaluation_queue_failure.json"
        if failure_path.exists():
            failure_path = ROOT / f"evaluation_queue_failure_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
        failure_path.write_text(
            json.dumps(failure, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        save("FAILED", phase="stopped", error=repr(exc))
        raise


if __name__ == "__main__":
    main()
