#!/usr/bin/env python3
"""Finish Stage31 only after every Known-only branch and T0 head is frozen.

Run this inside a named project-local tmux task.  Test caches and evaluation
are deliberately unreachable until the all-five-head hash gate passes.
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from preflight import AUDIT, CELLS, OUT, PROJECT, sha
from test_unlock import ensure_all_heads_frozen

SELECT = PROJECT.parents[1] / "skills" / "using-superpowers" / "scripts" / "select_gpu.py"
TMUX = PROJECT.parents[1] / "skills" / "tmux-task-execution" / "scripts" / "tmux_task.sh"
QUEUE = OUT / "queue_progress.json"
PROGRESS = OUT / "finalization_progress.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save(status: str, **details: object) -> None:
    tmp = PROGRESS.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"status": status, "updated_at_utc": now(), **details}, indent=2) + "\n")
    tmp.replace(PROGRESS)


def frozen_hashes() -> dict[str, str]:
    before = json.loads((OUT / "source_hashes_before.json").read_text())
    after = {}
    for relative in before:
        path = AUDIT if relative == "stage15r_protocol_audit" else PROJECT / relative
        after[relative] = sha(path)
    return after


def require_unchanged_sources() -> dict[str, str]:
    before = json.loads((OUT / "source_hashes_before.json").read_text())
    after = frozen_hashes()
    if after != before:
        changed = [key for key in before if before[key] != after.get(key)]
        raise RuntimeError(f"frozen source hashes changed: {changed}")
    return after


def await_heads() -> None:
    last_status = None
    while True:
        if not QUEUE.is_file():
            raise RuntimeError("Stage31 bounded queue progress is absent")
        queue = json.loads(QUEUE.read_text())
        status = queue.get("status")
        if status != last_status:
            print(json.dumps({"event": "queue_state", "status": status, "at": now()}), flush=True)
            last_status = status
        if status == "FAILED":
            raise RuntimeError(f"Stage31 bounded queue failed: {queue.get('error')}")
        if status == "ALL_BRANCHES_AND_T0_COMPLETE_TEST_NOT_RUN":
            ensure_all_heads_frozen()
            return
        if status != "RUNNING":
            raise RuntimeError(f"unexpected Stage31 queue state: {status}")
        process = subprocess.run([str(TMUX), "status", "codex_stage31_bounded_queue_20260925"],
                                 cwd=PROJECT, capture_output=True, text=True)
        if process.returncode or "state=running" not in process.stdout:
            raise RuntimeError(f"Stage31 queue is not live while progress says RUNNING: {process.stdout}")
        time.sleep(30)


def run_command(command: list[str]) -> None:
    print(json.dumps({"event": "start", "command": command, "at": now()}), flush=True)
    subprocess.run(command, cwd=PROJECT, check=True)
    print(json.dumps({"event": "finished", "command": command, "at": now()}), flush=True)


def commands_for(dataset: str, protocol: str) -> list[list[str]]:
    script = OUT
    if dataset in ("iscx_vpn", "iscx_tor"):
        return [[sys.executable, str(script / name), "--dataset", dataset, "--phase", "test"]
                for name in ("build_iscx_mfr.py", "build_iscx_tf_fig.py")]
    if dataset == "vnat":
        return [[sys.executable, str(script / name), "--protocol", protocol, "--phase", "test"]
                for name in ("build_vnat_mfr.py", "build_vnat_tf_fig.py")]
    return [[sys.executable, str(script / name), "--phase", "test"]
            for name in ("build_ustc_mfr.py", "build_ustc_tf_fig.py")]


def test_cache_dirs(dataset: str, protocol: str) -> list[Path]:
    root = OUT / "input_caches" / dataset
    if dataset in ("vnat", "ustc"):
        root /= protocol
    return [root / "yatc_mfr_test", root / "tf_fig_test"]


def require_cache(path: Path) -> bool:
    if not path.exists():
        return False
    audit_path = path / "cache_audit.json"
    if not audit_path.is_file() or json.loads(audit_path.read_text()).get("status") != "PASS":
        raise RuntimeError(f"partial or failed Test cache; refusing overwrite: {path}")
    return True


def replay_results() -> list[dict[str, object]]:
    rows = []
    for dataset, protocol in CELLS:
        run = OUT / "runs" / dataset / protocol
        result_dir = run / "known_test_evaluation"
        report = json.loads((result_dir / "results.json").read_text())
        if report["status"] != "PASS" or not (result_dir / "SUCCESS").is_file():
            raise RuntimeError(f"incomplete Known Test result: {dataset}/{protocol}")
        with (result_dir / "sample_predictions.csv").open(newline="") as handle:
            samples = list(csv.DictReader(handle))
        logits = np.load(result_dir / "logits.npy", allow_pickle=False)
        classes = report["known_classes"]
        if logits.shape != (len(samples), len(classes)) or len(samples) != report["known_test_samples"]:
            raise RuntimeError(f"shape or sample-count mismatch: {dataset}/{protocol}")
        truth = [row["true_class"] for row in samples]
        predicted = [classes[int(idx)] for idx in logits.argmax(axis=1)]
        if predicted != [row["predicted_class"] for row in samples]:
            raise RuntimeError(f"prediction/logit mismatch: {dataset}/{protocol}")
        computed = {"accuracy": accuracy_score(truth, predicted),
                    "macro_f1": f1_score(truth, predicted, labels=classes, average="macro", zero_division=0),
                    "weighted_f1": f1_score(truth, predicted, labels=classes, average="weighted", zero_division=0)}
        for metric, value in computed.items():
            if abs(value - report[metric]) > 1e-10:
                raise RuntimeError(f"independent metric replay mismatch: {dataset}/{protocol}/{metric}")
        head = run / "T0_equal"
        frozen = json.loads((head / "known_validation_metrics.json").read_text())
        for name, expected in frozen["checkpoint_hashes"].items():
            if sha(head / name) != expected:
                raise RuntimeError(f"head checkpoint changed during evaluation: {dataset}/{protocol}/{name}")
        for branch, expected in report["branch_checkpoint_hashes"].items():
            if sha(run / branch / "model_best.pt") != expected:
                raise RuntimeError(f"branch checkpoint changed during evaluation: {dataset}/{protocol}/{branch}")
        rows.append({"dataset": dataset, "protocol": protocol, "known_test_samples": len(samples),
                     "known_classes": len(classes),
                     "known_validation_macro_f1": frozen["metrics"]["macro_f1"], **computed,
                     "sample_predictions_sha256": sha(result_dir / "sample_predictions.csv"),
                     "logits_sha256": sha(result_dir / "logits.npy")})
    return rows


def write_completion(rows: list[dict[str, object]], after: dict[str, str]) -> None:
    summary = OUT / "stage31_known_test_summary.csv"
    if summary.exists():
        raise RuntimeError(f"refusing to overwrite summary: {summary}")
    with summary.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "source_hashes_after.json").write_text(json.dumps(after, indent=2, sort_keys=True) + "\n")
    verification = {"status": "PENDING_BUNDLE_VALIDATION", "completed_at_utc": now(),
                    "cells": len(rows), "datasets": len({row["dataset"] for row in rows}),
                    "known_test_metric_replay": "PASS",
                    "frozen_source_hashes_unchanged": True,
                    "all_five_heads_frozen_before_test": True,
                    "unknown_test_used": False, "test_threshold_or_model_selection": False,
                    "test_score_scope": "development; not untouched external validation"}
    (OUT / "completion_verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    table = "\n".join(f"| {row['dataset']} | {row['protocol']} | {row['known_test_samples']} | "
                      f"{row['known_validation_macro_f1']:.6f} | {row['accuracy']:.6f} | "
                      f"{row['macro_f1']:.6f} | {row['weighted_f1']:.6f} |"
                      for row in rows)
    historical_notes = (OUT / "RESULTS.md").read_text()
    report = f"""# Stage 31 results — four-dataset representative-protocol pilot

Status: **PASS**, completed {now()}. This is a five-protocol closed-set pilot over four datasets, not a full protocol sweep or untouched external validation.

## Data and split

The exact five Stage15R frozen Known Train/Validation/Test protocols and class memberships were reused. Unknown flows were excluded from all fitting and evaluation in this closed-set task.

## Configuration and execution

TrafficFormer, FIG/TAGCN, and YaTC were independently fitted on Known Train. The Stage30 `T0_equal` adapter/head used fixed 1/3 feature weights and Known Validation selection. All five head hashes were verified before Known Test input extraction.

## Core results

| Dataset | Frozen protocol | Known Test flows | Known Val Macro-F1 | Test Accuracy | Test Macro-F1 | Test Weighted-F1 |
|---|---|---:|---:|---:|---:|---:|
{table}

## Preserved evidence

Unknown Test was not accessed. Test did not select features, weights, checkpoints, or thresholds. Frozen source hashes are unchanged. Prediction/logit consistency and all three metrics were independently replayed from saved sample-level outputs. Full per-class results, logits, checkpoint hashes, and training histories remain under `runs/`. Earlier failed recovery attempts and their logs remain preserved.

## Limitations

Input caveat: VNAT Medium-2025 includes the preregistered Stage31-only IPv6 YaTC-MFR adaptation for two Train/Val flows; this is not strict official YaTC input parity. Stage31's Test input builder uses the same fixed rule and records any Test adaptations in its cache audit. Earlier failed recovery attempts and their logs remain preserved.

## Conclusion and next step

No open-set conclusion follows from this closed-set test. Differences in label spaces and Known class composition make a pooled four-dataset accuracy inappropriate.

## Earlier execution notes

{historical_notes}
"""
    (OUT / "RESULTS.md").write_text(report)
    manifest = json.loads((OUT / "manifest.json").read_text())
    manifest["status"] = "success"
    manifest["updated_at_utc"] = now()
    manifest["core_results"] = [{"dataset": row["dataset"], "protocol": row["protocol"],
                                  "accuracy": row["accuracy"], "macro_f1": row["macro_f1"],
                                  "weighted_f1": row["weighted_f1"]} for row in rows]
    manifest["limitations"] = ["Five representative protocols, not all frozen protocols",
                               "Previously exposed Test; development evidence only",
                               "Two VNAT Medium-2025 Train/Val flows use documented IPv6 YaTC-MFR adaptation"]
    manifest["next_step"] = "Report four-dataset pilot; no automatic open-set experiment"
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def validate_bundle_and_index(rows: list[dict[str, object]]) -> None:
    scripts = PROJECT.parents[1] / "skills" / "experiment-data-preservation" / "scripts"
    run_command([sys.executable, str(scripts / "refresh_artifact_manifest.py"),
                 str(OUT), "--hash-max-bytes", str(64 * 1024 * 1024)])
    run_command([sys.executable, str(scripts / "validate_experiment_bundle.py"),
                 str(OUT), "--verify-hashes"])
    verification_path = OUT / "completion_verification.json"
    verification = json.loads(verification_path.read_text())
    verification["status"] = "PASS"
    verification["bundle_validation"] = "PASS"
    verification_path.write_text(json.dumps(verification, indent=2) + "\n")
    # The verification file changed; refresh its digest and revalidate.
    run_command([sys.executable, str(scripts / "refresh_artifact_manifest.py"),
                 str(OUT), "--hash-max-bytes", str(64 * 1024 * 1024)])
    run_command([sys.executable, str(scripts / "validate_experiment_bundle.py"),
                 str(OUT), "--verify-hashes"])
    index_path = PROJECT / "EXPERIMENT_RESULTS.md"
    index = index_path.read_text()
    marker = "- `stage31-four-dataset-three-view-equal-20260925-v1`:"
    entry = (f"{marker} `success / FIVE_FROZEN_PROTOCOL_TESTS`; four datasets, five matched Known Test "
             f"protocols; independent metric replay, frozen source hashes and bundle "
             f"validation PASS; Unknown Test unused; [bundle](stage31_four_dataset_three_view_equal/RESULTS.md).")
    lines = [line for line in index.splitlines() if not line.startswith(marker)]
    index_path.write_text("\n".join([*lines, entry]) + "\n")


def main() -> None:
    if (OUT / "completion_verification.json").exists():
        raise RuntimeError("Stage31 already finalized")
    try:
        save("WAITING_FOR_ALL_FROZEN_HEADS", test_features_opened=0)
        await_heads()
        require_unchanged_sources()
        save("BUILDING_KNOWN_TEST_INPUTS")
        for dataset, protocol in CELLS:
            for command, cache in zip(commands_for(dataset, protocol), test_cache_dirs(dataset, protocol), strict=True):
                if not require_cache(cache):
                    run_command(command)
                require_cache(cache)
        require_unchanged_sources()
        save("EVALUATING_KNOWN_TEST")
        for dataset, protocol in CELLS:
            run = OUT / "runs" / dataset / protocol
            if (run / "known_test_results.json").is_file():
                continue
            if (run / "known_test_evaluation").exists():
                raise RuntimeError(f"partial Known Test evaluation; refusing overwrite: {run}")
            command = [sys.executable, str(SELECT), "--allowed", "0,1,2", "--min-free-gb", "16",
                       "--", sys.executable, str(OUT / "evaluate_known_test.py"),
                       "--dataset", dataset, "--protocol", protocol]
            run_command(command)
        ensure_all_heads_frozen()
        after = require_unchanged_sources()
        save("REPLAYING_SAVED_TEST_OUTPUTS")
        rows = replay_results()
        write_completion(rows, after)
        run_command([sys.executable, str(OUT / "progress.py")])
        save("COMPLETE", completed_cells=len(rows))
        validate_bundle_and_index(rows)
        print(json.dumps({"status": "PASS", "completed_cells": len(rows)}), flush=True)
    except BaseException as exc:
        save("FAILED", error=repr(exc), traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
