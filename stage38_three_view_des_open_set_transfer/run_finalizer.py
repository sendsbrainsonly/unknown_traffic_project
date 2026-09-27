#!/usr/bin/env python3
"""Wait for all Stage38 evaluations and assemble a verified development report."""
from __future__ import annotations

import csv
import json
import os
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from run_iscx_known import ROOT, PROJECT, lock

WORKSPACE = PROJECT.parents[1]
PRESERVE = WORKSPACE / "skills/experiment-data-preservation/scripts"
PYTHON = "/home/birkenwald/data/SEU-WXY/conda_envs/2025-10-8-WXY-dgl_py310/bin/python"
START = time.monotonic()
METHODS = {"msp": "C0", "energy": "C1", "centroid": "D0", "des_v1": "D1"}
DATASETS = (("VNAT", None), ("ISCX-VPN", "iscx_vpn"), ("ISCXTor2016", "iscx_tor"))


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"empty final output: {path}")
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def update(status: str, phase: str, **details: object) -> None:
    value = {"status": status, "phase": phase,
             "updated_at_utc": datetime.now(timezone.utc).isoformat(),
             "elapsed_seconds": round(time.monotonic() - START, 1), **details}
    tmp = ROOT / "finalizer_queue_progress.json.tmp"
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp, ROOT / "finalizer_queue_progress.json")
    print(json.dumps(value), flush=True)


def wait_for_datasets() -> None:
    update("WAITING", "VPN_TOR_FROZEN_EVALUATION")
    while True:
        states = {}
        for dataset in ("iscx_vpn", "iscx_tor"):
            path = ROOT / f"{dataset}_detection_queue_progress.json"
            states[dataset] = json.loads(path.read_text())["status"] if path.is_file() else "PENDING"
        if any(value == "FAILED" for value in states.values()):
            raise RuntimeError(f"Stage38B dataset evaluation failed: {states}")
        if all(value == "COMPLETE" for value in states.values()):
            break
        time.sleep(30)
    for dataset in states:
        lock(dataset)
        path = ROOT / "runs" / dataset / "medium_seed2022/detection/completion_verification.json"
        if json.loads(path.read_text())["status"] != "PASS":
            raise RuntimeError(f"independent score verification failed: {dataset}")


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def collect() -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    all_scores, comparisons, per_app, closed = [], [], [], []
    for name, dataset in DATASETS:
        if dataset is None:
            root = ROOT / "vnat"
            source = read_csv(root / "stage38a_method_results.csv")
            result = {r["method"]: r for r in source}
            c = json.loads((root / "closed_set_metrics.json").read_text())
            closed.append({"dataset": name, "protocol": "medium_seed2025",
                           "known_test": c["known_test"], "accuracy": c["accuracy"],
                           "macro_f1": c["macro_f1"], "weighted_f1": c["weighted_f1"]})
            app_rows = read_csv(root / "per_unknown_application.csv")
            protocol = "medium_seed2025"
        else:
            root = ROOT / "runs" / dataset / "medium_seed2022/detection"
            result = {r["method"]: r for r in read_csv(root / "open_set_results.csv")}
            c = json.loads((root / "closed_set_results.json").read_text())
            closed.append({"dataset": name, "protocol": "medium_seed2022",
                           "known_test": c["known_test_samples"],
                           "accuracy": c["metrics"]["accuracy"],
                           "macro_f1": c["metrics"]["macro_f1"],
                           "weighted_f1": c["metrics"]["weighted_f1"]})
            app_rows = read_csv(root / "per_unknown_application.csv")
            protocol = "medium_seed2022"
        if set(result) != set(METHODS):
            raise RuntimeError(f"missing frozen scores: {name}/{set(result)}")
        for method, score_id in METHODS.items():
            row = result[method]
            all_scores.append({"dataset": name, "protocol": protocol, "score_id": score_id,
                               "method": method, **{key: row[key] for key in (
                                   "auroc", "auprc", "ufar", "known_frr", "known_acceptance",
                                   "threshold", "known_test", "unknown_test")}})
        for left, right in (("D1", "D0"), ("D0", "C0"), ("D1", "C0"), ("D1", "C1")):
            source = {r["score_id"]: r for r in all_scores if r["dataset"] == name}
            comparisons.append({"dataset": name, "protocol": protocol, "left": left, "right": right,
                **{f"delta_{key}": float(source[left][key]) - float(source[right][key])
                   for key in ("auroc", "auprc", "ufar", "known_frr")}})
        for item in app_rows:
            per_app.append({"dataset": name, "protocol": protocol,
                            "score_id": METHODS[item["method"]],
                            **{key: item[key] for key in (
                                "method", "unknown_application", "auroc", "auprc", "ufar",
                                "known_frr", "known_acceptance", "known_test", "unknown_test")}})
    return all_scores, comparisons, per_app, closed


def gate(comparisons: list[dict]) -> str:
    d1_c0 = [r for r in comparisons if r["left"] == "D1" and r["right"] == "C0"]
    d1_d0 = [r for r in comparisons if r["left"] == "D1" and r["right"] == "D0"]
    if all(r["delta_auroc"] <= 0 for r in d1_c0 + d1_d0):
        return "NO_USEFUL_GAIN"
    if (all(r["delta_auroc"] >= 0.01 and r["delta_auprc"] >= 0
            and r["delta_ufar"] <= 0 for r in d1_c0)
        and all(r["delta_auroc"] >= 0.01 for r in d1_d0)):
        return "PROMISING_DEVELOPMENT"
    return "MIXED_OR_INCONCLUSIVE"


def report(scores: list[dict], deltas: list[dict], closed: list[dict], conclusion: str) -> str:
    lines = ["# Stage 38 — Three-view DES open-set transfer", "",
             "## Data and split", "",
             "Development datasets: VNAT Stage14B medium_seed2025, ISCX-VPN and ISCXTor2016 Stage12 medium_seed2022. "
             "Known/Unknown membership and sample IDs use those frozen manifests. ISCX Known-only training uses the preregistered "
             "capture-derived coarse category mapping; it is not authoritative per-flow service ground truth.", "",
             "## Configuration and execution", "",
             "VNAT uses the frozen Stage33 six-class three-view encoder. ISCX-VPN/Tor use newly trained Stage31/32-style "
             "TrafficFormer, graph and YaTC branches, 64-dimensional adapters, and a fixed equal-weight feature fusion. "
             "Training and checkpoint selection use Known Train/Validation only. Scores are C0 MSP, C1 Energy, D0 empirical "
             "centroid distance, and D1 fixed global/local DES-v1 (k=10, 0.5/0.5). Each threshold is Known-Val P95 (higher).", "",
             "## Core results", "", "Known Test closed-set metrics:", "",
             "| Dataset | Accuracy | Macro-F1 | Weighted-F1 | Known Test |",
             "|---|---:|---:|---:|---:|"]
    for row in closed:
        lines.append(f"| {row['dataset']} | {float(row['accuracy']):.6f} | {float(row['macro_f1']):.6f} | "
                     f"{float(row['weighted_f1']):.6f} | {row['known_test']} |")
    lines.extend(["", "Open-set scores (higher AUROC/AUPRC, lower UFAR/FRR are better):", "",
                  "| Dataset | Score | AUROC | AUPRC | UFAR | Known FRR |",
                  "|---|---|---:|---:|---:|---:|"])
    for row in scores:
        lines.append(f"| {row['dataset']} | {row['score_id']} | {float(row['auroc']):.6f} | "
                     f"{float(row['auprc']):.6f} | {float(row['ufar']):.6f} | "
                     f"{float(row['known_frr']):.6f} |")
    lines.extend(["", "Paired differences on the same frozen representation:", "",
                  "| Dataset | Comparison | ΔAUROC | ΔAUPRC | ΔUFAR | ΔKnown FRR |",
                  "|---|---|---:|---:|---:|---:|"])
    for row in deltas:
        lines.append(f"| {row['dataset']} | {row['left']}−{row['right']} | {row['delta_auroc']:+.6f} | "
                     f"{row['delta_auprc']:+.6f} | {row['delta_ufar']:+.6f} | "
                     f"{row['delta_known_frr']:+.6f} |")
    lines.extend(["", "## Preserved evidence", "",
                  "`stage38_dataset_results.csv`, `stage38_paired_comparison.csv`, "
                  "`stage38_unknown_application_results.csv`, `stage38_closed_set_results.csv`, "
                  "each dataset's `detection/sample_scores.csv`, calibration, representations and checkpoint hashes, "
                  "named tmux logs, and `completion_verification.json` preserve the run and its replay.", "",
                  "## Limitations", "",
                  "One seed per dataset; the VNAT taxonomy followed earlier exposed Test results. ISCX service labels are "
                  "capture-derived and may not identify each individual flow's activity. These are development diagnostics, "
                  "not untouched external validation. C0/C1 are classifier-uncertainty baselines and are not Open-Detect "
                  "Native. Across-encoder historical comparisons cannot isolate the DES support mechanism.", "",
                  "## Conclusion and next step", "",
                  f"Final Gate: `{conclusion}`. The paired D1−D0 comparison tests the global/local support contribution "
                  "within one representation. The C0/C1 comparison measures operating performance on the same encoder. "
                  "Any further method choice needs a new frozen multi-setting evaluation; this run makes no method or "
                  "threshold update.", ""])
    return "\n".join(lines)


def main() -> None:
    try:
        if json.loads((ROOT / "vnat/stage38a_verification.json").read_text())["status"] != "PASS":
            raise RuntimeError("Stage38A frozen VNAT verification not PASS")
        wait_for_datasets()
        update("RUNNING", "AGGREGATE_AND_REPLAY")
        scores, deltas, applications, closed = collect()
        choice = gate(deltas)
        write_csv(ROOT / "stage38_dataset_results.csv", scores)
        write_csv(ROOT / "stage38_paired_comparison.csv", deltas)
        write_csv(ROOT / "stage38_unknown_application_results.csv", applications)
        write_csv(ROOT / "stage38_closed_set_results.csv", closed)
        (ROOT / "RESULTS.md").write_text(report(scores, deltas, closed, choice))
        write_json(ROOT / "completion_verification.json", {"status": "PASS", "gate": choice,
            "datasets": [name for name, _ in DATASETS], "dataset_score_rows": len(scores),
            "paired_rows": len(deltas), "unknown_application_rows": len(applications),
            "vnat_replay": "PASS", "iscx_replay": "PASS", "unknown_for_fitting": 0,
            "known_test_for_fitting": 0, "checkpoint_updates_after_selection": 0})
        manifest = json.loads((ROOT / "manifest.json").read_text())
        manifest["status"] = "success"
        manifest["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["core_results"] = [{"dataset": r["dataset"], "score": r["score_id"],
            "auroc": float(r["auroc"]), "auprc": float(r["auprc"]),
            "ufar": float(r["ufar"]), "known_frr": float(r["known_frr"])} for r in scores]
        manifest["execution"] = {"tmux_session": "stage38_finalizer_queue_v2_0926",
            "command": "run_finalizer.py", "exit_code": 0,
            "environment": "2025-10-8-WXY-dgl_py310", "physical_gpu_ids": [2, 3, 4]}
        manifest["limitations"] = ["single seed", "development datasets",
            "capture-derived ISCX coarse labels", "VNAT taxonomy after earlier Test exposure"]
        manifest["next_step"] = "Stop Stage38; retain frozen evidence. Any next evaluation needs a new preregistered protocol."
        write_json(ROOT / "manifest.json", manifest)
        update("COMPLETE", "STAGE38", gate=choice, dataset_score_rows=len(scores))
        subprocess.run([PYTHON, str(PRESERVE / "refresh_artifact_manifest.py"), str(ROOT)],
                       cwd=PROJECT, check=True)
        subprocess.run([PYTHON, str(PRESERVE / "validate_experiment_bundle.py"),
                        str(ROOT), "--verify-hashes"], cwd=PROJECT, check=True)
    except BaseException as exc:
        failure = ROOT / ("finalizer_queue_failure_v2.json" if (ROOT / "finalizer_queue_failure.json").exists()
                          else "finalizer_queue_failure.json")
        failure.write_text(json.dumps({
            "status": "FAIL", "error": repr(exc), "traceback": traceback.format_exc(),
            "updated_at_utc": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n")
        update("FAILED", "STOPPED", error=repr(exc))
        raise


if __name__ == "__main__":
    main()
