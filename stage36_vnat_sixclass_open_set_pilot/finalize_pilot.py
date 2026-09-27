#!/usr/bin/env python3
"""Summarize and inventory a verified diagnostic without selecting a method."""
from __future__ import annotations

import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
WORKSPACE = PROJECT.parents[1]
PYTHON = "/home/birkenwald/data/SEU-WXY/conda_envs/2025-10-8-WXY-dgl_py310/bin/python"
PRESERVE = WORKSPACE / "skills/experiment-data-preservation/scripts"


def main() -> None:
    verification = json.loads((ROOT / "completion_verification.json").read_text())
    if verification["status"] != "PASS":
        raise RuntimeError("metric replay not PASS")
    with (ROOT / "open_set_results.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 4 or {r["method"] for r in rows} != {"msp", "energy", "centroid", "des_v1"}:
        raise RuntimeError("unexpected score set")
    table = ["| Score | AUROC | AUPRC | UFAR | Known FRR |", "|---|---:|---:|---:|---:|"]
    for row in rows:
        table.append("| {method} | {auroc:.6f} | {auprc:.6f} | {ufar:.6f} | {known_frr:.6f} |".format(
            method=row["method"], **{name: float(row[name]) for name in
            ("auroc", "auprc", "ufar", "known_frr")}))
    report = "\n".join([
        "# Stage 36 — VNAT six-class frozen open-set pilot", "",
        "Status: `complete / DIAGNOSTIC_ONLY`. No encoder, adapter, or classifier was retrained.", "",
        "Stage 14B `medium_seed2025`: 15,704 Known Train, 1,960 Known Validation, "
        "1,960 Known Test, 3,825 Unknown Test (sftp, vimeo, zoiper). "
        "The six-class head is the frozen Stage33 exploratory head.", "",
        *table, "",
        "The four formulas were registered before Unknown packet values were read. "
        "Centroids and kNN-10 use Known Train only; distance normalization and all "
        "P95 thresholds use Known Validation only (`method=higher`, anomaly if score > threshold). "
        "Known Test/Unknown Test are evaluation-only. No winner is selected from these Test metrics.", "",
        "Per-Unknown-application results: `per_unknown_application.csv`; "
        "all flow IDs, raw scores and decisions: `sample_scores.csv`; "
        "thresholds: `calibration.json`; representations: `representations/`; "
        "frozen/parity/replay evidence: `preflight.json`, `parity.json`, "
        "`evaluation_audit.json`, `completion_verification.json`.", "",
        "Limitations: one protocol/seed; Stage33 six-class taxonomy followed previously "
        "exposed four-class Test outcomes; VNAT Known splits are flow-random rather than "
        "capture-disjoint; Vimeo is application-unknown but shares the Streaming service "
        "with Known Netflix/YouTube. This is not untouched external validation or "
        "Service-level Unknown detection. Native Open-Detect KL/prototype is not represented "
        "by these classifier-based scores. Stage20 Service-LOSO still requires Known-only "
        "branch retraining per protocol.", "",
        "Conclusion: report the four diagnostic scores and their absolute UFAR/FRR; "
        "do not infer cross-dataset generalization or tune a new score from this pilot.", "",
    ])
    (ROOT / "RESULTS.md").write_text(report)
    manifest_path = ROOT / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    override_path = ROOT / "launch_override.json"
    override = json.loads(override_path.read_text()) if override_path.is_file() else None
    if override is not None and (override["science_protocol_changed"] or override["physical_gpu_ids"] != [2]):
        raise RuntimeError("invalid Stage36 scheduling-only override")
    manifest.update({"updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete", "claim_scope": "diagnostic", "inputs": [
            {"dataset": "VNAT", "protocol": "Stage14B medium_seed2025",
             "freeze_hash": "c904daef04d2c8e79469191121325c0a6ceeaeb3e595ef9c4c7d7a62c980ae2f",
             "known_train": 15704, "known_validation": 1960, "known_test": 1960, "unknown_test": 3825}],
        "execution": {"tmux_session": "stage36_gpu2_now_queue" if override else "stage36_conditional_queue",
                      "command": "run_gpu2_now.py" if override else "run_conditional.py",
                      "exit_code": 0, "environment": "2025-10-8-WXY-dgl_py310",
                      "physical_gpu_ids": [2] if override else [0, 1], "gpu2_used": bool(override)},
        "configuration": {"files": ["EXPERIMENT_PLAN.md", "run_pilot.py"],
                          "parameters": {"k": 10, "threshold": "KnownVal P95 higher", "scores":
                              ["msp", "energy", "centroid", "des_v1"]}, "seeds": [2022]},
        "core_results": [{"method": row["method"], **{key: float(row[key]) for key in
            ("auroc", "auprc", "ufar", "known_frr", "known_acceptance")}} for row in rows],
        "limitations": ["single VNAT protocol/seed", "post-hoc Stage33 taxonomy",
                        "flow-random capture overlap", "application-level Unknown only",
                        "Open-Detect Native KL score not compared"],
        "next_step": "Review diagnostic results; separately preregister Service-LOSO Known-only retraining if authorized."})
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    subprocess.run([PYTHON, str(PRESERVE / "refresh_artifact_manifest.py"), str(ROOT)], check=True)
    subprocess.run([PYTHON, str(PRESERVE / "validate_experiment_bundle.py"), str(ROOT), "--verify-hashes"],
                   check=True)
    index = PROJECT / "EXPERIMENT_RESULTS.md"
    if "stage36-vnat-sixclass-open-set-20260925-v1` terminal:" not in index.read_text():
        with index.open("a") as handle:
            handle.write("\n- `stage36-vnat-sixclass-open-set-20260925-v1` terminal: "
                         "`complete / DIAGNOSTIC_ONLY`; four frozen scores, VNAT "
                         "medium_seed2025 application-level Unknown, no retraining or Test tuning. "
                         "[bundle](stage36_vnat_sixclass_open_set_pilot/RESULTS.md).\n")
    progress = PROJECT / "EXECUTION_PROGRESS.md"
    marker = "Stage36 terminal update " + datetime.now(timezone.utc).strftime("%Y-%m-%d") + ":"
    if "Stage36 terminal update " not in progress.read_text():
        with progress.open("a") as handle:
            handle.write("\n" + marker + " `complete / DIAGNOSTIC_ONLY`. The conditional queue "
                         "verified Stage34/35 terminal gates, ran frozen VNAT Unknown input "
                         "recovery and four preregistered scores with Known-Val-only "
                         "calibration, and independently replayed all saved decisions and "
                         "metrics. Stage14B/31/33 hashes stayed unchanged; the full "
                         "evaluation used only the user-selected physical GPU2. Actual "
                         "metrics and limitations are "
                         "in stage36_vnat_sixclass_open_set_pilot/RESULTS.md. No new "
                         "method was selected from Test.\n")
    print(json.dumps({"status": "PASS", "methods": 4, "unknown_test": 3825}), flush=True)


if __name__ == "__main__":
    main()
