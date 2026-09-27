#!/usr/bin/env python3
"""Finalize Stage41 results, experiment index, and independently checked bundle."""
from __future__ import annotations

import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from freeze_matched_protocol import PROJECT, ROOT, digest

WORKSPACE = PROJECT.parents[1]
SKILL_SCRIPTS = WORKSPACE / "skills/experiment-data-preservation/scripts"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    verified = json.loads((ROOT / "completion_verification.json").read_text())
    if verified["status"] != "PASS" or not verified["same_flow_test_ids"]:
        raise RuntimeError("comparison completion verification failed")
    records = read_csv(ROOT / "comparison_run_results.csv")
    if len(records) != 12:
        raise RuntimeError("expected 3 scenarios × 2 populations × 2 methods")
    for setting in ("A-1", "A-2", "A-3"):
        if json.loads((ROOT / setting / "od_score_audit.json").read_text())["status"] != "PASS":
            raise RuntimeError(f"{setting}: OD extraction audit failed")
        if setting != "A-2" and json.loads((ROOT / setting / "detection/evaluation_audit.json").read_text())["status"] != "PASS":
            raise RuntimeError(f"{setting}: three-view evaluation audit failed")
    report = (ROOT / "comparison_report.md").read_text()
    now = datetime.now(timezone.utc).isoformat()
    result_text = (
        "# Experiment results: stage41-a123-matched-open-set-20260927\n\n"
        "- Status: `success`\n- Experiment type: `open-set-comparison`\n"
        "- Claim scope: `diagnostic`\n- Completed (UTC): `" + now + "`\n\n"
        "## Data and split\n\n"
        "The same USTC flow IDs and original Train/Validation/Test memberships were used by both methods. "
        "A-1/A-2/A-3 hold out 1/3/5 complete Unknown classes, respectively. "
        "The deterministic 10%-per-class/source-split pool exactly reproduces the old Stage34/40 A-2 IDs. "
        "The balanced 1:1 Test subset was frozen before any score was read.\n\n"
        "## Configuration and execution\n\n"
        "Seed 2022. Open-Detect released-code ResNet18/prototype-KL is trained from scratch for each "
        "scenario with 100 epochs, Adam, batch 128, Known-Validation checkpoint selection. "
        "The three-view method uses official TrafficFormer/YaTC initialization, FIG graph branch, "
        "separate Known-only branch training, fixed equal feature fusion, and DES-v1 k=10 0.5/0.5 "
        "support score. A-2 reuses byte-identical frozen Stage34/40 weights/scores; A-1/A-3 are newly trained. "
        "Primary threshold: each method's Known-Validation P95; the Test-Youden row is labeled "
        "retrospective oracle. The interrupted A-1 attempt used physical GPU0; after the "
        "two-GPU cap, completed jobs run through named tmux sessions on physical GPUs 6/7, "
        "with live capacity selection.\n\n"
        "## Core results\n\n" + report + "\n"
        "## Preserved evidence\n\n"
        "`matched_protocol.json`, `common_flow_pool.csv`, per-scenario role manifests, "
        "`balanced_test_ids.csv`, three Open-Detect image pools/checkpoints/logs/sample scores, "
        "A-1/A-3 three-view caches/checkpoints/validation histories/Test scores, "
        "`comparison_run_results.csv`, `comparison_paired.csv`, `per_unknown_class.csv`, "
        "`comparison_report.md`, `completion_verification.json`, `manifest.json`, "
        "and project-local named `.tmux-task/stage41_*` execution logs. The initial A-1 "
        "GPU0 attempt was interrupted at 52/100 epochs after the user reduced Stage41 to two GPUs; "
        "its partial checkpoint and history are retained, while reported A-1 metrics use only the "
        "new complete `od_run_2gpu` checkpoint.\n\n"
        "## Limitations\n\n"
        "This is one local 10% seed, not the paper's exact five folds. A-1 has only 85 Unknown "
        "Test flows. The three-view method uses external pretrained weights while Open-Detect "
        "is randomly initialized; this compares complete methods on equal flows, not isolated "
        "architecture components. Balanced Test composition differs from natural prevalence. "
        "Test-oracle thresholds use Test labels and are not deployable validation results.\n\n"
        "## Conclusion and next step\n\n"
        "The same-flow paired result is in the table above. Stop after Stage41; do not modify "
        "historical frozen protocols or automatically start tuning.\n"
    )
    (ROOT / "RESULTS.md").write_text(result_text)
    manifest_path = ROOT / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["status"] = "success"
    manifest["inputs"] = [
        {"path": str(ROOT / "matched_protocol.json"), "sha256": digest(ROOT / "matched_protocol.json")},
        {"path": str(PROJECT / "opendetect_ustc_encoder_audit/outputs/input_alignment_manifest.csv"),
         "sha256": json.loads((ROOT / "matched_protocol.json").read_text())["source_sha256"]},
        {"path": str(ROOT / "balanced_test_ids.csv"), "sha256": digest(ROOT / "balanced_test_ids.csv")},
    ]
    manifest["execution"].update({"tmux_session": "stage41_a123_queue_2gpu_0927",
                                  "command": "python stage41_a123_matched_open_set_comparison/run_queue_2gpu.py",
                                  "environment": "/home/birkenwald/data/SEU-WXY/conda_envs/2025-10-8-WXY-dgl_py310",
                                  "physical_gpu_ids": [6, 7]})
    manifest["configuration"] = {"files": ["matched_protocol.json", "balanced_test_ids.csv"],
        "parameters": {"seed": 2022, "fraction": .1, "scenarios": ["A-1", "A-2", "A-3"],
                       "primary_methods": ["Open-Detect Native", "Three-view DES-v1"],
                       "threshold": "Known Validation P95", "retrospective_oracle": "Test Youden-J"},
        "seeds": [2022]}
    manifest["core_results"] = [
        {"scenario": r["scenario"], "method": r["method"],
         "population": "balanced_1to1", "auroc": float(r["auroc"]),
         "auprc": float(r["auprc"]), "unknown_f1_at_p95": float(r["p95_unknown_f1"]),
         "ufar_at_p95": float(r["p95_ufar"])}
        for r in records if r["population"] == "balanced_1to1"]
    manifest["limitations"] = ["single local 10% seed, not paper exact five folds",
        "A-1 only 85 Unknown Test flows", "external pretrained branches versus OD from scratch",
        "Test-Youden threshold is retrospective oracle",
        "initial A-1 GPU0 attempt interrupted at epoch 52; only the complete two-GPU retry enters metrics"]
    manifest["next_step"] = "Stop; no automatic hyperparameter search or new experiment."
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    progress = {"status": "SUCCESS", "phase": "COMPLETE", "updated_at_utc": now,
                "comparison_report": str(ROOT / "comparison_report.md"),
                "result_rows": len(records)}
    (ROOT / "queue_progress.json").write_text(json.dumps(progress, indent=2) + "\n")
    subprocess.run(["python", str(SKILL_SCRIPTS / "refresh_artifact_manifest.py"), str(ROOT)], check=True)
    subprocess.run(["python", str(SKILL_SCRIPTS / "validate_experiment_bundle.py"),
                    str(ROOT), "--verify-hashes"], check=True)
    index = PROJECT / "EXPERIMENT_RESULTS.md"
    with index.open("a", encoding="utf-8") as handle:
        handle.write("- Stage41 matched A-1/A-2/A-3 same-flow comparison: success; "
                     "[results](stage41_a123_matched_open_set_comparison/RESULTS.md).\n")
    print(json.dumps({"status": "PASS", "scenarios": 3, "result_rows": len(records),
                      "bundle_hash_validation": "PASS"}), flush=True)


if __name__ == "__main__":
    main()
