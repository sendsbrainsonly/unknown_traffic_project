#!/usr/bin/env python3
"""Background terminal evidence handoff after the one-shot Stage32 finalizer exits."""
from __future__ import annotations

import csv
import json
import subprocess
import sys
import time
from datetime import datetime, timezone

from common import DATASETS, PROJECT, ROOT

WORKSPACE = PROJECT.parent.parent
REFRESH = WORKSPACE / "skills" / "experiment-data-preservation" / "scripts" / "refresh_artifact_manifest.py"
VALIDATE = WORKSPACE / "skills" / "experiment-data-preservation" / "scripts" / "validate_experiment_bundle.py"
FINALIZER_STATUS = PROJECT / ".tmux-task" / "codex_stage32_three_coarse_test_20260925" / "exit.status"


def utc():
    return datetime.now(timezone.utc).isoformat()


def append_once(path, marker: str, message: str):
    text = path.read_text()
    if marker not in text:
        with path.open("a", encoding="utf-8") as f:
            f.write("\n" + message.rstrip() + "\n")


def rows(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    start = time.monotonic()
    while True:
        state = json.loads((ROOT / "progress.json").read_text())
        if state["status"] in ("TEST_COMPLETE", "TEST_FAILED") and FINALIZER_STATUS.is_file():
            break
        if time.monotonic() - start > 8 * 3600:
            raise TimeoutError("Stage32 finalizer did not reach a terminal state within 8 hours")
        time.sleep(30)
    success = state["status"] == "TEST_COMPLETE" and json.loads(
        (ROOT / "completion_verification.json").read_text())["status"] == "PASS"
    manifest_path = ROOT / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["updated_at_utc"] = utc()
    manifest["status"] = "success" if success else "partial"
    manifest["execution"]["exit_code"] = 0 if success else 1
    manifest["execution"]["tmux_session"] = "codex_stage32_three_coarse_test_20260925"
    manifest["execution"]["training_tmux_session"] = "codex_stage32_three_coarse_train_20260925"
    manifest["configuration"]["parameters"]["known_test_opened_only_after_three_heads_frozen"] = True
    if success:
        report_rows = rows(ROOT / "stage32_test_results.csv")
        comparisons = rows(ROOT / "stage32_paired_comparison.csv")
        manifest["core_results"] = report_rows
        manifest["limitations"] = [
            "single training seed; no seed-stability claim",
            "Known Test previously exposed; development evidence only",
            "VNAT flow-random split and capture-derived labels can inflate scores",
            "VNAT historical F2 paired comparison available only on Known Validation and different training seed",
            "historical ISCX Test comparator CSV read prefit for ID/truth audit; no Test feature values or metrics used for fitting",
        ]
        manifest["next_step"] = "Stop Stage32; no automatic open-set or tuning experiment"
        table = "\n".join(
            f"| {r['dataset']} | {r['known_test_samples']} | {float(r['known_test_accuracy']):.6f} | "
            f"{float(r['known_test_macro_f1']):.6f} | {float(r['known_test_weighted_f1']):.6f} |"
            for r in report_rows)
        text = f"""# Experiment results: {manifest['experiment_id']}

- Status: `success / THREE_DATASET_SINGLE_SEED_DIAGNOSTIC_COMPLETE`
- Experiment type: `benchmark`; claim scope: `diagnostic`
- Created (UTC): `{manifest['created_at_utc']}`; completed (UTC): `{utc()}`

## Data and split

Frozen Stage20 ISCX-VPN/Tor full-flow Known Train/Validation/Test and Stage14B VNAT `medium_seed2025` Known roles. Stage25 fixed VPN4/Tor5 semantic map; VNAT original-paper application→category map with four observed Known categories. USTC and all Unknown samples excluded. Exact ID/label/source preflight passed for 3/3 units; 19 protected hashes unchanged.

## Configuration and execution

One training seed 2022 per dataset; TrafficFormer + FIG/TAGCN + YaTC frozen three-view representations, Stage30/31 T0 fixed 1/3 feature fusion, Known-Train-only standardization, 30 adapter + 30 head epochs, Known-Val checkpoint selection. Training tmux `codex_stage32_three_coarse_train_20260925` exited 0; gated Test tmux `codex_stage32_three_coarse_test_20260925` exited 0. Three heads were frozen before Test values opened. No Unknown fitting or Test parameter selection.

## Core results

| Dataset | Known Test flows | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|---:|
{table}

Paired differences and bootstrap intervals are in `stage32_paired_comparison.csv`; the full per-dataset interpretation is in `stage32_report.md`. VNAT historical F2 pairs on Known Validation only, not Known Test.

## Preserved evidence

`preflight.json`, frozen hashes before/after, `progress.json`, three run directories (60 training epochs and both best checkpoints each), per-flow validation/Test predictions, Test logits and per-class metrics, `stage32_test_results.csv`, `stage32_paired_comparison.csv`, `stage32_report.md`, `completion_verification.json`, `manifest.json`, and named tmux logs.

## Limitations

One seed cannot establish stability. The Test sets were exposed in historical work, so these are development results. The prefit ISCX historical Test CSV was scanned for ID/truth coverage, without using its metrics to fit or select; this is not a strict untouched Test. VNAT's flow-random split and capture-derived labels can leak capture signals. A coarse-label score rise is not an improvement on the old fine-label task, and no open-set conclusion follows.

## Conclusion and next step

Stop Stage32 after this completed diagnostic. Do not tune on Test or automatically launch open-set evaluation.
"""
        (ROOT / "RESULTS.md").write_text(text)
        append_once(PROJECT / "EXPERIMENT_RESULTS.md", "`stage32-three-dataset-coarse-single-seed-20260925-v1` terminal:",
            "- `stage32-three-dataset-coarse-single-seed-20260925-v1` terminal: `success / THREE_DATASET_SINGLE_SEED_DIAGNOSTIC_COMPLETE`; "
            + "; ".join(f"{r['dataset']} Known Test Macro-F1={float(r['known_test_macro_f1']):.6f}" for r in report_rows)
            + "; all 3 heads, 3 Test units, saved-logit replay and frozen hashes PASS; one seed, previously exposed Test and VNAT flow-random split limit claims. [bundle](stage32_three_dataset_coarse_single_seed/RESULTS.md).")
        append_once(PROJECT / "EXECUTION_PROGRESS.md", "Stage32 terminal update 2026-09-25:",
            "Stage32 terminal update 2026-09-25: status `complete / THREE_DATASET_SINGLE_SEED_DIAGNOSTIC_COMPLETE`. "
            + "; ".join(f"{r['dataset']} Known Test Macro-F1={float(r['known_test_macro_f1']):.6f}" for r in report_rows)
            + "; all three 2022-seed coarse T0 heads, one-shot Known Test units, saved-logit independent replay and protected hashes passed. Historical ISCX same-flow comparisons and VNAT Known-Val-only F2 comparison are in `stage32_three_dataset_coarse_single_seed/stage32_report.md`; no VNAT paired Test gain is claimed. Stop Stage32; no additional tuning or next stage launched.")
    else:
        manifest["limitations"] = ["Stage32 Test finalizer failed; preserve all partial caches and logs", state.get("error", "unknown failure")]
        manifest["next_step"] = "Inspect failure evidence; do not automatically rerun or tune on Test"
        with (ROOT / "RESULTS.md").open("a", encoding="utf-8") as f:
            f.write(f"\n## Terminal failure — {utc()}\n\nStage32 Test finalizer failed: `{state.get('error')}`. "
                    "Completed training and partial Test caches/results are preserved; no complete Test claim.\n")
        append_once(PROJECT / "EXPERIMENT_RESULTS.md", "`stage32-three-dataset-coarse-single-seed-20260925-v1` terminal:",
            "- `stage32-three-dataset-coarse-single-seed-20260925-v1` terminal: `partial / TEST_FAILED`; three Known-only coarse heads completed, but one-shot Test finalizer failed. Preserve evidence; no full three-dataset Test claim. [bundle](stage32_three_dataset_coarse_single_seed/RESULTS.md).")
        append_once(PROJECT / "EXECUTION_PROGRESS.md", "Stage32 terminal update 2026-09-25:",
            f"Stage32 terminal update 2026-09-25: status `in_progress / TEST_FAILED`; finalizer error `{state.get('error')}`. Three coarse heads remain preserved. Inspect Stage32 Test failure/logs and do not automatically rerun or tune on Test.")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    subprocess.run([sys.executable, str(REFRESH), str(ROOT)], check=True)
    subprocess.run([sys.executable, str(VALIDATE), str(ROOT), "--verify-hashes"], check=True)
    print(json.dumps({"status": "BUNDLE_VALIDATED", "experiment_status": manifest["status"],
                      "datasets": DATASETS}), flush=True)


if __name__ == "__main__":
    main()
