#!/usr/bin/env python3
"""Durable post-queue report, bundle inventory and validation."""
from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
SKILL = PROJECT.parents[1] / "skills/experiment-data-preservation/scripts"
QUEUE_SESSION = os.environ.get("STAGE34_QUEUE_SESSION", "stage34_bounded_queue")
QUEUE_STATUS = PROJECT / ".tmux-task" / QUEUE_SESSION / "exit.status"


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    while not QUEUE_STATUS.exists():
        time.sleep(20)
    code = int(QUEUE_STATUS.read_text().strip())
    results = ROOT / "RESULTS.md"
    manifest_path = ROOT / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if code != 0:
        failure = json.loads((ROOT / "queue_failure.json").read_text())
        results.write_text(results.read_text() +
            f"\n## Terminal status ({now()})\n\n- `FAILED_INCOMPLETE`; queue exit={code}; stage: {failure.get('stage', 'unknown')}. No final two-dataset performance conclusion. Inspect `queue_failure.json` and named tmux logs; preserve all partial results.\n")
        manifest.update({"status": "failed", "updated_at_utc": now(), "next_step": "Diagnose queue_failure.json; preserve partial runs before any explicit retry"})
        manifest["execution"]["exit_code"] = code
    else:
        verification = json.loads((ROOT / "completion_verification.json").read_text())
        if verification["status"] != "PASS":
            raise RuntimeError("queue exited zero without PASS verification")
        u, c = verification["ustc"], verification["cicids2017"]
        def line(name, value):
            return f"| {name} | {len(value.get('known_classes', value.get('classes', [])))} | {value['known_test_samples']:,} | {value['accuracy']:.6f} | {value['macro_f1']:.6f} | {value['weighted_f1']:.6f} |"
        report = "# Stage34 USTC and CIC strict closed-set results\n\n"
        report += "| Dataset/protocol | Classes | Known Test | Accuracy | Macro-F1 | Weighted-F1 |\n|---|---:|---:|---:|---:|---:|\n"
        report += line("USTC A-2", u) + "\n" + line("CIC strict group-disjoint 3-class", c) + "\n\n"
        report += "## CIC per-class Test results\n\n| Class | Support | Precision | Recall | F1 |\n|---|---:|---:|---:|---:|\n"
        for row in c["per_class"]:
            report += f"| {row['class']} | {row['support']} | {row['precision']:.6f} | {row['recall']:.6f} | {row['f1']:.6f} |\n"
        report += "\n## Interpretation limits\n\n- USTC has 17 Known classes; CIC has three labels and is not an equal-difficulty comparison.\n"
        report += "- CIC Slowhttptest and PortScan each originate from one day/PCAP. Five-minute group disjointness does not remove this capture/day confound.\n"
        report += "- CIC Slowhttptest Test support is 132 flows in one group. No Unknown samples or Test-driven selection were used.\n"
        report += "- TrafficFormer raw-byte inputs retain network-header content; endpoint/capture shortcuts may inflate CIC scores.\n"
        report += "- This is development evidence, not untouched external validation. Historical experiments were not modified.\n"
        report += "\n## Reproducibility\n\nSee `ustc_a2_10pct_manifest.csv`, `cicids2017_three_class_manifest_v2.csv`, `cicids2017_group_assignments.csv`, `completion_verification.json`, and named `.tmux-task/stage34_*` logs.\n"
        (ROOT / "stage34_report.md").write_text(report)
        results.write_text(results.read_text() + f"\n## Terminal status ({now()})\n\n- `COMPLETE / DIAGNOSTIC`; USTC Test Acc/Macro-F1/Weighted-F1={u['accuracy']:.6f}/{u['macro_f1']:.6f}/{u['weighted_f1']:.6f}; CIC={c['accuracy']:.6f}/{c['macro_f1']:.6f}/{c['weighted_f1']:.6f}. Full table and limitations: [stage34_report.md](stage34_report.md). Both checkpoint/source gates and single-use Known Test evaluations passed.\n")
        manifest.update({"status": "complete", "updated_at_utc": now(),
                         "core_results": [{"dataset": "USTC-TFC2016", "protocol": "A-2 17-class 10%", "accuracy": u["accuracy"], "macro_f1": u["macro_f1"], "weighted_f1": u["weighted_f1"], "test_samples": u["known_test_samples"]},
                                          {"dataset": "CIC-IDS-2017", "protocol": "strict 3-class group-disjoint", "accuracy": c["accuracy"], "macro_f1": c["macro_f1"], "weighted_f1": c["weighted_f1"], "test_samples": c["known_test_samples"]}],
                         "next_step": "No automatic next-stage experiment; interpret the two tasks separately"})
        manifest["execution"]["exit_code"] = 0
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    subprocess.run(["python", str(SKILL / "refresh_artifact_manifest.py"), str(ROOT)], check=True, cwd=PROJECT)
    subprocess.run(["python", str(SKILL / "validate_experiment_bundle.py"), str(ROOT), "--verify-hashes"], check=True, cwd=PROJECT)
    print(json.dumps({"bundle_status": manifest["status"], "queue_exit": code, "verified_at": now()}), flush=True)


if __name__ == "__main__":
    main()
