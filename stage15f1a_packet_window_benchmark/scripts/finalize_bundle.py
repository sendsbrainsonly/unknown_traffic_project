#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone

from common import PROJECT, ROOT, read_json, sha256_file, write_json


def read_rows(name: str):
    with (ROOT / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    gate = read_json(ROOT / "gate_evaluation.json")
    rows = read_rows("window_pilot_results.csv")
    manifest = read_json(ROOT / "manifest.json")
    manifest.update({
        "updated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "success",
        "inputs": [
            {"path": str((PROJECT / "stage15r_representation_bottleneck_audit").resolve()), "role": "frozen E2 and Native Known-Validation lineage"},
            {"path": str((PROJECT / "stage15f_literature_guided_feature_benchmark").resolve()), "role": "frozen literature registry and packet-window coverage"},
            {"path": str((ROOT / "sequence_cache").resolve()), "role": "Known Train/Validation-only first-32 packet cache"},
        ],
        "code": {
            "revision": None,
            "dirty": None,
            "changes": [
                "Added Stage15F-1A cache builder, mask-safe sequence CNN, controlled grid, aggregation, hash audit, tests, and completion verifier.",
                "Preserved the failed USTC direction re-inference cache attempt; corrected builder retains frozen PKL direction verbatim.",
            ],
        },
        "execution": {
            "tmux_session": "stage15f1a_grid_20260919_001",
            "command": "python scripts/run_grid.py --gpu-ids <live-selected-physical-ids>",
            "exit_code": 0,
            "environment": "/home/birkenwald/data/SEU-WXY/conda_envs/2025-10-8-WXY-dgl_py310",
            "physical_gpu_ids": sorted({str(row["physical_gpu"]) for row in rows}),
        },
        "configuration": {
            "files": [{"path": str((ROOT / "config.json").resolve()), "sha256": sha256_file(ROOT / "config.json")}],
            "parameters": read_json(ROOT / "config.json"),
            "seeds": [2022],
        },
        "core_results": [
            {"name": "final_gate", "value": gate["gate"]},
            {"name": "best_longer_window", "value": gate["best_longer_window"]},
            {"name": "best_mean_delta_macro_f1_vs_t8", "value": gate["window_summaries"][gate["best_longer_window"].removeprefix("T")] if isinstance(next(iter(gate["window_summaries"])), str) else gate["window_summaries"][int(gate["best_longer_window"].removeprefix("T"))]},
            {"name": "formal_runs", "value": len(rows), "unit": "runs"},
            {"name": "known_test_samples_used", "value": 0},
            {"name": "unknown_test_samples_used", "value": 0},
            {"name": "t64_started", "value": False},
        ],
        "limitations": [
            "Five preregistered pilot protocols only; this is a Known-Validation diagnostic, not an open-set or final-Test claim.",
            "Historical Stage15R E2 used ordinary BatchNorm; formal T8/T16/T32 use one mask-safe correction, so historical-to-formal T8 changes are not window effects.",
            "One fixed training seed is used, matching the preregistered Stage15R pilot design.",
        ],
        "next_step": "Stop after Stage15F-1A; use the frozen gate to decide whether a later statistical, burst, or structured-byte benchmark should be preregistered.",
    })
    write_json(ROOT / "manifest.json", manifest)

    index = PROJECT / "EXPERIMENT_RESULTS.md"
    text = index.read_text(encoding="utf-8")
    experiment_id = manifest["experiment_id"]
    if experiment_id not in text:
        best = gate["best_longer_window"]
        summary = gate["window_summaries"][best.removeprefix("T")]
        line = f"- `{experiment_id}`: `success / {gate['gate']}`; 15/15 Known-only T8/T16/T32 pilots complete; {best} mean ΔMacro-F1 vs T8=`{summary['mean_delta_macro_f1']:+.6f}`, positive protocols=`{summary['positive_protocols']}/5`, worst=`{summary['worst_delta_macro_f1']:+.6f}`; cache/parity/protected-hash checks PASS, Known Test/Unknown Test use=`0/0`, T64 and multiview training not run; [bundle](stage15f1a_packet_window_benchmark/RESULTS.md).\n"
        index.write_text(text.rstrip() + "\n" + line, encoding="utf-8")

    progress = PROJECT / "EXECUTION_PROGRESS.md"
    progress_text = progress.read_text(encoding="utf-8")
    marker = "### Stage 15F-1A terminal record"
    if marker not in progress_text:
        best = gate["best_longer_window"]
        summary = gate["window_summaries"][best.removeprefix("T")]
        addition = f"\n\n{marker}\n\n- Status: `complete`\n- Final Gate: `{gate['gate']}`\n- Formal runs: `15/15` (T8/T16/T32 x five frozen Known-only pilots).\n- {best} mean Delta Macro-F1 vs T8: `{summary['mean_delta_macro_f1']:+.6f}`; positive protocols `{summary['positive_protocols']}/5`; worst `{summary['worst_delta_macro_f1']:+.6f}`.\n- Known Test / Unknown Test usage: `0 / 0`; T64 and multiview training: `NOT_RUN`.\n- Evidence: `stage15f1a_packet_window_benchmark/RESULTS.md` and `completion_verification.json`.\n"
        progress.write_text(progress_text.rstrip() + addition, encoding="utf-8")
    print(json.dumps({"status": "PASS", "experiment_id": experiment_id, "gate": gate["gate"]}, sort_keys=True))


if __name__ == "__main__":
    main()
