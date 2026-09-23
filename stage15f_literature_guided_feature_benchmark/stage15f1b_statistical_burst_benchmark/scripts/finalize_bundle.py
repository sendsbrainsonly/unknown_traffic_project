#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import datetime, timezone

from common import PROJECT, ROOT, read_csv, read_json, sha256_file, write_json


def main() -> None:
    gate = read_json(ROOT / "gate_evaluation.json")
    results = read_csv(ROOT / "statistical_pilot_results.csv")
    verification = read_json(ROOT / "completion_verification.json")
    if verification["status"] != "PASS":
        raise RuntimeError("completion verification is not PASS")
    manifest = read_json(ROOT / "manifest.json")
    manifest.update({
        "updated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "success",
        "inputs": [
            {"path": str((PROJECT / "stage15r_representation_bottleneck_audit").resolve()), "role": "frozen E3 S12 and LightGBM baseline"},
            {"path": str((PROJECT / "stage15f1a_packet_window_benchmark").resolve()), "role": "frozen T16 and Native comparison evidence"},
            {"path": str(ROOT.parent.resolve()), "role": "Stage 15F-0 frozen registry and lineage"},
        ],
        "code": {"revision": None, "dirty": None, "changes": ["Added controlled FULL_FLOW/EARLY_16 statistical and burst extraction, parity replay, 45-run LightGBM grid, aggregation, and completion verification."]},
        "execution": {"tmux_session": "stage15f1b_grid_20260920_001", "command": "python scripts/run_grid.py", "exit_code": 0, "environment": "/home/birkenwald/data/SEU-WXY/conda_envs/2025-10-8-WXY-dgl_py310", "physical_gpu_ids": []},
        "configuration": {"files": [{"path": str((ROOT / "config.json").resolve()), "sha256": sha256_file(ROOT / "config.json")}, {"path": str((ROOT / "feature_definitions.json").resolve()), "sha256": sha256_file(ROOT / "feature_definitions.json")}], "parameters": read_json(ROOT / "config.json"), "seeds": [2022]},
        "core_results": [
            {"name": "conclusions", "value": gate["conclusions"]},
            {"name": "S_ABCD_minus_S12_mean_delta_macro_f1", "value": gate["statistical_summary"]["mean_delta_macro_f1"]},
            {"name": "S_Burst_minus_S_ABCD_mean_delta_macro_f1", "value": gate["burst_summary"]["mean_delta_macro_f1"]},
            {"name": "formal_runs", "value": len(results), "unit": "runs"},
            {"name": "known_test_samples_used", "value": 0},
            {"name": "unknown_test_samples_used", "value": 0},
            {"name": "full15_status", "value": gate["full15_status"]},
        ],
        "limitations": ["Five frozen Known-Validation pilots with one seed.", "FULL_FLOW and EARLY_16 are different observation budgets.", "Gain importance is diagnostic rather than causal."],
        "next_step": "Stop after Stage 15F-1B. Do not start Stage 15F-1C, full15, open-set evaluation, or Byte-Behavior development automatically.",
    })
    write_json(ROOT / "manifest.json", manifest)
    index = PROJECT / "EXPERIMENT_RESULTS.md"
    text = index.read_text(encoding="utf-8")
    experiment_id = manifest["experiment_id"]
    if experiment_id not in text:
        text = text.rstrip() + f"\n- `{experiment_id}`: `success / {','.join(gate['conclusions'])}`; 45/45 Known-only FULL_FLOW/EARLY_16 pilots complete; S-ABCD-S12 mean ΔMacro-F1=`{gate['statistical_summary']['mean_delta_macro_f1']:+.6f}`, S-Burst-S-ABCD=`{gate['burst_summary']['mean_delta_macro_f1']:+.6f}`; Known Test/Unknown Test=`0/0`; full15 and Stage15F-1C not run; [bundle](stage15f_literature_guided_feature_benchmark/stage15f1b_statistical_burst_benchmark/RESULTS.md).\n"
        index.write_text(text, encoding="utf-8")
    progress = PROJECT / "EXECUTION_PROGRESS.md"
    progress_text = progress.read_text(encoding="utf-8")
    marker = "### Stage 15F-1B terminal record"
    if marker not in progress_text:
        progress_text = progress_text.rstrip() + f"\n\n{marker}\n\n- Status: `complete`.\n- Conclusions: `{', '.join(gate['conclusions'])}`.\n- Formal runs: `45/45`; S12 parity: `5/5 PASS`; protected hash and completion verification: `PASS`.\n- S-ABCD-S12 mean ΔMacro-F1: `{gate['statistical_summary']['mean_delta_macro_f1']:+.6f}`; S-Burst-S-ABCD: `{gate['burst_summary']['mean_delta_macro_f1']:+.6f}`.\n- Known Test / Unknown Test: `0 / 0`; full15, Stage15F-1C, open-set evaluation, DES/H1 changes, and Byte-Behavior model: `NOT_RUN`.\n- Evidence: `stage15f_literature_guided_feature_benchmark/stage15f1b_statistical_burst_benchmark/RESULTS.md`.\n"
        progress.write_text(progress_text, encoding="utf-8")
    print(json.dumps({"status": "PASS", "experiment_id": experiment_id, "conclusions": gate["conclusions"]}, sort_keys=True))


if __name__ == "__main__":
    main()
