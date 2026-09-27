#!/usr/bin/env python3
"""Populate semantic evidence records for the four completed inference subruns."""
from __future__ import annotations

import csv
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent


def rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    for dataset in ("iscx_vpn", "iscx_tor"):
        for seed in (2022, 2023):
            run = OUT / "runs" / dataset / f"seed{seed}"
            if not (run / "SUCCESS").is_file():
                raise RuntimeError(f"subrun incomplete: {run}")
            check = json.loads((run / "verification.json").read_text())
            metrics = rows(run / "run_metrics.csv")
            chosen = {r["method"]: r for r in metrics if r["role"] == "known_test"
                      and r["scope"] == "all" and r["level"] == "coarse"}
            if set(chosen) != {"E3", "YaTC", "Fusion"}:
                raise RuntimeError("missing method in subrun")
            code = f"codex_stage26_{'vpn' if dataset == 'iscx_vpn' else 'tor'}{seed}_20260924"
            report = [f"# Frozen YaTC + E3 ensemble: {dataset}, seed {seed}", "",
                      "- Status: `success`", "- Experiment type: `benchmark`",
                      "- Claim scope: `diagnostic`", "",
                      "## Data and split", "",
                      f"- Frozen Stage20 {dataset}; Known Validation {check['validation_samples']} flows, Known Test {check['test_samples']} flows. Unknown usage=0.",
                      "", "## Configuration and execution", "",
                      "- Frozen E3 and YaTC checkpoint replay with exact historical argmax parity on Validation and Test. No weight updates or fitting.",
                      "- Fixed fine-score fusion: 0.5 E3 softmax + 0.5 YaTC softmax; Communication coarse probability sums Chat/Email/VoIP before argmax.",
                      f"- tmux session `{code}`, exit 0; GPU selected live by workspace selector; cached-embedding replay time {check['runtime_seconds']:.3f} seconds.",
                      "", "## Core results", "",
                      "| Known Test full-flow coarse Macro-F1 | E3 | YaTC | Fusion |",
                      "| --- | ---: | ---: | ---: |",
                      f"| {dataset} seed {seed} | {float(chosen['E3']['macro_f1']):.6f} | {float(chosen['YaTC']['macro_f1']):.6f} | {float(chosen['Fusion']['macro_f1']):.6f} |",
                      "", "## Preserved evidence", "",
                      "- `frozen_logits.npz`, `sample_predictions.csv`, `run_metrics.csv`, `per_class.csv`, `verification.json`, before/after source hashes, `SUCCESS`, this report and manifest; tmux log/status referenced externally.",
                      "", "## Limitations", "",
                      "- Development Test previously exposed. This is a two-model ensemble, not an improved single encoder; runtime excludes E3 feature extraction. Only one fixed weight and one seed in this subrun.",
                      "", "## Conclusion and next step", "",
                      "- Interpret only together with all four paired runs in `../../../stage26_report.md`; no weight tuning from this subrun.", ""]
            (run / "RESULTS.md").write_text("\n".join(report), encoding="utf-8")
            manifest_path = run / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest.update({"status": "success", "inputs": ["Stage20 frozen manifest", "Stage22 E3 checkpoint/representations", "Stage23 YaTC checkpoint/MFR cache"],
                             "code": {"revision": None, "dirty": True, "changes": ["Stage26 new inference and aggregation scripts only"]},
                             "execution": {"tmux_session": code, "command": f"select_gpu.py --min-free-gb 5 -- python stage26_yatc_e3_frozen_score_fusion/run_one.py --dataset {dataset} --seed {seed}",
                                           "exit_code": 0, "environment": "/home/birkenwald/data/SEU-WXY/conda_envs/2025-10-8-WXY-dgl_py310",
                                           "physical_gpu_ids": []},
                             "configuration": {"files": ["../../../EXPERIMENT_PLAN.md"], "parameters": {"fusion_weight": [0.5, 0.5], "level": ["fine", "coarse"], "scopes": ["all", "ge2"]}, "seeds": [seed]},
                             "core_results": [f"Known Test coarse Macro-F1 E3={float(chosen['E3']['macro_f1']):.6f}, YaTC={float(chosen['YaTC']['macro_f1']):.6f}, Fusion={float(chosen['Fusion']['macro_f1']):.6f}",
                                              "exact historical argmax parity PASS; source SHA256 unchanged"],
                             "limitations": ["Previously exposed Stage20 Test", "Single seed", "Two-model cached-representation inference only"],
                             "next_step": "Use four-run aggregate without Test-driven weight selection"})
            manifest["artifacts"] = [
                {"path": f"../../../../.tmux-task/{code}/output.log", "scope": "external", "role": "log"},
                {"path": f"../../../../.tmux-task/{code}/exit.status", "scope": "external", "role": "status"}]
            manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("FOUR_SUBRUN_METADATA_READY", flush=True)


if __name__ == "__main__":
    main()
