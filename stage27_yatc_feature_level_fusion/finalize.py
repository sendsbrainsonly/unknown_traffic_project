#!/usr/bin/env python3
"""Finalize Stage 27 evidence bundles after independent replay passes."""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
ENV = "/home/birkenwald/data/SEU-WXY/conda_envs/2025-10-8-WXY-dgl_py310"
METHODS = ("E3", "YaTC", "F1_LinearConcat", "F2_EqualProjected", "F3_FeatureGate")


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def save_manifest(path: Path, value: dict) -> None:
    value["updated_at_utc"] = now()
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def external(session: str, relative_prefix: str) -> list[dict]:
    return [{"path": f"{relative_prefix}.tmux-task/{session}/{name}", "scope": "external", "role": role}
            for name, role in (("output.log", "log"), ("exit.status", "status"))]


def main() -> None:
    replay = json.loads((ROOT / "independent_verification.json").read_text())
    assert replay["status"] == "PASS" and replay["total_metric_rows_replayed"] == 120
    completion = json.loads((ROOT / "completion_verification.json").read_text())
    assert completion["status"] == "PASS" and completion["runs"] == 4
    setting = read_csv(ROOT / "setting_summary.csv")
    summary = {(r["dataset"], r["level"], r["method"]): float(r["macro_f1_mean"]) for r in setting}
    run_metrics = read_csv(ROOT / "run_metrics.csv")
    per_class = read_csv(ROOT / "per_class.csv")
    gates = read_csv(ROOT / "gate_weight_summary.csv")
    report = ROOT / "stage27_report.md"
    if "## Class-level fine-label change" not in report.read_text():
        lines = ["", "## Class-level fine-label change", "",
                 "Known Test class F1 averaged over the two fixed seeds; deltas are relative to frozen YaTC.", "",
                 "| Dataset | Service | E3 | YaTC | F1 concat−YaTC | F2 projected−YaTC | F3 gate−YaTC |",
                 "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
        grouped: dict[tuple[str, str, str], list[float]] = {}
        for row in per_class:
            if row["role"] == "known_test" and row["level"] == "fine":
                grouped.setdefault((row["dataset"], row["class"], row["method"]), []).append(float(row["f1"]))
        for dataset in ("iscx_vpn", "iscx_tor"):
            names = sorted({key[1] for key in grouped if key[0] == dataset})
            for name in names:
                values = {method: sum(grouped[(dataset, name, method)]) / 2 for method in METHODS}
                lines.append(f"| {dataset} | {name} | {values['E3']:.4f} | {values['YaTC']:.4f} | "
                             f"{values['F1_LinearConcat']-values['YaTC']:+.4f} | "
                             f"{values['F2_EqualProjected']-values['YaTC']:+.4f} | "
                             f"{values['F3_FeatureGate']-values['YaTC']:+.4f} |")
        lines += ["", "## Gate behavior and interpretation", ""]
        for dataset in ("iscx_vpn", "iscx_tor"):
            for seed in (2022, 2023):
                values = {r["branch"]: float(r["mean"]) for r in gates
                          if r["dataset"] == dataset and int(r["seed"]) == seed and r["role"] == "known_test"}
                lines.append(f"- {dataset}/{seed} mean gate TrafficFormer/FIG/YaTC: "
                             f"{values['TrafficFormer']:.6f}/{values['FIG']:.6f}/{values['YaTC']:.6f}.")
        lines += ["", "The gate often collapses toward TrafficFormer, so the adaptive mechanism did not consistently exploit YaTC or FIG. This is an observation, not a reason to retune it on the exposed Test set.",
                  "No open-set or Unknown detection result is produced by Stage 27.", ""]
        with report.open("a", encoding="utf-8") as stream:
            stream.write("\n".join(lines))

    for dataset in ("iscx_vpn", "iscx_tor"):
        for seed in (2022, 2023):
            run = ROOT / "runs" / dataset / f"seed{seed}"
            v = json.loads((run / "verification.json").read_text())
            assert v["status"] == "PASS" and (run / "SUCCESS").is_file()
            session = f"codex_stage27_{dataset}_{seed}_20260925"
            rows = [r for r in run_metrics if r["dataset"] == dataset and int(r["seed"]) == seed
                    and r["role"] == "known_test" and r["level"] == "fine"]
            scores = {r["method"]: float(r["macro_f1"]) for r in rows}
            assert set(scores) == set(METHODS)
            manifest_path = run / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest.update({
                "status": "success",
                "inputs": ["Stage20 frozen manifest", "Stage22 E3 representations/checkpoint",
                           "Stage23 YaTC best checkpoint and MFR cache"],
                "code": {"revision": None, "dirty": True, "changes": ["Stage27 independent scripts only"]},
                "execution": {"tmux_session": session, "command":
                              f"select_gpu.py --min-free-gb 5 -- python stage27_yatc_feature_level_fusion/run_one.py --dataset {dataset} --seed {seed}",
                              "exit_code": 0, "environment": ENV, "physical_gpu_ids": [0]},
                "configuration": {"files": ["../../EXPERIMENT_PLAN.md"],
                                  "parameters": {"epochs": 30, "batch_size": 256, "optimizer": "Adam",
                                                 "learning_rate": 0.001, "checkpoint_rule": "Known Validation fine Macro-F1",
                                                 "unknown_usage": 0, "encoder_updates": 0}, "seeds": [seed]},
                "core_results": [f"Known Test fine Macro-F1 {m}={scores[m]:.6f}" for m in METHODS] +
                                ["Independent prediction/metric/checkpoint replay PASS"],
                "artifacts": external(session, "../../../../"),
                "limitations": ["Stage20 Test was exposed previously; development evidence, not untouched validation.",
                                "Only two fixed seeds and two datasets; no open-set measurement."],
                "next_step": "Do not revise Stage27 formula using the exposed Test set.",
            })
            save_manifest(manifest_path, manifest)
            text = [f"# Stage 27 {dataset} seed {seed}", "", "- Status: success; diagnostic development run.",
                    "", "## Data and split", "", f"- Exact frozen Stage20 flow IDs; Known Train/Val/Test: "
                    f"{v['samples']['known_train']}/{v['samples']['known_validation']}/{v['samples']['known_test']}.",
                    "- Unknown usage: 0; encoders frozen.", "", "## Configuration and execution", "",
                    f"- Session `{session}`, physical GPU 0, exit 0; 30 epochs, Adam 1e-3, batch 256.",
                    "- Checkpoint selected by Known Validation fine Macro-F1; Test loaded afterward.",
                    "", "## Core results", "", "Known Test fine Macro-F1:", ""]
            text.extend(f"- {m}: {scores[m]:.6f}" for m in METHODS)
            text += ["", "- Frozen-source hashes unchanged; historical E3/YaTC predictions match; independent replay PASS.",
                     "", "## Preserved evidence", "", "- `features.npz`, `logits.npz`, `sample_predictions.csv`, "
                     "`run_metrics.csv`, `per_class.csv`, `training_history.csv`, `gate_weights.csv`, "
                     "all three best checkpoints and SHA256, verification files, session log/status.",
                     "", "## Limitations", "", "- Previously exposed Stage20 Test; no independent validation claim.",
                     "- Closed-set Known-only task; no Unknown or open-set metric.",
                     "", "## Conclusion and next step", "", "- Include in fixed Stage27 paired analysis; "
                     "do not tune a new fusion formula on this Test.", ""]
            (run / "RESULTS.md").write_text("\n".join(text), encoding="utf-8")

    root_manifest_path = ROOT / "manifest.json"
    root_manifest = json.loads(root_manifest_path.read_text())
    root_manifest.update({
        "status": "success",
        "inputs": ["Stage20 frozen manifest", "Stage22 frozen E3 representations/checkpoints",
                   "Stage23 frozen YaTC checkpoints/MFR caches"],
        "code": {"revision": None, "dirty": True, "changes": ["Stage27 independent scripts and evidence only"]},
        "execution": {"tmux_session": "; ".join(f"codex_stage27_{d}_{s}_20260925"
                         for d in ("iscx_vpn", "iscx_tor") for s in (2022, 2023)),
                      "command": "select_gpu.py --min-free-gb 5 -- python stage27_yatc_feature_level_fusion/run_one.py --dataset DATASET --seed SEED",
                      "exit_code": 0, "environment": ENV, "physical_gpu_ids": [0]},
        "configuration": {"files": ["EXPERIMENT_PLAN.md"], "parameters": {
            "feature_fusion": ["linear_concat", "equal_projected", "learned_feature_gate"],
            "known_only_normalization_and_training": True, "epochs": 30, "batch_size": 256,
            "unknown_usage": 0, "encoder_weight_updates": 0}, "seeds": [2022, 2023]},
        "core_results": [
            "4/4 runs, 120/120 independent metric replays PASS",
            f"VPN fine Macro-F1 E3/YaTC/F1/F2/F3=" + "/".join(f"{summary[('iscx_vpn','fine',m)]:.6f}" for m in METHODS),
            f"Tor fine Macro-F1 E3/YaTC/F1/F2/F3=" + "/".join(f"{summary[('iscx_tor','fine',m)]:.6f}" for m in METHODS),
            "F3 learned feature gate not stably superior; often collapses toward TrafficFormer"],
        "artifacts": sum((external(f"codex_stage27_{d}_{s}_20260925", "../")
                          for d in ("iscx_vpn", "iscx_tor") for s in (2022, 2023)), []) +
                     external("codex_stage27_aggregate_20260925", "../") +
                     external("codex_stage27_replay_20260925", "../"),
        "limitations": ["Stage20 Test was previously exposed; Stage27 is development evidence only.",
                        "Only two seeds and two datasets; no Unknown/open-set evaluation.",
                        "Feature-level adaptation is not literal ER-CMGI entropy/diffusion implementation."],
        "next_step": "Stop here; any further design needs a separate pre-registered protocol and fresh validation.",
    })
    save_manifest(root_manifest_path, root_manifest)
    text = ["# Stage 27 — YaTC feature-level fusion with E3", "", "- Status: success; diagnostic development benchmark.",
            "", "## Data and split", "", "- Exact frozen Stage20 ISCX-VPN/ISCXTor Known Train/Val/Test flow IDs, seeds 2022/2023.",
            "- Original fine Service labels primary; fixed Communication hard-remap secondary. Unknown usage: 0.",
            "", "## Configuration and execution", "", "- Frozen E3 896-D and YaTC 192-D penultimate features; "
            "Known-Train-only YaTC z-score. F1 linear concat, F2 equal projected, F3 learned feature gate.",
            "- Three new heads per run; Adam 1e-3, batch 256, 30 full epochs; select on Known Val fine Macro-F1.",
            "- Four named tmux GPU0 runs exit 0; source hashes unchanged; no encoder weight updates.",
            "", "## Core results", "", "Known Test fine Macro-F1, mean of two seeds:", "",
            "| Dataset | E3 | YaTC | F1 concat | F2 projection | F3 gate |",
            "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for dataset in ("iscx_vpn", "iscx_tor"):
        text.append(f"| {dataset} | " + " | ".join(f"{summary[(dataset,'fine',m)]:.6f}" for m in METHODS) + " |")
    text += ["", "- 120/120 metrics independently replayed from saved logits and predictions; all checkpoint hashes PASS.",
             "- F1/F2 sometimes add fine-label signal, especially Tor; F3 did not show stable improvement and often collapses to TrafficFormer.",
             "", "## Preserved evidence", "", "- `stage27_report.md`, `run_metrics.csv`, `per_class.csv`, "
             "`paired_comparison.csv`, `setting_summary.csv`, `gate_weight_summary.csv`, `independent_verification.json`, "
             "all four full run bundles and tmux logs/status.",
             "", "## Limitations", "", "- Stage20 Test was previously exposed, so the figures are development evidence only.",
             "- No Unknown detection/open-set result; cannot infer open-set benefit from closed-set Macro-F1.",
             "- Learned gate is a feature-level adaptation, not a reproduction of ER-CMGI.",
             "", "## Conclusion and next step", "", "- The intended feature-layer question is now tested. "
             "Do not tune another formula on this exposed Test or silently replace Stage26.", ""]
    (ROOT / "RESULTS.md").write_text("\n".join(text), encoding="utf-8")
    index = PROJECT / "EXPERIMENT_RESULTS.md"
    marker = "stage27-yatc-feature-level-fusion-20260925-v1"
    if marker not in index.read_text():
        with index.open("a", encoding="utf-8") as stream:
            stream.write(f"\n- `{marker}`: `success / FEATURE_LEVEL_DIAGNOSTIC`; 4/4 frozen-encoder "
                         "feature-level runs, 120 independent metric replays PASS; F1/F2 fine-label gains "
                         "are dataset-dependent, F3 gate not stable; Stage20 Test previously exposed; "
                         "[bundle](stage27_yatc_feature_level_fusion/RESULTS.md).\n")
    progress = PROJECT / "EXECUTION_PROGRESS.md"
    terminal = "Stage 27 terminal update 2026-09-25"
    if terminal not in progress.read_text():
        with progress.open("a", encoding="utf-8") as stream:
            stream.write("\n" + terminal + ": status `complete / FEATURE_LEVEL_DIAGNOSTIC`. "
                         "Four frozen E3+YaTC feature-level runs finished; 120/120 metrics independently "
                         "replayed, all source/checkpoint hashes passed, Unknown usage and encoder updates=0. "
                         "Fine Macro-F1 VPN E3/YaTC/F1/F2/F3=0.859493/0.887085/0.888900/0.884258/0.880846; "
                         "Tor=0.822178/0.827393/0.837946/0.842260/0.821191. F3 often collapsed toward "
                         "TrafficFormer. Stage20 Test was previously exposed; results are development only, "
                         "not independent validation. Stage26 probability averaging remains separate. "
                         "Full evidence: `stage27_yatc_feature_level_fusion/RESULTS.md`.\n")
    print("Stage27 finalized: 4 runs, root bundle, report, index, progress")


if __name__ == "__main__":
    main()
