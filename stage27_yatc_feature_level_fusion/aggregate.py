#!/usr/bin/env python3
"""Aggregate four fixed Stage27 feature-level fusion runs without model selection."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
METHODS = ("E3", "YaTC", "F1_LinearConcat", "F2_EqualProjected", "F3_FeatureGate")
FUSED = METHODS[2:]
DATASETS = ("iscx_vpn", "iscx_tor")
SEEDS = (2022, 2023)


def rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, data: list[dict]) -> None:
    if not data:
        raise RuntimeError(f"empty output {path}")
    with path.open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)


def main() -> None:
    all_metrics, all_class, comparisons, gate_summary = [], [], [], []
    for dataset in DATASETS:
        for seed in SEEDS:
            run = OUT / "runs" / dataset / f"seed{seed}"
            if not (run / "SUCCESS").is_file():
                raise RuntimeError(f"incomplete run: {run}")
            check = json.loads((run / "verification.json").read_text())
            before = json.loads((run / "source_hashes_before.json").read_text())
            after = json.loads((run / "source_hashes_after.json").read_text())
            if check["status"] != "PASS" or before != after or not check["source_hashes_unchanged"]:
                raise RuntimeError(f"frozen asset verification failed: {run}")
            if check["encoder_weight_updates"] or check["unknown_usage"] or not check["test_loaded_after_head_selection"]:
                raise RuntimeError(f"protocol violation: {run}")
            rr, cc = rows(run / "run_metrics.csv"), rows(run / "per_class.csv")
            if len(rr) != 3 * 2 * len(METHODS):
                raise RuntimeError(f"metric row count: {run}")
            all_metrics.extend(rr)
            all_class.extend(cc)
            test = {(r["level"], r["method"]): r for r in rr if r["role"] == "known_test"}
            if len(test) != 2 * len(METHODS):
                raise RuntimeError(f"duplicate/missing test metrics: {run}")
            for level in ("fine", "coarse"):
                for candidate in FUSED:
                    for baseline in ("E3", "YaTC", "F2_EqualProjected"):
                        if baseline == candidate:
                            continue
                        a, b = test[(level, candidate)], test[(level, baseline)]
                        comparisons.append({
                            "dataset": dataset, "seed": seed, "level": level, "candidate": candidate,
                            "baseline": baseline,
                            "delta_accuracy": float(a["accuracy"]) - float(b["accuracy"]),
                            "delta_macro_f1": float(a["macro_f1"]) - float(b["macro_f1"]),
                            "delta_weighted_f1": float(a["weighted_f1"]) - float(b["weighted_f1"]),
                        })
            weights = rows(run / "gate_weights.csv")
            expected = sum(check["samples"].values())
            if len(weights) != expected:
                raise RuntimeError(f"gate sample count: {run}")
            for role in ("known_train", "known_validation", "known_test"):
                group = [x for x in weights if x["role"] == role]
                for name, key in (("TrafficFormer", "weight_trafficformer"), ("FIG", "weight_fig"), ("YaTC", "weight_yatc")):
                    v = np.asarray([float(x[key]) for x in group])
                    gate_summary.append({"dataset": dataset, "seed": seed, "role": role,
                                         "branch": name, "samples": len(v), "mean": float(v.mean()),
                                         "std": float(v.std()), "p05": float(np.quantile(v, .05)),
                                         "median": float(np.median(v)), "p95": float(np.quantile(v, .95))})
    write_csv(OUT / "run_metrics.csv", all_metrics)
    write_csv(OUT / "per_class.csv", all_class)
    write_csv(OUT / "paired_comparison.csv", comparisons)
    write_csv(OUT / "gate_weight_summary.csv", gate_summary)
    grouped = defaultdict(list)
    for r in all_metrics:
        if r["role"] == "known_test":
            grouped[(r["dataset"], r["level"], r["method"])].append(r)
    summary = []
    for (dataset, level, method), group in sorted(grouped.items()):
        if len(group) != 2:
            raise RuntimeError(f"missing seed: {dataset}/{level}/{method}")
        item = {"dataset": dataset, "level": level, "method": method, "runs": len(group)}
        for metric in ("accuracy", "macro_f1", "weighted_f1"):
            values = np.asarray([float(x[metric]) for x in group])
            item[f"{metric}_mean"] = float(values.mean())
            item[f"{metric}_std"] = float(values.std())
        summary.append(item)
    write_csv(OUT / "setting_summary.csv", summary)
    lines = ["# Stage 27 — Frozen YaTC feature-level fusion", "",
             "Claim scope: development diagnostic; Stage20 Test was previously exposed.", "",
             "All four YaTC and E3 encoders/checkpoints remained frozen. Three newly trained Known-only feature heads were compared on exactly matched Stage20 flows. Stage26 probability averaging is a separate control, not this feature fusion.", "",
             "## Known Test Macro-F1 (mean of seeds 2022/2023)", "",
             "| Dataset | Labels | E3 | YaTC | F1 concat | F2 equal-projected | F3 feature gate |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    lookup = {(r["dataset"], r["level"], r["method"]): r for r in summary}
    for dataset in DATASETS:
        for level in ("fine", "coarse"):
            vals = [lookup[(dataset, level, m)]["macro_f1_mean"] for m in METHODS]
            lines.append(f"| {dataset} | {level} | " + " | ".join(f"{v:.6f}" for v in vals) + " |")
    lines += ["", "## Paired interpretation", ""]
    for method in FUSED:
        for dataset in DATASETS:
            delta = [float(r["delta_macro_f1"]) for r in comparisons
                     if r["candidate"] == method and r["baseline"] == "YaTC"
                     and r["dataset"] == dataset and r["level"] == "fine"]
            lines.append(f"- {method} − YaTC, {dataset} fine: mean {np.mean(delta):+.6f}; positive seeds {sum(x > 0 for x in delta)}/2.")
    lines += ["", "Feature concatenation and feature gating are distinct from Stage26 output-score averaging. The gate is learned from Known Train features only; no ER-CMGI latent entropy, diffusion, or Unknown data was used.",
              "The prior Stage20 Test exposure precludes independent confirmation and must not be used to revise the formula or choose a new width/weight.",
              ""]
    (OUT / "stage27_report.md").write_text("\n".join(lines), encoding="utf-8")
    verification = {"status": "PASS", "runs": 4, "methods_per_run": len(METHODS),
                    "frozen_source_hashes_unchanged": True, "encoder_weight_updates": 0,
                    "unknown_usage": 0, "test_loaded_after_head_selection": True,
                    "test_metric_rows": sum(r["role"] == "known_test" for r in all_metrics),
                    "test_previously_exposed": True}
    (OUT / "completion_verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    print(json.dumps(verification), flush=True)


if __name__ == "__main__":
    main()
