#!/usr/bin/env python3
"""Aggregate four frozen-score fusion runs without parameter selection."""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
DATASETS = ("iscx_vpn", "iscx_tor")
SEEDS = (2022, 2023)
METHODS = ("E3", "YaTC", "Fusion")


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    all_metrics, all_classes, all_samples, verify = [], [], [], []
    for dataset in DATASETS:
        for seed in SEEDS:
            run = OUT / "runs" / dataset / f"seed{seed}"
            if not (run / "SUCCESS").is_file():
                raise RuntimeError(f"run incomplete: {run}")
            check = json.loads((run / "verification.json").read_text())
            before = json.loads((run / "source_hashes_before.json").read_text())
            after = json.loads((run / "source_hashes_after.json").read_text())
            if before != after or check["e3_yatc_argmax_parity"] != "PASS" or check["weight_updates"] or check["unknown_usage"]:
                raise RuntimeError(f"integrity failure: {run}")
            verify.append(check)
            all_metrics.extend(read(run / "run_metrics.csv"))
            all_classes.extend(read(run / "per_class.csv"))
            all_samples.extend(read(run / "sample_predictions.csv"))
    if len(all_metrics) != 96 or len(all_samples) != 8852:
        raise RuntimeError(f"coverage mismatch: metrics={len(all_metrics)} samples={len(all_samples)}")
    write(OUT / "run_metrics.csv", all_metrics)
    write(OUT / "per_class.csv", all_classes)
    summary = []
    for dataset in DATASETS:
        for role in ("known_validation", "known_test"):
            for scope in ("all", "ge2"):
                for level in ("fine", "coarse"):
                    for method in METHODS:
                        part = [r for r in all_metrics if r["dataset"] == dataset and r["role"] == role
                                and r["scope"] == scope and r["level"] == level and r["method"] == method]
                        if len(part) != 2 or {int(r["seed"]) for r in part} != set(SEEDS):
                            raise RuntimeError("missing paired metric cells")
                        summary.append({"dataset": dataset, "role": role, "scope": scope, "level": level,
                                        "method": method, "seeds": 2,
                                        "samples_per_seed": int(part[0]["samples"]),
                                        **{f"{key}_mean": float(np.mean([float(r[key]) for r in part]))
                                           for key in ("accuracy", "macro_f1", "weighted_f1")},
                                        "macro_f1_std_pop": float(np.std([float(r["macro_f1"]) for r in part]))})
    write(OUT / "setting_summary.csv", summary)
    paired = []
    for dataset in DATASETS:
        for role in ("known_validation", "known_test"):
            for scope in ("all", "ge2"):
                for level in ("fine", "coarse"):
                    for seed in SEEDS:
                        group = {r["method"]: r for r in all_metrics if r["dataset"] == dataset and
                                 r["role"] == role and r["scope"] == scope and r["level"] == level and
                                 int(r["seed"]) == seed}
                        if set(group) != set(METHODS):
                            raise RuntimeError("missing method in pair")
                        for base in ("E3", "YaTC"):
                            paired.append({"dataset": dataset, "seed": seed, "role": role, "scope": scope,
                                           "level": level, "baseline": base, "candidate": "Fusion",
                                           **{f"delta_{key}": float(group["Fusion"][key])-float(group[base][key])
                                              for key in ("accuracy", "macro_f1", "weighted_f1")}})
    write(OUT / "paired_comparison.csv", paired)
    rescue = []
    for dataset in DATASETS:
        for seed in SEEDS:
            for role in ("known_validation", "known_test"):
                group = [r for r in all_samples if r["dataset"] == dataset and int(r["seed"]) == seed and r["role"] == role]
                if len(group) != len({r["flow_id"] for r in group}):
                    raise RuntimeError("duplicate flow in sample comparison")
                for scope in ("all", "ge2"):
                    rows = group if scope == "all" else [r for r in group if int(r["retained_ge2"])]
                    for level in ("fine", "coarse"):
                        for base in ("E3", "YaTC"):
                            for cls in sorted({r[f"true_{level}"] for r in rows}):
                                sub = [r for r in rows if r[f"true_{level}"] == cls]
                                c = Counter((r[f"pred_{base}_{level}"] == r[f"true_{level}"],
                                             r[f"pred_Fusion_{level}"] == r[f"true_{level}"]) for r in sub)
                                rescue.append({"dataset": dataset, "seed": seed, "role": role,
                                               "scope": scope, "level": level, "baseline": base,
                                               "class": cls, "samples": len(sub),
                                               "both_correct": c[(True, True)], "fusion_only_correct": c[(False, True)],
                                               "baseline_only_correct": c[(True, False)], "both_wrong": c[(False, False)]})
    write(OUT / "sample_rescue.csv", rescue)
    primary = [r for r in paired if r["role"] == "known_test" and r["scope"] == "all" and
               r["level"] == "coarse" and r["baseline"] == "YaTC"]
    by_dataset = {d: float(np.mean([r["delta_macro_f1"] for r in primary if r["dataset"] == d]))
                  for d in DATASETS}
    positive = sum(r["delta_macro_f1"] > 0 for r in primary)
    fine = [r for r in paired if r["role"] == "known_test" and r["scope"] == "all" and
            r["level"] == "fine" and r["baseline"] == "YaTC"]
    fine_delta = {d: float(np.mean([r["delta_macro_f1"] for r in fine if r["dataset"] == d]))
                  for d in DATASETS}
    gate = "FUSION_GAIN_EXPLORATORY" if all(x > 0 for x in by_dataset.values()) and positive >= 3 and all(x >= -0.005 for x in fine_delta.values()) else "FUSION_NOT_CONFIRMED"
    decision = {"gate": gate, "primary_coarse_delta_vs_yatc_by_dataset": by_dataset,
                "positive_primary_cells": positive, "total_primary_cells": 4,
                "fine_delta_vs_yatc_by_dataset": fine_delta,
                "weight_search": False, "unknown_usage": 0, "weight_updates": 0,
                "frozen_checkpoint_argmax_parity": "PASS", "source_hashes_unchanged": True,
                "stage20_test_previously_exposed": True}
    (OUT / "completion_verification.json").write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    lines = ["# Stage 26 — Frozen YaTC + E3 probability fusion", "",
             "This is a matched Stage20 development experiment; Test was previously exposed. No encoder training, weight search or Unknown use.",
             "", "## Full-flow Known Test Macro-F1 (mean ± population std, two seeds)", "",
             "| Dataset | Label task | E3 | YaTC | Equal-weight fusion | ΔFusion−YaTC |",
             "| --- | --- | ---: | ---: | ---: | ---: |"]
    for dataset in DATASETS:
        for level in ("fine", "coarse"):
            cells = {r["method"]: r for r in summary if r["dataset"] == dataset and r["role"] == "known_test"
                     and r["scope"] == "all" and r["level"] == level}
            delta = cells["Fusion"]["macro_f1_mean"]-cells["YaTC"]["macro_f1_mean"]
            fmt = lambda r: f"{r['macro_f1_mean']:.6f} ± {r['macro_f1_std_pop']:.6f}"
            lines.append(f"| {dataset} | {level} | {fmt(cells['E3'])} | {fmt(cells['YaTC'])} | {fmt(cells['Fusion'])} | {delta:+.6f} |")
    lines.extend(["", "## Primary gate and interpretation", "",
                  f"- Gate: `{gate}`. Coarse ΔFusion−YaTC by dataset: VPN {by_dataset['iscx_vpn']:+.6f}, Tor {by_dataset['iscx_tor']:+.6f}; positive paired cells {positive}/4.",
                  "- Fine task is retained as a guardrail; both original model checkpoints are unchanged. A two-model ensemble adds inference and storage cost; it is not a single improved E3 encoder.",
                  "- Coarse probabilities are obtained by summing the three Communication Service probabilities before argmax. This differs from Stage25's hard-prediction remap; compare within this Stage26 table only.",
                  "- Individual results, per-class metrics, conditional ≥2-packet results and sample-level rescue/degradation are preserved in the CSVs. Any isolated Test gain is development evidence, not a new weight-selection signal.",
                  "- YaTC and E3 use different representations, pretraining and training budgets; this comparison cannot attribute a gain or failure to one architectural module.",
                  "", "## Integrity", "",
                  "- Four/four frozen checkpoint pairs passed exact historical Validation/Test argmax parity; all per-run source SHA256 values unchanged.",
                  "- No Unknown data, Test fitting, encoder training, weight tuning or threshold calibration.",
                  ""])
    (OUT / "stage26_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(decision, indent=2), flush=True)


if __name__ == "__main__":
    main()
