#!/usr/bin/env python3
"""Join frozen Stage23 baselines and Stage23B adapted runs for review."""
from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path

OUT = Path(__file__).resolve().parents[1]
PROJECT = OUT.parent
BASE = PROJECT / "stage23_closed_set_method_table/stage23_full_run_results.csv"
BASE_CLASS = PROJECT / "stage23_closed_set_method_table/stage23_all_method_per_class.csv"
NEW = OUT / "stage23b_run_results.csv"
NEW_CLASS = OUT / "stage23b_per_class.csv"
METRICS = ("accuracy", "macro_f1", "weighted_f1")


def load(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"empty output: {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float) -> str:
    return f"{value:.4f}"


def main() -> None:
    base_rows = [r for r in load(BASE) if r["dataset"] in {"iscx_vpn", "iscx_tor"}]
    new_rows = load(NEW)
    all_rows = []
    for r in base_rows:
        all_rows.append({"dataset": r["dataset"], "seed": int(r["seed"]),
                         "method": r["method"], "method_scope": "Stage23 existing result",
                         **{m: float(r[m]) for m in METRICS},
                         "train_flows": int(r["train_flows"]),
                         "validation_flows": int(r["validation_flows"]),
                         "test_flows": int(r["test_flows"]),
                         "sample_parity": r["sample_parity"]})
    for r in new_rows:
        all_rows.append({"dataset": r["dataset"], "seed": int(r["seed"]),
                         "method": r["method"], "method_scope": r["claim_scope"],
                         **{m: float(r[m]) for m in METRICS},
                         "train_flows": int(r["train_flows"]),
                         "validation_flows": int(r["validation_flows"]),
                         "test_flows": int(r["test_flows"]),
                         "sample_parity": r["sample_parity"]})
    expected_rows = 28
    if len(all_rows) != expected_rows:
        raise RuntimeError(f"expected {expected_rows} run rows, got {len(all_rows)}")
    for dataset in ("iscx_vpn", "iscx_tor"):
        ds_rows = [r for r in all_rows if r["dataset"] == dataset]
        expected = {"train": 8764, "validation": 1098, "test": 1093} if dataset == "iscx_vpn" else {
            "train": 8946, "validation": 1118, "test": 1117}
        for r in ds_rows:
            for split, value in expected.items():
                if r[f"{split}_flows"] != value:
                    raise RuntimeError(f"sample-count mismatch: {dataset} {r['method']} {split}")
            if r["sample_parity"] != "PASS":
                raise RuntimeError(f"sample parity not PASS: {dataset} {r['method']}")
    all_rows.sort(key=lambda r: (r["dataset"], r["method"], r["seed"]))
    write(OUT / "stage23b_all_methods_run_level.csv", all_rows)

    grouped = defaultdict(list)
    for r in all_rows:
        grouped[(r["dataset"], r["method"])].append(r)
    summary = []
    for (dataset, method), group in sorted(grouped.items()):
        row = {"dataset": dataset, "method": method,
               "method_scope": group[0]["method_scope"], "seeds": "2022;2023"}
        for metric in METRICS:
            vals = [r[metric] for r in group]
            row[f"{metric}_mean"] = statistics.mean(vals)
            row[f"{metric}_std_population"] = statistics.pstdev(vals)
        summary.append(row)
    write(OUT / "stage23b_all_methods_summary.csv", summary)

    old_class = [r for r in load(BASE_CLASS)
                 if r["dataset"] in {"iscx_vpn", "iscx_tor"} and r["role"] == "known_test"]
    new_class = load(NEW_CLASS)
    per_class = []
    for r in old_class:
        per_class.append({"dataset": r["dataset"], "seed": int(r["seed"]),
                          "method": r["method"], "service": r["service"],
                          "precision": float(r["precision"]), "recall": float(r["recall"]),
                          "f1": float(r["f1"]), "support": int(r["support"]),
                          "method_scope": "Stage23 existing result"})
    for r in new_class:
        per_class.append({"dataset": r["dataset"], "seed": int(r["seed"]),
                          "method": r["method"], "service": r["service"],
                          "precision": float(r["precision"]), "recall": float(r["recall"]),
                          "f1": float(r["f1"]), "support": int(r["support"]),
                          "method_scope": "adapted"})
    if len(per_class) != 182:
        raise RuntimeError(f"expected 182 per-class rows, got {len(per_class)}")
    per_class.sort(key=lambda r: (r["dataset"], r["service"], r["method"], r["seed"]))
    write(OUT / "stage23b_all_methods_per_class.csv", per_class)

    class_grouped = defaultdict(list)
    for r in per_class:
        class_grouped[(r["dataset"], r["service"], r["method"])].append(r)

    # Paired differences are computed on the same frozen dataset/seed rows.
    paired = []
    baseline_lookup = {(r["dataset"], r["seed"], r["method"]): r for r in all_rows}
    candidates = ["TFE-GNN-8-UDP-short", "Trident-early8-86D"]
    baselines = sorted({r["method"] for r in base_rows})
    for candidate in candidates:
        for baseline in baselines:
            for dataset in ("iscx_vpn", "iscx_tor"):
                for seed in (2022, 2023):
                    c = baseline_lookup[dataset, seed, candidate]
                    b = baseline_lookup[dataset, seed, baseline]
                    paired.append({"dataset": dataset, "seed": seed, "candidate": candidate,
                                   "baseline": baseline,
                                   **{f"delta_{m}": c[m] - b[m] for m in METRICS}})
    write(OUT / "stage23b_all_methods_paired.csv", paired)
    paired_groups = defaultdict(list)
    for r in paired:
        paired_groups[(r["dataset"], r["candidate"], r["baseline"])].append(r)
    paired_summary = []
    for (dataset, candidate, baseline), group in sorted(paired_groups.items()):
        row = {"dataset": dataset, "candidate": candidate, "baseline": baseline,
               "seeds": "2022;2023"}
        for metric in METRICS:
            values = [r[f"delta_{metric}"] for r in group]
            row[f"delta_{metric}_mean"] = statistics.mean(values)
            row[f"delta_{metric}_std_population"] = statistics.pstdev(values)
            row[f"positive_seeds_{metric}"] = sum(x > 0 for x in values)
        paired_summary.append(row)
    write(OUT / "stage23b_all_methods_paired_summary.csv", paired_summary)

    report = [
        "# Stage 23B: all-method matched comparison",
        "",
        "All rows use the frozen Stage20 ISCX-VPN and ISCXTor Service splits,",
        "seeds 2022/2023. Stage23B rows are adapted methods; Stage23 rows are",
        "the previously completed matched baselines. TFE and Trident Stage23B",
        "results are not author-native reproductions.",
        "",
        "## ISCX-VPN: per-seed and mean ± population SD",
        "",
        "| Method | Acc 2022 | Acc 2023 | Acc mean±SD | Macro-F1 2022 | Macro-F1 2023 | Macro-F1 mean±SD | Weighted-F1 mean±SD |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in ("iscx_vpn", "iscx_tor"):
        if dataset == "iscx_tor":
            report += ["", "## ISCXTor2016: per-seed and mean ± population SD", "",
                       "| Method | Acc 2022 | Acc 2023 | Acc mean±SD | Macro-F1 2022 | Macro-F1 2023 | Macro-F1 mean±SD | Weighted-F1 mean±SD |",
                       "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for method in sorted({r["method"] for r in all_rows}):
            group = sorted([r for r in all_rows if r["dataset"] == dataset and r["method"] == method],
                           key=lambda x: x["seed"])
            acc = "{:.4f} ± {:.4f}".format(statistics.mean(r["accuracy"] for r in group),
                                            statistics.pstdev(r["accuracy"] for r in group))
            macro = "{:.4f} ± {:.4f}".format(statistics.mean(r["macro_f1"] for r in group),
                                              statistics.pstdev(r["macro_f1"] for r in group))
            weighted = "{:.4f} ± {:.4f}".format(statistics.mean(r["weighted_f1"] for r in group),
                                                 statistics.pstdev(r["weighted_f1"] for r in group))
            report.append("| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                method, fmt(group[0]["accuracy"]), fmt(group[1]["accuracy"]), acc,
                fmt(group[0]["macro_f1"]), fmt(group[1]["macro_f1"]), macro, weighted))
    aliases = {
        "OURS-E3-T8-pretrained": "E3-T8", "Open-Detect corrected-paper": "OpenDetect",
        "RoNeTC-runnable-reconstruction": "RoNeTC", "TFE-GNN-8-UDP-short": "TFE-GNN-adapt",
        "TrafficFormer-pretrained": "TrafficFormer", "Trident-early8-86D": "Trident-adapt",
        "YaTC-official-pretrained-Stage20": "YaTC",
    }
    report += ["", "## Per-service F1 averaged across seeds", ""]
    for dataset, title in (("iscx_vpn", "ISCX-VPN"), ("iscx_tor", "ISCXTor2016")):
        report += [f"### {title}", "", "| Service | " + " | ".join(aliases[m] for m in sorted(aliases)) + " |",
                   "|---|" + "---:|" * len(aliases)]
        services = sorted({r["service"] for r in per_class if r["dataset"] == dataset})
        for service in services:
            values = []
            for method in sorted(aliases):
                group = class_grouped.get((dataset, service, method), [])
                values.append(fmt(statistics.mean(r["f1"] for r in group)) if group else "—")
            report.append("| " + service + " | " + " | ".join(values) + " |")
    report += ["", "## Paired mean differences: adaptation minus baseline", "",
               "Positive values favor the Stage23B adaptation. Values are means of the two seed-level paired differences.",
               "", "| Dataset | Candidate | Baseline | ΔAccuracy | ΔMacro-F1 | ΔWeighted-F1 |", "|---|---|---|---:|---:|---:|"]
    for r in paired_summary:
        report.append("| {} | {} | {} | {:+.4f} | {:+.4f} | {:+.4f} |".format(
            r["dataset"], r["candidate"], r["baseline"], r["delta_accuracy_mean"],
            r["delta_macro_f1_mean"], r["delta_weighted_f1_mean"]))
    report += [
        "",
        "## Detailed source tables",
        "",
        "- `stage23b_all_methods_run_level.csv`: 28 run rows × Accuracy/Macro-F1/Weighted-F1.",
        "- `stage23b_all_methods_paired.csv`: 40 seed-paired rows, both adapted methods against all five baselines.",
        "- `stage23b_all_methods_paired_summary.csv`: 20 mean paired comparisons.",
        "- `stage23b_all_methods_per_class.csv`: 182 Known-Test service rows with precision, recall, F1 and support.",
        "- `stage23b_subgroup_analysis.csv`: TCP/UDP and 1–2, 3–4, 5–8 packet bins for Stage23B adaptations.",
        "",
        "The five existing baselines are TrafficFormer-pretrained, OURS-E3-T8-pretrained,",
        "Open-Detect corrected-paper, RoNeTC-runnable-reconstruction, and YaTC-official-pretrained-Stage20.",
        "All seven methods have matching Train/Validation/Test counts per dataset and PASS sample parity.",
    ]
    (OUT / "stage23b_all_methods_comparison.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"PASS: runs={len(all_rows)} summaries={len(summary)} per_class={len(per_class)} paired={len(paired)}")


if __name__ == "__main__":
    main()
