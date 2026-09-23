#!/usr/bin/env python3
from __future__ import annotations

import csv
import datetime as dt
import json
import subprocess
from pathlib import Path

from stage22_common import CONFIG, OUT, PROJECT, read_json, write_json


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def fmt(value: str) -> str:
    return f"{float(value):.6f}"


def main() -> None:
    config = read_json(CONFIG)
    aggregate = load_csv(OUT / "aggregate_metrics.csv")
    deltas = load_csv(OUT / "e3_vs_branch_deltas.csv")
    queue = read_json(OUT / "queue_status.json")
    lookup = {(row["dataset"], row["encoder"]): row for row in aggregate}
    lines = [
        f"# Experiment results: {config['experiment_id']}",
        "",
        "- Status: `success / CLOSED_SET_DIAGNOSTIC_COMPLETE`",
        "- Experiment type: `benchmark`",
        "- Claim scope: `diagnostic`",
        "- Objective: compare official-pretrained TrafficFormer and the E3 fusion on identical frozen Stage20 samples.",
        "",
        "## Data and split",
        "",
        "- Reused Stage20 `closed_service_manifest.csv` without modification.",
        "- ISCX-VPN: 10,955 flows; Train/Validation/Test = 8,764/1,098/1,093; six Services.",
        "- ISCXTor2016: 11,181 flows; Train/Validation/Test = 8,946/1,118/1,117; seven Services.",
        "- Seeds: 2022 and 2023. Test samples used for checkpoint selection: 0.",
        "",
        "## Configuration and execution",
        "",
        f"- Official pretrained SHA256: `{config['pretrained_model_sha256']}`.",
        "- TrafficFormer: official 120k pretrained initialization, 20 epochs, batch 64, LR 6e-5, first pooling, sequence length 320.",
        "- TAGCN: unchanged Stage21 settings, 50 epochs, batch 64, LR 1e-3, hidden 128, K=2.",
        "- E3: Known-Train-only branch z-score, 768+128 concatenation, linear head, 30 epochs.",
        "- All checkpoints selected by Known Validation Macro-F1; Known Test was evaluated only after selection.",
        "",
        "## Core results",
        "",
        "| Dataset | Encoder | Accuracy mean±std | Macro-F1 mean±std | Weighted-F1 mean±std |",
        "|---|---|---:|---:|---:|",
    ]
    for dataset in ("iscx_vpn", "iscx_tor"):
        for encoder in ("E1", "E3"):
            row = lookup[(dataset, encoder)]
            lines.append(
                f"| {dataset} | {encoder} | {fmt(row['mean_accuracy'])}±{fmt(row['std_accuracy'])} | "
                f"{fmt(row['mean_macro_f1'])}±{fmt(row['std_macro_f1'])} | {fmt(row['mean_weighted_f1'])}±{fmt(row['std_weighted_f1'])} |"
            )
    lines.extend(["", "### E3 minus TrafficFormer", "", "| Dataset | Mean ΔAccuracy | Mean ΔMacro-F1 | Mean ΔWeighted-F1 | Positive Macro-F1 seeds |", "|---|---:|---:|---:|---:|"])
    for dataset in ("iscx_vpn", "iscx_tor"):
        rows = [row for row in deltas if row["dataset"] == dataset and row["comparison"] == "E3_minus_E1"]
        dacc = sum(float(row["delta_accuracy"]) for row in rows) / len(rows)
        dmacro = sum(float(row["delta_macro_f1"]) for row in rows) / len(rows)
        dweighted = sum(float(row["delta_weighted_f1"]) for row in rows) / len(rows)
        positive = sum(float(row["delta_macro_f1"]) > 0 for row in rows)
        lines.append(f"| {dataset} | {dacc:+.6f} | {dmacro:+.6f} | {dweighted:+.6f} | {positive}/{len(rows)} |")
    lines.extend([
        "",
        "## Preserved evidence",
        "",
        "- Four run directories with E1/E2/E3 checkpoints, training histories, representations and validation/test predictions.",
        "- `run_metrics.csv`, `aggregate_metrics.csv`, `per_class_metrics.csv`, `confusion_matrices.csv` and paired-delta tables.",
        "- `preflight.json`, protected input hashes, queue logs and `completion_verification.json`.",
        "",
        "## Limitations",
        "",
        "- The public TrafficFormer README does not establish the official pretraining corpus membership; this experiment is not a strict Unknown-Free open-set result.",
        "- Service labels are capture-derived weak labels and Stage20 is not uniformly capture-disjoint.",
        "- Only two seeds are included, so standard deviations are descriptive.",
        "- Stage21-to-Stage22 differences combine official pretraining and a 3-to-20 epoch budget increase; they are not a pure pretraining ablation.",
        "- This stage reports closed-set performance only and does not run unknown detection.",
        "",
        "## Conclusion and next step",
        "",
        "- Interpret E3 versus E1 only from the paired same-run deltas above. Do not compare these scores directly with historical TrafficFormer runs that used different flow populations.",
        "- Stop after the closed-set comparison; any open-set evaluation requires a separate frozen protocol decision about external pretraining exposure.",
        "",
    ])
    (OUT / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")

    manifest = read_json(OUT / "manifest.json")
    now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=PROJECT, text=True).strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        revision, dirty = None, None
    manifest.update({
        "updated_at_utc": now,
        "status": "success",
        "inputs": [
            {"path": str(PROJECT / "stage20_dual_coarse_service_protocol" / "closed_service_manifest.csv"), "role": "frozen split"},
            {"path": str(PROJECT / "stage21_coarse_service_ours_e3_benchmark" / "feature_cache"), "role": "frozen features"},
            {"path": str(PROJECT / "tf_runtime" / "code" / "models" / "pretrained_model.bin"), "role": "official pretrained initialization", "sha256": config["pretrained_model_sha256"]},
        ],
        "code": {"revision": revision, "dirty": dirty, "changes": ["new isolated Stage22 scripts and reports only"]},
        "execution": {
            "tmux_session": "stage22-pretrained-e3-grid-20260923",
            "command": "select_gpu.py --min-free-gb 36 --count 3 -- python stage22.../scripts/run_queue.py --device-indices 0,1,2",
            "exit_code": 0,
            "environment": "/home/birkenwald/data/SEU-WXY/conda_envs/2025-10-8-WXY-dgl_py310",
            "physical_gpu_ids": sorted({item.get("physical_gpu") for item in queue.values() if item.get("physical_gpu") is not None}),
        },
        "configuration": {"files": ["config.json"], "parameters": config["training"], "seeds": config["seeds"]},
        "core_results": [{"dataset": row["dataset"], "encoder": row["encoder"], "mean_accuracy": float(row["mean_accuracy"]), "mean_macro_f1": float(row["mean_macro_f1"]), "mean_weighted_f1": float(row["mean_weighted_f1"])} for row in aggregate if row["encoder"] in {"E1", "E3"}],
        "limitations": [
            "Official pretraining corpus exposure is unverified; not a strict Unknown-Free open-set result.",
            "Capture-derived weak labels and no uniform capture-disjoint split.",
            "Two seeds only.",
            "Stage21 comparison combines initialization and epoch-budget changes.",
        ],
        "next_step": "Stop after closed-set reporting; require a separate decision before any open-set use of the external pretrained checkpoint.",
    })
    write_json(OUT / "manifest.json", manifest)

    index = PROJECT / "EXPERIMENT_RESULTS.md"
    entry = f"- `{config['experiment_id']}`: completed official-pretrained TrafficFormer versus E3 closed-set same-sample comparison; [bundle]({OUT.name}/RESULTS.md)."
    existing = index.read_text(encoding="utf-8") if index.exists() else "# Experiment results index\n"
    if config["experiment_id"] not in existing:
        with index.open("a", encoding="utf-8") as handle:
            if not existing.endswith("\n"):
                handle.write("\n")
            handle.write(entry + "\n")


if __name__ == "__main__":
    main()
