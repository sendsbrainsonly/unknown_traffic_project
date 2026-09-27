#!/usr/bin/env python3
"""Independent Known-Validation-only Stage24 pilot aggregation and hash audit."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

OUT = Path(__file__).resolve().parents[1]
PROJECT = OUT.parent
METHODS = ("S1", "G1", "G2", "G2-shuffle")
DATASETS = ("iscx_vpn", "iscx_tor")
SEEDS = (2022, 2023)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def mean_std(values: list[float]) -> tuple[float, float]:
    return float(np.mean(values)), float(np.std(values, ddof=0))


def metric_for(rows: list[dict]) -> dict:
    true = [row["true_class"] for row in rows]
    pred = [row["pred_candidate"] for row in rows]
    labels = sorted(set(true))
    _p, _r, f1, support = precision_recall_fscore_support(
        true, pred, labels=labels, zero_division=0)
    return {"accuracy": float(accuracy_score(true, pred)),
            "macro_f1": float(np.mean(f1)),
            "weighted_f1": float(np.average(f1, weights=support))}


def main() -> None:
    if not (OUT / "QUEUE_COMPLETE").is_file():
        raise RuntimeError("16-run pilot queue is not verified complete")
    before = json.loads((OUT / "protected_hashes_before.json").read_text())
    source_paths = {"stage20_manifest": PROJECT / "stage20_dual_coarse_service_protocol" / "closed_service_manifest.csv"}
    for dataset in DATASETS:
        source_paths[f"stage21_{dataset}_cache"] = (PROJECT / "stage21_coarse_service_ours_e3_benchmark" /
                                                      "feature_cache" / dataset / "e3_t8_inputs.npz")
        for seed in SEEDS:
            run = PROJECT / "stage22_pretrained_trafficformer_e3_closed_set_comparison" / "runs" / dataset / f"seed{seed}"
            for name in ("E1_model_best.pt", "E2_model_best.pt", "E3_model_best.pt", "representations.npz"):
                source_paths[f"stage22_{dataset}_{seed}_{name}"] = run / name
    after = {name: sha256(path) for name, path in source_paths.items()}
    if before != after:
        write_json(OUT / "protected_hashes_after.json", after)
        raise RuntimeError("protected Stage20–22 assets changed")
    write_json(OUT / "protected_hashes_after.json", after)

    result_rows = []
    per_class_rows = []
    rescue_rows = []
    count_rows = []
    baseline_by_cell = {}
    for dataset in DATASETS:
        for seed in SEEDS:
            for method in METHODS:
                run = OUT / "runs" / "pilot" / dataset / f"seed{seed}" / method
                if not (run / "SUCCESS").is_file():
                    raise RuntimeError(f"run not complete: {run}")
                result = json.loads((run / "result.json").read_text())
                if result["known_test_feature_usage"] or result["unknown_test_feature_usage"]:
                    raise RuntimeError(f"Test usage flag in {run}")
                if result["input_hashes"]["stage20_manifest_sha256"] != before["stage20_manifest"]:
                    raise RuntimeError(f"source hash mismatch: {run}")
                baseline = (result["validation"]["B0"], result["validation"]["B1"])
                key = (dataset, seed)
                if key in baseline_by_cell and baseline_by_cell[key] != baseline:
                    raise RuntimeError(f"baseline inconsistency: {key}")
                baseline_by_cell[key] = baseline
                current = result["validation"][method]
                result_rows.append({"dataset": dataset, "seed": seed, "method": method,
                                    "train_samples": result["train_samples"],
                                    "val_samples": result["val_samples"],
                                    "val_accuracy": current["accuracy"],
                                    "val_macro_f1": current["macro_f1"],
                                    "val_weighted_f1": current["weighted_f1"],
                                    "baseline_e1_macro_f1": baseline[0]["macro_f1"],
                                    "baseline_e3_macro_f1": baseline[1]["macro_f1"],
                                    "delta_macro_vs_e1": current["macro_f1"] - baseline[0]["macro_f1"],
                                    "delta_macro_vs_e3": current["macro_f1"] - baseline[1]["macro_f1"],
                                    "branch_val_macro_f1": result["validation"]["branch"]["macro_f1"],
                                    "branch_best_epoch": result["branch_best_epoch"],
                                    "fusion_best_epoch": result["fusion_best_epoch"],
                                    "parameter_count": result["parameter_count"],
                                    "wall_seconds": result["wall_seconds"],
                                    "peak_gpu_memory_allocated_bytes": result["peak_gpu_memory_allocated_bytes"]})
                class_rows = read_csv(run / "validation_per_class.csv")
                per_class_rows.extend(row for row in class_rows if row["method"] in (method, "B0", "B1"))
                predictions = read_csv(run / "validation_predictions.csv")
                if len(predictions) != result["val_samples"] or len({r["flow_id"] for r in predictions}) != len(predictions):
                    raise RuntimeError(f"prediction coverage mismatch: {run}")
                grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
                for row in predictions:
                    n = int(row["packet_count"])
                    bucket = "1" if n == 1 else "2" if n == 2 else "3-8"
                    for group in ((row["true_class"], bucket), ("ALL", bucket)):
                        grouped[group].append({"true_class": row["true_class"],
                                               "pred_candidate": row[f"pred_{method}"],
                                               "pred_B0": row["pred_B0"], "pred_B1": row["pred_B1"]})
                for (class_name, bucket), values in grouped.items():
                    base = sum(v["pred_B0"] == v["true_class"] for v in values)
                    new = sum(v["pred_candidate"] == v["true_class"] for v in values)
                    rescue_rows.append({"dataset": dataset, "seed": seed, "method": method,
                                        "class": class_name, "packet_bucket": bucket,
                                        "samples": len(values),
                                        "baseline_e1_correct": base, "candidate_correct": new,
                                        "e1_only_correct": sum(v["pred_B0"] == v["true_class"] and
                                                               v["pred_candidate"] != v["true_class"] for v in values),
                                        "candidate_only_correct": sum(v["pred_B0"] != v["true_class"] and
                                                                     v["pred_candidate"] == v["true_class"] for v in values),
                                        "both_correct": sum(v["pred_B0"] == v["true_class"] and
                                                            v["pred_candidate"] == v["true_class"] for v in values),
                                        "both_wrong": sum(v["pred_B0"] != v["true_class"] and
                                                          v["pred_candidate"] != v["true_class"] for v in values)})
                    count_rows.append({"dataset": dataset, "seed": seed, "method": method,
                                       "class": class_name, "packet_bucket": bucket,
                                       "samples": len(values), "e1_accuracy": base / len(values),
                                       "candidate_accuracy": new / len(values),
                                       "delta_accuracy": (new - base) / len(values)})

    write_csv(OUT / "pilot_run_results.csv", result_rows, list(result_rows[0]))
    write_csv(OUT / "pilot_per_class.csv", per_class_rows, list(per_class_rows[0]))
    write_csv(OUT / "pilot_error_rescue.csv", rescue_rows, list(rescue_rows[0]))
    write_csv(OUT / "pilot_packet_count_results.csv", count_rows, list(count_rows[0]))
    summaries = []
    for dataset in DATASETS:
        for method in METHODS:
            rows = [row for row in result_rows if row["dataset"] == dataset and row["method"] == method]
            val_mean, val_std = mean_std([row["val_macro_f1"] for row in rows])
            delta_e1, _ = mean_std([row["delta_macro_vs_e1"] for row in rows])
            delta_e3, _ = mean_std([row["delta_macro_vs_e3"] for row in rows])
            summaries.append({"dataset": dataset, "method": method, "seeds": len(rows),
                              "val_macro_f1_mean": val_mean, "val_macro_f1_std_pop": val_std,
                              "delta_macro_vs_e1_mean": delta_e1,
                              "delta_macro_vs_e3_mean": delta_e3,
                              "positive_vs_e1": sum(row["delta_macro_vs_e1"] > 0 for row in rows),
                              "positive_vs_e3": sum(row["delta_macro_vs_e3"] > 0 for row in rows)})
    write_csv(OUT / "pilot_dataset_summary.csv", summaries, list(summaries[0]))

    # Pilot continuation is deliberately conservative: a candidate must beat
    # both frozen baselines on mean Macro-F1 in both datasets and on >=3/4 cells.
    # This is only a progression check; the formal five-seed gate is separate.
    candidates = []
    for method in ("S1", "G1", "G2"):
        rows = [row for row in result_rows if row["method"] == method]
        by_dataset = [next(s for s in summaries if s["method"] == method and s["dataset"] == d)
                      for d in DATASETS]
        passed = (all(s["delta_macro_vs_e1_mean"] > 0 and s["delta_macro_vs_e3_mean"] > 0
                      for s in by_dataset)
                  and sum(r["delta_macro_vs_e1"] > 0 and r["delta_macro_vs_e3"] > 0
                          for r in rows) >= 3)
        candidates.append({"method": method, "pilot_continuation_gate": passed,
                           "worst_dataset_delta_vs_e1": min(s["delta_macro_vs_e1_mean"] for s in by_dataset),
                           "worst_dataset_delta_vs_e3": min(s["delta_macro_vs_e3_mean"] for s in by_dataset)})
    viable = [r for r in candidates if r["pilot_continuation_gate"]]
    chosen = max(viable, key=lambda row: (min(row["worst_dataset_delta_vs_e1"],
                                          row["worst_dataset_delta_vs_e3"]),
                                          -("S1", "G1", "G2").index(row["method"]))) if viable else None
    decision = "PILOT_CANDIDATE_FOR_FIVE_SEED_CONFIRMATION" if chosen else "PILOT_NO_GLOBAL_GAIN"
    preflight = json.loads((OUT / "preflight.json").read_text())
    verification = {"status": "PASS", "pilot_runs_verified": len(result_rows),
                    "known_test_feature_usage": 0, "unknown_test_feature_usage": 0,
                    "protected_assets_checked": len(after), "protected_hashes_unchanged": True,
                    "pilot_gate": decision, "chosen_candidate": chosen["method"] if chosen else None,
                    "interflow_gate": preflight["interflow_stage20_gate"],
                    "formal_five_seed_gate_complete": False,
                    "claim_scope": "Known Validation pilot only; Stage20 Test exposed"}
    write_json(OUT / "completion_verification.json", verification)
    write_json(OUT / "pilot_candidate_decision.json", {"decision": decision,
                                                         "candidate_rows": candidates,
                                                         "chosen": chosen})
    lines = ["# Stage 24 — 24A pilot / 24B context-feasibility results", "",
             f"Status: `{decision}`. This is a Known Train/Validation pilot, not a Test or Unknown Detection result.",
             "", "## Frozen evidence and integrity", "",
             f"- Stage20/21/22 protected files: {len(after)}/{len(after)} before/after SHA256 unchanged.",
             "- 16/16 preregistered pilot run bundles passed independent artifact validation.",
             "- Known Test feature usage = 0; Unknown Test feature usage = 0.",
             "", "## Known Validation Macro-F1", "",
             "| Dataset | Candidate | Macro-F1 mean ± population std | Δ vs E1 | Δ vs E3 | Wins vs E1/E3 |",
             "| --- | --- | ---: | ---: | ---: | ---: |"]
    for row in summaries:
        lines.append(f"| {row['dataset']} | {row['method']} | {row['val_macro_f1_mean']:.6f} ± {row['val_macro_f1_std_pop']:.6f} | {row['delta_macro_vs_e1_mean']:+.6f} | {row['delta_macro_vs_e3_mean']:+.6f} | {row['positive_vs_e1']}/{row['positive_vs_e3']} of 2 |")
    lines.extend(["", "The 2022/2023 E1 checkpoints and representations are shared exactly within each paired cell.",
                  "S1 and G1 test early-eight behavior information; G2 uses packet-to-burst-to-flow hierarchy; G2-shuffle is a fixed-seed burst-order negative control.",
                  "", "## Short-flow and cross-flow context constraints", ""])
    for dataset in DATASETS:
        f = preflight["findings"][dataset]
        lines.append(f"- {dataset}: {f['one_or_two_packet_flows']}/{f['known_train_validation_flows']} Known Train/Val flows have 1–2 packets ({100*f['one_or_two_ratio']:.2f}%); {f['non_monotone_provenance_packet_timestamp_flows']} provenance flows have non-monotone packet timestamps.")
    lines.extend(["- Full-class capture-disjoint cross-flow graph: **NOT FEASIBLE on the frozen Stage20 VPN task**, because P2P has only one capture and current flow-random roles share captures.",
                  "- Frozen provenance lacks an endpoint/session identifier; causal previous-flow counts are only a read-only capture-time inventory, not an authorized edge set.",
                  "- No cross-flow GNN was trained. Inter-flow work requires a separate feasible context protocol, not a retrofit to this single-flow table.",
                  "", "## Decision", "",
                  f"- Pilot continuation gate: `{decision}`.",
                  f"- Chosen candidate for any five-seed confirmation: `{chosen['method'] if chosen else 'none'}`.",
                  "- A two-seed pilot cannot establish stable five-seed gains. Existing Stage20 Test was already exposed, so it is not untouched validation.",
                  "- No open-set score or Unknown-Free claim is produced; official TrafficFormer pretraining exposure remains unresolved.",
                  "", "## Evidence", "",
                  "- `pilot_run_results.csv`, `pilot_dataset_summary.csv`, `pilot_per_class.csv`, `pilot_error_rescue.csv`, `pilot_packet_count_results.csv`, individual run directories, protected hashes, and `completion_verification.json`.",
                  ""])
    (OUT / "stage24_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(verification, indent=2), flush=True)


if __name__ == "__main__":
    main()
