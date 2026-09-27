#!/usr/bin/env python3
"""Independently replay 8 adapted Test rows and compare paired Stage23 baselines."""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

OUT = Path(__file__).resolve().parents[1]
PROJECT = OUT.parent
MANIFEST = PROJECT / "stage20_dual_coarse_service_protocol/closed_service_manifest.csv"
BASE = PROJECT / "stage23_closed_set_method_table/stage23_full_run_results.csv"
FROZEN = "6404d496f50a6f7f6f397d125ad0cd653c9df152d8db1b9ed2ad2310967950bb"
BASE_FROZEN = "b10cbf5ccead40312768301c5379027ce983f4bcb8ec3cb31338f5b80ced0bdf"
METHODS = {"tfe": "TFE-GNN-8-UDP-short", "trident": "Trident-early8-86D"}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    if path.exists():
        raise FileExistsError(path)
    with path.open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    if digest(MANIFEST) != FROZEN:
        raise RuntimeError("Stage20 manifest changed")
    if digest(BASE) != BASE_FROZEN:
        raise RuntimeError("original Stage23 baseline results changed")
    original = read_csv(MANIFEST)
    baselines = {(r["dataset"], int(r["seed"]), r["method"]): r for r in read_csv(BASE)}
    rows, per_class, paired, subgroups, checks = [], [], [], [], 0
    for dataset in ("iscx_vpn", "iscx_tor"):
        input_dir = OUT / "inputs" / dataset / "test"
        input_ids = np.load(input_dir / "flow_ids.npy", allow_pickle=False)
        protocols = np.load(input_dir / "trident86.npy", mmap_mode="r", allow_pickle=False)[:, 0]
        packet_counts = np.load(input_dir / "packet_counts.npy", allow_pickle=False)
        if not (len(input_ids) == len(protocols) == len(packet_counts)):
            raise RuntimeError(f"Test input array lengths differ: {dataset}")
        input_meta = {str(uid): (int(proto), int(n_packet))
                      for uid, proto, n_packet in zip(input_ids, protocols, packet_counts)}
        if len(input_meta) != len(input_ids):
            raise RuntimeError(f"duplicate Test input flow_id: {dataset}")
        frozen_test = {r["flow_id"]: r for r in original
                       if r["dataset"] == dataset and r["closed_role"] == "known_test"}
        expected_train = sum(r["dataset"] == dataset and r["closed_role"] == "known_train" for r in original)
        expected_val = sum(r["dataset"] == dataset and r["closed_role"] == "known_validation" for r in original)
        expected_test = len(frozen_test)
        for seed in (2022, 2023):
            for key, method in METHODS.items():
                run = OUT / "runs" / key / dataset / f"seed{seed}"
                result = json.loads((run / "result.json").read_text(encoding="utf-8"))
                selection = json.loads((run / "selection.json").read_text(encoding="utf-8"))
                if not (run / "SUCCESS").is_file() or not (run / "SELECTION_COMPLETE").is_file():
                    raise RuntimeError(f"missing completion markers {run}")
                if selection["stage20_manifest_sha256"] != FROZEN or result["stage20_manifest_sha256"] != FROZEN:
                    raise RuntimeError(f"protocol hash mismatch {run}")
                if result["checkpoint_sha256"] != selection["checkpoint_sha256"]:
                    raise RuntimeError(f"checkpoint record mismatch {run}")
                cp = run / (selection["checkpoint_file"] if key == "tfe" else "selected_model.joblib")
                if digest(cp) != selection["checkpoint_sha256"]:
                    raise RuntimeError(f"checkpoint bytes changed {cp}")
                predictions = read_csv(run / "known_test_predictions.csv")
                actual_ids = [r["flow_id"] for r in predictions]
                if len(predictions) != expected_test or len(set(actual_ids)) != expected_test or set(actual_ids) != set(frozen_test):
                    raise RuntimeError(f"test flow membership mismatch {run}")
                if set(actual_ids) != set(input_meta):
                    raise RuntimeError(f"test input/prediction membership mismatch {run}")
                for r in predictions:
                    frozen = frozen_test[r["flow_id"]]
                    if r["true_service"] != frozen["service_label"] or r["role"] != "known_test":
                        raise RuntimeError(f"label/role mismatch {run}/{r['flow_id']}")
                    if r["method"] != method or int(r["seed"]) != seed or r["dataset"] != dataset:
                        raise RuntimeError(f"method/seed mismatch {run}")
                    if int(r["correct"]) != int(r["true_service"] == r["predicted_service"]):
                        raise RuntimeError(f"correct flag mismatch {run}")
                    checks += 1
                services = sorted({r["service_label"] for r in frozen_test.values()})
                truth = np.asarray([r["true_service"] for r in predictions])
                pred = np.asarray([r["predicted_service"] for r in predictions])
                score = {
                    "accuracy": float(accuracy_score(truth, pred)),
                    "macro_f1": float(f1_score(truth, pred, labels=services, average="macro", zero_division=0)),
                    "weighted_f1": float(f1_score(truth, pred, labels=services, average="weighted", zero_division=0)),
                }
                for metric, value in score.items():
                    if abs(result["test"][metric] - value) > 1e-10:
                        raise RuntimeError(f"metric replay mismatch {run}/{metric}")
                    checks += 1
                p, recall, f, n = precision_recall_fscore_support(truth, pred, labels=services, zero_division=0)
                for i, service in enumerate(services):
                    per_class.append({"dataset": dataset, "seed": seed, "method": method,
                                      "service": service, "precision": float(p[i]),
                                      "recall": float(recall[i]), "f1": float(f[i]), "support": int(n[i])})
                for subgroup_type, subgroup_values in (
                    ("protocol", {"TCP": {6}, "UDP": {17}}),
                    ("packet_count", {"1-2": {1, 2}, "3-4": {3, 4}, "5-8": {5, 6, 7, 8}}),
                ):
                    for subgroup, accepted in subgroup_values.items():
                        chosen = [i for i, uid in enumerate(actual_ids)
                                  if (input_meta[uid][0] if subgroup_type == "protocol"
                                      else input_meta[uid][1]) in accepted]
                        if not chosen:
                            continue
                        subgroups.append({"dataset": dataset, "seed": seed, "method": method,
                                          "subgroup_type": subgroup_type, "subgroup": subgroup,
                                          "test_flows": len(chosen),
                                          "accuracy": float(accuracy_score(truth[chosen], pred[chosen])),
                                          "macro_f1_fixed_classes": float(f1_score(
                                              truth[chosen], pred[chosen], labels=services,
                                              average="macro", zero_division=0))})
                rows.append({"dataset": dataset, "seed": seed, "method": method,
                             "claim_scope": "adapted-not-author-native", "classes": len(services),
                             "train_flows": expected_train, "validation_flows": expected_val,
                             "test_flows": expected_test, **score,
                             "best_epoch": selection.get("best_epoch", 10),
                             "checkpoint_sha256": selection["checkpoint_sha256"],
                             "stage20_manifest_sha256": FROZEN, "sample_parity": "PASS"})
                for baseline in ("TrafficFormer-pretrained", "OURS-E3-T8-pretrained"):
                    previous = baselines[dataset, seed, baseline]
                    paired.append({"dataset": dataset, "seed": seed, "candidate": method,
                                   "baseline": baseline, "delta_accuracy": score["accuracy"] - float(previous["accuracy"]),
                                   "delta_macro_f1": score["macro_f1"] - float(previous["macro_f1"]),
                                   "delta_weighted_f1": score["weighted_f1"] - float(previous["weighted_f1"])})
                checks += 5
    write_csv(OUT / "stage23b_run_results.csv", rows)
    write_csv(OUT / "stage23b_per_class.csv", per_class)
    write_csv(OUT / "stage23b_paired_comparison.csv", paired)
    write_csv(OUT / "stage23b_subgroup_analysis.csv", subgroups)
    summary = []
    for dataset in ("iscx_vpn", "iscx_tor"):
        for method in METHODS.values():
            group = [r for r in rows if r["dataset"] == dataset and r["method"] == method]
            summary.append({"dataset": dataset, "method": method, "seeds": "2022;2023",
                            **{metric + "_mean": float(np.mean([r[metric] for r in group]))
                               for metric in ("accuracy", "macro_f1", "weighted_f1")},
                            **{metric + "_std_population": float(np.std([r[metric] for r in group]))
                               for metric in ("accuracy", "macro_f1", "weighted_f1")}})
    write_csv(OUT / "stage23b_summary.csv", summary)
    verification = {"status": "PASS", "checks": checks, "runs": len(rows),
                    "frozen_manifest_sha256": digest(MANIFEST),
                    "original_stage23_sha256": digest(BASE),
                    "known_test_predictions": sum(r["test_flows"] for r in rows),
                    "unknown_test_used": False,
                    "original_stage23_modified": digest(BASE) != BASE_FROZEN}
    (OUT / "completion_verification.json").write_text(json.dumps(verification, indent=2) + "\n", encoding="utf-8")
    lines = [
        "\n## Terminal same-flow adapted results\n",
        "Status: COMPLETE / 8 of 8 adapted runs independently replayed. These are",
        "Stage20 same-flow **adaptations**, not native paper implementations.",
        "No Unknown Test was used; Known Test was materialized after all Known-Val selections.",
        "",
        "| Dataset | Method | Accuracy mean±std | Macro-F1 mean±std | Weighted-F1 mean±std |",
        "|---|---|---:|---:|---:|",
    ]
    for r in summary:
        lines.append(
            f"| {r['dataset']} | {r['method']} | "
            f"{r['accuracy_mean']:.6f} ± {r['accuracy_std_population']:.6f} | "
            f"{r['macro_f1_mean']:.6f} ± {r['macro_f1_std_population']:.6f} | "
            f"{r['weighted_f1_mean']:.6f} ± {r['weighted_f1_std_population']:.6f} |"
        )
    lines.extend([
        "",
        "Paired differences to the pre-existing TrafficFormer and E3 rows are in",
        "stage23b_paired_comparison.csv; all eight individual scores are retained",
        "in stage23b_run_results.csv, with per-class, TCP/UDP, packet-count",
        "subgroup results and predictions. Subgroup macro-F1 retains all frozen",
        "dataset services as the label universe.",
        "The original Stage23 five-method table was not modified.",
        f"Stage20 frozen manifest SHA256 after evaluation: {FROZEN}.",
        f"Independent checks: {checks}.",
    ])
    report_path = OUT / "RESULTS.md"
    report = report_path.read_text(encoding="utf-8")
    report = report.replace("- Status: `running`", "- Status: `complete`", 1)
    report = report.replace(
        "- No final Known Test metrics yet; do not compare these preliminary validation\n"
        "  values to the completed Stage23 Test table.",
        "- Final Known Test results follow below; validation values above were used only for selection.",
        1,
    )
    report_path.write_text(report + "\n" + "\n".join(lines) + "\n", encoding="utf-8")
    manifest_path = OUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "complete"
    manifest["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["core_results"] = summary
    manifest["execution"]["exit_code"] = 0
    manifest["limitations"] = [
        "TFE-GNN and Trident inputs are explicit eight-packet all-flow adaptations, not author-native results.",
        "Stage20 Service labels are capture-derived and split is not uniformly capture-disjoint.",
    ]
    manifest["next_step"] = "Scientific interpretation of paired adapted results; do not overwrite Stage23 original."
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (PROJECT / "EXPERIMENT_RESULTS.md").open("a", encoding="utf-8") as f:
        f.write("\nStage 23B COMPLETE: adapted TFE-GNN/Trident 8/8 same-Stage20-flow Test rows "
                "replayed; see stage23b_tfe_trident_same_flow_adaptation/RESULTS.md.\n")
    print(json.dumps(verification), flush=True)


if __name__ == "__main__":
    main()
