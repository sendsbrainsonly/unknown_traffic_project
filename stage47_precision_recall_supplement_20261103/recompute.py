#!/usr/bin/env python3
"""Supplement frozen Known-Test metrics; no model/data fitting or inference."""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
S47 = PROJECT / "stage47_original_service_current_three_view"
FROZEN = PROJECT / "stage20_dual_coarse_service_protocol/closed_service_manifest.csv"
METHODS = {
    "TrafficFormer-pretrained": "TrafficFormer（预训练）",
    "YaTC-official-pretrained-Stage20": "YaTC（预训练）",
    "Open-Detect corrected-paper": "Open-Detect（corrected-paper）",
    "RoNeTC-runnable-reconstruction": "RoNeTC（重建版）",
    "TFE-GNN-8-UDP-short": "TFE-GNN（8包、UDP/短流适配）",
    "Trident-early8-86D": "Trident（前8包、86维适配）",
    "OURS-E3-T8-pretrained": "本文早期 E3 双视图",
    "OURS-current-threeview-equal": "本文当前三视图等权融合",
}
DATASETS = {"iscx_vpn": ("ISCX-VPN（6类）", 8764, 1098, 1093),
            "iscx_tor": ("ISCXTor2016（7类）", 8946, 1118, 1117)}
METRICS = ("accuracy", "macro_precision", "macro_recall", "macro_f1",
           "weighted_precision", "weighted_recall", "weighted_f1",
           "micro_precision", "micro_recall", "micro_f1")
TOL = 1e-12


def now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(name, rows):
    with (ROOT / name).open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(name, value):
    (ROOT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source(dataset, seed, method):
    if method in ("TrafficFormer-pretrained", "OURS-E3-T8-pretrained"):
        path = PROJECT / f"stage22_pretrained_trafficformer_e3_closed_set_comparison/runs/{dataset}/seed{seed}/predictions.csv"
        return path, {"encoder": "E1" if method == "TrafficFormer-pretrained" else "E3"}
    routes = {
        "YaTC-official-pretrained-Stage20": f"stage23_closed_set_method_table/runs/yatc_stage20/{dataset}/seed{seed}_formal/test_predictions_cuda.csv",
        "Open-Detect corrected-paper": f"stage23_closed_set_method_table/runs/opendetect_corrected_paper/{dataset}/seed{seed}/predictions.csv",
        "RoNeTC-runnable-reconstruction": f"stage23_closed_set_method_table/runs/ronetc_stage20/{dataset}/seed{seed}_formal/predictions.csv",
        "TFE-GNN-8-UDP-short": f"stage23b_tfe_trident_same_flow_adaptation/runs/tfe/{dataset}/seed{seed}/known_test_predictions.csv",
        "Trident-early8-86D": f"stage23b_tfe_trident_same_flow_adaptation/runs/trident/{dataset}/seed{seed}/known_test_predictions.csv",
        "OURS-current-threeview-equal": f"stage47_original_service_current_three_view/runs/{dataset}/seed{seed}/known_test/predictions.csv",
    }
    return PROJECT / routes[method], {}


def expected_truth(dataset):
    rows = [r for r in read_csv(FROZEN) if r["dataset"] == dataset and r["closed_role"] == "known_test"]
    truth = {r["flow_id"]: r["service_label"] for r in rows}
    assert len(rows) == len(truth) == DATASETS[dataset][3], "frozen count/ID error"
    return truth


def load_predictions(dataset, seed, method, expected):
    path, filters = source(dataset, seed, method)
    rows = [r for r in read_csv(path) if r["role"] == "known_test"
            and all(r[k] == value for k, value in filters.items())]
    indexed = {r["flow_id"]: r for r in rows}
    assert len(rows) == len(indexed) == len(expected), f"duplicate/missing sample: {path}"
    assert indexed.keys() == expected.keys(), f"membership mismatch: {path}"
    classes = sorted(set(expected.values()))
    y, pred = [], []
    for uid in sorted(expected):
        row = indexed[uid]
        truth = row.get("true_service", row.get("true_class"))
        predicted = row.get("predicted_service", row.get("predicted_class"))
        assert truth == expected[uid] and predicted in classes, f"label mismatch: {path}"
        assert row["dataset"] == dataset and int(row["seed"]) == seed
        if "method" in row:
            assert row["method"] == method
        if "correct" in row:
            assert int(row["correct"]) == int(truth == predicted)
        y.append(truth)
        pred.append(predicted)
    return path, classes, y, pred


def from_confusion(cm):
    support, predicted, tp = cm.sum(1), cm.sum(0), cm.diagonal()
    precision = np.divide(tp, predicted, out=np.zeros(len(tp), dtype=float), where=predicted != 0)
    recall = np.divide(tp, support, out=np.zeros(len(tp), dtype=float), where=support != 0)
    f1 = np.divide(2 * tp, support + predicted, out=np.zeros(len(tp), dtype=float), where=support + predicted != 0)
    accuracy = float(tp.sum() / cm.sum())
    result = {"accuracy": accuracy}
    for metric, values in (("precision", precision), ("recall", recall), ("f1", f1)):
        result[f"macro_{metric}"] = float(values.mean())
        result[f"weighted_{metric}"] = float(np.average(values, weights=support))
        result[f"micro_{metric}"] = accuracy
    return result


def aggregate(rows):
    summaries = []
    for dataset in DATASETS:
        for method in METHODS:
            pair = [r for r in rows if r["dataset"] == dataset and r["method"] == method]
            assert sorted(int(r["seed"]) for r in pair) == [2022, 2023]
            summaries.append({"dataset": dataset, "method": method, "num_seeds": 2,
                **{f"{metric}_{kind}": float(op([float(r[metric]) for r in pair]))
                   for metric in METRICS for kind, op in (("mean", np.mean),
                       ("population_sd", lambda values: np.std(values, ddof=0)))}})
    return summaries


def report(summaries):
    lines = ["# ISCX 已知类分类：Precision / Recall 补充", "",
        "Status: success / PREDICTION_AND_METRIC_REPLAY_PASS", "",
        "Known Test；seeds 2022、2023；均值 ± 总体标准差（ddof=0）；单位 %。", "",
        "Macro 为每类等权平均；Weighted 为按真实类别支持数加权。单标签多分类下 Weighted Recall、Micro Precision/Recall/F1 均等于 Accuracy，已逐 run 验证。", "",
        "Macro-F1 是各类 F1 的均值，不是表中 Macro Precision 与 Macro Recall 的调和均值。", ""]
    columns = [("accuracy", "Accuracy"), ("macro_precision", "P-macro"),
               ("macro_recall", "R-macro"), ("macro_f1", "F1-macro"),
               ("weighted_precision", "P-weighted"), ("weighted_recall", "R-weighted"),
               ("weighted_f1", "F1-weighted")]
    for dataset, (name, train, val, test) in DATASETS.items():
        lines += ["## " + name, "", f"Train / Validation / Test = {train:,} / {val:,} / {test:,}。", "",
                  "| 方法 | " + " | ".join(label for _, label in columns) + " |",
                  "|---|" + "---:|" * len(columns)]
        for row in [r for r in summaries if r["dataset"] == dataset]:
            cells = [f"{row[key + '_mean'] * 100:.2f} ± {row[key + '_population_sd'] * 100:.2f}" for key, _ in columns]
            lines.append("| " + METHODS[row["method"]] + " | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["## 来源与限制", "",
        "从 Stage22、Stage23、Stage23B、Stage47 的原始逐样本预测补算；仅选择 Known Test 行。32/32 run 的 flow ID、真实标签、样本数均与 Stage20 冻结清单核对，Accuracy/Macro-F1/Weighted-F1 均须与原 Stage47 数值一致。", "",
        "固定样本、固定预测的事后指标补充；不训练、不推理、不选择模型或阈值。原标签、划分、预测和历史结果文件保持不变。VPN 为6类、Tor为7类，不能混入后来的4/5类粗任务结果。", "",
        "RoNeTC为重建版，TFE-GNN和Trident为所列适配版。两个seed和已暴露Test仅支持本开发协议内比较。"]
    return "\n".join(lines) + "\n"


def compute():
    original = read_csv(S47 / "all_methods_run_results.csv")
    assert len(original) == 32
    refs = {(r["dataset"], int(r["seed"]), r["method"]): r for r in original}
    sources = {FROZEN, S47 / "all_methods_run_results.csv", S47 / "method_summary.csv"}
    for dataset in DATASETS:
        for seed in (2022, 2023):
            sources.update(source(dataset, seed, method)[0] for method in METHODS)
    hashes = {str(p.relative_to(PROJECT)): sha(p) for p in sorted(sources)}
    write_json("source_hashes_before.json", hashes)
    all_rows, per_class, matrices, audits = [], [], [], []
    for dataset in DATASETS:
        expected = expected_truth(dataset)
        for method in METHODS:
            for seed in (2022, 2023):
                path, classes, truth, pred = load_predictions(dataset, seed, method, expected)
                identity = {"dataset": dataset, "seed": seed, "method": method}
                values = {"accuracy": float(accuracy_score(truth, pred))}
                for average in ("macro", "weighted", "micro"):
                    p, r, f, _ = precision_recall_fscore_support(truth, pred, labels=classes, average=average, zero_division=0)
                    values.update({f"{average}_precision": float(p), f"{average}_recall": float(r), f"{average}_f1": float(f)})
                cm = confusion_matrix(truth, pred, labels=classes)
                manual = from_confusion(cm)
                assert all(abs(values[k] - manual[k]) < TOL for k in METRICS), identity
                ref = refs[(dataset, seed, method)]
                assert all(abs(values[k] - float(ref[k])) < TOL for k in ("accuracy", "macro_f1", "weighted_f1")), identity
                all_rows.append({**identity, "test_flows": len(truth), **{k: values[k] for k in METRICS},
                    "source_predictions": str(path.relative_to(PROJECT)), "sample_parity": "PASS"})
                p, r, f, s = precision_recall_fscore_support(truth, pred, labels=classes, zero_division=0)
                per_class.extend({**identity, "class": c, "precision": float(p[i]), "recall": float(r[i]),
                    "f1": float(f[i]), "support": int(s[i])} for i, c in enumerate(classes))
                matrices.extend({**identity, "true_class": c, "predicted_class": d, "count": int(cm[i, j])}
                    for i, c in enumerate(classes) for j, d in enumerate(classes))
                audits.append({**identity, "source_path": str(path.relative_to(PROJECT)), "sha256": hashes[str(path.relative_to(PROJECT))],
                    "test_flows": len(truth), "class_count": len(classes), "membership_and_labels": "PASS",
                    "previous_accuracy_f1_replay": "PASS", "confusion_metric_replay": "PASS"})
    summaries = aggregate(all_rows)
    prior_summary = {(r["dataset"], r["method"]): r for r in read_csv(S47 / "method_summary.csv")}
    for row in summaries:
        prior = prior_summary[(row["dataset"], row["method"])]
        for metric in ("accuracy", "macro_f1", "weighted_f1"):
            for stat in ("mean", "population_sd"):
                assert abs(row[f"{metric}_{stat}"] - float(prior[f"{metric}_{stat}"])) < TOL
    assert hashes == {str(p.relative_to(PROJECT)): sha(p) for p in sorted(sources)}
    write_json("source_hashes_after.json", hashes)
    for name, rows in (("run_metrics.csv", all_rows), ("method_summary.csv", summaries),
                       ("per_class_metrics.csv", per_class), ("confusion_matrices.csv", matrices), ("source_audit.csv", audits)):
        write_csv(name, rows)
    (ROOT / "table_precision_recall.md").write_text(report(summaries), encoding="utf-8")
    manifest = json.loads((ROOT / "manifest.json").read_text())
    manifest.update({"inputs": [{"path": str(PROJECT / path), "sha256": value} for path, value in hashes.items()],
        "code": {"revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip(),
                 "dirty": True, "changes": ["new isolated posthoc metric supplement; original experiment sources read-only"]},
        "configuration": {"files": ["recompute.py"], "seeds": [2022, 2023], "parameters": {
            "role": "known_test", "averages": ["macro", "weighted", "micro"], "zero_division": 0,
            "std_ddof": 0, "replay_absolute_tolerance": TOL}},
        "execution": {"tmux_session": "iscx_pr_compute_20261103", "command": "python stage47_precision_recall_supplement_20261103/recompute.py compute",
            "exit_code": None, "environment": sys.prefix, "physical_gpu_ids": [],
            "log_path": str(PROJECT / ".tmux-task/iscx_pr_compute_20261103/output.log"),
            "exit_status_path": str(PROJECT / ".tmux-task/iscx_pr_compute_20261103/exit.status")},
        "core_results": summaries, "artifacts": [{"scope": "external", "role": "frozen-input", "path": str(Path("..") / path),
            "size_bytes": (PROJECT / path).stat().st_size, "sha256": value, "hash_status": "verified"} for path, value in hashes.items()],
        "limitations": ["two seeds; development Test previously exposed", "method adaptations preserved", "VPN6/Tor7 task"],
        "next_step": "independent replay of saved supplement and source hash verification"})
    write_json("manifest.json", manifest)
    print(json.dumps({"status": "COMPUTED", "runs": len(all_rows), "summaries": len(summaries), "per_class_rows": len(per_class), "source_files": len(hashes)}))


def verify():
    status_path = PROJECT / ".tmux-task/iscx_pr_compute_20261103/exit.status"
    assert status_path.read_text().strip() == "0", "computation exit status not successful"
    hashes = json.loads((ROOT / "source_hashes_before.json").read_text())
    assert hashes == json.loads((ROOT / "source_hashes_after.json").read_text())
    assert all(sha(PROJECT / path) == value for path, value in hashes.items())
    rows = read_csv(ROOT / "run_metrics.csv")
    assert len(rows) == len({(r["dataset"], r["seed"], r["method"]) for r in rows}) == 32
    for row in rows:
        dataset, seed, method = row["dataset"], int(row["seed"]), row["method"]
        _, classes, truth, pred = load_predictions(dataset, seed, method, expected_truth(dataset))
        matrix = np.zeros((len(classes), len(classes)), dtype=np.int64)
        for y, p in zip(truth, pred, strict=True):
            matrix[classes.index(y), classes.index(p)] += 1
        manual = from_confusion(matrix)
        assert all(abs(manual[key] - float(row[key])) < TOL for key in METRICS)
    summaries = aggregate(rows)
    saved = read_csv(ROOT / "method_summary.csv")
    for row, ref in zip(summaries, saved, strict=True):
        assert row["dataset"] == ref["dataset"] and row["method"] == ref["method"]
        assert all(abs(row[key] - float(ref[key])) < TOL for key in row if key not in ("dataset", "method", "num_seeds"))
    assert report(summaries) == (ROOT / "table_precision_recall.md").read_text(encoding="utf-8")
    verification = {"status": "PASS", "verified_at_utc": now(), "runs": 32, "summaries": 16,
        "per_class_rows": len(read_csv(ROOT / "per_class_metrics.csv")), "source_files_unchanged": len(hashes),
        "all_96_original_run_metrics_reproduced": True, "all_96_original_summary_cells_reproduced": True,
        "independent_confusion_metric_replay": True, "weighted_recall_equals_accuracy": True,
        "training_used": False, "gpu_used": False, "unknown_used": False, "test_parameter_selection": False}
    write_json("completion_verification.json", verification)
    manifest = json.loads((ROOT / "manifest.json").read_text())
    absolute_refs = [item for item in manifest["artifacts"]
                     if item.get("scope") == "external" and Path(item["path"]).is_absolute()]
    if absolute_refs:
        with (ROOT / "archive_failure_manifest.json").open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        for item in absolute_refs:
            item["path"] = str(Path("..") / Path(item["path"]).relative_to(PROJECT))
        manifest["execution"]["archive_schema_correction"] = {
            "failed_session": "iscx_pr_archive_20261103", "exit_code": 1,
            "reason": "validator requires relative artifact paths even for external references",
            "repair": "metadata only: absolute input references changed to equivalent relative references",
            "metrics_changed": False}
    manifest.update({"status": "success", "updated_at_utc": now(), "next_step": "none; metric supplement complete"})
    manifest["execution"]["exit_code"] = 0
    manifest["execution"]["verification_tmux_session"] = "iscx_pr_replay_20261103"
    write_json("manifest.json", manifest)
    text = "\n".join(["# Experiment results: stage47-iscx-known-test-precision-recall-20261103", "",
        "- Status: `success` / `PREDICTION_AND_METRIC_REPLAY_PASS`", "- Experiment type: `posthoc-metric-supplement`",
        "- Claim scope: `diagnostic`", "- Objective: Supplement frozen ISCX-VPN/Tor closed-set Precision and Recall.", "",
        "## Data and split", "", "VPN6: 8,764 / 1,098 / 1,093; Tor7: 8,946 / 1,118 / 1,117 (Train/Val/Test). Seeds2022/2023; unchanged Stage20 membership.", "",
        "## Configuration and execution", "", "CPU metric recomputation from existing Known-Test predictions; no model execution, fitting, threshold or checkpoint selection. Fixed workspace Conda, tmux. Macro/Weighted/Micro; zero_division=0; population SD ddof=0.", "",
        "## Core results", "", "32/32 runs, 16 summaries, 208 per-class rows. All original 96 run-level Accuracy/F1 values and 96 summary cells reproduced within 1e-12. See [full comparison table](table_precision_recall.md).", "",
        "## Preserved evidence", "", "recompute.py; run_metrics.csv; method_summary.csv; per_class_metrics.csv; confusion_matrices.csv; source_audit.csv; source_hashes_before/after.json; completion_verification.json; manifest.json; project-local tmux logs referenced by manifest. Original source files retained in place and SHA256 registered.", "",
        "## Limitations", "", "Two-seed development evaluation with exposed Test; baseline reconstruction/adaptation identities preserved. VPN6/Tor7 cannot be substituted for later coarse4/5 tasks. Averaged Macro-F1 is not the harmonic mean of averaged Precision/Recall.", "",
        "## Conclusion and next step", "", "Precision/Recall supplemental computation complete; no new training required. No further experiment started.", ""])
    (ROOT / "RESULTS.md").write_text(text, encoding="utf-8")
    print(json.dumps(verification))


if __name__ == "__main__":
    assert sys.prefix == "/home/birkenwald/data/SEU-WXY/conda_envs/2025-10-8-WXY-dgl_py310", sys.prefix
    {"compute": compute, "verify": verify}[sys.argv[1]]()
