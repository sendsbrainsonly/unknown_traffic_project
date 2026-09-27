#!/usr/bin/env python3
"""Independent saved-logit replay and Stage32 three-dataset diagnostic report."""
from __future__ import annotations

import csv
import json

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from common import COARSE_CLASSES, DATASETS, ROOT, freeze_sources, sha256, write_json


def csv_rows(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_csv(path, rows):
    if not rows or path.exists():
        raise RuntimeError(f"empty/existing CSV: {path}")
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if (ROOT / "completion_verification.json").exists():
        raise FileExistsError("Stage32 independent replay already exists")
    rows, comparisons, replay = [], [], []
    selected = json.loads((ROOT / "selected_heads_before_test.json").read_text())
    if freeze_sources() != selected["source_hashes"]:
        raise RuntimeError("frozen input changed before independent replay")
    for dataset in DATASETS:
        result = ROOT / "runs" / dataset / "known_test_evaluation"
        if not (result / "SUCCESS").is_file():
            raise RuntimeError(f"missing complete Stage32 Test evidence: {dataset}")
        report = json.loads((result / "results.json").read_text())
        prediction = csv_rows(result / "sample_predictions.csv")
        logits = np.load(result / "logits.npy", allow_pickle=False)
        classes = COARSE_CLASSES[dataset]
        if len(prediction) != report["known_test_samples"] or logits.shape != (len(prediction), len(classes)):
            raise RuntimeError(f"saved Test shape mismatch: {dataset}")
        truth = np.asarray([classes.index(r["true_coarse"]) for r in prediction], dtype=np.int64)
        inferred = logits.argmax(1)
        if any(classes[int(i)] != r["pred_coarse"] or int(r["correct"]) != int(i == y)
               for i, y, r in zip(inferred, truth, prediction, strict=True)):
            raise RuntimeError(f"saved-logit decision replay mismatch: {dataset}")
        got = {"accuracy": float(accuracy_score(truth, inferred)),
               "macro_f1": float(f1_score(truth, inferred, labels=range(len(classes)), average="macro", zero_division=0)),
               "weighted_f1": float(f1_score(truth, inferred, labels=range(len(classes)), average="weighted", zero_division=0))}
        if any(abs(got[k] - report["metrics"][k]) > 1e-12 for k in got):
            raise RuntimeError(f"reported metric replay mismatch: {dataset}")
        p, r, f, support = precision_recall_fscore_support(truth, inferred, labels=range(len(classes)), zero_division=0)
        if any(abs(float(f[i]) - report["per_class"][i]["f1"]) > 1e-12 or
               int(support[i]) != report["per_class"][i]["support"] for i in range(len(classes))):
            raise RuntimeError(f"reported per-class replay mismatch: {dataset}")
        ckpt = json.loads((ROOT / "runs" / dataset / "known_validation_metrics.json").read_text())["checkpoint_hashes"]
        if ckpt != selected["head_hashes"][dataset] or any(
            sha256(ROOT / "runs" / dataset / name) != digest for name, digest in ckpt.items()):
            raise RuntimeError(f"checkpoint hash changed: {dataset}")
        val = json.loads((ROOT / "runs" / dataset / "known_validation_metrics.json").read_text())
        rows.append({"dataset": dataset, "coarse_classes": len(classes),
                     "train_seed": 2022, "known_validation_macro_f1": val["metrics"]["macro_f1"],
                     "known_test_samples": len(prediction), **{f"known_test_{k}": v for k, v in got.items()},
                     "claim_scope": "previously_exposed_test_development"})
        comparisons.extend(csv_rows(result / "paired_comparison.csv"))
        replay.append({"dataset": dataset, "saved_logits_replayed": len(prediction),
                       "metrics_replayed": 3, "per_class_rows_replayed": len(classes),
                       "checkpoint_hashes_unchanged": True})
    save_csv(ROOT / "stage32_test_results.csv", rows)
    save_csv(ROOT / "stage32_paired_comparison.csv", comparisons)
    source_after = freeze_sources()
    write_json(ROOT / "frozen_source_hashes_after.json", source_after)
    if source_after != selected["source_hashes"]:
        raise RuntimeError("frozen source changed after Test")
    write_json(ROOT / "completion_verification.json", {"status": "PASS",
        "test_units": len(rows), "saved_logit_replay": replay,
        "frozen_source_hashes_unchanged": True, "unknown_usage": 0,
        "test_threshold_or_model_selection": 0,
        "vnat_test_baseline_available": False,
        "limitations": ["single training seed", "previously exposed Test",
            "VNAT medium_seed2025 uses flow-random split with possible capture leakage",
            "VNAT historical F2 pairing is Known Validation only and uses a different training seed"]})
    lines = ["# Stage 32 — 三数据集粗粒度三路等权闭集实验", "",
             "Status: completed diagnostic; all scores are on the frozen **Known-only closed-set** task.", "",
             "## 单 seed 结果", "",
             "| 数据集 | 类别数 | Known Test 流数 | Val Macro-F1 | Test Accuracy | Test Macro-F1 | Test Weighted-F1 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['dataset']} | {row['coarse_classes']} | {row['known_test_samples']} | "
            f"{row['known_validation_macro_f1']:.6f} | {row['known_test_accuracy']:.6f} | "
            f"{row['known_test_macro_f1']:.6f} | {row['known_test_weighted_f1']:.6f} |")
    lines += ["", "## 同流、同粗标签配对", "",
              "| 数据集 | 评价集 | 历史对照 | 历史 Macro-F1 | 新 T0 Macro-F1 | ΔMacro-F1 | 95% bootstrap CI |",
              "|---|---|---|---:|---:|---:|---:|"]
    for r in comparisons:
        lines.append(f"| {r['dataset']} | {r['role']} | {r['baseline']} | "
            f"{float(r['baseline_macro_f1']):.6f} | {float(r['new_macro_f1']):.6f} | "
            f"{float(r['delta_macro_f1']):+.6f} | [{float(r['bootstrap_delta_macro_f1_low']):+.6f}, "
            f"{float(r['bootstrap_delta_macro_f1_high']):+.6f}] |")
    lines += ["", "## 解释边界", "",
              "- ISCX 两项 Test 对照是 seed 2022 的 exact-flow 历史结果；VNAT 历史 F2 仅与 Known Validation 配对，且其训练 seed 为 2025。VNAT Test 没有经同流冻结重放验证的旧 F2 对照，不能声称 Test 相对 F2 的增益。",
              "- 粗标签抹平了部分原细类错误；高分不是原细粒度任务性能提升，也不是开集性能。三路原 fine encoder 与 fine head 保留。",
              "- VNAT 按 flow 随机划分，同 capture 可能同时在 Train/Val/Test，且标签主要来自 capture；特别高的分数不证明跨 capture 或跨环境泛化。",
              "- 三个 Test 曾在历史项目中暴露，因此是 development evidence；仅一个训练 seed，不能宣称稳定多 seed 优势。",
              "- 所有对照和主模型均未使用 Unknown 流拟合；Test 未参与 checkpoint、阈值或参数选择。",
              "", "完整逐流预测、逐类指标、训练曲线和 checkpoint 位于 `runs/`；独立重放见 `completion_verification.json`。", ""]
    (ROOT / "stage32_report.md").write_text("\n".join(lines))
    print(json.dumps({"status": "PASS", "datasets": len(rows), "test_results": rows,
                      "comparisons": [{"dataset": r["dataset"], "baseline": r["baseline"],
                                       "delta_macro_f1": r["delta_macro_f1"]} for r in comparisons]}), flush=True)


if __name__ == "__main__":
    main()
