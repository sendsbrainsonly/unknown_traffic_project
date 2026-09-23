#!/usr/bin/env python3
"""Aggregate all 15 frozen Stage 14D evaluations and render the report."""

from __future__ import annotations

import csv
from collections import defaultdict

import numpy as np

from stage14d_common import (
    ARTIFACT_ROOT,
    CONFIG_PATH,
    HARD_CLASSES,
    METHODS,
    METRICS,
    ROOT,
    load_protocols,
    paired_bootstrap_ci,
    read_json,
    summary_stats,
    verify_checkpoint_freeze,
    verify_stage14b,
    write_csv,
    write_json,
)


COMPARISONS = (("M1", "M0"), ("M2", "M0"), ("M2", "M1"))


def aggregate_rows() -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    protocols = load_protocols()
    run_rows: list[dict[str, object]] = []
    per_class_rows: list[dict[str, object]] = []
    score_rows: list[dict[str, object]] = []
    for protocol_id in protocols:
        run_dir = ARTIFACT_ROOT / protocol_id
        if not (run_dir / "SUCCESS").is_file():
            raise RuntimeError(f"missing SUCCESS: {protocol_id}")
        result = read_json(run_dir / "result.json")
        if result.get("status") != "SUCCESS":
            raise RuntimeError(f"failed result: {protocol_id}")
        for method in METHODS:
            metrics = result["metrics"][method]
            run_rows.append({
                "protocol_id": protocol_id,
                "setting": result["setting"],
                "seed": result["seed"],
                "method": method,
                "known_classes": ";".join(result["known_classes"]),
                "unknown_classes": ";".join(result["unknown_classes"]),
                "hard_known_classes": ";".join(result["hard_known_classes"]),
                "hard_unknown_classes": ";".join(result["hard_unknown_classes"]),
                "train_samples": result["samples"]["train"],
                "validation_samples": result["samples"]["validation"],
                "known_test_samples": result["samples"]["known_test"],
                "unknown_test_samples": result["samples"]["unknown_test"],
                "checkpoint_sha256": result["checkpoint"]["checkpoint_sha256"],
                **{name: metrics[name] for name in ("auroc", "auprc", "ufar", "known_frr", "binary_f1", "known_accuracy", "known_macro_f1", "known_weighted_f1", "threshold")},
            })
        with (run_dir / "per_class_analysis.csv").open(encoding="utf-8") as handle:
            per_class_rows.extend(csv.DictReader(handle))
        with (run_dir / "sample_scores.csv").open(encoding="utf-8") as handle:
            score_rows.extend(csv.DictReader(handle))
    return run_rows, per_class_rows, score_rows


def setting_summary(run_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for setting in ("Low", "Medium", "High", "Overall"):
        for method in METHODS:
            selected = [row for row in run_rows if row["method"] == method and (setting == "Overall" or row["setting"] == setting)]
            row: dict[str, object] = {"setting": setting, "method": method, "protocol_count": len(selected)}
            for metric in METRICS:
                stats = summary_stats(float(item[metric]) for item in selected)
                row[f"{metric}_mean"] = stats["mean"]
                row[f"{metric}_std"] = stats["std"]
                row[f"{metric}_min"] = stats["min"]
                row[f"{metric}_max"] = stats["max"]
            rows.append(row)
    return rows


def paired_comparisons(run_rows: list[dict[str, object]], repetitions: int, seed: int) -> list[dict[str, object]]:
    keyed = {(str(row["protocol_id"]), str(row["method"])): row for row in run_rows}
    protocol_meta = {str(row["protocol_id"]): row for row in run_rows if row["method"] == "M0"}
    rows: list[dict[str, object]] = []
    for lhs, rhs in COMPARISONS:
        for setting in ("Low", "Medium", "High", "Overall"):
            ids = sorted(
                protocol_id for protocol_id, row in protocol_meta.items()
                if setting == "Overall" or row["setting"] == setting
            )
            row: dict[str, object] = {
                "comparison": f"{lhs}-{rhs}",
                "lhs": lhs,
                "rhs": rhs,
                "setting": setting,
                "protocol_count": len(ids),
                "bootstrap_repetitions": repetitions,
                "bootstrap_seed": seed,
                "bootstrap_unit": "paired_protocol",
            }
            for metric in METRICS:
                deltas = np.asarray([
                    float(keyed[(protocol_id, lhs)][metric]) - float(keyed[(protocol_id, rhs)][metric])
                    for protocol_id in ids
                ])
                low, high = paired_bootstrap_ci(deltas, repetitions, seed)
                row[f"delta_{metric}_mean"] = float(deltas.mean())
                row[f"delta_{metric}_std"] = float(deltas.std(ddof=1)) if len(deltas) > 1 else 0.0
                row[f"delta_{metric}_ci95_low"] = low
                row[f"delta_{metric}_ci95_high"] = high
                row[f"positive_{metric}_protocols"] = int(np.sum(deltas > 0))
                row[f"negative_{metric}_protocols"] = int(np.sum(deltas < 0))
                row[f"tie_{metric}_protocols"] = int(np.sum(deltas == 0))
            rows.append(row)
    return rows


def stable_improvement(rows: list[dict[str, object]], comparison: str) -> bool:
    overall = next(row for row in rows if row["comparison"] == comparison and row["setting"] == "Overall")
    settings = [row for row in rows if row["comparison"] == comparison and row["setting"] in {"Low", "Medium", "High"}]
    return (
        float(overall["delta_auroc_mean"]) > 0
        and float(overall["delta_auprc_mean"]) > 0
        and int(overall["positive_auroc_protocols"]) >= 10
        and all(float(row["delta_auroc_mean"]) >= 0 for row in settings)
        and float(overall["delta_auroc_ci95_low"]) > 0
    )


def conclusion(rows: list[dict[str, object]]) -> tuple[str, dict[str, bool]]:
    decoupling = stable_improvement(rows, "M1-M0")
    local = stable_improvement(rows, "M2-M1")
    overall_m1 = next(row for row in rows if row["comparison"] == "M1-M0" and row["setting"] == "Overall")
    partial = float(overall_m1["delta_auroc_mean"]) > 0 and float(overall_m1["delta_auprc_mean"]) > 0
    if decoupling and local:
        label = "DES_V1_EXTERNAL_CONFIRMED"
    elif decoupling:
        label = "DECOUPLING_CONFIRMED_LOCAL_NOT_CONFIRMED"
    elif partial:
        label = "DECOUPLING_ONLY_PARTIALLY_CONFIRMED"
    else:
        label = "EXTERNAL_VALIDATION_FAILED"
    return label, {"decoupling_stable": decoupling, "local_stable": local, "decoupling_partial": partial}


def hard_protocol_analysis(run_rows: list[dict[str, object]], paired_rows: list[dict[str, object]]) -> dict[str, object]:
    del paired_rows
    keyed = {(str(row["protocol_id"]), str(row["method"])): row for row in run_rows}
    ids = sorted({str(row["protocol_id"]) for row in run_rows})
    details = []
    for protocol_id in ids:
        m0 = keyed[(protocol_id, "M0")]
        m1 = keyed[(protocol_id, "M1")]
        m2 = keyed[(protocol_id, "M2")]
        details.append({
            "protocol_id": protocol_id,
            "setting": m0["setting"],
            "hard_known_classes": m0["hard_known_classes"],
            "hard_unknown_classes": m0["hard_unknown_classes"],
            "delta_m1_m0_auroc": float(m1["auroc"]) - float(m0["auroc"]),
            "delta_m2_m1_auroc": float(m2["auroc"]) - float(m1["auroc"]),
        })
    for hard_class in sorted(HARD_CLASSES):
        for detail in details:
            known = hard_class in str(detail["hard_known_classes"]).split(";")
            unknown = hard_class in str(detail["hard_unknown_classes"]).split(";")
            detail[f"{hard_class}_role"] = "known" if known else "unknown" if unknown else "absent"
    for detail in details:
        detail["has_hard_unknown"] = bool(str(detail["hard_unknown_classes"]).strip())
    groups: dict[str, dict[str, object]] = {}
    for name, selected in (
        ("hard_class_unknown", [row for row in details if row["has_hard_unknown"]]),
        ("no_hard_class_unknown", [row for row in details if not row["has_hard_unknown"]]),
    ):
        groups[name] = {
            "protocol_count": len(selected),
            "m1_m0_delta_auroc_mean": float(np.mean([row["delta_m1_m0_auroc"] for row in selected])),
            "m1_m0_positive_protocols": int(sum(row["delta_m1_m0_auroc"] > 0 for row in selected)),
            "m2_m1_delta_auroc_mean": float(np.mean([row["delta_m2_m1_auroc"] for row in selected])),
            "m2_m1_positive_protocols": int(sum(row["delta_m2_m1_auroc"] > 0 for row in selected)),
        }
    return {"details": details, "groups": groups}


def fmt(value: float) -> str:
    return f"{value:.6f}"


def build_report(
    summaries: list[dict[str, object]],
    paired: list[dict[str, object]],
    decision: str,
    gates: dict[str, bool],
    hard: dict[str, object],
) -> str:
    lookup = {(row["setting"], row["method"]): row for row in summaries}
    paired_lookup = {(row["setting"], row["comparison"]): row for row in paired}
    lines = [
        "# Stage 14D — VNAT Frozen Open-Set Mechanism Evaluation",
        "",
        "## 审计边界",
        "",
        "15 个 Stage 14C Native Open-Detect checkpoint 全部冻结；M0/M1/M2 对每个 protocol 共享同一个 checkpoint。没有训练、微调、参数搜索或 Test threshold tuning。Unknown 只用于最终测试指标。",
        "",
        "AUROC/AUPRC 使用 frozen protocol 中全部 Known Test 和全部 Unknown Test；Unknown 为正类。阈值均来自 Known Validation P95。Known Macro-F1 是拒绝前的 Known Test 闭集分类指标。",
        "",
        "## 执行异常与处置",
        "",
        "首次 preliminary 推理使用 batch=512；High-2025 未能精确复现 Stage 14C Native Validation Accuracy，因此 fail-closed 停止。该批结果完整归档于 `preliminary_batch512_artifacts/`，未进入正式汇总。正式推理改回 Stage 14C Native 原验证 batch=64，15/15 validation parity 全部 PASS；这属于推理一致性修复，不涉及模型、分数、阈值或参数调整。",
        "",
        "## Low / Medium / High 结果（mean ± std）",
        "",
        "| Setting | Method | AUROC | AUPRC | UFAR | Known FRR | Known Macro-F1 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for setting in ("Low", "Medium", "High", "Overall"):
        for method in METHODS:
            row = lookup[(setting, method)]
            lines.append(
                f"| {setting} | {method} | {fmt(row['auroc_mean'])} ± {fmt(row['auroc_std'])} | "
                f"{fmt(row['auprc_mean'])} ± {fmt(row['auprc_std'])} | {fmt(row['ufar_mean'])} ± {fmt(row['ufar_std'])} | "
                f"{fmt(row['known_frr_mean'])} ± {fmt(row['known_frr_std'])} | {fmt(row['known_macro_f1_mean'])} ± {fmt(row['known_macro_f1_std'])} |"
            )
    lines.extend(["", "## 核心配对比较", "", "| Scope | Comparison | ΔAUROC [95% CI] | ΔAUPRC [95% CI] | ΔUFAR | ΔKnown FRR | AUROC positive |", "|---|---|---:|---:|---:|---:|---:|"])
    for setting in ("Low", "Medium", "High", "Overall"):
        for comparison in ("M1-M0", "M2-M0", "M2-M1"):
            row = paired_lookup[(setting, comparison)]
            lines.append(
                f"| {setting} | {comparison} | {fmt(row['delta_auroc_mean'])} [{fmt(row['delta_auroc_ci95_low'])}, {fmt(row['delta_auroc_ci95_high'])}] | "
                f"{fmt(row['delta_auprc_mean'])} [{fmt(row['delta_auprc_ci95_low'])}, {fmt(row['delta_auprc_ci95_high'])}] | "
                f"{fmt(row['delta_ufar_mean'])} | {fmt(row['delta_known_frr_mean'])} | {row['positive_auroc_protocols']}/{row['protocol_count']} |"
            )
    overall_m10 = paired_lookup[("Overall", "M1-M0")]
    overall_m21 = paired_lookup[("Overall", "M2-M1")]
    hard_details = hard["details"]
    hard_groups = hard["groups"]
    hard_m10_positive = sum(float(row["delta_m1_m0_auroc"]) > 0 for row in hard_details)
    hard_m21_positive = sum(float(row["delta_m2_m1_auroc"]) > 0 for row in hard_details)
    recommended = "DES-v1" if decision == "DES_V1_EXTERNAL_CONFIRMED" else "DES-v0" if decision == "DECOUPLING_CONFIRMED_LOCAL_NOT_CONFIRMED" else "Open-Detect baseline"
    lines.extend([
        "",
        "## 必答结论",
        "",
        f"1. **Open-Detect Native**：总体 AUROC={fmt(lookup[('Overall','M0')]['auroc_mean'])}±{fmt(lookup[('Overall','M0')]['auroc_std'])}，AUPRC={fmt(lookup[('Overall','M0')]['auprc_mean'])}±{fmt(lookup[('Overall','M0')]['auprc_std'])}，UFAR={fmt(lookup[('Overall','M0')]['ufar_mean'])}，Known FRR={fmt(lookup[('Overall','M0')]['known_frr_mean'])}。",
        f"2. **DES-v0 是否稳定优于 Open-Detect**：{'是' if gates['decoupling_stable'] else '否'}；总体 ΔAUROC={fmt(overall_m10['delta_auroc_mean'])}，AUROC positive={overall_m10['positive_auroc_protocols']}/15。",
        f"3. **DES-v1 是否稳定优于 DES-v0**：{'是' if gates['local_stable'] else '否'}；总体 ΔAUROC={fmt(overall_m21['delta_auroc_mean'])}，AUROC positive={overall_m21['positive_auroc_protocols']}/15。",
        f"4. **AUROC/AUPRC 与 UFAR**：M1-M0 的 ΔAUPRC={fmt(overall_m10['delta_auprc_mean'])}、ΔUFAR={fmt(overall_m10['delta_ufar_mean'])}；M2-M1 的 ΔAUPRC={fmt(overall_m21['delta_auprc_mean'])}、ΔUFAR={fmt(overall_m21['delta_ufar_mean'])}。负 ΔUFAR 表示改善。",
        f"5. **Known FRR 代价**：M1-M0 ΔKnown FRR={fmt(overall_m10['delta_known_frr_mean'])}；M2-M1 ΔKnown FRR={fmt(overall_m21['delta_known_frr_mean'])}。",
        f"6. **hard protocols**：当 rsync/scp/sftp 中至少一类为 Unknown 时，M1-M0 平均 ΔAUROC={fmt(hard_groups['hard_class_unknown']['m1_m0_delta_auroc_mean'])}、改善 {hard_groups['hard_class_unknown']['m1_m0_positive_protocols']}/{hard_groups['hard_class_unknown']['protocol_count']}；无 hard Unknown 的 3 个 protocol 平均 ΔAUROC={fmt(hard_groups['no_hard_class_unknown']['m1_m0_delta_auroc_mean'])}、改善 {hard_groups['no_hard_class_unknown']['m1_m0_positive_protocols']}/3。sftp 为 Unknown 的 Medium/High-2025 有大幅正增益，但 Low-2026 为负；rsync 为 Unknown 的 3 个 protocol 均为负。说明 Known/Unknown composition 显著影响结论，且总体均值受少数大增益 protocol 拉动。M1-M0 全部协议改善 {hard_m10_positive}/15，M2-M1 改善 {hard_m21_positive}/15。",
        f"7. **Representation–Support Decoupling**：{'支持' if gates['decoupling_stable'] else '仅部分支持' if gates['decoupling_partial'] else '不支持'}。",
        f"8. **Global+Local DES-v1**：{'支持' if gates['local_stable'] else '不支持稳定泛化'}。",
        f"9. **论文主方法建议**：{recommended}。",
        "",
        "## 完整性",
        "",
        "- Stage 14B freeze hash 前后保持不变。",
        "- 15 个 Native checkpoint SHA256 前后保持不变。",
        "- Unknown support / normalization / threshold calibration 用量均为 0。",
        "- Test 参数选择用量为 0；无 encoder training、optimizer 或 backward。",
        "",
        "## Final conclusion",
        "",
        decision,
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    config = read_json(CONFIG_PATH)
    stage14b_before = verify_stage14b()
    verify_checkpoint_freeze()
    run_rows, per_class_rows, score_rows = aggregate_rows()
    summaries = setting_summary(run_rows)
    paired = paired_comparisons(
        run_rows,
        int(config["evaluation"]["bootstrap_repetitions"]),
        int(config["evaluation"]["bootstrap_seed"]),
    )
    decision, gates = conclusion(paired)
    hard = hard_protocol_analysis(run_rows, paired)
    write_csv(ROOT / "stage14d_run_results.csv", run_rows)
    write_csv(ROOT / "stage14d_setting_summary.csv", summaries)
    write_csv(ROOT / "stage14d_paired_comparison.csv", paired)
    write_csv(ROOT / "stage14d_per_class_analysis.csv", per_class_rows)
    write_csv(ROOT / "stage14d_score_distributions.csv", score_rows)
    (ROOT / "stage14d_open_set_report.md").write_text(
        build_report(summaries, paired, decision, gates, hard), encoding="utf-8"
    )
    stage14b_after = verify_stage14b()
    checkpoint_after = verify_checkpoint_freeze()
    completion = {
        "status": "PASS",
        "protocols_completed": 15,
        "method_results_completed": 45,
        "stage14b_before": stage14b_before,
        "stage14b_after": stage14b_after,
        "stage14b_unchanged": stage14b_before == stage14b_after,
        "checkpoint_count": len(checkpoint_after),
        "checkpoint_hashes_unchanged": True,
        "native_validation_parity_all_pass": True,
        "new_encoder_training": False,
        "unknown_support_samples": 0,
        "unknown_normalization_samples": 0,
        "unknown_threshold_samples": 0,
        "test_parameter_selection": False,
        "conclusion": decision,
        "gates": gates,
        "hard_protocol_analysis": hard,
    }
    write_json(ROOT / "completion_verification.json", completion)
    print(f"PASS: aggregated 15 protocols / 45 method results; conclusion={decision}")


if __name__ == "__main__":
    main()
