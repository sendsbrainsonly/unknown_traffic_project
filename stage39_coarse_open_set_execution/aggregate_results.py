"""Aggregate all predeclared Stage39 settings without selecting a best fold."""
from __future__ import annotations

import csv
import json
from collections import defaultdict

import numpy as np

from freeze_protocols import ROOT, sha, write_json


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    if not rows:
        raise RuntimeError(f"empty aggregate table: {path}")
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def num(row, key):
    return float(row[key])


def render_table(rows, keys):
    header = "| " + " | ".join(keys) + " |"
    bar = "|" + "|".join("---" for _ in keys) + "|"
    body = ["| " + " | ".join(str(row.get(key, "")) for key in keys) + " |" for row in rows]
    return "\n".join([header, bar, *body])


def aggregate():
    if (ROOT / "stage39_run_results.csv").exists():
        raise FileExistsError("Stage39 aggregate already exists; preserve prior output")
    queue = json.loads((ROOT / "evaluation_queue_progress.json").read_text(encoding="utf-8"))
    if queue["status"] not in ("DETECTION_COMPLETE", "AGGREGATING"):
        raise RuntimeError("all frozen evaluations must finish before aggregation")
    freeze = json.loads((ROOT / "freeze_verification.json").read_text(encoding="utf-8"))
    if freeze["status"] != "PASS" or freeze["settings"] != 13 or freeze["new_training_settings"] != 12:
        raise RuntimeError("Stage39 freeze record invalid")
    for path, digest in freeze["source_hashes"].items():
        from pathlib import Path
        if sha(Path(path)) != digest:
            raise RuntimeError(f"protected Stage12/14B source changed: {path}")
    planned = [r for r in read_csv(ROOT / "protocol_index.csv") if r["new_training_required"] == "True"]
    if len(planned) != 12:
        raise RuntimeError("frozen setting count changed")
    results, per_service, domains = [], [], []
    for item in planned:
        dataset, slug = item["dataset"], item["slug"]
        fold = ROOT / "settings" / dataset / slug
        protocol = json.loads((fold / "protocol.json").read_text(encoding="utf-8"))
        if protocol["status"] != "FROZEN_PRETRAIN" or \
           sha(fold / "role_manifest.csv") != protocol["role_manifest_sha256"] or \
           item["role_manifest_sha256"] != protocol["role_manifest_sha256"]:
            raise RuntimeError(f"frozen protocol/hash mismatch: {dataset}/{slug}")
        out = fold / "detection"
        for marker in ("calibration_verification.json", "evaluation_audit.json",
                       "completion_verification.json", "deployment_verification.json"):
            if json.loads((out / marker).read_text(encoding="utf-8"))["status"] != "PASS":
                raise RuntimeError(f"setting verification failed: {dataset}/{slug}/{marker}")
        calibration = json.loads((out / "calibration.json").read_text(encoding="utf-8"))
        run_name = "vnat" if dataset == "VNAT" else dataset
        run = fold / "runs" / run_name / protocol["protocol_id"]
        for name, expected in calibration["checkpoint_hashes"].items():
            path = fold / "role_manifest.csv" if name == "role_manifest.csv" else run / name
            if sha(path) != expected:
                raise RuntimeError(f"frozen checkpoint changed: {dataset}/{slug}/{name}")
        primary = read_csv(out / "open_set_metrics.csv")
        deployment = read_csv(out / "deployment_metrics.csv")
        if len(primary) != 6 or len(deployment) != 6:
            raise RuntimeError(f"P90/P95/P99 x two methods incomplete: {dataset}/{slug}")
        closed = json.loads((out / "known_test_closed_set.json").read_text(encoding="utf-8"))
        if closed["status"] != "PASS":
            raise RuntimeError(f"Known Test closed-set metric missing: {dataset}/{slug}")
        for group, table in (("primary", primary), ("deployment", deployment)):
            for row in table:
                q = int(row["percentile"])
                result = {"dataset": dataset, "heldout_service": item["heldout_service"],
                    "fold": slug, "setting_seed": protocol["training_seed"],
                    "evaluation_group": group, "method": row["method"], "percentile": q,
                    "known_val_macro_f1": calibration["known_validation_metrics"]["macro_f1"],
                    "known_test_macro_f1_pre_reject": closed["before_rejection"]["macro_f1"],
                    "known_test_macro_f1_post_des_v1_p95":
                        closed["after_des_v1_p95"]["macro_f1_reject_as_error"],
                    "auroc": num(row, "auroc"), "auprc": num(row, "auprc"),
                    "ufar": num(row, "ufar"), "known_frr": num(row, "known_frr"),
                    "known_acceptance": num(row, "known_acceptance"),
                    "unknown_prevalence": num(row, "unknown_prevalence"),
                    "known_n": int(row["known_test"]), "unknown_n": int(row["unknown_test"]),
                    "auxiliary_known_n": int(row["auxiliary_known_n"]) if group == "deployment" else 0,
                    "threshold": num(row, "threshold"),
                    "goal_auroc_ge_0_90": num(row, "auroc") >= .90,
                    "goal_ufar_le_0_30": num(row, "ufar") <= .30,
                    "goal_known_frr_le_0_07": num(row, "known_frr") <= .07}
                if group == "deployment":
                    result["known_service_macro_f1_pre_reject"] = num(row, "known_service_macro_f1_pre_reject")
                    result["known_service_macro_f1_post_reject"] = num(row, "known_service_macro_f1_post_reject")
                else:
                    result["known_service_macro_f1_pre_reject"] = ""
                    result["known_service_macro_f1_post_reject"] = ""
                results.append(result)
        for row in read_csv(out / "per_unknown_service.csv"):
            per_service.append(row)
        for row in read_csv(out / "domain_metrics.csv"):
            domains.append(row)
    write_csv(ROOT / "stage39_run_results.csv", results)
    write_csv(ROOT / "stage39_per_unknown_service.csv", per_service)
    write_csv(ROOT / "stage39_domain_metrics.csv", domains)
    grouped = defaultdict(list)
    for row in results:
        if row["percentile"] == 95:
            grouped[(row["dataset"], row["evaluation_group"], row["method"])].append(row)
    summaries = []
    for (dataset, group, method), rows in sorted(grouped.items()):
        if len(rows) != 4:
            raise RuntimeError(f"expected four service holdouts: {dataset}/{group}/{method}")
        summaries.append({"dataset": dataset, "evaluation_group": group, "method": method,
            "settings": 4, "mean_auroc": float(np.mean([r["auroc"] for r in rows])),
            "std_auroc": float(np.std([r["auroc"] for r in rows], ddof=1)),
            "worst_auroc": float(min(r["auroc"] for r in rows)),
            "mean_auprc": float(np.mean([r["auprc"] for r in rows])),
            "mean_ufar": float(np.mean([r["ufar"] for r in rows])),
            "worst_ufar": float(max(r["ufar"] for r in rows)),
            "mean_known_frr": float(np.mean([r["known_frr"] for r in rows])),
            "mean_known_test_macro_f1_pre_reject": float(np.mean([
                r["known_test_macro_f1_pre_reject"] for r in rows])),
            "goals_auroc_pass": sum(r["goal_auroc_ge_0_90"] for r in rows),
            "goals_ufar_pass": sum(r["goal_ufar_le_0_30"] for r in rows),
            "goals_frr_pass": sum(r["goal_known_frr_le_0_07"] for r in rows),
            "all_three_goals_pass": sum(r["goal_auroc_ge_0_90"] and
                                         r["goal_ufar_le_0_30"] and r["goal_known_frr_le_0_07"]
                                         for r in rows)})
    write_csv(ROOT / "stage39_dataset_summary.csv", summaries)
    control = read_csv(ROOT / "settings/iscx_tor/p2p_only/results.csv")
    if len(control) != 4 or any(r["checkpoint_reused"] != "True" for r in control):
        raise RuntimeError("retrospective P2P-only control invalid")
    des = [r for r in results if r["evaluation_group"] == "primary" and
           r["method"] == "des_v1" and r["percentile"] == 95]
    report_rows = [{"Dataset": r["dataset"], "Held-out service": r["heldout_service"],
        "Known F1": f"{r['known_test_macro_f1_pre_reject']:.4f}",
        "AUROC": f"{r['auroc']:.4f}", "AUPRC": f"{r['auprc']:.4f}",
        "UFAR": f"{r['ufar']:.4f}", "Known FRR": f"{r['known_frr']:.4f}"} for r in des]
    summary_rows = [{"Dataset": r["dataset"], "Group": r["evaluation_group"],
        "Method": r["method"], "Mean AUROC": f"{r['mean_auroc']:.4f}",
        "Worst AUROC": f"{r['worst_auroc']:.4f}", "Mean UFAR": f"{r['mean_ufar']:.4f}",
        "Mean FRR": f"{r['mean_known_frr']:.4f}",
        "All goals": f"{r['all_three_goals_pass']}/4"} for r in summaries]
    report = ["# Stage 39 — Coarse-service open-set results", "",
        "Development evidence on previously exposed data. Twelve predeclared whole-service holdouts; one seed 2022. No fold was removed or selected after Test evaluation.",
        "", "## DES-v1 per held-out setting (primary Known Test vs Unknown service, P95)", "",
        render_table(report_rows, list(report_rows[0])), "",
        "## Dataset summary (P95)", "", render_table(summary_rows, list(summary_rows[0])), "",
        "The `deployment` rows additionally count unseen applications belonging to retained Known services as Known. AUROC/AUPRC depend on the Unknown prevalence in each row; see the run CSV. P90/P99 and MSP are also preserved; P95 is the fixed primary operating point.",
        "", "## Retrospective control", "",
        "ISCXTor P2P-only reuses the Stage38 checkpoint and calibration; it is not a newly trained Stage39 fold. Its DES-v1 AUROC is 0.913526 and P95 UFAR is 0.706000. Do not pool it with the four newly trained Tor settings.",
        "", "## Interpretation limits", "",
        "The label taxonomy changed from unseen application to unseen service; these numbers are not a same-task improvement over Stage38. ISCX service labels are capture-derived; VNAT service families are development taxonomy. Flow-random split/capture overlap and domain imbalance remain. High Known Macro-F1 alone does not establish useful Unknown rejection.",
        "", "Source role manifests, checkpoints, calibration, per-sample scores, per-service/domain metrics and replay markers are preserved by setting. No threshold, fusion weight or model was selected from Test outcomes.", ""]
    (ROOT / "stage39_report.md").write_text("\n".join(report), encoding="utf-8")
    write_json(ROOT / "aggregate_verification.json", {"status": "PASS", "new_training_settings": 12,
        "retrospective_control_settings": 1, "run_result_rows": len(results),
        "per_unknown_service_rows": len(per_service), "domain_rows": len(domains),
        "source_hashes_unchanged": True, "checkpoint_hashes_unchanged": True,
        "all_setting_replays_pass": True, "unknown_used_for_fitting": 0,
        "test_used_for_parameter_selection": 0,
        "p95_des_v1_primary_settings_meeting_all_three_goals": sum(
            r["goal_auroc_ge_0_90"] and r["goal_ufar_le_0_30"] and
            r["goal_known_frr_le_0_07"] for r in des)})
    print(json.dumps({"status": "PASS", "stage39_new_settings": 12,
                      "p95_des_v1_primary_mean_auroc": float(np.mean([r["auroc"] for r in des])),
                      "p95_des_v1_primary_mean_ufar": float(np.mean([r["ufar"] for r in des]))}), flush=True)


if __name__ == "__main__":
    aggregate()
