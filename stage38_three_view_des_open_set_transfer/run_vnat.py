#!/usr/bin/env python3
"""Run the frozen VNAT part of Stage 38 in an isolated output directory.

The tested score implementation is reused from Stage 36 without writing to
that historical experiment. Stage 38 only changes the experiment location.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
OUT = ROOT / "vnat"
HISTORICAL = PROJECT / "stage36_vnat_sixclass_open_set_pilot"
sys.path.insert(0, str(HISTORICAL))
import run_pilot as pilot  # noqa: E402

pilot.ROOT = OUT
METHODS = {"msp": "C0", "energy": "C1", "centroid": "D0", "des_v1": "D1"}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def write_once(path: Path, obj: object) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")


def protected_hashes() -> dict[str, str]:
    files = [
        pilot.S14 / "vnat_open_set_protocol.json",
        pilot.S14 / "vnat_split_manifest.csv",
        pilot.S33 / "selected_head_before_test.json",
        pilot.S33 / "run_six.py",
        pilot.S33 / "runs/vnat/adapters_best.pt",
        pilot.S33 / "runs/vnat/T0_equal_best.pt",
        pilot.S31 / "train_equal_fusion.py",
        HISTORICAL / "EXPERIMENT_PLAN.md",
        HISTORICAL / "RESULTS.md",
        HISTORICAL / "manifest.json",
        HISTORICAL / "preflight.json",
        HISTORICAL / "parity.json",
        HISTORICAL / "known_smoke.json",
        HISTORICAL / "run_pilot.py",
    ]
    files += [pilot.S31 / "runs/vnat/medium_seed2025" / name / "model_best.pt"
              for name in pilot.VIEWS]
    return {str(path.relative_to(PROJECT)): digest(path) for path in files}


def preflight() -> None:
    OUT.mkdir(exist_ok=False)
    old = json.loads((HISTORICAL / "preflight.json").read_text())
    parity = json.loads((HISTORICAL / "parity.json").read_text())
    smoke = json.loads((HISTORICAL / "known_smoke.json").read_text())
    current = pilot.frozen_hashes()
    if old["status"] != "PASS" or old["source_hashes"] != current:
        raise RuntimeError("Stage 36 frozen source audit no longer matches")
    if parity["status"] != "PASS" or parity["exact_array_comparisons"] != 66:
        raise RuntimeError("Stage 36 Known Validation packet parity not PASS")
    if smoke["status"] != "PASS" or smoke["max_abs_logit_difference"] != 0:
        raise RuntimeError("Stage 36 frozen head parity not PASS")
    write_once(OUT / "historical_hashes_before.json", protected_hashes())
    pilot.preflight()
    if json.loads((OUT / "preflight.json").read_text())["source_hashes"] != current:
        raise RuntimeError("new preflight differs from frozen Stage 36 source")
    write_once(OUT / "stage36_gate_recheck.json", {
        "status": "PASS", "stage36_packet_comparisons": 66,
        "stage36_known_validation_logits_max_abs_diff": 0.0,
        "stage36_unchanged": True, "stage38_protocol": "Stage14B medium_seed2025",
        "unknown_values_read": 0, "test_values_read": 0,
    })


def verify() -> None:
    pilot.verify()
    before = json.loads((OUT / "historical_hashes_before.json").read_text())
    after = protected_hashes()
    write_once(OUT / "historical_hashes_after.json", after)
    if after != before:
        raise RuntimeError("historical Stage 14B/31/32/33/36 source changed")
    write_once(OUT / "stage38a_verification.json", {
        "status": "PASS", "stage36_score_replay": "PASS",
        "protected_hashes_unchanged": True, "protected_files": len(before),
        "stage14b_freeze_hash": pilot.FREEZE,
        "unknown_fit_count": 0, "test_fit_count": 0,
        "encoder_training_count": 0,
    })


def finalize() -> None:
    if json.loads((OUT / "stage38a_verification.json").read_text())["status"] != "PASS":
        raise RuntimeError("Stage 38A independent verification not PASS")
    with (OUT / "open_set_results.csv").open(newline="", encoding="utf-8") as handle:
        original = list(csv.DictReader(handle))
    if {row["method"] for row in original} != set(METHODS):
        raise RuntimeError("registered score set mismatch")
    rows = [{"score_id": METHODS[row["method"]], **row} for row in original]
    with (OUT / "stage38a_method_results.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lookup = {row["score_id"]: row for row in rows}
    paired = []
    for left, right in (("D1", "D0"), ("D0", "C0"), ("D1", "C0"),
                        ("D0", "C1"), ("D1", "C1")):
        paired.append({"dataset": "VNAT", "protocol": pilot.PROTOCOL,
                       "left": left, "right": right,
                       **{f"delta_{key}": float(lookup[left][key]) - float(lookup[right][key])
                          for key in ("auroc", "auprc", "ufar", "known_frr")}})
    with (OUT / "stage38a_paired_comparison.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)
    known = np.load(OUT / "representations/known_test.npz", allow_pickle=False)
    ids = known["flow_ids"].astype(str).tolist()
    frozen = pilot.frozen_rows()["known_test"]
    if ids != [item["flow_uid"] for item in frozen]:
        raise RuntimeError("Known Test representation ID mismatch")
    truth = np.asarray([pilot.stage33.CLASSES.index(pilot.stage33.LABEL_MAP[r["application"]])
                        for r in frozen])
    pred = known["logits"].argmax(1)
    closed = {"accuracy": float(accuracy_score(truth, pred)),
              "macro_f1": float(f1_score(truth, pred, average="macro")),
              "weighted_f1": float(f1_score(truth, pred, average="weighted")),
              "known_test": len(truth)}
    write_once(OUT / "closed_set_metrics.json", closed)
    table = ["| Score | AUROC | AUPRC | UFAR | Known FRR |",
             "|---|---:|---:|---:|---:|"]
    for row in rows:
        table.append("| {score_id} | {auroc:.6f} | {auprc:.6f} | {ufar:.6f} | {known_frr:.6f} |".format(
            score_id=row["score_id"], **{name: float(row[name]) for name in
                ("auroc", "auprc", "ufar", "known_frr")}))
    (ROOT / "RESULTS.md").write_text("\n".join([
        "# Stage 38 — 三路融合表示的 DES 开集迁移实验", "",
        "状态：`in_progress / VNAT 38A complete; VPN/Tor 38B pending`。",
        "本阶段是单 seed、已暴露数据上的开发性诊断。", "",
        "## VNAT Stage 38A", "",
        "Stage14B `medium_seed2025`：Known Train/Val/Test 为 15,704/1,960/1,960，",
        "Unknown Test 3,825（sftp、vimeo、zoiper）。冻结 Stage33 六类三路模型；",
        "Stage14B/31/32/33/36 共 17 个受保护文件哈希前后一致。", "",
        "Known Test Accuracy={accuracy:.6f}，Macro-F1={macro_f1:.6f}，"
        "Weighted-F1={weighted_f1:.6f}。".format(**closed), "",
        *table, "",
        "C0=MSP，C1=Energy，D0=DES-v0，D1=DES-v1。",
        "中心和 kNN-10 只拟合 Known Train；距离归一化与四个 P95 阈值只用 Known Validation。",
        "逐流分数及判决见 `vnat/sample_scores.csv`；逐 Unknown 类见 "
        "`vnat/per_unknown_application.csv`；配对差值见 "
        "`vnat/stage38a_paired_comparison.csv`。不根据 Test 选分数。", "",
        "独立重放与原始权重哈希检查见 `vnat/stage38a_verification.json`。",
        "VNAT 六类映射是在较早四类 Test 暴露后确定，且 split 非 capture-disjoint；",
        "Vimeo 与 Known Streaming 服务语义邻近。因此结果不能称为独立外部验证。", "",
        "## 后续", "",
        "Stage38B ISCX-VPN/ISCXTor Known-only 训练及开集评估尚未完成；",
        "Stage38 整体 Gate 暂不判定。", "",
    ]), encoding="utf-8")
    manifest = json.loads((ROOT / "manifest.json").read_text())
    manifest.update({"updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running", "claim_scope": "diagnostic",
        "inputs": [{"dataset": "VNAT", "protocol": pilot.PROTOCOL,
                    "freeze_hash": pilot.FREEZE, "known_train": 15704,
                    "known_validation": 1960, "known_test": 1960, "unknown_test": 3825}],
        "execution": {"tmux_session": "stage38_gpu2_queue_0926",
                      "command": "python stage38_three_view_des_open_set_transfer/run_queue.py",
                      "exit_code": 0, "environment": "2025-10-8-WXY-dgl_py310",
                      "physical_gpu_ids": [2]},
        "configuration": {"files": ["STAGE38_THREE_VIEW_DES_OPEN_SET_TRANSFER_PLAN.md",
                                    "stage38_three_view_des_open_set_transfer/run_vnat.py"],
                          "parameters": {"scores": METHODS, "k": 10,
                                         "threshold": "Known Validation P95 higher"},
                          "seeds": [2022]},
        "core_results": [{"dataset": "VNAT", "score": row["score_id"],
                          **{key: float(row[key]) for key in
                             ("auroc", "auprc", "ufar", "known_frr")}} for row in rows],
        "limitations": ["single seed", "previous Test exposure", "VNAT capture overlap",
                        "Stage38B pending"],
        "next_step": "Run Stage38B ISCX-VPN/Tor Known-only audit and training without changing scores."})
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"phase": "finalize", "status": "PASS", "closed": closed,
                      "methods": rows}), flush=True)


PHASES = {"preflight": preflight, "parity": pilot.parity,
          "known_smoke": pilot.known_smoke, "extract": pilot.extract_unknown,
          "evaluate": pilot.evaluate, "verify": verify, "finalize": finalize}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=PHASES)
    PHASES[parser.parse_args().phase]()
