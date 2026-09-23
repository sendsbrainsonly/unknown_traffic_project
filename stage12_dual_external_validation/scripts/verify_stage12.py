#!/usr/bin/env python3
"""Verify Stage 12 protocol boundaries before or after the one-shot test."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from stage12_common import CONFIG_PATH, STAGE_ROOT, load_config, sha256_file, write_json


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def expected_runs(config: dict) -> list[tuple[str, str, int]]:
    rows = []
    for dataset in config["datasets"]:
        protocol = json.loads(
            (STAGE_ROOT / "protocol" / dataset / "unknown_class_protocol.json").read_text(encoding="utf-8")
        )
        for setting in protocol["settings"]:
            for seed in config["training_seeds"]:
                rows.append((dataset, setting, int(seed)))
    return rows


def verify_protocol(config: dict) -> dict:
    audit = json.loads((STAGE_ROOT / "outputs" / "summary" / "dataset_audit_status.json").read_text(encoding="utf-8"))
    require(audit["status"] == "PASS", "dataset/provenance audit is not PASS")
    provenance = json.loads((STAGE_ROOT / "outputs" / "summary" / "provenance_audit.json").read_text(encoding="utf-8"))
    verification_path = STAGE_ROOT / "outputs" / "summary" / "provenance_verification.json"
    if not verification_path.exists():
        write_json(verification_path, provenance)
    require(provenance["status"] == "PASS", "provenance audit is not PASS")
    require(provenance["config_sha256"] == sha256_file(CONFIG_PATH), "audit predates current Stage 12 config")
    dataset_rows = {}
    for dataset in config["datasets"]:
        root = STAGE_ROOT / "protocol" / dataset
        with (root / "dataset_inventory.csv").open(encoding="utf-8", newline="") as handle:
            inventory = list(csv.DictReader(handle))
        for row in inventory:
            source = Path(row["absolute_path"])
            require(source.is_file(), f"source file disappeared after audit: {source}")
            require(source.stat().st_size == int(row["size_bytes"]), f"source file size changed after audit: {source}")
        freeze = json.loads((root / "protocol_freeze.json").read_text(encoding="utf-8"))
        unknown = json.loads((root / "unknown_class_protocol.json").read_text(encoding="utf-8"))
        label_map = json.loads((root / "canonical_label_map.json").read_text(encoding="utf-8"))
        require(freeze["status"] == "PROTOCOL_FROZEN_PRETRAIN", f"bad protocol state: {dataset}")
        require(freeze["test_metrics_opened"] is False, f"protocol freeze says test opened: {dataset}")
        require(freeze["unknown_protocol_sha256"] == sha256_file(root / "unknown_class_protocol.json"), f"unknown protocol hash mismatch: {dataset}")
        require(freeze["split_manifest_sha256"] == sha256_file(root / "split_manifest.csv"), f"split manifest hash mismatch: {dataset}")
        require(unknown["nested"] is True, f"unknown settings are not nested: {dataset}")
        require(set(label_map["canonical_class_to_label"]) == set(unknown["eligible_classes_canonical_sorted"]), f"label/protocol class mismatch: {dataset}")
        for setting, spec in unknown["settings"].items():
            require(not (set(spec["known_classes"]) & set(spec["unknown_classes"])), f"Known/Unknown overlap: {dataset}/{setting}")
            setting_protocol = freeze["settings"][setting]
            require(setting_protocol["unknown_free"] is True, f"Unknown-free flag failed: {dataset}/{setting}")
            require(not any(setting_protocol["prohibited_overlaps"].values()), f"prohibited split overlap: {dataset}/{setting}")
            require(set(setting_protocol["known_classes"]) == set(spec["known_classes"]), f"Known class mismatch: {dataset}/{setting}")
            require(set(setting_protocol["unknown_classes"]) == set(spec["unknown_classes"]), f"Unknown class mismatch: {dataset}/{setting}")
        dataset_rows[dataset] = {
            "eligible_class_count": freeze["eligible_class_count"],
            "settings": len(unknown["settings"]),
            "split_policy": freeze["dataset_split_policy"],
            "fallback_classes": freeze["group_aware_not_feasible_classes"],
        }
    return dataset_rows


def verify_pretest(config: dict) -> dict:
    runs = expected_runs(config)
    checkpoint_hashes = set()
    for dataset, setting, seed in runs:
        root = STAGE_ROOT / "runs" / dataset / setting / f"seed{seed}"
        required = (
            "config.json", "train_manifest.csv", "val_manifest.csv", "known_test_manifest.csv",
            "unknown_test_manifest.csv", "training_log.csv", "model_best.pt", "checkpoint.sha256",
            "native_threshold.json", "des_centroids.npy", "des_centroids.sha256",
            "des_threshold.json", "pretest_freeze.json", "manifest.json", "READY_FOR_ONE_SHOT_TEST",
        )
        for name in required:
            require((root / name).is_file(), f"missing required pretest artifact {name}: {root}")
        require((root / "READY_FOR_ONE_SHOT_TEST").is_file(), f"missing READY marker: {root}")
        freeze = json.loads((root / "pretest_freeze.json").read_text(encoding="utf-8"))
        require(freeze["status"] == "READY_FOR_ONE_SHOT_TEST", f"bad run status: {root}")
        require(freeze["test_data_loaded"] is False, f"test loaded before opening: {root}")
        require(freeze["unknown_data_loaded"] is False, f"unknown loaded before opening: {root}")
        require(freeze["test_metrics_computed"] is False, f"test metric computed before opening: {root}")
        require(freeze["checkpoint_sha256"] == sha256_file(root / "model_best.pt"), f"checkpoint hash mismatch: {root}")
        require(freeze["des_centroids_sha256"] == sha256_file(root / "des_centroids.npy"), f"centroid hash mismatch: {root}")
        require((root / "checkpoint.sha256").read_text(encoding="utf-8").split()[0] == freeze["checkpoint_sha256"], f"checkpoint sidecar mismatch: {root}")
        require((root / "des_centroids.sha256").read_text(encoding="utf-8").split()[0] == freeze["des_centroids_sha256"], f"centroid sidecar mismatch: {root}")
        require(freeze["config_hash"] == sha256_file(CONFIG_PATH), f"run config hash mismatch: {root}")
        checkpoint_hashes.add(freeze["checkpoint_sha256"])
    require(not (STAGE_ROOT / "outputs" / "summary" / "FINAL_TEST_OPENING.json").exists(), "FINAL_TEST_OPENING already exists during pretest verification")
    return {"expected_runs": len(runs), "ready_runs": len(runs), "unique_checkpoint_hashes": len(checkpoint_hashes)}


def verify_final(config: dict) -> dict:
    opening_path = STAGE_ROOT / "outputs" / "summary" / "FINAL_TEST_OPENING.json"
    require(opening_path.is_file(), "FINAL_TEST_OPENING is missing")
    opening = json.loads(opening_path.read_text(encoding="utf-8"))
    runs = expected_runs(config)
    method_rows = 0
    for dataset, setting, seed in runs:
        root = STAGE_ROOT / "runs" / dataset / setting / f"seed{seed}"
        for name in (
            "final_results.json", "per_unknown_results.csv", "absorption_matrix.csv",
            "manifest.json", "SUCCESS",
        ):
            require((root / name).is_file(), f"missing required final artifact {name}: {root}")
        require((root / "SUCCESS").is_file(), f"missing final SUCCESS: {root}")
        result = json.loads((root / "final_results.json").read_text(encoding="utf-8"))
        require(result["final_test_opening_code_hash"] == opening["evaluation_code_hash"], f"opening hash mismatch: {root}")
        require(set(result["methods"]) == set(config["methods"]), f"method set mismatch: {root}")
        require(result["post_test_method_modification"] is False, f"post-test method modification recorded: {root}")
        require(result["forbidden_methods_run"] is False, f"forbidden method recorded: {root}")
        method_rows += len(result["methods"])
    summary = STAGE_ROOT / "outputs" / "summary"
    required_outputs = (
        "des_vs_opendetect_summary.csv", "cross_dataset_comparison.csv",
        "final_gate.json", "final_gate.md",
    )
    for name in required_outputs:
        require((summary / name).is_file(), f"missing summary output: {name}")
    for dataset in config["datasets"]:
        output = STAGE_ROOT / "outputs" / dataset
        for name in (
            "run_level_results.csv", "setting_summary.csv", "per_unknown_class_results.csv",
            "absorption_analysis.csv", "bootstrap_results.csv", "per_unknown_delta_summary.csv",
            "absorption_hubs.csv",
        ):
            require((output / name).is_file(), f"missing dataset output {dataset}/{name}")
    decision = json.loads((summary / "final_gate.json").read_text(encoding="utf-8"))
    require(decision["decision"] in {"EXTERNAL_CONFIRMED", "PARTIAL_CONFIRMATION", "NOT_CONFIRMED"}, "invalid final gate")
    require(decision["post_test_method_modification"] is False, "gate records a post-test method modification")
    require(decision["forbidden_methods_run"] is False, "gate records a forbidden method")
    return {"expected_runs": len(runs), "successful_runs": len(runs), "method_result_rows": method_rows, "decision": decision["decision"]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("pretest", "final"), required=True)
    args = parser.parse_args()
    config = load_config()
    report = {"status": "PASS", "phase": args.phase, "protocol": verify_protocol(config)}
    report[args.phase] = verify_pretest(config) if args.phase == "pretest" else verify_final(config)
    output = STAGE_ROOT / "outputs" / "summary" / f"verification_{args.phase}.json"
    write_json(output, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
