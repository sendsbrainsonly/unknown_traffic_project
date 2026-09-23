#!/usr/bin/env python3
"""Independent, PCAP-free verification of the completed Stage 8A freeze."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import joblib
import numpy as np

from common import STAGE8_ROOT, VALID_SETTINGS, load_fold, require, sha256_file, verify_all_provenance


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def verify_bundle(setting: str) -> int:
    root = STAGE8_ROOT / "artifacts" / setting
    manifest = load_json(root / "bundle_manifest.json")
    require(manifest["status"] == "FROZEN", f"{setting}: bundle not frozen")
    lines = (root / "bundle_hashes.sha256").read_text(encoding="utf-8").splitlines()
    require(len(lines) == len(manifest["files"]), f"{setting}: bundle hash count mismatch")
    for line, item in zip(lines, manifest["files"]):
        expected, relative = line.split(maxsplit=1)
        require(expected == item["sha256"] and relative == item["path"], f"{setting}: bundle ledger mismatch")
        require(sha256_file((root / relative).resolve()) == expected, f"{setting}: frozen file hash mismatch: {relative}")
    return len(lines)


def verify_setting(setting: str) -> dict[str, object]:
    fold = load_fold(setting)
    artifact = STAGE8_ROOT / "artifacts" / setting
    output = STAGE8_ROOT / "outputs" / setting
    train = np.load(artifact / "mu_train.npy", mmap_mode="r", allow_pickle=False)
    val = np.load(artifact / "mu_val.npy", mmap_mode="r", allow_pickle=False)
    require(train.shape == (int(fold["known_train_count"]), 128), f"{setting}: train mu shape mismatch")
    require(val.shape == (int(fold["known_validation_count"]), 128), f"{setting}: val mu shape mismatch")
    require(np.isfinite(train).all() and np.isfinite(val).all(), f"{setting}: non-finite mu")
    representation = load_json(output / "representation_audit.json")
    require(not representation["train"]["collapse_detected"] and not representation["validation"]["collapse_detected"], f"{setting}: representation collapse")
    fit = load_json(output / "density_boundary_metadata.json")
    require(fit["scaler_fit_split"] == "KNOWN_TRAIN_ONLY" and fit["pca_fit_split"] == "KNOWN_TRAIN_ONLY", f"{setting}: transform leakage")
    require(fit["k1_fit_split"] == "KNOWN_TRAIN_ONLY" and fit["k2_fit_split"] == "KNOWN_TRAIN_ONLY", f"{setting}: density leakage")
    require(fit["threshold_fit_split"] == "KNOWN_VALIDATION_ONLY", f"{setting}: threshold leakage")
    k1 = joblib.load(artifact / "single_k1/models.joblib")
    k2 = joblib.load(artifact / "multi_k2/models.joblib")
    require(list(k1) == fold["known_classes"] and list(k2) == fold["known_classes"], f"{setting}: model class ordering mismatch")
    require(all(model.n_components == 1 and model.covariance_type == "full" for model in k1.values()), f"{setting}: K1 contract mismatch")
    require(all(model.n_components == 2 and model.covariance_type == "full" for model in k2.values()), f"{setting}: K2 contract mismatch")
    diagnostics = csv_rows(output / "k2_component_diagnostics.csv")
    require(len(diagnostics) == len(fold["known_classes"]), f"{setting}: K2 diagnostic rows missing")
    access = load_json(output / "file_access_audit.json")
    require(access["status"] == "PASS", f"{setting}: access audit failed")
    require(int(access["pcap_files_opened"]) == 0, f"{setting}: Stage 8A opened PCAP")
    require(int(access["known_test_opened"]) == 0 and int(access["unknown_test_opened"]) == 0, f"{setting}: Test access")
    require(access["known_test_mu_generated"] is False and access["unknown_mu_generated"] is False, f"{setting}: forbidden mu generated")
    prohibited = ("mu_test.npy", "mu_unknown.npy", "test_manifest.csv", "unknown_manifest.csv", "z_test_pca64.npy", "z_unknown_pca64.npy")
    require(not any((artifact / name).exists() for name in prohibited), f"{setting}: forbidden Test/Unknown artifact exists")
    classification = load_json(output / "known_val_classification_summary.json")
    require(classification["parity"] == "PASS", f"{setting}: Stage 7 validation replay failed")
    thresholds = csv_rows(artifact / "component_p05_thresholds.csv")
    require(len(thresholds) == 2 * len(fold["known_classes"]), f"{setting}: fallback map incomplete")
    bundle_files = verify_bundle(setting)
    return {
        "setting": setting,
        "mu_train_shape": list(train.shape),
        "mu_val_shape": list(val.shape),
        "pca_explained_variance": float(fit["pca_cumulative_explained_variance"]),
        "k1_all_converged": bool(fit["k1_all_converged"]),
        "k2_all_converged": bool(fit["k2_all_converged"]),
        "bundle_files_verified": bundle_files,
        "known_test_opened": 0,
        "unknown_test_opened": 0,
        "status": "PASS",
    }


def main() -> None:
    provenance = verify_all_provenance()
    results = [verify_setting(setting) for setting in VALID_SETTINGS]
    evaluation = load_json(STAGE8_ROOT / "configs/evaluation_config.json")
    require(evaluation["method_order"] == ["Native", "Single-Full-K1", "Multi-Global-K2", "Class-P05-K2", "Component-P05-K2"], "evaluation method list changed")
    require(evaluation["excluded_methods"] == ["DGSB-v1", "DGSB-v2"], "DGSB exclusion changed")
    gate = load_json(STAGE8_ROOT / "outputs/summary/final_gate.json")
    require(gate["gate"] == "READY_FOR_FINAL_TEST" and all(gate["checks"].values()), "final gate mismatch")
    print(json.dumps({"status": "PASS", "provenance": provenance["status"], "settings": results, "evaluation_config_frozen": True, "known_test_opened": 0, "unknown_test_opened": 0, "unknown_inference_executed": False, "final_gate": gate["gate"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
