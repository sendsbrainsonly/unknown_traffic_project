#!/usr/bin/env python3
"""Independent, PCAP-free verification of the frozen Stage 8B package."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

from stage8b_common import (
    STAGE8A_ROOT,
    STAGE8B_ROOT,
    VALID_SETTINGS,
    load_json,
    read_csv,
    require,
    score_k2_matrices,
    sha256_file,
    verify_all_provenance,
    verify_hash_lines,
)


BASELINES = ["Native", "Single-Full-K1", "Multi-Global-K2", "Class-P05-K2", "Component-P05-K2"]


def verify_setting(setting: str) -> dict[str, object]:
    artifact = STAGE8B_ROOT / "artifacts" / setting
    output = STAGE8B_ROOT / "outputs" / setting
    hash_count = verify_hash_lines(artifact / "dgsbv2_hashes.sha256", artifact)
    require(hash_count == 5, f"{setting}: expected five rule assets")
    access = load_json(output / "file_access_audit.json")
    require(access["status"] == "PASS", f"{setting}: access audit failed")
    require(access["known_test_opened"] == 0 and access["unknown_test_opened"] == 0, f"{setting}: Test opened")
    require(access["known_test_mu_generated"] is False and access["unknown_mu_generated"] is False, f"{setting}: Test mu generated")
    require(access["unknown_inference_executed"] is False, f"{setting}: Unknown inference executed")

    z_path = STAGE8A_ROOT / f"artifacts/{setting}/z_val_pca64.npy"
    model_path = STAGE8A_ROOT / f"artifacts/{setting}/multi_k2/models.joblib"
    manifest_path = STAGE8A_ROOT / f"artifacts/{setting}/val_manifest.csv"
    z_val = np.load(z_path, mmap_mode="r", allow_pickle=False)
    models = joblib.load(model_path)
    manifest = read_csv(manifest_path)
    names, class_scores, local_scores = score_k2_matrices(models, np.asarray(z_val))
    predicted_class_indices = np.argmax(class_scores, axis=1)
    global_scores = class_scores[np.arange(len(z_val)), predicted_class_indices]
    blocks = local_scores.reshape(len(z_val), len(names), 2)[np.arange(len(z_val)), predicted_class_indices]
    predicted_components = np.argmax(blocks, axis=1)
    local_scores_selected = blocks[np.arange(len(z_val)), predicted_components]
    threshold_rows = read_csv(artifact / "dgsbv2_local_thresholds.csv")
    threshold_lookup = {
        (row["class_name"], int(row["component_id"])): float(row["effective_threshold"])
        for row in threshold_rows
    }
    predicted_names = [names[index] for index in predicted_class_indices]
    thresholds = np.asarray([threshold_lookup[(name, int(component))] for name, component in zip(predicted_names, predicted_components)])
    global_threshold = float(load_json(artifact / "dgsbv2_global_threshold.json")["threshold"])
    global_pass = global_scores >= global_threshold
    local_pass = local_scores_selected >= thresholds
    both_pass = global_pass & local_pass
    summary = load_json(output / "calibration_summary.json")
    require(len(manifest) == summary["validation_n"], f"{setting}: validation count changed")
    require(np.isclose(global_pass.mean(), summary["global_acceptance"], rtol=0, atol=1e-15), f"{setting}: global acceptance mismatch")
    require(np.isclose(local_pass.mean(), summary["local_acceptance"], rtol=0, atol=1e-15), f"{setting}: local acceptance mismatch")
    require(np.isclose(both_pass.mean(), summary["dgsbv2_acceptance"], rtol=0, atol=1e-15), f"{setting}: AND acceptance mismatch")
    require(summary["quantile_search_performed"] is False, f"{setting}: quantile search recorded")
    require(summary["post_local_global_recalibration_performed"] is False, f"{setting}: global recalibration recorded")
    prohibited = ["mu_test.npy", "mu_unknown.npy", "z_test_pca64.npy", "z_unknown_pca64.npy", "unknown_scores.npy"]
    require(not any((artifact / name).exists() for name in prohibited), f"{setting}: prohibited Test artifact")
    return {
        "setting": setting,
        "hash_files_verified": hash_count,
        "global_acceptance": float(global_pass.mean()),
        "local_acceptance": float(local_pass.mean()),
        "dgsbv2_acceptance": float(both_pass.mean()),
        "gate_status": summary["gate_status"],
        "status": "PASS",
    }


def main() -> None:
    provenance = verify_all_provenance()
    settings = [verify_setting(setting) for setting in VALID_SETTINGS]
    old_config = load_json(STAGE8A_ROOT / "configs/evaluation_config.json")
    new_config_path = STAGE8B_ROOT / "configs/evaluation_config_v2.json"
    new_config = load_json(new_config_path)
    require(new_config["method_order"] == BASELINES + ["DGSB-v2"], "method order mismatch")
    for setting in VALID_SETTINGS:
        for method in BASELINES:
            require(
                old_config["settings"][setting]["methods"][method] == new_config["settings"][setting]["methods"][method],
                f"existing method changed: {setting}/{method}",
            )
    require(new_config["final_test_metrics"]["paired_bootstrap_iterations"] == 1000, "bootstrap count changed")
    gate = load_json(STAGE8B_ROOT / "outputs/summary/final_gate.json")
    require(gate["gate"] == "READY_FOR_ONE_SHOT_FINAL_TEST" and all(gate["checks"].values()), "final gate failed")
    require(gate["evaluation_config_v2_sha256"] == sha256_file(new_config_path), "evaluation config hash changed")
    payload = {
        "status": "PASS",
        "provenance": provenance["status"],
        "settings": settings,
        "existing_methods_modified": 0,
        "known_test_opened": 0,
        "known_test_mu_generated": False,
        "unknown_test_opened": 0,
        "unknown_mu_generated": False,
        "unknown_inference_executed": False,
        "final_gate": gate["gate"],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
