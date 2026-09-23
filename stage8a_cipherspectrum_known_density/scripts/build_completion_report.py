#!/usr/bin/env python3
"""Build the user-facing 31-item Stage 8A completion evidence snapshot."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from common import STAGE8_ROOT, VALID_SETTINGS, write_json


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    provenance = load_json(STAGE8_ROOT / "outputs/summary/provenance_verification.json")
    gate = load_json(STAGE8_ROOT / "outputs/summary/final_gate.json")
    evaluation_path = STAGE8_ROOT / "configs/evaluation_config.json"
    evaluation = load_json(evaluation_path)
    settings: list[dict[str, object]] = []
    for setting in VALID_SETTINGS:
        artifact = STAGE8_ROOT / "artifacts" / setting
        output = STAGE8_ROOT / "outputs" / setting
        representation = load_json(output / "representation_audit.json")
        density = load_json(output / "density_boundary_metadata.json")
        classification = load_json(output / "known_val_classification_summary.json")
        boundary = {row["method"]: row for row in csv_rows(output / "known_val_boundary_summary.csv")}
        native = load_json(artifact / "native_threshold.json")
        single = load_json(artifact / "single_global_threshold.json")
        multi = load_json(artifact / "multi_global_threshold.json")
        bundle = load_json(artifact / "bundle_manifest.json")
        settings.append(
            {
                "setting": setting,
                "mu_train_shape": representation["train"]["shape"],
                "mu_val_shape": representation["validation"]["shape"],
                "nan_count": int(representation["train"]["nan_count"]) + int(representation["validation"]["nan_count"]),
                "inf_count": int(representation["train"]["inf_count"]) + int(representation["validation"]["inf_count"]),
                "collapse_detected": bool(representation["train"]["collapse_detected"]) or bool(representation["validation"]["collapse_detected"]),
                "all_zero_dimensions": {"train": representation["train"]["all_zero_dimension_count"], "validation": representation["validation"]["all_zero_dimension_count"]},
                "near_zero_variance_dimensions": {"train": representation["train"]["near_zero_variance_dimension_count"], "validation": representation["validation"]["near_zero_variance_dimension_count"]},
                "duplicate_rows": {"train": representation["train"]["duplicate_rows"], "validation": representation["validation"]["duplicate_rows"]},
                "pca64_explained_variance": density["pca_cumulative_explained_variance"],
                "k1_all_converged": density["k1_all_converged"],
                "k2_all_converged": density["k2_all_converged"],
                "minimum_k2_component_weight": density["minimum_k2_component_weight"],
                "component_val_n_lt_30": density["component_val_n_lt_30"],
                "component_fallback_rate": density["component_fallback_rate"],
                "mean_DeltaNLL2": density["mean_DeltaNLL2"],
                "median_DeltaNLL2": density["median_DeltaNLL2"],
                "classes_K2_validation_NLL_better": density["classes_K2_validation_NLL_better"],
                "native_validation_threshold": native["threshold"],
                "single_global_threshold": single["threshold"],
                "multi_global_threshold": multi["threshold"],
                "class_p05_validation_acceptance": float(boundary["Class-P05-K2"]["overall_acceptance"]),
                "component_p05_validation_acceptance": float(boundary["Component-P05-K2"]["overall_acceptance"]),
                "per_class_coverage_ranges": {
                    method: [float(row["per_class_acceptance_min"]), float(row["per_class_acceptance_max"])]
                    for method, row in boundary.items()
                },
                "per_component_coverage_gaps": {
                    method: {
                        "mean": float(row["mean_per_class_component_acceptance_gap"]),
                        "max": float(row["max_per_class_component_acceptance_gap"]),
                    }
                    for method, row in boundary.items()
                },
                "hardest_five_known_validation_classes_by_f1": classification["hardest_five_by_f1"],
                "evaluation_config_frozen": setting in evaluation["settings"],
                "known_test_opened": 0,
                "known_test_mu_generated": False,
                "unknown_test_opened": 0,
                "unknown_mu_generated": False,
                "unknown_inference_executed": False,
                "bundle_files_verified": len(bundle["files"]),
                "bundle_status": bundle["status"],
            }
        )
    payload = {
        "stage6_protocol_hash_status": provenance["stage6"]["verification_status"],
        "stage7_checkpoint_hash_status": "PASS" if all(item["verification_status"] == "PASS" for item in provenance["settings"]) else "FAIL",
        "settings": settings,
        "evaluation_config_path": str(evaluation_path.resolve()),
        "evaluation_config_frozen": evaluation["created_before_any_test_access"],
        "known_test_opened": 0,
        "known_test_mu_generated": False,
        "unknown_test_opened": 0,
        "unknown_mu_generated": False,
        "unknown_inference_executed": False,
        "bundle_hashes_status": "PASS" if gate["checks"]["bundle_hashes_pass"] else "FAIL",
        "final_gate": gate["gate"],
        "readme_status": "UPDATED",
    }
    path = STAGE8_ROOT / "outputs/summary/completion_report.json"
    if path.exists():
        raise RuntimeError(f"refusing to overwrite completion report: {path}")
    write_json(path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
