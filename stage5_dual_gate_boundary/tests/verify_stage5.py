#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs"
CONFIG = json.loads((ROOT / "configs" / "stage5_freeze_config.json").read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha_file(sha_path: Path) -> None:
    for line in sha_path.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split(maxsplit=1)
        path = sha_path.parent / relative.strip()
        assert path.is_file(), path
        assert sha256_file(path) == expected, path


def verify(output_root: Path = OUTPUT_ROOT) -> dict[str, object]:
    rule_root = output_root / "rule_freeze"
    protocol_root = output_root / "cstnet_protocol"
    required_rule = {
        "unknown_access_audit.md",
        "known_only_sanity.csv",
        "dgsb_calibration.csv",
        "class_thresholds.csv",
        "local_component_thresholds.csv",
        "dgsb_rule.md",
        "dgsb_rule.json",
        "dgsb_rule.sha256",
        "known_only_run_metadata.json",
    }
    required_protocol = {
        "class_inventory.csv",
        "eligibility_thresholds.csv",
        "grouping_audit.csv",
        "fold_selection_audit.csv",
        "cstnet_open_set_protocol.md",
        "cstnet_open_set_protocol.json",
        "low_fold.json",
        "medium_fold.json",
        "high_fold.json",
        "split_manifest.csv",
        "split_hashes.sha256",
    }
    assert required_rule <= {path.name for path in rule_root.iterdir() if path.is_file()}
    assert required_protocol <= {path.name for path in protocol_root.iterdir() if path.is_file()}
    verify_sha_file(rule_root / "dgsb_rule.sha256")
    verify_sha_file(protocol_root / "split_hashes.sha256")

    rule = json.loads((rule_root / "dgsb_rule.json").read_text(encoding="utf-8"))
    method = CONFIG["method"]
    assert rule["model"] == {
        "K": 2,
        "covariance_type": "full",
        "pca_dimension": 64,
        "refit_performed": False,
        "reg_covar": 0.001,
    }
    assert rule["local_gate"]["quantile"] == 0.05
    assert rule["local_gate"]["minimum_validation_samples"] == 30
    assert rule["unknown_samples_used_for_calibration"] == 0
    assert rule["ustc_unknown_inputs_accessed"] == 0
    assert rule["created_before_cstnet_unknown_evaluation"] is True

    metadata = json.loads((rule_root / "known_only_run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["fit_operations_performed"] == []
    assert metadata["ustc_unknown_inputs_accessed"] == 0
    assert metadata["cstnet_training_executed"] is False
    assert metadata["cstnet_unknown_scores_accessed"] is False
    blocked = ("test_unknown", "frozen_final_predictions", "posthoc_unknown", "transition")
    assert all(not any(token in path.lower() for token in blocked) for path in metadata["accessed_input_files"])
    for item in metadata["verified_known_asset_hashes"]:
        path = Path(item["file"])
        assert path.is_file(), path
        assert sha256_file(path) == item["sha256"], path

    sanity = pd.read_csv(rule_root / "known_only_sanity.csv")
    assert len(sanity) == 1248
    assert set(sanity["setting"]) == {"A-1", "A-2", "A-3"}
    assert set(sanity["split"]) == {"Known Validation", "Known Test"}
    assert set(sanity["rule"]) == {"Global", "Class-P05", "Component-P05", "DGSB"}
    overall = sanity[sanity["level"] == "overall"].copy()
    dgsb_val = overall[(overall["rule"] == "DGSB") & (overall["split"] == "Known Validation")]
    dgsb_test = overall[(overall["rule"] == "DGSB") & (overall["split"] == "Known Test")]
    target_error = (dgsb_val["acceptance_rate"] - method["global_target_known_validation_acceptance"]).abs()
    metric_reasons: list[str] = []
    if (target_error > method["known_validation_target_tolerance"]).any():
        metric_reasons.append("DGSB validation acceptance outside frozen tolerance")
    if (dgsb_test["acceptance_rate"] < method["minimum_noncollapse_acceptance"]).any():
        metric_reasons.append("DGSB Known Test acceptance collapse")
    for split in ("Known Validation", "Known Test"):
        selected = overall[overall["split"] == split].set_index(["setting", "rule"])
        for setting in ("A-1", "A-2", "A-3"):
            increase = (
                selected.loc[(setting, "DGSB"), "mean_component_acceptance_gap"]
                - selected.loc[(setting, "Global"), "mean_component_acceptance_gap"]
            )
            if increase > method["maximum_mean_component_gap_increase_vs_global"]:
                metric_reasons.append(f"{setting}/{split}: DGSB component gap increase {increase}")

    class_inventory = pd.read_csv(protocol_root / "class_inventory.csv")
    thresholds = pd.read_csv(protocol_root / "eligibility_thresholds.csv")
    assert len(class_inventory) == 120
    assert int(class_inventory["eligible_for_open_set"].sum()) == 119
    observed_thresholds = dict(
        zip(thresholds["minimum_total_samples_per_class"], thresholds["eligible_classes"])
    )
    assert observed_thresholds == {50: 119, 100: 119, 200: 98, 500: 3}
    protocol = json.loads((protocol_root / "cstnet_open_set_protocol.json").read_text(encoding="utf-8"))
    assert protocol["created_before_unknown_evaluation"] is True
    assert protocol["cstnet_training_executed"] is False
    assert protocol["cstnet_unknown_scores_accessed"] is False
    assert protocol["grouping"]["domain_endpoint_risk"] == "PROTOCOL_RISK"

    manifest = pd.read_csv(protocol_root / "split_manifest.csv")
    assert len(manifest) == 46372 * 3
    for fold_name, expected_unknown in (("low", 6), ("medium", 18), ("high", 30)):
        fold = json.loads((protocol_root / f"{fold_name}_fold.json").read_text(encoding="utf-8"))
        assert fold["unknown_class_count"] == expected_unknown
        assert fold["known_class_count"] == 119 - expected_unknown
        assert fold["group_disjoint"] is True
        assert fold["active_group_overlap_count"] == 0
        rows = manifest[manifest["fold"] == fold_name]
        active = rows[rows["split"].isin(["known_train", "known_validation", "known_test", "unknown_final_test"])]
        assert (active.groupby("group_id")["split"].nunique() == 1).all()
        unknown_classes = set(fold["unknown_classes"])
        assert set(rows.loc[rows["split"] == "unknown_final_test", "class_name"]) == unknown_classes
        assert not set(rows.loc[rows["split"].str.startswith("known_"), "class_name"]) & unknown_classes

    forbidden_suffixes = {".pt", ".bin", ".joblib", ".npy", ".parquet"}
    assert not [path for path in ROOT.rglob("*") if path.is_file() and path.suffix.lower() in forbidden_suffixes]
    gate = "READY" if not metric_reasons else "NOT READY"
    return {
        "gate": gate,
        "metric_reasons": metric_reasons,
        "dgsb_rule_sha256": sha256_file(rule_root / "dgsb_rule.json"),
        "raw_classes": 120,
        "eligible_classes": 119,
        "split_hashes": "PASS",
        "cstnet_training_executed": False,
        "cstnet_unknown_scores_accessed": False,
    }


def main() -> None:
    print(json.dumps(verify(), sort_keys=True))


if __name__ == "__main__":
    main()
