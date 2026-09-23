#!/usr/bin/env python3
"""Completion checks for the materialized Stage 4 diagnosis evidence."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
OUTPUT = ROOT / "outputs"
sys.path.insert(0, str(ROOT / "scripts"))
import run_boundary_diagnosis as stage4  # noqa: E402


REQUIRED = {
    "provenance_snapshot.md", "boundary_leakage_audit.md", "global_class_coverage.csv",
    "global_component_coverage.csv", "boundary_heterogeneity.csv", "class_thresholds.csv",
    "local_component_thresholds.csv", "known_validation_boundary_comparison.csv",
    "known_test_boundary_generalization.csv", "htbot_tinba_case_study.csv",
    "virut_neris_case_study.csv", "posthoc_unknown_boundary_results.csv",
    "boundary_factor_correlations.csv", "fallback_summary.csv",
    "external_validation_protocol_candidate.md", "audit_summary.md", "run_metadata.json",
    "RESULTS.md", "manifest.json",
}
HEADINGS = {
    "## Motivation", "## Why Density Fit Is Not Enough", "## Frozen Stage3 Assets",
    "## Global Boundary", "## Class-Calibrated Boundary", "## Local Component Boundary",
    "## Known Validation Coverage", "## Known Test Generalization", "## Htbot–Tinba Case",
    "## Virut–Neris Case", "## Post-Hoc Validity Warning", "## Final Diagnosis",
    "## Candidate External Rule", "## Next Step",
}


def main() -> None:
    missing = sorted(name for name in REQUIRED if not (OUTPUT / name).is_file())
    assert not missing, missing
    assert HEADINGS <= set((ROOT / "README.md").read_text(encoding="utf-8").splitlines())

    class_cov = pd.read_csv(OUTPUT / "global_class_coverage.csv")
    comp_cov = pd.read_csv(OUTPUT / "global_component_coverage.csv")
    hetero = pd.read_csv(OUTPUT / "boundary_heterogeneity.csv")
    class_tau = pd.read_csv(OUTPUT / "class_thresholds.csv")
    local_tau = pd.read_csv(OUTPUT / "local_component_thresholds.csv")
    validation = pd.read_csv(OUTPUT / "known_validation_boundary_comparison.csv")
    known_test = pd.read_csv(OUTPUT / "known_test_boundary_generalization.csv")
    assert len(class_cov) == 51 and len(comp_cov) == 102 and len(hetero) == 51
    assert len(class_tau) == 51 and len(local_tau) == 102
    assert len(validation) == 468 and len(known_test) == 468
    assert int(class_tau["unknown_samples_used"].sum()) == 0
    assert int(local_tau["unknown_samples_used"].sum()) == 0
    assert set(class_tau["quantile"]) == {0.05} and set(local_tau["quantile"]) == {0.05}
    assert int(local_tau["fallback_to_class_threshold"].sum()) == 3

    virut = comp_cov[(comp_cov.setting == "A-3") & (comp_cov.known_class == "Virut")].sort_values("component_id")
    htbot = comp_cov[(comp_cov.setting == "A-1") & (comp_cov.known_class == "Htbot")].sort_values("component_id")
    assert np.allclose(virut["global_acceptance_rate"], [0.0972037283621837, 0.9863227823368504])
    assert np.allclose(htbot["global_acceptance_rate"], [0.7978494623655914, 0.8165680473372781])
    maximum = hetero.sort_values("val_global_component_acceptance_gap", ascending=False).iloc[0]
    assert (maximum.setting, maximum.known_class) == ("A-3", "Virut")
    assert np.isclose(maximum.val_global_component_acceptance_gap, 0.8891190539746666)

    def mean_gap(frame: pd.DataFrame, rule: str) -> float:
        return float(frame[(frame.rule == rule) & (frame.level == "class")]["component_acceptance_gap"].mean())
    assert mean_gap(validation, "Component-P05") < mean_gap(validation, "Class-P05")
    assert mean_gap(known_test, "Component-P05") < mean_gap(known_test, "Class-P05")

    ht_case = pd.read_csv(OUTPUT / "htbot_tinba_case_study.csv")
    vir_case = pd.read_csv(OUTPUT / "virut_neris_case_study.csv")
    assert len(ht_case) == 850 and int(ht_case["focus_case"].sum()) == 86
    assert set(ht_case.loc[ht_case.focus_case.astype(bool), "case_component_id"]) == {1}
    assert int((~ht_case.loc[ht_case.focus_case.astype(bool), "local_p05_accept"].astype(bool)).sum()) == 0
    assert len(vir_case) == 236 and vir_case["focus_case"].astype(bool).all()
    assert vir_case["case_component_id"].value_counts().to_dict() == {0: 179, 1: 57}
    assert int((~vir_case["local_p05_accept"].astype(bool)).sum()) == 0
    assert set(vir_case["local_p05_predicted_class"]) == {"Virut"}
    assert set(vir_case["local_p05_predicted_component"]) == {0}

    posthoc = pd.read_csv(OUTPUT / "posthoc_unknown_boundary_results.csv")
    assert len(posthoc) == 36 and set(posthoc["validity"]) == {"POST_HOC_DIAGNOSTIC_ONLY"}
    assert int(posthoc["unknown_used_for_calibration"].sum()) == 0
    observed = posthoc[posthoc.unknown_class.eq("__ALL__")].set_index(["setting", "rule"])["unknown_false_acceptance_rate"]
    assert np.isclose(observed.loc[("A-1", "Global")], 0.6235294117647059)
    assert np.isclose(observed.loc[("A-3", "Component-P05")], 0.2853677469435271)

    metadata = json.loads((OUTPUT / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["prediction_replay_all_pass"] is True
    assert sum(row["rows"] for row in metadata["prediction_replay"]) == 146733
    assert all(row["class_and_binary_parity"] for row in metadata["prediction_replay"])
    assert all(row["native_score_max_abs_error"] == row["single_score_max_abs_error"] == row["multi_score_max_abs_error"] == 0 for row in metadata["prediction_replay"])
    assert metadata["fit_operations_performed"] == []
    assert metadata["unknown_calibration_samples"] == 0
    assert metadata["execution_order"][-1] == "post-hoc Unknown Test mechanism"
    assert metadata["diagnosis"] == "C — LOCAL BOUNDARY NECESSARY"
    assert metadata["external_training_executed"] is False
    for setting in stage4.SETTINGS:
        digest, count = stage4.tree_hash(stage4.STAGE3_ROOT / "outputs" / setting)
        assert digest == stage4.PACKAGE_HASHES[setting]
        assert count == metadata["stage3_package_hashes_after"][setting]["files"]

    protocol = (OUTPUT / "external_validation_protocol_candidate.md").read_text(encoding="utf-8")
    assert "Component-P05" in protocol and "CSTNET-TLS1.3" in protocol and "No CSTNET training" in protocol
    assert "Stage 4 Local Boundary Diagnosis | ✅ 完成" in (PROJECT / "README.md").read_text(encoding="utf-8")
    assert "stage4-local-boundary-diagnosis-20260913" in (PROJECT / "EXPERIMENT_RESULTS.md").read_text(encoding="utf-8")
    print(json.dumps({"status": "PASS", "required_files": len(REQUIRED), "replayed_rows": 146733, "diagnosis": metadata["diagnosis"]}, sort_keys=True))


if __name__ == "__main__":
    main()
