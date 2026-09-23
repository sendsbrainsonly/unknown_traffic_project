#!/usr/bin/env python3
"""Terminal verification for the complete Stage 11A result bundle."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from stage11a_common import CONFIG_PATH, STAGE_ROOT, read_json, sha256_file


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    config = read_json(CONFIG_PATH)
    provenance = read_json(STAGE_ROOT / "outputs" / "summary" / "provenance_verification.json")
    require(provenance["status"] == "PASS" and provenance["phase"] == "after", "final provenance is not PASS/after")
    require(not provenance["cipherspectrum_stage9_sample_level_files_opened"], "CipherSpectrum sample-level Test was opened")
    require(not provenance["frozen_experiment_modified"], "a frozen experiment changed")
    dap_count = 0
    b0_count = 0
    for scenario, scenario_cfg in config["scenarios"].items():
        for fold, seed in enumerate(config["seeds"]):
            run = STAGE_ROOT / "runs" / scenario / f"fold{fold}_seed{seed}"
            required = [
                "config.json", "training_log.csv", "prototype_drift_history.csv", "model_best.pt",
                "checkpoint.sha256", "results.json", "density_models.joblib", "RESULTS.md", "manifest.json", "SUCCESS",
            ]
            require(all((run / name).is_file() for name in required), f"incomplete DAP run: {run}")
            require((run / "SUCCESS").read_text().strip() == "STAGE11A_DAP_FORMAL_SUCCESS", f"wrong DAP marker: {run}")
            result = read_json(run / "results.json")
            training = result["training"]
            require(training["prototype_in_optimizer"] is False, f"prototype in optimizer: {run}")
            require(training["prototype_update"] == "ANCHOR_EVERY_EPOCH", f"wrong anchor schedule: {run}")
            require(training["unknown_used_for_training_or_selection"] is False, f"unknown leakage: {run}")
            require(result["scenario"] == scenario and result["seed"] == seed, f"result identity mismatch: {run}")
            require(set(result["density_diagnostics"]["methods"]) == {"K1", "K2"}, f"wrong density matrix: {run}")
            require(result["density_diagnostics"]["methods"]["K1"]["components"] == 1, f"wrong K1: {run}")
            require(result["density_diagnostics"]["methods"]["K2"]["components"] == 2, f"wrong K2: {run}")
            drift = csv_rows(run / "prototype_drift_history.csv")
            expected_rows = (int(training["stop_epoch"]) + 1) * int(scenario_cfg["known_count"])
            require(len(drift) == expected_rows, f"incomplete epoch x class drift history: {run}")
            epochs = sorted({int(row["epoch"]) for row in drift})
            require(epochs == list(range(int(training["stop_epoch"]) + 1)), f"missing anchor epochs: {run}")
            require(all(row["prototype_source"] == "KNOWN_TRAIN_DETERMINISTIC_MU_ONLY" for row in drift), f"wrong anchor source: {run}")
            require(max(float(row["after_anchor_gap"]) for row in drift) < 1e-5, f"anchor did not zero gap: {run}")
            checkpoint_hash = (run / "checkpoint.sha256").read_text().split()[0]
            require(checkpoint_hash == sha256_file(run / "model_best.pt"), f"DAP checkpoint hash mismatch: {run}")
            dap_count += 1

            b0 = STAGE_ROOT / "artifacts" / "b0" / scenario / f"fold{fold}_seed{seed}"
            require((b0 / "SUCCESS").is_file(), f"missing B0 diagnostic: {b0}")
            b0_result = read_json(b0 / "results.json")
            parity = b0_result["frozen_b0"]["native_recompute_parity"]
            require(parity["status"] == "PASS", f"B0 parity failed: {b0}")
            require(b0_result["frozen_b0"]["retrained"] is False, f"B0 was retrained: {b0}")
            b0_count += 1

    required_summary = [
        "provenance_verification.json", "training_summary.csv", "prototype_alignment.csv",
        "native_detection_results.csv", "k1_k2_diagnostics.csv", "paired_seed_comparison.csv",
        "scenario_summary.csv", "mechanism_case.md", "final_gate.md", "final_gate.json",
    ]
    summary = STAGE_ROOT / "outputs" / "summary"
    require(all((summary / name).is_file() for name in required_summary), "missing required summary output")
    require(len(csv_rows(summary / "training_summary.csv")) == 30, "training summary must have 30 B0/DAP rows")
    require(len(csv_rows(summary / "native_detection_results.csv")) == 30, "Native summary must have 30 rows")
    require(len(csv_rows(summary / "k1_k2_diagnostics.csv")) == 60, "K1/K2 summary must have 60 rows")
    require(len(csv_rows(summary / "paired_seed_comparison.csv")) == 15, "paired summary must have 15 rows")
    gate = read_json(summary / "final_gate.json")
    require(gate["final_gate"] in {"GO", "CONDITIONAL_GO", "NO_GO"}, "invalid final gate")
    require(gate["mechanism_case"] in {"P1", "P2", "P3", "P4"}, "invalid mechanism case")
    require(gate["next_stage_started"] is False, "next stage was started")
    require(not (STAGE_ROOT.parent / "stage11b_data_anchored_covariance").exists(), "unexpected Stage11B directory")
    print(json.dumps({
        "status": "PASS",
        "dap_formal_runs": dap_count,
        "b0_frozen_evaluations": b0_count,
        "comparison_cells": dap_count + b0_count,
        "native_rows": 30,
        "k1_k2_rows": 60,
        "paired_rows": 15,
        "final_gate": gate["final_gate"],
        "mechanism_case": gate["mechanism_case"],
        "cipherspectrum_stage9_sample_level_test_used": False,
        "frozen_experiment_modified": False,
        "next_stage_started": False,
    }, indent=2))


if __name__ == "__main__":
    main()
