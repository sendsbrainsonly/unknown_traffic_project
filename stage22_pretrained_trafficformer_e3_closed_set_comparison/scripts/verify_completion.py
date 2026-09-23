#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from stage22_common import CONFIG, OUT, RUN_ROOT, protected_assets, read_json, sha256_file, write_json


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def check(items: list[dict], name: str, passed: bool, detail) -> None:
    items.append({"name": name, "passed": bool(passed), "detail": detail})


def main() -> None:
    config = read_json(CONFIG)
    checks: list[dict] = []
    after = {name: {"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size} for name, path in protected_assets().items()}
    write_json(OUT / "protected_asset_hashes_after.json", after)
    before = read_json(OUT / "protected_asset_hashes_before.json")
    check(checks, "protected_assets_unchanged", before == after, {name: before[name]["sha256"] == after[name]["sha256"] for name in before})

    expected_runs = [(dataset, int(seed)) for dataset in config["datasets"] for seed in config["seeds"]]
    completed = []
    for dataset, seed in expected_runs:
        run = RUN_ROOT / dataset / f"seed{seed}"
        success = run / "SUCCESS"
        manifest_path = run / "run_manifest.json"
        check(checks, f"{dataset}_{seed}_success_marker", success.is_file(), str(success))
        check(checks, f"{dataset}_{seed}_manifest", manifest_path.is_file(), str(manifest_path))
        if not manifest_path.is_file():
            continue
        manifest = read_json(manifest_path)
        check(checks, f"{dataset}_{seed}_status", manifest.get("status") == "SUCCESS", manifest.get("status"))
        check(checks, f"{dataset}_{seed}_pretrained_hash", manifest.get("pretrained_model_sha256") == config["pretrained_model_sha256"], manifest.get("pretrained_model_sha256"))
        check(checks, f"{dataset}_{seed}_test_selection_zero", manifest.get("test_selection_samples") == 0, manifest.get("test_selection_samples"))
        for artifact in ("E1_model_best.pt", "E2_model_best.pt", "E3_model_best.pt", "results.csv", "predictions.csv", "training_history.json", "representations.npz", "pretrained_load_report.json"):
            path = run / artifact
            check(checks, f"{dataset}_{seed}_{artifact}", path.is_file() and path.stat().st_size > 0, str(path))
        results = {row["encoder"]: row for row in load_csv(run / "results.csv")}
        predictions = load_csv(run / "predictions.csv")
        services = config["datasets"][dataset]["services"]
        service_to_id = {service: index for index, service in enumerate(services)}
        for encoder in ("E1", "E2", "E3"):
            rows = [row for row in predictions if row["role"] == "known_test" and row["encoder"] == encoder]
            y_true = np.asarray([service_to_id[row["true_service"]] for row in rows])
            y_pred = np.asarray([service_to_id[row["predicted_service"]] for row in rows])
            expected_count = config["datasets"][dataset]["expected_test"]
            check(checks, f"{dataset}_{seed}_{encoder}_test_count", len(rows) == expected_count, len(rows))
            recomputed_accuracy = float(accuracy_score(y_true, y_pred))
            recomputed_macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
            check(checks, f"{dataset}_{seed}_{encoder}_accuracy_replay", abs(recomputed_accuracy - float(results[encoder]["test_accuracy"])) < 1e-12, recomputed_accuracy)
            check(checks, f"{dataset}_{seed}_{encoder}_macro_f1_replay", abs(recomputed_macro_f1 - float(results[encoder]["test_macro_f1"])) < 1e-12, recomputed_macro_f1)
        completed.append((dataset, seed))

    check(checks, "exact_run_scope", sorted(completed) == sorted(expected_runs), completed)
    for artifact in ("run_metrics.csv", "aggregate_metrics.csv", "per_class_metrics.csv", "per_class_aggregate.csv", "confusion_matrices.csv", "validation_test_gap.csv", "e3_vs_branch_deltas.csv", "paired_vs_stage21_random3.csv", "aggregate_summary.json"):
        path = OUT / artifact
        check(checks, f"aggregate_{artifact}", path.is_file() and path.stat().st_size > 0, str(path))
    passed = all(item["passed"] for item in checks)
    result = {"status": "PASS" if passed else "FAIL", "checks": len(checks), "failures": [item for item in checks if not item["passed"]], "details": checks}
    write_json(OUT / "completion_verification.json", result)
    print(json.dumps({"status": result["status"], "checks": len(checks), "failures": len(result["failures"])}, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
