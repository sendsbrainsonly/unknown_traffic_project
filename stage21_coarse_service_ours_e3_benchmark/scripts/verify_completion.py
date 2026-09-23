#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from stage21_common import CONFIG, OUT, RUN_ROOT, cache_path, read_json, sha256_file, write_json


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    config = read_json(CONFIG)
    checks: list[dict] = []
    failures: list[str] = []

    def check(name: str, passed: bool, detail) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})
        if not passed:
            failures.append(name)

    expected_runs = [(dataset, int(seed)) for dataset in ("iscx_vpn", "iscx_tor") for seed in config["seeds"]]
    check("exact_formal_run_scope", len(expected_runs) == 4, expected_runs)
    check("interrupted_seed2024_excluded", 2024 not in config["seeds"], config["seeds"])
    for dataset, seed in expected_runs:
        run = RUN_ROOT / dataset / f"seed{seed}"
        check(f"{dataset}_{seed}_success_marker", (run / "SUCCESS").read_text().strip() == "SUCCESS", str(run / "SUCCESS"))
        manifest = read_json(run / "run_manifest.json")
        check(f"{dataset}_{seed}_manifest_success", manifest["status"] == "SUCCESS", manifest["status"])
        check(f"{dataset}_{seed}_test_not_used_for_selection", manifest["test_selection_samples"] == 0, manifest["test_selection_samples"])
        check(f"{dataset}_{seed}_cache_hash", manifest["source_cache_sha256"] == sha256_file(cache_path(dataset)), manifest["source_cache_sha256"])
        for name, digest in manifest["artifacts"].items():
            artifact = run / name
            check(f"{dataset}_{seed}_artifact_{name}", artifact.is_file() and sha256_file(artifact) == digest, digest)

        history = read_json(run / "training_history.json")
        results = {row["encoder"]: row for row in load_csv(run / "results.csv")}
        for encoder in ("E1", "E2", "E3"):
            best = max(history[encoder], key=lambda row: (float(row["val_macro_f1"]), -int(row["epoch"])))
            check(f"{dataset}_{seed}_{encoder}_best_epoch", int(results[encoder]["best_epoch"]) == int(best["epoch"]), {"reported": results[encoder]["best_epoch"], "recomputed": best["epoch"]})
            check(f"{dataset}_{seed}_{encoder}_best_validation_macro_f1", abs(float(results[encoder]["validation_macro_f1"]) - float(best["val_macro_f1"])) < 1e-12, {"reported": results[encoder]["validation_macro_f1"], "recomputed": best["val_macro_f1"]})

        predictions = load_csv(run / "test_predictions.csv")
        for encoder in ("E1", "E2", "E3"):
            rows = [row for row in predictions if row["encoder"] == encoder]
            truth = [row["true_service"] for row in rows]
            pred = [row["predicted_service"] for row in rows]
            recomputed_accuracy = accuracy_score(truth, pred)
            recomputed_macro_f1 = f1_score(truth, pred, average="macro", zero_division=0)
            check(f"{dataset}_{seed}_{encoder}_test_accuracy_replay", abs(float(results[encoder]["test_accuracy"]) - recomputed_accuracy) < 1e-12, recomputed_accuracy)
            check(f"{dataset}_{seed}_{encoder}_test_macro_f1_replay", abs(float(results[encoder]["test_macro_f1"]) - recomputed_macro_f1) < 1e-12, recomputed_macro_f1)

    aggregate = read_json(OUT / "aggregate_summary.json")
    multiview = read_json(OUT / "multiview_summary.json")
    check("aggregate_formal_runs", aggregate["formal_runs"] == 4, aggregate["formal_runs"])
    check("aggregate_result_rows", aggregate["result_rows"] == 12, aggregate["result_rows"])
    check("multiview_metric_families", len(multiview["metric_families"]) == 6, multiview["metric_families"])
    grid = read_json(OUT / "training_grid_execution.json")
    check("two_seed_grid_success", grid["failed"] == 0 and grid["successful_or_skipped"] == 4, grid)

    result = {"status": "PASS" if not failures else "FAIL", "checks": len(checks), "failures": failures, "details": checks}
    write_json(OUT / "completion_verification.json", result)
    print(json.dumps({"status": result["status"], "checks": len(checks), "failures": failures}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
