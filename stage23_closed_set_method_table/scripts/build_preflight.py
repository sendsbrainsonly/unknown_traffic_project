#!/usr/bin/env python3
"""Audit Stage20 membership and import only truly paired Stage22 closed-set rows."""

from __future__ import annotations

import csv
import hashlib
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


OUT = Path(__file__).resolve().parents[1]
PROJECT = OUT.parent
STAGE20 = PROJECT / "stage20_dual_coarse_service_protocol"
STAGE21 = PROJECT / "stage21_coarse_service_ours_e3_benchmark"
STAGE22 = PROJECT / "stage22_pretrained_trafficformer_e3_closed_set_comparison"
MANIFEST = STAGE20 / "closed_service_manifest.csv"
SEEDS = (2022, 2023)
ENCODERS = {"E1": "TrafficFormer-pretrained", "E3": "OURS-E3-T8-pretrained"}
ROLES = ("known_train", "known_validation", "known_test")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def npz_rows(path: Path) -> tuple[int, tuple[int, ...]]:
    """Read only the NPY header in the ZIP; do not load image values."""
    with zipfile.ZipFile(path) as archive, archive.open("data.npy") as member:
        version = np.lib.format.read_magic(member)
        if version == (1, 0):
            shape, _, _ = np.lib.format.read_array_header_1_0(member)
        elif version == (2, 0):
            shape, _, _ = np.lib.format.read_array_header_2_0(member)
        else:
            raise ValueError(f"unsupported NPY version in {path}: {version}")
    return int(shape[0]), tuple(map(int, shape))


def main() -> None:
    failures: list[str] = []
    checks: list[dict] = []

    def check(name: str, condition: bool, detail: object) -> None:
        checks.append({"name": name, "passed": bool(condition), "detail": detail})
        if not condition:
            failures.append(name)

    config = read_json(STAGE22 / "config.json")
    rows = read_csv(MANIFEST)
    manifest_hash = sha256_file(MANIFEST)
    check("manifest_nonempty", bool(rows), len(rows))
    check("expected_datasets", set(config["datasets"]) == {"iscx_vpn", "iscx_tor"}, list(config["datasets"]))
    check("only_closed_roles", {row["closed_role"] for row in rows} == set(ROLES), Counter(row["closed_role"] for row in rows))
    check("unique_dataset_flow_ids", len({(row["dataset"], row["flow_id"]) for row in rows}) == len(rows), len(rows))

    by_dataset: dict[str, dict[str, dict[str, dict[str, str]]]] = defaultdict(lambda: defaultdict(dict))
    hashes_by_role: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    source_indices: dict[tuple[str, str], int] = {}
    for row in rows:
        dataset, role, flow_id = row["dataset"], row["closed_role"], row["flow_id"]
        by_dataset[dataset][role][flow_id] = row
        hashes_by_role[dataset][role].add(row["image_sha256"])
        key = (row["source_pool"], row["source_role"])
        source_indices[key] = max(source_indices.get(key, -1), int(row["source_index"]))

    source_audit = []
    for (pool, source_role), max_index in sorted(source_indices.items()):
        pool_path = Path(pool).resolve()
        check(f"source_pool_inside_project_{len(source_audit)}", pool_path.is_relative_to(PROJECT), str(pool_path))
        source_file = pool_path / f"pool_{source_role}.npz"
        exists = source_file.is_file()
        check(f"source_file_exists_{len(source_audit)}", exists, str(source_file))
        count, shape = npz_rows(source_file) if exists else (0, ())
        check(f"source_index_in_bounds_{len(source_audit)}", max_index < count, {"max_index": max_index, "rows": count})
        source_audit.append({"path": str(source_file), "rows": count, "shape": shape, "max_referenced_index": max_index})

    inventory = []
    for dataset, expected in config["datasets"].items():
        role_map = by_dataset[dataset]
        check(f"{dataset}_total", sum(map(len, role_map.values())) == expected["expected_flows"], {role: len(role_map[role]) for role in ROLES})
        check(f"{dataset}_services", {row["service_label"] for role in ROLES for row in role_map[role].values()} == set(expected["services"]), expected["services"])
        for role in ROLES:
            wanted = expected["expected_" + role.removeprefix("known_")]
            check(f"{dataset}_{role}_count", len(role_map[role]) == wanted, len(role_map[role]))
            count = Counter(row["service_label"] for row in role_map[role].values())
            check(f"{dataset}_{role}_every_service", set(count) == set(expected["services"]), dict(count))
            for service in expected["services"]:
                inventory.append({"dataset": dataset, "role": role, "service": service, "flows": count[service]})
        role_hashes = hashes_by_role[dataset]
        for left_index, left in enumerate(ROLES):
            for right in ROLES[left_index + 1:]:
                overlap = role_hashes[left] & role_hashes[right]
                check(f"{dataset}_{left}_{right}_image_disjoint", not overlap, len(overlap))
        cache = read_json(STAGE21 / "feature_cache" / dataset / "cache_audit.json")
        check(f"{dataset}_cache_membership", cache["status"] == "PASS" and cache["stage20_membership_missing"] == 0 and cache["flows"] == expected["expected_flows"], cache)
        check(f"{dataset}_cache_source_hash", cache["source_hashes"]["stage20_closed_manifest"] == manifest_hash, cache["source_hashes"]["stage20_closed_manifest"])

    result_rows = []
    sample_checks = []
    for dataset, expected in config["datasets"].items():
        for seed in SEEDS:
            run = STAGE22 / "runs" / dataset / f"seed{seed}"
            run_manifest = read_json(run / "run_manifest.json")
            check(f"{dataset}_{seed}_run_success", run_manifest["status"] == "SUCCESS" and (run / "SUCCESS").is_file(), run_manifest["status"])
            check(f"{dataset}_{seed}_source_manifest", run_manifest["source_manifest_sha256"] == manifest_hash, run_manifest["source_manifest_sha256"])
            check(f"{dataset}_{seed}_test_selection_zero", run_manifest["test_selection_samples"] == 0, run_manifest["test_selection_samples"])
            pred_path = run / "predictions.csv"
            pred_hash = sha256_file(pred_path)
            check(f"{dataset}_{seed}_prediction_hash", pred_hash == run_manifest["artifacts"]["predictions.csv"], pred_hash)
            predictions = read_csv(pred_path)
            for encoder, method in ENCODERS.items():
                for role in ("known_validation", "known_test"):
                    selected = [row for row in predictions if row["encoder"] == encoder and row["role"] == role]
                    expected_map = by_dataset[dataset][role]
                    selected_ids = [row["flow_id"] for row in selected]
                    identifiers_match = len(selected_ids) == len(expected_map) and set(selected_ids) == set(expected_map)
                    labels_match = all(expected_map.get(row["flow_id"], {}).get("service_label") == row["true_service"] for row in selected)
                    check(f"{dataset}_{seed}_{encoder}_{role}_ids", identifiers_match, {"predictions": len(selected), "manifest": len(expected_map)})
                    check(f"{dataset}_{seed}_{encoder}_{role}_labels", labels_match, len(selected))
                    sample_checks.append({"dataset": dataset, "seed": seed, "method": method, "role": role, "rows": len(selected), "ids_match": identifiers_match, "labels_match": labels_match})
                    if role == "known_test" and selected:
                        labels = expected["services"]
                        accuracy = float(accuracy_score([row["true_service"] for row in selected], [row["predicted_service"] for row in selected]))
                        macro_f1 = float(f1_score([row["true_service"] for row in selected], [row["predicted_service"] for row in selected], labels=labels, average="macro", zero_division=0))
                        weighted_f1 = float(f1_score([row["true_service"] for row in selected], [row["predicted_service"] for row in selected], labels=labels, average="weighted", zero_division=0))
                        published = next(row for row in read_csv(run / "results.csv") if row["encoder"] == encoder)
                        for metric, value in (("accuracy", accuracy), ("macro_f1", macro_f1), ("weighted_f1", weighted_f1)):
                            check(f"{dataset}_{seed}_{encoder}_{metric}_replay", abs(value - float(published["test_" + metric])) < 1e-12, value)
                        result_rows.append({
                            "dataset": dataset, "seed": seed, "method": method, "source_stage": "Stage22",
                            "classes": len(labels), "train_flows": len(by_dataset[dataset]["known_train"]),
                            "validation_flows": len(by_dataset[dataset]["known_validation"]),
                            "test_flows": len(selected), "accuracy": accuracy, "macro_f1": macro_f1,
                            "weighted_f1": weighted_f1, "manifest_sha256": manifest_hash,
                            "sample_parity": "PASS" if identifiers_match and labels_match else "FAIL",
                            "checkpoint_selection": run_manifest["checkpoint_selection"],
                            "status": "REUSED_VERIFIED" if identifiers_match and labels_match else "INVALID",
                        })

    methods = [
        ("TrafficFormer", "REUSED_VERIFIED", "Stage22 E1; full Stage20 coverage and prediction parity", ""),
        ("OURS-E3-T8", "REUSED_VERIFIED", "Stage22 E3; full Stage20 coverage and prediction parity", ""),
        ("Open-Detect", "INPUT_FEASIBLE_NOT_TRAINED", "Stage12 32x32 source arrays and Stage20 source indices checked", "Build Stage20-only closed-set OD adapter; Known-Val selection"),
        ("YaTC", "ADAPTER_REQUIRED", "Existing local result uses author MFR images, not Stage20 flow IDs", "Reconstruct 40x40 MFR from exact Stage20 flows; audit coverage"),
        ("ET-BERT", "ADAPTER_REQUIRED", "Existing local packet/flow TSVs have different samples and labels", "Generate TSV for exact Stage20 flow IDs; audit packet policy"),
        ("TFE-GNN", "ADAPTER_REQUIRED", "Existing graphs use different flow/segment population", "Build header/payload graphs on exact Stage20 flows"),
        ("RoNeTC", "ADAPTER_REQUIRED", "Stage19 view cache covers most Stage20 flows; recovered TOR tail needs view parity", "Audit/recover three 8-packet views for every Stage20 flow"),
        ("Trident", "ADAPTER_REQUIRED", "Existing 86-D stats use USTC, not Stage20 flows", "Extract same 86-D features for exact Stage20 flow IDs"),
        ("UnDiff", "NOT_ELIGIBLE_NATIVE", "One-class anomaly detector, no native 6/7-way service classifier", "Keep separate anomaly-detection panel"),
    ]
    method_rows = [{"method": method, "status": status, "evidence": evidence, "next_action": action} for method, status, evidence, action in methods]
    write_csv(OUT / "sample_inventory.csv", inventory, ["dataset", "role", "service", "flows"])
    write_csv(OUT / "sample_parity.csv", sample_checks, ["dataset", "seed", "method", "role", "rows", "ids_match", "labels_match"])
    write_csv(OUT / "closed_set_run_results.csv", result_rows, [
        "dataset", "seed", "method", "source_stage", "classes", "train_flows", "validation_flows",
        "test_flows", "accuracy", "macro_f1", "weighted_f1", "manifest_sha256",
        "sample_parity", "checkpoint_selection", "status",
    ])
    write_csv(OUT / "method_readiness.csv", method_rows, ["method", "status", "evidence", "next_action"])
    report = {
        "status": "PASS" if not failures else "FAIL",
        "checks": len(checks), "failures": failures, "details": checks,
        "stage20_manifest_sha256": manifest_hash, "stage20_rows": len(rows),
        "source_npz_audit": source_audit, "verified_baseline_rows": len(result_rows),
        "new_model_runs": 0, "unknown_test_reads": 0,
    }
    write_json(OUT / "preflight_verification.json", report)
    print(json.dumps({key: report[key] for key in ("status", "checks", "failures", "stage20_rows", "verified_baseline_rows", "new_model_runs")}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
