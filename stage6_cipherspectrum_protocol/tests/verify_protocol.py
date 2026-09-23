#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


STAGE_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = STAGE_ROOT / "outputs"
ALLOWED_ROLES = {
    "KNOWN_TRAIN", "KNOWN_VALIDATION", "KNOWN_TEST", "UNKNOWN_TEST",
    "PURGED_GROUP_OVERLAP", "NOT_ACTIVE_FOR_SETTING",
}
SETTINGS = {"low": (38, 2, 42), "medium": (34, 6, 43), "high": (30, 10, 44)}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_hash_file(path: Path) -> tuple[int, list[str]]:
    failures = []
    count = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        expected, relative = line.split(maxsplit=1)
        target = (path.parent / relative.strip()).resolve()
        count += 1
        if not target.is_file() or sha256_file(target) != expected:
            failures.append(str(target))
    return count, failures


def main() -> None:
    required = [
        "corpus/canonical120k_manifest.csv", "corpus/canonical120k_summary.csv", "corpus/canonical120k.sha256",
        "protocol/cipherspectrum_open_set_protocol.md", "protocol/cipherspectrum_open_set_protocol.json",
        "protocol/preprocessing_freeze.json", "protocol/protocol_hashes.sha256", "protocol/run_metadata.json",
        "splits/low_fold.json", "splits/medium_fold.json", "splits/high_fold.json", "splits/split_manifest.csv",
        "audits/mix_exclusion_audit.md", "audits/fold_selection_audit.csv", "audits/calibration_capacity.csv",
        "audits/component_fallback_stress.csv", "audits/input_leakage_summary.md", "audits/unknown_free_ledger.md",
        "audits/split_integrity_audit.csv", "audits/final_gate.md",
    ]
    missing = [name for name in required if not (OUTPUT_ROOT / name).is_file()]
    assert not missing, f"missing outputs: {missing}"

    canonical = read_csv(OUTPUT_ROOT / "corpus/canonical120k_manifest.csv")
    assert len(canonical) == 120_000
    assert len({row["sample_id"] for row in canonical}) == 120_000
    assert all(row["split_group_id"] for row in canonical)
    assert all(row["cipher_source"] in {"aes-128-gcm", "aes-256-gcm", "chacha20-poly1305"} for row in canonical)
    assert all(row["class_name"] != "getpocket.com" for row in canonical)
    assert all("__MACOSX" not in row["pcap_path"] and not Path(row["pcap_path"]).name.startswith("._") for row in canonical)
    assert all(Path(row["pcap_path"]).is_file() for row in canonical)
    class_counts = Counter(row["class_name"] for row in canonical)
    strata = Counter((row["class_name"], row["cipher_source"]) for row in canonical)
    assert len(class_counts) == 40 and set(class_counts.values()) == {3000}
    assert len(strata) == 120 and set(strata.values()) == {1000}
    corpus_hash_count, corpus_hash_failures = verify_hash_file(OUTPUT_ROOT / "corpus/canonical120k.sha256")
    assert corpus_hash_count == 2 and not corpus_hash_failures

    protocol = json.loads((OUTPUT_ROOT / "protocol/cipherspectrum_open_set_protocol.json").read_text(encoding="utf-8"))
    assert protocol["created_before_unknown_evaluation"] is True
    assert protocol["unknown_inference_executed"] is False
    assert protocol["training_executed"] is False
    assert protocol["embedding_or_mu_extracted"] is False
    assert protocol["density_or_boundary_fit"] is False
    assert protocol["unknown_free"]["status"] == "PASS"
    assert protocol["dgsb_v1_status"] == "NOT YET CLEARED FOR EXTERNAL EVALUATION"
    assert protocol["final_gate"] == "READY"

    manifest_path = OUTPUT_ROOT / "splits/split_manifest.csv"
    role_counts = Counter()
    class_role_counts = Counter()
    source_role_counts: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    groups_by_role: dict[tuple[str, str], set[str]] = defaultdict(set)
    sample_setting_counts = Counter()
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            setting, role = row["setting"], row["role"]
            assert setting in SETTINGS and role in ALLOWED_ROLES
            role_counts[(setting, role)] += 1
            class_role_counts[(setting, row["class_name"], role)] += 1
            source_role_counts[(setting, row["class_name"], role)].add(row["cipher_source"])
            groups_by_role[(setting, role)].add(row["split_group_id"])
            sample_setting_counts[(setting, row["sample_id"])] += 1
    assert len(sample_setting_counts) == 360_000 and set(sample_setting_counts.values()) == {1}

    integrity_rows = read_csv(OUTPUT_ROOT / "audits/split_integrity_audit.csv")
    assert len(integrity_rows) == 3
    for setting, (known_n, unknown_n, seed) in SETTINGS.items():
        fold = json.loads((OUTPUT_ROOT / f"splits/{setting}_fold.json").read_text(encoding="utf-8"))
        assert fold["seed"] == seed
        assert len(fold["known_classes"]) == known_n and len(fold["unknown_classes"]) == unknown_n
        assert set(fold["known_classes"]).isdisjoint(fold["unknown_classes"])
        assert set(fold["known_classes"]) | set(fold["unknown_classes"]) == set(class_counts)
        assert role_counts[(setting, "UNKNOWN_TEST")] == unknown_n * 3000 == fold["unknown_test_count"]
        assert role_counts[(setting, "KNOWN_TRAIN")] == fold["known_train_count"]
        assert role_counts[(setting, "KNOWN_VALIDATION")] == fold["known_validation_count"]
        assert role_counts[(setting, "KNOWN_TEST")] == fold["known_test_count"]
        assert role_counts[(setting, "PURGED_GROUP_OVERLAP")] == fold["purged_known_count"]
        active_roles = ("KNOWN_TRAIN", "KNOWN_VALIDATION", "KNOWN_TEST")
        for left_index, left in enumerate(active_roles):
            for right in active_roles[left_index + 1 :]:
                assert not (groups_by_role[(setting, left)] & groups_by_role[(setting, right)])
            assert not (groups_by_role[(setting, left)] & groups_by_role[(setting, "UNKNOWN_TEST")])
        for class_name in fold["known_classes"]:
            assert class_role_counts[(setting, class_name, "KNOWN_TRAIN")] >= 500
            assert class_role_counts[(setting, class_name, "KNOWN_VALIDATION")] >= 100
            assert class_role_counts[(setting, class_name, "KNOWN_TEST")] >= 100
            assert len(source_role_counts[(setting, class_name, "KNOWN_TRAIN")]) >= 2
            assert len(source_role_counts[(setting, class_name, "KNOWN_VALIDATION")]) >= 2
            assert len(source_role_counts[(setting, class_name, "KNOWN_TEST")]) >= 2
        for class_name in fold["unknown_classes"]:
            assert class_role_counts[(setting, class_name, "UNKNOWN_TEST")] == 3000
            assert all(class_role_counts[(setting, class_name, role)] == 0 for role in active_roles)

    selection = read_csv(OUTPUT_ROOT / "audits/fold_selection_audit.csv")
    for setting in SETTINGS:
        accepted = [row for row in selection if row["setting"] == setting and row["accepted"] == "True"]
        assert len(accepted) == 1
    calibration = read_csv(OUTPUT_ROOT / "audits/calibration_capacity.csv")
    for setting in SETTINGS:
        val_summary = next(row for row in calibration if row["setting"] == setting and row["record_type"] == "DISTRIBUTION_SUMMARY" and row["distribution"] == "VALIDATION")
        test_summary = next(row for row in calibration if row["setting"] == setting and row["record_type"] == "DISTRIBUTION_SUMMARY" and row["distribution"] == "TEST")
        assert float(val_summary["min"]) >= 100 and float(test_summary["min"]) >= 100

    protocol_hash_count, protocol_hash_failures = verify_hash_file(OUTPUT_ROOT / "protocol/protocol_hashes.sha256")
    assert protocol_hash_count >= 9 and not protocol_hash_failures
    ledger = (OUTPUT_ROOT / "audits/unknown_free_ledger.md").read_text(encoding="utf-8")
    assert "UNKNOWN_FREE_AUDIT = PASS" in ledger and "FINAL TEST ONLY" in ledger
    forbidden_suffixes = {".pt", ".pth", ".bin", ".npy", ".npz", ".parquet"}
    forbidden_outputs = [path for path in STAGE_ROOT.rglob("*") if path.is_file() and path.suffix.lower() in forbidden_suffixes]
    assert not forbidden_outputs, f"model/data result artifacts unexpectedly present: {forbidden_outputs}"
    print(json.dumps({
        "verification": "PASS",
        "canonical_samples": len(canonical),
        "class_count": len(class_counts),
        "split_manifest_rows": sum(role_counts.values()),
        "protocol_hashes": protocol_hash_count,
        "corpus_hashes": corpus_hash_count,
        "unknown_inference_executed": False,
        "final_gate": "READY",
    }, indent=2))


if __name__ == "__main__":
    main()
