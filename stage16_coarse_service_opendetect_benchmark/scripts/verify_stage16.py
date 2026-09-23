#!/usr/bin/env python3
"""Fail-closed verification for the Stage 16 blocked audit bundle."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


OUT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    check = json.loads((OUT / "completion_verification.json").read_text(encoding="utf-8"))
    failures: list[str] = []
    if check["method_identity_gate"] != "FAIL":
        failures.append("identity gate must fail")
    if check["data_parity_gate"] != "PASS":
        failures.append("data parity gate must pass")
    manifest = rows(OUT / "matched_service_manifest.csv")
    if len(manifest) != 3065:
        failures.append(f"manifest row count={len(manifest)}")
    train = sum(row["split_role"] == "known_train" for row in manifest)
    val = sum(row["split_role"] == "known_validation" for row in manifest)
    if (train, val) != (2730, 335):
        failures.append(f"A counts={(train, val)}")
    c_train = sum(
        row["split_role"] == "known_train" and row["in_protocol_C"] == "1"
        for row in manifest
    )
    c_val = sum(
        row["split_role"] == "known_validation" and row["in_protocol_C"] == "1"
        for row in manifest
    )
    if (c_train, c_val) != (1367, 184):
        failures.append(f"C counts={(c_train, c_val)}")
    od = rows(OUT / "opendetect_service_results.csv")
    if len(od) != 3 or any(row["status"] != "NOT_RUN_IDENTITY_GATE" for row in od):
        failures.append("Open-Detect NOT_RUN markers invalid")
    if any(row["accuracy"] or row["macro_f1"] or row["weighted_f1"] for row in od):
        failures.append("fabricated Open-Detect metrics present")
    before = json.loads((OUT / "protected_asset_hashes_before.json").read_text())
    after = json.loads((OUT / "protected_asset_hashes_after.json").read_text())
    if before != after:
        failures.append("protected hash snapshots differ")
    for path, expected in after.items():
        actual = sha256_file(Path(path))
        if actual != expected:
            failures.append(f"protected asset changed: {path}")
    for name in check["required_files"]:
        if not (OUT / name).is_file():
            failures.append(f"missing required file: {name}")
    if len(list((OUT / "confusion_matrices").glob("*.csv"))) != 3:
        failures.append("expected three historical confusion matrices")
    if failures:
        raise SystemExit("FAIL\n" + "\n".join(failures))
    print("PASS")
    print("identity_gate=FAIL_EXPECTED")
    print("data_parity_gate=PASS")
    print("A=2730/335 C=1367/184")
    print("new_training_runs=0")
    print("protected_assets=UNCHANGED")


if __name__ == "__main__":
    main()
