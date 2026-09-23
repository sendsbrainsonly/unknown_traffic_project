#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


OUT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    required = [
        OUT / "config.json", OUT / "closed_service_manifest.csv",
        OUT / "loso_service_protocol_manifest.csv", OUT / "protocol_audit.json",
        OUT / "protocol_summary.json", OUT / "SUCCESS",
    ]
    for path in required:
        assert path.is_file() and path.stat().st_size > 0, path
    summary = json.loads((OUT / "protocol_summary.json").read_text(encoding="utf-8"))
    audit = json.loads((OUT / "protocol_audit.json").read_text(encoding="utf-8"))
    assert summary["status"] == "PROTOCOL_FROZEN_NOT_TRAINED"
    assert audit["status"] == "PASS"
    assert audit["protocol_count"] == 13
    for name, expected in summary["output_hashes"].items():
        assert sha256_file(OUT / name) == expected, name

    with (OUT / "loso_service_protocol_manifest.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == summary["loso_manifest_rows"]
    protocols = defaultdict(list)
    for row in rows:
        protocols[row["protocol_id"]].append(row)
    assert len(protocols) == 13
    for protocol_id, protocol in protocols.items():
        unknown_service = {row["unknown_service"] for row in protocol}
        assert len(unknown_service) == 1
        unknown_service = next(iter(unknown_service))
        assert all(
            (row["service_label"] == unknown_service) == (row["role"] == "unknown_test")
            for row in protocol
        ), protocol_id
        by_role = defaultdict(list)
        for row in protocol:
            by_role[row["role"]].append(row)
        assert set(by_role) == {"known_train", "known_validation", "known_test", "unknown_test"}
        for field in ("flow_id", "image_sha256"):
            sets = {role: {row[field] for row in values} for role, values in by_role.items()}
            roles = sorted(sets)
            for index, left in enumerate(roles):
                for right in roles[index + 1:]:
                    assert not sets[left] & sets[right], (protocol_id, field, left, right)
    verification = {
        "status": "PASS",
        "protocols": len(protocols),
        "manifest_rows": len(rows),
        "strict_unknown_free": True,
        "flow_role_overlap": 0,
        "exact_image_role_overlap": 0,
        "training_started": False,
    }
    (OUT / "completion_verification.json").write_text(
        json.dumps(verification, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
