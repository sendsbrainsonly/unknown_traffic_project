#!/usr/bin/env python3
"""Audit whether YaTC's five-packet MFR can be built for every Stage20 flow.

Metadata/provenance only: no PCAP bytes or model/Test features are read.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


OUT = Path(__file__).resolve().parents[1]
PROJECT = OUT.parent
STAGE20 = PROJECT / "stage20_dual_coarse_service_protocol"
STAGE12 = PROJECT / "stage12_dual_external_validation"
MANIFEST = STAGE20 / "closed_service_manifest.csv"
FROZEN_HASH = "6404d496f50a6f7f6f397d125ad0cd653c9df152d8db1b9ed2ad2310967950bb"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict], names: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if sha256(MANIFEST) != FROZEN_HASH:
        raise RuntimeError("Stage20 manifest hash changed")
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 22136:
        raise RuntimeError("unexpected Stage20 row count")
    stage12_cfg = json.loads((STAGE12 / "configs" / "stage12_config.json").read_text(encoding="utf-8"))
    target = {(row["source_pool"], row["flow_id"]): row for row in rows}
    if len(target) != len(rows):
        raise RuntimeError("duplicate source-pool/flow ID")
    found: dict[tuple[str, str], list[dict]] = {}
    for pool in sorted({row["source_pool"] for row in rows}):
        for path in sorted(Path(pool).glob("provenance_*.jsonl")):
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    record = json.loads(line)
                    key = (pool, record["flow_id_sha256"])
                    if key not in target:
                        continue
                    if key in found or record["image_sha256"] != target[key]["image_sha256"]:
                        raise RuntimeError(f"duplicate or mismatched provenance: {key}")
                    found[key] = record["packet_refs"]
    if set(found) != set(target):
        raise RuntimeError(f"missing provenance for {len(set(target) - set(found))} frozen flows")
    flow_rows = []
    captures = defaultdict(lambda: {"flows": 0, "selected_refs": 0})
    for key, row in target.items():
        refs = found[key]
        if not refs or any(int(ref["packet_number"]) <= 0 for ref in refs[:5]):
            raise RuntimeError(f"missing/invalid first-five packet refs: {row['flow_id']}")
        selected = refs[:5]
        root = Path(stage12_cfg["datasets"][row["dataset"]]["pcap_root"])
        pcap = root / row["source_file"]
        stat = captures[str(pcap)]
        stat["flows"] += 1
        stat["selected_refs"] += len(selected)
        flow_rows.append({
            "dataset": row["dataset"], "flow_id": row["flow_id"],
            "role": row["closed_role"], "service": row["service_label"],
            "capture_group": row["capture_group"], "source_file": row["source_file"],
            "source_pool": row["source_pool"], "available_packet_refs": len(refs),
            "mfr_packet_refs": len(selected), "pcap_path": str(pcap),
            "pcap_exists": int(pcap.is_file()),
        })
    capture_rows = []
    for path, values in sorted(captures.items()):
        pcap = Path(path)
        capture_rows.append({
            "pcap_path": path, "exists": int(pcap.is_file()),
            "size_bytes": pcap.stat().st_size if pcap.is_file() else 0,
            **values,
        })
    missing_pcaps = [row["pcap_path"] for row in capture_rows if not row["exists"]]
    status = "METADATA_READY_RAW_PACKET_CHECK_PENDING" if not missing_pcaps else "BLOCKED_MISSING_PCAP"
    write_csv(OUT / "yatc_mfr_flow_inventory.csv", flow_rows, list(flow_rows[0]))
    write_csv(OUT / "yatc_mfr_capture_inventory.csv", capture_rows, list(capture_rows[0]))
    summary = {
        "status": status, "stage20_manifest_sha256": FROZEN_HASH,
        "flows": len(flow_rows), "captured_files": len(capture_rows),
        "capture_bytes_total": sum(row["size_bytes"] for row in capture_rows),
        "missing_pcaps": missing_pcaps,
        "packet_ref_distribution": dict(sorted(Counter(row["mfr_packet_refs"] for row in flow_rows).items())),
        "by_dataset": {name: {
            "flows": sum(row["dataset"] == name for row in flow_rows),
            "shorter_than_five": sum(row["dataset"] == name and row["mfr_packet_refs"] < 5 for row in flow_rows),
        } for name in ("iscx_vpn", "iscx_tor")},
        "raw_pcap_bytes_read": 0, "test_feature_values_read": 0, "raw_dataset_modified": False,
        "limitation": "IPv4/IP layer availability and exact author MFR bytes require a separate raw-packet reconstruction audit",
    }
    (OUT / "yatc_mfr_input_audit.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if sha256(MANIFEST) != FROZEN_HASH:
        raise RuntimeError("Stage20 manifest changed during YaTC audit")
    print(json.dumps(summary, indent=2))
    if missing_pcaps:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
