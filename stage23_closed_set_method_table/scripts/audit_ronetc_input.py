#!/usr/bin/env python3
"""Read-only-source audit of exact Stage20 flows against Stage19 RoNeTC views."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


OUT = Path(__file__).resolve().parents[1]
PROJECT = OUT.parent
STAGE19 = PROJECT / "stage19_ronetc_four_dataset_comparison"
STAGE12 = PROJECT / "stage12_dual_external_validation"
STAGE20 = PROJECT / "stage20_dual_coarse_service_protocol"
FROZEN_HASH = "6404d496f50a6f7f6f397d125ad0cd653c9df152d8db1b9ed2ad2310967950bb"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    manifest = STAGE20 / "closed_service_manifest.csv"
    if file_sha256(manifest) != FROZEN_HASH:
        raise RuntimeError("Stage20 frozen manifest hash changed")
    rows = csv_rows(manifest)
    if len(rows) != 22136:
        raise RuntimeError("unexpected Stage20 row count")
    config = json.loads((STAGE12 / "configs" / "stage12_config.json").read_text(encoding="utf-8"))
    coverage, missing_rows, audits = [], [], {}
    for dataset in ("iscx_vpn", "iscx_tor"):
        target = [row for row in rows if row["dataset"] == dataset]
        cache = STAGE19 / "byte_cache" / dataset
        old_audit = json.loads((cache / "cache_audit.json").read_text(encoding="utf-8"))
        id_path, view_path = cache / "flow_uids.npy", cache / "views.npy"
        if old_audit["status"] != "PASS" or file_sha256(id_path) != old_audit["flow_uids_sha256"] or file_sha256(view_path) != old_audit["views_sha256"]:
            raise RuntimeError(f"Stage19 cache integrity failed for {dataset}")
        ids = [str(value) for value in np.load(id_path, mmap_mode="r", allow_pickle=False)]
        shape = np.load(view_path, mmap_mode="r", allow_pickle=False).shape
        if len(set(ids)) != len(ids) or shape != (len(ids), 3, 8, 64):
            raise RuntimeError(f"Stage19 cache shape/IDs invalid for {dataset}")
        cached = set(ids)
        missing = [row for row in target if row["flow_id"] not in cached]
        by_group = defaultdict(lambda: [0, 0])
        for row in target:
            key = (row["closed_role"], row["service_label"])
            by_group[key][0] += 1
            by_group[key][1] += int(row["flow_id"] in cached)
        for (role, service), (total, hit) in sorted(by_group.items()):
            coverage.append({
                "dataset": dataset, "role": role, "service": service,
                "stage20_flows": total, "cached_flows": hit, "missing_flows": total - hit,
                "coverage_ratio": hit / total,
            })
        missing_ids = {row["flow_id"] for row in missing}
        provenance = {}
        pools = {Path(row["source_pool"]) for row in missing}
        for pool in sorted(pools):
            for path in sorted(pool.glob("provenance_*.jsonl")):
                with path.open(encoding="utf-8") as handle:
                    for line in handle:
                        record = json.loads(line)
                        flow_id = record["flow_id_sha256"]
                        if flow_id in missing_ids:
                            if flow_id in provenance:
                                raise RuntimeError(f"duplicate provenance for {flow_id}")
                            provenance[flow_id] = record
        if set(provenance) != missing_ids:
            raise RuntimeError(f"{dataset}: {len(missing_ids - set(provenance))} missing provenance records")
        pcap_root = Path(config["datasets"][dataset]["pcap_root"])
        missing_pcaps = set()
        for row in missing:
            record = provenance[row["flow_id"]]
            refs = record["packet_refs"][:8]
            if record["image_sha256"] != row["image_sha256"] or not refs:
                raise RuntimeError(f"provenance/image mismatch for {row['flow_id']}")
            if any(int(ref["packet_number"]) < 1 for ref in refs):
                raise RuntimeError(f"invalid packet index for {row['flow_id']}")
            pcap = pcap_root / row["source_file"]
            if not pcap.is_file():
                missing_pcaps.add(str(pcap))
            missing_rows.append({
                "dataset": dataset, "flow_id": row["flow_id"],
                "role": row["closed_role"], "service": row["service_label"],
                "source_pool": str(Path(row["source_pool"]).name),
                "source_file": row["source_file"], "pcap_path": str(pcap),
                "pcap_readable": int(pcap.is_file()),
                "packet_refs": len(refs), "first_packet_number": int(refs[0]["packet_number"]),
                "last_packet_number": int(refs[-1]["packet_number"]),
            })
        audits[dataset] = {
            "stage20_flows": len(target), "cached_flows": len(target) - len(missing),
            "missing_flows": len(missing), "cache_flow_count": len(ids),
            "cache_shape": shape, "cache_integrity": "PASS",
            "missing_provenance_records": 0, "missing_pcap_paths": sorted(missing_pcaps),
            "missing_role_counts": dict(Counter(row["closed_role"] for row in missing)),
            "missing_service_counts": dict(Counter(row["service_label"] for row in missing)),
        }
    status = "PASS_FULL_COVERAGE" if not missing_rows else "RECOVERABLE_GAP" if all(
        row["pcap_readable"] for row in missing_rows
    ) else "BLOCKED_MISSING_PCAP"
    write_csv(OUT / "ronetc_input_coverage.csv", coverage,
              ["dataset", "role", "service", "stage20_flows", "cached_flows", "missing_flows", "coverage_ratio"])
    write_csv(OUT / "ronetc_missing_flows.csv", missing_rows,
              ["dataset", "flow_id", "role", "service", "source_pool", "source_file", "pcap_path",
               "pcap_readable", "packet_refs", "first_packet_number", "last_packet_number"])
    payload = {
        "status": status, "stage20_manifest_sha256": FROZEN_HASH,
        "datasets": audits, "raw_pcap_reads": 0, "raw_dataset_modified": False,
        "next_step": "Recover only the missing flows from exact Stage12 packet_refs before formal training",
    }
    (OUT / "ronetc_input_audit.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if file_sha256(manifest) != FROZEN_HASH:
        raise RuntimeError("Stage20 manifest changed during audit")
    print(json.dumps(payload, indent=2))
    if status == "BLOCKED_MISSING_PCAP":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
