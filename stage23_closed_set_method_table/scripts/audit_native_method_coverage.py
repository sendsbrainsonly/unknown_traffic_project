#!/usr/bin/env python3
"""Read-only Stage23 native-input feasibility counts from frozen flow caches."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
MANIFEST = PROJECT / "stage20_dual_coarse_service_protocol" / "closed_service_manifest.csv"
FROZEN = "6404d496f50a6f7f6f397d125ad0cd653c9df152d8db1b9ed2ad2310967950bb"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if sha(MANIFEST) != FROZEN:
        raise RuntimeError("Stage20 manifest hash changed")
    rows = list(csv.DictReader(MANIFEST.open(newline="", encoding="utf-8")))
    counts = Counter()
    for row in rows:
        counts[row["dataset"], row["closed_role"], "total"] += 1
    for dataset in ("iscx_vpn", "iscx_tor"):
        audit = json.loads((ROOT / "yatc_stage20_mfr_cache" / dataset / "cache_audit_trainval.json").read_text())
        if audit["status"] != "PASS" or audit["stage20_manifest_sha256"] != FROZEN:
            raise RuntimeError("YaTC byte cache not verified")
        for role in ("known_train", "known_validation"):
            image = ROOT / "yatc_stage20_mfr_cache" / dataset / f"{role}_mfr.npy"
            if sha(image) != audit["roles"][role]["mfr_sha256"]:
                raise RuntimeError("YaTC MFR content hash changed")
            flat = np.load(image, mmap_mode="r", allow_pickle=False).reshape(-1, 1600)
            protocol = flat[:, 9]
            counts[dataset, role, "tcp_first_packet"] = int(np.count_nonzero(protocol == 6))
            counts[dataset, role, "udp_first_packet"] = int(np.count_nonzero(protocol == 17))
            counts[dataset, role, "other_first_packet"] = int(
                len(flat) - counts[dataset, role, "tcp_first_packet"] -
                counts[dataset, role, "udp_first_packet"]
            )
    packet_count_audit = json.loads((ROOT / "yatc_mfr_input_audit.json").read_text())
    result = {
        "status": "PASS_TRAIN_VAL_NATIVE_COVERAGE_AUDIT",
        "stage20_manifest_sha256": FROZEN,
        "counts": [
            {"dataset": dataset, "role": role, "metric": metric, "count": value}
            for (dataset, role, metric), value in sorted(counts.items())
        ],
        "etbert_strict_policy": "minimum 3 packets and materialized flow PCAP >= 5 KiB",
        "etbert_one_two_packet_count_all_roles": sum(
            int(packet_count_audit["packet_ref_distribution"][str(k)]) for k in (1, 2)
        ),
        "stage20_packet_ref_distribution": packet_count_audit["packet_ref_distribution"],
        "tfe_strict_policy": "TCP flow with at least one payload packet; first 50 payload packets",
        "trident_native_scope": "86 full-bidirectional-flow statistics; official USTC preprocessing unavailable",
        "note": "First-packet transport is a conservative input-coverage diagnostic, not a full TFE payload/graph audit.",
    }
    (ROOT / "native_method_coverage_audit_v2.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": result["status"], "counts": result["counts"]}))


if __name__ == "__main__":
    main()
