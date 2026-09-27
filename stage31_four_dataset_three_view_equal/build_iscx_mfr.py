#!/usr/bin/env python3
"""Recover exact YaTC first-five-packet MFR for frozen Stage12 Known roles."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scapy.utils import RawPcapReader

from preflight import OUT, PROJECT, sha
from test_unlock import ensure_all_heads_frozen

sys.path.insert(0, str(PROJECT / "stage23_closed_set_method_table" / "scripts"))
from build_yatc_mfr import author_mfr_packet  # noqa: E402

STAGE12 = PROJECT / "stage12_dual_external_validation"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("iscx_vpn", "iscx_tor"), required=True)
    parser.add_argument("--phase", choices=("trainval", "test"), default="trainval")
    args = parser.parse_args()
    if args.phase == "test":
        ensure_all_heads_frozen()
    dataset = args.dataset
    result = OUT / "input_caches" / dataset / ("yatc_mfr" if args.phase == "trainval" else "yatc_mfr_test")
    if result.exists():
        raise RuntimeError(f"refusing to overwrite existing cache: {result}")
    result.mkdir(parents=True)
    split_path = STAGE12 / "protocol" / dataset / "split_manifest.csv"
    with split_path.open(newline="", encoding="utf-8") as handle:
        roles = ("known_train", "known_validation") if args.phase == "trainval" else ("known_test",)
        selected = [r for r in csv.DictReader(handle) if r["setting"] == "medium" and
                    r["role"] in roles]
    if not selected or len({r["flow_id_sha256"] for r in selected}) != len(selected):
        raise RuntimeError("missing or duplicate frozen Known flows")
    pool_candidates = list((STAGE12 / "artifacts" / dataset / "preprocessing_pool").glob("raw_prepared_parallel_v*"))
    pool_candidates = [p for p in pool_candidates if (p / "provenance_train.jsonl").is_file()]
    if len(pool_candidates) != 1:
        raise RuntimeError(f"expected exactly one source pool, found {pool_candidates}")
    pool = pool_candidates[0]
    target = {r["flow_id_sha256"]: r for r in selected}
    refs = {}
    for part in ("train", "validation", "test"):
        path = pool / f"provenance_{part}.jsonl"
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                item = json.loads(line)
                uid = item["flow_id_sha256"]
                if uid not in target:
                    continue
                if uid in refs or item["image_sha256"] != target[uid]["image_sha256"]:
                    raise RuntimeError(f"duplicate or mismatched packet provenance: {uid}")
                current = item["packet_refs"][:5]
                if not current:
                    raise RuntimeError(f"empty packet refs: {uid}")
                refs[uid] = current
    if set(refs) != set(target):
        raise RuntimeError(f"missing flow refs: {len(set(target)-set(refs))}")
    config = json.loads((STAGE12 / "configs" / "stage12_config.json").read_text())
    pcap_root = Path(config["datasets"][dataset]["pcap_root"])
    by_pcap = defaultdict(lambda: defaultdict(list))
    for uid, row in target.items():
        pcap = pcap_root / row["source_file"]
        if not pcap.is_file():
            raise FileNotFoundError(pcap)
        for slot, ref in enumerate(refs[uid]):
            by_pcap[pcap][int(ref["packet_number"])].append((row["role"], uid, slot))
    arrays, indices = {}, {}
    for role in roles:
        ids = sorted(uid for uid, row in target.items() if row["role"] == role)
        indices[role] = {uid: i for i, uid in enumerate(ids)}
        np.save(result / f"{role}_flow_ids.npy", np.asarray(ids, dtype="U80"), allow_pickle=False)
        arrays[role] = np.lib.format.open_memmap(result / f"{role}_mfr.npy", mode="w+", dtype=np.uint8,
                                                 shape=(len(ids), 40, 40))
        arrays[role][:] = 0
    found = Counter()
    captures = []
    for index, (pcap, packet_map) in enumerate(sorted(by_pcap.items(), key=lambda x: str(x[0])), 1):
        last = max(packet_map)
        scanned = 0
        with RawPcapReader(str(pcap)) as reader:
            for number, (raw, _) in enumerate(reader, 1):
                scanned = number
                for role, uid, slot in packet_map.get(number, ()):
                    content = np.frombuffer(author_mfr_packet(raw), dtype=np.uint8)
                    arrays[role][indices[role][uid]].reshape(-1)[slot*320:(slot+1)*320] = content
                    found[role, uid] += 1
                if number >= last:
                    break
        captures.append({"pcap": str(pcap), "last_packet_read": scanned,
                         "last_packet_required": last, "references": sum(map(len, packet_map.values()))})
        print(json.dumps({"dataset": dataset, "capture": index, "total_captures": len(by_pcap),
                          "last_packet_read": scanned}), flush=True)
    missing = [uid for uid, row in target.items() if found[row["role"], uid] != len(refs[uid])]
    if missing:
        (result / "failure.json").write_text(json.dumps({"missing_count": len(missing), "sample": missing[:50]}, indent=2))
        raise RuntimeError(f"MFR packet coverage failed for {len(missing)} flows")
    for array in arrays.values():
        array.flush()
    audit = {"status": "PASS", "dataset": dataset, "protocol": "medium_seed2022",
             "roles": {role: len(index) for role, index in indices.items()},
             "split_sha256": sha(split_path), "pool_provenance": str(pool),
             "first_five_packet_refs": sum(len(v) for v in refs.values()),
             "captured_pcap_count": len(captures), "capture_audit": captures,
             "known_test_values_loaded": len(selected) if args.phase == "test" else 0,
             "unknown_values_loaded": 0,
             "mfr_method": "same first-five 40x40 author rule as Stage23"}
    (result / "cache_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "dataset": dataset, "flows": len(selected),
                      "captures": len(captures)}), flush=True)


if __name__ == "__main__":
    main()
