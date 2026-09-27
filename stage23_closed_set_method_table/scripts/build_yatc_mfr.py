#!/usr/bin/env python3
"""Reconstruct author-style YaTC 40x40 MFR from exact Stage20 packet refs.

The first five packets of each frozen flow are used, zero-padded like YaTC.
No original PCAP is modified; all outputs reside in the Stage23 bundle.
Known Test MFR is a separate post-checkpoint-selection action.
"""

from __future__ import annotations

import argparse
import binascii
import csv
import hashlib
import json
import traceback
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scapy.all import Ether, IP, IPv6, Raw
from scapy.utils import RawPcapReader


OUT = Path(__file__).resolve().parents[1]
PROJECT = OUT.parent
STAGE20 = PROJECT / "stage20_dual_coarse_service_protocol"
STAGE12 = PROJECT / "stage12_dual_external_validation"
MANIFEST = STAGE20 / "closed_service_manifest.csv"
ROOT = OUT / "yatc_stage20_mfr_cache"
FROZEN_HASH = "6404d496f50a6f7f6f397d125ad0cd653c9df152d8db1b9ed2ad2310967950bb"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def author_mfr_packet(raw_bytes: bytes) -> bytes:
    """Exact YaTC data_process.py header/payload hex rule for one IPv4 packet."""
    if raw_bytes and raw_bytes[0] >> 4 == 4:
        packet = IP(raw_bytes)
    elif raw_bytes and raw_bytes[0] >> 4 == 6:
        packet = IPv6(raw_bytes)
    else:
        packet = Ether(raw_bytes)
    if IP not in packet:
        raise ValueError("YaTC MFR requires an IPv4 IP layer")
    header = binascii.hexlify(bytes(packet[IP])).decode()
    try:
        payload = binascii.hexlify(bytes(packet[Raw])).decode()
        header = header.replace(payload, "")
    except (IndexError, KeyError):
        payload = ""
    header = (header[:160]).ljust(160, "0")
    payload = (payload[:480]).ljust(480, "0")
    result = bytes.fromhex(header + payload)
    if len(result) != 320:
        raise RuntimeError("YaTC packet representation not 320 bytes")
    return result


def main(dataset: str, phase: str) -> None:
    if sha256(MANIFEST) != FROZEN_HASH:
        raise RuntimeError("Stage20 manifest hash changed")
    previous = json.loads((OUT / "yatc_mfr_input_audit.json").read_text(encoding="utf-8"))
    if previous["status"] != "METADATA_READY_RAW_PACKET_CHECK_PENDING":
        raise RuntimeError("YaTC metadata preflight not passed")
    if phase == "test":
        successful = list((OUT / "runs" / "yatc_stage20" / dataset).glob("seed*_formal/SUCCESS"))
        if not successful:
            raise RuntimeError("Known Test MFR must wait until a complete YaTC checkpoint selection")
    roles = ("known_train", "known_validation") if phase == "trainval" else ("known_test",)
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        target = [row for row in csv.DictReader(handle) if row["dataset"] == dataset and row["closed_role"] in roles]
    target.sort(key=lambda row: (row["closed_role"], row["flow_id"]))
    if not target:
        raise RuntimeError("empty frozen target")
    by_role: dict[str, list[dict[str, str]]] = {role: [row for row in target if row["closed_role"] == role] for role in roles}
    wanted = {(row["source_pool"], row["flow_id"]): row for row in target}
    if len(wanted) != len(target):
        raise RuntimeError("duplicate frozen flow source keys")
    records: dict[tuple[str, str], list[dict]] = {}
    for pool in sorted({row["source_pool"] for row in target}):
        for path in sorted(Path(pool).glob("provenance_*.jsonl")):
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    record = json.loads(line)
                    key = (pool, record["flow_id_sha256"])
                    if key not in wanted:
                        continue
                    if key in records or record["image_sha256"] != wanted[key]["image_sha256"]:
                        raise RuntimeError(f"duplicate/mismatched provenance: {key}")
                    refs = record["packet_refs"][:5]
                    if not refs or any(int(ref["packet_number"]) <= 0 for ref in refs):
                        raise RuntimeError(f"empty/invalid packet refs: {key}")
                    records[key] = refs
    if set(records) != set(wanted):
        raise RuntimeError(f"missing provenance for {len(set(wanted) - set(records))} flows")

    cfg = json.loads((STAGE12 / "configs" / "stage12_config.json").read_text(encoding="utf-8"))
    pcap_root = Path(cfg["datasets"][dataset]["pcap_root"])
    selected: dict[Path, dict[int, list[tuple[str, str, int]]]] = defaultdict(lambda: defaultdict(list))
    for key, row in wanted.items():
        pcap = pcap_root / row["source_file"]
        if not pcap.is_file():
            raise FileNotFoundError(pcap)
        for slot, ref in enumerate(records[key]):
            selected[pcap][int(ref["packet_number"])].append((row["closed_role"], row["flow_id"], slot))

    output = ROOT / dataset
    output.mkdir(parents=True, exist_ok=True)
    for role in roles:
        for name in (f"{role}_mfr.npy", f"{role}_flow_ids.npy"):
            if (output / name).exists():
                raise FileExistsError(f"preserve existing YaTC cache: {output / name}")
    arrays = {}
    indices = {}
    try:
        for role, role_rows in by_role.items():
            ids = [row["flow_id"] for row in role_rows]
            np.save(output / f"{role}_flow_ids.npy", np.asarray(ids, dtype="U80"), allow_pickle=False)
            indices[role] = {uid: index for index, uid in enumerate(ids)}
            arrays[role] = np.lib.format.open_memmap(
                output / f"{role}_mfr.npy", mode="w+", dtype=np.uint8,
                shape=(len(role_rows), 40, 40)
            )
        found = Counter()
        invalid = []
        capture_audit = []
        for capture_index, (pcap, packet_map) in enumerate(sorted(selected.items(), key=lambda item: str(item[0])), start=1):
            max_target = max(packet_map)
            scanned = 0
            reader = RawPcapReader(str(pcap))
            try:
                for packet_number, (raw_bytes, _meta) in enumerate(reader, start=1):
                    scanned = packet_number
                    requests = packet_map.get(packet_number, ())
                    if requests:
                        try:
                            content = author_mfr_packet(raw_bytes)
                        except (ValueError, TypeError, IndexError, KeyError) as exc:
                            invalid.extend({"pcap": str(pcap), "packet_number": packet_number,
                                            "flow_id": uid, "error": str(exc)} for _role, uid, _slot in requests)
                        else:
                            bytes320 = np.frombuffer(content, dtype=np.uint8)
                            for role, uid, slot in requests:
                                arrays[role][indices[role][uid]].reshape(-1)[slot * 320:(slot + 1) * 320] = bytes320
                                found[role, uid] += 1
                    if packet_number >= max_target:
                        break
            finally:
                reader.close()
            capture_audit.append({
                "pcap": str(pcap), "pcap_size_bytes": pcap.stat().st_size,
                "packets_scanned": scanned, "last_required_packet": max_target,
                "selected_packet_numbers": len(packet_map),
                "selected_references": sum(map(len, packet_map.values())),
            })
            print(json.dumps({"event": "capture", "dataset": dataset, "phase": phase,
                              "index": capture_index, "total": len(selected),
                              "name": pcap.name, "packets_scanned": scanned}), flush=True)
        missing = [
            (row["closed_role"], row["flow_id"])
            for key, row in wanted.items() if found[row["closed_role"], row["flow_id"]] != len(records[key])
        ]
        if invalid or missing:
            write_json(output / f"packet_issues_{phase}.json", {
                "invalid_packets": len(invalid), "invalid_preview": invalid[:100],
                "incomplete_flows": len(missing), "incomplete_preview": missing[:100],
            })
            raise RuntimeError(f"YaTC MFR coverage failed: {len(invalid)} invalid packets, {len(missing)} incomplete flows")
        role_audits = {}
        for role, value in arrays.items():
            value.flush()
            arr = np.load(output / f"{role}_mfr.npy", mmap_mode="r", allow_pickle=False)
            if arr.shape != (len(by_role[role]), 40, 40):
                raise RuntimeError(f"YaTC {role} shape mismatch")
            role_audits[role] = {
                "flows": len(by_role[role]), "mfr_shape": list(arr.shape),
                "flow_ids_sha256": sha256(output / f"{role}_flow_ids.npy"),
                "mfr_sha256": sha256(output / f"{role}_mfr.npy"),
                "selected_packet_references": sum(len(records[row["source_pool"], row["flow_id"]]) for row in by_role[role]),
                "padded_flows": sum(len(records[row["source_pool"], row["flow_id"]]) < 5 for row in by_role[role]),
            }
        if sha256(MANIFEST) != FROZEN_HASH:
            raise RuntimeError("Stage20 manifest changed during YaTC MFR construction")
        report = {
            "status": "PASS", "dataset": dataset, "phase": phase,
            "representation": "YaTC original first-5-packet 40x40 MFR hex rule",
            "stage20_manifest_sha256": FROZEN_HASH,
            "roles": role_audits, "capture_audit": capture_audit,
            "raw_pcap_modified": False,
        }
        write_json(output / f"cache_audit_{phase}.json", report)
        (output / f"SUCCESS_{phase}").write_text("SUCCESS\n", encoding="utf-8")
        print(json.dumps({"status": "PASS", "dataset": dataset, "phase": phase,
                          "roles": {role: data["flows"] for role, data in role_audits.items()}}), flush=True)
    except BaseException as exc:
        write_json(output / f"FAILURE_{phase}.json", {
            "status": "FAILED", "error_type": type(exc).__name__, "error": str(exc),
            "traceback": traceback.format_exc(), "preserve_partial_artifacts": True,
        })
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("iscx_vpn", "iscx_tor"), required=True)
    parser.add_argument("--phase", choices=("trainval", "test"), default="trainval")
    args = parser.parse_args()
    main(args.dataset, args.phase)
