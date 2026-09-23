#!/usr/bin/env python3
"""Build a protocol-neutral, flow-ID-aligned Open-Detect image cache for VNAT.

The cache is preprocessing evidence only.  Per-run builders copy only frozen
Known Train/Validation rows into model-visible arrays.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import zip_longest
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap
from scapy.utils import RawPcapReader

from common import (
    BYTES_PER_PACKET,
    CACHE_ROOT,
    EXPECTED_FREEZE_HASH,
    FLOW_BYTES,
    IMAGE_SIDE,
    PACKETS_PER_FLOW,
    encode_packet,
    iter_split_rows,
    sha256_file,
    tshark_flow_id,
    verify_freeze,
)


TSHARK_FIELDS = (
    "ip.src", "ipv6.src", "ip.dst", "ipv6.dst", "ip.proto", "ipv6.nxt",
    "tcp.srcport", "tcp.dstport", "udp.srcport", "udp.dstport",
    "tcp.stream", "udp.stream",
)


def ethernet_compatible_frame(raw_frame: bytes) -> bytes:
    """Wrap raw-IP captures for the existing Ethernet-expecting adapter.

    VNAT release PCAPs use DLT_RAW.  The Open-Detect adapter discards the
    Ethernet header and encodes the IP packet, so a zero-address Ethernet
    envelope preserves every model-visible IP/payload byte.
    """
    if not raw_frame:
        raise ValueError("empty raw frame")
    version = raw_frame[0] >> 4
    if version == 4:
        return b"\x00" * 12 + b"\x08\x00" + raw_frame
    if version == 6:
        return b"\x00" * 12 + b"\x86\xdd" + raw_frame
    return raw_frame


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_clean_flow_rows() -> list[dict[str, str]]:
    # Every frozen protocol contains every clean flow exactly once.  Reading one
    # protocol avoids treating the 15 protocol expansions as 351,735 samples.
    rows = list(iter_split_rows("low_seed2022"))
    if len(rows) != 23_449:
        raise RuntimeError(f"expected 23,449 clean flows, found {len(rows)}")
    uids = [row["flow_uid"] for row in rows]
    if len(set(uids)) != len(uids):
        raise RuntimeError("duplicate flow_uid in frozen clean-pool projection")
    return sorted(rows, key=lambda row: row["flow_uid"])


def scan_capture(
    capture_id: str,
    pcap_path: Path,
    targets: dict[str, dict[str, object]],
    flat_images: np.memmap,
    packets_used: np.memmap,
    observed_counts: Counter[str],
    failures: list[dict[str, object]],
    stderr_path: Path,
) -> dict[str, object]:
    command = [
        "tshark", "-n", "-r", str(pcap_path), "-T", "fields",
        "-E", "separator=/t", "-E", "occurrence=f",
    ]
    for field in TSHARK_FIELDS:
        command.extend(["-e", field])
    packet_rows = 0
    sentinel = object()
    with stderr_path.open("w", encoding="utf-8") as stderr_handle:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=stderr_handle,
            text=True,
            bufsize=1024 * 1024,
        )
        assert process.stdout is not None
        with RawPcapReader(str(pcap_path)) as raw_reader:
            for packet_rows, pair in enumerate(
                zip_longest(process.stdout, raw_reader, fillvalue=sentinel), start=1
            ):
                line, raw_entry = pair
                if line is sentinel or raw_entry is sentinel:
                    process.kill()
                    raise RuntimeError(
                        f"{capture_id}: tshark/RawPcapReader packet-row count mismatch at {packet_rows}"
                    )
                values = str(line).rstrip("\r\n").split("\t")
                if len(values) < len(TSHARK_FIELDS):
                    values += [""] * (len(TSHARK_FIELDS) - len(values))
                flow_id = tshark_flow_id(values)
                if flow_id is None or flow_id not in targets:
                    continue
                observed_counts[f"{capture_id}|{flow_id}"] += 1
                target = targets[flow_id]
                index = int(target["cache_index"])
                used = int(packets_used[index])
                if used >= PACKETS_PER_FLOW:
                    continue
                raw_frame = raw_entry[0]
                try:
                    encoded = encode_packet(ethernet_compatible_frame(bytes(raw_frame)))
                except Exception as exc:
                    # The released Open-Detect reader catches any per-packet
                    # preprocessing exception and emits an all-zero 128-byte
                    # packet. Preserve that behavior while auditing each use.
                    failures.append({
                        "capture_id": capture_id,
                        "source_flow_id": flow_id,
                        "flow_uid": target["flow_uid"],
                        "packet_ordinal_in_flow": used + 1,
                        "error": f"released_zero_fill: {type(exc).__name__}: {exc}",
                    })
                    encoded = np.zeros(BYTES_PER_PACKET, dtype=np.uint8)
                start = used * BYTES_PER_PACKET
                flat_images[index, start : start + BYTES_PER_PACKET] = encoded
                packets_used[index] = used + 1
        returncode = process.wait()
    if returncode != 0:
        tail = stderr_path.read_text(encoding="utf-8", errors="replace")[-2000:]
        raise RuntimeError(f"{capture_id}: tshark exit {returncode}: {tail}")
    return {
        "capture_id": capture_id,
        "pcap_path": str(pcap_path),
        "target_flows": len(targets),
        "packet_rows": packet_rows,
        "tshark_stderr": str(stderr_path),
    }


def main() -> None:
    if CACHE_ROOT.exists():
        raise RuntimeError(f"refusing to overwrite existing cache: {CACHE_ROOT}")
    CACHE_ROOT.mkdir(parents=True)
    stderr_root = CACHE_ROOT / "tshark_stderr"
    stderr_root.mkdir()
    freeze = verify_freeze()
    rows = load_clean_flow_rows()

    images_path = CACHE_ROOT / "images.npy"
    flat_images = open_memmap(
        images_path, mode="w+", dtype=np.uint8, shape=(len(rows), FLOW_BYTES)
    )
    flat_images[:] = 0
    packets_used = open_memmap(
        CACHE_ROOT / "packets_used.npy", mode="w+", dtype=np.uint8, shape=(len(rows),)
    )
    packets_used[:] = 0

    by_capture: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    capture_paths: dict[str, Path] = {}
    for index, row in enumerate(rows):
        flow_id = row["source_flow_id"]
        capture_id = row["capture_id"]
        if flow_id in by_capture[capture_id]:
            raise RuntimeError(f"duplicate source flow id in capture: {capture_id}/{flow_id}")
        by_capture[capture_id][flow_id] = {
            "cache_index": index,
            "flow_uid": row["flow_uid"],
            "expected_packet_count": int(row["packet_count"]),
        }
        capture_paths[capture_id] = Path(row["source_pcap_path"])

    failures: list[dict[str, object]] = []
    observed_counts: Counter[str] = Counter()
    capture_audit = []
    for capture_number, capture_id in enumerate(sorted(by_capture), start=1):
        pcap_path = capture_paths[capture_id]
        if not pcap_path.is_file():
            raise FileNotFoundError(pcap_path)
        capture_audit.append(
            scan_capture(
                capture_id,
                pcap_path,
                by_capture[capture_id],
                flat_images,
                packets_used,
                observed_counts,
                failures,
                stderr_root / f"{capture_id}.log",
            )
        )
        print(json.dumps({
            "event": "capture_complete",
            "capture": capture_id,
            "capture_number": capture_number,
            "capture_total": len(by_capture),
            "target_flows": len(by_capture[capture_id]),
            "encoding_failures": len(failures),
        }, sort_keys=True), flush=True)

    flat_images.flush()
    packets_used.flush()
    count_mismatches = []
    cache_manifest = []
    for index, row in enumerate(rows):
        key = f"{row['capture_id']}|{row['source_flow_id']}"
        observed = observed_counts[key]
        expected = int(row["packet_count"])
        if observed != expected:
            count_mismatches.append({
                "flow_uid": row["flow_uid"], "capture_id": row["capture_id"],
                "source_flow_id": row["source_flow_id"], "expected": expected, "observed": observed,
            })
        image = np.asarray(flat_images[index]).reshape(IMAGE_SIDE, IMAGE_SIDE)
        cache_manifest.append({
            "cache_index": index,
            "flow_uid": row["flow_uid"],
            "application": row["application"],
            "vpn_status": row["vpn_status"],
            "capture_id": row["capture_id"],
            "source_flow_id": row["source_flow_id"],
            "packet_count_expected": expected,
            "packet_count_observed": observed,
            "packets_used": int(packets_used[index]),
            "image_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
            "source_pcap_path": row["source_pcap_path"],
        })
    np.save(CACHE_ROOT / "flow_uids.npy", np.asarray([row["flow_uid"] for row in rows], dtype="U64"), allow_pickle=False)
    np.save(CACHE_ROOT / "applications.npy", np.asarray([row["application"] for row in rows], dtype="U16"), allow_pickle=False)
    write_csv(CACHE_ROOT / "cache_manifest.csv", cache_manifest)
    write_csv(CACHE_ROOT / "capture_audit.csv", capture_audit)
    failure_fields = ["capture_id", "source_flow_id", "flow_uid", "packet_ordinal_in_flow", "error"]
    with (CACHE_ROOT / "encoding_failures.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=failure_fields)
        writer.writeheader()
        writer.writerows(failures)
    mismatch_fields = ["flow_uid", "capture_id", "source_flow_id", "expected", "observed"]
    with (CACHE_ROOT / "packet_count_mismatches.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=mismatch_fields)
        writer.writeheader()
        writer.writerows(count_mismatches)

    zero_packet_flows = int(np.count_nonzero(np.asarray(packets_used) == 0))
    audit = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if not count_mismatches and zero_packet_flows == 0 else "FAIL",
        "scope": "protocol-neutral cache; not a model fit and never directly exposed wholesale to a run",
        "freeze_hash": freeze["freeze_hash"],
        "clean_flow_count": len(rows),
        "capture_count": len(by_capture),
        "released_zero_fill_packet_fallbacks": len(failures),
        "packet_count_mismatches": len(count_mismatches),
        "zero_packet_flows": zero_packet_flows,
        "flows_with_fewer_than_8_packets": int(np.count_nonzero(np.asarray(packets_used) < PACKETS_PER_FLOW)),
        "representation": "first 8 packets; each packet 80 IP/header bytes + 48 Raw payload bytes; IPv4 src/dst zeroed; zero padding",
        "flow_identity": "exact Stage 14A tshark tcp.stream/udp.stream/other-IP grouping",
        "images_sha256": sha256_file(images_path),
        "flow_uids_sha256": sha256_file(CACHE_ROOT / "flow_uids.npy"),
        "cache_manifest_sha256": sha256_file(CACHE_ROOT / "cache_manifest.csv"),
        "unknown_role_used_for_model_fit": False,
        "known_test_role_used_for_model_fit": False,
    }
    (CACHE_ROOT / "cache_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    if audit["status"] != "PASS":
        raise RuntimeError(f"flow image cache audit failed: {audit}")
    print(json.dumps(audit, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
