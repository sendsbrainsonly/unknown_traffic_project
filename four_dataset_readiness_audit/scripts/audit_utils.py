from __future__ import annotations

import csv
import hashlib
import math
import os
import struct
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, Sequence


SEED = 20260913
AUDIT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = AUDIT_ROOT.parent
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]
DATASET_ROOT = WORKSPACE_ROOT / "Dataset"
OUTPUT_ROOT = AUDIT_ROOT / "outputs"


def ensure_output_dirs() -> None:
    for name in ("ustc", "cipherspectrum", "cstnet", "cicids2017", "summary"):
        (OUTPUT_ROOT / name).mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: Iterable[Mapping[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def entropy_bits(counter: Counter[object]) -> float:
    total = sum(counter.values())
    if total == 0:
        return 0.0
    return -sum((count / total) * math.log2(count / total) for count in counter.values())


def top_share(counter: Counter[object], n: int) -> tuple[str, float]:
    total = sum(counter.values())
    if not total:
        return "", 0.0
    common = counter.most_common(n)
    return str(common[0][0]), sum(count for _, count in common) / total


def normalized_mutual_information(joint: Counter[tuple[str, str]]) -> float:
    total = sum(joint.values())
    if total == 0:
        return 0.0
    left = Counter()
    right = Counter()
    for (a, b), count in joint.items():
        left[a] += count
        right[b] += count
    mi = 0.0
    for (a, b), count in joint.items():
        p_ab = count / total
        mi += p_ab * math.log((count * total) / (left[a] * right[b]))
    h_left = -sum((count / total) * math.log(count / total) for count in left.values())
    h_right = -sum((count / total) * math.log(count / total) for count in right.values())
    denominator = math.sqrt(h_left * h_right)
    return mi / denominator if denominator else 0.0


def directory_inventory(path: Path) -> dict[str, int]:
    result = {
        "total_size": 0,
        "file_count": 0,
        "pcap_count": 0,
        "csv_count": 0,
        "parquet_count": 0,
        "other_count": 0,
    }
    for root, _, files in os.walk(path):
        for filename in files:
            item = Path(root) / filename
            try:
                result["total_size"] += item.stat().st_size
            except OSError:
                continue
            result["file_count"] += 1
            suffix = item.suffix.lower()
            if suffix in {".pcap", ".pcapng"}:
                result["pcap_count"] += 1
            elif suffix == ".csv":
                result["csv_count"] += 1
            elif suffix == ".parquet":
                result["parquet_count"] += 1
            else:
                result["other_count"] += 1
    return result


def _pcap_format(magic: bytes) -> tuple[str, float] | None:
    formats = {
        b"\xd4\xc3\xb2\xa1": ("<", 1e-6),
        b"\xa1\xb2\xc3\xd4": (">", 1e-6),
        b"\x4d\x3c\xb2\xa1": ("<", 1e-9),
        b"\xa1\xb2\x3c\x4d": (">", 1e-9),
    }
    return formats.get(magic)


def _frame_endpoints(raw: bytes) -> tuple[bytes, bytes] | None:
    """Return directional network+transport endpoints without third-party parsing."""
    if len(raw) < 14:
        return None
    offset = 14
    ethertype = int.from_bytes(raw[12:14], "big")
    while ethertype in {0x8100, 0x88A8} and len(raw) >= offset + 4:
        ethertype = int.from_bytes(raw[offset + 2 : offset + 4], "big")
        offset += 4
    if ethertype == 0x0800 and len(raw) >= offset + 20:
        ihl = (raw[offset] & 0x0F) * 4
        if ihl < 20 or len(raw) < offset + ihl:
            return None
        protocol = raw[offset + 9]
        src_ip, dst_ip = raw[offset + 12 : offset + 16], raw[offset + 16 : offset + 20]
        transport = offset + ihl
    elif ethertype == 0x86DD and len(raw) >= offset + 40:
        protocol = raw[offset + 6]
        src_ip, dst_ip = raw[offset + 8 : offset + 24], raw[offset + 24 : offset + 40]
        transport = offset + 40
    else:
        return None
    if protocol in {6, 17} and len(raw) >= transport + 4:
        src_port, dst_port = raw[transport : transport + 2], raw[transport + 2 : transport + 4]
    else:
        src_port = dst_port = b"\x00\x00"
    return bytes(src_ip) + bytes(src_port), bytes(dst_ip) + bytes(dst_port)


def scan_classic_pcap(path: Path, fingerprint_packets: int = 8) -> dict[str, object]:
    """One-pass SHA, structural check, and metadata-free first-N fingerprint."""
    digest = hashlib.sha256()
    first_entries: list[bytes] = []
    packet_count = 0
    first_timestamp: float | None = None
    last_timestamp: float | None = None
    initiator: bytes | None = None
    readable = False
    error = ""
    try:
        with path.open("rb") as handle:
            global_header = handle.read(24)
            digest.update(global_header)
            if len(global_header) != 24:
                raise ValueError("truncated_global_header")
            fmt = _pcap_format(global_header[:4])
            if fmt is None:
                raise ValueError(f"unsupported_magic:{global_header[:4].hex()}")
            endian, scale = fmt
            first_packet_ts: float | None = None
            while True:
                header = handle.read(16)
                if not header:
                    break
                digest.update(header)
                if len(header) != 16:
                    raise ValueError("truncated_packet_header")
                ts_sec, ts_frac, incl_len, orig_len = struct.unpack(endian + "IIII", header)
                if incl_len > 256 * 1024 * 1024:
                    raise ValueError(f"implausible_incl_len:{incl_len}")
                raw = handle.read(incl_len)
                digest.update(raw)
                if len(raw) != incl_len:
                    raise ValueError("truncated_packet_data")
                timestamp = ts_sec + ts_frac * scale
                if first_timestamp is None:
                    first_timestamp = timestamp
                    first_packet_ts = timestamp
                last_timestamp = timestamp
                if packet_count < fingerprint_packets:
                    delta_ns = int(round((timestamp - float(first_packet_ts)) * 1_000_000_000))
                    content_hash = hashlib.sha256(raw).digest()
                    endpoints = _frame_endpoints(raw)
                    if endpoints is None:
                        direction = 0
                    else:
                        source, destination = endpoints
                        if initiator is None:
                            initiator = source
                        direction = 1 if source == initiator else -1 if destination == initiator else 0
                    first_entries.append(
                        struct.pack(">IIqb", incl_len, orig_len, delta_ns, direction) + content_hash
                    )
                packet_count += 1
            readable = True
    except (OSError, ValueError, struct.error) as exc:
        error = str(exc)
        try:
            full_sha = sha256_file(path)
        except OSError:
            full_sha = ""
    else:
        full_sha = digest.hexdigest()
    normalized = hashlib.sha256()
    normalized.update(struct.pack(">Q", packet_count))
    for entry in first_entries:
        normalized.update(entry)
    return {
        "absolute_path": str(path.resolve()),
        "size_bytes": path.stat().st_size if path.exists() else 0,
        "sha256": full_sha,
        "readable": readable,
        "packet_count": packet_count,
        "first_packet_timestamp": first_timestamp,
        "last_packet_timestamp": last_timestamp,
        "normalized_fingerprint": normalized.hexdigest() if readable else "",
        "error": error,
    }
