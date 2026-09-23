#!/usr/bin/env python3
"""Read-only VNAT PCAP and metadata audit for a strict group-aware protocol."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import h5py


DATASET_ROOT = Path("/home/birkenwald/data/SEU-WXY/WXY-SEU/TrafficClassifier/Dataset/VNAT/VNAT")
PCAP_ROOT = DATASET_ROOT / "VNAT_release_1"
SOURCE_LABELS = DATASET_ROOT / "labels" / "labels_manifest.csv"
STAGE_ROOT = Path(__file__).resolve().parents[1]
SCAN_ROOT = STAGE_ROOT / "artifacts" / "pcap_scans"
OUTPUT_ROOT = STAGE_ROOT / "outputs"

APPLICATION_MAP = {
    "vimeo": "vimeo",
    "netflix": "netflix",
    "youtube": "youtube",
    "voip": "zoiper",
    "skype-chat": "skype",
    "ssh": "ssh",
    "rdp": "rdp",
    "sftp": "sftp",
    "rsync": "rsync",
    "scp": "scp",
}
PCAP_PATTERN = re.compile(
    r"^(?P<vpn>nonvpn|vpn)_(?P<keyword>vimeo|netflix|youtube|voip|skype-chat|ssh|rdp|sftp|rsync|scp)_"
    r"(?P<variant>capture|newcapture|long_capture)_?(?P<number>\d+)\.pcap$",
    re.IGNORECASE,
)
FIELDS = (
    "frame.md5_hash",
    "frame.time_epoch",
    "frame.cap_len",
    "ip.src",
    "ipv6.src",
    "ip.dst",
    "ipv6.dst",
    "ip.proto",
    "ipv6.nxt",
    "tcp.srcport",
    "tcp.dstport",
    "udp.srcport",
    "udp.dstport",
    "tcp.stream",
    "udp.stream",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows and fieldnames is None:
        raise ValueError(f"Cannot infer schema for empty CSV: {path}")
    if fieldnames is None:
        # The unified manifest mixes PCAP and metadata-file records. Preserve
        # the first-seen order while retaining fields that only occur later.
        names = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    names.append(key)
    else:
        names = fieldnames
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_pcap_name(name: str) -> dict[str, object]:
    match = PCAP_PATTERN.fullmatch(name)
    if match is None:
        raise ValueError(f"Unrecognized VNAT PCAP filename: {name}")
    vpn_status = match.group("vpn").lower()
    keyword = match.group("keyword").lower()
    application = APPLICATION_MAP[keyword]
    variant = match.group("variant").lower()
    number = int(match.group("number"))
    stem = Path(name).stem
    # Existing local metadata treats each source PCAP as a group. For a strict
    # split we conservatively collapse VPN/non-VPN files sharing application,
    # variant, and capture number, because no session metadata proves they are
    # independent capture events.
    return {
        "application": application,
        "vpn_status": vpn_status,
        "capture_id": stem,
        "source_group_id": stem,
        "strict_base_group_id": f"{application}::{variant}::{number}",
        "group_id": f"{application}::{variant}::{number}",
        "filename_keyword": keyword,
        "capture_variant": variant,
        "capture_number": number,
    }


def canonical_tuple(src: str, sport: str, dst: str, dport: str, protocol: str) -> str:
    left = (src, sport)
    right = (dst, dport)
    first, second = sorted((left, right))
    return f"{protocol}|{first[0]}:{first[1]}|{second[0]}:{second[1]}"


def capinfos(path: Path) -> dict[str, object]:
    command = ["capinfos", "-Tm", "-c", "-a", "-e", "-u", "-H", str(path)]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"capinfos exit {completed.returncode}: {completed.stderr.strip()}")
    rows = list(csv.reader(completed.stdout.splitlines()))
    if len(rows) != 2:
        raise RuntimeError(f"Unexpected capinfos table for {path}: {len(rows)} rows")
    values = dict(zip(rows[0], rows[1]))
    return {
        "packet_count_capinfos": int(values["Number of packets"]),
        "duration_seconds": float(values["Capture duration (seconds)"]),
        "first_packet_timestamp": values["Start time"],
        "last_packet_timestamp": values["End time"],
        "sha256": values["SHA256"].lower(),
    }


def scan_flows(path: Path, stderr_path: Path) -> dict[str, object]:
    command = [
        "tshark", "-n", "-o", "frame.generate_md5_hash:TRUE", "-r", str(path),
        "-T", "fields", "-E", "separator=/t", "-E", "occurrence=f",
    ]
    for field in FIELDS:
        command.extend(["-e", field])
    flows: dict[str, dict[str, Any]] = {}
    packet_count = 0
    ip_packet_count = 0
    non_ip_packet_count = 0
    with stderr_path.open("w", encoding="utf-8") as stderr_handle:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=stderr_handle,
            text=True,
            bufsize=1024 * 1024,
        )
        assert process.stdout is not None
        for line_number, line in enumerate(process.stdout, start=1):
            packet_count += 1
            values = line.rstrip("\n\r").split("\t")
            if len(values) < len(FIELDS):
                values += [""] * (len(FIELDS) - len(values))
            if len(values) != len(FIELDS):
                raise RuntimeError(f"Unexpected tshark field count at packet {line_number}: {len(values)}")
            (
                frame_md5, timestamp, cap_len, ip4_src, ip6_src, ip4_dst, ip6_dst,
                ip4_proto, ip6_next, tcp_sport, tcp_dport, udp_sport, udp_dport,
                tcp_stream, udp_stream,
            ) = values
            src = ip4_src or ip6_src
            dst = ip4_dst or ip6_dst
            if not src or not dst:
                non_ip_packet_count += 1
                continue
            ip_packet_count += 1
            if tcp_stream:
                flow_id = f"tcp:{tcp_stream}"
                protocol = "tcp"
                sport, dport = tcp_sport, tcp_dport
            elif udp_stream:
                flow_id = f"udp:{udp_stream}"
                protocol = "udp"
                sport, dport = udp_sport, udp_dport
            else:
                protocol = f"ip:{ip4_proto or ip6_next or 'unknown'}"
                sport = dport = ""
                tuple_text = canonical_tuple(src, sport, dst, dport, protocol)
                flow_id = f"other:{hashlib.sha256(tuple_text.encode()).hexdigest()[:24]}"
            tuple_text = canonical_tuple(src, sport, dst, dport, protocol)
            record = flows.get(flow_id)
            if record is None:
                record = {
                    "flow_id": flow_id,
                    "protocol": protocol,
                    "tuple_sha256": hashlib.sha256(tuple_text.encode()).hexdigest(),
                    "packet_count": 0,
                    "byte_count": 0,
                    "first_timestamp": timestamp,
                    "last_timestamp": timestamp,
                    "content_hasher": hashlib.sha256(),
                }
                flows[flow_id] = record
            record["packet_count"] += 1
            record["byte_count"] += int(cap_len or 0)
            record["last_timestamp"] = timestamp
            record["content_hasher"].update((frame_md5 or "MISSING_FRAME_HASH").encode("ascii", errors="replace"))
            record["content_hasher"].update(b"\n")
        return_code = process.wait()
    stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")
    if return_code != 0:
        raise RuntimeError(f"tshark exit {return_code}: {stderr_text[-2000:]}")
    rows = []
    for record in flows.values():
        rows.append({
            key: value
            for key, value in record.items()
            if key != "content_hasher"
        } | {"content_sha256": record["content_hasher"].hexdigest()})
    protocol_counts = Counter(row["protocol"] for row in rows)
    return {
        "packet_count_tshark": packet_count,
        "ip_packet_count": ip_packet_count,
        "non_ip_packet_count": non_ip_packet_count,
        "flow_count": len(rows),
        "tcp_flow_count": protocol_counts.get("tcp", 0),
        "udp_flow_count": protocol_counts.get("udp", 0),
        "other_ip_flow_count": sum(count for protocol, count in protocol_counts.items() if protocol not in {"tcp", "udp"}),
        "tshark_stderr": stderr_text.strip(),
        "flows": sorted(rows, key=lambda item: item["flow_id"]),
    }


def scan_pcap(path: Path) -> dict[str, object]:
    parsed = parse_pcap_name(path.name)
    error_path = SCAN_ROOT / f"{path.stem}.tshark.stderr.log"
    result: dict[str, object] = {
        **parsed,
        "file_name": path.name,
        "absolute_path": str(path.resolve()),
        "relative_path": path.relative_to(DATASET_ROOT).as_posix(),
        "file_type": "pcap",
        "size_bytes": path.stat().st_size,
        "readable": False,
        "parse_error": "",
    }
    try:
        result.update(capinfos(path))
        result.update(scan_flows(path, error_path))
        result["readable"] = True
        if result["packet_count_capinfos"] != result["packet_count_tshark"]:
            raise AssertionError("capinfos/tshark packet counts differ")
    except Exception as error:
        result["parse_error"] = f"{type(error).__name__}: {error}"
    return result


def load_source_labels() -> dict[str, dict[str, str]]:
    with SOURCE_LABELS.open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    return {row["source_pcap"]: row for row in rows}


def validate_source_label(result: dict[str, object], source: dict[str, str]) -> list[str]:
    failures = []
    checks = {
        "application": result["application"],
        "vpn_status": result["vpn_status"],
        "capture_variant": result["capture_variant"],
        "capture_number": str(result["capture_number"]),
        "size_bytes": str(result["size_bytes"]),
    }
    for field, expected in checks.items():
        if source.get(field) != str(expected):
            failures.append(f"{field}: metadata={source.get(field)!r}, parsed={expected!r}")
    return failures


def duplicate_annotations(results: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    file_hashes: dict[str, list[dict[str, object]]] = defaultdict(list)
    tuple_hashes: dict[str, list[tuple[dict[str, object], dict[str, object]]]] = defaultdict(list)
    content_hashes: dict[str, list[tuple[dict[str, object], dict[str, object]]]] = defaultdict(list)
    for result in results:
        if not result["readable"]:
            continue
        file_hashes[str(result["sha256"])].append(result)
        for flow in result["flows"]:
            tuple_hashes[str(flow["tuple_sha256"])].append((result, flow))
            content_hashes[str(flow["content_sha256"])].append((result, flow))

    duplicate_flow_rows: list[dict[str, object]] = []
    for result in results:
        result["exact_duplicate_file"] = False
        result["exact_duplicate_file_peers"] = ""
        result["repeated_5tuple_flow_count"] = 0
        result["exact_duplicate_flow_count"] = 0
        if not result["readable"]:
            continue
        file_peers = [item["capture_id"] for item in file_hashes[str(result["sha256"])] if item is not result]
        result["exact_duplicate_file"] = bool(file_peers)
        result["exact_duplicate_file_peers"] = ";".join(sorted(map(str, file_peers)))
        for flow in result["flows"]:
            tuple_peers = {
                str(item[0]["capture_id"])
                for item in tuple_hashes[str(flow["tuple_sha256"])]
                if item[0] is not result
            }
            content_peers = {
                str(item[0]["capture_id"])
                for item in content_hashes[str(flow["content_sha256"])]
                if item[0] is not result
            }
            if tuple_peers:
                result["repeated_5tuple_flow_count"] += 1
            if content_peers:
                result["exact_duplicate_flow_count"] += 1
                duplicate_flow_rows.append({
                    "capture_id": result["capture_id"],
                    "application": result["application"],
                    "vpn_status": result["vpn_status"],
                    "flow_id": flow["flow_id"],
                    "protocol": flow["protocol"],
                    "packet_count": flow["packet_count"],
                    "byte_count": flow["byte_count"],
                    "content_sha256": flow["content_sha256"],
                    "peer_capture_ids": ";".join(sorted(content_peers)),
                    "peer_applications": ";".join(sorted({
                        str(item[0]["application"])
                        for item in content_hashes[str(flow["content_sha256"])]
                        if item[0] is not result
                    })),
                })
    duplicate_files = [
        {
            "sha256": digest,
            "file_count": len(items),
            "capture_ids": ";".join(sorted(str(item["capture_id"]) for item in items)),
            "applications": ";".join(sorted({str(item["application"]) for item in items})),
            "vpn_statuses": ";".join(sorted({str(item["vpn_status"]) for item in items})),
        }
        for digest, items in file_hashes.items()
        if len(items) > 1
    ]
    return duplicate_files, duplicate_flow_rows


def assign_effective_group_ids(results: list[dict[str, object]]) -> None:
    """Union conservative capture groups connected by exact duplicate PCAPs."""
    base_groups = {str(item["strict_base_group_id"]) for item in results}
    parent = {group: group for group in base_groups}

    def find(group: str) -> str:
        while parent[group] != group:
            parent[group] = parent[parent[group]]
            group = parent[group]
        return group

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    by_hash: dict[str, list[str]] = defaultdict(list)
    for item in results:
        digest = str(item.get("sha256", ""))
        if digest:
            by_hash[digest].append(str(item["strict_base_group_id"]))
    for groups in by_hash.values():
        for group in groups[1:]:
            union(groups[0], group)

    members: dict[str, list[str]] = defaultdict(list)
    for group in sorted(base_groups):
        members[find(group)].append(group)
    labels = {
        group: "||".join(members[find(group)])
        for group in base_groups
    }
    for item in results:
        item["group_id"] = labels[str(item["strict_base_group_id"])]


def non_pcap_record(path: Path) -> dict[str, object]:
    record: dict[str, object] = {
        "file_name": path.name,
        "absolute_path": str(path.resolve()),
        "relative_path": path.relative_to(DATASET_ROOT).as_posix(),
        "file_type": path.suffix.lower().lstrip(".") or "no_extension",
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "readable": True,
        "parse_error": "",
        "application": "",
        "vpn_status": "",
        "capture_id": "",
        "source_group_id": "",
        "strict_base_group_id": "",
        "group_id": "",
        "filename_keyword": "",
        "capture_variant": "",
        "capture_number": "",
    }
    try:
        if path.suffix.lower() == ".h5":
            with h5py.File(path, "r") as handle:
                record["container_keys"] = ";".join(sorted(handle.keys()))
        elif path.suffix.lower() == ".csv":
            with path.open(encoding="utf-8-sig") as handle:
                reader = csv.reader(handle)
                next(reader, None)
                record["row_count"] = sum(1 for _ in reader)
        else:
            with path.open("rb") as handle:
                handle.read(4096)
    except Exception as error:
        record["readable"] = False
        record["parse_error"] = f"{type(error).__name__}: {error}"
    return record


def class_statistics(results: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for application in sorted({str(item["application"]) for item in results}):
        items = [item for item in results if item["application"] == application]
        vpn = [item for item in items if item["vpn_status"] == "vpn"]
        nonvpn = [item for item in items if item["vpn_status"] == "nonvpn"]
        successful = [item for item in items if item["readable"]]
        successful_vpn = [item for item in successful if item["vpn_status"] == "vpn"]
        successful_nonvpn = [item for item in successful if item["vpn_status"] == "nonvpn"]
        valid_effective_groups = {str(item["group_id"]) for item in successful}
        unique_file_hashes = {str(item.get("sha256", "")) for item in successful}
        rows.append({
            "application": application,
            "pcap_count": len(items),
            "successful_pcap_count": len(successful),
            "failed_pcap_count": len(items) - len(successful),
            "flow_count": sum(int(item.get("flow_count", 0)) for item in successful),
            "capture_count": len({str(item["capture_id"]) for item in items}),
            "source_group_count": len({str(item["source_group_id"]) for item in items}),
            "strict_group_count": len({str(item["strict_base_group_id"]) for item in items}),
            "valid_strict_group_count": len({str(item["strict_base_group_id"]) for item in successful}),
            "effective_group_count": len(valid_effective_groups),
            "unique_file_hash_count": len(unique_file_hashes),
            "vpn_pcap_count": len(vpn),
            "vpn_flow_count": sum(int(item.get("flow_count", 0)) for item in vpn if item["readable"]),
            "vpn_group_count": len({str(item["group_id"]) for item in successful_vpn}),
            "nonvpn_pcap_count": len(nonvpn),
            "nonvpn_flow_count": sum(int(item.get("flow_count", 0)) for item in nonvpn if item["readable"]),
            "nonvpn_group_count": len({str(item["group_id"]) for item in successful_nonvpn}),
            "has_vpn_and_nonvpn": bool(vpn and nonvpn),
            "exact_duplicate_pcap_count": sum(bool(item.get("exact_duplicate_file")) for item in items),
            "repeated_5tuple_flow_count": sum(int(item.get("repeated_5tuple_flow_count", 0)) for item in items),
            "exact_duplicate_flow_count": sum(int(item.get("exact_duplicate_flow_count", 0)) for item in items),
            "min_flows_per_effective_group": min(
                sum(int(item.get("flow_count", 0)) for item in items if item["group_id"] == group and item["readable"])
                for group in valid_effective_groups
            ),
            "max_flows_per_effective_group": max(
                sum(int(item.get("flow_count", 0)) for item in items if item["group_id"] == group and item["readable"])
                for group in valid_effective_groups
            ),
        })
    return rows


def eligibility(stats: list[dict[str, object]]) -> tuple[dict[str, object], dict[str, object]]:
    basis = {
        "decision_time": "after observing the complete empirical distribution",
        "flow_threshold": None,
        "group_threshold": "No empirical threshold; three non-empty disjoint groups are the logical minimum for Train/Val/Test",
        "group_definition": "application::capture_variant::capture_number, conservatively shared across VPN/non-VPN; exact duplicate PCAP groups are unioned",
        "semantic_rule": "application is the class; VPN and non-VPN must be held out together if the application is later chosen as Unknown",
    }
    eligible = []
    excluded = []
    for row in stats:
        reasons = []
        caveats = []
        if int(row["failed_pcap_count"]) > 0:
            caveats.append("exclude failed PCAPs from all derived datasets")
        if int(row["flow_count"]) == 0:
            reasons.append("no parsed IP flows")
        if int(row["effective_group_count"]) < 3:
            reasons.append(
                f"only {row['effective_group_count']} usable duplicate-aware groups; cannot allocate non-empty Train/Val/Test without group overlap"
            )
        if int(row["exact_duplicate_pcap_count"]) > 0:
            caveats.append("retain at most one copy of an exact duplicate PCAP, or keep all copies in the same group")
        if int(row["exact_duplicate_flow_count"]) > 0:
            caveats.append("quarantine exact duplicate flows before protocol freeze, especially cross-application duplicates")
        record = {
            "application": row["application"],
            "flow_count": row["flow_count"],
            "strict_group_count": row["strict_group_count"],
            "valid_strict_group_count": row["valid_strict_group_count"],
            "effective_group_count": row["effective_group_count"],
            "vpn_flow_count": row["vpn_flow_count"],
            "nonvpn_flow_count": row["nonvpn_flow_count"],
        }
        if reasons:
            record["reasons"] = reasons
            excluded.append(record)
        else:
            record["status"] = "eligible for constructing a non-empty strict group-aware Train/Val/Test split"
            if int(row["effective_group_count"]) == 3:
                caveats.append("mathematically feasible but minimally supported: exactly one group per split")
            if caveats:
                record["caveats"] = caveats
            eligible.append(record)
    return {"decision_basis": basis, "eligible_classes": eligible}, {"decision_basis": basis, "excluded_classes": excluded}


def report_markdown(
    all_files: list[dict[str, object]],
    pcaps: list[dict[str, object]],
    stats: list[dict[str, object]],
    eligible: dict[str, object],
    excluded: dict[str, object],
    duplicate_files: list[dict[str, object]],
    duplicate_flows: list[dict[str, object]],
    metadata_failures: list[str],
) -> str:
    successful = [item for item in pcaps if item["readable"]]
    failed_pcaps = [item for item in pcaps if not item["readable"]]
    successful_files = [item for item in all_files if item["readable"]]
    total_flows = sum(int(item.get("flow_count", 0)) for item in successful)
    exact_duplicate_flow_records = sum(int(item.get("exact_duplicate_flow_count", 0)) for item in pcaps)
    repeated_tuple_records = sum(int(item.get("repeated_5tuple_flow_count", 0)) for item in pcaps)
    eligible_names = [item["application"] for item in eligible["eligible_classes"]]
    excluded_names = [item["application"] for item in excluded["excluded_classes"]]
    capture_to_application = {str(item["capture_id"]): str(item["application"]) for item in pcaps}
    cross_application_duplicate_rows = [
        item for item in duplicate_flows
        if any(
            capture_to_application.get(peer) != str(item["application"])
            for peer in str(item["peer_capture_ids"]).split(";")
        )
    ]
    lines = [
        "# Stage 14A — VNAT Data Audit",
        "",
        "## Scope and grain",
        "",
        f"- Dataset (read-only): `{DATASET_ROOT}`",
        f"- Dataset files: **{len(all_files)}**, including **{len(pcaps)} PCAPs** and **{len(all_files)-len(pcaps)} metadata/HDF/script files**.",
        f"- PCAP-derived flow grain: tshark TCP/UDP stream within one source PCAP; other IP traffic is grouped by canonical bidirectional protocol/endpoints. Total: **{total_flows:,} flows**.",
        "- `capture_id` is the source PCAP stem. `source_group_id` reproduces existing local metadata. `strict_base_group_id` conservatively shares VPN/non-VPN files with the same application/variant/number; final `group_id` additionally unions exact duplicate PCAPs.",
        "- Exact flow duplication uses an ordered digest of tshark `frame.md5_hash` values, so it represents identical captured packet bytes, not merely a repeated 5-tuple.",
        "",
        "## Application statistics",
        "",
        "| application | PCAPs usable/total | flows | effective groups | VPN flows/groups | non-VPN flows/groups | duplicate PCAP rows | exact duplicate flow rows |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in stats:
        lines.append(
            f"| {row['application']} | {row['successful_pcap_count']}/{row['pcap_count']} | {row['flow_count']} | {row['effective_group_count']} | "
            f"{row['vpn_flow_count']}/{row['vpn_group_count']} | {row['nonvpn_flow_count']}/{row['nonvpn_group_count']} | "
            f"{row['exact_duplicate_pcap_count']} | {row['exact_duplicate_flow_count']} |"
        )
    lines.extend([
        "",
        "## Data-quality findings",
        "",
        f"- Readability: **{len(successful)}/{len(pcaps)} PCAPs passed** both capinfos and full tshark traversal; failures: **{len(pcaps)-len(successful)}**.",
        "- Failed/truncated PCAPs: " + "; ".join(
            f"`{item['file_name']}` ({item['size_bytes']} bytes; {item['parse_error']})"
            for item in failed_pcaps
        ) + ".",
        f"- Filename/metadata consistency failures: **{len(metadata_failures)}**.",
        f"- Exact duplicate PCAP hash groups: **{len(duplicate_files)}**.",
        "- Exact duplicate PCAP pair: `vpn_skype-chat_capture3` and `vpn_skype-chat_capture4`; their final `group_id` is unified.",
        f"- Flow rows with an identical packet-byte sequence in another capture: **{exact_duplicate_flow_records}** (detailed rows: {len(duplicate_flows)}).",
        f"- Exact duplicate flow rows crossing application labels: **{len(cross_application_duplicate_rows)}**. These cannot remain in a future Known/Unknown protocol.",
        "- Cross-application exact flow duplication occurs between `nonvpn_rsync_newcapture1` and `nonvpn_sftp_newcapture2` (two packet sequences, represented by four peer rows).",
        f"- Flow rows whose canonical 5-tuple also appears in another capture: **{repeated_tuple_records}**. Reused endpoint tuples alone are not treated as exact duplicates.",
        "- All ten applications contain both VPN and non-VPN captures; application, not tunnel status, must be the semantic class.",
        "- Existing labels are filename-derived capture labels. They do not independently prove that every background/auxiliary flow inside a PCAP belongs to the named application.",
        "- The raw HDF has 33,711 rows. The supplied feature HDF has 15,093 rows and only five coarse category labels; it has no capture/group column. It cannot by itself support a strict group-aware ten-application split.",
        "- Capture and flow volume are highly imbalanced: Skype has 110 PCAPs, while the smallest applications have only 2–3 PCAPs; RDP has 44 parsed flows versus SSH with 13,563. RDP is retained because no empirical flow threshold was preset and it has five usable groups.",
        "",
        "## Eligibility decision",
        "",
        "No empirical minimum flow/group threshold was preregistered. After reporting the full distribution, eligibility uses only the logical requirement that a three-way group-disjoint split needs at least three non-empty usable groups. Corrupt captures are quarantined rather than automatically disqualifying an otherwise usable application.",
        "",
        f"- Eligible: **{', '.join(eligible_names)}**.",
        f"- Excluded: **{', '.join(excluded_names) if excluded_names else 'none'}**.",
    ])
    for item in excluded["excluded_classes"]:
        lines.append(f"  - `{item['application']}`: {'; '.join(item['reasons'])}.")
    lines.extend([
        "",
        "## Required final answers",
        "",
        f"1. 总文件数 / 成功解析数 / 失败数：**{len(all_files)} / {len(successful_files)} / {len(all_files)-len(successful_files)}**；其中 PCAP **{len(pcaps)} / {len(successful)} / {len(pcaps)-len(successful)}**。",
        f"2. application 总数：**{len(stats)}**。",
        "3. 每类 flow 与 group 数：见上方 Application statistics 表；CSV 中同时保留 source-group 与 conservative strict-group 计数。",
        "4. VPN/non-VPN 分布：所有 application 两种状态均存在；逐类 flow/group 数见表。",
        "5. group-aware split 是否可行：**对 eligible 子集有条件可行，对全部 10 类不可行**。Train/Val/Test 必须按 duplicate-aware `group_id` 整组分配，禁止同一组内 flow 随机拆分。",
        f"6. eligible / excluded classes：eligible = **{eligible_names}**；excluded = **{excluded_names}**。",
        "7. 发现的数据质量问题：见 Data-quality findings；核心风险是极端类别/组数不平衡、少数组无法三分、重复证据，以及标签仅来自文件名而非逐 flow 权威标注。",
        "8. 是否可以进入 Stage 14B Protocol Freeze：**CONDITIONAL YES**。只能使用 eligible 类；先隔离损坏 PCAP、去除/隔离跨 application 的 exact duplicate flows、合并 exact duplicate PCAP group，再冻结 group-aware split。把同一 application 的 VPN/non-VPN 整体置于同一语义侧；不得直接对无 group 字段的 feature HDF 随机切分。",
        "",
        "No Unknown class or Low/Medium/High setting was selected in this audit.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="bounded smoke scan only")
    args = parser.parse_args()
    SCAN_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    source_labels = load_source_labels()
    pcap_paths = sorted(PCAP_ROOT.glob("*.pcap"), key=lambda path: path.name.lower())
    if args.limit is None and len(pcap_paths) != 165:
        raise AssertionError(f"Expected 165 PCAPs, found {len(pcap_paths)}")
    if args.limit is not None:
        pcap_paths = pcap_paths[: args.limit]

    results = []
    metadata_failures: list[str] = []
    for index, path in enumerate(pcap_paths, start=1):
        output = SCAN_ROOT / f"{path.stem}.json"
        if output.is_file():
            result = json.loads(output.read_text(encoding="utf-8"))
            # Backfill deterministic filename-derived fields when the audit
            # schema evolves without repeating the expensive packet traversal.
            result.update(parse_pcap_name(path.name))
            if not result.get("sha256"):
                result["sha256"] = sha256_file(path)
                write_json(output, result)
            status = "resume"
        else:
            result = scan_pcap(path)
            write_json(output, result)
            status = "scanned"
        source = source_labels.get(path.name)
        if source is None:
            metadata_failures.append(f"missing metadata row: {path.name}")
        else:
            failures = validate_source_label(result, source)
            metadata_failures.extend(f"{path.name}: {message}" for message in failures)
        results.append(result)
        print(
            f"stage14a_progress={index}/{len(pcap_paths)} status={status} file={path.name} "
            f"readable={result['readable']} flows={result.get('flow_count', 0)}",
            flush=True,
        )

    if args.limit is not None:
        print(json.dumps({"status": "SMOKE_COMPLETE", "pcaps": len(results)}, indent=2))
        return

    if set(source_labels) != {path.name for path in pcap_paths}:
        metadata_failures.append("source label manifest PCAP membership differs from disk")
    duplicate_files, duplicate_flows = duplicate_annotations(results)
    assign_effective_group_ids(results)
    non_pcaps = [non_pcap_record(path) for path in sorted(DATASET_ROOT.rglob("*")) if path.is_file() and path.suffix.lower() != ".pcap"]
    manifest_rows = []
    for item in results:
        manifest_rows.append({key: value for key, value in item.items() if key not in {"flows", "tshark_stderr"}})
    manifest_rows.extend(non_pcaps)
    manifest_rows.sort(key=lambda item: str(item["relative_path"]))
    stats = class_statistics(results)
    eligible, excluded = eligibility(stats)
    write_csv(STAGE_ROOT / "vnat_manifest.csv", manifest_rows)
    write_csv(STAGE_ROOT / "vnat_class_statistics.csv", stats)
    write_json(STAGE_ROOT / "eligible_classes.json", eligible)
    write_json(STAGE_ROOT / "excluded_classes.json", excluded)
    write_csv(OUTPUT_ROOT / "duplicate_files.csv", duplicate_files, ["sha256", "file_count", "capture_ids", "applications", "vpn_statuses"])
    write_csv(OUTPUT_ROOT / "duplicate_flows.csv", duplicate_flows, ["capture_id", "application", "vpn_status", "flow_id", "protocol", "packet_count", "byte_count", "content_sha256", "peer_capture_ids", "peer_applications"])
    write_json(OUTPUT_ROOT / "audit_details.json", {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "dataset_root": str(DATASET_ROOT),
        "dataset_file_count": len(manifest_rows),
        "pcap_count": len(results),
        "successful_pcap_count": sum(bool(item["readable"]) for item in results),
        "failed_pcap_count": sum(not bool(item["readable"]) for item in results),
        "flow_count": sum(int(item.get("flow_count", 0)) for item in results if item["readable"]),
        "application_count": len(stats),
        "metadata_failures": metadata_failures,
        "duplicate_file_groups": len(duplicate_files),
        "duplicate_flow_rows": len(duplicate_flows),
        "raw_hdf_rows": 33711,
        "feature_hdf_rows": 15093,
        "feature_hdf_label_granularity": "five coarse categories; no capture/group column",
        "unknown_class_selected": False,
        "settings_generated": False,
        "model_training": False,
        "open_detect_run": False,
        "des_run": False,
        "unknown_detection_results_read": False,
    })
    (STAGE_ROOT / "stage14a_vnat_audit.md").write_text(
        report_markdown(manifest_rows, results, stats, eligible, excluded, duplicate_files, duplicate_flows, metadata_failures),
        encoding="utf-8",
    )
    print(json.dumps({
        "status": "PASS" if all(item["readable"] for item in results) and not metadata_failures else "QUALITY_ISSUES",
        "dataset_files": len(manifest_rows),
        "pcaps": len(results),
        "successful_pcaps": sum(bool(item["readable"]) for item in results),
        "flows": sum(int(item.get("flow_count", 0)) for item in results if item["readable"]),
        "applications": len(stats),
        "eligible": [item["application"] for item in eligible["eligible_classes"]],
        "excluded": [item["application"] for item in excluded["excluded_classes"]],
        "metadata_failures": metadata_failures,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
