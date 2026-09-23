#!/usr/bin/env python3
"""Reconstruct the frozen VNAT Stage 14B flow lineage without changing it."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import pickle
import re
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np


PROJECT = Path(__file__).resolve().parents[2]
AUDIT = PROJECT / "stage14b5_vnat_flow_retention_audit"
ARTIFACTS = AUDIT / "artifacts"
STAGE14A = PROJECT / "stage14a_vnat_data_audit"
STAGE14B = PROJECT / "stage14b_vnat_protocol_freeze"
DATASET = Path(
    "/home/birkenwald/data/SEU-WXY/WXY-SEU/TrafficClassifier/Dataset/VNAT/VNAT"
)
MANIFEST = STAGE14A / "vnat_manifest.csv"
SCAN_ROOT = STAGE14A / "artifacts" / "pcap_scans"
CLASS_AUDIT = STAGE14B / "vnat_final_class_audit.csv"
DUPLICATE_FILES = STAGE14A / "outputs" / "duplicate_files.csv"
DUPLICATE_FLOWS = STAGE14A / "outputs" / "duplicate_flows.csv"
PROTOCOL = STAGE14B / "vnat_open_set_protocol.json"
SPLIT = STAGE14B / "vnat_split_manifest.csv"
FREEZE_HASHES = STAGE14B / "outputs" / "freeze_hashes.json"
OFFICIAL_H5 = DATASET / "VNAT_Dataframe_release_1.h5"
CORRUPT_PREFIX = ARTIFACTS / "corrupt_prefix_flow_counts.json"
EXPECTED_FREEZE_HASH = "c904daef04d2c8e79469191121325c0a6ceeaeb3e595ef9c4c7d7a62c980ae2f"


class NumpyOnlyUnpickler(pickle.Unpickler):
    """Restricted decoder for the ndarray pickle in the official HDF5 file."""

    ALLOWED = {
        ("numpy", "dtype"),
        ("numpy", "ndarray"),
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy.core.multiarray", "scalar"),
        ("numpy._core.multiarray", "scalar"),
    }

    def find_class(self, module: str, name: str):  # type: ignore[override]
        if (module, name) not in self.ALLOWED:
            raise pickle.UnpicklingError(f"blocked global: {module}.{name}")
        imported = __import__(module, fromlist=[name])
        return getattr(imported, name)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def freeze_state() -> dict[str, str]:
    recorded = json.loads(FREEZE_HASHES.read_text(encoding="utf-8"))
    state = {
        "freeze_hash": str(recorded["freeze_hash"]),
        "vnat_open_set_protocol_sha256": sha256_file(PROTOCOL),
        "vnat_split_manifest_sha256": sha256_file(SPLIT),
    }
    if state["freeze_hash"] != EXPECTED_FREEZE_HASH:
        raise AssertionError(f"Unexpected freeze hash: {state['freeze_hash']}")
    if state["vnat_open_set_protocol_sha256"] != recorded["vnat_open_set_protocol_sha256"]:
        raise AssertionError("Protocol JSON differs from frozen recorded hash")
    if state["vnat_split_manifest_sha256"] != recorded["vnat_split_manifest_sha256"]:
        raise AssertionError("Split manifest differs from frozen recorded hash")
    return state


def load_official_counts(pcap_by_file: dict[str, dict[str, str]]) -> tuple[Counter[str], list[dict[str, Any]]]:
    with h5py.File(OFFICIAL_H5, "r") as handle:
        raw = handle["data/block0_values"][0].tobytes()
    decoded = NumpyOnlyUnpickler(io.BytesIO(raw)).load()
    if not isinstance(decoded, np.ndarray) or decoded.shape != (33711, 5):
        raise AssertionError(f"Unexpected official dataframe shape: {getattr(decoded, 'shape', None)}")
    counts = Counter(str(value) for value in decoded[:, 4])
    unknown_names = sorted(set(counts) - set(pcap_by_file))
    if unknown_names:
        raise AssertionError(f"Official H5 contains unmapped PCAP names: {unknown_names}")
    rows = []
    for file_name in sorted(pcap_by_file):
        pcap = pcap_by_file[file_name]
        scan_path = SCAN_ROOT / f"{pcap['capture_id']}.json"
        scan = json.loads(scan_path.read_text(encoding="utf-8"))
        rows.append({
            "application": pcap["application"],
            "vpn_status": pcap["vpn_status"],
            "capture_id": pcap["capture_id"],
            "file_name": file_name,
            "pcap_readable_in_stage14a": pcap["readable"],
            "official_connection_count": counts[file_name],
            "our_stage14a_flow_count": int(scan.get("flow_count", 0)),
            "official_minus_our": counts[file_name] - int(scan.get("flow_count", 0)),
            "comparison_note": (
                "failed PCAP: local Stage 14A contribution is zero; official count is not the same flow definition"
                if pcap["readable"] != "True"
                else "definition comparison only; official connection construction is not the local tshark stream construction"
            ),
        })
    if sum(counts.values()) != 33711:
        raise AssertionError("Official H5 connection counts do not sum to 33,711")
    return counts, rows


def main() -> None:
    AUDIT.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    before = freeze_state()
    (ARTIFACTS / "freeze_hash_before.json").write_text(
        json.dumps({"checked_at_utc": utc_now(), **before}, indent=2) + "\n",
        encoding="utf-8",
    )

    manifest = read_csv(MANIFEST)
    pcaps = [row for row in manifest if row["file_type"] == "pcap"]
    pcap_by_file = {row["file_name"]: row for row in pcaps}
    readable = [row for row in pcaps if row["readable"] == "True"]
    failed = [row for row in pcaps if row["readable"] != "True"]
    if (len(pcaps), len(readable), len(failed)) != (165, 163, 2):
        raise AssertionError("Unexpected PCAP inventory counts")

    scans: dict[str, dict[str, Any]] = {
        row["capture_id"]: json.loads((SCAN_ROOT / f"{row['capture_id']}.json").read_text(encoding="utf-8"))
        for row in pcaps
    }
    successful_packet_count = sum(int(scans[row["capture_id"]]["packet_count_tshark"]) for row in readable)
    successful_ip_packets = sum(int(scans[row["capture_id"]]["ip_packet_count"]) for row in readable)
    successful_non_ip_packets = sum(int(scans[row["capture_id"]]["non_ip_packet_count"]) for row in readable)
    stage14a_flows = sum(int(scans[row["capture_id"]]["flow_count"]) for row in readable)
    tcp_flows = sum(int(scans[row["capture_id"]]["tcp_flow_count"]) for row in readable)
    udp_flows = sum(int(scans[row["capture_id"]]["udp_flow_count"]) for row in readable)
    other_ip_flows = sum(int(scans[row["capture_id"]]["other_ip_flow_count"]) for row in readable)
    all_readable_flows = [flow for row in readable for flow in scans[row["capture_id"]]["flows"]]
    one_packet_flows = sum(int(flow["packet_count"]) == 1 for flow in all_readable_flows)
    under_five_packet_flows = sum(int(flow["packet_count"]) < 5 for flow in all_readable_flows)
    if successful_packet_count != successful_ip_packets + successful_non_ip_packets:
        raise AssertionError("Packet accounting does not close")
    if stage14a_flows != tcp_flows + udp_flows + other_ip_flows or stage14a_flows != 23454:
        raise AssertionError("Stage 14A flow accounting does not close")

    failed_prefix_packets: dict[str, int] = {}
    for row in failed:
        match = re.search(r"after reading (\d+) packets", row["parse_error"])
        failed_prefix_packets[row["capture_id"]] = int(match.group(1)) if match else 0

    if not CORRUPT_PREFIX.exists():
        raise FileNotFoundError(f"Missing diagnostic prefix scan: {CORRUPT_PREFIX}")
    prefix_payload = json.loads(CORRUPT_PREFIX.read_text(encoding="utf-8"))
    prefix_by_name = {row["file_name"]: row for row in prefix_payload["results"]}

    official_counts, official_rows = load_official_counts(pcap_by_file)
    for row in official_rows:
        prefix = prefix_by_name.get(row["file_name"])
        row["readable_prefix_diagnostic_flow_count"] = prefix["readable_prefix_flow_count"] if prefix else ""
        row["readable_prefix_packet_rows"] = prefix["packet_rows_emitted_before_error"] if prefix else ""
    write_csv(ARTIFACTS / "official_connection_comparison.csv", official_rows)
    official_failed_total = sum(official_counts[row["file_name"]] for row in failed)
    prefix_failed_total = sum(int(prefix_by_name[row["file_name"]]["readable_prefix_flow_count"]) for row in failed)

    duplicate_files = read_csv(DUPLICATE_FILES)
    duplicate_flows = read_csv(DUPLICATE_FLOWS)
    class_audit = read_csv(CLASS_AUDIT)
    dropped_duplicate_capture = "vpn_skype-chat_capture4"
    duplicate_pcap_flow_count = int(scans[dropped_duplicate_capture]["flow_count"])
    cross_app_rows = [
        row for row in duplicate_flows
        if any(peer != row["application"] for peer in row["peer_applications"].split(";") if peer)
    ]
    if len(duplicate_files) != 1 or duplicate_pcap_flow_count != 1 or len(cross_app_rows) != 4:
        raise AssertionError("Duplicate evidence changed")
    final_flows = sum(int(row["cleaned_flow_count"]) for row in class_audit)
    if stage14a_flows - duplicate_pcap_flow_count - len(cross_app_rows) != final_flows or final_flows != 23449:
        raise AssertionError("Flow conservation failed")

    tshark_version = subprocess.run(
        ["tshark", "--version"], capture_output=True, text=True, check=True
    ).stdout.splitlines()[0]

    lineage_rows: list[dict[str, Any]] = [
        {
            "step": 1, "stage": "raw_pcap_inventory", "input_count": 165, "input_unit": "PCAP",
            "output_count": 165, "output_unit": "PCAP", "removed_count": 0,
            "category": "inventory", "reason": "all source PCAPs inventoried",
            "code_file": "stage14a_vnat_data_audit/scripts/audit_vnat.py", "function": "main/scan_pcap dispatch",
            "parameters": "*.pcap under VNAT_release_1", "included_in_flow_conservation": False,
        },
        {
            "step": 2, "stage": "pcap_readability_gate", "input_count": 165, "input_unit": "PCAP",
            "output_count": 163, "output_unit": "PCAP", "removed_count": 2,
            "category": "parser failure; corrupted PCAP", "reason": "capinfos returned non-zero for a mid-packet truncation",
            "code_file": "stage14a_vnat_data_audit/scripts/audit_vnat.py:134-149,247-268", "function": "capinfos; scan_pcap",
            "parameters": "capinfos -Tm -c -a -e -u -H; any exception leaves readable=False",
            "included_in_flow_conservation": False,
        },
        {
            "step": "2a", "stage": "corrupt_pcap_flow_exclusion_diagnostic",
            "input_count": stage14a_flows + prefix_failed_total, "input_unit": "locally groupable flow over complete packet records",
            "output_count": stage14a_flows, "output_unit": "flow accepted by Stage 14A", "removed_count": prefix_failed_total,
            "category": "parser failure; corrupted PCAP",
            "reason": "audit-only grouping found flows in the complete packet rows emitted before each truncation error; frozen pipeline rejects both whole PCAPs",
            "code_file": "stage14a_vnat_data_audit/scripts/audit_vnat.py:134-149,247-268; stage14b5_vnat_flow_retention_audit/scripts/scan_corrupt_prefixes.py",
            "function": "capinfos; scan_pcap; diagnostic scan",
            "parameters": "same tcp.stream/udp.stream/other-IP grouping; diagnostic only; never inserted into Stage 14B",
            "included_in_flow_conservation": "diagnostic raw-complete-packet-record ledger",
        },
        {
            "step": 3, "stage": "packet_parsing_successful_pcaps", "input_count": successful_packet_count,
            "input_unit": "packet", "output_count": successful_packet_count, "output_unit": "packet", "removed_count": 0,
            "category": "packet parsing", "reason": "capinfos and full tshark traversal counts agree for all 163 PCAPs",
            "code_file": "stage14a_vnat_data_audit/scripts/audit_vnat.py:152-265", "function": "scan_flows; scan_pcap",
            "parameters": f"{tshark_version}; -n; frame.generate_md5_hash:TRUE; no display filter",
            "included_in_flow_conservation": False,
        },
        {
            "step": 4, "stage": "ip_packet_eligibility", "input_count": successful_packet_count,
            "input_unit": "packet", "output_count": successful_ip_packets, "output_unit": "packet",
            "removed_count": successful_non_ip_packets, "category": "protocol filter",
            "reason": "packets lacking both IPv4/IPv6 source and destination do not enter a flow",
            "code_file": "stage14a_vnat_data_audit/scripts/audit_vnat.py:184-189", "function": "scan_flows",
            "parameters": "src=ip.src or ipv6.src; dst=ip.dst or ipv6.dst",
            "included_in_flow_conservation": False,
        },
        {
            "step": 5, "stage": "flow_session_construction", "input_count": successful_ip_packets,
            "input_unit": "IP packet", "output_count": stage14a_flows, "output_unit": "flow",
            "removed_count": "not_applicable_different_units", "category": "flow construction rule",
            "reason": "bidirectional TCP/UDP tshark stream IDs; other IP grouped by canonical endpoints+protocol",
            "code_file": "stage14a_vnat_data_audit/scripts/audit_vnat.py:127-131,190-220", "function": "canonical_tuple; scan_flows",
            "parameters": "tcp.stream / udp.stream; no explicit timeout; no packet/duration/payload threshold",
            "included_in_flow_conservation": False,
        },
        {
            "step": 6, "stage": "flow_feature_summary", "input_count": stage14a_flows, "input_unit": "flow",
            "output_count": stage14a_flows, "output_unit": "flow", "removed_count": 0,
            "category": "feature extraction", "reason": "metadata summary only: counts, bytes, timestamps and hashes; no ML features",
            "code_file": "stage14a_vnat_data_audit/scripts/audit_vnat.py:205-243", "function": "scan_flows",
            "parameters": "frame.cap_len; first/last timestamp; tuple/content SHA256",
            "included_in_flow_conservation": True,
        },
        {
            "step": 7, "stage": "general_flow_filtering", "input_count": stage14a_flows, "input_unit": "flow",
            "output_count": stage14a_flows, "output_unit": "flow", "removed_count": 0,
            "category": "filtering", "reason": "no short-flow, payload, handshake, port, duration or packet-count filter exists",
            "code_file": "stage14a_vnat_data_audit/scripts/audit_vnat.py:152-243", "function": "scan_flows",
            "parameters": "none", "included_in_flow_conservation": True,
        },
        {
            "step": 8, "stage": "duplicate_pcap_removal", "input_count": stage14a_flows, "input_unit": "flow",
            "output_count": stage14a_flows - duplicate_pcap_flow_count, "output_unit": "flow",
            "removed_count": duplicate_pcap_flow_count, "category": "duplicate PCAP",
            "reason": "capture4 byte-identical to capture3; lexicographically retain capture3 and drop capture4",
            "code_file": "stage14b_vnat_protocol_freeze/scripts/freeze_vnat_protocol.py:77-85,133-135",
            "function": "integrity_exclusions; build_clean_flows", "parameters": "full-file SHA256 equality",
            "included_in_flow_conservation": True,
        },
        {
            "step": 9, "stage": "cross_application_duplicate_removal",
            "input_count": stage14a_flows - duplicate_pcap_flow_count, "input_unit": "flow",
            "output_count": final_flows, "output_unit": "flow", "removed_count": len(cross_app_rows),
            "category": "cross-application duplicate", "reason": "two identical packet sequences appear once under rsync and once under sftp; all four rows quarantined",
            "code_file": "stage14b_vnat_protocol_freeze/scripts/freeze_vnat_protocol.py:65-75,136-140",
            "function": "integrity_exclusions; build_clean_flows", "parameters": "content_sha256 peer application differs",
            "included_in_flow_conservation": True,
        },
        {
            "step": 10, "stage": "application_labeling", "input_count": final_flows, "input_unit": "flow",
            "output_count": final_flows, "output_unit": "flow", "removed_count": 0,
            "category": "application labeling", "reason": "every retained flow inherits application and VPN status parsed from its source PCAP filename",
            "code_file": "stage14a_vnat_data_audit/scripts/audit_vnat.py:29-45,100-124; stage14b_vnat_protocol_freeze/scripts/freeze_vnat_protocol.py:145-161",
            "function": "parse_pcap_name; build_clean_flows", "parameters": "voip->zoiper; skype-chat->skype; remaining keywords identity",
            "included_in_flow_conservation": True,
        },
        {
            "step": 11, "stage": "stage14b_clean_pool", "input_count": final_flows, "input_unit": "flow",
            "output_count": final_flows, "output_unit": "flow", "removed_count": 0,
            "category": "final", "reason": "frozen Stage 14B clean pool before protocol expansion",
            "code_file": "stage14b_vnat_protocol_freeze/scripts/freeze_vnat_protocol.py:98-217", "function": "build_clean_flows",
            "parameters": "does not read test outcomes or train a model", "included_in_flow_conservation": True,
        },
    ]
    write_csv(AUDIT / "stage14b5_flow_lineage.csv", lineage_rows)

    filter_rows = [
        ("F01", "pcap_readability_gate", "parser failure / corrupted PCAP", True, "PCAP", 165, 2, 163,
         "capinfos non-zero on truncated PCAP; scan_flows is never called", "audit_vnat.py:134-149,247-268", "capinfos -Tm -c -a -e -u -H"),
        ("F02", "packet_to_flow", "non-IP packet exclusion", True, "packet", successful_packet_count,
         successful_non_ip_packets, successful_ip_packets, "requires IPv4/IPv6 src and dst", "audit_vnat.py:184-189", "no display filter"),
        ("F03", "flow_construction", "non-TCP/UDP protocol filter", False, "flow", stage14a_flows, 0, stage14a_flows,
         "non-TCP/UDP IP is retained as ip:<protocol>", "audit_vnat.py:198-203", "canonical endpoints + protocol"),
        ("F04", "flow_filtering", "single-packet flow filter", False, "flow", stage14a_flows, 0, stage14a_flows,
         f"no threshold; {one_packet_flows} single-packet flows retained", "audit_vnat.py:205-243", "none"),
        ("F05", "flow_filtering", "short-flow / packet-count filter", False, "flow", stage14a_flows, 0, stage14a_flows,
         f"no threshold; {under_five_packet_flows} flows with <5 packets retained", "audit_vnat.py:205-243", "none"),
        ("F06", "flow_filtering", "empty-payload filter", False, "flow", stage14a_flows, 0, stage14a_flows,
         "payload fields are not extracted, so payload emptiness is neither measured nor filtered", "audit_vnat.py:46-62,152-243", "none"),
        ("F07", "flow_filtering", "invalid 5-tuple filter", False, "flow", stage14a_flows, 0, stage14a_flows,
         "no explicit validation; TCP/UDP stream IDs drive grouping and other IP intentionally has blank ports", "audit_vnat.py:190-203", "none"),
        ("F08", "flow_filtering", "TCP completion filter", False, "flow", stage14a_flows, 0, stage14a_flows,
         "SYN/FIN/RST state is not extracted or checked", "audit_vnat.py:46-62,152-243", "none"),
        ("F09", "flow_filtering", "port/protocol allowlist", False, "flow", stage14a_flows, 0, stage14a_flows,
         "no port allow/deny list and all IP protocols retained", "audit_vnat.py:190-203", "none"),
        ("F10", "flow_filtering", "flow duration threshold", False, "flow", stage14a_flows, 0, stage14a_flows,
         "first/last timestamps recorded but never thresholded", "audit_vnat.py:205-243", "none"),
        ("F11", "flow_construction", "flow timeout/session rule", True, "configuration", "", "", "",
         "no explicit timeout; TCP/UDP sessionization delegated to tshark tcp.stream/udp.stream defaults; other IP has no time split", "audit_vnat.py:190-203", tshark_version),
        ("F12", "deduplication", "exact duplicate PCAP", True, "flow", stage14a_flows, 1, stage14a_flows - 1,
         "drop all flow contribution from vpn_skype-chat_capture4; retain capture3", "freeze_vnat_protocol.py:77-85,133-135", "full-file SHA256"),
        ("F13", "deduplication", "cross-application duplicate flow", True, "flow", stage14a_flows - 1, 4, final_flows,
         "quarantine both copies of two packet-identical rsync/sftp sequences", "freeze_vnat_protocol.py:65-75,136-140", "content_sha256"),
        ("F14", "labeling", "missing label", False, "flow", final_flows, 0, final_flows,
         "all retained PCAP names matched the frozen filename grammar and application map", "audit_vnat.py:29-45,100-124", "filename-derived"),
    ]
    write_csv(AUDIT / "stage14b5_filter_breakdown.csv", [
        {
            "rule_id": rule_id, "stage": stage, "category": category, "applied": applied,
            "unit": unit, "input_count": input_count, "removed_count": removed_count,
            "retained_count": retained_count, "actual_logic_or_evidence": evidence,
            "code_location": code_location, "parameters": parameters,
        }
        for rule_id, stage, category, applied, unit, input_count, removed_count, retained_count,
        evidence, code_location, parameters in filter_rows
    ])

    official_by_class = Counter()
    official_by_class_vpn = Counter()
    official_corrupt_by_class = Counter()
    prefix_by_class = Counter()
    for row in official_rows:
        app, vpn = row["application"], row["vpn_status"]
        official_by_class[app] += int(row["official_connection_count"])
        official_by_class_vpn[(app, vpn)] += int(row["official_connection_count"])
        if row["pcap_readable_in_stage14a"] != "True":
            official_corrupt_by_class[app] += int(row["official_connection_count"])
            prefix_by_class[app] += int(row["readable_prefix_diagnostic_flow_count"] or 0)

    raw_by_class_vpn = Counter()
    for pcap in readable:
        raw_by_class_vpn[(pcap["application"], pcap["vpn_status"])] += int(scans[pcap["capture_id"]]["flow_count"])
    class_rows = []
    for row in class_audit:
        app = row["application"]
        raw = int(row["raw_readable_flow_count"])
        final = int(row["cleaned_flow_count"])
        removed = raw - final
        reasons = []
        if int(row["removed_duplicate_pcap_flows"]):
            reasons.append(f"duplicate PCAP flow={row['removed_duplicate_pcap_flows']}")
        if int(row["removed_cross_application_duplicate_flows"]):
            reasons.append(f"cross-app duplicate rows={row['removed_cross_application_duplicate_flows']}")
        if int(row["failed_pcap_count"]):
            reasons.append(f"failed PCAP excluded before local flow construction={row['failed_pcap_ids']}")
        if not reasons:
            reasons.append("none after successful flow construction")
        class_rows.append({
            "application": app,
            "raw_flow_count": raw,
            "removed_flow_count": removed,
            "final_flow_count": final,
            "retention_rate": f"{final / raw:.8f}" if raw else "",
            "vpn_raw": raw_by_class_vpn[(app, "vpn")],
            "vpn_final": row["vpn_flow_count"],
            "nonvpn_raw": raw_by_class_vpn[(app, "nonvpn")],
            "nonvpn_final": row["nonvpn_flow_count"],
            "main_removal_reason": "; ".join(reasons),
            "failed_pcap_count": row["failed_pcap_count"],
            "official_connection_count_all_pcaps": official_by_class[app],
            "official_connections_in_failed_pcaps": official_corrupt_by_class[app],
            "corrupt_readable_prefix_diagnostic_flows": prefix_by_class[app],
            "official_minus_local_raw_readable": official_by_class[app] - official_corrupt_by_class[app] - raw,
            "official_vpn_connections": official_by_class_vpn[(app, "vpn")],
            "official_nonvpn_connections": official_by_class_vpn[(app, "nonvpn")],
        })
    write_csv(AUDIT / "stage14b5_class_retention.csv", class_rows)

    failed_lines = []
    official_failed_total = 0
    prefix_failed_total = 0
    for pcap in failed:
        official_n = official_counts[pcap["file_name"]]
        prefix = prefix_by_name[pcap["file_name"]]
        official_failed_total += official_n
        prefix_failed_total += int(prefix["readable_prefix_flow_count"])
        failed_lines.append(
            f"| {pcap['capture_id']} | {failed_prefix_packets[pcap['capture_id']]:,} | "
            f"{prefix['packet_rows_emitted_before_error']:,} | {prefix['readable_prefix_flow_count']:,} | "
            f"{official_n:,} | 0 |"
        )

    comparable_official = sum(
        int(row["official_connection_count"])
        for row in official_rows if row["pcap_readable_in_stage14a"] == "True"
    )
    official_after_same_integrity_drops = comparable_official - 1 - 4
    class_table = "\n".join(
        f"| {row['application']} | {row['raw_flow_count']} | {row['removed_flow_count']} | "
        f"{row['final_flow_count']} | {float(row['retention_rate']):.4%} | "
        f"{row['vpn_raw']}/{row['vpn_final']} | {row['nonvpn_raw']}/{row['nonvpn_final']} | "
        f"{row['official_connection_count_all_pcaps']} | {row['main_removal_reason']} |"
        for row in class_rows
    )
    markdown = f"""# Stage 14B.5 — VNAT Flow Retention & Filtering Audit

## Audit boundary and verdict

- **Verdict: `PASS_WITH_EXPLAINED_DIFFERENCE`.** The frozen Stage 14B clean pool closes exactly as `23,454 - 1 - 4 = 23,449` flows.
- This audit did not regenerate Stage 14B, did not train a model, and did not run Open-Detect/DES.
- Frozen protocol identity before the audit: `{before['freeze_hash']}`; protocol JSON `{before['vnat_open_set_protocol_sha256']}`; split manifest `{before['vnat_split_manifest_sha256']}`.
- The raw-to-clean story has one important boundary: a PCAP has no unique "raw flow count" until a sessionization definition is chosen. The two truncated PCAPs were rejected before Stage 14A flow construction, so their exact counterfactual Stage 14A full-file flow count is unknowable. We report both a read-only readable-prefix diagnostic and the official connection dataframe count, without treating either as interchangeable with the frozen local flow definition.

## Exact local quantity conservation

```text
Stage 14A flows from 163 fully readable PCAPs = 23,454
- exact duplicate PCAP contribution            =      1
- cross-application duplicate flow rows        =      4
- all other flow filters                       =      0
--------------------------------------------------------
Stage 14B clean pool                           = 23,449
```

The five successfully constructed flows that did not enter the clean pool are fully identified: one UDP flow from duplicate `vpn_skype-chat_capture4`, plus four non-TCP/UDP rows representing two packet-identical sequences present under both `rsync` and `sftp`. There is no unresolved gap inside the accepted Stage 14A flow pool.

For the broader raw-file audit, tshark emitted all complete packet records before each truncation error and the diagnostic applied the same grouping semantics:

```text
Locally groupable flows over all complete packet rows = 34,009
- flows in two whole rejected truncated PCAPs          = 10,555
- exact duplicate PCAP contribution                    =      1
- cross-application duplicate flow rows                =      4
----------------------------------------------------------------
Stage 14B clean pool                                    = 23,449
```

Thus **10,560 locally groupable flows are absent from the final pool when the corrupt-PCAP gate is included**, while only five of those existed as Stage 14A flow rows and were then deleted. This second ledger is diagnostic-only; it does not mutate or repopulate Stage 14B.

## PCAP and packet path

- Source inventory: 165 PCAPs.
- Fully passed `capinfos` and complete `tshark` traversal: 163; rejected as truncated: 2.
- Packets in the 163 successful PCAPs: **{successful_packet_count:,}**; IP-eligible packets: **{successful_ip_packets:,}**; non-IP packets excluded before flow grouping: **{successful_non_ip_packets:,}**.
- Resulting flows: **{stage14a_flows:,}** = TCP {tcp_flows:,} + UDP {udp_flows:,} + other-IP {other_ip_flows:,}.
- Single-packet flows retained: **{one_packet_flows:,}**. Flows with fewer than five packets retained: **{under_five_packet_flows:,}**.

### Two truncated PCAPs

| capture | capinfos packets before error | tshark prefix rows | prefix flows under local grouping | official H5 connections | Stage 14B contribution |
|---|---:|---:|---:|---:|---:|
{chr(10).join(failed_lines)}

The current pipeline loses the entire contribution of both captures because `scan_pcap()` calls `capinfos()` first, and any non-zero return jumps to the exception handler before `scan_flows()` is invoked. Thus the **pipeline-observed contribution is exactly zero**. The official H5 contains **{official_failed_total:,}** connections for those two filenames; the read-only diagnostic found **{prefix_failed_total:,}** locally grouped flows in bytes emitted before the truncation error. These are loss indicators, not additions to the frozen protocol, and the unknown missing tail prevents claiming an exact full-file counterfactual.

## Actual construction and filtering rules

1. **PCAP parsing gate** — `capinfos -Tm -c -a -e -u -H`; any non-zero status rejects the whole PCAP (`audit_vnat.py:134-149,247-268`).
2. **Packet parsing** — `tshark -n -o frame.generate_md5_hash:TRUE -r ... -T fields`; no display/capture filter (`audit_vnat.py:152-158`). Runtime: `{tshark_version}`.
3. **Packet eligibility** — only packets with an IPv4 or IPv6 source and destination enter flow construction (`audit_vnat.py:184-189`). This is a non-IP exclusion, not a TCP/UDP-only filter.
4. **Session construction** — TCP uses `tcp.stream`; UDP uses `udp.stream`; other IP protocols are retained and grouped bidirectionally by canonical endpoints plus IP protocol (`audit_vnat.py:190-203`). No explicit timeout is configured. Therefore TCP/UDP session boundaries depend on the installed tshark defaults; other-IP grouping has no time split.
5. **Flow summary** — packet count, captured byte count, first/last timestamp, tuple hash and ordered packet-MD5 content hash are recorded (`audit_vnat.py:205-243`). This is not the official 129-feature H5 extractor.
6. **No general flow filter** — no single/short-flow threshold, payload-length threshold, flow-duration threshold, TCP-completion requirement, port allowlist, or malformed-5-tuple rejection exists. Empty payload cannot be tested because payload fields are not extracted.
7. **Duplicate PCAP** — Stage 14A detects identical full-file SHA256; Stage 14B lexicographically retains `vpn_skype-chat_capture3` and drops `capture4` (`freeze_vnat_protocol.py:77-85,133-135`). Both contain one identical UDP flow, so exactly one flow is removed.
8. **Cross-application duplicate flows** — ordered packet hashes form `content_sha256`; all four rows spanning `nonvpn_rsync_newcapture1` and `nonvpn_sftp_newcapture2` are quarantined (`freeze_vnat_protocol.py:65-75,136-140`).
9. **Labels** — filename grammar maps `voip -> zoiper`, `skype-chat -> skype`, and all other keywords to the same application name; each flow inherits its PCAP's application and VPN status (`audit_vnat.py:29-45,100-124`). VPN and non-VPN use exactly the same extractor and filters.

The complete machine-readable rule table, including explicitly absent filters, is `stage14b5_filter_breakdown.csv`.

## Per-application retention

Here `raw_flow_count` means flows successfully constructed by Stage 14A from fully readable PCAPs; it does not silently substitute official H5 connections for failed-capture flows.

| application | raw local | removed after construction | final | retention | VPN raw/final | non-VPN raw/final | official all-PCAP connections | main reason |
|---|---:|---:|---:|---:|---:|---:|---:|---|
{class_table}

`rdp`, `netflix`, `youtube`, and `ssh` have **no post-construction filtering at all**. Their apparent size differences are source/capture and sessionization properties, not preprocessing removals. `scp` is affected by whole-PCAP rejection of `nonvpn_scp_long_capture1`; `skype` is affected by one truncated PCAP and one duplicate-PCAP flow; `rsync` and `sftp` each lose two cross-application duplicate rows.

## Official H5 comparison

- Official `VNAT_Dataframe_release_1.h5`: **33,711** connection rows across all 165 filenames.
- Official rows belonging to the two failed local PCAPs: **{official_failed_total:,}**; official rows for the 163 locally readable PCAPs: **{comparable_official:,}**.
- Local Stage 14A flow rows for those readable PCAPs: **23,454**. Difference (official minus local): **{comparable_official - stage14a_flows:+,}**.
- Applying the same one duplicate-PCAP and four cross-app row exclusions gives an indicative official count of **{official_after_same_integrity_drops:,}**, versus local **23,449**, still **{official_after_same_integrity_drops - final_flows:+,}**.

This difference is **not a filter loss**: the official artifact stores prebuilt `connection` objects, while this pipeline uses tshark stream IDs and a custom rule for other IP protocols. No one-to-one connection matching or official extractor code exists in the audited local pipeline. Per-capture, per-application and VPN/non-VPN counts are in `artifacts/official_connection_comparison.csv` and `stage14b5_class_retention.csv`.

## Fairness assessment

- No class-dependent packet/flow filter and no VPN-specific filter was found.
- The main fairness risk is whole-capture exclusion: `scp` loses a large non-VPN capture and `skype` loses one VPN capture. This can alter class/domain composition, even though the exclusion is integrity-driven and applied before any model result.
- Filename-derived application labels and tshark default sessionization are reproducible in this environment but are not proven equivalent to the official connection dataframe definition.
- Duplicate removal is appropriate for leakage control: the duplicate Skype capture is not double-counted, and cross-application packet-identical rows cannot straddle Known/Unknown semantics.

## Required answers

1. **Why 23,449?** Exactly `23,454 successful Stage 14A flows - 1 duplicate-PCAP flow - 4 cross-application duplicate rows`.
2. **How many were deleted/filtered?** Five after successful Stage 14A flow construction. Counting all complete packet rows present in the two truncated files, 10,555 additional locally groupable flows were excluded at the whole-PCAP gate, so the diagnostic raw-file-to-clean total absent is **10,560**. The unavailable intended tail remains unknowable. Official H5 reports {official_failed_total:,} connections for the two files.
3. **Rules?** Non-IP packets are excluded before grouping; two corrupted PCAPs fail the parser gate; one exact duplicate PCAP and four cross-app duplicate rows are removed. All queried short/payload/TCP/port/duration filters are absent.
4. **Code/functions?** `audit_vnat.py::{{capinfos,scan_pcap,scan_flows,canonical_tuple,parse_pcap_name,duplicate_annotations}}` and `freeze_vnat_protocol.py::{{integrity_exclusions,build_clean_flows}}`; exact locations are in the CSVs above.
5. **Key parameters?** `capinfos -Tm -c -a -e -u -H`; tshark `-n`, `frame.generate_md5_hash:TRUE`, listed fields, no display filter, no explicit timeout, and no flow-size/duration/payload thresholds.
6. **Unexpected filtering?** No hidden post-construction filter was found. Whole-PCAP rejection before partial salvage is consequential but explicit in control flow; official/local sessionization differs.
7. **Can 23,449 be strictly explained?** Yes, from the successful Stage 14A pool. The intended full-capture counterfactual for truncated tails cannot be reconstructed exactly.
8. **Potential external-validation fairness impact?** Yes: integrity exclusion changes `scp` non-VPN and `skype` VPN coverage, and official/local connection definitions differ. There is no evidence of outcome-driven or class-specific filtering.
9. **Freeze hash before/after?** Verified below and machine-recorded in `artifacts/freeze_hash_before.json` / `freeze_hash_after.json`.
10. **Final conclusion:** `PASS_WITH_EXPLAINED_DIFFERENCE`.
"""
    (AUDIT / "stage14b5_flow_audit.md").write_text(markdown, encoding="utf-8")

    after = freeze_state()
    if after != before:
        raise AssertionError(f"Frozen inputs changed during audit: before={before}, after={after}")
    (ARTIFACTS / "freeze_hash_after.json").write_text(
        json.dumps({"checked_at_utc": utc_now(), **after, "matches_before": True}, indent=2) + "\n",
        encoding="utf-8",
    )
    with (AUDIT / "stage14b5_flow_audit.md").open("a", encoding="utf-8") as handle:
        handle.write(
            f"\nFreeze hash after audit: `{after['freeze_hash']}` — **PASS, unchanged**. "
            f"Protocol JSON and split manifest SHA256 also match their before-audit values.\n"
        )

    summary = {
        "generated_at_utc": utc_now(),
        "verdict": "PASS_WITH_EXPLAINED_DIFFERENCE",
        "pcap_total": len(pcaps),
        "pcap_success": len(readable),
        "pcap_failed": len(failed),
        "stage14a_flows": stage14a_flows,
        "duplicate_pcap_flows_removed": duplicate_pcap_flow_count,
        "cross_application_rows_removed": len(cross_app_rows),
        "final_clean_flows": final_flows,
        "official_connections": sum(official_counts.values()),
        "official_failed_pcap_connections": official_failed_total,
        "prefix_diagnostic_flows": prefix_failed_total,
        "freeze_hash_before": before["freeze_hash"],
        "freeze_hash_after": after["freeze_hash"],
        "freeze_unchanged": after == before,
    }
    (ARTIFACTS / "audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
