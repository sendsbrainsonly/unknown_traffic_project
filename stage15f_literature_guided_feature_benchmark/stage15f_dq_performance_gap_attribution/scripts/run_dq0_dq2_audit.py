#!/usr/bin/env python3
"""Execute Stage 15F-DQ DQ-0 through DQ-2 on the shared VPN captures.

The script never opens Known Test or Unknown Test arrays.  It reconstructs the
three projects' different flow semantics without silently merging them.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


PROJECT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parents[1]
S12 = PROJECT / "stage12_dual_external_validation"
TFE = WORKSPACE / "Projects/TFE-GNN"
TF = WORKSPACE / "Projects/TrafficFormer"

TFE_PREP = TFE / "artifacts/paper-reproduction-v2/iscx-vpn/flows/prepare_manifest.json"
TFE_DATASET = TFE / "artifacts/paper-reproduction-v2/iscx-vpn/paper_text_split_seed32/dataset_manifest.json"
TF_DATA = TF / "reproduction/datasets/iscxvpn_trafficformer"
SPLIT = S12 / "protocol/iscx_vpn/split_manifest.csv"
PREP = S12 / "protocol/iscx_vpn/preprocessing_manifest.json"
CAPTURE_AUDIT = PROJECT / "stage15r_representation_bottleneck_audit/feature_cache/iscx_vpn/capture_audit.csv"
FLOW_STATS = PROJECT / "stage15r_representation_bottleneck_audit/feature_cache/iscx_vpn/flow_statistics.csv"
RUN = S12 / "runs/iscx_vpn/medium/seed2022"

CANONICAL_SERVICE = {
    "chat": "Chat",
    "email": "Email",
    "file": "File-Transfer",
    "file_transfer": "File-Transfer",
    "p2p": "P2P",
    "streaming": "Streaming",
    "voip": "VoIP",
    "Chat": "Chat",
    "Email": "Email",
    "File": "File-Transfer",
    "File-Transfer": "File-Transfer",
    "P2P": "P2P",
    "Streaming": "Streaming",
    "VoIP": "VoIP",
}

FIELDS = (
    "frame.number", "frame.time_epoch", "frame.len", "frame.cap_len",
    "ip.src", "ip.dst", "ip.proto",
    "tcp.srcport", "tcp.dstport", "udp.srcport", "udp.dstport",
    "tcp.flags.syn", "tcp.flags.ack", "tcp.flags.fin", "tcp.flags.reset",
    "ip.frag_offset",
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_frame(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def sha256_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def text(values: list[str], index: int) -> str:
    return values[index] if index < len(values) else ""


def flag(value: str) -> bool:
    return value.lower() in {"1", "true", "set"}


def canonical_key(protocol: str, source_ip: str, source_port: str, destination_ip: str, destination_port: str) -> tuple[str, str, str]:
    endpoints = sorted(((source_ip, str(source_port)), (destination_ip, str(destination_port))))
    return protocol.lower(), f"{endpoints[0][0]}:{endpoints[0][1]}", f"{endpoints[1][0]}:{endpoints[1][1]}"


def flow_token(key: tuple[str, str, str]) -> str:
    return "|".join(key)


def packet_key(values: list[str]) -> tuple[str, str, str] | None:
    # tshark may expose reassembled TCP/UDP fields on non-first IPv4 fragments,
    # while Scapy's packet.haslayer(TCP/UDP) used by both source pipelines does
    # not.  Exclude those rows to preserve source-code semantics.
    fragment_offset = text(values, 15)
    if fragment_offset and int(fragment_offset) > 0:
        return None
    src, dst = text(values, 4), text(values, 5)
    ip_protocol = text(values, 6)
    tcp_s, tcp_d = text(values, 7), text(values, 8)
    udp_s, udp_d = text(values, 9), text(values, 10)
    # Require the outer IPv4 protocol to agree with the transport fields.
    # Otherwise tshark may expose an inner UDP/TCP header carried by ICMP,
    # whereas Scapy's direct TCP/UDP layer test does not accept that packet.
    if ip_protocol == "6" and src and dst and tcp_s and tcp_d:
        return canonical_key("tcp", src, tcp_s, dst, tcp_d)
    if ip_protocol == "17" and src and dst and udp_s and udp_d:
        return canonical_key("udp", src, udp_s, dst, udp_d)
    return None


def parse_tfe_name(name: str) -> tuple[str, str, str]:
    match = re.search(r"\.(TCP|UDP)_([0-9-]+)_(\d+)_([0-9-]+)_(\d+)\.pcap$", name)
    if match is None:
        raise ValueError(f"cannot parse TFE CATE flow name: {name}")
    protocol, src_ip, src_port, dst_ip, dst_port = match.groups()
    return canonical_key(protocol, src_ip.replace("-", "."), src_port, dst_ip.replace("-", "."), dst_port)


@dataclass
class NativeState:
    sequence: int = 0
    current: dict[str, Any] | None = None
    last_timestamp: float | None = None
    closed: bool = False
    syn_only: bool = False


def new_native_row(capture: dict[str, Any], key: tuple[str, str, str], sequence: int, timestamp: float) -> dict[str, Any]:
    capture_id = f"class={capture['application']}|capture={capture['capture_index']}|name={Path(capture['pcap_path']).name}"
    session_id = f"{capture_id}|{flow_token(key)}|session={sequence}"
    protocol, endpoint_a, endpoint_b = key
    ip_a, port_a = endpoint_a.rsplit(":", 1)
    ip_b, port_b = endpoint_b.rsplit(":", 1)
    return {
        "dataset": "ISCX-VPN",
        "flow_id": hashlib.sha256(session_id.encode()).hexdigest(),
        "native_session_id": session_id,
        "capture_id": capture_id,
        "source_file": capture["source_file"],
        "pcap_path": capture["pcap_path"],
        "application": capture["application"],
        "service": capture["service"],
        "vpn_state": "VPN",
        "ip_protocol": protocol.upper(),
        "endpoint_a_ip": ip_a,
        "endpoint_a_port": int(port_a),
        "endpoint_b_ip": ip_b,
        "endpoint_b_port": int(port_b),
        "bidirectional_five_tuple": flow_token(key),
        "native_session_sequence": sequence,
        "flow_start_time": timestamp,
        "flow_end_time": timestamp,
        "packet_count": 0,
        "total_bytes": 0,
        "captured_bytes": 0,
    }


def load_native_membership() -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    split = pd.read_csv(SPLIT, dtype=str)
    membership: dict[str, dict[str, Any]] = {}
    for uid, group in split.groupby("flow_id_sha256", sort=False):
        fields = ["canonical_class", "source_file", "domain_state", "official_category", "image_sha256"]
        for name in fields:
            if group[name].nunique(dropna=False) != 1:
                raise RuntimeError(f"inconsistent frozen membership for {uid}: {name}")
        roles = {row.setting: row.role for row in group.itertuples()}
        first = group.iloc[0]
        membership[uid] = {
            "canonical_class": first.canonical_class,
            "source_file": first.source_file,
            "domain_state": first.domain_state,
            "official_category": first.official_category,
            "image_sha256": first.image_sha256,
            "roles": roles,
        }
    return membership, split


def load_tfe() -> tuple[dict[str, dict[str, Any]], set[str], set[str]]:
    prep = read_json(TFE_PREP)
    dataset = read_json(TFE_DATASET)
    skipped_empty = {Path(value).name.replace(".npz", ".pcap") for value in dataset.get("skipped_empty", [])}
    skipped_anomalous = {Path(value).name.replace(".npz", ".pcap") for value in dataset.get("skipped_anomalous", [])}
    sources: dict[str, dict[str, Any]] = {}
    for source in prep["sources"]:
        path = str(Path(source["source"]).resolve())
        by_key: defaultdict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
        for flow in source["flows"]:
            item = dict(flow)
            item["key"] = parse_tfe_name(flow["name"])
            item["canonical_service"] = CANONICAL_SERVICE[flow["category"]]
            item["paper_text_eligible"] = flow["name"] not in skipped_empty | skipped_anomalous
            by_key[item["key"]].append(item)
        sources[path] = {**source, "by_key": dict(by_key)}
    return sources, skipped_empty, skipped_anomalous


def load_tf_audit() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    audit = pd.read_csv(TF_DATA / "processing_audit.tsv", sep="\t")
    audit["source_file"] = audit["source_file"].map(lambda value: str(Path(value).resolve()))
    return audit, {source: group.copy() for source, group in audit.groupby("source_file", sort=False)}


def finalize_state(state: NativeState, rows: list[dict[str, Any]]) -> None:
    if state.current is not None:
        rows.append(state.current)
        state.current = None


def reconstruct_capture(
    capture: dict[str, Any],
    tfe_source: dict[str, Any],
    tf_rows: pd.DataFrame,
    membership: dict[str, dict[str, Any]],
    timeout: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pcap = Path(capture["pcap_path"])
    command = ["tshark", "-n", "-r", str(pcap), "-T", "fields", "-E", "separator=/t", "-E", "occurrence=f"]
    for field_name in FIELDS:
        command.extend(["-e", field_name])
    stderr_path = OUT / "tshark_stderr" / f"{capture['application']}_{capture['capture_index']:03d}_{pcap.stem}.log"
    states: dict[tuple[str, str, str], NativeState] = {}
    parents: defaultdict[tuple[str, str, str], dict[str, Any]] = defaultdict(
        lambda: {"packet_count": 0, "frame_bytes": 0, "captured_bytes": 0, "start": math.inf, "end": -math.inf}
    )
    native_rows: list[dict[str, Any]] = []
    packet_rows = eligible_packets = 0
    with stderr_path.open("w", encoding="utf-8") as stderr_handle:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=stderr_handle, text=True, bufsize=1024 * 1024)
        assert process.stdout is not None
        for line in process.stdout:
            packet_rows += 1
            values = line.rstrip("\r\n").split("\t")
            key = packet_key(values)
            if key is None:
                continue
            eligible_packets += 1
            timestamp = float(text(values, 1))
            frame_bytes = int(text(values, 2) or 0)
            captured_bytes = int(text(values, 3) or frame_bytes)
            parent = parents[key]
            parent["packet_count"] += 1
            parent["frame_bytes"] += frame_bytes
            parent["captured_bytes"] += captured_bytes
            parent["start"] = min(parent["start"], timestamp)
            parent["end"] = max(parent["end"], timestamp)

            state = states.setdefault(key, NativeState())
            new_syn = key[0] == "tcp" and flag(text(values, 11)) and not flag(text(values, 12))
            timed_out = state.last_timestamp is not None and timestamp - state.last_timestamp > timeout
            syn_starts_new = new_syn and state.current is not None and not state.syn_only
            if state.current is None or state.closed or timed_out or syn_starts_new:
                finalize_state(state, native_rows)
                state.current = new_native_row(capture, key, state.sequence, timestamp)
                state.sequence += 1
                state.closed = False
                state.syn_only = new_syn
            assert state.current is not None
            state.current["packet_count"] += 1
            state.current["total_bytes"] += frame_bytes
            state.current["captured_bytes"] += captured_bytes
            state.current["flow_end_time"] = timestamp
            state.last_timestamp = timestamp
            if not new_syn:
                state.syn_only = False
            if key[0] == "tcp" and (flag(text(values, 13)) or flag(text(values, 14))):
                state.closed = True
        returncode = process.wait()
    if returncode != 0:
        raise RuntimeError(f"tshark failed ({returncode}) for {pcap}; see {stderr_path}")
    for state in states.values():
        finalize_state(state, native_rows)

    expected_counter = Counter((int(row.packets), int(row.bytes), str(row.status)) for row in tf_rows.itertuples())
    observed_counter = Counter()
    for parent in parents.values():
        if parent["captured_bytes"] < 2048:
            status = "filtered:size_lt_2kb"
        elif parent["packet_count"] < 3:
            status = "filtered:packets_lt_3"
        else:
            status = "success"
        observed_counter[(parent["packet_count"], parent["captured_bytes"], status)] += 1
    tf_exact_parity = expected_counter == observed_counter

    cate_by_key = tfe_source["by_key"]
    for row in native_rows:
        key = (row["ip_protocol"].lower(), row["bidirectional_five_tuple"].split("|")[1], row["bidirectional_five_tuple"].split("|")[2])
        parent = parents[key]
        parent_id = hashlib.sha256(f"{pcap.resolve()}\0{flow_token(key)}".encode()).hexdigest()
        row.update(
            {
                "trafficformer_parent_flow_id": parent_id,
                "trafficformer_parent_packet_count": parent["packet_count"],
                "trafficformer_parent_frame_bytes": parent["frame_bytes"],
                "trafficformer_parent_captured_bytes": parent["captured_bytes"],
                "trafficformer_eligibility": bool(parent["captured_bytes"] >= 2048 and parent["packet_count"] >= 3),
                "trafficformer_audit_exact_parity": tf_exact_parity,
                "trafficformer_flow_relation": "NATIVE_SESSION_TO_CAPTURE_WIDE_BIDIRECTIONAL_5TUPLE",
            }
        )
        candidates = cate_by_key.get(key, [])
        if not candidates:
            cate_status, reason = "UNMATCHED", "NO_CATE_5TUPLE"
        elif len(candidates) > 1:
            cate_status, reason = "AMBIGUOUS", "MULTIPLE_CATE_ROWS_FOR_5TUPLE"
        elif int(candidates[0]["packet_count"]) != int(parent["packet_count"]):
            cate_status, reason = "MATCH_FAILED", "CATE_PARENT_PACKET_COUNT_MISMATCH"
        else:
            cate_status, reason = "MATCHED", "UNIQUE_CAPTURE_5TUPLE_PARENT"
        row["cate_match_status"] = cate_status
        row["cate_match_reason"] = reason
        row["cate_candidate_count"] = len(candidates)
        row["cate_flow_name"] = candidates[0]["name"] if len(candidates) == 1 else ""
        row["cate_service"] = candidates[0]["canonical_service"] if len(candidates) == 1 else ""
        row["cate_parent_packet_count"] = candidates[0]["packet_count"] if len(candidates) == 1 else ""
        row["cate_paper_text_eligible"] = candidates[0]["paper_text_eligible"] if len(candidates) == 1 else False
        row["cate_flow_relation"] = "NATIVE_SESSION_TO_CAPTURE_WIDE_CATE_5TUPLE"
        meta = membership.get(row["flow_id"])
        row["native_pool_membership"] = meta is not None
        row["native_image_sha256"] = meta["image_sha256"] if meta else ""
        row["native_roles"] = json.dumps(meta["roles"], sort_keys=True) if meta else "{}"
        row["native_medium_role"] = meta["roles"].get("medium", "") if meta else ""
        row["native_flow_definition"] = "bidirectional IPv4 TCP/UDP; 60s idle timeout; TCP SYN/FIN/RST boundaries"

    expected_selected = {
        uid for uid, meta in membership.items()
        if str(meta["source_file"]) == str(capture["source_file"])
    }
    observed_selected = {row["flow_id"] for row in native_rows if row["native_pool_membership"]}
    missing_selected = expected_selected - observed_selected
    if missing_selected:
        raise RuntimeError(f"failed to reconstruct {len(missing_selected)} frozen Native flows for {pcap}")
    audit = {
        "source_file": str(pcap.resolve()),
        "packet_rows": packet_rows,
        "ipv4_tcp_udp_packet_rows": eligible_packets,
        "native_sessions": len(native_rows),
        "native_selected_sessions": len(observed_selected),
        "trafficformer_parent_flows": len(parents),
        "trafficformer_expected_rows": len(tf_rows),
        "trafficformer_exact_multiset_parity": tf_exact_parity,
        "tfe_cate_parent_flows": sum(len(items) for items in cate_by_key.values()),
        "tfe_expected_flows": int(tfe_source["expected_flows"]),
        "stderr_path": str(stderr_path),
    }
    return native_rows, audit


def summarize_subset(frame: pd.DataFrame, set_id: str, definition: str) -> list[dict[str, Any]]:
    rows = []
    groupings: list[tuple[str, Iterable[tuple[str, pd.DataFrame]]]] = [
        ("overall", [("ALL", frame)]),
        ("application", frame.groupby("application", sort=True)),
        ("service", frame.groupby("service", sort=True)),
    ]
    for grouping, groups in groupings:
        for value, part in groups:
            packets = part["packet_count"].astype(int)
            values = part["total_bytes"].astype(float)
            rows.append(
                {
                    "set_id": set_id,
                    "set_definition": definition,
                    "grouping": grouping,
                    "group_value": value,
                    "flow_count": len(part),
                    "capture_count": part["source_file"].nunique(),
                    "one_packet_count": int((packets == 1).sum()),
                    "one_packet_ratio": float((packets == 1).mean()) if len(part) else math.nan,
                    "two_packet_count": int((packets == 2).sum()),
                    "two_packet_ratio": float((packets == 2).mean()) if len(part) else math.nan,
                    "short_le2_count": int((packets <= 2).sum()),
                    "short_le2_ratio": float((packets <= 2).mean()) if len(part) else math.nan,
                    "mean_packet_count": float(packets.mean()) if len(part) else math.nan,
                    "median_total_bytes": float(values.median()) if len(part) else math.nan,
                    "p25_total_bytes": float(values.quantile(0.25)) if len(part) else math.nan,
                    "p75_total_bytes": float(values.quantile(0.75)) if len(part) else math.nan,
                    "applications": "|".join(sorted(part["application"].unique())),
                    "services": "|".join(sorted(part["service"].unique())),
                }
            )
    return rows


def packet_group(count: int) -> str:
    if count == 1:
        return "1"
    if count == 2:
        return "2"
    if count <= 7:
        return "3-7"
    if count <= 15:
        return "8-15"
    return ">=16"


def analysis_row(frame: pd.DataFrame, dimension: str, value: str, class_names: list[str]) -> dict[str, Any]:
    true = frame["true_application"].to_numpy()
    pred = frame["predicted_application"].to_numpy()
    correct = true == pred
    present = sorted(set(true))
    return {
        "dataset": "ISCX-VPN",
        "protocol": "medium_seed2022_known_validation_shared31",
        "dimension": dimension,
        "group_value": value,
        "sample_count": len(frame),
        "correct_count": int(correct.sum()),
        "error_count": int((~correct).sum()),
        "accuracy": float(correct.mean()) if len(frame) else math.nan,
        "per_class_recall_interpretation": "accuracy within true-class row; otherwise see confusion_json",
        "macro_f1_within_subset": float(f1_score(true, pred, labels=present, average="macro", zero_division=0)) if present else math.nan,
        "macro_f1_class_set": "|".join(present),
        "macro_f1_warning": "subset/possibly missing classes; not comparable to global Macro-F1",
        "confusion_json": json.dumps(Counter(pred), sort_keys=True),
        "all_model_classes": "|".join(class_names),
    }


def build_error_analyses(flow_manifest: pd.DataFrame, shared_sources: set[str], split: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    val = pd.read_csv(RUN / "val_manifest.csv", dtype=str)
    bundle = np.load(RUN / "known_train_validation_features.npz", allow_pickle=False)
    y_true = bundle["validation_labels_reindexed"]
    y_pred = bundle["validation_m0_predictions"]
    config = read_json(RUN / "config.json")
    class_names = list(config["known_classes"])
    if len(val) != len(y_true) or len(val) != len(y_pred):
        raise RuntimeError("validation manifest/prediction length mismatch")
    true_names = np.asarray([class_names[int(value)] for value in y_true])
    predicted_names = np.asarray([class_names[int(value)] for value in y_pred])
    if not np.array_equal(true_names, val["canonical_class"].to_numpy()):
        raise RuntimeError("validation prediction order does not match val_manifest.csv")
    val = val.copy()
    val["true_application"] = true_names
    val["predicted_application"] = predicted_names
    val["correct"] = true_names == predicted_names
    shared_names = {Path(path).name for path in shared_sources}
    val = val[val["source_file"].map(lambda value: Path(value).name in shared_names)].copy()

    flow_lookup = flow_manifest[flow_manifest["native_pool_membership"]].set_index("flow_id", drop=False)
    missing = sorted(set(val["flow_id_sha256"]) - set(flow_lookup.index))
    if missing:
        raise RuntimeError(f"missing {len(missing)} shared validation flow rows")
    val["packet_count"] = val["flow_id_sha256"].map(flow_lookup["packet_count"]).astype(int)
    val["total_bytes"] = val["flow_id_sha256"].map(flow_lookup["total_bytes"]).astype(int)
    val["cate_match_status"] = val["flow_id_sha256"].map(flow_lookup["cate_match_status"])
    val["capture"] = val["source_file"]
    val["service"] = val["official_category"]
    val["packet_group"] = val["packet_count"].map(packet_group)

    mapping: dict[str, list[str]] = {}
    unique = split.drop_duplicates("flow_id_sha256")
    for application, group in unique.groupby("canonical_class", sort=True):
        mapping[application] = sorted(group["official_category"].unique())
    map_value = {application: values[0] if len(values) == 1 else "UNMAPPED" for application, values in mapping.items()}
    val["mapped_true_service"] = val["true_application"].map(map_value)
    val["mapped_predicted_service"] = val["predicted_application"].map(map_value)
    categories = []
    for row in val.itertuples():
        if row.correct:
            categories.append("FINE_CORRECT")
        elif row.mapped_true_service == "UNMAPPED" or row.mapped_predicted_service == "UNMAPPED":
            categories.append("FINE_WRONG_COARSE_UNMAPPED")
        elif row.mapped_true_service == row.mapped_predicted_service:
            categories.append("FINE_WRONG_COARSE_CORRECT")
        else:
            categories.append("FINE_WRONG_COARSE_WRONG")
    val["hierarchical_outcome"] = categories

    analyses = [analysis_row(val, "overall", "ALL", class_names)]
    for dimension in ("packet_group", "true_application", "service", "domain_state", "capture", "cate_match_status"):
        for value, group in val.groupby(dimension, sort=True):
            analyses.append(analysis_row(group, dimension, str(value), class_names))
    for dimensions in (
        ("true_application", "packet_group"),
        ("service", "packet_group"),
        ("capture", "packet_group"),
        ("cate_match_status", "packet_group"),
        ("cate_match_status", "service"),
        ("cate_match_status", "true_application"),
        ("cate_match_status", "capture"),
    ):
        for values, group in val.groupby(list(dimensions), sort=True):
            analyses.append(analysis_row(group, " x ".join(dimensions), " x ".join(map(str, values)), class_names))

    coarse_rows = []
    for dimension in ("overall", "true_application", "service", "capture", "packet_group", "domain_state"):
        groups = [("ALL", val)] if dimension == "overall" else val.groupby(dimension, sort=True)
        for value, group in groups:
            counts = Counter(group["hierarchical_outcome"])
            fine_errors = int((~group["correct"]).sum())
            mappable_fine_errors = counts["FINE_WRONG_COARSE_CORRECT"] + counts["FINE_WRONG_COARSE_WRONG"]
            coarse_rows.append(
                {
                    "dataset": "ISCX-VPN",
                    "protocol": "medium_seed2022_known_validation_shared31",
                    "dimension": dimension,
                    "group_value": value,
                    "sample_count": len(group),
                    "fine_correct": counts["FINE_CORRECT"],
                    "fine_wrong_coarse_correct": counts["FINE_WRONG_COARSE_CORRECT"],
                    "fine_wrong_coarse_wrong": counts["FINE_WRONG_COARSE_WRONG"],
                    "fine_wrong_coarse_unmapped": counts["FINE_WRONG_COARSE_UNMAPPED"],
                    "fine_error_count": fine_errors,
                    "mappable_fine_error_count": mappable_fine_errors,
                    "eliminated_fraction_among_all_fine_errors": counts["FINE_WRONG_COARSE_CORRECT"] / fine_errors if fine_errors else math.nan,
                    "eliminated_fraction_among_mappable_fine_errors": counts["FINE_WRONG_COARSE_CORRECT"] / mappable_fine_errors if mappable_fine_errors else math.nan,
                    "mapping_coverage": float(((group["mapped_true_service"] != "UNMAPPED") & (group["mapped_predicted_service"] != "UNMAPPED")).mean()),
                }
            )

    cate_rows = []
    for dimensions in (("cate_match_status",), ("cate_match_status", "packet_group"), ("cate_match_status", "true_application"), ("cate_match_status", "service"), ("cate_match_status", "capture")):
        for values, group in val.groupby(list(dimensions), sort=True):
            if not isinstance(values, tuple):
                values = (values,)
            cate_rows.append(
                {
                    "scope": "KNOWN_VALIDATION_SHARED31",
                    "dimensions": " x ".join(dimensions),
                    "group_value": " x ".join(map(str, values)),
                    "flow_count": len(group),
                    "error_count": int((~group["correct"]).sum()),
                    "error_rate": float((~group["correct"]).mean()),
                    "applications": "|".join(sorted(group["true_application"].unique())),
                    "services": "|".join(sorted(group["service"].unique())),
                    "causal_warning": "descriptive only; composition differs across CATE statuses",
                }
            )

    overall = coarse_rows[0]
    meta = {
        "known_validation_total_full_medium": int(len(y_true)),
        "known_validation_shared31": int(len(val)),
        "fine_accuracy_shared31": float(accuracy_score(val["true_application"], val["predicted_application"])),
        "fine_macro_f1_shared31": float(f1_score(val["true_application"], val["predicted_application"], labels=sorted(val["true_application"].unique()), average="macro", zero_division=0)),
        "fine_errors": int((~val["correct"]).sum()),
        "fine_wrong_coarse_correct": int(overall["fine_wrong_coarse_correct"]),
        "fine_wrong_coarse_unmapped": int(overall["fine_wrong_coarse_unmapped"]),
        "mapping_coverage": float(overall["mapping_coverage"]),
        "known_test_samples_used": 0,
        "unknown_test_samples_used": 0,
    }
    return val, pd.DataFrame(analyses), pd.DataFrame(coarse_rows), pd.DataFrame(cate_rows), meta


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tshark_stderr").mkdir(exist_ok=True)
    membership, split = load_native_membership()
    tfe_sources, skipped_empty, skipped_anomalous = load_tfe()
    tf_audit, tf_sources = load_tf_audit()
    capture_audit = pd.read_csv(CAPTURE_AUDIT)
    prep = read_json(PREP)
    timeout = float(read_json(S12 / "configs/stage12_config.json")["preprocessing"]["session_timeout_seconds"])

    prep_lookup = {}
    for class_spec in prep["classes"]:
        for capture_index, raw in enumerate(class_spec["captures"]):
            prep_lookup[str(Path(raw).resolve())] = (class_spec["name"], capture_index)
    service_by_source = (
        split.drop_duplicates("flow_id_sha256").groupby("source_file")["official_category"]
        .agg(lambda values: sorted(set(values))).to_dict()
    )
    selected_by_source = split.drop_duplicates("flow_id_sha256").groupby("source_file").size().to_dict()

    pcap_rows = []
    shared_paths = set(tfe_sources) & set(tf_sources)
    if len(shared_paths) != 31:
        raise RuntimeError(f"expected 31 TFE/TrafficFormer shared sources, found {len(shared_paths)}")
    for row in capture_audit.itertuples():
        path = str(Path(row.pcap_path).resolve())
        source_file = str(row.source_file)
        services = service_by_source.get(source_file, [])
        tfe_source = tfe_sources.get(path)
        tf_source = tf_sources.get(path)
        actual_hash = sha256_file(Path(path)) if path in shared_paths else str(row.pcap_sha256)
        if path in shared_paths and actual_hash != str(row.pcap_sha256):
            raise RuntimeError(f"shared PCAP hash changed: {path}")
        tfe_raw = sum(len(items) for items in tfe_source["by_key"].values()) if tfe_source else 0
        tfe_valid = sum(int(item["paper_text_eligible"]) for items in tfe_source["by_key"].values() for item in items) if tfe_source else 0
        pcap_rows.append(
            {
                "dataset": "ISCX-VPN",
                "capture_id": f"class={row.class_name}|capture={int(row.capture_index)}|name={Path(path).name}",
                "pcap_path": path,
                "source_file": source_file,
                "pcap_size_bytes": Path(path).stat().st_size,
                "pcap_sha256": actual_hash,
                "pcap_sha256_source": "live_rehash" if path in shared_paths else "Stage15R verified capture audit",
                "vpn_nonvpn": "VPN" if "VPN-PCAP" in source_file and not source_file.startswith("NonVPN") else "NonVPN",
                "tor_nontor": "NOT_APPLICABLE",
                "application": row.class_name,
                "service": "|".join(services) if services else "UNKNOWN",
                "native_included": True,
                "native_selected_flow_count": int(selected_by_source.get(source_file, 0)),
                "tfe_gnn_included": path in tfe_sources,
                "tfe_gnn_cate_flow_count": tfe_raw,
                "tfe_gnn_paper_text_valid_flow_count": tfe_valid,
                "trafficformer_included": path in tf_sources,
                "trafficformer_parent_flow_count": int(len(tf_source)) if tf_source is not None else 0,
                "trafficformer_eligible_flow_count": int((tf_source["status"] == "success").sum()) if tf_source is not None else 0,
                "shared_all_three": path in shared_paths,
                "native_filter": "IPv4 TCP/UDP; stable-hash cap 2000/application; no packet/byte minimum",
                "tfe_filter": "CATE TCP target five-tuples; paper-text removes empty payload and >10000-packet anomalous flows",
                "trafficformer_filter": "capture-wide bidirectional IPv4 TCP/UDP five-tuple; bytes <2048 first, then packets <3; captured packet bytes",
            }
        )
    pcap_frame = pd.DataFrame(pcap_rows).sort_values(["application", "source_file"])
    write_frame(OUT / "cross_project_pcap_manifest.csv", pcap_frame)

    shared_capture_rows = []
    for path in sorted(shared_paths):
        application, capture_index = prep_lookup[path]
        native_row = capture_audit[capture_audit["pcap_path"].map(lambda value: str(Path(value).resolve())) == path].iloc[0]
        services = service_by_source.get(str(native_row.source_file), [])
        if len(services) != 1:
            raise RuntimeError(f"shared capture lacks a unique capture-level service: {path} {services}")
        shared_capture_rows.append(
            {
                "pcap_path": path,
                "source_file": str(native_row.source_file),
                "application": application,
                "capture_index": int(capture_index),
                "service": services[0],
            }
        )

    all_flow_rows: list[dict[str, Any]] = []
    reconstruction_rows = []
    for index, capture in enumerate(shared_capture_rows, start=1):
        path = capture["pcap_path"]
        rows, audit = reconstruct_capture(capture, tfe_sources[path], tf_sources[path], membership, timeout)
        all_flow_rows.extend(rows)
        reconstruction_rows.append(audit)
        print(json.dumps({"event": "capture_complete", "index": index, "total": len(shared_capture_rows), **audit}), flush=True)
    flow_frame = pd.DataFrame(all_flow_rows).sort_values(["application", "source_file", "flow_start_time", "flow_id"])
    write_frame(OUT / "cross_project_flow_manifest.csv", flow_frame)
    write_frame(OUT / "flow_reconstruction_audit.csv", pd.DataFrame(reconstruction_rows))

    pool = flow_frame[flow_frame["native_pool_membership"]].copy()
    sets = {
        "A": (pool, "Native selected pool rows from the shared 31 PCAPs"),
        "B": (pool[pool["cate_match_status"] == "MATCHED"], "A intersect unique packet-count-consistent CATE five-tuple match"),
        "C": (pool[pool["trafficformer_eligibility"]], "A whose capture-wide TrafficFormer parent five-tuple passes >=2048 captured bytes and >=3 packets"),
        "D": (pool[(pool["cate_match_status"] == "MATCHED") & pool["trafficformer_eligibility"]], "B intersect C"),
    }
    eligibility_rows = []
    for name, (frame, definition) in sets.items():
        eligibility_rows.extend(summarize_subset(frame, name, definition))
    write_frame(OUT / "sample_eligibility_summary.csv", pd.DataFrame(eligibility_rows))

    validation, short_analysis, coarse_analysis, cate_validation, prediction_meta = build_error_analyses(flow_frame, shared_paths, split)
    write_frame(OUT / "known_validation_shared31_predictions.csv", validation)
    write_frame(OUT / "short_flow_error_analysis.csv", short_analysis)
    write_frame(OUT / "fine_to_coarse_error_analysis.csv", coarse_analysis)

    cate_pool_rows = []
    for dimensions in (("cate_match_status",), ("cate_match_status", "application"), ("cate_match_status", "service")):
        for values, group in pool.groupby(list(dimensions), sort=True):
            if not isinstance(values, tuple):
                values = (values,)
            cate_pool_rows.append(
                {
                    "scope": "NATIVE_POOL_SHARED31",
                    "dimensions": " x ".join(dimensions),
                    "group_value": " x ".join(map(str, values)),
                    "flow_count": len(group),
                    "error_count": "NOT_APPLICABLE",
                    "error_rate": "NOT_APPLICABLE",
                    "applications": "|".join(sorted(group["application"].unique())),
                    "services": "|".join(sorted(group["service"].unique())),
                    "causal_warning": "CATE status is a selection indicator, not authoritative label truth",
                }
            )
    write_frame(OUT / "cate_matching_analysis.csv", pd.concat([pd.DataFrame(cate_pool_rows), cate_validation], ignore_index=True))

    unique_split = split.drop_duplicates("flow_id_sha256")
    mapping = {}
    for application, group in unique_split.groupby("canonical_class", sort=True):
        services = sorted(group["official_category"].unique())
        mapping[application] = {
            "status": "MAPPED" if len(services) == 1 else "UNMAPPED_AMBIGUOUS",
            "service": services[0] if len(services) == 1 else "UNMAPPED",
            "possible_capture_level_services": services,
            "evidence": "Stage12 frozen split manifest official_category derived from capture/activity filename",
        }
    write_json(
        OUT / "label_mapping.json",
        {
            "dataset": "ISCX-VPN",
            "mapping_scope": "fixed application to service audit; no validation-driven merging",
            "application_mapping": mapping,
            "capture_level_service_is_available": True,
            "warning": "Facebook, Hangouts and Skype are multi-service applications; predicted application cannot be deterministically mapped to one service without an extra rule.",
        },
    )

    train_manifest = pd.read_csv(RUN / "train_manifest.csv", dtype=str)
    val_manifest = pd.read_csv(RUN / "val_manifest.csv", dtype=str)
    shared_names = {Path(path).name for path in shared_paths}
    train_shared = train_manifest[train_manifest["source_file"].map(lambda value: Path(value).name in shared_names)].copy()
    val_shared = val_manifest[val_manifest["source_file"].map(lambda value: Path(value).name in shared_names)].copy()
    support = {
        "fine_train": train_shared.groupby("canonical_class").size().to_dict(),
        "fine_validation": val_shared.groupby("canonical_class").size().to_dict(),
        "service_train": train_shared.groupby("official_category").size().to_dict(),
        "service_validation": val_shared.groupby("official_category").size().to_dict(),
    }
    known_names = set(read_json(RUN / "config.json")["known_classes"])
    ambiguous_known = sorted(
        name for name in known_names
        if mapping.get(name, {"status": "UNMAPPED"})["status"] != "MAPPED"
    )
    gate = {
        "flow_reconstruction": "PASS" if all(row["trafficformer_exact_multiset_parity"] for row in reconstruction_rows) else "FAIL",
        "shared_pcap_count": len(shared_paths),
        "native_selected_shared_flow_count": len(pool),
        "cate_status_counts": pool["cate_match_status"].value_counts().to_dict(),
        "trafficformer_exact_parity_captures": int(sum(row["trafficformer_exact_multiset_parity"] for row in reconstruction_rows)),
        "application_to_service_full_mapping": "FAIL" if ambiguous_known else "PASS",
        "ambiguous_known_applications": ambiguous_known,
        "prediction_mapping_coverage": prediction_meta["mapping_coverage"],
        "fine_to_coarse_full_same_flow_comparison": "BLOCKED" if prediction_meta["mapping_coverage"] < 1.0 else "PASS",
        "capture_group_split_status": "NOT_RUN_DQ6",
        "dq3_training_status": "NOT_RUN_PENDING_GATE_DECISION",
        "support": support,
        "known_test_samples_used": 0,
        "unknown_test_samples_used": 0,
    }
    write_json(OUT / "dq0_dq2_summary.json", {"gate": gate, "prediction": prediction_meta})
    print(json.dumps({"status": "DQ0_DQ2_COMPLETE", **gate, "prediction": prediction_meta}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
