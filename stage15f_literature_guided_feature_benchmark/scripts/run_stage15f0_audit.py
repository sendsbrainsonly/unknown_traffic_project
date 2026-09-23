#!/usr/bin/env python3
"""Aggregate Stage 15F-0 feature availability, packet-window, and encryption audits."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from common import PROJECT_ROOT, ROOT, WINDOWS, iter_csv, read_json, sha256_file, write_csv, write_json


@dataclass
class CoverageAccumulator:
    flow_count: int = 0
    packet_counts: Counter[int] = field(default_factory=Counter)
    packet_count_sum: int = 0
    total_frame_bytes: int = 0
    first_frame_bytes: int = 0
    total_duration: float = 0.0
    first_duration: float = 0.0
    complete_count: int = 0
    truncated_count: int = 0
    padded_flow_count: int = 0
    padding_ratio_sum: float = 0.0
    byte_coverage_sum: float = 0.0
    byte_coverage_count: int = 0
    time_coverage_sum: float = 0.0
    time_coverage_count: int = 0
    time_coverage_undefined: int = 0

    def add(self, row: dict[str, str], window: int) -> None:
        count = int(row["packet_count"])
        total_bytes = int(row["total_frame_bytes"])
        first_bytes = int(row[f"frame_bytes_first_{window}"])
        duration = float(row["duration_seconds"])
        first_duration = float(row[f"time_seconds_first_{window}"])
        self.flow_count += 1
        self.packet_counts[count] += 1
        self.packet_count_sum += count
        self.total_frame_bytes += total_bytes
        self.first_frame_bytes += first_bytes
        self.total_duration += duration
        self.first_duration += first_duration
        complete = row[f"complete_first_{window}"].lower() == "true"
        self.complete_count += int(complete)
        self.truncated_count += int(not complete)
        self.padded_flow_count += int(count < window)
        self.padding_ratio_sum += float(row[f"padding_ratio_first_{window}"])
        byte_coverage = row[f"byte_coverage_first_{window}"]
        if byte_coverage != "":
            self.byte_coverage_sum += float(byte_coverage)
            self.byte_coverage_count += 1
        time_coverage = row[f"time_coverage_first_{window}"]
        if time_coverage != "":
            self.time_coverage_sum += float(time_coverage)
            self.time_coverage_count += 1
        else:
            self.time_coverage_undefined += 1


def weighted_quantile(counts: Counter[int], q: float) -> float:
    total = sum(counts.values())
    if total <= 0:
        return math.nan
    target = q * (total - 1)
    cumulative = 0
    for value in sorted(counts):
        next_cumulative = cumulative + counts[value]
        if target < next_cumulative:
            return float(value)
        cumulative = next_cumulative
    return float(max(counts))


def load_memberships() -> tuple[dict[str, dict[str, list[tuple[str, str, str, str]]]], dict[tuple[str, str], str]]:
    memberships: dict[str, dict[str, list[tuple[str, str, str, str]]]] = {
        name: defaultdict(list) for name in ("ustc", "iscx_vpn", "iscx_tor", "vnat")
    }
    settings: dict[tuple[str, str], str] = {}
    for protocol in ("A-1", "A-2", "A-3"):
        path = PROJECT_ROOT / "stage3_unknown_utility" / "outputs" / protocol / "data_manifest.csv"
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row["known_or_unknown"].lower() != "known":
                    continue
                split = "known_train" if row["used_encoder_train"].lower() == "true" else (
                    "known_validation" if row["used_encoder_val"].lower() == "true" else ""
                )
                if split:
                    memberships["ustc"][row["flow_id"]].append((protocol, protocol, row["class_name"], split))
                    settings[("ustc", protocol)] = protocol
    for dataset in ("iscx_vpn", "iscx_tor"):
        path = PROJECT_ROOT / "stage12_dual_external_validation" / "protocol" / dataset / "split_manifest.csv"
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row["role"] not in {"known_train", "known_validation"}:
                    continue
                protocol = row["setting"]
                memberships[dataset][row["flow_id_sha256"]].append(
                    (protocol, protocol, row["canonical_class"], row["role"])
                )
                settings[(dataset, protocol)] = protocol
    path = PROJECT_ROOT / "stage14b_vnat_protocol_freeze" / "vnat_split_manifest.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["class_role"] != "known" or row["split"] not in {"train", "validation"}:
                continue
            split = "known_train" if row["split"] == "train" else "known_validation"
            memberships["vnat"][row["flow_uid"]].append(
                (row["protocol_id"], row["setting"], row["application"], split)
            )
            settings[("vnat", row["protocol_id"])] = row["setting"]
    for dataset, mapping in memberships.items():
        for uid, values in mapping.items():
            if len(values) != len(set(values)):
                raise RuntimeError(f"duplicate membership: {dataset} {uid}")
    return memberships, settings


def classify_encryption(dataset: str, row: dict[str, str]) -> tuple[str, str, str]:
    domain = row["domain_state"].lower()
    klass = row["class_name"].lower()
    tokens = set(row["protocol_tokens"].lower().split(":")) - {""}
    if domain == "vpn":
        return "UNDETERMINED_WITHIN_TUNNEL", "VPN", "frozen VPN metadata; no inner-flow claim"
    if domain == "tor":
        return "UNDETERMINED_WITHIN_TUNNEL", "TOR", "frozen Tor metadata; no inner-flow claim"
    if any(token.startswith("quic") for token in tokens):
        return "TLS_QUIC", "NONE_OBSERVED", "packet-layer protocol token"
    if any(token.startswith(("tls", "ssl")) for token in tokens):
        return "TLS_QUIC", "NONE_OBSERVED", "packet-layer protocol token"
    if any(token.startswith("ssh") for token in tokens):
        return "SSH", "NONE_OBSERVED", "packet-layer protocol token"
    if dataset == "vnat" and klass in {"ssh", "scp", "sftp", "rsync"}:
        return "SSH_OR_SECURE_TRANSFER_INFERRED", "NONE_OBSERVED", "application label inference; packet dissector did not prove every flow"
    if tokens & {"http", "ftp", "smtp", "imap", "pop", "pop3"}:
        return "VISIBLE_PLAINTEXT", "NONE_OBSERVED", "packet-layer protocol token"
    return "UNDETERMINED", "NONE_OBSERVED", "no decisive packet-layer or tunnel evidence"


def aggregate() -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    memberships, settings = load_memberships()
    coverage: dict[tuple[str, str, str, str, str, int], CoverageAccumulator] = defaultdict(CoverageAccumulator)
    encryption: Counter[tuple[str, str, str, str, str, str, str]] = Counter()
    seen_cache: dict[str, int] = {}
    unmatched_memberships: dict[str, int] = {}
    for dataset in ("ustc", "iscx_vpn", "iscx_tor", "vnat"):
        cache_path = ROOT / "window_cache" / dataset / "flow_window_metrics.csv.gz"
        seen = set()
        for row in iter_csv(cache_path):
            uid = row["flow_uid"]
            values = memberships[dataset].get(uid, [])
            if not values:
                raise RuntimeError(f"cache contains non-development flow: {dataset} {uid}")
            seen.add(uid)
            transport_regime, tunnel_regime, evidence = classify_encryption(dataset, row)
            for protocol, setting, class_name, split in values:
                for scope in (split, "known_train_validation_union"):
                    for group_class in (class_name, "__ALL__"):
                        for window in WINDOWS:
                            coverage[(dataset, protocol, setting, scope, group_class, window)].add(row, window)
                encryption[(dataset, protocol, setting, class_name, transport_regime, tunnel_regime, evidence)] += 1
        seen_cache[dataset] = len(seen)
        missing = set(memberships[dataset]) - seen
        unmatched_memberships[dataset] = len(missing)
        if missing:
            raise RuntimeError(f"{dataset}: {len(missing)} memberships missing from window cache")
    coverage_rows: list[dict[str, object]] = []
    for key in sorted(coverage):
        dataset, protocol, setting, scope, class_name, window = key
        agg = coverage[key]
        n = agg.flow_count
        coverage_rows.append(
            {
                "dataset": dataset,
                "protocol_id": protocol,
                "setting": setting,
                "role_scope": scope,
                "class_name": class_name,
                "window_packets": window,
                "flow_count": n,
                "packet_count_min": min(agg.packet_counts),
                "packet_count_p01": weighted_quantile(agg.packet_counts, 0.01),
                "packet_count_p25": weighted_quantile(agg.packet_counts, 0.25),
                "packet_count_median": weighted_quantile(agg.packet_counts, 0.50),
                "packet_count_p75": weighted_quantile(agg.packet_counts, 0.75),
                "packet_count_p99": weighted_quantile(agg.packet_counts, 0.99),
                "packet_count_max": max(agg.packet_counts),
                "packet_count_mean": agg.packet_count_sum / n,
                "complete_flow_count": agg.complete_count,
                "complete_coverage_ratio": agg.complete_count / n,
                "truncated_flow_count": agg.truncated_count,
                "truncation_ratio": agg.truncated_count / n,
                "padded_flow_count": agg.padded_flow_count,
                "padded_flow_ratio": agg.padded_flow_count / n,
                "padding_slot_ratio_mean": agg.padding_ratio_sum / n,
                "first_n_byte_coverage_mean_per_flow": agg.byte_coverage_sum / agg.byte_coverage_count if agg.byte_coverage_count else "",
                "first_n_byte_coverage_byte_weighted": agg.first_frame_bytes / agg.total_frame_bytes if agg.total_frame_bytes else "",
                "first_n_time_coverage_mean_per_flow": agg.time_coverage_sum / agg.time_coverage_count if agg.time_coverage_count else "",
                "first_n_time_coverage_duration_weighted": agg.first_duration / agg.total_duration if agg.total_duration else "",
                "time_coverage_defined_flows": agg.time_coverage_count,
                "time_coverage_undefined_zero_duration_truncated_flows": agg.time_coverage_undefined,
                "early_n_or_full_flow": "FULL_FLOW" if agg.complete_count == n else ("EARLY_N" if agg.complete_count == 0 else "MIXED_EARLY_N_AND_FULL_FLOW"),
                "byte_definition": "captured frame bytes",
                "time_definition": "elapsed first-to-Nth packet / full first-to-last duration",
            }
        )
    encryption_rows = [
        {
            "dataset": key[0],
            "protocol_id": key[1],
            "setting": key[2],
            "class_name": key[3],
            "application_transport_encryption": key[4],
            "tunnel_encryption": key[5],
            "evidence": key[6],
            "known_train_validation_flow_count": count,
        }
        for key, count in sorted(encryption.items())
    ]
    audit = {
        "status": "PASS",
        "known_test_feature_values_used": 0,
        "unknown_test_feature_values_used": 0,
        "cache_flow_counts": seen_cache,
        "unmatched_membership_counts": unmatched_memberships,
        "protocol_counts": {dataset: len({key[1] for key in coverage if key[0] == dataset}) for dataset in memberships},
        "coverage_row_count": len(coverage_rows),
        "encryption_row_count": len(encryption_rows),
    }
    return coverage_rows, encryption_rows, audit


def dataset_availability() -> list[dict[str, object]]:
    vnat_h5 = Path("/home/birkenwald/data/SEU-WXY/WXY-SEU/TrafficClassifier/Dataset/VNAT/VNAT/VNAT_Dataframe_release_1.h5")
    tor_csv_root = Path("/home/birkenwald/data/SEU-WXY/WXY-SEU/TrafficClassifier/Dataset/ISCXTor2016/ISCXTor2016（whole）/raw")
    rows = []
    dataset_roots = {
        "ustc": Path("/home/birkenwald/data/SEU-WXY/WXY-SEU/TrafficClassifier/Dataset/ustc-tfc2016"),
        "vnat": Path("/home/birkenwald/data/SEU-WXY/WXY-SEU/TrafficClassifier/Dataset/VNAT"),
        "iscx_vpn": Path("/home/birkenwald/data/SEU-WXY/WXY-SEU/TrafficClassifier/Dataset/ISCX-VPN"),
        "iscx_tor": Path("/home/birkenwald/data/SEU-WXY/WXY-SEU/TrafficClassifier/Dataset/ISCXTor2016"),
    }
    specifications = {
        "ustc": {
            "raw_source": "project data/flows/*.pkl derived from USTC PCAP; raw frames retained",
            "tabular_source": "data/flow_index.csv; Stage-0 PKLs",
            "lineage": "opendetect_ustc_encoder_audit/outputs/input_alignment_manifest.csv",
            "granularity": "bidirectional five-tuple; no timeout/session split",
            "flow_id": "PRESERVED",
            "packet_index": "DERIVABLE_FROM_STORED_ORDER",
            "timestamp": "PRESERVED",
            "direction": "PRESERVED_INITIATOR_RELATIVE",
            "frame_length": "PRESERVED_CAPLEN_AND_WIRELEN",
            "ip_length": "RECOVERABLE_FROM_RAW_FRAME_IPV4_HEADER",
            "transport_payload_length": "RECOVERABLE_FROM_RAW_FRAME",
            "header_bytes": "RECOVERABLE_FROM_RAW_FRAME_WITH_LAYER_PARSER",
            "payload_bytes": "RECOVERABLE_FROM_RAW_FRAME_WITH_LAYER_PARSER",
            "valid_packet_mask": "DERIVABLE_FROM_PACKET_COUNT_AND_WINDOW",
            "caveat": "Stage-0 filtered non-IPv4 and non-TCP/UDP; encryption regime not pre-labeled",
        },
        "vnat": {
            "raw_source": "VNAT_release_1 PCAPs",
            "tabular_source": f"official raw connection H5 present={vnat_h5.is_file()} ({vnat_h5.stat().st_size if vnat_h5.is_file() else 0} bytes); frozen Stage14B CSV",
            "lineage": "Stage14B flow_uid -> capture_id + tshark tcp.stream/udp.stream",
            "granularity": "tshark stream within capture",
            "flow_id": "PRESERVED",
            "packet_index": "RECOVERABLE_BY_PCAP_REPLAY",
            "timestamp": "RECOVERABLE_BY_PCAP_REPLAY",
            "direction": "RECOVERABLE_FIRST_ENDPOINT_RELATIVE",
            "frame_length": "RECOVERABLE_BY_PCAP_REPLAY",
            "ip_length": "RECOVERABLE_BY_PCAP_REPLAY",
            "transport_payload_length": "RECOVERABLE_BY_PCAP_REPLAY",
            "header_bytes": "RECOVERABLE_FROM_RAW_PACKET_WITH_LAYER_PARSER",
            "payload_bytes": "RECOVERABLE_FROM_RAW_PACKET_WITH_LAYER_PARSER",
            "valid_packet_mask": "DERIVABLE_FROM_PACKET_COUNT_AND_WINDOW",
            "caveat": "VPN metadata denotes outer tunnel; inner application headers/payload are not observable",
        },
        "iscx_vpn": {
            "raw_source": "ISCX-VPN aggregate PCAP/PCAPNG listed by frozen preprocessing manifest",
            "tabular_source": "Stage12 split_manifest.csv; no compatible official per-flow feature table used",
            "lineage": "Stage12 SHA256 session ID; bidirectional IPv4 TCP/UDP, 60s timeout, TCP boundaries",
            "granularity": "bidirectional session",
            "flow_id": "PRESERVED",
            "packet_index": "RECOVERABLE_BY_PCAP_REPLAY",
            "timestamp": "RECOVERABLE_BY_PCAP_REPLAY",
            "direction": "RECOVERABLE_FIRST_ENDPOINT_RELATIVE",
            "frame_length": "RECOVERABLE_BY_PCAP_REPLAY",
            "ip_length": "RECOVERABLE_BY_PCAP_REPLAY",
            "transport_payload_length": "RECOVERABLE_BY_PCAP_REPLAY",
            "header_bytes": "RECOVERABLE_FROM_RAW_PACKET_WITH_LAYER_PARSER",
            "payload_bytes": "RECOVERABLE_FROM_RAW_PACKET_WITH_LAYER_PARSER",
            "valid_packet_mask": "DERIVABLE_FROM_PACKET_COUNT_AND_WINDOW",
            "caveat": "VPN captures expose outer tunnel traffic, not inner application headers/payload",
        },
        "iscx_tor": {
            "raw_source": "ISCXTor aggregate PCAPs listed by frozen preprocessing manifest",
            "tabular_source": f"Scenario CSV files present={len(list(tor_csv_root.glob('*.csv')))}; not aligned to frozen Stage12 flow IDs",
            "lineage": "Stage12 SHA256 session ID; bidirectional IPv4 TCP/UDP, 60s timeout, TCP boundaries",
            "granularity": "bidirectional session",
            "flow_id": "PRESERVED",
            "packet_index": "RECOVERABLE_BY_PCAP_REPLAY",
            "timestamp": "RECOVERABLE_BY_PCAP_REPLAY",
            "direction": "RECOVERABLE_FIRST_ENDPOINT_RELATIVE",
            "frame_length": "RECOVERABLE_BY_PCAP_REPLAY",
            "ip_length": "RECOVERABLE_BY_PCAP_REPLAY",
            "transport_payload_length": "RECOVERABLE_BY_PCAP_REPLAY",
            "header_bytes": "RECOVERABLE_FROM_RAW_PACKET_WITH_LAYER_PARSER",
            "payload_bytes": "RECOVERABLE_FROM_RAW_PACKET_WITH_LAYER_PARSER",
            "valid_packet_mask": "DERIVABLE_FROM_PACKET_COUNT_AND_WINDOW",
            "caveat": "Tor captures expose outer Tor traffic, not inner application headers/payload",
        },
    }
    for dataset, spec in specifications.items():
        cache = ROOT / "window_cache" / dataset / "cache_audit.json"
        payload = read_json(cache)
        assert isinstance(payload, dict)
        local_root = dataset_roots[dataset]
        local_files = [path for path in local_root.rglob("*") if path.is_file()]
        pcaps = [path for path in local_files if path.suffix.lower() in {".pcap", ".pcapng"}]
        csvs = [path for path in local_files if path.suffix.lower() == ".csv"]
        h5s = [path for path in local_files if path.suffix.lower() in {".h5", ".hdf5"}]
        parquet = [path for path in local_files if path.suffix.lower() in {".parquet", ".feather"}]
        rows.append(
            {
                "dataset": dataset,
                **spec,
                "dataset_root": str(local_root),
                "pcap_file_count": len(pcaps),
                "pcap_total_bytes": sum(path.stat().st_size for path in pcaps),
                "csv_file_count": len(csvs),
                "h5_file_count": len(h5s),
                "parquet_or_feather_file_count": len(parquet),
                "tabular_total_bytes": sum(path.stat().st_size for path in csvs + h5s + parquet),
                "window_cache_status": payload["status"],
                "window_cache_scope": payload["scope"],
                "window_cache_flows": payload["recovered_flows"],
                "window_cache_sha256": sha256_file(cache),
                "unknown_test_feature_values_used": payload["unknown_test_values_stored"],
                "known_test_feature_values_used": payload["known_test_values_stored"],
            }
        )
    return rows


def main() -> None:
    coverage, encryption, audit = aggregate()
    coverage_fields = list(coverage[0])
    write_csv(ROOT / "packet_window_coverage.csv", coverage, coverage_fields)
    write_csv(ROOT / "encryption_visibility_audit.csv", encryption, list(encryption[0]))
    availability = dataset_availability()
    write_csv(ROOT / "dataset_feature_availability.csv", availability, list(availability[0]))
    audit.update(
        {
            "packet_window_coverage_sha256": sha256_file(ROOT / "packet_window_coverage.csv"),
            "encryption_visibility_audit_sha256": sha256_file(ROOT / "encryption_visibility_audit.csv"),
            "dataset_feature_availability_sha256": sha256_file(ROOT / "dataset_feature_availability.csv"),
        }
    )
    write_json(ROOT / "stage15f0_aggregation_audit.json", audit)
    print(json.dumps(audit, sort_keys=True))


if __name__ == "__main__":
    main()
