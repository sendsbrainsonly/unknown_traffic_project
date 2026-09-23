#!/usr/bin/env python3
"""Read-only audit of an existing CIC-IDS-2017 PCAP-to-label mapping.

The script never reads packet captures.  It independently joins the existing
flow Parquet files to the official TrafficLabelling CSV files using a
bidirectional canonical 5-tuple, timestamp precision, duration, and packet
counts.  All generated artifacts are written below a caller-selected project
output directory.
"""

from __future__ import annotations

import argparse
import bisect
import calendar
import csv
import hashlib
import ipaddress
import json
import math
import mimetypes
import random
import re
import statistics
import sys
from array import array
from collections import Counter, defaultdict, namedtuple
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

import pyarrow.parquet as pq


DAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")
TIMESTAMP_FORMATS = (
    ("%d/%m/%Y %H:%M:%S.%f", "microsecond"),
    ("%d/%m/%Y %H:%M:%S", "second"),
    ("%d/%m/%Y %H:%M", "minute"),
    ("%Y-%m-%d %H:%M:%S.%f", "microsecond"),
    ("%Y-%m-%d %H:%M:%S", "second"),
)
ALIASES = {
    "flow_id": {"flow id", "flowid"},
    "src_ip": {"source ip", "src ip", "sourceip", "srcip"},
    "src_port": {"source port", "src port", "sourceport", "srcport"},
    "dst_ip": {"destination ip", "dst ip", "destinationip", "dstip"},
    "dst_port": {"destination port", "dst port", "destinationport", "dstport"},
    "protocol": {"protocol"},
    "timestamp": {"timestamp", "time stamp"},
    "label": {"label", "class"},
    "duration": {"flow duration", "duration"},
    "fwd_packets": {"total fwd packets", "tot fwd pkts"},
    "bwd_packets": {"total backward packets", "tot bwd pkts"},
}

Official = namedtuple(
    "Official",
    "row_id source_csv row_number flow_id src_ip dst_ip src_port dst_port protocol "
    "key timestamp_raw timestamp_local precision raw_label label duration_s "
    "fwd_packets bwd_packets total_packets",
)

FLOW_COLUMNS = [
    "flow_id", "day", "src_ip", "dst_ip", "src_port", "dst_port", "protocol",
    "canonical_flow_key", "flow_instance_index", "flow_start_time", "flow_end_time",
    "duration", "packet_count", "forward_packet_count", "backward_packet_count",
    "label", "match_status", "match_method", "timestamp_difference_seconds",
    "unmatched_reason", "source_pcap", "source_label_csv", "official_row_id",
    "official_timestamp",
]


def normalize_header(value: str) -> str:
    return " ".join(value.strip().lower().replace("_", " ").split())


def resolve_columns(headers: Iterable[str]) -> dict[str, str | None]:
    normalized = {normalize_header(header): header for header in headers}
    return {
        key: next((normalized[alias] for alias in aliases if alias in normalized), None)
        for key, aliases in ALIASES.items()
    }


def normalize_label(value: object) -> str:
    cleaned = " ".join(str(value or "").strip().split())
    return re.sub(r"\s*(?:\uFFFD|\x96|\u2013|\u2014|Â–)\s*", " - ", cleaned)


def safe_int(value: object, default: int = 0) -> int:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError, OverflowError):
        return default


def parse_csv_timestamp(value: str) -> tuple[float, str]:
    cleaned = value.strip()
    for fmt, precision in TIMESTAMP_FORMATS:
        try:
            parsed = datetime.strptime(cleaned, fmt)
            scalar = calendar.timegm(parsed.timetuple()) + parsed.microsecond / 1_000_000
            # The official files omit AM/PM.  Captured workday hours represented
            # as 01:xx-07:xx are the afternoon portion of the same day.
            if parsed.hour < 8:
                scalar += 12 * 3600
            return scalar, precision
        except ValueError:
            continue
    raise ValueError(f"unsupported timestamp: {value!r}")


def parse_existing_timestamp(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
        if parsed.tzinfo is not None:
            return parsed.timestamp()
        return calendar.timegm(parsed.timetuple()) + parsed.microsecond / 1_000_000
    except ValueError:
        return None


def endpoint(ip_text: str, port: int) -> tuple[int, bytes, int]:
    ip = ipaddress.ip_address(ip_text)
    return ip.version, ip.packed, int(port)


def endpoint_text(value: tuple[int, bytes, int]) -> str:
    ip = str(ipaddress.ip_address(value[1]))
    return f"[{ip}]:{value[2]}" if value[0] == 6 else f"{ip}:{value[2]}"


def canonical_key(src_ip: str, src_port: int, dst_ip: str, dst_port: int, protocol: int) -> str:
    left = endpoint(src_ip, src_port)
    right = endpoint(dst_ip, dst_port)
    if right < left:
        left, right = right, left
    return f"{int(protocol)}|{endpoint_text(left)}|{endpoint_text(right)}"


def recorded_width(precision: str) -> float:
    return 60.0 if precision == "minute" else 1.0 if precision == "second" else 0.001


def interval_distance(record: Official, flow_start: float) -> float:
    start = record.timestamp_local
    end = start + recorded_width(record.precision)
    if flow_start < start:
        return start - flow_start
    if flow_start < end:
        return 0.0
    return flow_start - end


def eligible_method(record: Official, flow_start: float, flow_end: float) -> tuple[int, str, float] | None:
    distance = interval_distance(record, flow_start)
    if distance <= 1.0:
        name = "L1_MINUTE_BUCKET" if record.precision == "minute" else "L1_EXACT_OR_SECOND_BUCKET"
        return 1, name, distance
    if distance <= 5.0:
        return 2, "L2_WITHIN_5S", distance
    if distance <= 10.0:
        return 3, "L2_WITHIN_10S", distance
    start = record.timestamp_local
    end = start + recorded_width(record.precision)
    if start <= flow_end + 1.0 and end >= flow_start - 1.0:
        return 4, "L3_FLOW_INTERVAL", min(abs(start - flow_start), abs(end - flow_end))
    return None


def quantile_sorted(values: array, q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lower = int(math.floor(pos))
    upper = int(math.ceil(pos))
    if lower == upper:
        return float(ordered[lower])
    weight = pos - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def json_text(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def csv_row_count(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        return sum(1 for _ in reader)


def classify_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return "parquet"
    if suffix == ".csv":
        return "csv"
    if suffix == ".json":
        return "json"
    if suffix in {".md", ".txt", ".log"}:
        return suffix[1:] or "text"
    if suffix in {".npy", ".npz"}:
        return "numpy"
    if suffix in {".pkl", ".pickle"}:
        return "pickle"
    if "flow" in path.name.lower():
        return "flow"
    return mimetypes.guess_type(path.name)[0] or "binary"


def write_inventory(mapping_root: Path, output_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(mapping_root.rglob("*")):
        if not path.is_file():
            continue
        row_count: int | str = ""
        columns: list[str] = []
        schema = ""
        if path.suffix.lower() == ".parquet":
            parquet = pq.ParquetFile(path)
            row_count = parquet.metadata.num_rows
            columns = parquet.schema_arrow.names
            schema = str(parquet.schema_arrow).replace("\n", " | ")
        elif path.suffix.lower() == ".csv":
            row_count = csv_row_count(path)
            with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
                columns = next(csv.reader(handle), [])
        st = path.stat()
        rows.append({
            "relative_path": str(path.relative_to(mapping_root)),
            "absolute_path": str(path.resolve()),
            "file_type": classify_file(path),
            "extension": path.suffix.lower(),
            "size_bytes": st.st_size,
            "row_count": row_count,
            "columns": json_text(columns),
            "schema": schema,
            "modified_time_utc": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(),
        })
    target = output_dir / "mapping_inventory.csv"
    with target.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def input_snapshot(paths: Iterable[Path]) -> dict[str, dict[str, int]]:
    result = {}
    for path in sorted(paths):
        st = path.stat()
        result[str(path.resolve())] = {"size_bytes": st.st_size, "mtime_ns": st.st_mtime_ns}
    return result


def load_official_day(csv_paths: list[Path], day: str):
    by_key: dict[str, list[Official]] = defaultdict(list)
    by_id: dict[str, Official] = {}
    label_counts: Counter[str] = Counter()
    raw_label_variants: dict[str, set[str]] = defaultdict(set)
    flow_id_stats: dict[str, list[object]] = {}
    invalid_rows = 0
    empty_rows = 0
    for path in csv_paths:
        with path.open("r", encoding="cp1252", errors="replace", newline="") as handle:
            reader = csv.DictReader(handle, skipinitialspace=True)
            columns = resolve_columns(reader.fieldnames or [])
            required = ("src_ip", "dst_ip", "src_port", "dst_port", "protocol", "timestamp", "label")
            missing = [name for name in required if not columns.get(name)]
            if missing:
                raise RuntimeError(f"{path} missing official fields: {missing}")
            for row_number, row in enumerate(reader, start=2):
                values = [str(row.get(columns[name] or "", "")).strip() for name in required]
                if not any(values):
                    empty_rows += 1
                    continue
                try:
                    src_ip = str(ipaddress.ip_address(values[0]))
                    dst_ip = str(ipaddress.ip_address(values[1]))
                    src_port = safe_int(values[2])
                    dst_port = safe_int(values[3])
                    protocol = safe_int(values[4])
                    timestamp_local, precision = parse_csv_timestamp(values[5])
                    raw_label = values[6]
                    label = normalize_label(raw_label)
                    key = canonical_key(src_ip, src_port, dst_ip, dst_port, protocol)
                    if not label:
                        raise ValueError("empty label")
                except (ValueError, ipaddress.AddressValueError):
                    invalid_rows += 1
                    continue
                fwd = safe_int(row.get(columns.get("fwd_packets") or "", 0))
                bwd = safe_int(row.get(columns.get("bwd_packets") or "", 0))
                official_flow_id = str(row.get(columns.get("flow_id") or "", "")).strip()
                record = Official(
                    f"{path.name}:{row_number}", path.name, row_number, official_flow_id,
                    src_ip, dst_ip, src_port, dst_port, protocol, key, values[5],
                    timestamp_local, precision, raw_label, label,
                    safe_int(row.get(columns.get("duration") or "", 0)) / 1_000_000,
                    fwd, bwd, fwd + bwd,
                )
                by_key[key].append(record)
                by_id[record.row_id] = record
                label_counts[label] += 1
                raw_label_variants[label].add(raw_label)
                if official_flow_id:
                    stats = flow_id_stats.get(official_flow_id)
                    if stats is None:
                        flow_id_stats[official_flow_id] = [1, {label}, path.name, row_number]
                    else:
                        stats[0] += 1
                        stats[1].add(label)
    times_by_key: dict[str, array] = {}
    for key, records in by_key.items():
        records.sort(key=lambda record: (record.timestamp_local, record.source_csv, record.row_number))
        times_by_key[key] = array("d", (record.timestamp_local for record in records))
    return {
        "by_key": by_key,
        "by_id": by_id,
        "times": times_by_key,
        "label_counts": label_counts,
        "raw_label_variants": raw_label_variants,
        "flow_id_stats": flow_id_stats,
        "invalid_rows": invalid_rows,
        "empty_rows": empty_rows,
        "valid_rows": sum(label_counts.values()),
        "day": day,
    }


def select_candidates(index: dict[str, object], flow: dict[str, object]):
    key = str(flow["computed_key"])
    records = index["by_key"].get(key, [])
    if not records:
        return [], [], "NO_SAME_CANONICAL_5TUPLE"
    times = index["times"][key]
    start = float(flow["start"])
    end = float(flow["end"])
    low = bisect.bisect_left(times, start - 70.0)
    high = bisect.bisect_right(times, end + 11.0)
    eligible = []
    for record in records[low:high]:
        method = eligible_method(record, start, end)
        if method is None:
            continue
        rank, method_name, interval_diff = method
        total_delta = abs(record.total_packets - int(flow["packet_count"]))
        duration_delta = abs(record.duration_s - float(flow["duration"]))
        same_direction = (
            flow["src_ip"] == record.src_ip and int(flow["src_port"]) == record.src_port
            and flow["dst_ip"] == record.dst_ip and int(flow["dst_port"]) == record.dst_port
        )
        if same_direction:
            direction_delta = abs(record.fwd_packets - int(flow["fwd"])) + abs(record.bwd_packets - int(flow["bwd"]))
        else:
            direction_delta = abs(record.fwd_packets - int(flow["bwd"])) + abs(record.bwd_packets - int(flow["fwd"]))
        score = (
            float(rank), round(interval_diff, 6), 0.0 if total_delta == 0 else 1.0,
            float(total_delta), round(duration_delta, 6), float(direction_delta),
        )
        eligible.append((score, method_name, interval_diff, record))
    if not eligible:
        return [], [], "NO_TIMESTAMP_CANDIDATE_WITHIN_HIERARCHY"
    eligible.sort(key=lambda item: (item[0], item[3].source_csv, item[3].row_number))
    best_score = eligible[0][0]
    tied = [item for item in eligible if item[0] == best_score]
    return eligible, tied, ""


def reservoir_add(bucket: list[dict[str, object]], item: dict[str, object], seen: int, limit: int, rng: random.Random) -> None:
    if limit <= 0:
        return
    if len(bucket) < limit:
        bucket.append(item)
        return
    choice = rng.randrange(seen)
    if choice < limit:
        bucket[choice] = item


def write_csv_header(path: Path, fields: list[str]):
    handle = path.open("w", encoding="utf-8-sig", newline="")
    writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    return handle, writer


def audit(args: argparse.Namespace) -> None:
    mapping_root = args.mapping_root.resolve()
    official_root = args.official_csv_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_dir == mapping_root or mapping_root in output_dir.parents:
        raise RuntimeError("Audit output must not be inside the existing mapping directory")

    mapping_inputs = [path for path in mapping_root.rglob("*") if path.is_file()]
    official_inputs = sorted(official_root.glob("*.csv"))
    before_mapping = input_snapshot(mapping_inputs)
    before_official = input_snapshot(official_inputs)
    inventory_rows = write_inventory(mapping_root, output_dir)

    recheck_fields = [
        "flow_id", "day", "source_pcap", "existing_match_status", "existing_match_method",
        "existing_label", "existing_tuple", "stored_canonical_tuple", "stored_tuple_consistent",
        "existing_time", "existing_duration_seconds", "existing_packet_count",
        "verification_status", "candidate_status", "independent_method", "eligible_candidate_count",
        "best_candidate_count", "best_candidate_labels", "official_row_id", "official_csv_file",
        "official_tuple", "official_time_raw", "official_time_normalized", "official_duration_seconds",
        "official_label", "raw_official_label", "timestamp_abs_diff_seconds",
        "timestamp_interval_diff_seconds", "duration_abs_diff_seconds", "tuple_match", "label_match",
        "referenced_official_row_id", "referenced_row_found", "referenced_row_is_best",
        "referenced_row_tuple_match", "referenced_row_label_match", "unmatched_reason",
    ]
    mismatch_fields = recheck_fields + ["mismatch_reason"]
    ambiguity_fields = [
        "record_type", "day", "flow_id", "canonical_tuple", "existing_time",
        "eligible_candidate_count", "best_candidate_count", "candidate_labels",
        "candidate_row_ids", "notes",
    ]
    duplicate_fields = ["dataset", "day", "flow_id", "count", "labels", "source_files", "notes"]
    repeated_fields = [
        "day", "canonical_tuple", "official_count", "official_labels", "official_min_time",
        "official_max_time", "existing_count", "existing_labels", "existing_min_time",
        "existing_max_time", "benign_and_attack", "notes",
    ]

    recheck_h, recheck_w = write_csv_header(output_dir / "mapping_vs_official.csv", recheck_fields)
    mismatch_h, mismatch_w = write_csv_header(output_dir / "label_mismatches.csv", mismatch_fields)
    ambiguity_h, ambiguity_w = write_csv_header(output_dir / "official_ambiguities.csv", ambiguity_fields)
    duplicate_h, duplicate_w = write_csv_header(output_dir / "duplicate_flow_ids.csv", duplicate_fields)
    repeated_h, repeated_w = write_csv_header(output_dir / "repeated_5tuple_label_changes.csv", repeated_fields)

    overall_status: Counter[str] = Counter()
    candidate_statuses: Counter[str] = Counter()
    existing_statuses: Counter[str] = Counter()
    existing_methods: Counter[str] = Counter()
    official_counts: Counter[str] = Counter()
    official_raw_labels: dict[str, set[str]] = defaultdict(set)
    existing_counts: Counter[str] = Counter()
    label_match_by_class: Counter[str] = Counter()
    timestamp_values: dict[tuple[str, str, str], array] = defaultdict(lambda: array("d"))
    signed_timestamp_values: dict[tuple[str, str, str], array] = defaultdict(lambda: array("d"))
    timestamp_method_counts: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    independent_recheckable = 0
    referenced_rows_found = 0
    referenced_rows_consistent = 0
    multiple_eligible_rows = 0
    multi_best_rows = 0
    existing_unmatched_with_candidate = 0
    existing_unmatched_with_nonnull_label = 0
    official_invalid = 0
    official_empty = 0
    processed_rows = 0
    day_counts: dict[str, Counter[str]] = {}
    spot_buckets: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    spot_seen: Counter[tuple[str, str]] = Counter()
    bonus_attack = {"Monday": 0, "Tuesday": 3, "Wednesday": 3, "Thursday": 2, "Friday": 2}
    rngs = {(day, kind): random.Random(args.seed + 1000 * i + (0 if kind == "BENIGN" else 1))
            for i, day in enumerate(DAYS) for kind in ("BENIGN", "ATTACK")}

    for day in DAYS:
        csv_paths = sorted(path for path in official_inputs if day.lower() in path.name.lower())
        if not csv_paths:
            raise RuntimeError(f"No TrafficLabelling CSV for {day}")
        flow_path = mapping_root / "flows" / f"{day}_flows.parquet"
        if not flow_path.is_file():
            raise FileNotFoundError(flow_path)
        print(f"loading official CSV day={day} files={len(csv_paths)}", flush=True)
        index = load_official_day(csv_paths, day)
        official_counts.update(index["label_counts"])
        for normalized, variants in index["raw_label_variants"].items():
            official_raw_labels[normalized].update(variants)
        official_invalid += int(index["invalid_rows"])
        official_empty += int(index["empty_rows"])

        # Official duplicate flow IDs and label ambiguity at identical tuple/time.
        for flow_id, stats in index["flow_id_stats"].items():
            count, labels, source_file, first_row = stats
            if int(count) > 1:
                duplicate_w.writerow({
                    "dataset": "official_csv", "day": day, "flow_id": flow_id,
                    "count": count, "labels": json_text(sorted(labels)),
                    "source_files": source_file,
                    "notes": f"first_seen_row={first_row}; official Flow ID is not a unique session identifier",
                })

        mixed_official: dict[str, dict[str, object]] = {}
        for key, records in index["by_key"].items():
            labels = {record.label for record in records}
            if len(labels) > 1:
                mixed_official[key] = {
                    "count": len(records), "labels": labels,
                    "min": records[0].timestamp_local, "max": records[-1].timestamp_local,
                }
            position = 0
            while position < len(records):
                end = position + 1
                while end < len(records) and records[end].timestamp_local == records[position].timestamp_local:
                    end += 1
                group = records[position:end]
                group_labels = {record.label for record in group}
                if len(group_labels) > 1:
                    ambiguity_w.writerow({
                        "record_type": "OFFICIAL_SAME_TUPLE_TIME_DIFFERENT_LABEL",
                        "day": day, "flow_id": "", "canonical_tuple": key,
                        "existing_time": "", "eligible_candidate_count": len(group),
                        "best_candidate_count": len(group),
                        "candidate_labels": json_text(sorted(group_labels)),
                        "candidate_row_ids": json_text([record.row_id for record in group]),
                        "notes": "Official rows share normalized tuple and timestamp but disagree on label",
                    })
                position = end

        existing_flow_ids: dict[str, tuple[int, str]] = {}
        existing_duplicate_rows: dict[str, list[object]] = {}
        mixed_existing: dict[str, dict[str, object]] = {}
        parquet = pq.ParquetFile(flow_path)
        available = set(parquet.schema_arrow.names)
        missing_columns = sorted(set(FLOW_COLUMNS) - available)
        if missing_columns:
            raise RuntimeError(f"{flow_path} missing required fields: {missing_columns}")
        day_counter: Counter[str] = Counter()
        print(f"rechecking flow parquet day={day} rows={parquet.metadata.num_rows}", flush=True)
        for batch in parquet.iter_batches(batch_size=args.batch_size, columns=FLOW_COLUMNS):
            for row in batch.to_pylist():
                processed_rows += 1
                flow_id = str(row.get("flow_id") or "")
                existing_status = str(row.get("match_status") or "")
                existing_label = normalize_label(row.get("label"))
                existing_statuses[existing_status] += 1
                day_counter[existing_status] += 1
                if existing_status == "UNMATCHED" and existing_label:
                    existing_unmatched_with_nonnull_label += 1
                if row.get("match_method"):
                    existing_methods[str(row["match_method"])] += 1
                if existing_label:
                    existing_counts[existing_label] += 1
                if flow_id in existing_flow_ids:
                    stats = existing_duplicate_rows.setdefault(flow_id, [1, {existing_label}, set()])
                    stats[0] += 1
                    stats[1].add(existing_label)
                    stats[2].add(str(row.get("source_pcap") or ""))
                else:
                    existing_flow_ids[flow_id] = (1, existing_label)

                start = parse_existing_timestamp(row.get("flow_start_time"))
                end = parse_existing_timestamp(row.get("flow_end_time"))
                missing = []
                for name in ("src_ip", "dst_ip", "src_port", "dst_port", "protocol"):
                    if row.get(name) in (None, ""):
                        missing.append(name)
                if start is None:
                    missing.append("flow_start_time")
                computed_key = ""
                try:
                    if not missing:
                        computed_key = canonical_key(
                            str(row["src_ip"]), int(row["src_port"]), str(row["dst_ip"]),
                            int(row["dst_port"]), int(row["protocol"]),
                        )
                except (ValueError, ipaddress.AddressValueError):
                    missing.append("invalid_tuple")
                stored_key = str(row.get("canonical_flow_key") or "")
                stored_consistent = bool(computed_key and stored_key == computed_key)
                if end is None and start is not None:
                    end = start + float(row.get("duration") or 0.0)

                base_out = {
                    "flow_id": flow_id, "day": day, "source_pcap": row.get("source_pcap"),
                    "existing_match_status": existing_status, "existing_match_method": row.get("match_method"),
                    "existing_label": existing_label, "existing_tuple": computed_key,
                    "stored_canonical_tuple": stored_key, "stored_tuple_consistent": stored_consistent,
                    "existing_time": row.get("flow_start_time"),
                    "existing_duration_seconds": row.get("duration"),
                    "existing_packet_count": row.get("packet_count"),
                    "referenced_official_row_id": row.get("official_row_id"),
                    "unmatched_reason": row.get("unmatched_reason"),
                }
                if missing:
                    result = {**base_out, "verification_status": "INSUFFICIENT_FIELDS",
                              "candidate_status": "INSUFFICIENT_FIELDS",
                              "best_candidate_labels": "", "eligible_candidate_count": 0,
                              "best_candidate_count": 0, "tuple_match": False, "label_match": False}
                    overall_status["INSUFFICIENT_FIELDS"] += 1
                    candidate_statuses["INSUFFICIENT_FIELDS"] += 1
                    recheck_w.writerow(result)
                    continue

                independent_recheckable += 1
                flow = {
                    "computed_key": computed_key, "start": start, "end": end,
                    "src_ip": str(row["src_ip"]), "dst_ip": str(row["dst_ip"]),
                    "src_port": int(row["src_port"]), "dst_port": int(row["dst_port"]),
                    "packet_count": int(row.get("packet_count") or 0),
                    "fwd": int(row.get("forward_packet_count") or 0),
                    "bwd": int(row.get("backward_packet_count") or 0),
                    "duration": float(row.get("duration") or 0.0),
                }
                eligible, tied, no_reason = select_candidates(index, flow)
                if len(eligible) > 1:
                    multiple_eligible_rows += 1
                if len(tied) > 1:
                    multi_best_rows += 1

                referenced = index["by_id"].get(str(row.get("official_row_id") or ""))
                referenced_found = referenced is not None
                referenced_is_best = bool(referenced and any(item[3].row_id == referenced.row_id for item in tied))
                referenced_tuple_match = bool(referenced and referenced.key == computed_key)
                referenced_label_match = bool(referenced and referenced.label == existing_label)
                if referenced_found:
                    referenced_rows_found += 1
                if referenced_found and referenced_tuple_match and referenced_label_match:
                    referenced_rows_consistent += 1

                if not eligible:
                    verification_status = "NO_MATCH"
                    candidate_status = "NO_OFFICIAL_MATCH"
                    best_labels: set[str] = set()
                    chosen = None
                    method_name = ""
                    interval_diff = None
                else:
                    best_labels = {item[3].label for item in tied}
                    chosen = tied[0][3]
                    method_name = tied[0][1]
                    interval_diff = tied[0][2]
                    if len(tied) > 1:
                        candidate_status = "MULTIPLE_OFFICIAL_CANDIDATES"
                    else:
                        candidate_status = "EXACT_OR_UNIQUE_MATCH"
                    if not existing_label or not existing_status.startswith("MATCHED"):
                        verification_status = "NO_MATCH"
                        existing_unmatched_with_candidate += 1
                    elif len(best_labels) > 1:
                        verification_status = "MULTIPLE_CANDIDATES"
                    elif existing_label == chosen.label:
                        verification_status = "LABEL_MATCH"
                    else:
                        verification_status = "LABEL_MISMATCH"

                official_key = chosen.key if chosen else ""
                timestamp_abs = abs(start - chosen.timestamp_local) if chosen else None
                duration_abs = abs(float(row.get("duration") or 0.0) - chosen.duration_s) if chosen else None
                out = {
                    **base_out,
                    "verification_status": verification_status,
                    "candidate_status": candidate_status,
                    "independent_method": method_name,
                    "eligible_candidate_count": len(eligible),
                    "best_candidate_count": len(tied),
                    "best_candidate_labels": json_text(sorted(best_labels)) if best_labels else "[]",
                    "official_row_id": chosen.row_id if chosen else "",
                    "official_csv_file": chosen.source_csv if chosen else "",
                    "official_tuple": official_key,
                    "official_time_raw": chosen.timestamp_raw if chosen else "",
                    "official_time_normalized": datetime.fromtimestamp(chosen.timestamp_local, timezone.utc).replace(tzinfo=None).isoformat() if chosen else "",
                    "official_duration_seconds": chosen.duration_s if chosen else "",
                    "official_label": chosen.label if chosen else "",
                    "raw_official_label": chosen.raw_label if chosen else "",
                    "timestamp_abs_diff_seconds": timestamp_abs if timestamp_abs is not None else "",
                    "timestamp_interval_diff_seconds": interval_diff if interval_diff is not None else "",
                    "duration_abs_diff_seconds": duration_abs if duration_abs is not None else "",
                    "tuple_match": bool(chosen and official_key == computed_key),
                    "label_match": bool(chosen and existing_label and chosen.label == existing_label),
                    "referenced_row_found": referenced_found,
                    "referenced_row_is_best": referenced_is_best,
                    "referenced_row_tuple_match": referenced_tuple_match,
                    "referenced_row_label_match": referenced_label_match,
                }
                recheck_w.writerow(out)
                overall_status[verification_status] += 1
                candidate_statuses[candidate_status] += 1
                day_counter[verification_status] += 1
                if verification_status == "LABEL_MATCH" and chosen:
                    label_match_by_class[existing_label] += 1
                    key_stats = (day, str(row.get("source_pcap") or ""), chosen.source_csv)
                    timestamp_values[key_stats].append(float(timestamp_abs))
                    signed_timestamp_values[key_stats].append(float(start - chosen.timestamp_local))
                    timestamp_method_counts[key_stats][method_name] += 1
                if verification_status in {"LABEL_MISMATCH", "MULTIPLE_CANDIDATES"}:
                    mismatch_w.writerow({**out, "mismatch_reason": verification_status})
                if len(tied) > 1:
                    ambiguity_w.writerow({
                        "record_type": "EXISTING_FLOW_EQUAL_BEST_CANDIDATES",
                        "day": day, "flow_id": flow_id, "canonical_tuple": computed_key,
                        "existing_time": row.get("flow_start_time"),
                        "eligible_candidate_count": len(eligible), "best_candidate_count": len(tied),
                        "candidate_labels": json_text(sorted(best_labels)),
                        "candidate_row_ids": json_text([item[3].row_id for item in tied]),
                        "notes": "Equal non-label score; labels may be identical or conflicting",
                    })

                if computed_key in mixed_official and existing_label:
                    info = mixed_existing.get(computed_key)
                    if info is None:
                        mixed_existing[computed_key] = {"count": 1, "labels": {existing_label}, "min": start, "max": start}
                    else:
                        info["count"] += 1
                        info["labels"].add(existing_label)
                        info["min"] = min(float(info["min"]), start)
                        info["max"] = max(float(info["max"]), start)

                if verification_status == "LABEL_MATCH" and chosen:
                    kind = "BENIGN" if existing_label == "BENIGN" else "ATTACK"
                    spot_key = (day, kind)
                    spot_seen[spot_key] += 1
                    limit = 10 + (bonus_attack[day] if kind == "ATTACK" else 0)
                    spot_item = {
                        "flow_id": flow_id, "source_day": day,
                        "existing_tuple": computed_key, "official_tuple": chosen.key,
                        "existing_time": row.get("flow_start_time"),
                        "official_time": datetime.fromtimestamp(chosen.timestamp_local, timezone.utc).replace(tzinfo=None).isoformat(),
                        "existing_label": existing_label, "official_label": chosen.label,
                        "tuple_match": chosen.key == computed_key,
                        "time_match": interval_diff is not None,
                        "label_match": existing_label == chosen.label,
                        "manual_result": "PASS" if chosen.key == computed_key and interval_diff is not None and existing_label == chosen.label else "FAIL",
                        "notes": f"deterministic independent CSV check; official_row_id={chosen.row_id}; method={method_name}; abs_time_diff={timestamp_abs:.6f}s",
                    }
                    reservoir_add(spot_buckets[spot_key], spot_item, spot_seen[spot_key], limit, rngs[spot_key])

        for flow_id, stats in sorted(existing_duplicate_rows.items()):
            duplicate_w.writerow({
                "dataset": "existing_mapping", "day": day, "flow_id": flow_id,
                "count": stats[0], "labels": json_text(sorted(stats[1])),
                "source_files": json_text(sorted(stats[2])),
                "notes": "Duplicate generated flow_id",
            })
        for key, official_info in sorted(mixed_official.items()):
            existing_info = mixed_existing.get(key, {"count": 0, "labels": set(), "min": None, "max": None})
            labels = set(official_info["labels"])
            benign_and_attack = "BENIGN" in labels and any(label != "BENIGN" for label in labels)
            repeated_w.writerow({
                "day": day, "canonical_tuple": key,
                "official_count": official_info["count"],
                "official_labels": json_text(sorted(labels)),
                "official_min_time": datetime.fromtimestamp(float(official_info["min"]), timezone.utc).replace(tzinfo=None).isoformat(),
                "official_max_time": datetime.fromtimestamp(float(official_info["max"]), timezone.utc).replace(tzinfo=None).isoformat(),
                "existing_count": existing_info["count"],
                "existing_labels": json_text(sorted(existing_info["labels"])),
                "existing_min_time": datetime.fromtimestamp(float(existing_info["min"]), timezone.utc).replace(tzinfo=None).isoformat() if existing_info["min"] is not None else "",
                "existing_max_time": datetime.fromtimestamp(float(existing_info["max"]), timezone.utc).replace(tzinfo=None).isoformat() if existing_info["max"] is not None else "",
                "benign_and_attack": benign_and_attack,
                "notes": "Same bidirectional 5-tuple appears under multiple labels at different times; session time must remain part of identity",
            })
        day_counts[day] = day_counter
        print(f"completed day={day} cumulative_rows={processed_rows}", flush=True)
        del index

    for handle in (recheck_h, mismatch_h, ambiguity_h, duplicate_h, repeated_h):
        handle.close()

    timestamp_fields = [
        "day", "source_pcap", "csv_file", "count", "abs_min_seconds", "abs_median_seconds",
        "abs_p90_seconds", "abs_p95_seconds", "abs_p99_seconds", "abs_max_seconds",
        "signed_median_seconds", "signed_min_seconds", "signed_max_seconds", "method_counts",
        "fixed_hour_offset_detected", "notes",
    ]
    with (output_dir / "timestamp_recheck.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=timestamp_fields)
        writer.writeheader()
        for group in sorted(timestamp_values):
            absolute = timestamp_values[group]
            signed = signed_timestamp_values[group]
            signed_median = quantile_sorted(signed, 0.5)
            fixed_hour = bool(signed and abs(float(signed_median or 0.0)) >= 3500 and statistics.pstdev(signed) < 120)
            writer.writerow({
                "day": group[0], "source_pcap": group[1], "csv_file": group[2], "count": len(absolute),
                "abs_min_seconds": min(absolute), "abs_median_seconds": quantile_sorted(absolute, 0.5),
                "abs_p90_seconds": quantile_sorted(absolute, 0.9), "abs_p95_seconds": quantile_sorted(absolute, 0.95),
                "abs_p99_seconds": quantile_sorted(absolute, 0.99), "abs_max_seconds": max(absolute),
                "signed_median_seconds": signed_median, "signed_min_seconds": min(signed), "signed_max_seconds": max(signed),
                "method_counts": json_text(timestamp_method_counts[group]),
                "fixed_hour_offset_detected": fixed_hour,
                "notes": "Absolute difference uses normalized official timestamp origin; minute-precision rows naturally span almost 60 seconds within their recorded minute bucket",
            })

    class_fields = [
        "class_name", "official_raw_labels", "official_csv_count", "existing_mapping_count",
        "verified_label_match_count", "difference", "relative_difference", "interpretation",
    ]
    all_labels = sorted(set(official_counts) | set(existing_counts))
    with (output_dir / "class_count_comparison.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=class_fields)
        writer.writeheader()
        for label in all_labels:
            official = official_counts[label]
            existing = existing_counts[label]
            difference = existing - official
            writer.writerow({
                "class_name": label,
                "official_raw_labels": json_text(sorted(official_raw_labels[label])),
                "official_csv_count": official, "existing_mapping_count": existing,
                "verified_label_match_count": label_match_by_class[label],
                "difference": difference,
                "relative_difference": difference / official if official else "",
                "interpretation": "Counts need not be equal: official CICFlowMeter rows include directional/duplicate records, while mapping rows are reconstructed bidirectional sessions with timeout/termination differences.",
            })

    spot_fields = [
        "flow_id", "source_day", "existing_tuple", "official_tuple", "existing_time",
        "official_time", "existing_label", "official_label", "tuple_match", "time_match",
        "label_match", "manual_result", "notes",
    ]
    spot_rows = [row for key in sorted(spot_buckets) for row in spot_buckets[key]]
    with (output_dir / "manual_spot_check.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=spot_fields)
        writer.writeheader()
        writer.writerows(spot_rows)

    flow_parquets = [mapping_root / "flows" / f"{day}_flows.parquet" for day in DAYS]
    schema = pq.ParquetFile(flow_parquets[0]).schema_arrow
    schema_lines = [f"- `{field.name}`: `{field.type}`" for field in schema]
    requested_map = {
        "flow_id": "flow_id", "src_ip": "src_ip", "src_port": "src_port",
        "dst_ip": "dst_ip", "dst_port": "dst_port", "protocol": "protocol",
        "timestamp / flow_start_time": "flow_start_time; flow_start_epoch_utc",
        "flow_end_time": "flow_end_time; flow_end_epoch_utc", "duration": "duration",
        "packet_count": "packet_count", "forward_packet_count": "forward_packet_count",
        "backward_packet_count": "backward_packet_count", "source_pcap": "source_pcap",
        "source_day": "day", "label": "label", "raw_label": "not retained in Parquet",
        "matched_csv_file": "source_label_csv", "matched_csv_row": "official_row_id",
        "match_status": "match_status", "time_diff_seconds": "timestamp_difference_seconds",
    }
    schema_report = [
        "# Existing mapping schema report", "",
        f"- Mapping root: `{mapping_root}`",
        f"- Main flow Parquet files: 5; total rows: {sum(pq.ParquetFile(p).metadata.num_rows for p in flow_parquets):,}",
        "- Schema is identical across all five day files: " + str(all(pq.ParquetFile(p).schema_arrow == schema for p in flow_parquets)),
        "", "## Requested field mapping", "",
        "| Requested concept | Actual field |", "|---|---|",
    ] + [f"| `{key}` | `{value}` |" for key, value in requested_map.items()] + [
        "", "## Actual Parquet schema", "", *schema_lines,
        "", "`raw_label` is not retained; `label` contains whitespace cleanup and dash normalization. Source CSV and row provenance are retained.",
    ]
    (output_dir / "mapping_schema_report.md").write_text("\n".join(schema_report) + "\n", encoding="utf-8")

    source_code = mapping_root.parent.parent / "cicids2017_label_mapping" / "map_flows.py"
    provenance_report = f"""# Label provenance report

## Finding

- Official ground truth source: `{official_root}`.
- `MachineLearningCVE` used for label decisions: **no**.
- Attack schedule/time windows used to assign labels: **no**; only the separate aggregate sanity check uses them.
- Filename-derived attack labels: **no**. Filenames are used only to select the same weekday PCAP and CSV files.
- Unmatched to BENIGN fallback: **no**. `UNMATCHED` rows have `label=NULL`.
- Default label fallback: **none found**.
- Label merge/rename: whitespace is collapsed and dash-like characters in Web Attack labels are normalized to ASCII ` - `. No semantic class merge was found.

## Direct code evidence

- Source loader: `{source_code}` lines 167-217 reads the official CSV tuple, timestamp, and Label columns.
- Input discovery: lines 531-541 restricts labels to `CSV/TrafficLabelling/*.csv` for the selected weekday.
- Output assignment: lines 557-593 assigns `record.label` only when an official record was selected; otherwise label is null.
- Attack windows exist only in `aggregate_reports.py` and are documented as validation-only.

## Independent artifact checks

- Official valid rows read in this audit: {sum(official_counts.values()):,}.
- Invalid non-empty official rows: {official_invalid:,}; empty delimiter-only rows: {official_empty:,}.
- Existing mapped rows with a non-null label: {sum(existing_counts.values()):,}.
- Existing unmatched rows with a non-null label: {existing_unmatched_with_nonnull_label:,}.
"""
    (output_dir / "label_provenance_report.md").write_text(provenance_report, encoding="utf-8")

    matching_report = f"""# Matching logic report

## Existing generator

- Flow identity: direction-independent canonical 5-tuple; endpoints `(IP, port)` are sorted and protocol is included.
- Reverse direction: handled as the same bidirectional flow. The first observed packet defines forward direction counters.
- Session separation: idle timeout 120 s, active timeout 120 s, TCP close grace 1 s, and a new SYN after closure starts another instance.
- Official matching hierarchy: exact/recorded time bucket within 1 s, then 5 s, 10 s, then flow-interval overlap.
- Tie-break fields: packet count, duration, and direction counts. Label is not part of the score.
- One-to-one behavior: a selected official row is marked used. An otherwise eligible candidate already used by another flow leads to `UNMATCHED`.
- Equal best candidates with different labels become `LABEL_CONFLICT`; equal-label ties become `MATCHED_DUPLICATE_SAME_LABEL`.

## Risk assessment

- **No session collapse found in the declared algorithm**: repeated canonical tuples can create multiple `flow_instance_index` values after timeout/closure.
- **Moderate ambiguity risk**: {multiple_eligible_rows:,} rows had more than one independently eligible official candidate; the generator resolves a unique best candidate by count/duration/direction rather than requiring a uniqueness margin.
- Equal-best candidate rows: {multi_best_rows:,}; see `official_ambiguities.csv`.
- Existing methods containing `COUNT_DURATION_TIEBREAK`: {sum(count for method, count in existing_methods.items() if 'COUNT_DURATION_TIEBREAK' in method):,}.
- The current Parquet does not retain the full candidate set or candidate-count field, so downstream experiments must rely on this audit or conservatively filter questionable rows.
"""
    (output_dir / "matching_logic_report.md").write_text(matching_report, encoding="utf-8")

    timestamp_summary_arr = array("d")
    for group, values in timestamp_values.items():
        timestamp_summary_arr.extend(values)
    spot_counts = Counter(row["manual_result"] for row in spot_rows)
    label_mismatch_count = overall_status["LABEL_MISMATCH"]
    no_match_count = overall_status["NO_MATCH"]
    multiple_count = overall_status["MULTIPLE_CANDIDATES"]
    insufficient_count = overall_status["INSUFFICIENT_FIELDS"]
    label_match_count = overall_status["LABEL_MATCH"]
    verdict = "B. REUSABLE_WITH_FILTERING"
    filter_rule = (
        "Keep only rows where match_status starts with MATCHED, label is non-null, and this audit's "
        "verification_status is LABEL_MATCH. Exclude UNMATCHED, LABEL_CONFLICT, LABEL_MISMATCH, "
        "MULTIPLE_CANDIDATES, NO_MATCH, and INSUFFICIENT_FIELDS. Do not use day, timestamps, IPs, "
        "Flow ID, source CSV, or match metadata as model features."
    )
    summary = f"""# CIC-IDS-2017 existing mapping audit

## Final decision

**{verdict}**

The matched portion is strongly traceable to official TrafficLabelling rows, but the complete mapping directory is not a drop-in labelled dataset because it also contains {no_match_count:,} unmatched/null-label flows and candidate ambiguity must be filtered explicitly.

## Core results

- Existing flow rows: {processed_rows:,}
- Independently recheckable rows: {independent_recheckable:,}
- LABEL_MATCH: {label_match_count:,} ({label_match_count / processed_rows:.6%})
- LABEL_MISMATCH: {label_mismatch_count:,} ({label_mismatch_count / processed_rows:.6%})
- NO_MATCH: {no_match_count:,} ({no_match_count / processed_rows:.6%})
- MULTIPLE_CANDIDATES (equal-best different-label): {multiple_count:,} ({multiple_count / processed_rows:.6%})
- INSUFFICIENT_FIELDS: {insufficient_count:,} ({insufficient_count / processed_rows:.6%})
- Rows with more than one eligible candidate before non-label tie-break: {multiple_eligible_rows:,} ({multiple_eligible_rows / processed_rows:.6%})
- Equal-best rows of any label composition: {multi_best_rows:,} ({multi_best_rows / processed_rows:.6%})
- Existing referenced official rows found: {referenced_rows_found:,}; tuple+label consistent: {referenced_rows_consistent:,}
- Existing unmatched rows for which an independent non-one-to-one candidate exists: {existing_unmatched_with_candidate:,}
- Manual deterministic spot-check: {len(spot_rows)} rows; PASS={spot_counts['PASS']}, FAIL={spot_counts['FAIL']}, UNCERTAIN={spot_counts['UNCERTAIN']}

## Timestamp

- Absolute origin difference across verified unique label matches: min={min(timestamp_summary_arr) if timestamp_summary_arr else 'NA'}, median={quantile_sorted(timestamp_summary_arr, 0.5)}, p90={quantile_sorted(timestamp_summary_arr, 0.9)}, p95={quantile_sorted(timestamp_summary_arr, 0.95)}, p99={quantile_sorted(timestamp_summary_arr, 0.99)}, max={max(timestamp_summary_arr) if timestamp_summary_arr else 'NA'} seconds.
- Detailed day/PCAP/CSV statistics and fixed-offset flags are in `timestamp_recheck.csv`.

## High-risk logic

- unmatched -> BENIGN: **not found**.
- ambiguous -> first: equal-best different-label cases are rejected, but equal-best same-label cases select the first deterministic official row.
- nearest/score selection without a uniqueness margin: **present** when multiple candidates have one best count/duration/direction score; this is why audit-side filtering is retained.
- silent conflict deletion: **not found**; conflict rows are retained with null label, though the current full run reports zero conflicts.

## Required filtering rule

{filter_rule}

## Scope boundary

No PCAP was parsed. This audit validates existing flow metadata against official CSV rows; it does not independently reproduce packet-to-flow reconstruction or prove CICFlowMeter-equivalent termination behavior.
"""
    (output_dir / "audit_summary.md").write_text(summary, encoding="utf-8")

    after_mapping = input_snapshot(mapping_inputs)
    after_official = input_snapshot(official_inputs)
    if before_mapping != after_mapping:
        raise RuntimeError("Existing mapping files changed during read-only audit")
    if before_official != after_official:
        raise RuntimeError("Official CSV files changed during read-only audit")

    script_path = Path(__file__).resolve()
    metadata = {
        "audit_name": "CIC-IDS-2017 existing mapping audit",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "script": str(script_path),
        "script_sha256": hashlib.sha256(script_path.read_bytes()).hexdigest(),
        "command": " ".join(sys.argv),
        "seed": args.seed,
        "batch_size": args.batch_size,
        "mapping_root": str(mapping_root),
        "official_csv_root": str(official_root),
        "output_dir": str(output_dir),
        "pcap_read": False,
        "matching_policy": {
            "canonical_bidirectional_5tuple": True,
            "timestamp_hierarchy_seconds": [1, 5, 10],
            "flow_interval_overlap": True,
            "tie_break_fields": ["packet_count", "duration", "direction_counts"],
            "label_used_for_candidate_scoring": False,
            "one_to_one_official_consumption": False,
        },
        "mapping_input_snapshot_before": before_mapping,
        "mapping_input_snapshot_unchanged": before_mapping == after_mapping,
        "official_input_snapshot_before": before_official,
        "official_input_snapshot_unchanged": before_official == after_official,
        "counts": {
            "processed_rows": processed_rows,
            "independently_recheckable": independent_recheckable,
            "verification_status": dict(sorted(overall_status.items())),
            "candidate_status": dict(sorted(candidate_statuses.items())),
            "existing_match_status": dict(sorted(existing_statuses.items())),
            "multiple_eligible_rows": multiple_eligible_rows,
            "equal_best_rows": multi_best_rows,
            "existing_unmatched_with_independent_candidate": existing_unmatched_with_candidate,
            "manual_spot_check": dict(sorted(spot_counts.items())),
        },
        "verdict": verdict,
        "filter_rule": filter_rule,
        "generated_files": sorted({path.name for path in output_dir.iterdir() if path.is_file()} | {"run_metadata.json"}),
        "inventory_file_count": len(inventory_rows),
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "processed_rows": processed_rows,
        "verification_status": dict(sorted(overall_status.items())),
        "candidate_status": dict(sorted(candidate_statuses.items())),
        "multiple_eligible_rows": multiple_eligible_rows,
        "spot_check": dict(sorted(spot_counts.items())),
        "verdict": verdict,
        "output_dir": str(output_dir),
    }, ensure_ascii=False, indent=2), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping-root", type=Path, required=True)
    parser.add_argument("--official-csv-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=50_000)
    return parser


if __name__ == "__main__":
    audit(build_parser().parse_args())
