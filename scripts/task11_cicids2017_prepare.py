#!/usr/bin/env python3
"""Prepare the fixed CIC-IDS-2017 seven-class TrafficFormer audit cohort.

This script consumes the *existing* mapping and its independent verification
CSV.  It never assigns labels and never reruns PCAP-to-label matching.  PCAPs
are streamed once only to recover the first five raw packets of already
selected, already bounded flow IDs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import ipaddress
import json
import os
import random
import sys
import time
import zlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.trafficformer_input import (  # noqa: E402
    TrafficFormerFormat,
    encode_flow,
)


CLASSES = (
    "BENIGN",
    "DDoS",
    "DoS Hulk",
    "PortScan",
    "DoS GoldenEye",
    "DoS Slowhttptest",
    "DoS slowloris",
)
DAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")
SEED = 42
CLASS_CAP = 100_000
STRICT_EXPECTED = 2_086_657
FORMAT = TrafficFormerFormat(
    policy="compatible_min1",
    min_packets=1,
    max_packets=5,
    ethernet_offset_bytes=14,
    bytes_per_packet=64,
    add_sep_before_packet=True,
    seq_length=320,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(stage: str, class_name: str, day: str) -> int:
    token = f"{stage}|{class_name}|{day}".encode("utf-8")
    return (SEED + zlib.crc32(token)) % (2**32)


def is_true(value: object) -> bool:
    return str(value).strip().lower() == "true"


def is_strict(row: dict[str, str]) -> bool:
    return (
        row["verification_status"] == "LABEL_MATCH"
        and is_true(row["stored_tuple_consistent"])
        and is_true(row["referenced_row_found"])
        and is_true(row["referenced_row_tuple_match"])
        and is_true(row["referenced_row_label_match"])
        and bool(row["existing_label"].strip())
        and row["existing_match_status"].startswith("MATCHED")
    )


def largest_remainder(counts: dict[str, int], target: int) -> dict[str, int]:
    """Allocate an integer target proportionally with deterministic ties."""
    total = sum(counts.values())
    if not 0 <= target <= total:
        raise ValueError(f"invalid allocation target={target}, total={total}")
    if total == 0:
        return {key: 0 for key in counts}
    exact = {key: counts[key] * target / total for key in counts}
    result = {key: int(np.floor(value)) for key, value in exact.items()}
    remaining = target - sum(result.values())
    order = sorted(counts, key=lambda key: (-(exact[key] - result[key]), key))
    for key in order[:remaining]:
        result[key] += 1
    return result


def select_verified_rows(audit_csv: Path) -> tuple[list[dict[str, object]], dict]:
    target_rows: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    strict_total = 0
    verification_counts: Counter[str] = Counter()
    with audit_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "flow_id", "day", "source_pcap", "existing_label",
            "existing_match_status", "verification_status",
            "stored_tuple_consistent", "referenced_row_found",
            "referenced_row_tuple_match", "referenced_row_label_match",
        }
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"audit CSV missing fields: {sorted(missing)}")
        for row in reader:
            verification_counts[row["verification_status"]] += 1
            if not is_strict(row):
                continue
            strict_total += 1
            label = row["existing_label"]
            if label in CLASSES:
                target_rows[(label, row["day"])].append({
                    "flow_id": row["flow_id"],
                    "class_name": label,
                    "source_day": row["day"],
                    "source_pcap": row["source_pcap"],
                    "verification_status": row["verification_status"],
                    "existing_match_status": row["existing_match_status"],
                    "stored_tuple_consistent": True,
                    "referenced_row_found": True,
                    "referenced_row_tuple_match": True,
                    "referenced_row_label_match": True,
                })
    if strict_total != STRICT_EXPECTED:
        raise ValueError(
            f"strict mapping count drift: {strict_total:,} != {STRICT_EXPECTED:,}"
        )

    selected: list[dict[str, object]] = []
    sampling_rows: list[dict[str, object]] = []
    for class_name in CLASSES:
        day_counts = {
            day: len(target_rows[(class_name, day)])
            for day in DAYS if target_rows[(class_name, day)]
        }
        original_count = sum(day_counts.values())
        if original_count == 0:
            raise ValueError(f"required class absent after strict filter: {class_name}")
        selected_count = min(original_count, CLASS_CAP)
        quotas = largest_remainder(day_counts, selected_count)
        for day in sorted(day_counts, key=DAYS.index):
            rows = target_rows[(class_name, day)]
            quota = quotas[day]
            if quota < len(rows):
                rng = np.random.default_rng(stable_seed("sampling", class_name, day))
                chosen = np.sort(rng.choice(len(rows), size=quota, replace=False))
                rows = [rows[int(index)] for index in chosen]
            selected.extend(rows)
            sampling_rows.append({
                "class_name": class_name,
                "original_count": original_count,
                "selected_count": selected_count,
                "source_day": day,
                "day_original_count": day_counts[day],
                "day_selected_count": quota,
                "seed": SEED,
            })
    return selected, {
        "strict_total": strict_total,
        "verification_counts": dict(verification_counts),
        "sampling_rows": sampling_rows,
    }


def assign_splits(rows: list[dict[str, object]]) -> None:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["class_name"]), str(row["source_day"]))].append(row)
    for class_name in CLASSES:
        counts = {
            day: len(grouped[(class_name, day)])
            for day in DAYS if grouped[(class_name, day)]
        }
        total = sum(counts.values())
        val_target = int(round(total * 0.20))
        val_quotas = largest_remainder(counts, val_target)
        for day in counts:
            group = grouped[(class_name, day)]
            order = list(range(len(group)))
            random.Random(stable_seed("split", class_name, day)).shuffle(order)
            val_indices = set(order[:val_quotas[day]])
            for index, row in enumerate(group):
                row["split"] = "val" if index in val_indices else "train"


def join_flow_boundaries(rows: list[dict[str, object]], mapping_root: Path) -> None:
    selected = {str(row["flow_id"]): row for row in rows}
    found: set[str] = set()
    columns = [
        "flow_id", "day", "source_pcap", "label", "match_status",
        "canonical_flow_key", "first_packet_index", "last_packet_index",
        "packet_count",
    ]
    for day in DAYS:
        path = mapping_root / "flows" / f"{day}_flows.parquet"
        table = pq.read_table(path, columns=columns)
        data = table.to_pydict()
        for index, flow_id in enumerate(data["flow_id"]):
            row = selected.get(flow_id)
            if row is None:
                continue
            if flow_id in found:
                raise ValueError(f"duplicate selected flow in Parquet: {flow_id}")
            found.add(flow_id)
            checks = {
                "day": (data["day"][index], row["source_day"]),
                "source_pcap": (data["source_pcap"][index], row["source_pcap"]),
                "label": (data["label"][index], row["class_name"]),
            }
            bad = {key: value for key, value in checks.items() if value[0] != value[1]}
            if bad or not str(data["match_status"][index]).startswith("MATCHED"):
                raise ValueError(f"mapping/audit mismatch for {flow_id}: {bad}")
            row.update({
                "canonical_flow_key": data["canonical_flow_key"][index],
                "first_packet_index": int(data["first_packet_index"][index]),
                "last_packet_index": int(data["last_packet_index"][index]),
                "packet_count": int(data["packet_count"][index]),
            })
    missing = set(selected) - found
    if missing:
        raise ValueError(f"selected flows absent from mapping Parquet: {len(missing)}")


def endpoint_text(value: tuple[int, bytes, int]) -> str:
    version, address, port = value
    ip = str(ipaddress.ip_address(address))
    return f"[{ip}]:{port}" if version == 6 else f"{ip}:{port}"


def packet_key(raw: bytes, dpkt) -> str | None:
    try:
        frame = dpkt.ethernet.Ethernet(raw)
        network = frame.data
        if isinstance(network, dpkt.ip.IP):
            version, protocol = 4, int(network.p)
        elif isinstance(network, dpkt.ip6.IP6):
            version, protocol = 6, int(network.nxt)
        else:
            return None
        transport = network.data
        if isinstance(transport, dpkt.tcp.TCP):
            protocol, src_port, dst_port = 6, int(transport.sport), int(transport.dport)
        elif isinstance(transport, dpkt.udp.UDP):
            protocol, src_port, dst_port = 17, int(transport.sport), int(transport.dport)
        else:
            src_port = dst_port = 0
        src = (version, bytes(network.src), src_port)
        dst = (version, bytes(network.dst), dst_port)
        left, right = (src, dst) if src <= dst else (dst, src)
        return f"{protocol}|{endpoint_text(left)}|{endpoint_text(right)}"
    except (dpkt.UnpackError, ValueError, OSError, IndexError):
        return None


def make_smoke_ids(rows: list[dict[str, object]]) -> set[str]:
    result: set[str] = set()
    limits = {"train": 512, "val": 128}
    for class_name in CLASSES[:3]:
        for split in ("train", "val"):
            group = sorted(
                (row for row in rows if row["class_name"] == class_name and row["split"] == split),
                key=lambda row: str(row["flow_id"]),
            )
            rng = np.random.default_rng(stable_seed("smoke", class_name, split))
            chosen = rng.choice(len(group), size=min(limits[split], len(group)), replace=False)
            result.update(str(group[int(index)]["flow_id"]) for index in chosen)
    return result


def write_manifest(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def extract_and_write(
    rows: list[dict[str, object]], data_root: Path, output_root: Path, vendor: Path
) -> dict:
    sys.path.insert(0, str(vendor))
    import dpkt  # type: ignore

    input_dir = output_root / "artifacts" / "input"
    smoke_dir = output_root / "artifacts" / "smoke" / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    smoke_dir.mkdir(parents=True, exist_ok=True)
    smoke_ids = make_smoke_ids(rows)
    label_ids = {name: index for index, name in enumerate(CLASSES)}
    smoke_label_ids = {name: index for index, name in enumerate(CLASSES[:3])}

    paths = {
        "train": input_dir / "train_dataset.tsv",
        "val": input_dir / "valid_dataset.tsv",
        "smoke_train": smoke_dir / "train_dataset.tsv",
        "smoke_val": smoke_dir / "valid_dataset.tsv",
    }
    maps = {
        "train": input_dir / "flow_map_train.csv",
        "val": input_dir / "flow_map_val.csv",
        "smoke_train": smoke_dir / "flow_map_train.csv",
        "smoke_val": smoke_dir / "flow_map_val.csv",
    }
    handles = {key: path.open("w", encoding="utf-8", newline="") for key, path in paths.items()}
    map_handles = {key: path.open("w", encoding="utf-8", newline="") for key, path in maps.items()}
    tsv_writers = {key: csv.writer(handle, delimiter="\t", lineterminator="\n") for key, handle in handles.items()}
    map_fields = [
        "row_index", "flow_id", "class_name", "label_id", "source_day", "split",
        "source_pcap", "packet_count", "used_packet_count", "token_count", "text_sha256",
    ]
    map_writers = {key: csv.DictWriter(handle, fieldnames=map_fields) for key, handle in map_handles.items()}
    for writer in tsv_writers.values():
        writer.writerow(["label", "text_a"])
    for writer in map_writers.values():
        writer.writeheader()
    row_counts: Counter[str] = Counter()
    observed_counts: Counter[str] = Counter()

    def emit(row: dict[str, object], raw_packets: list[bytes]) -> None:
        expected = min(FORMAT.max_packets, int(row["packet_count"]))
        if len(raw_packets) != expected:
            raise ValueError(
                f"{row['flow_id']}: recovered {len(raw_packets)} packets, expected {expected}"
            )
        pseudo_packets = [(0.0, len(raw), len(raw), "", raw) for raw in raw_packets]
        encoded = encode_flow(pseudo_packets, FORMAT)
        if encoded is None:
            raise ValueError(f"unexpected filtered flow: {row['flow_id']}")
        split = str(row["split"])
        targets = [split]
        if str(row["flow_id"]) in smoke_ids:
            targets.append(f"smoke_{split}")
        for target in targets:
            label_map = smoke_label_ids if target.startswith("smoke_") else label_ids
            label_id = label_map[str(row["class_name"])]
            tsv_writers[target].writerow([label_id, encoded.text])
            map_writers[target].writerow({
                "row_index": row_counts[target],
                "flow_id": row["flow_id"],
                "class_name": row["class_name"],
                "label_id": label_id,
                "source_day": row["source_day"],
                "split": split,
                "source_pcap": row["source_pcap"],
                "packet_count": row["packet_count"],
                "used_packet_count": encoded.used_packet_count,
                "token_count": encoded.token_count,
                "text_sha256": hashlib.sha256(encoded.text.encode("utf-8")).hexdigest(),
            })
            row_counts[target] += 1

    try:
        for day in DAYS:
            day_rows = [row for row in rows if row["source_day"] == day]
            by_key: dict[str, list[dict[str, object]]] = defaultdict(list)
            for row in day_rows:
                row["_raw_packets"] = []
                row["_observed_packet_count"] = 0
                by_key[str(row["canonical_flow_key"])].append(row)
            for intervals in by_key.values():
                intervals.sort(key=lambda row: int(row["first_packet_index"]))
            positions = {key: 0 for key in by_key}
            pcap_names = {str(row["source_pcap"]) for row in day_rows}
            if len(pcap_names) != 1:
                raise ValueError(f"{day}: expected one PCAP, found {sorted(pcap_names)}")
            pcap_path = data_root / "PCAPs" / next(iter(pcap_names))
            started = time.time()
            packet_seen = 0
            with pcap_path.open("rb") as handle:
                reader = dpkt.pcapng.Reader(handle)
                if reader.datalink() != dpkt.pcap.DLT_EN10MB:
                    raise ValueError(f"unsupported datalink for {pcap_path}")
                for packet_index, (_, raw) in enumerate(reader):
                    packet_seen += 1
                    key = packet_key(raw, dpkt)
                    if key is None or key not in by_key:
                        continue
                    intervals = by_key[key]
                    pos = positions[key]
                    while pos < len(intervals) and packet_index > int(intervals[pos]["last_packet_index"]):
                        stale = intervals[pos]
                        raise ValueError(f"missed final packet boundary for {stale['flow_id']}")
                    if pos >= len(intervals):
                        continue
                    row = intervals[pos]
                    first = int(row["first_packet_index"])
                    last = int(row["last_packet_index"])
                    if first <= packet_index <= last:
                        row["_observed_packet_count"] = int(row["_observed_packet_count"]) + 1
                        packets = row["_raw_packets"]
                        assert isinstance(packets, list)
                        if len(packets) < FORMAT.max_packets:
                            packets.append(bytes(raw))
                        if packet_index == last:
                            observed = int(row["_observed_packet_count"])
                            if observed != int(row["packet_count"]):
                                raise ValueError(
                                    f"{row['flow_id']}: interval packet count {observed} "
                                    f"!= mapping {row['packet_count']}"
                                )
                            emit(row, packets)
                            observed_counts[day] += observed
                            del row["_raw_packets"]
                            del row["_observed_packet_count"]
                            positions[key] = pos + 1
                    if packet_seen % 5_000_000 == 0:
                        print(
                            f"{day}: packets={packet_seen:,} emitted="
                            f"{row_counts['train'] + row_counts['val']:,} "
                            f"elapsed={time.time() - started:.1f}s",
                            flush=True,
                        )
            unfinished = sum(len(by_key[key]) - positions[key] for key in by_key)
            if unfinished:
                raise ValueError(f"{day}: {unfinished} selected intervals unfinished at EOF")
            print(
                f"{day}: complete packets={packet_seen:,}, flows={len(day_rows):,}, "
                f"elapsed={time.time() - started:.1f}s",
                flush=True,
            )
    finally:
        for handle in handles.values():
            handle.close()
        for handle in map_handles.values():
            handle.close()

    expected = Counter(str(row["split"]) for row in rows)
    if row_counts["train"] != expected["train"] or row_counts["val"] != expected["val"]:
        raise ValueError(f"TSV count mismatch: {dict(row_counts)} vs {dict(expected)}")
    return {
        "row_counts": dict(row_counts),
        "observed_selected_packets_by_day": dict(observed_counts),
        "paths": {key: str(path) for key, path in paths.items()},
        "sha256": {str(path): sha256_file(path) for path in [*paths.values(), *maps.values()]},
    }


def write_protocol(output_root: Path, args: argparse.Namespace) -> None:
    text = f"""# CIC-IDS-2017 TrafficFormer protocol

## Reused USTC Stage 1 protocol

- TrafficFormer/UER source: `tf_runtime/code`, upstream commit `6d0ba64d82e74fb130c6c7301ef20885dbfbdf29`.
- Base checkpoint: `{args.pretrained_model}`; SHA-256 `{sha256_file(args.pretrained_model)}`.
- Base checkpoint provenance: TrafficFormer official Google Drive object `1pR6ZaWE7MWFDQWiF4LDzSyjSq0Gj3kV7`, archive member `nomoe_bertflow_pre-trained_model.bin-120000`; local verification is recorded in `tf_runtime/PRETRAINED_MODEL.md`.
- Vocabulary: `{args.vocab}`; SHA-256 `{sha256_file(args.vocab)}`.
- BERT config: `{args.bert_config}`; SHA-256 `{sha256_file(args.bert_config)}`.
- Packet construction: remove the 14-byte Ethernet header; keep at most 64 following bytes from each of at most five observed packets; prepend `[SEP]` to every packet; use the same overlapping two-byte bigrams.
- Short flows: retain every non-empty flow (`compatible_min1`); do not synthesize packets. UER pads the token sequence tail with token ID 0.
- Token sequence: prepend `[CLS]`, truncate/pad to 320 tokens, `word_pos_seg`, fully-visible Transformer mask.
- Architecture: BERT-base-sized TrafficFormer encoder (12 layers, hidden 768, 12 heads); pooling is `first`.
- `z_t`: encoder output `output[:, 0, :]` before `output_layer_1` and the classifier head; dimension 768.
- USTC optimization reference: AdamW, linear scheduler, learning rate `6e-5`, three epochs, early-stop patience 3, seed 7, effective batch 32.

## CIC-IDS-2017 controlled differences

- A fresh seven-class classification head is initialized and trained from the same base checkpoint; the USTC 20-class fine-tuned head is never used.
- Sampling and train/validation splitting use seed 42 as requested. Model training retains the USTC seed 7.
- The user-selected physical GPU is 3. Training uses the USTC batch size 32 directly on that GPU; a bounded backward-pass smoke test is required before the formal run.
- No test split, Unknown class, FIG, `z_g`, `z_f`, BIC selection, adaptive K, or label remapping is used.

Therefore the input representation, tokenizer, architecture, base checkpoint, optimization target, and `z_t` definition are the same as the actual USTC Stage 1 run. The execution mechanism differs only by mathematically equivalent gradient accumulation and the required new seven-class head.
"""
    (output_root / "trafficformer_protocol.md").write_text(text, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    default_dataset = Path(
        "/home/birkenwald/data/SEU-WXY/WXY-SEU/TrafficClassifier/Dataset/"
        "CIC-IDS-2017/CIC-IDS-2017(whole)"
    )
    parser.add_argument("--data-root", type=Path, default=default_dataset)
    parser.add_argument(
        "--mapping-root", type=Path,
        default=default_dataset / "outputs" / "cicids2017_pcap_label_mapping",
    )
    parser.add_argument(
        "--audit-csv", type=Path,
        default=PROJECT_ROOT / "outputs/cicids2017_existing_mapping_audit/mapping_vs_official.csv",
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=PROJECT_ROOT / "outputs/cicids2017_gaussian_audit",
    )
    parser.add_argument(
        "--vendor", type=Path,
        default=default_dataset / "cicids2017_label_mapping/vendor",
    )
    parser.add_argument(
        "--pretrained-model", type=Path,
        default=PROJECT_ROOT / "tf_runtime/code/models/pretrained_model.bin",
    )
    parser.add_argument(
        "--vocab", type=Path,
        default=PROJECT_ROOT / "tf_runtime/code/models/encryptd_vocab.txt",
    )
    parser.add_argument(
        "--bert-config", type=Path,
        default=PROJECT_ROOT / "tf_runtime/code/models/bert/base_config.json",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_root = args.output_root.resolve()
    protected = (PROJECT_ROOT / "outputs/stage2").resolve()
    if output_root == protected or protected in output_root.parents:
        raise ValueError("refusing to write into outputs/stage2")
    if output_root.exists() and any(output_root.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output is non-empty; use --overwrite: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    started = time.time()
    selected, audit = select_verified_rows(args.audit_csv.resolve())
    assign_splits(selected)
    join_flow_boundaries(selected, args.mapping_root.resolve())
    selected.sort(key=lambda row: (DAYS.index(str(row["source_day"])), int(row["first_packet_index"])))

    sampling_fields = [
        "class_name", "original_count", "selected_count", "source_day",
        "day_original_count", "day_selected_count", "seed",
    ]
    write_manifest(output_root / "sampling_manifest.csv", audit["sampling_rows"], sampling_fields)
    split_fields = [
        "flow_id", "class_name", "label_id", "source_day", "split", "source_pcap",
        "canonical_flow_key", "first_packet_index", "last_packet_index", "packet_count",
        "verification_status", "existing_match_status", "stored_tuple_consistent",
        "referenced_row_found", "referenced_row_tuple_match", "referenced_row_label_match",
    ]
    label_ids = {name: index for index, name in enumerate(CLASSES)}
    for row in selected:
        row["label_id"] = label_ids[str(row["class_name"])]
    write_manifest(output_root / "split_manifest.csv", selected, split_fields)
    write_protocol(output_root, args)

    extraction = extract_and_write(
        selected, args.data_root.resolve(), output_root, args.vendor.resolve()
    )
    input_dir = output_root / "artifacts" / "input"
    (input_dir / "label_map.json").write_text(
        json.dumps({
            "class_to_id": label_ids,
            "id_to_class": {str(value): key for key, value in label_ids.items()},
        }, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    smoke_ids = {name: index for index, name in enumerate(CLASSES[:3])}
    (output_root / "artifacts/smoke/input/label_map.json").write_text(
        json.dumps({
            "class_to_id": smoke_ids,
            "id_to_class": {str(value): key for key, value in smoke_ids.items()},
        }, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    class_split = Counter((str(row["class_name"]), str(row["split"])) for row in selected)
    metadata = {
        "stage": "prepare",
        "seed": SEED,
        "class_cap": CLASS_CAP,
        "strict_expected": STRICT_EXPECTED,
        "strict_observed": audit["strict_total"],
        "classes": list(CLASSES),
        "selected_total": len(selected),
        "class_split_counts": {
            f"{name}|{split}": count for (name, split), count in sorted(class_split.items())
        },
        "audit_csv": str(args.audit_csv.resolve()),
        "audit_csv_sha256": sha256_file(args.audit_csv.resolve()),
        "mapping_root": str(args.mapping_root.resolve()),
        "data_root": str(args.data_root.resolve()),
        "format": FORMAT.__dict__,
        "extraction": extraction,
        "command": " ".join(sys.argv),
        "elapsed_seconds": time.time() - started,
    }
    (output_root / "artifacts/preparation_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
