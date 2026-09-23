#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq

from audit_utils import (
    DATASET_ROOT,
    OUTPUT_ROOT,
    PROJECT_ROOT,
    SEED,
    directory_inventory,
    ensure_output_dirs,
    entropy_bits,
    normalized_mutual_information,
    percentile,
    scan_classic_pcap,
    sha256_file,
    top_share,
    write_csv,
)


DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
CIC_INPUT_LABELS = {
    "BENIGN",
    "FTP-Patator",
    "SSH-Patator",
    "DoS Hulk",
    "DoS GoldenEye",
    "DoS slowloris",
    "DoS Slowhttptest",
    "Bot",
    "PortScan",
    "DDoS",
}

# Eligibility thresholds are fixed before inspecting the resulting concentration values.
CIC_TOO_SMALL = 100
CIC_PRIMARY_MIN = 1_000
CIC_SEVERE_ENDPOINT_TOP1 = 0.95
CIC_MIN_TIME_GROUPS = 3


def locate_datasets() -> dict[str, Path]:
    candidates = {
        "USTC-TFC2016": ["ustc-tfc2016", "USTC-TFC2016"],
        "CipherSpectrum": ["CipherSpectrum", "ipherSpectrum", "CipherSpectrum(sok)", "ipherSpectrum(sok)"],
        "CSTNET-TLS1.3": ["CSTNET-TLS1.3", "CSTNET"],
        "CIC-IDS-2017": ["CIC-IDS-2017", "CICIDS2017"],
    }
    children = [item for item in DATASET_ROOT.iterdir() if item.is_dir()]
    resolved: dict[str, Path] = {}
    for dataset, names in candidates.items():
        match = next((item for name in names for item in children if item.name.lower() == name.lower()), None)
        if match is None:
            match = next(
                (item for item in children if any(name.lower() in item.name.lower() for name in names)),
                None,
            )
        if match is None:
            raise FileNotFoundError(f"unable to locate {dataset} under {DATASET_ROOT}")
        resolved[dataset] = match.resolve()
    rows = []
    for dataset, path in resolved.items():
        inventory = directory_inventory(path)
        rows.append({"dataset": dataset, "resolved_path": path, "exists": True, **inventory})
    write_csv(
        OUTPUT_ROOT / "summary/dataset_path_resolution.csv",
        rows,
        ["dataset", "resolved_path", "exists", "total_size", "file_count", "pcap_count", "csv_count", "parquet_count", "other_count"],
    )
    return resolved


def audit_ustc(root: Path) -> None:
    out = OUTPUT_ROOT / "ustc"
    whole = root / "USTC-TFC2016（whole）"
    canonical = whole / "extracted/V1/1.DataSet(USTC-TFC2016)"
    pcaps = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in {".pcap", ".pcapng"})
    with ThreadPoolExecutor(max_workers=6) as pool:
        scans = list(pool.map(scan_classic_pcap, pcaps))
    integrity_rows = []
    by_hash: dict[str, list[str]] = defaultdict(list)
    canonical_valid = 0
    for path, scan in zip(pcaps, scans):
        is_canonical = canonical in path.parents
        is_metadata = path.name.startswith("._") or "__MACOSX" in path.parts
        if is_canonical and scan["readable"] and int(scan["packet_count"]) > 0:
            canonical_valid += 1
        if scan["sha256"]:
            by_hash[str(scan["sha256"])].append(str(path.resolve()))
        integrity_rows.append({
            "file_name": path.name,
            "relative_path": str(path.relative_to(root)),
            "is_canonical_extracted": is_canonical,
            "is_macos_metadata": is_metadata,
            **scan,
        })
    write_csv(
        out / "file_integrity.csv",
        integrity_rows,
        ["file_name", "relative_path", "absolute_path", "size_bytes", "sha256", "readable", "packet_count", "first_packet_timestamp", "last_packet_timestamp", "normalized_fingerprint", "is_canonical_extracted", "is_macos_metadata", "error"],
    )
    duplicate_rows = []
    for digest, paths in sorted(by_hash.items()):
        if len(paths) > 1:
            duplicate_rows.append({"sha256": digest, "file_count": len(paths), "paths": " | ".join(paths)})
    write_csv(out / "duplicate_summary.csv", duplicate_rows, ["sha256", "file_count", "paths"])

    label_map_path = PROJECT_ROOT / "data/fig_graph/all_flows/label_map.json"
    split_summary_path = PROJECT_ROOT / "data/splits/compatible_min1/split_summary.json"
    label_map = json.loads(label_map_path.read_text(encoding="utf-8"))["class_to_id"]
    split = json.loads(split_summary_path.read_text(encoding="utf-8"))
    source_pcaps: dict[str, list[str]] = defaultdict(list)
    for path in sorted(canonical.rglob("*.pcap")):
        stem = path.stem
        logical = "SMB" if stem.startswith("SMB-") else "Weibo" if stem.startswith("Weibo-") else stem
        source_pcaps[logical].append(str(path.relative_to(canonical)))
    pkl_root = PROJECT_ROOT / "data/flows"
    source_pkls: dict[str, list[str]] = defaultdict(list)
    for path in sorted(pkl_root.rglob("*.pkl")):
        stem = path.stem
        logical = "SMB" if stem.startswith("SMB-") else "Weibo" if stem.startswith("Weibo-") else stem
        source_pkls[logical].append(str(path.relative_to(pkl_root)))
    class_rows = []
    for class_name, label_id in sorted(label_map.items(), key=lambda pair: pair[1]):
        flow_count = sum(int(split["per_split"][part]["per_class"][str(label_id)]) for part in ("train", "val", "test"))
        class_rows.append({
            "class_name": class_name,
            "label_id": label_id,
            "logical_class": True,
            "flow_count": flow_count,
            "source_pcap_count": len(source_pcaps[class_name]),
            "source_pcap_files": " | ".join(source_pcaps[class_name]),
            "source_pkl_count": len(source_pkls[class_name]),
            "source_pkl_files": " | ".join(source_pkls[class_name]),
            "merge_policy": "SMB-1+SMB-2" if class_name == "SMB" else "Weibo-1..4" if class_name == "Weibo" else "NONE",
        })
    write_csv(out / "class_inventory.csv", class_rows, list(class_rows[0]))

    open_detect = PROJECT_ROOT / "opendetect_ustc_encoder_audit/artifacts"
    checks = {
        "label_map": label_map_path,
        "split_summary": split_summary_path,
        "train_split_ids": PROJECT_ROOT / "data/splits/compatible_min1/train.txt",
        "val_split_ids": PROJECT_ROOT / "data/splits/compatible_min1/val.txt",
        "test_split_ids": PROJECT_ROOT / "data/splits/compatible_min1/test.txt",
        "opendetect_train_images": open_detect / "train_images.npy",
        "opendetect_val_images": open_detect / "val_images.npy",
        "opendetect_test_images": open_detect / "test_images.npy",
        "stage3_class_inventory": PROJECT_ROOT / "stage3_protocol/class_inventory.csv",
        "stage3_split_hashes": PROJECT_ROOT / "stage3_protocol/split_hashes.sha256",
        "stage4_class_thresholds": PROJECT_ROOT / "stage4_local_boundary_diagnosis/outputs/class_thresholds.csv",
        "stage5_protocol_hashes": PROJECT_ROOT / "stage5_dual_gate_boundary/outputs/cstnet_protocol/split_hashes.sha256",
    }
    ready = (
        len(label_map) == 20
        and int(split["total_flows"]) == 489_101
        and canonical_valid == 24
        and all(path.exists() for path in checks.values())
    )
    lines = [
        "# USTC Existing Split and Asset Summary",
        "",
        f"- Logical classes: **{len(label_map)}**.",
        f"- Final flows: **{split['total_flows']:,}** (train {split['per_split']['train']['flows']:,}, validation {split['per_split']['val']['flows']:,}, test {split['per_split']['test']['flows']:,}).",
        f"- Canonical extracted PCAPs readable/non-empty: **{canonical_valid}/24**.",
        "- SMB policy: **SMB-1 and SMB-2 merge into logical class SMB**.",
        "- Weibo policy: four captures merge into logical class Weibo.",
        "",
        "## Existing assets",
        "",
    ]
    lines.extend(f"- {name}: {'FOUND' if path.exists() else 'MISSING'} — `{path}`" for name, path in checks.items())
    (out / "existing_split_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "readiness.md").write_text(
        "# USTC Readiness\n\n"
        f"- USTC_DATA_READY = **{'YES' if ready else 'NO'}**\n"
        "- INDEPENDENT_EXTERNAL_VALIDATION = **NO**\n"
        "- Role = **Development / Mechanism benchmark**\n\n"
        "The Unknown test and related diagnostics have already informed method development; this prevents a new independent external claim even though the local data and frozen assets are usable.\n",
        encoding="utf-8",
    )


def read_cipher_manifest() -> tuple[Path, list[dict[str, str]]]:
    path = PROJECT_ROOT / "cipherspectrum_labeling/outputs/cipherspectrum_label_manifest.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        return path, list(csv.DictReader(handle))


def audit_cipher_inventory(root: Path) -> None:
    del root
    out = OUTPUT_ROOT / "cipherspectrum"
    _, rows = read_cipher_manifest()
    counters: dict[tuple[str, str], dict[str, object]] = {}
    groups: dict[str, dict[str, set[str] | int]] = {}
    for row in rows:
        key = (row["capture_group_directory"], row["canonical_label"])
        item = counters.setdefault(key, {"count": 0, "files": set(), "collections": set()})
        item["count"] = int(item["count"]) + 1
        item["files"].add(Path(row["relative_path"]).name)
        item["collections"].add(row["split_group_id"])
        group = groups.setdefault(row["split_group_id"], {"count": 0, "classes": set(), "sources": set()})
        group["count"] = int(group["count"]) + 1
        group["classes"].add(row["canonical_label"])
        group["sources"].add(row["capture_group_directory"])
    source_rows = []
    for (source, label), values in sorted(counters.items()):
        official = label != "getpocket.com" and source in {"aes-128-gcm", "aes-256-gcm", "chacha20-poly1305", "mix"}
        source_rows.append({
            "source_group": source,
            "class_name": label,
            "pcap_count": values["count"],
            "unique_filename_count": len(values["files"]),
            "unique_collection_count": len(values["collections"]),
            "corpus_status": "OFFICIAL40" if official else "NON_PRIMARY_EXTRA_CLASS",
        })
    write_csv(out / "source_class_counts.csv", source_rows, list(source_rows[0]))
    sizes = [int(value["count"]) for value in groups.values()]
    group_rows = [{
        "scope": "all_local41",
        "group_count": len(groups),
        "cross_class_group_count": sum(len(value["classes"]) > 1 for value in groups.values()),
        "cross_source_group_count": sum(len(value["sources"]) > 1 for value in groups.values()),
        "group_size_min": min(sizes),
        "group_size_p25": percentile(sizes, 0.25),
        "group_size_median": percentile(sizes, 0.50),
        "group_size_p75": percentile(sizes, 0.75),
        "group_size_p95": percentile(sizes, 0.95),
        "group_size_max": max(sizes),
        "split_group_id_available": True,
        "required_split_rule": "all rows sharing split_group_id stay in one partition",
    }]
    write_csv(out / "grouping_audit.csv", group_rows, list(group_rows[0]))

    official = [row for row in rows if row["official40_eligible"].lower() == "true"]
    canonical = [row for row in official if row["capture_group_directory"] != "mix"]
    per_class = Counter(row["canonical_label"] for row in canonical)
    calibration_rows = []
    for split_name, val_fraction in (("80/10/10", 0.10), ("70/15/15", 0.15)):
        val_counts = [round(count * val_fraction) for count in per_class.values()]
        calibration_rows.append({
            "record_type": "validation_distribution",
            "split_scheme": split_name,
            "component_weight_scenario": "ALL",
            "class_count": len(val_counts),
            "validation_min": min(val_counts),
            "validation_p25": percentile(val_counts, 0.25),
            "validation_median": percentile(val_counts, 0.50),
            "validation_p75": percentile(val_counts, 0.75),
            "validation_max": max(val_counts),
            "small_component_validation_min": "",
            "classes_expected_n_lt_30": "",
        })
        for scenario, small_weight in (("50/50", 0.50), ("80/20", 0.20), ("90/10", 0.10), ("95/5", 0.05)):
            small = [math.floor(count * small_weight) for count in val_counts]
            calibration_rows.append({
                "record_type": "K2_component_stress",
                "split_scheme": split_name,
                "component_weight_scenario": scenario,
                "class_count": len(small),
                "validation_min": min(val_counts),
                "validation_p25": percentile(val_counts, 0.25),
                "validation_median": percentile(val_counts, 0.50),
                "validation_p75": percentile(val_counts, 0.75),
                "validation_max": max(val_counts),
                "small_component_validation_min": min(small),
                "classes_expected_n_lt_30": sum(value < 30 for value in small),
            })
    write_csv(out / "calibration_capacity.csv", calibration_rows, list(calibration_rows[0]))


def audit_cstnet() -> None:
    out = OUTPUT_ROOT / "cstnet"
    protocol_root = PROJECT_ROOT / "stage5_dual_gate_boundary/outputs/cstnet_protocol"
    hash_rows = []
    for line in (protocol_root / "split_hashes.sha256").read_text(encoding="utf-8").splitlines():
        expected, relative = line.split(maxsplit=1)
        path = protocol_root / relative.strip()
        actual = sha256_file(path)
        hash_rows.append((relative.strip(), expected, actual, expected == actual))
    inventory_rows = list(csv.DictReader((protocol_root / "class_inventory.csv").open(encoding="utf-8", newline="")))
    original_classes = len(inventory_rows)
    readable = sum(int(row["pcap_count"]) for row in inventory_rows)
    eligible = [row for row in inventory_rows if row["eligible_for_open_set"].lower() == "true"]
    excluded = [row["class_name"] for row in inventory_rows if row["eligible_for_open_set"].lower() != "true"]
    (out / "frozen_protocol_verification.md").write_text(
        "# CSTNET Frozen Protocol Verification\n\n"
        f"- Protocol SHA-256: **{'PASS' if all(row[3] for row in hash_rows) else 'FAIL'} ({sum(row[3] for row in hash_rows)}/{len(hash_rows)})**.\n"
        "- Stage 5.5 broader frozen integrity record: **26/26 PASS**.\n"
        f"- Original classes: **{original_classes}**.\n"
        f"- Readable per-flow PCAPs: **{readable:,}**.\n"
        f"- Eligible/excluded: **{len(eligible)}/{len(excluded)}**; excluded: **{', '.join(excluded)}**.\n"
        "- Frozen folds: Low seed 42, Medium seed 43, High seed 44.\n"
        "- No split was regenerated and no protocol file was modified.\n\n"
        "## Hash details\n\n"
        + "\n".join(f"- {name}: {'PASS' if passed else 'FAIL'}" for name, _, _, passed in hash_rows)
        + "\n",
        encoding="utf-8",
    )
    split_rows = list(csv.DictReader((PROJECT_ROOT / "stage5_5_cstnet_preflight/outputs/calibration_sample_audit/class_split_counts.csv").open(encoding="utf-8", newline="")))
    fallback_rows = list(csv.DictReader((PROJECT_ROOT / "stage5_5_cstnet_preflight/outputs/calibration_sample_audit/fallback_risk_summary.csv").open(encoding="utf-8", newline="")))
    calibration_rows = []
    for fold in ("low", "medium", "high"):
        vals = [int(row["validation_count"]) for row in split_rows if row["fold"] == fold]
        calibration_rows.append({
            "record_type": "known_validation_distribution", "fold": fold, "scenario": "ALL",
            "known_class_count": len(vals), "min": min(vals), "p05": percentile(vals, 0.05),
            "p25": percentile(vals, 0.25), "median": percentile(vals, 0.50), "p75": percentile(vals, 0.75), "max": max(vals),
            "n_lt_20": sum(value < 20 for value in vals), "n_lt_30": sum(value < 30 for value in vals), "n_lt_50": sum(value < 50 for value in vals),
            "classes_expected_fallback_n_lt_30": "", "expected_fallback_rate": "",
        })
    for row in fallback_rows:
        calibration_rows.append({
            "record_type": "K2_component_stress", "fold": row["fold"], "scenario": row["component_weight_scenario"],
            "known_class_count": row["known_class_count"], "min": "", "p05": "", "p25": "", "median": "", "p75": "", "max": "",
            "n_lt_20": "", "n_lt_30": "", "n_lt_50": "",
            "classes_expected_fallback_n_lt_30": row["classes_expected_fallback_n_lt_30"], "expected_fallback_rate": row["expected_fallback_rate"],
        })
    write_csv(out / "calibration_capacity.csv", calibration_rows, list(calibration_rows[0]))

    leakage_path = PROJECT_ROOT / "stage5_5_cstnet_preflight/outputs/cstnet_input_audit/domain_leakage_audit.csv"
    with leakage_path.open(encoding="utf-8", newline="") as handle:
        leakage = list(csv.DictReader(handle))
    valid = [row for row in leakage if row["input_valid"].lower() == "true"]
    count_true = lambda field: sum(row[field].lower() == "true" for row in valid)
    (out / "input_leakage_summary.md").write_text(
        "# CSTNET Input Leakage Summary\n\n"
        f"- Reused fixed audit sample: **{len(leakage)} PCAPs; {len(valid)} valid 32x32 inputs**.\n"
        f"- Full class-domain visible: **{count_true('domain_string_visible')}/{len(valid)} ({count_true('domain_string_visible')/len(valid):.2%})**.\n"
        f"- Parsed SNI visible: **{count_true('sni_visible')}/{len(valid)} ({count_true('sni_visible')/len(valid):.2%})**.\n"
        f"- Domain/token union visible: **{count_true('domain_or_token_visible')}/{len(valid)} ({count_true('domain_or_token_visible')/len(valid):.2%})**.\n"
        "- IP fields: **MASKED (4,800/4,800 checked packet fields)**.\n"
        "- Transport ports: **VISIBLE (4,800/4,800 checked packet headers)**.\n"
        "- Direct domain leakage level: **LOW**. Port visibility remains a representation-level shortcut risk.\n",
        encoding="utf-8",
    )


def audit_cicids(root: Path) -> None:
    out = OUTPUT_ROOT / "cicids2017"
    whole = root / "CIC-IDS-2017(whole)"
    mapping = whole / "outputs/cicids2017_pcap_label_mapping"
    columns = [
        "flow_id", "day", "src_ip", "dst_ip", "src_port", "dst_port", "protocol",
        "canonical_flow_key", "flow_start_epoch_utc", "flow_end_epoch_utc", "packet_count",
        "first_packet_index", "last_packet_index", "label", "match_status", "source_pcap",
    ]
    counts = Counter()
    day_joint = Counter()
    pcap_joint = Counter()
    hour_joint = Counter()
    first_ts: dict[str, float] = {}
    last_ts: dict[str, float] = {}
    pcaps: dict[str, set[str]] = defaultdict(set)
    days: dict[str, set[str]] = defaultdict(set)
    time_groups: dict[str, Counter[str]] = defaultdict(Counter)
    endpoint: dict[str, dict[str, Counter[object]]] = defaultdict(lambda: defaultdict(Counter))
    sample_seen = Counter()
    samples: dict[str, list[dict[str, object]]] = defaultdict(list)
    rngs = {label: random.Random(f"{SEED}:{label}") for label in CIC_INPUT_LABELS}
    parquet_paths = [mapping / "flows" / f"{day}_flows.parquet" for day in DAYS]
    for parquet_path in parquet_paths:
        parquet = pq.ParquetFile(parquet_path)
        for batch in parquet.iter_batches(batch_size=65_536, columns=columns):
            data = batch.to_pydict()
            for index, label_value in enumerate(data["label"]):
                status = str(data["match_status"][index] or "")
                if label_value is None or not status.startswith("MATCHED"):
                    continue
                label = str(label_value)
                day = str(data["day"][index])
                pcap = str(data["source_pcap"][index])
                start = float(data["flow_start_epoch_utc"][index])
                end = float(data["flow_end_epoch_utc"][index])
                counts[label] += 1
                day_joint[(label, day)] += 1
                pcap_joint[(label, pcap)] += 1
                hour_joint[(label, day, datetime.fromtimestamp(start, tz=timezone.utc).hour)] += 1
                first_ts[label] = min(first_ts.get(label, start), start)
                last_ts[label] = max(last_ts.get(label, end), end)
                pcaps[label].add(pcap)
                days[label].add(day)
                time_groups[label][f"{day}:{int(start // 300)}"] += 1
                for field in ("src_ip", "dst_ip", "src_port", "dst_port"):
                    endpoint[label][field][data[field][index]] += 1
                if label in CIC_INPUT_LABELS and (label != "BENIGN" or day == "Tuesday"):
                    sample_seen[label] += 1
                    candidate = {key: data[key][index] for key in columns}
                    reservoir = samples[label]
                    if len(reservoir) < 20:
                        reservoir.append(candidate)
                    else:
                        slot = rngs[label].randrange(sample_seen[label])
                        if slot < 20:
                            reservoir[slot] = candidate
        print(f"CICIDS aggregate complete: {parquet_path.name}", flush=True)

    class_rows = []
    endpoint_rows = []
    concentration: dict[str, dict[str, float]] = defaultdict(dict)
    for label in sorted(counts):
        class_rows.append({
            "label": label, "strict_flow_count": counts[label], "pcap_count": len(pcaps[label]), "day_count": len(days[label]),
            "first_timestamp": datetime.fromtimestamp(first_ts[label], tz=timezone.utc).isoformat(),
            "last_timestamp": datetime.fromtimestamp(last_ts[label], tz=timezone.utc).isoformat(),
            "unique_src_ip": len(endpoint[label]["src_ip"]), "unique_dst_ip": len(endpoint[label]["dst_ip"]),
            "unique_src_port": len(endpoint[label]["src_port"]), "unique_dst_port": len(endpoint[label]["dst_port"]),
            "capture_window_5m_count": len(time_groups[label]),
        })
        for field in ("src_ip", "dst_ip", "src_port", "dst_port"):
            counter = endpoint[label][field]
            top1_value, top1 = top_share(counter, 1)
            _, top3 = top_share(counter, 3)
            concentration[label][field] = top1
            endpoint_rows.append({
                "label": label, "feature": field, "unique_value_count": len(counter), "top1_value": top1_value,
                "top1_share": top1, "top3_share": top3, "entropy_bits": entropy_bits(counter),
            })
    write_csv(out / "class_inventory.csv", class_rows, list(class_rows[0]))
    write_csv(out / "endpoint_port_shortcut.csv", endpoint_rows, list(endpoint_rows[0]))

    day_rows = []
    for label in sorted(counts):
        row = {"label": label, **{day: day_joint[(label, day)] for day in DAYS}}
        top_day = max((day_joint[(label, day)] for day in DAYS), default=0) / counts[label]
        row.update({"top_day_share": top_day, "day_count": len(days[label]), "risk": "TEMPORAL_CAPTURE_CONFOUNDING_RISK" if top_day >= 0.95 else "LOWER"})
        day_rows.append(row)
    write_csv(out / "class_day_matrix.csv", day_rows, ["label", *DAYS, "top_day_share", "day_count", "risk"])
    pcap_rows = [
        {"label": label, "source_pcap": pcap, "strict_flow_count": value, "within_class_share": value / counts[label], "risk": "TEMPORAL_CAPTURE_CONFOUNDING_RISK" if value / counts[label] >= 0.95 else "LOWER"}
        for (label, pcap), value in sorted(pcap_joint.items())
    ]
    write_csv(out / "class_pcap_matrix.csv", pcap_rows, list(pcap_rows[0]))
    hour_rows = [
        {"label": label, "day": day, "hour_utc": hour, "strict_flow_count": value, "within_class_share": value / counts[label]}
        for (label, day, hour), value in sorted(hour_joint.items())
    ]
    write_csv(out / "class_hour_distribution.csv", hour_rows, list(hour_rows[0]))

    sanity_fail = set()
    sanity_path = mapping / "reports/sanity_check.csv"
    with sanity_path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["status"] == "FAIL":
                sanity_fail.add(row["label"])
    eligibility_rows = []
    primary = []
    for label in sorted(counts):
        severe_endpoint = (
            concentration[label]["dst_ip"] >= CIC_SEVERE_ENDPOINT_TOP1
            and concentration[label]["dst_port"] >= CIC_SEVERE_ENDPOINT_TOP1
        )
        if counts[label] < CIC_TOO_SMALL:
            eligibility = "TOO_SMALL"
            reason = f"strict_flow_count<{CIC_TOO_SMALL}"
        elif label in sanity_fail:
            eligibility = "MAPPING_RISK"
            reason = "existing official-schedule sanity check FAIL; official per-flow label retained"
        elif len(time_groups[label]) < CIC_MIN_TIME_GROUPS:
            eligibility = "SEVERE_CAPTURE_CONFOUNDING"
            reason = f"fewer than {CIC_MIN_TIME_GROUPS} distinct five-minute capture groups"
        elif severe_endpoint:
            eligibility = "SEVERE_ENDPOINT_SHORTCUT"
            reason = f"dst_ip and dst_port top1 shares both >= {CIC_SEVERE_ENDPOINT_TOP1:.2f}"
        elif counts[label] < CIC_PRIMARY_MIN:
            eligibility = "SECONDARY_ONLY"
            reason = f"{CIC_TOO_SMALL}<=strict_flow_count<{CIC_PRIMARY_MIN}"
        else:
            eligibility = "PRIMARY_ELIGIBLE"
            reason = "passes fixed count, mapping, time-group, and severe endpoint gates"
            primary.append(label)
        eligibility_rows.append({
            "label": label, "strict_flow_count": counts[label], "eligibility": eligibility, "reason": reason,
            "day_count": len(days[label]), "pcap_count": len(pcaps[label]), "capture_window_5m_count": len(time_groups[label]),
            "dst_ip_top1_share": concentration[label]["dst_ip"], "dst_port_top1_share": concentration[label]["dst_port"],
            "schedule_sanity": "FAIL" if label in sanity_fail else "PASS_OR_NOT_APPLICABLE",
        })
    write_csv(out / "class_eligibility.csv", eligibility_rows, list(eligibility_rows[0]))

    day_nmi = normalized_mutual_information(day_joint)
    pcap_nmi = normalized_mutual_information(pcap_joint)
    feasible = bool(primary) and all(len(time_groups[label]) >= CIC_MIN_TIME_GROUPS for label in primary)
    split_lines = [
        "# CIC-IDS-2017 Group-aware Split Feasibility",
        "",
        f"- Strict reliable matched flows: **{sum(counts.values()):,}**.",
        f"- class ↔ day NMI: **{day_nmi:.6f}**.",
        f"- class ↔ PCAP NMI: **{pcap_nmi:.6f}**.",
        f"- Fixed thresholds: TOO_SMALL < {CIC_TOO_SMALL}; PRIMARY minimum {CIC_PRIMARY_MIN}; severe endpoint if dst IP and dst port top-1 shares are each >= {CIC_SEVERE_ENDPOINT_TOP1:.2f}; minimum {CIC_MIN_TIME_GROUPS} five-minute groups.",
        f"- PRIMARY_ELIGIBLE classes: **{', '.join(primary) if primary else 'NONE'}**.",
        f"- Group-aware Known Train/Validation/Test feasibility: **{'FEASIBLE' if feasible else 'NOT_FEASIBLE'}**.",
        "- Candidate group: `(source_pcap, floor(flow_start_epoch_utc / 300))`; all flows in a five-minute block must remain in one partition.",
        "- This is a feasibility audit only: no Unknown classes or split assignments were frozen.",
        "- The high NMI and one-day-per-attack structure must be reported as temporal/capture confounding; a random flow split is forbidden.",
    ]
    (out / "split_feasibility.md").write_text("\n".join(split_lines) + "\n", encoding="utf-8")
    (out / "confounding_summary.json").write_text(json.dumps({
        "strict_matched_flows": sum(counts.values()), "label_count": len(counts),
        "class_day_nmi": day_nmi, "class_pcap_nmi": pcap_nmi,
        "primary_eligible": primary, "group_aware_split_feasible": feasible,
        "group_definition": "source_pcap + five-minute UTC flow-start block",
    }, indent=2) + "\n", encoding="utf-8")
    flat_samples = [row for label in sorted(samples) for row in samples[label]]
    (out / "sample_candidates.json").write_text(json.dumps(flat_samples, indent=2, default=str) + "\n", encoding="utf-8")


def main() -> None:
    ensure_output_dirs()
    paths = locate_datasets()
    audit_ustc(paths["USTC-TFC2016"])
    print("USTC audit complete", flush=True)
    audit_cipher_inventory(paths["CipherSpectrum"])
    print("CipherSpectrum inventory/group/calibration audit complete", flush=True)
    audit_cstnet()
    print("CSTNET reuse audit complete", flush=True)
    audit_cicids(paths["CIC-IDS-2017"])
    print("CICIDS aggregate audit complete", flush=True)


if __name__ == "__main__":
    main()
