#!/usr/bin/env python3
"""Build Stage 15R-0 protocol, input-lineage, feature and regime audits."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from common import (
    CONFIG_PATH, OPENDETECT_ROOT, PROJECT_ROOT, ROOT, STAGE12_ROOT, STAGE14B_ROOT,
    STAGE14C5_ROOT, STAGE14C_INPUT_ROOT, STAGE3_PROTOCOL_ROOT, USTC_AUDIT_ROOT,
    packet_bin, read_csv, read_json, sha256_file, write_csv, write_json,
)


DATASET_NAMES = ("ustc", "vnat", "iscx_vpn", "iscx_tor")


def lightgbm_available() -> bool:
    vendor = ROOT / "vendor"
    if str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))
    return importlib.util.find_spec("lightgbm") is not None


def class_protocol_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    # Stage 12 has one frozen class composition per setting and five training seeds.
    for dataset in ("iscx_vpn", "iscx_tor"):
        protocol = read_json(STAGE12_ROOT / "protocol" / dataset / "unknown_class_protocol.json")
        split_rows = read_csv(STAGE12_ROOT / "protocol" / dataset / "split_manifest.csv")
        original_classes = sorted({row["canonical_class"] for row in read_csv(STAGE12_ROOT / "protocol" / dataset / "dataset_inventory.csv") if row.get("canonical_class")})
        eligible = list(protocol["eligible_classes_canonical_sorted"])
        for setting, item in protocol["settings"].items():
            for seed in range(2022, 2027):
                protocol_id = f"{setting}_seed{seed}"
                for class_name in eligible:
                    class_role = "unknown" if class_name in item["unknown_classes"] else "known"
                    selected = [row for row in split_rows if row["setting"] == setting and row["canonical_class"] == class_name]
                    counts = Counter(row["role"] for row in selected)
                    rows.append({
                        "dataset": dataset, "protocol_id": protocol_id, "setting": setting, "seed": seed,
                        "semantic_class_type": "application/service from controlled-capture filename",
                        "original_class_count": len(original_classes), "eligible_class_count": len(eligible),
                        "class_name": class_name, "class_role": class_role,
                        "known_train_samples": counts.get("known_train", 0),
                        "known_validation_samples": counts.get("known_validation", 0),
                        "known_test_samples": counts.get("known_test", 0),
                        "unknown_test_samples": counts.get("unknown_test", 0),
                        "source_group_count": len({row["source_file"] for row in selected}),
                        "domain_states": ";".join(sorted({row["domain_state"] for row in selected})),
                    })
    # VNAT has 15 independently frozen class compositions.
    document = read_json(STAGE14B_ROOT / "vnat_open_set_protocol.json")
    split_rows = read_csv(STAGE14B_ROOT / "vnat_split_manifest.csv")
    by_protocol_class: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in split_rows:
        by_protocol_class[(row["protocol_id"], row["application"])].append(row)
    for protocol in document["protocols"]:
        for class_name in document["retained_applications"]:
            selected = by_protocol_class[(protocol["protocol_id"], class_name)]
            counts = Counter(row["split"] for row in selected)
            role = "unknown" if class_name in protocol["unknown_applications"] else "known"
            rows.append({
                "dataset": "vnat", "protocol_id": protocol["protocol_id"], "setting": protocol["setting"], "seed": protocol["seed"],
                "semantic_class_type": "application; VPN/non-VPN retained as metadata",
                "original_class_count": 10, "eligible_class_count": 10, "class_name": class_name, "class_role": role,
                "known_train_samples": counts.get("train", 0) if role == "known" else 0,
                "known_validation_samples": counts.get("validation", 0) if role == "known" else 0,
                "known_test_samples": counts.get("test", 0) if role == "known" else 0,
                "unknown_test_samples": counts.get("unknown_test", 0) if role == "unknown" else 0,
                "source_group_count": len({row["group_id"] for row in selected}),
                "domain_states": ";".join(sorted({row["vpn_status"] for row in selected})),
            })
    # USTC has three frozen class-holdout scenarios.
    split_rows = read_csv(STAGE3_PROTOCOL_ROOT / "split_manifest.csv")
    for row in split_rows:
        rows.append({
            "dataset": "ustc", "protocol_id": row["scenario"], "setting": row["scenario"], "seed": 2022,
            "semantic_class_type": "application for benign traffic or malware family for malicious traffic",
            "original_class_count": 20, "eligible_class_count": 20, "class_name": row["class_name"],
            "class_role": row["class_role"].lower(),
            "known_train_samples": int(row["source_train_count"]) if row["class_role"] == "Known" else 0,
            "known_validation_samples": int(row["source_val_count"]) if row["class_role"] == "Known" else 0,
            "known_test_samples": int(row["source_test_count"]) if row["class_role"] == "Known" else 0,
            "unknown_test_samples": int(row["source_test_count"]) if row["class_role"] == "Unknown" else 0,
            "source_group_count": 1,
            "domain_states": "Malware" if row["class_name"] in {"Cridex", "Geodo", "Htbot", "Miuref", "Neris", "Nsis-ay", "Shifu", "Tinba", "Virut", "Zeus"} else "Benign",
        })
    return rows


def lineage_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    # Stage 12 arrays and manifests.
    for dataset in ("iscx_vpn", "iscx_tor"):
        protocol_doc = read_json(STAGE12_ROOT / "protocol" / dataset / "unknown_class_protocol.json")
        split_manifest_path = STAGE12_ROOT / "protocol" / dataset / "split_manifest.csv"
        split_rows = read_csv(split_manifest_path)
        for setting in protocol_doc["settings"]:
            base = STAGE12_ROOT / "artifacts" / dataset / "protocol" / setting
            for role, file_name in (("known_train", "known_train.npz"), ("known_validation", "known_validation.npz"), ("known_test", "known_test.npz"), ("unknown_test", "unknown_test.npz")):
                path = base / file_name
                values_loaded = role in {"known_train", "known_validation"}
                role_count = sum(
                    row["setting"] == setting and row["role"] == role
                    for row in split_rows
                )
                if values_loaded:
                    bundle = np.load(path, mmap_mode="r", allow_pickle=False)
                    data, target = bundle["data"], bundle["target"]
                    samples, shape, dtype = len(data), str(list(data.shape)), str(data.dtype)
                    label_samples = len(target)
                    label_min, label_max = int(target.min()), int(target.max())
                    label_alignment: bool | str = len(data) == len(target) == role_count
                else:
                    # Strict boundary: checksum and manifest counts are metadata; the
                    # Known/Unknown Test NPZ members are intentionally never opened.
                    samples, shape, dtype = role_count, "NOT_OPENED", "NOT_OPENED"
                    label_samples = role_count
                    label_min, label_max = "", ""
                    label_alignment = "MANIFEST_COUNT_ONLY"
                rows.append({
                    "dataset": dataset, "protocol_id": f"{setting}_seed2022", "role": role,
                    "array_path": str(path.resolve()), "array_sha256": sha256_file(path),
                    "samples": samples, "shape": shape, "dtype": dtype,
                    "label_samples": label_samples, "label_min": label_min, "label_max": label_max,
                    "manifest_path": str(split_manifest_path.resolve()),
                    "manifest_sha256": sha256_file(split_manifest_path),
                    "label_alignment": label_alignment, "values_loaded": values_loaded,
                    "unknown_loaded_for_audit": False,
                    "notes": "Known Train/Validation values inspected" if values_loaded else "file hashed and frozen manifest counted; Test values not opened",
                })
    # VNAT inputs: strictly inspect only train/validation arrays here.
    for protocol_id in ("medium_seed2025", "medium_seed2026"):
        base = STAGE14C_INPUT_ROOT / "runs" / protocol_id / "inputs"
        for role in ("train", "validation"):
            data_path = base / f"{role}_images.npy"
            label_path = base / f"{role}_labels.npy"
            data = np.load(data_path, mmap_mode="r", allow_pickle=False)
            target = np.load(label_path, mmap_mode="r", allow_pickle=False)
            rows.append({
                "dataset": "vnat", "protocol_id": protocol_id, "role": f"known_{role}",
                "array_path": str(data_path.resolve()), "array_sha256": sha256_file(data_path),
                "samples": len(data), "shape": str(list(data.shape)), "dtype": str(data.dtype),
                "label_samples": len(target), "label_min": int(target.min()), "label_max": int(target.max()),
                "manifest_path": str((base / "input_manifest.csv").resolve()), "manifest_sha256": sha256_file(base / "input_manifest.csv"),
                "label_alignment": len(data) == len(target), "values_loaded": True, "unknown_loaded_for_audit": False,
                "notes": "reuses Stage 14C frozen Known-only inputs",
            })
    # USTC aligned arrays.  Protocol-specific filtering is by frozen labels.
    summary = read_json(USTC_AUDIT_ROOT / "outputs" / "input_alignment_summary.json")
    for role in ("train", "val"):
        path = USTC_AUDIT_ROOT / "artifacts" / f"{role}_images.npy"
        data = np.load(path, mmap_mode="r", allow_pickle=False)
        label_path = PROJECT_ROOT / "outputs" / "stage1" / "modelA" / "embeddings" / f"labels_{role}.npy"
        target = np.load(label_path, mmap_mode="r", allow_pickle=False)
        rows.append({
            "dataset": "ustc", "protocol_id": "A-2", "role": "known_train" if role == "train" else "known_validation",
            "array_path": str(path.resolve()), "array_sha256": sha256_file(path),
            "samples": len(data), "shape": str(list(data.shape)), "dtype": str(data.dtype),
            "label_samples": len(target), "label_min": int(target.min()), "label_max": int(target.max()),
            "manifest_path": str((USTC_AUDIT_ROOT / "outputs" / "input_alignment_manifest.csv").resolve()),
            "manifest_sha256": sha256_file(USTC_AUDIT_ROOT / "outputs" / "input_alignment_manifest.csv"),
            "label_alignment": len(data) == len(target) and bool(summary["all_inputs_valid"]), "values_loaded": True, "unknown_loaded_for_audit": False,
            "notes": "full 20-class aligned array; pilot loader must restrict to frozen A-2 Known labels before constructing a Dataset",
        })
    # USTC Known/Unknown Test share the historical full test array.  Record only
    # its immutable hash and frozen A-2 manifest counts; never open Test values.
    ustc_test_path = USTC_AUDIT_ROOT / "artifacts" / "test_images.npy"
    stage3_rows = [row for row in read_csv(STAGE3_PROTOCOL_ROOT / "split_manifest.csv") if row["scenario"] == "A-2"]
    for role, class_role in (("known_test", "Known"), ("unknown_test", "Unknown")):
        samples = sum(int(row["source_test_count"]) for row in stage3_rows if row["class_role"] == class_role)
        rows.append({
            "dataset": "ustc", "protocol_id": "A-2", "role": role,
            "array_path": str(ustc_test_path.resolve()), "array_sha256": sha256_file(ustc_test_path),
            "samples": samples, "shape": "NOT_OPENED", "dtype": "NOT_OPENED",
            "label_samples": samples, "label_min": "", "label_max": "",
            "manifest_path": str((STAGE3_PROTOCOL_ROOT / "split_manifest.csv").resolve()),
            "manifest_sha256": sha256_file(STAGE3_PROTOCOL_ROOT / "split_manifest.csv"),
            "label_alignment": "MANIFEST_COUNT_ONLY", "values_loaded": False,
            "unknown_loaded_for_audit": False,
            "notes": "shared Test array hashed and frozen manifest counted; Test values not opened",
        })
    return rows


def load_feature_rows(dataset: str) -> list[dict[str, str]]:
    if dataset in {"ustc", "iscx_vpn", "iscx_tor"}:
        path = ROOT / "feature_cache" / dataset / "flow_statistics.csv"
        if not path.is_file():
            return []
        return read_csv(path)
    if dataset == "vnat":
        cache = STAGE14C5_ROOT / "raw_feature_cache"
        uids = np.load(cache / "flow_uids.npy", allow_pickle=False)
        stats = np.load(cache / "flow_statistics.npy", mmap_mode="r", allow_pickle=False)
        directions = np.load(cache / "packet_directions.npy", mmap_mode="r", allow_pickle=False)
        masks = np.load(cache / "packet_mask.npy", mmap_mode="r", allow_pickle=False)
        manifest = {row["flow_uid"]: row for row in read_csv(STAGE14B_ROOT / "vnat_split_manifest.csv")}
        output = []
        for index, uid_raw in enumerate(uids):
            uid = str(uid_raw)
            row = manifest[uid]
            packet_count, total_bytes, duration, length_mean, length_std, iat_mean, iat_std, ratio = map(float, stats[index])
            seq = directions[index][masks[index].astype(bool)]
            seq = seq[seq != 0]
            changes = int(np.sum(seq[1:] != seq[:-1])) if len(seq) > 1 else 0
            output.append({
                "dataset": "vnat", "flow_uid": uid, "class_name": row["application"], "source_file": row["source_pcap_path"],
                "domain_state": row["vpn_status"], "official_category": row["application"],
                "packet_count": int(packet_count), "total_bytes": int(total_bytes), "payload_bytes": "",
                "duration_seconds": duration, "packet_length_mean": length_mean, "packet_length_std": length_std,
                "iat_mean_seconds": iat_mean, "iat_std_seconds": iat_std,
                "forward_packets": "", "reverse_packets": "", "forward_reverse_ratio": ratio,
                "direction_changes": changes,
                "padding_ratio_packet_slots": max(0, 8 - min(int(packet_count), 8)) / 8,
                "truncated_after_packet_8": int(packet_count) > 8,
                "encryption_regime": "VPN tunnel" if row["vpn_status"] == "vpn" else ("SSH" if row["application"] in {"ssh", "scp", "sftp"} else "undetermined"),
                "protocol_tokens": "",
            })
        return output
    raise KeyError(dataset)


def packet_statistics() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    regime_rows: list[dict[str, object]] = []
    for dataset in DATASET_NAMES:
        features = load_feature_rows(dataset)
        if not features:
            rows.append({"dataset": dataset, "class_name": "__CACHE_NOT_READY__", "packet_bin": "NOT_AVAILABLE", "flow_count": 0, "metric_scope": "NOT_RUN", "packet_count_mean": "", "duration_mean": "", "total_bytes_mean": "", "payload_bytes_mean": "", "padding_ratio_mean": "", "truncation_ratio": "", "direction_changes_mean": "", "iat_mean_seconds": ""})
            continue
        by_class: dict[str, list[dict[str, str]]] = defaultdict(list)
        by_regime: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
        for row in features:
            by_class[row["class_name"]].append(row)
            by_regime[(row["class_name"], row["encryption_regime"])].append(row)
        for class_name, class_rows in sorted(by_class.items()):
            for bin_name in ("1", "2-4", "5-10", "11-20", ">20"):
                selected = [row for row in class_rows if packet_bin(int(float(row["packet_count"]))) == bin_name]
                if not selected:
                    continue
                def mean(field: str) -> float | str:
                    values = [float(row[field]) for row in selected if row.get(field, "") not in {"", None}]
                    return float(np.mean(values)) if values else ""
                def quantile(field: str, q: float) -> float | str:
                    values = [float(row[field]) for row in selected if row.get(field, "") not in {"", None}]
                    return float(np.quantile(values, q)) if values else ""
                rows.append({
                    "dataset": dataset, "class_name": class_name, "packet_bin": bin_name, "flow_count": len(selected),
                    "metric_scope": ("known_train_validation_union_only; exact full-flow statistics except first8 direction changes" if dataset == "vnat" else "exact full-flow statistics except noted payload fields"),
                    "packet_count_mean": mean("packet_count"), "duration_mean": mean("duration_seconds"),
                    "total_bytes_mean": mean("total_bytes"), "payload_bytes_mean": mean("payload_bytes"),
                    "padding_ratio_mean": mean("padding_ratio_packet_slots"),
                    "truncation_ratio": float(np.mean([str(row["truncated_after_packet_8"]).lower() in {"true", "1"} for row in selected])),
                    "direction_changes_mean": mean("direction_changes"), "iat_mean_seconds": mean("iat_mean_seconds"),
                    "packet_count_median": quantile("packet_count", 0.5),
                    "packet_count_p1": quantile("packet_count", 0.01), "packet_count_p99": quantile("packet_count", 0.99),
                    "duration_median": quantile("duration_seconds", 0.5),
                    "duration_p1": quantile("duration_seconds", 0.01), "duration_p99": quantile("duration_seconds", 0.99),
                    "total_bytes_median": quantile("total_bytes", 0.5),
                    "total_bytes_p1": quantile("total_bytes", 0.01), "total_bytes_p99": quantile("total_bytes", 0.99),
                    "payload_bytes_median": quantile("payload_bytes", 0.5),
                    "padding_ratio_median": quantile("padding_ratio_packet_slots", 0.5),
                    "iat_median_seconds": quantile("iat_mean_seconds", 0.5),
                })
        for (class_name, regime), selected in sorted(by_regime.items()):
            regime_rows.append({
                "dataset": dataset, "class_name": class_name, "encryption_regime": regime,
                "flow_count": len(selected), "evidence": ("actual tshark frame.protocols" if dataset == "iscx_vpn" else "actual Scapy packet-layer tokens") if dataset.startswith("iscx") else ("VPN metadata or application-semantic inference; otherwise undetermined" if dataset == "vnat" else "not parsed from Stage-0 raw bytes"),
                "packet_count_mean": float(np.mean([float(row["packet_count"]) for row in selected])),
                "duration_mean": float(np.mean([float(row["duration_seconds"]) for row in selected])),
                "padding_ratio_mean": float(np.mean([float(row["padding_ratio_packet_slots"]) for row in selected])),
                "truncation_ratio": float(np.mean([str(row["truncated_after_packet_8"]).lower() in {"true", "1"} for row in selected])),
            })
    return rows, regime_rows


def feature_report(cache_ready: dict[str, bool], vnat_scope: dict[str, object]) -> str:
    lightgbm = lightgbm_available()
    return f"""# Stage 15R-0 — Current feature pipeline audit

## Verified Open-Detect input semantics

- The active raw-PCAP pipeline uses bidirectional IPv4 TCP/UDP flow/session keys, a 60-second idle timeout, and TCP SYN/FIN/RST session boundaries.
- Each flow image contains at most the first 8 packets. Each packet contributes 128 bytes: 80 bytes from the zero-addressed IPv4/header serialization and 48 bytes from Scapy `Raw` payload. Eight packet blocks are concatenated and reshaped row-major to `32×32` uint8.
- Ethernet bytes are not included in the 128-byte packet block because serialization starts at the IPv4 layer. IPv4 source/destination addresses are zeroed. Other header fields, including transport ports when present, remain visible.
- Packet boundaries are implicit fixed 128-byte slots. Direction and IAT are not represented in E0. Missing packet slots are zero padded; packets after packet 8 are truncated.
- The encoder receives `ToTensor()` values in `[0,1]`. No dataset-level mean/std scaler is fitted for E0. Training uses random crop and horizontal flip; validation is deterministic `ToTensor()`.
- Stage 12 and VNAT Native use the same 32×32 representation semantics, but their flow construction differs: Stage 12 replays aggregate PCAPs with a 60-second timeout/TCP boundaries, while VNAT uses tshark stream IDs from its own clean-pool construction. USTC uses preserved Stage-0 five-tuple flow PKLs.

## Lineage and alignment

- USTC: `input_alignment_manifest.csv` maps every historical Stage-1 flow ID to exactly one Stage-0 PKL flow and one image row; the existing alignment gate reports all inputs valid.
- VNAT: Stage 14C input manifests map `flow_uid → cache_index → image/label`; Stage 14B supplies frozen application and VPN metadata. Known Train/Validation manifests are used by pilot development.
- VNAT's feature cache intentionally contains the union of flows that appear as Known Train/Validation in at least one frozen protocol: {vnat_scope['cache_flow_count']:,}/{vnat_scope['clean_pool_flow_count']:,}. The two omitted rows occur only as Known Test in every protocol in which they are Known; their feature values were therefore not opened. This is a development-visibility boundary, not a preprocessing loss.
- ISCX-VPN/ISCXTor: each image has a SHA-256 flow ID, image hash, source capture and frozen role. Exact-image groups were kept disjoint by the Stage 12 protocol logic.
- Unknown Test values are not loaded by the pilot input builders. Stage 15R-0 may count frozen manifest rows, but does not use Unknown feature values or outcomes for feature/model selection.

## Recoverability audit

| Dataset | Exact full-flow statistics cache | First-8 length/IAT/direction | Payload bytes | Encryption evidence |
|---|---|---|---|---|
| USTC | {'ready' if cache_ready['ustc'] else 'not ready'} | recoverable from Stage-0 PKLs | not separately audited in Stage-0 tuple | currently undetermined |
| VNAT | ready from Stage 14C.5 cache | ready | unavailable in existing cache | VPN metadata; SSH application inference; otherwise undetermined |
| ISCX-VPN | {'ready' if cache_ready['iscx_vpn'] else 'not ready'} | replayed from source PCAP with tshark | tshark TCP/UDP payload length | actual `frame.protocols` plus VPN metadata |
| ISCXTor | {'ready' if cache_ready['iscx_tor'] else 'not ready'} | replayed from source PCAP with Scapy `PcapReader` | Scapy TCP/UDP payload length | actual Scapy packet-layer tokens plus Tor metadata |

## Data and task-definition differences from published baselines

- Current ISCX-VPN is a 16-application pool and its Medium pilot has 13 Known applications. The local YaTC reproduction reports a 7-class `ISCXVPN2016_MFR` task, while the local ET-BERT official processed run is a balanced 12-class packet-level service task. Their reported Accuracy cannot be compared directly with the current 13/15-class application-level flow task.
- Current ISCXTor has 10 eligible canonical classes and the Medium pilot has 8 Known classes. YaTC uses its own MFR preprocessing and large-scale masked-autoencoder pretraining; its published 99.72% result is not an architecture-only comparison to the present raw-byte Open-Detect flow image.
- Current USTC mixes ten benign applications with ten malware families and has a much larger aligned training pool. Its high Accuracy does not by itself prove that the same representation is adequate for fine-grained encrypted application pairs.

## Dependency gate

- `lightgbm` available in the fixed environment: **{lightgbm}**.
- E3 uses the real project-local LightGBM implementation when the availability flag is true. The shared Conda environment remains unchanged; a scikit-learn estimator is never relabeled as LightGBM.
"""


def main() -> None:
    protocol_rows = class_protocol_rows()
    lineage = lineage_rows()
    cache_ready = {name: (ROOT / "feature_cache" / name / "cache_audit.json").is_file() for name in ("ustc", "iscx_vpn", "iscx_tor")}
    cache_ready["vnat"] = (STAGE14C5_ROOT / "raw_feature_cache" / "cache_audit.json").is_file()
    packet_rows, regime_rows = packet_statistics()
    write_csv(ROOT / "dataset_protocol_audit.csv", protocol_rows)
    write_csv(ROOT / "input_shape_and_lineage_audit.csv", lineage)
    write_csv(ROOT / "packet_flow_statistics.csv", packet_rows)
    write_csv(ROOT / "encryption_regime_audit.csv", regime_rows, ["dataset", "class_name", "encryption_regime", "flow_count", "evidence", "packet_count_mean", "duration_mean", "padding_ratio_mean", "truncation_ratio"])
    clean_rows = read_csv(STAGE14B_ROOT / "vnat_split_manifest.csv")
    clean_ids = {row["flow_uid"] for row in clean_rows}
    cache_ids = set(map(str, np.load(STAGE14C5_ROOT / "raw_feature_cache" / "flow_uids.npy", allow_pickle=False)))
    missing_ids = sorted(clean_ids - cache_ids)
    extra_ids = sorted(cache_ids - clean_ids)
    metadata_by_id = {row["flow_uid"]: row for row in clean_rows}
    vnat_scope = {
        "clean_pool_flow_count": len(clean_ids),
        "cache_flow_count": len(cache_ids),
        "missing_from_development_cache": missing_ids,
        "extra_in_development_cache": extra_ids,
        "missing_roles": sorted({(metadata_by_id[uid]["class_role"], metadata_by_id[uid]["split"]) for uid in missing_ids}),
        "explanation": "Stage14C5 cache is the union of Known Train/Validation flow IDs; rows visible only as Known Test are intentionally not loaded",
        "known_or_unknown_test_feature_values_loaded": False,
    }
    write_json(ROOT / "vnat_development_cache_scope_audit.json", vnat_scope)
    (ROOT / "feature_pipeline_audit.md").write_text(feature_report(cache_ready, vnat_scope), encoding="utf-8")
    source_hashes = {
        "stage12_config": sha256_file(STAGE12_ROOT / "configs" / "stage12_config.json"),
        "stage12_vpn_split": sha256_file(STAGE12_ROOT / "protocol" / "iscx_vpn" / "split_manifest.csv"),
        "stage12_tor_split": sha256_file(STAGE12_ROOT / "protocol" / "iscx_tor" / "split_manifest.csv"),
        "stage14b_protocol": sha256_file(STAGE14B_ROOT / "vnat_open_set_protocol.json"),
        "stage14b_split": sha256_file(STAGE14B_ROOT / "vnat_split_manifest.csv"),
        "stage3_protocol": sha256_file(STAGE3_PROTOCOL_ROOT / "stage3_protocol.json"),
        "ustc_alignment": sha256_file(USTC_AUDIT_ROOT / "outputs" / "input_alignment_manifest.csv"),
        "stage15r_config": sha256_file(CONFIG_PATH),
    }
    write_json(ROOT / "source_hashes_before.json", source_hashes)
    write_json(ROOT / "stage15r0_status.json", {
        "status": "PASS" if all(cache_ready.values()) else "PARTIAL_CACHE_BUILD_REQUIRED",
        "cache_ready": cache_ready,
        "dataset_protocol_rows": len(protocol_rows), "lineage_rows": len(lineage),
        "packet_stat_rows": len(packet_rows), "encryption_rows": len(regime_rows),
        "unknown_test_used_for_feature_or_model_selection": False,
        "known_test_used_for_feature_or_model_selection": False,
        "lightgbm_available": lightgbm_available(),
        "vnat_development_cache_scope": vnat_scope,
    })
    print(json.dumps(read_json(ROOT / "stage15r0_status.json"), indent=2))


if __name__ == "__main__":
    main()
