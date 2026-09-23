#!/usr/bin/env python3
from __future__ import annotations

import csv
import importlib.util
import json
import random
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from audit_utils import OUTPUT_ROOT, PROJECT_ROOT, SEED, scan_classic_pcap, write_csv


CIPHER_DATA_ROOT = PROJECT_ROOT.parents[1] / "Dataset/ipherSpectrum(sok)/ipherSpectrum(sok)"
FULL_MANIFEST = PROJECT_ROOT / "cipherspectrum_labeling/outputs/cipherspectrum_label_manifest.csv"
OFFICIAL_MANIFEST = PROJECT_ROOT / "cipherspectrum_labeling/outputs/cipherspectrum_official40_manifest.csv"
SOURCES = ("aes-128-gcm", "aes-256-gcm", "chacha20-poly1305")


def scan_worker(path_text: str) -> dict[str, object]:
    return scan_classic_pcap(Path(path_text), fingerprint_packets=8)


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def run_provenance() -> dict[str, object]:
    out = OUTPUT_ROOT / "cipherspectrum"
    rows = load_rows(FULL_MANIFEST)
    paths = [CIPHER_DATA_ROOT / row["relative_path"] for row in rows]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"CipherSpectrum manifest paths missing: {len(missing)}")
    with ProcessPoolExecutor(max_workers=8) as pool:
        scans = list(pool.map(scan_worker, map(str, paths), chunksize=64))
    fingerprint_rows = []
    by_relative: dict[str, dict[str, object]] = {}
    for index, (row, scan) in enumerate(zip(rows, scans), start=1):
        result = {
            "flow_id": row["flow_id"], "relative_path": row["relative_path"],
            "source_group": row["capture_group_directory"], "class_name": row["canonical_label"],
            "official40_eligible": row["official40_eligible"], "split_group_id": row["split_group_id"],
            **scan,
        }
        fingerprint_rows.append(result)
        by_relative[row["relative_path"]] = result
        if index % 10_000 == 0:
            print(f"CipherSpectrum SHA/fingerprint: {index:,}/{len(rows):,}", flush=True)
    write_csv(
        out / "all_pcap_fingerprints.csv", fingerprint_rows,
        ["flow_id", "relative_path", "absolute_path", "source_group", "class_name", "official40_eligible", "split_group_id", "size_bytes", "sha256", "normalized_fingerprint", "readable", "packet_count", "first_packet_timestamp", "last_packet_timestamp", "error"],
    )

    official = [row for row in rows if row["official40_eligible"].lower() == "true"]
    canonical = [row for row in official if row["capture_group_directory"] in SOURCES]
    mix = [row for row in official if row["capture_group_directory"] == "mix"]
    sha_index: dict[str, list[dict[str, str]]] = defaultdict(list)
    fp_index: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in canonical:
        scan = by_relative[row["relative_path"]]
        sha_index[str(scan["sha256"])].append(row)
        fp_index[str(scan["normalized_fingerprint"])].append(row)

    pairs = []
    exact_mix: set[str] = set()
    normalized_mix: set[str] = set()
    per_source_exact: dict[str, set[str]] = defaultdict(set)
    per_source_normalized: dict[str, set[str]] = defaultdict(set)
    for mix_row in mix:
        mix_scan = by_relative[mix_row["relative_path"]]
        exact_matches = sha_index.get(str(mix_scan["sha256"]), [])
        fp_matches = fp_index.get(str(mix_scan["normalized_fingerprint"]), [])
        if exact_matches:
            exact_mix.add(mix_row["flow_id"])
        if fp_matches:
            normalized_mix.add(mix_row["flow_id"])
        emitted: set[tuple[str, str]] = set()
        for match_type, matches in (("EXACT_SHA256", exact_matches), ("NORMALIZED_ONLY", fp_matches)):
            for other in matches:
                if match_type == "NORMALIZED_ONLY" and other in exact_matches:
                    continue
                key = (match_type, other["flow_id"])
                if key in emitted:
                    continue
                emitted.add(key)
                source = other["capture_group_directory"]
                if match_type == "EXACT_SHA256":
                    per_source_exact[source].add(mix_row["flow_id"])
                per_source_normalized[source].add(mix_row["flow_id"])
                pairs.append({
                    "match_type": match_type, "mix_flow_id": mix_row["flow_id"], "mix_relative_path": mix_row["relative_path"],
                    "canonical_flow_id": other["flow_id"], "canonical_relative_path": other["relative_path"], "canonical_source_group": source,
                    "sha256": mix_scan["sha256"], "normalized_fingerprint": mix_scan["normalized_fingerprint"],
                    "same_split_group_id": mix_row["split_group_id"] == other["split_group_id"],
                    "same_file_stem": Path(mix_row["relative_path"]).stem == Path(other["relative_path"]).stem,
                })
    write_csv(
        out / "mix_duplicate_pairs.csv", pairs,
        ["match_type", "mix_flow_id", "mix_relative_path", "canonical_flow_id", "canonical_relative_path", "canonical_source_group", "sha256", "normalized_fingerprint", "same_split_group_id", "same_file_stem"],
    )

    source_groups = {
        source: {row["split_group_id"] for row in canonical if row["capture_group_directory"] == source}
        for source in SOURCES
    }
    mix_groups = {row["split_group_id"] for row in mix}
    mix_stems = {Path(row["relative_path"]).stem for row in mix}
    audit_rows = []
    for source in (*SOURCES, "ANY_CANONICAL"):
        groups = set().union(*source_groups.values()) if source == "ANY_CANONICAL" else source_groups[source]
        exact_set = exact_mix if source == "ANY_CANONICAL" else per_source_exact[source]
        normalized_set = normalized_mix if source == "ANY_CANONICAL" else per_source_normalized[source]
        source_stems = {
            Path(row["relative_path"]).stem for row in canonical
            if source == "ANY_CANONICAL" or row["capture_group_directory"] == source
        }
        audit_rows.append({
            "comparison": f"mix_vs_{source}", "mix_pcap_count": len(mix),
            "exact_duplicate_mix_count": len(exact_set), "exact_duplicate_rate": len(exact_set) / len(mix),
            "normalized_fingerprint_overlap_mix_count": len(normalized_set), "normalized_fingerprint_overlap_rate": len(normalized_set) / len(mix),
            "normalized_only_mix_count": len(normalized_set - exact_set),
            "shared_split_group_count": len(mix_groups & groups),
            "mix_split_group_count": len(mix_groups), "shared_file_stem_count": len(mix_stems & source_stems),
        })
    write_csv(out / "mix_provenance_audit.csv", audit_rows, list(audit_rows[0]))
    exact_rate = len(exact_mix) / len(mix)
    normalized_rate = len(normalized_mix) / len(mix)
    shared_groups = len(mix_groups & set().union(*source_groups.values()))
    shared_group_rate = shared_groups / len(mix_groups)
    if normalized_rate >= 0.50 or exact_rate >= 0.10:
        status = "DERIVED_OR_DUPLICATED"
    elif normalized_mix or exact_mix or shared_groups:
        status = "PARTIALLY_OVERLAPPING"
    else:
        status = "UNCERTAIN"
    summary = {
        "mix_status": status, "mix_pcap_count": len(mix), "exact_duplicate_mix_count": len(exact_mix),
        "normalized_fingerprint_overlap_mix_count": len(normalized_mix), "mix_unique_split_groups": len(mix_groups),
        "shared_split_groups_with_canonical": shared_groups, "shared_split_group_rate": shared_group_rate,
        "canonical_primary_candidate_pcap_count": len(canonical), "recommend_exclude_mix_from_primary": status != "INDEPENDENT",
    }
    (out / "mix_provenance_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (out / "mix_provenance_summary.md").write_text(
        "# CipherSpectrum MIX Provenance Summary\n\n"
        f"- MIX_STATUS = **{status}**.\n"
        f"- MIX PCAPs: **{len(mix):,}**.\n"
        f"- Exact SHA-256 duplicate MIX files: **{len(exact_mix):,} ({exact_rate:.2%})**.\n"
        f"- Normalized packet-fingerprint overlap: **{len(normalized_mix):,} ({normalized_rate:.2%})**.\n"
        f"- MIX collection groups shared with canonical sources: **{shared_groups:,}/{len(mix_groups):,} ({shared_group_rate:.2%})**.\n"
        f"- Canonical three-cipher candidate: **{len(canonical):,} PCAPs (40 classes x 3,000)**.\n"
        f"- Primary recommendation: **{'exclude MIX' if status != 'INDEPENDENT' else 'MIX may be retained'}**. No source file was deleted or changed.\n\n"
        "The normalized fingerprint excludes the PCAP global header and combines packet count with the first eight packets' lengths, direction, relative timestamps, and complete frame-byte hashes. Shared collection IDs alone establish collection/session dependence, not byte identity.\n",
        encoding="utf-8",
    )
    return summary


def load_preflight_module():
    path = PROJECT_ROOT / "stage5_5_cstnet_preflight/scripts/run_preflight_audit.py"
    spec = importlib.util.spec_from_file_location("cstnet_preflight_reuse", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_input_audit() -> dict[str, object]:
    out = OUTPUT_ROOT / "cipherspectrum"
    rows = load_rows(OFFICIAL_MANIFEST)
    labels = sorted({row["canonical_label"] for row in rows})[:20]
    selected = []
    for label in labels:
        candidates = sorted((row for row in rows if row["canonical_label"] == label), key=lambda row: row["flow_id"])
        rng = random.Random(f"{SEED}:{label}")
        selected.extend(candidates[index] for index in sorted(rng.sample(range(len(candidates)), min(20, len(candidates)))))
    module = load_preflight_module()
    manifest_rows = []
    leakage_rows = []
    for index, row in enumerate(selected):
        path = CIPHER_DATA_ROOT / row["relative_path"]
        sample_id = f"cipher-{index:04d}"
        encoding = module.encode_pcap(path, sample_id)
        raw = encoding.raw_blob.lower()
        model = encoding.image.tobytes().lower() if encoding.image is not None else b""
        label_bytes = row["canonical_label"].lower().encode("ascii", errors="ignore")
        tokens = [token.encode("ascii") for token in row["canonical_label"].lower().replace("-", ".").split(".") if len(token) >= 4]
        raw_domain = bool(label_bytes and label_bytes in raw)
        model_domain = bool(label_bytes and label_bytes in model)
        raw_sni, raw_snis = module.parse_tls_sni(encoding.raw_tcp_blobs)
        model_sni, model_snis = module.parse_tls_sni(encoding.retained_tcp_blobs)
        model_token = any(token in model for token in tokens)
        direct = model_domain or model_sni
        manifest_rows.append({
            "sample_id": sample_id, "flow_id": row["flow_id"], "class_name": row["canonical_label"], "source_group": row["capture_group_directory"],
            "relative_path": row["relative_path"], "seed": SEED, "packets_found": encoding.packets_found, "packets_used": encoding.packets_used,
            "input_valid": encoding.input_valid, "failure_reason": encoding.failure_reason, "source_bytes_retained": encoding.source_bytes_retained,
            "ip_fields_checked": encoding.ip_fields_checked, "ip_fields_masked": encoding.ip_fields_masked,
            "port_fields_checked": encoding.port_fields_checked, "port_fields_visible": encoding.port_fields_visible,
        })
        leakage_rows.append({
            "sample_id": sample_id, "class_name": row["canonical_label"], "raw_pcap_class_domain_visible": raw_domain,
            "raw_pcap_sni_parse_success": raw_sni, "raw_sni_values": json.dumps(raw_snis),
            "model_input_class_domain_visible": model_domain, "model_input_sni_visible": model_sni,
            "model_input_sni_values": json.dumps(model_snis), "model_input_class_token_visible": model_token,
            "model_input_direct_domain_or_sni": direct, "model_input_domain_sni_or_token": direct or model_token,
            "input_valid": encoding.input_valid,
        })
        if (index + 1) % 50 == 0:
            print(f"CipherSpectrum Open-Detect input sample: {index + 1}/{len(selected)}", flush=True)
    write_csv(out / "raw_input_manifest.csv", manifest_rows, list(manifest_rows[0]))
    write_csv(out / "domain_sni_leakage.csv", leakage_rows, list(leakage_rows[0]))
    valid = [row for row in leakage_rows if row["input_valid"]]
    count = lambda key: sum(bool(row[key]) for row in valid)
    raw_domain_count = count("raw_pcap_class_domain_visible")
    domain_count = count("model_input_class_domain_visible")
    sni_count = count("model_input_sni_visible")
    token_count = count("model_input_class_token_visible")
    union_count = count("model_input_domain_sni_or_token")
    total_ip = sum(int(row["ip_fields_checked"]) for row in manifest_rows)
    masked_ip = sum(int(row["ip_fields_masked"]) for row in manifest_rows)
    total_port = sum(int(row["port_fields_checked"]) for row in manifest_rows)
    visible_port = sum(int(row["port_fields_visible"]) for row in manifest_rows)
    union_rate = union_count / len(valid) if valid else 1.0
    level = "LOW" if union_rate < 0.10 else "MODERATE" if union_rate < 0.50 else "HIGH"
    (out / "endpoint_input_audit.md").write_text(
        "# CipherSpectrum Open-Detect Input Audit\n\n"
        f"- Fixed sample: seed **{SEED}**, {len(labels)} classes x up to 20 PCAPs = **{len(selected)}**; valid inputs **{len(valid)}**.\n"
        f"- Raw PCAP full class-domain visible: **{raw_domain_count}/{len(valid)} ({raw_domain_count/len(valid):.2%})**.\n"
        f"- Actual 32x32 input full class-domain visible: **{domain_count}/{len(valid)} ({domain_count/len(valid):.2%})**.\n"
        f"- Actual input parsed SNI visible: **{sni_count}/{len(valid)} ({sni_count/len(valid):.2%})**.\n"
        f"- Actual input class token visible: **{token_count}/{len(valid)} ({token_count/len(valid):.2%})**.\n"
        f"- Actual input domain/SNI/token union: **{union_count}/{len(valid)} ({union_rate:.2%})**.\n"
        f"- IP masking: **{masked_ip}/{total_ip}** checked packet fields masked.\n"
        f"- Port retention: **{visible_port}/{total_port}** checked transport headers preserve ports.\n"
        f"- CIPHERSPECTRUM_DIRECT_DOMAIN_LEAKAGE = **{level}**.\n\n"
        "Raw visibility and model-input visibility are reported separately. Port visibility is a potential endpoint shortcut, not proof that a classifier uses it.\n",
        encoding="utf-8",
    )
    return {"valid": len(valid), "domain": domain_count, "sni": sni_count, "token": token_count, "union": union_count, "level": level}


def main() -> None:
    provenance = run_provenance()
    leakage = run_input_audit()
    print(json.dumps({"provenance": provenance, "leakage": leakage}, indent=2), flush=True)


if __name__ == "__main__":
    main()
