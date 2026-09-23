#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from audit_utils import OUTPUT_ROOT, PROJECT_ROOT, directory_inventory, scan_classic_pcap, sha256_file, write_csv


AUDIT_ROOT = Path(__file__).resolve().parents[1]
CIPHER_ROOT = PROJECT_ROOT.parents[1] / "Dataset/ipherSpectrum(sok)/ipherSpectrum(sok)"
CIPHER_MANIFEST = PROJECT_ROOT / "cipherspectrum_labeling/outputs/cipherspectrum_label_manifest.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def bool_value(value: object) -> bool:
    return str(value).lower() == "true"


def complete_cipher_fingerprints() -> tuple[int, list[str]]:
    path = OUTPUT_ROOT / "cipherspectrum/all_pcap_fingerprints.csv"
    rows = read_csv(path)
    known = {row["relative_path"] for row in rows}
    actual = sorted(
        item for item in CIPHER_ROOT.rglob("*")
        if item.is_file() and item.suffix.lower() in {".pcap", ".pcapng"}
    )
    extras = [item for item in actual if str(item.relative_to(CIPHER_ROOT)) not in known]
    for item in extras:
        relative = str(item.relative_to(CIPHER_ROOT))
        scan = scan_classic_pcap(item)
        rows.append({
            "flow_id": "", "relative_path": relative, "source_group": relative.split("/", 1)[0],
            "class_name": "__MACOSX_METADATA", "official40_eligible": False, "split_group_id": "", **scan,
        })
    fields = ["flow_id", "relative_path", "absolute_path", "source_group", "class_name", "official40_eligible", "split_group_id", "size_bytes", "sha256", "normalized_fingerprint", "readable", "packet_count", "first_packet_timestamp", "last_packet_timestamp", "error"]
    write_csv(path, rows, fields)
    write_csv(
        OUTPUT_ROOT / "cipherspectrum/unmanifested_pcap.csv",
        [{"relative_path": str(item.relative_to(CIPHER_ROOT)), "classification": "MACOS_RESOURCE_FORK_NON_DATA", **scan_classic_pcap(item)} for item in extras],
        ["relative_path", "absolute_path", "classification", "size_bytes", "sha256", "readable", "packet_count", "error"],
    )
    return len(actual), [str(item.relative_to(CIPHER_ROOT)) for item in extras]


def rewrite_cipher_grouping() -> dict[str, dict[str, float | int]]:
    rows = read_csv(CIPHER_MANIFEST)
    scopes = {
        "official40": [row for row in rows if bool_value(row["official40_eligible"])],
        "all_local41": rows,
    }
    output = []
    summaries = {}
    for scope, selected in scopes.items():
        groups: dict[str, dict[str, object]] = {}
        for row in selected:
            group = groups.setdefault(row["split_group_id"], {"count": 0, "classes": set(), "sources": set()})
            group["count"] = int(group["count"]) + 1
            group["classes"].add(row["canonical_label"])
            group["sources"].add(row["capture_group_directory"])
        sizes = sorted(int(group["count"]) for group in groups.values())

        def pct(q: float) -> float:
            position = (len(sizes) - 1) * q
            low, high = int(position), min(int(position) + 1, len(sizes) - 1)
            weight = position - low
            return sizes[low] * (1 - weight) + sizes[high] * weight

        result = {
            "scope": scope, "flow_count": len(selected), "group_count": len(groups),
            "cross_class_group_count": sum(len(group["classes"]) > 1 for group in groups.values()),
            "cross_source_group_count": sum(len(group["sources"]) > 1 for group in groups.values()),
            "group_size_min": min(sizes), "group_size_p25": pct(0.25), "group_size_median": pct(0.50),
            "group_size_p75": pct(0.75), "group_size_p95": pct(0.95), "group_size_max": max(sizes),
            "split_group_id_available": True, "required_split_rule": "all rows sharing split_group_id stay in one partition",
        }
        output.append(result)
        summaries[scope] = result
    write_csv(OUTPUT_ROOT / "cipherspectrum/grouping_audit.csv", output, list(output[0]))
    return summaries


def load_summary() -> dict[str, object]:
    path_rows = {row["dataset"]: row for row in read_csv(OUTPUT_ROOT / "summary/dataset_path_resolution.csv")}
    ustc_classes = read_csv(OUTPUT_ROOT / "ustc/class_inventory.csv")
    mix = json.loads((OUTPUT_ROOT / "cipherspectrum/mix_provenance_summary.json").read_text(encoding="utf-8"))
    cipher_leak = read_csv(OUTPUT_ROOT / "cipherspectrum/domain_sni_leakage.csv")
    cipher_valid = [row for row in cipher_leak if bool_value(row["input_valid"])]
    leak_counts = {
        "domain": sum(bool_value(row["model_input_class_domain_visible"]) for row in cipher_valid),
        "sni": sum(bool_value(row["model_input_sni_visible"]) for row in cipher_valid),
        "token": sum(bool_value(row["model_input_class_token_visible"]) for row in cipher_valid),
        "union": sum(bool_value(row["model_input_domain_sni_or_token"]) for row in cipher_valid),
        "valid": len(cipher_valid),
    }
    leak_rate = leak_counts["union"] / leak_counts["valid"] if leak_counts["valid"] else 1.0
    cipher_leak_level = "LOW" if leak_rate < 0.10 else "MODERATE" if leak_rate < 0.50 else "HIGH"
    cic_inventory = read_csv(OUTPUT_ROOT / "cicids2017/class_inventory.csv")
    cic_eligibility = read_csv(OUTPUT_ROOT / "cicids2017/class_eligibility.csv")
    cic_conf = json.loads((OUTPUT_ROOT / "cicids2017/confounding_summary.json").read_text(encoding="utf-8"))
    cic_input = read_csv(OUTPUT_ROOT / "cicids2017/raw_input_manifest.csv")
    endpoint = read_csv(OUTPUT_ROOT / "cicids2017/endpoint_port_shortcut.csv")
    endpoint_by_label: dict[str, dict[str, float]] = defaultdict(dict)
    for row in endpoint:
        endpoint_by_label[row["label"]][row["feature"]] = float(row["top1_share"])
    endpoint_rank = sorted(
        (
            {"label": label, "dst_ip_top1": values.get("dst_ip", 0.0), "dst_port_top1": values.get("dst_port", 0.0), "combined_min": min(values.get("dst_ip", 0.0), values.get("dst_port", 0.0))}
            for label, values in endpoint_by_label.items()
        ),
        key=lambda row: (-row["combined_min"], row["label"]),
    )
    return {
        "paths": path_rows, "ustc_classes": ustc_classes, "mix": mix, "cipher_leak_counts": leak_counts,
        "cipher_leak_level": cipher_leak_level, "cic_inventory": cic_inventory,
        "cic_eligibility": cic_eligibility, "cic_conf": cic_conf, "cic_input": cic_input,
        "endpoint_rank": endpoint_rank,
    }


def write_matrix(summary: dict[str, object], grouping: dict[str, dict[str, float | int]]) -> list[dict[str, object]]:
    paths = summary["paths"]
    mix = summary["mix"]
    cipher_level = summary["cipher_leak_level"]
    cic_primary = summary["cic_conf"]["primary_eligible"]
    cipher_ready = "READY" if cipher_level == "LOW" and mix["mix_status"] != "UNCERTAIN" else "CONDITIONALLY_READY" if cipher_level != "HIGH" else "NOT_READY"
    cic_ready = "CONDITIONALLY_READY" if summary["cic_conf"]["group_aware_split_feasible"] and len(cic_primary) >= 2 else "NOT_READY"
    rows = [
        {
            "dataset": "USTC-TFC2016", "role": "Development / Mechanism", "raw_pcap_available": "YES",
            "label_quality": "20-class logical mapping verified; SMB-1/2 merged", "class_count": 20, "usable_class_count": 20,
            "sample_count": 489101, "group_id_available": "NO authoritative session group", "group_split_ready": "EXISTING FLOW SPLIT FOUND",
            "direct_domain_leakage": "NOT RE-AUDITED", "endpoint_port_risk": "NOT RE-AUDITED",
            "calibration_capacity": "existing development split available", "component_p05_feasibility": "class-dependent; not reassessed",
            "unknown_split_status": "existing frozen development folds", "independent_validation_status": "DEVELOPMENT_ONLY",
            "blocking_issue": "Unknown test informed development", "final_readiness": "READY",
        },
        {
            "dataset": "CipherSpectrum", "role": "Primary Independent External Validation", "raw_pcap_available": "YES",
            "label_quality": "official40 directory-label manifest; getpocket excluded", "class_count": 40, "usable_class_count": 40,
            "sample_count": 120000, "group_id_available": "YES: split_group_id", "group_split_ready": "FEASIBLE; not frozen",
            "direct_domain_leakage": cipher_level, "endpoint_port_risk": "PORT_VISIBLE; group control required",
            "calibration_capacity": "300/class (80/10/10) or 450/class (70/15/15)",
            "component_p05_feasibility": "15 or 22/class; below n=30 fallback threshold",
            "unknown_split_status": "NOT FROZEN", "independent_validation_status": "INDEPENDENT_CANDIDATE",
            "blocking_issue": f"MIX={mix['mix_status']}; exclude MIX and freeze group-aware split",
            "final_readiness": cipher_ready,
        },
        {
            "dataset": "CSTNET-TLS1.3", "role": "Secondary Sparse-Calibration Stress Test", "raw_pcap_available": "YES",
            "label_quality": "directory labels; 119 eligible of 120", "class_count": 120, "usable_class_count": 119,
            "sample_count": 46372, "group_id_available": "YES: frozen one-minute proxy", "group_split_ready": "YES: hashes PASS",
            "direct_domain_leakage": "LOW (0/600)", "endpoint_port_risk": "PORT_VISIBLE; endpoint novelty unresolved",
            "calibration_capacity": "sparse: validation medians 44/42/39", "component_p05_feasibility": "fallback expected for all classes",
            "unknown_split_status": "FROZEN Low/Medium/High", "independent_validation_status": "INDEPENDENT_CANDIDATE",
            "blocking_issue": "component calibration sparse; protocol v2 needed before method run",
            "final_readiness": "CONDITIONALLY_READY",
        },
        {
            "dataset": "CIC-IDS-2017", "role": "Independent NIDS / Unknown-Attack Validation", "raw_pcap_available": "YES",
            "label_quality": "strict official flow-CSV matches; 3 schedule sanity-risk labels", "class_count": 15,
            "usable_class_count": len(cic_primary), "sample_count": summary["cic_conf"]["strict_matched_flows"],
            "group_id_available": "candidate five-minute capture block", "group_split_ready": "FEASIBLE; not frozen",
            "direct_domain_leakage": "NOT A DOMAIN-LABEL DATASET", "endpoint_port_risk": "HIGH for multiple attack classes",
            "calibration_capacity": "large overall; only 3 PRIMARY_ELIGIBLE", "component_p05_feasibility": "not evaluated before class policy freeze",
            "unknown_split_status": "NOT FROZEN", "independent_validation_status": "INDEPENDENT_BUT_CONFOUNDED",
            "blocking_issue": "one-day/PCAP attack binding, endpoint shortcuts, class policy and Unknown split unfrozen",
            "final_readiness": cic_ready,
        },
    ]
    write_csv(OUTPUT_ROOT / "summary/four_dataset_readiness_matrix.csv", rows, list(rows[0]))
    return rows


def make_readme(summary: dict[str, object], matrix: list[dict[str, object]], all_cipher_pcaps: int, extra_paths: list[str], grouping: dict[str, dict[str, float | int]]) -> None:
    paths = summary["paths"]
    mix = summary["mix"]
    leak = summary["cipher_leak_counts"]
    cic_inventory = summary["cic_inventory"]
    eligibility = summary["cic_eligibility"]
    primary = [row["label"] for row in eligibility if row["eligibility"] == "PRIMARY_ELIGIBLE"]
    excluded = [f"{row['label']} ({row['eligibility']})" for row in eligibility if row["eligibility"] != "PRIMARY_ELIGIBLE"]
    matrix_lines = ["| Dataset | Role | Samples | Usable classes | Final readiness |", "|---|---|---:|---:|---|"]
    matrix_lines.extend(f"| {row['dataset']} | {row['role']} | {int(row['sample_count']):,} | {row['usable_class_count']} | {row['final_readiness']} |" for row in matrix)
    class_lines = ["| CICIDS label | Strict flows | Eligibility |", "|---|---:|---|"]
    by_elig = {row["label"]: row["eligibility"] for row in eligibility}
    class_lines.extend(f"| {row['label']} | {int(row['strict_flow_count']):,} | {by_elig[row['label']]} |" for row in cic_inventory)
    endpoint_lines = ["| Label | dst IP top-1 | dst port top-1 |", "|---|---:|---:|"]
    endpoint_lines.extend(f"| {row['label']} | {row['dst_ip_top1']:.2%} | {row['dst_port_top1']:.2%} |" for row in summary["endpoint_rank"][:10])
    readme = f"""# Four-Dataset Final Data Readiness Audit

## Goal

Audit only local data completeness, label provenance, grouping, duplicate/provenance risk, representation leakage, calibration capacity, and experiment readiness. No encoder training, embedding extraction, density fitting, thresholding, Unknown inference, or formal metric evaluation was performed.

## Dataset Paths

- USTC-TFC2016: `{paths['USTC-TFC2016']['resolved_path']}`
- CipherSpectrum: `{paths['CipherSpectrum']['resolved_path']}`
- CSTNET-TLS1.3: `{paths['CSTNET-TLS1.3']['resolved_path']}`
- CIC-IDS-2017: `{paths['CIC-IDS-2017']['resolved_path']}`

## USTC

The existing logical map remains **20 classes / 489,101 flows**. SMB-1 and SMB-2 are merged into SMB; Weibo-1..4 are merged into Weibo. All 24 canonical extracted PCAPs are readable/non-empty. Existing train/validation/test manifests, Open-Detect reconstructed images, and Stage3/4/5 protocol/hash assets are present. Data readiness is **READY**, but independent-validation status is **DEVELOPMENT_ONLY**.

## CipherSpectrum

The existing official40 manifest has **160,200 PCAPs**, exactly 4,005 per class. The conservative primary candidate excludes MIX and has **120,000 PCAPs**, 3,000 per class. `getpocket.com` remains a **NON_PRIMARY_EXTRA_CLASS**. The physical tree has **{all_cipher_pcaps:,} PCAP-suffixed files**; the two non-manifest files are macOS resource forks: `{'; '.join(extra_paths)}`.

## CSTNET-TLS1.3

Frozen protocol SHA-256 verification is **PASS (10/10)**; the broader Stage 5.5 frozen record remains 26/26 PASS. The source has 120 classes and 46,372 readable per-flow PCAPs; 119 classes are eligible and `chia.net` is excluded.

## CIC-IDS-2017

Strict reliable official flow-CSV matching yields **{int(summary['cic_conf']['strict_matched_flows']):,} flows / {len(cic_inventory)} labels**. All attack labels occur in exactly one day and one PCAP. The sample-weighted class↔day and class↔PCAP NMI are both **{summary['cic_conf']['class_day_nmi']:.6f}**; this moderate aggregate number is suppressed by the dominant multi-day BENIGN class and does not negate the 100% per-attack day binding.

{chr(10).join(class_lines)}

## Duplicate/Provenance Risks

MIX_STATUS is **{mix['mix_status']}**. Exact SHA duplicates are **{mix['exact_duplicate_mix_count']:,}** and normalized packet-fingerprint overlaps are **{mix['normalized_fingerprint_overlap_mix_count']:,}** among 40,200 MIX PCAPs. MIX shares **{mix['shared_split_groups_with_canonical']:,}/{mix['mix_unique_split_groups']:,}** collection groups with canonical sources. Primary experiments should **{'exclude MIX' if mix['recommend_exclude_mix_from_primary'] else 'retain MIX only with the audited provenance rule'}**; no file was deleted.

## Leakage Risks

CipherSpectrum actual Open-Detect inputs: full domain **{leak['domain']}/{leak['valid']}**, parsed SNI **{leak['sni']}/{leak['valid']}**, class token **{leak['token']}/{leak['valid']}**, union **{leak['union']}/{leak['valid']}**; direct level **{summary['cipher_leak_level']}**. CSTNET remains 0/600 for direct domain/SNI/token, with IP masked and ports visible. CICIDS endpoint concentration is substantial:

{chr(10).join(endpoint_lines)}

## Calibration Capacity

CipherSpectrum canonical-three view gives 300 validation samples/class under 80/10/10 and 450/class under 70/15/15. A hypothetical K=2 95/5 small component receives only 15 or 22 samples/class, below the n=30 fallback boundary. CSTNET Known-validation medians are 44/42/39 for Low/Medium/High; K=2 fallback stress is severe, so its correct role is **SPARSE_CALIBRATION_STRESS_BENCHMARK**, not the primary component-boundary benchmark.

## Group Split Feasibility

CipherSpectrum `split_group_id` is available. In official40 there are **{grouping['official40']['group_count']:,} groups**, including **{grouping['official40']['cross_class_group_count']:,} cross-class and {grouping['official40']['cross_source_group_count']:,} cross-source groups**; every shared group must remain in one partition. CSTNET has a frozen group-aware protocol. CICIDS can use `(source_pcap, five-minute flow-start block)` and is count-feasible for the three current primary classes, but no Unknown policy or split was frozen.

## Four-Dataset Comparison

{chr(10).join(matrix_lines)}

## Final Readiness

- USTC: **READY** for development/mechanism only.
- CipherSpectrum: **{next(row['final_readiness'] for row in matrix if row['dataset']=='CipherSpectrum')}**, using the canonical-three 120k corpus and a future group-aware split.
- CSTNET: **CONDITIONALLY_READY** because component calibration is sparse.
- CIC-IDS-2017: **{next(row['final_readiness'] for row in matrix if row['dataset']=='CIC-IDS-2017')}**; only {len(primary)} classes currently pass the fixed primary gates.

## Recommended Experimental Roles

- USTC = Development / Mechanism: supported.
- CipherSpectrum = Primary Independent External Validation: supported only with MIX excluded and `split_group_id` enforced.
- CSTNET = Secondary Sparse-Calibration Stress Test: supported conditionally.
- CIC-IDS-2017 = Independent NIDS / Unknown-Attack Validation: conditionally supported; not yet a clean 15-class benchmark.

## Blocking Issues

- CipherSpectrum: freeze the canonical-three group-aware split; do not allow `split_group_id` overlap.
- CSTNET: freeze a calibration-aware protocol revision before method execution; do not change the existing audit protocol silently.
- CICIDS: resolve/freeze the class-eligibility and Unknown policy; retain schedule-risk labels as risk, not silently relabel them; use grouped time blocks; report endpoint/capture confounding.
- Excluded from CICIDS primary under fixed gates: {', '.join(excluded)}.

## Next Step

The data-readiness audit is complete. The project is **not yet ready to launch one unified formal four-dataset experiment** because three downstream protocol freezes remain. The next authorized step would be protocol design/freeze only; this audit did not start training or evaluation.
"""
    (AUDIT_ROOT / "README.md").write_text(readme, encoding="utf-8")


def write_result_bundle(summary: dict[str, object], matrix: list[dict[str, object]]) -> None:
    now = datetime.now(timezone.utc).isoformat()
    results = "# Experiment results: four_dataset_final_readiness_audit_20260913_v1\n\n"
    results += "- Status: `complete`\n- Experiment type: `data-readiness-audit`\n- Claim scope: `diagnostic`\n"
    results += f"- Updated (UTC): `{now}`\n\n## Core results\n\n"
    for row in matrix:
        results += f"- {row['dataset']}: **{row['final_readiness']}**, samples={int(row['sample_count']):,}, usable_classes={row['usable_class_count']}.\n"
    results += f"- CipherSpectrum MIX: **{summary['mix']['mix_status']}**, exact={summary['mix']['exact_duplicate_mix_count']:,}, normalized={summary['mix']['normalized_fingerprint_overlap_mix_count']:,}.\n"
    results += f"- CICIDS strict matched: **{summary['cic_conf']['strict_matched_flows']:,}**, labels=15, primary={', '.join(summary['cic_conf']['primary_eligible'])}.\n"
    results += "\n## Boundaries and limitations\n\n- No model training, embedding extraction, GMM, threshold fitting, Unknown inference, or formal metric evaluation.\n- Directory and endpoint labels remain dataset-semantic risks; readiness does not imply paper-equivalent validity.\n- Original dataset files were read only.\n\n## Preserved evidence\n\n- `manifest.json`\n- Per-dataset CSV/Markdown files under `ustc/`, `cipherspectrum/`, `cstnet/`, and `cicids2017/`.\n- Unified matrix under `summary/`.\n"
    (OUTPUT_ROOT / "RESULTS.md").write_text(results, encoding="utf-8")

    (AUDIT_ROOT / "EXPERIMENT_INDEX.md").write_text(
        "# Experiment Index\n\n| Experiment ID | Type | Status | Evidence |\n|---|---|---|---|\n| four_dataset_final_readiness_audit_20260913_v1 | data-readiness-audit | complete | `outputs/RESULTS.md`, `outputs/manifest.json` |\n",
        encoding="utf-8",
    )

    artifact_rows = []
    for path in sorted(item for item in AUDIT_ROOT.rglob("*") if item.is_file() and item.name != "manifest.json" and ".tmux-task" not in item.parts and "__pycache__" not in item.parts):
        artifact_rows.append({"path": str(path.relative_to(AUDIT_ROOT)), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    source_inputs = [
        PROJECT_ROOT / "data/fig_graph/all_flows/label_map.json",
        PROJECT_ROOT / "data/splits/compatible_min1/split_summary.json",
        PROJECT_ROOT / "cipherspectrum_labeling/outputs/cipherspectrum_official40_manifest.csv",
        PROJECT_ROOT / "stage5_dual_gate_boundary/outputs/cstnet_protocol/split_hashes.sha256",
        PROJECT_ROOT.parents[1] / "Dataset/CIC-IDS-2017/CIC-IDS-2017(whole)/outputs/cicids2017_pcap_label_mapping/reports/sanity_check.csv",
    ]
    manifest = {
        "schema_version": 1, "experiment_id": "four_dataset_final_readiness_audit_20260913_v1",
        "created_at_utc": "2026-09-13T03:36:31Z", "updated_at_utc": now, "status": "complete",
        "experiment_type": "data-readiness-audit", "claim_scope": "diagnostic",
        "objective": "Audit four local datasets for integrity, labels, grouping, provenance, leakage, calibration capacity, and readiness without modeling.",
        "inputs": [{"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path), "read_only": True} for path in source_inputs],
        "execution": {
            "tmux_sessions": ["four_data_inventory_run_20260913", "four_data_cipher_full_20260913", "four_data_cicids_input_20260913", "four_data_finalize_20260913", "four_data_verify_20260913"],
            "commands": [
                "python -u four_dataset_readiness_audit/scripts/run_inventory_audits.py",
                "python -u four_dataset_readiness_audit/scripts/run_cipher_audits.py",
                "python -u four_dataset_readiness_audit/scripts/run_cicids_input_audit.py",
                "python -u four_dataset_readiness_audit/scripts/finalize_readiness_audit.py",
                "python -u four_dataset_readiness_audit/tests/verify_readiness_audit.py",
            ],
            "environment": "/home/birkenwald/data/SEU-WXY/conda_envs/2025-10-8-WXY-dgl_py310", "physical_gpu_ids": [],
        },
        "configuration": {
            "seed": 20260913, "cipher_input_max": "20 classes x 20 PCAPs", "cicids_input_max": "10 classes x 20 flows",
            "cicids_primary_thresholds": {"too_small": 100, "primary_min": 1000, "severe_endpoint_top1": 0.95, "min_five_minute_groups": 3},
        },
        "core_results": matrix,
        "artifacts": artifact_rows,
        "limitations": ["diagnostic data audit only", "no model or Unknown evaluation", "CICIDS attacks are day/PCAP bound"],
        "next_step": "freeze downstream group-aware protocols only after explicit authorization",
    }
    (OUTPUT_ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    all_cipher_pcaps, extra_paths = complete_cipher_fingerprints()
    grouping = rewrite_cipher_grouping()
    summary = load_summary()
    matrix = write_matrix(summary, grouping)
    make_readme(summary, matrix, all_cipher_pcaps, extra_paths, grouping)
    final = {
        "all_cipher_pcap_suffix_files": all_cipher_pcaps, "cipher_unmanifested_resource_forks": extra_paths,
        "matrix": matrix, "mix": summary["mix"], "cipher_leak_counts": summary["cipher_leak_counts"],
        "cipher_leak_level": summary["cipher_leak_level"], "cicids": summary["cic_conf"],
        "cicids_input_valid": sum(bool_value(row["input_valid"]) for row in summary["cic_input"]),
        "cicids_input_total": len(summary["cic_input"]), "cicids_endpoint_rank": summary["endpoint_rank"],
        "cicids_eligibility": summary["cic_eligibility"],
    }
    (OUTPUT_ROOT / "summary/audit_summary.json").write_text(json.dumps(final, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_result_bundle(summary, matrix)
    print(json.dumps(final, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
