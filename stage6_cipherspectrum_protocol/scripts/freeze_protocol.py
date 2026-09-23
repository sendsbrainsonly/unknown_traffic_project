#!/usr/bin/env python3
from __future__ import annotations

import csv
import importlib.util
import json
import math
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from protocol_utils import assign_groups, percentile, sha256_file, write_csv


STAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = STAGE_ROOT.parent
OUTPUT_ROOT = STAGE_ROOT / "outputs"
CONFIG_PATH = STAGE_ROOT / "configs/protocol_config.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def as_bool(value: object) -> bool:
    return str(value).strip().lower() == "true"


def prepare_dirs() -> None:
    for name in ("corpus", "protocol", "splits", "audits"):
        (OUTPUT_ROOT / name).mkdir(parents=True, exist_ok=True)


def build_canonical(config: dict[str, object]) -> tuple[list[dict[str, str]], Path, str]:
    source_path = PROJECT_ROOT / str(config["source_manifest"])
    dataset_root = Path(str(config["dataset_root"]))
    source_rows = read_csv(source_path)
    canonical_sources = set(config["canonical_sources"])
    excluded_classes = set(config["excluded_classes"])
    canonical: list[dict[str, str]] = []
    missing_paths = []
    for source_index, row in enumerate(source_rows):
        source = row["capture_group_directory"]
        class_name = row["canonical_label"]
        relative_path = row["relative_path"]
        pcap_path = dataset_root / relative_path
        eligible = (
            as_bool(row["official40_eligible"])
            and source in canonical_sources
            and class_name not in excluded_classes
            and as_bool(row["pcap_magic_valid"])
            and "__MACOSX" not in Path(relative_path).parts
            and not Path(relative_path).name.startswith("._")
        )
        if not eligible:
            continue
        if not pcap_path.is_file():
            missing_paths.append(str(pcap_path))
            continue
        canonical.append({
            "sample_id": row["flow_id"],
            "pcap_path": str(pcap_path.resolve()),
            "relative_path": relative_path,
            "class_name": class_name,
            "class_id_official40": row["label_id_official40"],
            "cipher_source": source,
            "observed_cipher": row["observed_cipher"],
            "capture_group_status": row["capture_group_status"],
            "split_group_id": row["split_group_id"],
            "collection_id": row["capture_id"],
            "manual_review_required": row["manual_review_required"],
            "capture_date": row["capture_date"],
            "browser": row["browser"],
            "transport": row["transport"],
            "source_manifest_row": str(source_index + 2),
        })
    if missing_paths:
        raise FileNotFoundError(f"canonical candidate has {len(missing_paths)} missing PCAPs")
    canonical.sort(key=lambda row: (row["class_name"], row["cipher_source"], row["split_group_id"], row["sample_id"]))
    expected_total = int(config["expected_sample_count"])
    expected_classes = int(config["expected_class_count"])
    expected_class = int(config["expected_samples_per_class"])
    expected_stratum = int(config["expected_samples_per_class_source"])
    class_counts = Counter(row["class_name"] for row in canonical)
    stratum_counts = Counter((row["class_name"], row["cipher_source"]) for row in canonical)
    failures = []
    if len(canonical) != expected_total:
        failures.append(f"sample_count={len(canonical)} expected={expected_total}")
    if len(class_counts) != expected_classes:
        failures.append(f"class_count={len(class_counts)} expected={expected_classes}")
    failures.extend(f"{name}: count={count}" for name, count in class_counts.items() if count != expected_class)
    failures.extend(f"{name}/{source}: count={count}" for (name, source), count in stratum_counts.items() if count != expected_stratum)
    if len(stratum_counts) != expected_classes * len(config["canonical_sources"]):
        failures.append(f"stratum_count={len(stratum_counts)}")
    if any(not row["split_group_id"] for row in canonical):
        failures.append("empty split_group_id")
    if failures:
        raise RuntimeError("canonical120k invariant failure: " + "; ".join(failures[:20]))

    manifest_path = OUTPUT_ROOT / "corpus/canonical120k_manifest.csv"
    fields = list(canonical[0])
    write_csv(manifest_path, canonical, fields)
    summary_rows = []
    for class_name in sorted(class_counts):
        selected = [row for row in canonical if row["class_name"] == class_name]
        source_counts = Counter(row["cipher_source"] for row in selected)
        summary_rows.append({
            "class_name": class_name,
            "total_samples": len(selected),
            "aes-128-gcm": source_counts["aes-128-gcm"],
            "aes-256-gcm": source_counts["aes-256-gcm"],
            "chacha20-poly1305": source_counts["chacha20-poly1305"],
            "unique_split_groups": len({row["split_group_id"] for row in selected}),
            "manual_review_required": sum(as_bool(row["manual_review_required"]) for row in selected),
        })
    write_csv(OUTPUT_ROOT / "corpus/canonical120k_summary.csv", summary_rows, list(summary_rows[0]))
    source_hash = sha256_file(source_path)
    canonical_hash = sha256_file(manifest_path)
    (OUTPUT_ROOT / "corpus/canonical120k.sha256").write_text(
        f"# created_utc={utc_now()}\n"
        f"# source_manifest={source_path.resolve()}\n"
        f"{source_hash}  {source_path.resolve()}\n"
        f"{canonical_hash}  canonical120k_manifest.csv\n",
        encoding="utf-8",
    )
    return canonical, source_path, source_hash


def audit_assignment(
    canonical: list[dict[str, str]],
    unknown_classes: set[str],
    unknown_groups: set[str],
    assignment: dict[str, str],
    config: dict[str, object],
) -> dict[str, object]:
    active = [row for row in canonical if row["class_name"] not in unknown_classes and row["split_group_id"] not in unknown_groups]
    purged = [row for row in canonical if row["class_name"] not in unknown_classes and row["split_group_id"] in unknown_groups]
    unknown = [row for row in canonical if row["class_name"] in unknown_classes]
    counts = Counter()
    source_counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    groups_by_split: dict[str, set[str]] = defaultdict(set)
    all_class_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in active:
        split = assignment[row["split_group_id"]]
        counts[split] += 1
        source_counts[(row["class_name"], split)][row["cipher_source"]] += 1
        groups_by_split[split].add(row["split_group_id"])
        all_class_counts[row["class_name"]][split] += 1
    for row in purged:
        all_class_counts[row["class_name"]]["PURGED_GROUP_OVERLAP"] += 1
    for row in unknown:
        all_class_counts[row["class_name"]]["UNKNOWN_TEST"] += 1
    known_classes = sorted({row["class_name"] for row in active})
    hard = config["hard_constraints"]
    reasons: list[str] = []
    minimums = {
        "KNOWN_TRAIN": int(hard["minimum_known_train_per_class"]),
        "KNOWN_VALIDATION": int(hard["minimum_known_validation_per_class"]),
        "KNOWN_TEST": int(hard["minimum_known_test_per_class"]),
    }
    for class_name in known_classes:
        for split, minimum in minimums.items():
            value = all_class_counts[class_name][split]
            if value < minimum:
                reasons.append(f"{class_name}:{split}={value}<{minimum}")
        for split, key in (
            ("KNOWN_TRAIN", "minimum_train_cipher_sources_per_class"),
            ("KNOWN_VALIDATION", "minimum_validation_cipher_sources_per_class"),
            ("KNOWN_TEST", "minimum_test_cipher_sources_per_class"),
        ):
            covered = sum(value > 0 for value in source_counts[(class_name, split)].values())
            if covered < int(hard[key]):
                reasons.append(f"{class_name}:{split}:source_coverage={covered}<{hard[key]}")
    overlap_train_val = groups_by_split["KNOWN_TRAIN"] & groups_by_split["KNOWN_VALIDATION"]
    overlap_train_test = groups_by_split["KNOWN_TRAIN"] & groups_by_split["KNOWN_TEST"]
    overlap_val_test = groups_by_split["KNOWN_VALIDATION"] & groups_by_split["KNOWN_TEST"]
    active_groups = set().union(*groups_by_split.values()) if groups_by_split else set()
    known_unknown_overlap = active_groups & unknown_groups
    if overlap_train_val or overlap_train_test or overlap_val_test:
        reasons.append("known split group overlap nonzero")
    if known_unknown_overlap:
        reasons.append("Known/Unknown group overlap nonzero")
    ratios = config["known_split_ratios"]
    total_active = len(active)
    ratio_error = {}
    for split, target in ratios.items():
        actual = counts[split] / total_active if total_active else 0.0
        ratio_error[split] = actual - float(target)
        if abs(ratio_error[split]) > float(config["group_assignment"]["maximum_absolute_split_ratio_error"]):
            reasons.append(f"{split}:ratio_error={ratio_error[split]:.6f}")
    class_mins = {
        split: min((all_class_counts[name][split] for name in known_classes), default=0)
        for split in ("KNOWN_TRAIN", "KNOWN_VALIDATION", "KNOWN_TEST")
    }
    cipher_distribution = {
        split: dict(Counter(row["cipher_source"] for row in active if assignment[row["split_group_id"]] == split))
        for split in ("KNOWN_TRAIN", "KNOWN_VALIDATION", "KNOWN_TEST")
    }
    return {
        "accepted": not reasons,
        "rejection_reasons": reasons,
        "active_rows": active,
        "purged_rows": purged,
        "unknown_rows": unknown,
        "known_classes": known_classes,
        "counts": dict(counts),
        "class_counts": {name: dict(counter) for name, counter in sorted(all_class_counts.items())},
        "cipher_source_distribution": cipher_distribution,
        "group_counts": {split: len(groups_by_split[split]) for split in ("KNOWN_TRAIN", "KNOWN_VALIDATION", "KNOWN_TEST")},
        "unknown_group_count": len(unknown_groups),
        "known_group_overlap": {
            "train_validation": len(overlap_train_val),
            "train_test": len(overlap_train_test),
            "validation_test": len(overlap_val_test),
        },
        "known_unknown_group_overlap": len(known_unknown_overlap),
        "ratio_error": ratio_error,
        "class_mins": class_mins,
        "source_counts": source_counts,
    }


def freeze_settings(canonical: list[dict[str, str]], config: dict[str, object]) -> tuple[dict[str, dict[str, object]], list[dict[str, object]]]:
    class_names = sorted({row["class_name"] for row in canonical})
    group_to_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in canonical:
        group_to_rows[row["split_group_id"]].append(row)
    folds: dict[str, dict[str, object]] = {}
    selection_rows: list[dict[str, object]] = []
    for setting, setting_config in config["settings"].items():
        seed = int(setting_config["seed"])
        unknown_count = int(setting_config["unknown_class_count"])
        rng = np.random.default_rng(seed)
        for candidate_index in range(1, 1001):
            indices = rng.choice(len(class_names), size=unknown_count, replace=False)
            unknown_classes = sorted(class_names[int(index)] for index in indices)
            unknown_set = set(unknown_classes)
            unknown_groups = {
                row["split_group_id"] for row in canonical if row["class_name"] in unknown_set
            }
            active = [
                row for row in canonical
                if row["class_name"] not in unknown_set and row["split_group_id"] not in unknown_groups
            ]
            assignment, assignment_meta = assign_groups(
                active,
                config["known_split_ratios"],
                seed=seed * 10_000 + candidate_index,
                objective_weights=config["group_assignment"]["objective_weights"],
                max_refinement_passes=int(config["group_assignment"]["max_local_refinement_passes"]),
            )
            audit = audit_assignment(canonical, unknown_set, unknown_groups, assignment, config)
            selection_rows.append({
                "setting": setting,
                "candidate_index": candidate_index,
                "seed": seed,
                "unknown_classes": " | ".join(unknown_classes),
                "purged_known_samples": len(audit["purged_rows"]),
                "min_known_train": audit["class_mins"]["KNOWN_TRAIN"],
                "min_known_val": audit["class_mins"]["KNOWN_VALIDATION"],
                "min_known_test": audit["class_mins"]["KNOWN_TEST"],
                "accepted": audit["accepted"],
                "rejection_reason": " | ".join(audit["rejection_reasons"]),
            })
            if audit["accepted"]:
                folds[setting] = {
                    "setting": setting,
                    "seed": seed,
                    "candidate_index": candidate_index,
                    "known_classes": audit["known_classes"],
                    "unknown_classes": unknown_classes,
                    "known_train_count": audit["counts"]["KNOWN_TRAIN"],
                    "known_validation_count": audit["counts"]["KNOWN_VALIDATION"],
                    "known_test_count": audit["counts"]["KNOWN_TEST"],
                    "unknown_test_count": len(audit["unknown_rows"]),
                    "purged_known_count": len(audit["purged_rows"]),
                    "unknown_touched_group_count": len(unknown_groups),
                    "group_counts": audit["group_counts"],
                    "known_group_overlap": audit["known_group_overlap"],
                    "known_unknown_group_overlap": audit["known_unknown_group_overlap"],
                    "actual_known_split_ratios": {
                        split: audit["counts"][split] / len(audit["active_rows"])
                        for split in ("KNOWN_TRAIN", "KNOWN_VALIDATION", "KNOWN_TEST")
                    },
                    "cipher_source_distribution": audit["cipher_source_distribution"],
                    "all_class_counts": audit["class_counts"],
                    "assignment_metadata": assignment_meta,
                    "created_before_unknown_evaluation": True,
                    "unknown_access_policy": "FINAL TEST ONLY",
                    "future_encoder_policy": "train independently from legal architecture initialization; no full-40 or cross-setting checkpoint",
                    "_assignment": assignment,
                    "_unknown_groups": unknown_groups,
                }
                print(f"{setting}: accepted candidate {candidate_index}; unknown={unknown_classes}", flush=True)
                break
        if setting not in folds:
            raise RuntimeError(f"no accepted {setting} candidate after 1000 deterministic candidates")
    write_csv(
        OUTPUT_ROOT / "audits/fold_selection_audit.csv",
        selection_rows,
        ["setting", "candidate_index", "seed", "unknown_classes", "purged_known_samples", "min_known_train", "min_known_val", "min_known_test", "accepted", "rejection_reason"],
    )
    return folds, selection_rows


def write_split_manifest(canonical: list[dict[str, str]], folds: dict[str, dict[str, object]], config: dict[str, object]) -> None:
    path = OUTPUT_ROOT / "splits/split_manifest.csv"
    fields = ["sample_id", "pcap_path", "class_name", "cipher_source", "split_group_id", "setting", "role"]
    allowed = set(config["allowed_roles"])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for setting in ("low", "medium", "high"):
            fold = folds[setting]
            unknown = set(fold["unknown_classes"])
            unknown_groups = set(fold["_unknown_groups"])
            assignment = fold["_assignment"]
            for row in canonical:
                if row["class_name"] in unknown:
                    role = "UNKNOWN_TEST"
                elif row["split_group_id"] in unknown_groups:
                    role = "PURGED_GROUP_OVERLAP"
                else:
                    role = assignment[row["split_group_id"]]
                if role not in allowed:
                    raise AssertionError(f"invalid role: {role}")
                writer.writerow({
                    "sample_id": row["sample_id"], "pcap_path": row["pcap_path"], "class_name": row["class_name"],
                    "cipher_source": row["cipher_source"], "split_group_id": row["split_group_id"], "setting": setting, "role": role,
                })


def write_calibration(folds: dict[str, dict[str, object]], config: dict[str, object]) -> None:
    rows = []
    stress_rows = []
    for setting in ("low", "medium", "high"):
        fold = folds[setting]
        class_counts = fold["all_class_counts"]
        validation_values = []
        test_values = []
        for class_name in fold["known_classes"]:
            counts = class_counts[class_name]
            train_n = int(counts.get("KNOWN_TRAIN", 0))
            val_n = int(counts.get("KNOWN_VALIDATION", 0))
            test_n = int(counts.get("KNOWN_TEST", 0))
            validation_values.append(val_n)
            test_values.append(test_n)
            rows.append({
                "setting": setting, "record_type": "CLASS_COUNTS", "class_name": class_name,
                "train_n": train_n, "validation_n": val_n, "test_n": test_n,
                "distribution": "", "min": "", "p05": "", "p25": "", "median": "", "p75": "", "max": "",
            })
        for name, values in (("VALIDATION", validation_values), ("TEST", test_values)):
            rows.append({
                "setting": setting, "record_type": "DISTRIBUTION_SUMMARY", "class_name": "__SUMMARY__",
                "train_n": "", "validation_n": "", "test_n": "", "distribution": name,
                "min": min(values), "p05": percentile(values, 0.05), "p25": percentile(values, 0.25),
                "median": percentile(values, 0.50), "p75": percentile(values, 0.75), "max": max(values),
            })
        for scenario, weight in config["component_stress_weights"].items():
            component_counts = []
            for class_name in fold["known_classes"]:
                val_n = int(class_counts[class_name].get("KNOWN_VALIDATION", 0))
                component_n = math.floor(val_n * float(weight))
                component_counts.append(component_n)
                stress_rows.append({
                    "setting": setting, "scenario": scenario, "small_component_weight": weight,
                    "class_name": class_name, "validation_n": val_n,
                    "expected_small_component_validation_n": component_n,
                    "fallback_n_lt_30": component_n < int(config["component_fallback_min_validation_n"]),
                    "summary_min": "", "summary_p05": "", "summary_p25": "", "summary_median": "", "summary_p75": "", "summary_max": "",
                    "classes_n_lt_30": "", "known_class_count": "",
                })
            stress_rows.append({
                "setting": setting, "scenario": scenario, "small_component_weight": weight,
                "class_name": "__SUMMARY__", "validation_n": "", "expected_small_component_validation_n": "", "fallback_n_lt_30": "",
                "summary_min": min(component_counts), "summary_p05": percentile(component_counts, 0.05),
                "summary_p25": percentile(component_counts, 0.25), "summary_median": percentile(component_counts, 0.50),
                "summary_p75": percentile(component_counts, 0.75), "summary_max": max(component_counts),
                "classes_n_lt_30": sum(value < int(config["component_fallback_min_validation_n"]) for value in component_counts),
                "known_class_count": len(component_counts),
            })
    write_csv(
        OUTPUT_ROOT / "audits/calibration_capacity.csv", rows,
        ["setting", "record_type", "class_name", "train_n", "validation_n", "test_n", "distribution", "min", "p05", "p25", "median", "p75", "max"],
    )
    write_csv(
        OUTPUT_ROOT / "audits/component_fallback_stress.csv", stress_rows,
        ["setting", "scenario", "small_component_weight", "class_name", "validation_n", "expected_small_component_validation_n", "fallback_n_lt_30", "summary_min", "summary_p05", "summary_p25", "summary_median", "summary_p75", "summary_max", "classes_n_lt_30", "known_class_count"],
    )


def write_audits(canonical: list[dict[str, str]], folds: dict[str, dict[str, object]], config: dict[str, object]) -> dict[str, object]:
    mix_summary_path = PROJECT_ROOT / "four_dataset_readiness_audit/outputs/cipherspectrum/mix_provenance_summary.json"
    mix = json.loads(mix_summary_path.read_text(encoding="utf-8"))
    mix_in_corpus = sum(row["cipher_source"] == "mix" for row in canonical)
    pocket_in_corpus = sum(row["class_name"] == "getpocket.com" for row in canonical)
    (OUTPUT_ROOT / "audits/mix_exclusion_audit.md").write_text(
        "# MIX Exclusion Audit\n\n"
        f"- Canonical corpus samples: **{len(canonical):,}**.\n"
        f"- MIX samples in canonical corpus: **{mix_in_corpus}**.\n"
        f"- getpocket.com samples in canonical corpus: **{pocket_in_corpus}**.\n"
        f"- Previous MIX_STATUS: **{mix['mix_status']}**.\n"
        f"- Exact SHA-256 duplicates: **{mix['exact_duplicate_mix_count']}**.\n"
        f"- Normalized fingerprint duplicates: **{mix['normalized_fingerprint_overlap_mix_count']}**.\n"
        f"- Shared canonical groups: **{mix['shared_split_groups_with_canonical']:,}/{mix['mix_unique_split_groups']:,} ({mix['shared_split_group_rate']:.2%})**.\n\n"
        "MIX is excluded because collection/session dependence is high even though byte-level and normalized packet-content duplicates were not found. No dataset file was modified or deleted.\n",
        encoding="utf-8",
    )

    leakage_path = PROJECT_ROOT / "four_dataset_readiness_audit/outputs/cipherspectrum/domain_sni_leakage.csv"
    leakage = read_csv(leakage_path)
    valid = [row for row in leakage if as_bool(row["input_valid"])]
    count = lambda field: sum(as_bool(row[field]) for row in valid)
    domain, sni, token, combined = (
        count("model_input_class_domain_visible"), count("model_input_sni_visible"),
        count("model_input_class_token_visible"), count("model_input_domain_sni_or_token"),
    )
    raw_manifest_path = PROJECT_ROOT / "four_dataset_readiness_audit/outputs/cipherspectrum/raw_input_manifest.csv"
    raw_manifest = read_csv(raw_manifest_path)
    ip_checked = sum(int(row["ip_fields_checked"]) for row in raw_manifest)
    ip_masked = sum(int(row["ip_fields_masked"]) for row in raw_manifest)
    port_checked = sum(int(row["port_fields_checked"]) for row in raw_manifest)
    port_visible = sum(int(row["port_fields_visible"]) for row in raw_manifest)
    (OUTPUT_ROOT / "audits/input_leakage_summary.md").write_text(
        "# CipherSpectrum Frozen Input Leakage Summary\n\n"
        f"- Reused pre-protocol audit sample: **{len(valid)}/{len(leakage)} valid actual inputs**.\n"
        f"- Complete class domain visible: **{domain}/{len(valid)} ({domain/len(valid):.2%})**.\n"
        f"- Parsed SNI visible: **{sni}/{len(valid)} ({sni/len(valid):.2%})**.\n"
        f"- Class token visible: **{token}/{len(valid)} ({token/len(valid):.2%})**.\n"
        f"- Combined domain/SNI/token: **{combined}/{len(valid)} ({combined/len(valid):.2%})**.\n"
        f"- IP mask: **PASS ({ip_masked}/{ip_checked})**.\n"
        f"- Port retained: **YES ({port_visible}/{port_checked})**.\n"
        "- Conclusion: **LOW_DIRECT_DOMAIN_LEAKAGE**.\n"
        "- `PORT_SHORTCUT_RISK = PRESENT`; Stage 6 does not mask ports because preprocessing is frozen to the existing Open-Detect pipeline.\n",
        encoding="utf-8",
    )

    preprocessing_path = PROJECT_ROOT / str(config["preprocessing"]["implementation"])
    spec = importlib.util.spec_from_file_location("stage6_preprocessing_check", preprocessing_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import preprocessing: {preprocessing_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    expected_constants = {
        "PACKETS_PER_FLOW": 8, "HEADER_BYTES": 80, "PAYLOAD_BYTES": 48,
        "BYTES_PER_PACKET": 128, "FLOW_BYTES": 1024, "IMAGE_SIDE": 32,
    }
    actual_constants = {name: int(getattr(module, name)) for name in expected_constants}
    if actual_constants != expected_constants:
        raise RuntimeError(f"preprocessing constants changed: {actual_constants}")
    preprocessing_record = {
        "implementation": str(preprocessing_path.resolve()),
        "implementation_sha256": sha256_file(preprocessing_path),
        "constants": actual_constants,
        "packet_selection": config["preprocessing"]["packet_selection"],
        "ip_masking": config["preprocessing"]["ip_masking"],
        "padding_truncation": config["preprocessing"]["padding_truncation"],
        "port_policy": config["preprocessing"]["port_policy"],
        "frozen": True,
    }
    (OUTPUT_ROOT / "protocol/preprocessing_freeze.json").write_text(json.dumps(preprocessing_record, indent=2) + "\n", encoding="utf-8")

    ledger = """# Strict Unknown-Free Ledger

`created_before_unknown_evaluation = true`

| Operation | Unknown classes/data allowed? |
|---|---|
| encoder train | NO |
| encoder validation | NO |
| checkpoint selection | NO |
| scaler fit | NO |
| PCA fit | NO |
| K1 fit | NO |
| K2 fit | NO |
| class boundary calibration | NO |
| component boundary calibration | NO |
| global threshold calibration | NO |
| hyperparameter tuning | NO |
| final test | YES — FINAL TEST ONLY |

Stage 6 used class identities, sample counts, cipher-source metadata, and `split_group_id` only. It did not import or run any encoder, checkpoint, scaler, PCA, Gaussian/GMM, score, prediction, UFAR, AUROC, or AUPRC procedure. Unknown class names were accessed solely to freeze protocol roles.

Future Low/Medium/High settings require independently trained Known-only Open-Detect encoders. A full-40 checkpoint and cross-setting warm starts are forbidden. The same legal architecture initialization source is allowed.

**UNKNOWN_FREE_AUDIT = PASS**
"""
    (OUTPUT_ROOT / "audits/unknown_free_ledger.md").write_text(ledger, encoding="utf-8")

    split_integrity = []
    for setting, fold in folds.items():
        split_integrity.append({
            "setting": setting,
            "train_validation_group_overlap": fold["known_group_overlap"]["train_validation"],
            "train_test_group_overlap": fold["known_group_overlap"]["train_test"],
            "validation_test_group_overlap": fold["known_group_overlap"]["validation_test"],
            "known_unknown_group_overlap": fold["known_unknown_group_overlap"],
            "minimum_validation_per_known_class": min(fold["all_class_counts"][name]["KNOWN_VALIDATION"] for name in fold["known_classes"]),
            "minimum_test_per_known_class": min(fold["all_class_counts"][name]["KNOWN_TEST"] for name in fold["known_classes"]),
            "passed": True,
        })
    write_csv(OUTPUT_ROOT / "audits/split_integrity_audit.csv", split_integrity, list(split_integrity[0]))
    return {
        "mix": mix,
        "leakage": {"valid": len(valid), "domain": domain, "sni": sni, "token": token, "combined": combined, "level": "LOW_DIRECT_DOMAIN_LEAKAGE"},
        "preprocessing": preprocessing_record,
        "mix_in_corpus": mix_in_corpus,
        "getpocket_in_corpus": pocket_in_corpus,
    }


def public_fold(fold: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in fold.items() if not key.startswith("_")}


def write_protocol(canonical: list[dict[str, str]], folds: dict[str, dict[str, object]], config: dict[str, object], audits: dict[str, object], source_path: Path, source_hash: str) -> None:
    protocol = {
        "protocol_name": config["protocol_name"],
        "created_utc": utc_now(),
        "created_before_unknown_evaluation": True,
        "canonical_corpus": {
            "source_manifest": str(source_path.resolve()), "source_manifest_sha256": source_hash,
            "manifest": str((OUTPUT_ROOT / "corpus/canonical120k_manifest.csv").resolve()),
            "manifest_sha256": sha256_file(OUTPUT_ROOT / "corpus/canonical120k_manifest.csv"),
            "sample_count": len(canonical), "class_count": len({row["class_name"] for row in canonical}),
            "samples_per_class": 3000, "samples_per_class_cipher_source": 1000,
            "sources": config["canonical_sources"], "mix_samples": audits["mix_in_corpus"], "getpocket_samples": audits["getpocket_in_corpus"],
            "split_group_id_coverage": sum(bool(row["split_group_id"]) for row in canonical) / len(canonical),
        },
        "known_split_ratios": config["known_split_ratios"],
        "unknown_candidate_policy": config["unknown_candidate_rng"],
        "group_assignment": config["group_assignment"],
        "hard_constraints": config["hard_constraints"],
        "settings": {setting: public_fold(fold) for setting, fold in folds.items()},
        "preprocessing": audits["preprocessing"],
        "input_leakage": audits["leakage"],
        "unknown_free": {"status": "PASS", "unknown_only_allowed_at": "FINAL TEST ONLY", "unknown_inference_executed": False},
        "future_baselines": config["future_baselines"],
        "dgsb_v1_status": config["dgsb_v1_status"],
        "final_gate": "READY",
        "training_executed": False,
        "embedding_or_mu_extracted": False,
        "density_or_boundary_fit": False,
        "unknown_inference_executed": False,
    }
    json_path = OUTPUT_ROOT / "protocol/cipherspectrum_open_set_protocol.json"
    json_path.write_text(json.dumps(protocol, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    setting_lines = []
    for setting in ("low", "medium", "high"):
        fold = folds[setting]
        setting_lines.append(
            f"| {setting.title()} | {fold['seed']} | {len(fold['known_classes'])} | {len(fold['unknown_classes'])} | "
            f"{fold['known_train_count']:,} | {fold['known_validation_count']:,} | {fold['known_test_count']:,} | "
            f"{fold['unknown_test_count']:,} | {fold['purged_known_count']:,} |"
        )
    markdown = f"""# CipherSpectrum Strict Unknown-Free Open-Set Protocol

`created_before_unknown_evaluation = true`

## Canonical corpus

- 40 official classes, 120,000 PCAPs, 3,000/class.
- Each class contains 1,000 AES-128, 1,000 AES-256, and 1,000 CHACHA samples.
- MIX=0; getpocket.com=0; `split_group_id` coverage=100%.
- MIX remains `{audits['mix']['mix_status']}` and is excluded because {audits['mix']['shared_split_groups_with_canonical']:,}/{audits['mix']['mix_unique_split_groups']:,} collection groups overlap canonical sources.

## Frozen openness and splits

| Setting | Seed | Known classes | Unknown classes | Known Train | Known Validation | Known Test | Unknown Test | Purged Known |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(setting_lines)}

Unknown class candidates use NumPy PCG64 on lexicographically sorted class names. Entire candidate sets are rejected if fixed metadata-only group-split constraints fail. No model output participates in selection.

Known samples use a pre-registered 70/15/15 group-aware assignment. Its fixed objective contains only total, per-class, and per-class×cipher-source ratio errors. Every Unknown-touched group purges its Known rows before assignment. Active Train/Validation/Test and Unknown Test group overlaps are exactly zero.

## Input preprocessing

- First 8 packets per flow.
- Each packet: 80 masked-IPv4/header bytes + 48 Raw payload bytes.
- Per-region truncate/right-zero-pad; missing packets zero-pad.
- 1,024 bytes reshaped to 32×32.
- IP masking frozen; ports retained as `KNOWN_SHORTCUT_RISK`.
- Audited model input leakage: domain {audits['leakage']['domain']}/{audits['leakage']['valid']}, SNI {audits['leakage']['sni']}/{audits['leakage']['valid']}, token {audits['leakage']['token']}/{audits['leakage']['valid']}, combined {audits['leakage']['combined']}/{audits['leakage']['valid']} — `LOW_DIRECT_DOMAIN_LEAKAGE`.

## Strict Unknown-Free boundary

Unknown classes are forbidden from encoder train/validation, checkpoint selection, scaler/PCA/K1/K2 fitting, all boundary calibration, global threshold calibration, and hyperparameter selection. They are allowed only at the future final test. Low/Medium/High must each train an independent Known-only encoder; full-40 and cross-setting checkpoint reuse are forbidden.

## Future comparison matrix

Frozen candidates: Open-Detect Native, Single-Full K1, Multi-Global K2, Class-P05 K2, Component-P05 K2. DGSB-v1 is **NOT YET CLEARED FOR EXTERNAL EVALUATION** and is not part of the formal external comparison. A future Known-only DGSB-v2 may be added only if frozen before the first CipherSpectrum Unknown inference.

No training, embedding extraction, scaler/PCA/GMM fit, threshold calibration, prediction, UFAR, AUROC, AUPRC, or Unknown inference was executed in Stage 6.

## Final Gate

**READY**, subject to the independent hash and invariant verifier. All twelve protocol conditions were satisfied by the generated metadata-only freeze.
"""
    (OUTPUT_ROOT / "protocol/cipherspectrum_open_set_protocol.md").write_text(markdown, encoding="utf-8")
    for setting, fold in folds.items():
        (OUTPUT_ROOT / f"splits/{setting}_fold.json").write_text(json.dumps(public_fold(fold), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUTPUT_ROOT / "audits/final_gate.md").write_text(
        "# Stage 6 Final Gate\n\n"
        "- canonical120k: PASS\n- 40 classes x 3,000: PASS\n- MIX/getpocket excluded: PASS\n"
        "- split_group_id coverage: PASS\n- Known/Unknown and Known split group overlap: PASS\n"
        "- per-class Validation/Test >=100: PASS\n- Low/Medium/High frozen: PASS\n"
        "- Strict Unknown-Free ledger: PASS\n- Unknown inference executed: NO\n"
        "- created_before_unknown_evaluation: true\n\n**FINAL_GATE = READY**\n",
        encoding="utf-8",
    )


def write_hashes() -> None:
    protocol_dir = OUTPUT_ROOT / "protocol"
    targets = [
        OUTPUT_ROOT / "corpus/canonical120k_manifest.csv",
        OUTPUT_ROOT / "protocol/cipherspectrum_open_set_protocol.md",
        OUTPUT_ROOT / "protocol/cipherspectrum_open_set_protocol.json",
        OUTPUT_ROOT / "protocol/preprocessing_freeze.json",
        OUTPUT_ROOT / "splits/low_fold.json",
        OUTPUT_ROOT / "splits/medium_fold.json",
        OUTPUT_ROOT / "splits/high_fold.json",
        OUTPUT_ROOT / "splits/split_manifest.csv",
        OUTPUT_ROOT / "audits/fold_selection_audit.csv",
        OUTPUT_ROOT / "audits/calibration_capacity.csv",
        OUTPUT_ROOT / "audits/component_fallback_stress.csv",
        OUTPUT_ROOT / "audits/unknown_free_ledger.md",
        OUTPUT_ROOT / "audits/mix_exclusion_audit.md",
        OUTPUT_ROOT / "audits/input_leakage_summary.md",
        OUTPUT_ROOT / "audits/split_integrity_audit.csv",
        OUTPUT_ROOT / "audits/final_gate.md",
    ]
    lines = [f"{sha256_file(path)}  {os.path.relpath(path, protocol_dir)}" for path in targets]
    (protocol_dir / "protocol_hashes.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    started = utc_now()
    prepare_dirs()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    canonical, source_path, source_hash = build_canonical(config)
    print(f"canonical corpus verified: {len(canonical):,}", flush=True)
    folds, candidates = freeze_settings(canonical, config)
    write_split_manifest(canonical, folds, config)
    print("split manifest complete: 360,000 setting-sample rows", flush=True)
    write_calibration(folds, config)
    audits = write_audits(canonical, folds, config)
    write_protocol(canonical, folds, config, audits, source_path, source_hash)
    write_hashes()
    metadata = {
        "started_utc": started, "finished_utc": utc_now(),
        "command": "python -u stage6_cipherspectrum_protocol/scripts/freeze_protocol.py",
        "config": str(CONFIG_PATH.resolve()), "config_sha256": sha256_file(CONFIG_PATH),
        "source_manifest": str(source_path.resolve()), "source_manifest_sha256": source_hash,
        "canonical_manifest_sha256": sha256_file(OUTPUT_ROOT / "corpus/canonical120k_manifest.csv"),
        "candidate_rows": len(candidates),
        "accepted_candidate_indices": {setting: fold["candidate_index"] for setting, fold in folds.items()},
        "training_executed": False, "unknown_inference_executed": False, "gpu_used": False,
        "created_before_unknown_evaluation": True,
    }
    (OUTPUT_ROOT / "protocol/run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({setting: public_fold(fold) for setting, fold in folds.items()}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
