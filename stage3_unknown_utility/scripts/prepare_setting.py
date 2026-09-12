#!/usr/bin/env python3
"""Build the auditable, strict Unknown-free usage manifest for one setting."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from common import (
    AUDIT_ROOT,
    EXPECTED_EXECUTION_PLAN_RAW_SHA256,
    EXPECTED_PROTOCOL_CANONICAL_SHA256,
    STAGE3_ROOT,
    load_fold,
    load_label_maps,
    sha256_file,
    split_arrays,
    verify_frozen_inputs,
)


FIELDS = (
    "flow_id",
    "class_name",
    "known_or_unknown",
    "original_split",
    "used_encoder_train",
    "used_encoder_val",
    "used_scaler_fit",
    "used_pca_fit",
    "used_density_fit",
    "used_threshold_calibration",
    "used_final_test",
)
FORBIDDEN_UNKNOWN_FLAGS = (
    "used_encoder_train",
    "used_encoder_val",
    "used_scaler_fit",
    "used_pca_fit",
    "used_density_fit",
    "used_threshold_calibration",
)


def usage_flags(split: str, role: str) -> dict[str, bool]:
    known = role == "Known"
    return {
        "used_encoder_train": known and split == "train",
        "used_encoder_val": known and split == "val",
        "used_scaler_fit": known and split == "train",
        "used_pca_fit": known and split == "train",
        "used_density_fit": known and split == "train",
        "used_threshold_calibration": known and split == "val",
        "used_final_test": split == "test",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", required=True)
    args = parser.parse_args()
    verify_frozen_inputs()
    fold = load_fold(args.setting)
    _, id_to_class = load_label_maps()
    output_dir = STAGE3_ROOT / "outputs" / args.setting
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "data_manifest.csv"
    known_names = set(str(value) for value in fold["known_classes"])
    unknown_names = set(str(value) for value in fold["unknown_classes"])
    counts: Counter[tuple[str, str, str]] = Counter()
    forbidden_unknown_counts = Counter({flag: 0 for flag in FORBIDDEN_UNKNOWN_FLAGS})
    unique_ids: set[str] = set()

    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for split in ("train", "val", "test"):
            flow_ids, labels = split_arrays(split)
            for flow_id, source_label in zip(flow_ids, labels):
                flow_id = str(flow_id)
                if flow_id in unique_ids:
                    raise RuntimeError(f"duplicate flow ID across source splits: {flow_id}")
                unique_ids.add(flow_id)
                class_name = id_to_class[int(source_label)]
                if class_name in known_names:
                    role = "Known"
                elif class_name in unknown_names:
                    role = "Unknown"
                else:
                    raise RuntimeError(f"class absent from fold: {class_name}")
                flags = usage_flags(split, role)
                counts[(role, split, "rows")] += 1
                for flag, value in flags.items():
                    if value:
                        counts[(role, split, flag)] += 1
                        if role == "Unknown" and flag in FORBIDDEN_UNKNOWN_FLAGS:
                            forbidden_unknown_counts[flag] += 1
                writer.writerow(
                    {
                        "flow_id": flow_id,
                        "class_name": class_name,
                        "known_or_unknown": role,
                        "original_split": split,
                        **{key: str(value).lower() for key, value in flags.items()},
                    }
                )

    expected = fold["formal_counts"]
    observed = {
        "known_train": counts[("Known", "train", "rows")],
        "known_validation": counts[("Known", "val", "rows")],
        "known_final_test": counts[("Known", "test", "rows")],
        "unknown_final_test": counts[("Unknown", "test", "rows")],
    }
    if observed != expected:
        raise RuntimeError(f"frozen fold count mismatch: expected={expected}, observed={observed}")
    if any(forbidden_unknown_counts.values()):
        raise RuntimeError(f"Unknown leakage detected: {dict(forbidden_unknown_counts)}")
    alignment_summary = json.loads(
        (AUDIT_ROOT / "outputs/input_alignment_summary.json").read_text(encoding="utf-8")
    )
    if not alignment_summary["all_inputs_valid"]:
        raise RuntimeError("Open-Detect raw-byte alignment gate is not PASS")

    audit = {
        "setting": args.setting,
        "status": "PASS",
        "protocol_canonical_sha256": EXPECTED_PROTOCOL_CANONICAL_SHA256,
        "execution_plan_sha256": EXPECTED_EXECUTION_PLAN_RAW_SHA256,
        "data_manifest_sha256": sha256_file(manifest_path),
        "known_classes": fold["known_classes"],
        "unknown_classes": fold["unknown_classes"],
        "counts": observed,
        "unknown_forbidden_usage_counts": dict(forbidden_unknown_counts),
        "unknown_train_rows_excluded": counts[("Unknown", "train", "rows")],
        "unknown_validation_rows_excluded": counts[("Unknown", "val", "rows")],
        "raw_byte_inputs_all_valid": True,
    }
    (output_dir / "leakage_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        f"# {args.setting} Strict Unknown-Free Leakage Audit",
        "",
        "- Status: **PASS**",
        f"- Known classes: {len(fold['known_classes'])} — {', '.join(fold['known_classes'])}",
        f"- Unknown classes: {len(fold['unknown_classes'])} — {', '.join(fold['unknown_classes'])}",
        f"- Known train / validation / final-test: {observed['known_train']} / {observed['known_validation']} / {observed['known_final_test']}",
        f"- Unknown final-test: {observed['unknown_final_test']}",
        f"- Excluded Unknown source-train/source-validation rows: {audit['unknown_train_rows_excluded']} / {audit['unknown_validation_rows_excluded']}",
        f"- Data manifest SHA-256: `{audit['data_manifest_sha256']}`",
        "",
        "## Mandatory assertions",
        "",
    ]
    for flag in FORBIDDEN_UNKNOWN_FLAGS:
        lines.append(f"- Unknown `{flag}` count = 0: **PASS**")
    lines += [
        "- Unknown is used only when `original_split=test` and `used_final_test=true`: **PASS**",
        "- Fixed source split retained; no new random flow split: **PASS**",
        "- Raw-byte input alignment gate: **PASS**",
        "",
        "This audit permits GPU training for this setting. It does not evaluate any Unknown score.",
    ]
    (output_dir / "unknown_leakage_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(audit, sort_keys=True))


if __name__ == "__main__":
    main()
