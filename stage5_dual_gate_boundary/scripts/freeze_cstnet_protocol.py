#!/usr/bin/env python3
"""Freeze CSTNET class-held-out folds without training or score access."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import struct
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


STAGE5_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = STAGE5_ROOT.parent
AUDIT_ROOT = PROJECT_ROOT / "dataset_suitability_audit" / "outputs"
CONFIG_PATH = STAGE5_ROOT / "configs" / "stage5_freeze_config.json"
VALID_PCAP_MAGIC = {
    bytes.fromhex("d4c3b2a1"): ("<", "microseconds"),
    bytes.fromhex("a1b2c3d4"): (">", "microseconds"),
    bytes.fromhex("4d3cb2a1"): ("<", "nanoseconds"),
    bytes.fromhex("a1b23c4d"): (">", "nanoseconds"),
}


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json(data: object) -> str:
    return json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def write_frozen(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        previous = path.read_text(encoding="utf-8")
        if previous != content:
            raise RuntimeError(f"refusing to modify frozen file: {path}")
        return
    path.write_text(content, encoding="utf-8")


def first_packet_timestamp(path: Path) -> int:
    with path.open("rb") as handle:
        header = handle.read(28)
    if len(header) < 28 or header[:4] not in VALID_PCAP_MAGIC:
        raise RuntimeError(f"invalid or short PCAP header: {path}")
    endian, _ = VALID_PCAP_MAGIC[header[:4]]
    return int(struct.unpack(endian + "I", header[24:28])[0])


def inventory_fingerprint(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for row in frame.sort_values("relative_path").itertuples(index=False):
        line = f"{row.relative_path}\0{int(row.size_bytes)}\0{int(row.first_packet_timestamp_sec)}\n"
        digest.update(line.encode("utf-8"))
    return digest.hexdigest()


def select_unknown_classes(
    frame: pd.DataFrame,
    eligible_classes: list[str],
    unknown_count: int,
    seed: int,
    minimum_samples: int,
    minimum_groups: int,
) -> tuple[list[str], pd.DataFrame, list[dict[str, object]], set[str]]:
    """Use one seeded RNG stream; reject candidates only on data-quality constraints."""
    rng = np.random.default_rng(seed)
    eligible_set = set(eligible_classes)
    audit_rows: list[dict[str, object]] = []
    for attempt in range(1, 10001):
        unknown = sorted(rng.choice(eligible_classes, size=unknown_count, replace=False).tolist())
        unknown_groups = set(frame.loc[frame["class_name"].isin(unknown), "group_id"])
        known_classes = sorted(eligible_set - set(unknown))
        known = frame[frame["class_name"].isin(known_classes)]
        kept = known[~known["group_id"].isin(unknown_groups)].copy()
        counts = kept.groupby("class_name").size().reindex(known_classes, fill_value=0)
        groups = kept.groupby("class_name")["group_id"].nunique().reindex(known_classes, fill_value=0)
        failed_counts = counts[counts < minimum_samples]
        failed_groups = groups[groups < minimum_groups]
        accepted = failed_counts.empty and failed_groups.empty
        reasons: list[str] = []
        if not failed_counts.empty:
            reasons.append(
                "post_group_purge_sample_count_below_"
                + str(minimum_samples)
                + ":"
                + "|".join(f"{name}={int(value)}" for name, value in failed_counts.items())
            )
        if not failed_groups.empty:
            reasons.append(
                "post_group_purge_group_count_below_"
                + str(minimum_groups)
                + ":"
                + "|".join(f"{name}={int(value)}" for name, value in failed_groups.items())
            )
        audit_rows.append(
            {
                "attempt": attempt,
                "seed": seed,
                "candidate_unknown_classes": "|".join(unknown),
                "accepted": accepted,
                "rejection_reason": "NONE" if accepted else ";".join(reasons),
                "minimum_retained_known_samples": int(counts.min()),
                "minimum_retained_known_groups": int(groups.min()),
                "known_samples_before_group_purge": len(known),
                "known_samples_after_group_purge": len(kept),
                "purged_known_samples": len(known) - len(kept),
            }
        )
        if accepted:
            return unknown, kept, audit_rows, unknown_groups
    raise RuntimeError("no deterministic class fold satisfied the frozen data-quality constraints")


def assign_known_splits(
    frame: pd.DataFrame,
    seed: int,
    n_splits: int,
) -> tuple[pd.Series, int, int, dict[str, object]]:
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_ids = np.full(len(frame), -1, dtype=np.int64)
    for fold_id, (_, indices) in enumerate(
        splitter.split(np.zeros(len(frame)), frame["class_name"], groups=frame["group_id"])
    ):
        fold_ids[indices] = fold_id
    if np.any(fold_ids < 0):
        raise RuntimeError("StratifiedGroupKFold left unassigned samples")
    working = frame.copy()
    working["fold_id"] = fold_ids
    table = (
        working.groupby(["class_name", "fold_id"]).size().unstack(fill_value=0).reindex(columns=range(n_splits), fill_value=0)
    )
    totals = table.sum(axis=1)
    target_fraction = 1.0 / n_splits
    candidates: list[tuple[float, float, float, int, int]] = []
    for validation_fold, test_fold in itertools.permutations(range(n_splits), 2):
        minimum_count = min(int(table[validation_fold].min()), int(table[test_fold].min()))
        classwise_deviation = float(
            (
                (table[validation_fold] - target_fraction * totals).abs()
                + (table[test_fold] - target_fraction * totals).abs()
            ).sum()
        )
        global_deviation = float(
            abs(int(table[validation_fold].sum()) - target_fraction * len(working))
            + abs(int(table[test_fold].sum()) - target_fraction * len(working))
        )
        candidates.append(
            (-minimum_count, classwise_deviation, global_deviation, validation_fold, test_fold)
        )
    best = min(candidates)
    validation_fold, test_fold = best[-2], best[-1]
    split = pd.Series("known_train", index=frame.index, dtype=object)
    split.iloc[np.flatnonzero(fold_ids == validation_fold)] = "known_validation"
    split.iloc[np.flatnonzero(fold_ids == test_fold)] = "known_test"
    details = {
        "validation_fold_id": validation_fold,
        "test_fold_id": test_fold,
        "minimum_per_class_validation_or_test_count": int(-best[0]),
        "classwise_L1_deviation": best[1],
        "global_size_deviation": best[2],
    }
    return split, validation_fold, test_fold, details


def active_group_disjoint(manifest: pd.DataFrame) -> bool:
    active = manifest[
        manifest["split"].isin(
            ["known_train", "known_validation", "known_test", "unknown_final_test"]
        )
    ]
    return bool((active.groupby("group_id")["split"].nunique() == 1).all())


def run(output_root: Path) -> dict[str, object]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    protocol_config = config["cstnet"]
    protocol_root = output_root / "cstnet_protocol"
    protocol_root.mkdir(parents=True, exist_ok=True)

    class_balance_path = AUDIT_ROOT / "class_balance.csv"
    pcap_inventory_path = AUDIT_ROOT / "pcap_inventory.csv"
    leakage_path = AUDIT_ROOT / "leakage_risk_audit.csv"
    source_class_path = AUDIT_ROOT / "class_inventory.csv"
    class_balance = pd.read_csv(class_balance_path)
    source_class = pd.read_csv(source_class_path)
    leakage = pd.read_csv(leakage_path)
    pcap = pd.read_csv(pcap_inventory_path)

    class_balance = class_balance[
        class_balance["dataset"].eq(protocol_config["dataset_name"])
        & class_balance["label_level"].eq(protocol_config["label_level"])
    ].copy()
    if len(class_balance) != 120:
        raise RuntimeError(f"expected 120 CSTNET classes, got {len(class_balance)}")
    pcap = pcap[
        pcap["dataset"].eq(protocol_config["dataset_name"])
        & pcap["readable_magic"].astype(bool)
        & ~pcap["macos_sidecar"].astype(bool)
    ].copy()
    if len(pcap) != 46372:
        raise RuntimeError(f"expected 46,372 readable CSTNET PCAPs, got {len(pcap)}")

    dataset_root = Path(protocol_config["dataset_root"])
    pcap["class_name"] = pcap["relative_path"].map(lambda value: Path(value).parent.name)
    timestamps = [first_packet_timestamp(dataset_root / path) for path in pcap["relative_path"]]
    pcap["first_packet_timestamp_sec"] = np.asarray(timestamps, dtype=np.int64)
    window = int(protocol_config["group_proxy_seconds"])
    pcap["group_epoch"] = pcap["first_packet_timestamp_sec"] // window
    pcap["group_id"] = pcap["group_epoch"].map(lambda value: f"utc-minute-{int(value)}")
    pcap["first_packet_timestamp_utc"] = pd.to_datetime(
        pcap["first_packet_timestamp_sec"], unit="s", utc=True
    ).dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    pcap_counts = pcap.groupby("class_name").size()
    source_labels = source_class[
        source_class["dataset"].eq(protocol_config["dataset_name"])
        & source_class["label_level"].eq(protocol_config["label_level"])
    ].set_index("class_name")
    risk = leakage[leakage["dataset"].eq(protocol_config["dataset_name"])].set_index("class_name")
    minimum = int(protocol_config["minimum_total_samples_per_class"])
    class_rows: list[dict[str, object]] = []
    for row in class_balance.sort_values("class_name").itertuples(index=False):
        name = str(row.class_name)
        sample_count = int(row.sample_count)
        if name not in pcap_counts or int(pcap_counts[name]) != sample_count:
            raise RuntimeError(f"{name}: class-count/PCAP-count mismatch")
        eligible = sample_count >= minimum
        provenance = source_labels.loc[name, "label_provenance"] if name in source_labels.index else "UNKNOWN"
        risk_level = risk.loc[name, "risk_level"] if name in risk.index else "UNKNOWN"
        class_rows.append(
            {
                "class_name": name,
                "sample_count": sample_count,
                "pcap_count": int(pcap_counts[name]),
                "domain_count": 1,
                "endpoint_count": "UNKNOWN_NOT_FLOW_PARSED",
                "eligible_for_open_set": eligible,
                "exclusion_reason": "NONE" if eligible else f"sample_count_{sample_count}_below_{minimum}",
                "label_provenance": provenance,
                "domain_endpoint_risk": risk_level,
            }
        )
    class_inventory = pd.DataFrame(class_rows)
    eligible_classes = sorted(
        class_inventory.loc[class_inventory["eligible_for_open_set"], "class_name"].tolist()
    )
    excluded_classes = sorted(
        class_inventory.loc[~class_inventory["eligible_for_open_set"], "class_name"].tolist()
    )
    exclusion_fraction = len(excluded_classes) / len(class_inventory)
    if exclusion_fraction > float(protocol_config["mass_exclusion_stop_fraction"]):
        raise RuntimeError("eligibility rule causes mass class deletion; protocol freeze stopped")

    threshold_rows = [
        {
            "minimum_total_samples_per_class": threshold,
            "eligible_classes": int((class_inventory["sample_count"] >= threshold).sum()),
            "excluded_classes": int((class_inventory["sample_count"] < threshold).sum()),
            "selected_rule": threshold == minimum,
            "selection_basis": "engineering minimum chosen before modeling",
        }
        for threshold in protocol_config["eligibility_thresholds_to_report"]
    ]
    threshold_frame = pd.DataFrame(threshold_rows)

    group_rows: list[dict[str, object]] = []
    for group_id, group in pcap.groupby("group_id", sort=True):
        class_names = sorted(group["class_name"].unique())
        epoch = int(group["group_epoch"].iloc[0])
        group_rows.append(
            {
                "group_id": group_id,
                "group_start_utc": datetime.fromtimestamp(epoch * window, timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                ),
                "group_size": len(group),
                "class_count": len(class_names),
                "cross_class_group": len(class_names) > 1,
                "class_names": "|".join(class_names),
                "group_source": "PCAP first-packet UTC timestamp floored to one minute",
            }
        )
    grouping_audit = pd.DataFrame(group_rows).sort_values("group_id")
    group_lookup = grouping_audit.set_index("group_id")
    pcap["group_class_count"] = pcap["group_id"].map(group_lookup["class_count"])
    cross_class_flow_ratio = float((pcap["group_class_count"] > 1).mean())
    dataset_fingerprint = inventory_fingerprint(pcap)

    all_manifests: list[pd.DataFrame] = []
    fold_summaries: dict[str, dict[str, object]] = {}
    selection_audits: list[pd.DataFrame] = []
    for fold_config in protocol_config["folds"]:
        fold_name = str(fold_config["name"])
        openness = float(fold_config["openness_ratio"])
        seed = int(fold_config["seed"])
        unknown_count = max(1, round(len(eligible_classes) * openness))
        unknown_classes, kept_known, attempts, unknown_groups = select_unknown_classes(
            pcap,
            eligible_classes,
            unknown_count,
            seed,
            int(protocol_config["minimum_known_samples_after_group_purge"]),
            int(protocol_config["minimum_known_groups_after_group_purge"]),
        )
        attempt_frame = pd.DataFrame(attempts)
        attempt_frame.insert(0, "fold", fold_name)
        selection_audits.append(attempt_frame)
        split, validation_fold, test_fold, selection_details = assign_known_splits(
            kept_known,
            seed,
            int(protocol_config["known_group_n_splits"]),
        )
        kept_known = kept_known.copy()
        kept_known["assigned_split"] = split.to_numpy()
        known_split_by_path = kept_known.set_index("relative_path")["assigned_split"]

        manifest = pcap[
            [
                "relative_path",
                "class_name",
                "size_bytes",
                "first_packet_timestamp_sec",
                "first_packet_timestamp_utc",
                "group_id",
                "group_class_count",
            ]
        ].copy()
        manifest.insert(0, "fold", fold_name)
        manifest["eligible_class"] = manifest["class_name"].isin(eligible_classes)
        manifest["class_role"] = "known"
        manifest.loc[manifest["class_name"].isin(unknown_classes), "class_role"] = "unknown"
        manifest.loc[manifest["class_name"].isin(excluded_classes), "class_role"] = "excluded_ineligible"
        manifest["split"] = "excluded_ineligible"
        manifest["exclusion_reason"] = "class_sample_count_below_100"
        unknown_mask = manifest["class_role"].eq("unknown")
        manifest.loc[unknown_mask, "split"] = "unknown_final_test"
        manifest.loc[unknown_mask, "exclusion_reason"] = "NONE"
        known_mask = manifest["class_role"].eq("known")
        overlap_mask = known_mask & manifest["group_id"].isin(unknown_groups)
        manifest.loc[overlap_mask, "split"] = "excluded_group_overlap_with_unknown"
        manifest.loc[overlap_mask, "exclusion_reason"] = "capture_group_shared_with_unknown_class"
        retained_mask = known_mask & ~overlap_mask
        manifest.loc[retained_mask, "split"] = manifest.loc[retained_mask, "relative_path"].map(
            known_split_by_path
        )
        manifest.loc[retained_mask, "exclusion_reason"] = "NONE"
        if manifest.loc[retained_mask, "split"].isna().any():
            raise RuntimeError(f"{fold_name}: retained Known path lacks group split")
        if not active_group_disjoint(manifest):
            raise RuntimeError(f"{fold_name}: active split group overlap detected")
        all_manifests.append(manifest)

        active_known = manifest[manifest["split"].isin(["known_train", "known_validation", "known_test"])]
        per_class_split = active_known.groupby(["class_name", "split"]).size().unstack(fill_value=0)
        counts = manifest["split"].value_counts().to_dict()
        fold_summary = {
            "fold_name": fold_name,
            "seed": seed,
            "openness_ratio_requested": openness,
            "eligible_class_count": len(eligible_classes),
            "unknown_class_count": len(unknown_classes),
            "known_class_count": len(eligible_classes) - len(unknown_classes),
            "unknown_classes": unknown_classes,
            "known_classes": sorted(set(eligible_classes) - set(unknown_classes)),
            "excluded_classes": excluded_classes,
            "accepted_selection_attempt": len(attempts),
            "rejected_selection_attempts": len(attempts) - 1,
            "selection_rejection_reasons": [
                row["rejection_reason"] for row in attempts if not row["accepted"]
            ],
            "capture_groups_containing_unknown": len(unknown_groups),
            "known_samples_purged_for_unknown_group_overlap": int(
                counts.get("excluded_group_overlap_with_unknown", 0)
            ),
            "active_group_overlap_count": 0,
            "group_disjoint": True,
            "known_split_counts": {
                split_name: int(counts.get(split_name, 0))
                for split_name in ("known_train", "known_validation", "known_test")
            },
            "unknown_final_test_samples": int(counts.get("unknown_final_test", 0)),
            "minimum_per_class_split_counts": {
                column: int(per_class_split[column].min()) for column in per_class_split.columns
            },
            "validation_fold_id": validation_fold,
            "test_fold_id": test_fold,
            "validation_test_fold_selection_details": selection_details,
            "created_before_unknown_evaluation": True,
            "unknown_score_accessed": False,
        }
        fold_summaries[fold_name] = fold_summary
        write_frozen(protocol_root / f"{fold_name}_fold.json", stable_json(fold_summary))

    split_manifest = pd.concat(all_manifests, ignore_index=True)
    fold_selection_audit = pd.concat(selection_audits, ignore_index=True)
    write_frozen(
        protocol_root / "class_inventory.csv",
        class_inventory.to_csv(index=False, lineterminator="\n"),
    )
    write_frozen(
        protocol_root / "eligibility_thresholds.csv",
        threshold_frame.to_csv(index=False, lineterminator="\n"),
    )
    write_frozen(
        protocol_root / "grouping_audit.csv",
        grouping_audit.to_csv(index=False, lineterminator="\n"),
    )
    write_frozen(
        protocol_root / "fold_selection_audit.csv",
        fold_selection_audit.to_csv(index=False, lineterminator="\n"),
    )
    write_frozen(
        protocol_root / "split_manifest.csv",
        split_manifest.to_csv(index=False, lineterminator="\n"),
    )

    audit_source_hashes = {
        path.name: sha256_file(path)
        for path in (class_balance_path, pcap_inventory_path, leakage_path, source_class_path)
    }
    protocol = {
        "protocol_name": "CSTNET-TLS1.3 Strict Unknown-Free External Validation",
        "status": "FROZEN_BEFORE_UNKNOWN_EVALUATION",
        "created_before_unknown_evaluation": True,
        "cstnet_training_executed": False,
        "cstnet_unknown_scores_accessed": False,
        "dataset": {
            "root": str(dataset_root),
            "raw_class_count": len(class_inventory),
            "readable_pcap_count": len(pcap),
            "inventory_fingerprint_type": "SHA256(relative_path, size_bytes, first_packet_timestamp_sec)",
            "inventory_fingerprint_sha256": dataset_fingerprint,
            "source_audit_hashes": audit_source_hashes,
        },
        "eligibility": {
            "rule": f"sample_count >= {minimum}",
            "basis": "engineering minimum chosen before modeling",
            "eligible_class_count": len(eligible_classes),
            "eligible_classes": eligible_classes,
            "excluded_class_count": len(excluded_classes),
            "excluded_classes": excluded_classes,
            "exclusion_fraction": exclusion_fraction,
            "reported_thresholds": threshold_rows,
        },
        "grouping": {
            "group_id": protocol_config["group_proxy"],
            "window_seconds": window,
            "reason": "no authoritative capture/session manifest exists; use a deterministic first-packet co-capture proxy",
            "raw_group_count": len(grouping_audit),
            "cross_class_group_count": int(grouping_audit["cross_class_group"].sum()),
            "cross_class_flow_ratio": cross_class_flow_ratio,
            "unknown_overlap_policy": protocol_config["unknown_group_overlap_policy"],
            "domain_endpoint_risk": "PROTOCOL_RISK",
            "risk_reason": "domain directory is the class label and endpoint metadata remains UNKNOWN_NOT_FLOW_PARSED",
        },
        "known_split": {
            "algorithm": protocol_config["known_group_split"],
            "n_splits": protocol_config["known_group_n_splits"],
            "target": protocol_config["known_split_target"],
            "validation_test_fold_selection": protocol_config[
                "validation_test_fold_selection"
            ],
        },
        "class_holdout_folds": fold_summaries,
        "strict_unknown_free_constraints": {
            "forbidden_for_unknown_classes": [
                "encoder train",
                "encoder validation",
                "checkpoint selection",
                "scaler",
                "PCA",
                "Gaussian/GMM",
                "boundary calibration",
                "hyperparameter tuning",
            ],
            "unknown_allowed_only_in": "future final test after method/protocol freeze",
        },
        "future_comparison_methods": [
            "Open-Detect Native",
            "Single-Full K1",
            "Multi-Global K2",
            "Class-P05 K2",
            "Component-P05 K2",
            "DGSB K2",
        ],
        "proposed_candidate": "DGSB K2",
        "performance_claim": "NONE_BEFORE_CSTNET_UNKNOWN_EVALUATION",
    }
    write_frozen(protocol_root / "cstnet_open_set_protocol.json", stable_json(protocol))

    fold_lines: list[str] = []
    for name in ("low", "medium", "high"):
        summary = fold_summaries[name]
        fold_lines.extend(
            [
                f"### {name.title()}",
                "",
                f"- Seed: `{summary['seed']}`",
                f"- Known/Unknown classes: `{summary['known_class_count']}/{summary['unknown_class_count']}`",
                f"- Unknown classes: `{', '.join(summary['unknown_classes'])}`",
                f"- Selection attempt: `{summary['accepted_selection_attempt']}`; rejected attempts: `{summary['rejected_selection_attempts']}`",
                f"- Known train/validation/test samples: `{summary['known_split_counts']['known_train']}/{summary['known_split_counts']['known_validation']}/{summary['known_split_counts']['known_test']}`",
                f"- Purged Known samples sharing an Unknown capture group: `{summary['known_samples_purged_for_unknown_group_overlap']}`",
                "",
            ]
        )
    protocol_md = f"""# CSTNET-TLS1.3 Strict Unknown-Free Protocol Freeze

## Scope

This document freezes class eligibility, group-aware Known splits and three
class-held-out folds. No CSTNET encoder training, representation export,
density fitting, threshold calibration or Unknown scoring was run.

## Dataset and Eligibility

- Original classes: `{len(class_inventory)}`
- Readable per-flow PCAPs: `{len(pcap)}`
- Engineering eligibility rule fixed before modeling: `sample_count >= {minimum}`
- Eligible/excluded classes: `{len(eligible_classes)}/{len(excluded_classes)}`
- Excluded: `{', '.join(excluded_classes)}`
- Inventory fingerprint: `{dataset_fingerprint}`

## Group-Aware Split

The source has no authoritative capture/session manifest. `group_id` is frozen
as the UTC one-minute window containing the first packet timestamp. Before
Known splitting, Known samples sharing any group with a selected Unknown class
are excluded. Remaining Known groups are assigned by a deterministic shuffled
10-fold `StratifiedGroupKFold`; two folds are selected for Validation and Test
using the frozen count-only lexicographic criterion, and eight folds form Train.

This makes active Train/Validation/Test/Unknown groups disjoint. It is a proxy,
not proof of true session identity.

## Domain and Endpoint Risk

`PROTOCOL_RISK`: the domain directory is also the class label. Endpoint metadata
was not flow-parsed by the source audit, so endpoint novelty remains unresolved.
The capture-window control does not eliminate that semantic shortcut risk.

## Frozen Folds

{''.join(fold_lines)}
## Strict Unknown-Free Boundary

Unknown classes are forbidden from encoder training/validation, checkpoint
selection, scaler/PCA/Gaussian/GMM fitting, boundary calibration and tuning.
They may be accessed only in the future final test, after this freeze.

`created_before_unknown_evaluation = true`
"""
    write_frozen(protocol_root / "cstnet_open_set_protocol.md", protocol_md)

    files_to_hash = [
        "class_inventory.csv",
        "eligibility_thresholds.csv",
        "grouping_audit.csv",
        "fold_selection_audit.csv",
        "low_fold.json",
        "medium_fold.json",
        "high_fold.json",
        "split_manifest.csv",
        "cstnet_open_set_protocol.json",
        "cstnet_open_set_protocol.md",
    ]
    hash_lines = [f"{sha256_file(protocol_root / name)}  {name}" for name in files_to_hash]
    write_frozen(protocol_root / "split_hashes.sha256", "\n".join(hash_lines) + "\n")
    return protocol


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=STAGE5_ROOT / "outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = run(args.output_root.resolve())
    print(
        json.dumps(
            {
                "status": protocol["status"],
                "raw_classes": protocol["dataset"]["raw_class_count"],
                "eligible_classes": protocol["eligibility"]["eligible_class_count"],
                "training_executed": protocol["cstnet_training_executed"],
                "unknown_scores_accessed": protocol["cstnet_unknown_scores_accessed"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
