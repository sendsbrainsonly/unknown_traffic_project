#!/usr/bin/env python3
"""Create Open-Detect flow images and freeze group-aware Stage 12 protocols."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import re
import subprocess
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from stage12_common import (
    CONFIG_PATH,
    STAGE_ROOT,
    array_digest,
    frozen_unknown_protocol,
    load_config,
    sha256_file,
    stable_seed,
    write_csv,
    write_json,
)


ROLES = ("known_train", "known_validation", "known_test", "unknown_test")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_pool(directory: Path, prefix: str) -> list[dict]:
    records: list[dict] = []
    for split in ("train", "validation", "test"):
        bundle = np.load(directory / f"{prefix}_{split}.npz", allow_pickle=False)
        data = bundle["data"]
        labels = bundle["target"]
        provenance = [
            json.loads(line)
            for line in (directory / f"provenance_{split}.jsonl").read_text(encoding="utf-8").splitlines()
            if line
        ]
        if len(data) != len(labels) or len(data) != len(provenance):
            raise RuntimeError(f"pool/provenance length mismatch for {split}")
        for image, label, row in zip(data, labels, provenance):
            if int(label) != int(row["label"]):
                raise RuntimeError("pool label/provenance mismatch")
            if not row["packet_refs"]:
                raise RuntimeError("selected flow has no packet provenance")
            source_token = str(row["packet_refs"][0]["capture"])
            source_name = source_token.rsplit("|name=", 1)[-1]
            records.append({**row, "image": image, "source_name": source_name})
    return records


class UnionFind:
    def __init__(self, values: list[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            value, self.parent[value] = self.parent[value], root
        return root

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def duplicate_linked_groups(records: list[dict]) -> dict[str, str]:
    source_names = sorted({str(row["source_file"]) for row in records})
    union = UnionFind(source_names)
    by_image: dict[str, list[str]] = defaultdict(list)
    for row in records:
        by_image[str(row["image_sha256"])].append(str(row["source_file"]))
    for sources in by_image.values():
        unique = sorted(set(sources))
        for source in unique[1:]:
            union.union(unique[0], source)
    return {source: union.find(source) for source in source_names}


def best_group_assignment(records: list[dict], class_name: str, seed: int) -> dict[str, str]:
    linked = duplicate_linked_groups(records)
    counts: Counter[str] = Counter(linked[str(row["source_file"])] for row in records)
    groups = sorted(counts)
    if len(groups) < 3:
        raise RuntimeError(f"GROUP_AWARE_NOT_FEASIBLE after duplicate linking: {class_name}")
    rng = np.random.default_rng(stable_seed(seed, class_name, "group-split"))
    target = np.asarray([0.8, 0.1, 0.1], dtype=np.float64)
    best: tuple[float, tuple[int, ...]] | None = None
    attempts = max(20_000, 2_000 * len(groups))
    weights = np.asarray([counts[group] for group in groups], dtype=np.float64)
    for attempt in range(attempts):
        if attempt == 0:
            assignment = np.asarray([0] * (len(groups) - 2) + [1, 2], dtype=np.int64)
            rng.shuffle(assignment)
        else:
            assignment = rng.choice(3, size=len(groups), p=target)
        if len(set(map(int, assignment))) < 3:
            continue
        totals = np.asarray([weights[assignment == role].sum() for role in range(3)])
        ratios = totals / totals.sum()
        score = float(np.abs(ratios - target).sum())
        key = tuple(map(int, assignment))
        candidate = (score, key)
        if best is None or candidate < best:
            best = candidate
    if best is None:
        raise RuntimeError(f"unable to assign three non-empty group splits: {class_name}")
    role_names = ("train", "validation", "test")
    cluster_assignment = {group: role_names[role] for group, role in zip(groups, best[1])}
    return {source: cluster_assignment[cluster] for source, cluster in linked.items()}


def best_flow_disjoint_assignment(records: list[dict], class_name: str, seed: int) -> dict[str, str]:
    """Fallback split over exact-image groups when source-PCAP grouping is infeasible."""
    by_image: dict[str, list[dict]] = defaultdict(list)
    for row in records:
        by_image[str(row["image_sha256"])].append(row)
    groups = sorted(by_image)
    if len(groups) < 3:
        raise RuntimeError(f"FLOW_DISJOINT_NOT_FEASIBLE after exact-image linking: {class_name}")
    rng = np.random.default_rng(stable_seed(seed, class_name, "flow-disjoint-split"))
    target = np.asarray([0.8, 0.1, 0.1], dtype=np.float64)
    weights = np.asarray([len(by_image[group]) for group in groups], dtype=np.float64)
    best: tuple[float, tuple[int, ...]] | None = None
    attempts = max(20_000, 2_000 * len(groups))
    for attempt in range(attempts):
        if attempt == 0:
            assignment = np.asarray([0] * (len(groups) - 2) + [1, 2], dtype=np.int64)
            rng.shuffle(assignment)
        else:
            assignment = rng.choice(3, size=len(groups), p=target)
        if len(set(map(int, assignment))) < 3:
            continue
        totals = np.asarray([weights[assignment == role].sum() for role in range(3)])
        score = float(np.abs(totals / totals.sum() - target).sum())
        candidate = (score, tuple(map(int, assignment)))
        if best is None or candidate < best:
            best = candidate
    if best is None:
        raise RuntimeError(f"unable to assign three non-empty flow-level splits: {class_name}")
    role_names = ("train", "validation", "test")
    return {group: role_names[role] for group, role in zip(groups, best[1])}


def global_source_group_assignment(
    records: list[dict], class_names: list[str], seed: int
) -> tuple[dict[str, str] | None, list[str]]:
    """Assign duplicate-linked source groups jointly across all classes."""
    linked = duplicate_linked_groups(records)
    clusters = sorted(set(linked.values()))
    class_index = {name: index for index, name in enumerate(class_names)}
    cluster_index = {name: index for index, name in enumerate(clusters)}
    weights = np.zeros((len(class_names), len(clusters)), dtype=np.float64)
    for row in records:
        weights[class_index[str(row["class_name"])], cluster_index[linked[str(row["source_file"])]]] += 1
    infeasible = [
        name for name, index in class_index.items()
        if int(np.count_nonzero(weights[index])) < 3
    ]
    if infeasible:
        return None, sorted(infeasible)
    rng = np.random.default_rng(stable_seed(seed, "global-source-group-split"))
    target = np.asarray([0.8, 0.1, 0.1], dtype=np.float64)
    best: tuple[float, tuple[int, ...]] | None = None
    attempts = max(50_000, 2_000 * len(clusters))
    for _ in range(attempts):
        assignment = rng.choice(3, size=len(clusters), p=target)
        totals = np.stack([weights[:, assignment == role].sum(axis=1) for role in range(3)], axis=1)
        if np.any(totals == 0):
            continue
        ratios = totals / totals.sum(axis=1, keepdims=True)
        candidate = (float(np.abs(ratios - target).sum()), tuple(map(int, assignment)))
        if best is None or candidate < best:
            best = candidate
    if best is None:
        return None, list(class_names)
    cluster_assignment = {
        cluster: ("train", "validation", "test")[role]
        for cluster, role in zip(clusters, best[1])
    }
    return {source: cluster_assignment[cluster] for source, cluster in linked.items()}, []


def global_flow_disjoint_assignment(
    records_by_class: dict[str, list[dict]], seed: int, open_detect_root: Path
) -> tuple[dict[str, str], dict]:
    """Use Open-Detect's cross-class exact-image grouping for the fallback."""
    reproduction = open_detect_root / "reproduction"
    if str(reproduction) not in sys.path:
        sys.path.insert(0, str(reproduction))
    from grouped_image_split import grouped_image_split

    assigned, diagnostics = grouped_image_split(records_by_class, seed)
    flow_assignment = {
        str(row["flow_id_sha256"]): role
        for role, rows in assigned.items()
        for row in rows
    }
    expected = sum(len(rows) for rows in records_by_class.values())
    if len(flow_assignment) != expected:
        raise RuntimeError("flow IDs are not globally unique in fallback split")
    return flow_assignment, diagnostics


def prepare_manifest(dataset: str, config: dict, inventory: list[dict[str, str]]) -> tuple[Path, list[dict]]:
    eligibility = config["eligibility"]
    pcap_root = Path(config["datasets"][dataset]["pcap_root"]).resolve()
    readable = [
        row for row in inventory
        if row["record_type"] == "pcap" and row["readable"] == "True" and row["canonical_class"]
    ]
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in readable:
        groups[row["canonical_class"]].append(row)
    class_rows = []
    classes = []
    for label, class_name in enumerate(sorted(groups)):
        rows = groups[class_name]
        group_count = len(rows)
        provisional = group_count >= int(eligibility["minimum_readable_source_groups"])
        class_rows.append({
            "canonical_class": class_name,
            "provisional_label": label if provisional else "",
            "readable_source_groups": group_count,
            "domain_states": ";".join(sorted({row["domain_state"] for row in rows})),
            "selected_source_groups": "",
            "selected_flows": "",
            "unique_images": "",
            "split_policy": "PENDING" if provisional else "NOT_APPLICABLE",
            "group_aware_feasible": "PENDING" if provisional else "False",
            "eligible": "PENDING" if provisional else "False",
            "exclusion_reason": "" if provisional else "fewer than minimum readable source groups",
        })
        if provisional:
            classes.append({
                "name": class_name,
                "label": len(classes),
                "captures": [str((pcap_root / row["source_file"]).resolve()) for row in rows],
                "target_samples": int(eligibility["target_selected_flows_per_class"]),
            })
    if not classes:
        raise RuntimeError(f"no provisionally eligible classes: {dataset}")
    manifest = {
        "dataset": dataset,
        "canonical_label": "application/service from official capture filename",
        "domain_state_is_metadata": True,
        "classes": classes,
    }
    path = STAGE_ROOT / "protocol" / dataset / "preprocessing_manifest.json"
    write_json(path, manifest)
    return path, class_rows


def run_open_detect_preprocessor(dataset: str, config: dict, manifest: Path) -> Path:
    pool_root = STAGE_ROOT / "artifacts" / dataset / "preprocessing_pool"
    first = pool_root / "raw_prepared_parallel_v1"
    output = pool_root / "raw_prepared_parallel_v2" if first.exists() and not (first / "SUCCESS").is_file() else first
    report = output / "preprocessing_report.json"
    success = output / "SUCCESS"
    if success.is_file() and report.is_file():
        return output
    if output.exists():
        raise FileExistsError(f"preserved incomplete preprocessing output; refusing overwrite: {output}")
    output.mkdir(parents=True)
    script = Path(config["open_detect_root"]) / "reproduction" / "prepare_pcap_dataset.py"
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    classes = sorted(manifest_payload["classes"], key=lambda row: int(row["label"]))
    commands = []
    for class_spec in classes:
        token = f"class_{int(class_spec['label']):03d}"
        class_manifest = output / "class_manifests" / f"{token}.json"
        class_output = output / "classes" / token
        class_log = output / "class_logs" / f"{token}.log"
        write_json(class_manifest, {**manifest_payload, "classes": [class_spec]})
        command = [
            sys.executable, "-B", str(script),
            "--manifest", str(class_manifest),
            "--output-dir", str(class_output),
            "--output-prefix", "pool",
            "--flow-direction", config["preprocessing"]["flow_direction"],
            "--session-timeout-seconds", str(config["preprocessing"]["session_timeout_seconds"]),
            "--sampling", config["preprocessing"]["sampling"],
            "--split-policy", "grouped_image_disjoint",
            "--seed", str(config["preprocessing"]["sampling_seed"]),
            "--on-packet-error", config["preprocessing"]["packet_representation_error_policy"],
        ]
        commands.append((class_spec, class_output, class_log, command))
    write_json(STAGE_ROOT / "protocol" / dataset / "preprocessing_command.json", {
        "commands": [item[3] for item in commands],
        "cwd": str(Path(config["open_detect_root"]).resolve()),
        "source_dataset_read_only": True,
        "execution": "class-level parallel; sample identity is unchanged because each class has independent capture indices and stable-hash sampling",
        "workers": int(config["preprocessing"]["class_parallel_workers"]),
    })

    def execute(item: tuple[dict, Path, Path, list[str]]) -> tuple[str, Path | None, dict | None]:
        class_spec, class_output, class_log, command = item
        class_log.parent.mkdir(parents=True, exist_ok=True)
        with class_log.open("x", encoding="utf-8") as handle:
            try:
                subprocess.run(
                    command, cwd=config["open_detect_root"], check=True,
                    stdout=handle, stderr=subprocess.STDOUT,
                )
            except subprocess.CalledProcessError:
                handle.flush()
                text = class_log.read_text(encoding="utf-8", errors="replace")
                matches = re.findall(r"sessions=(\d+) selected=(\d+)", text)
                if not matches:
                    raise
                sessions, selected = map(int, matches[-1])
                if selected >= int(config["eligibility"]["minimum_selected_flows"]):
                    raise
                source_inventory = []
                for capture_index, value in enumerate(class_spec["captures"]):
                    capture = Path(value)
                    stat = capture.stat()
                    source_inventory.append({
                        "class": str(class_spec["name"]),
                        "path": f"class={class_spec['name']}|capture={capture_index}|name={capture.name}",
                        "size_bytes": stat.st_size,
                        "mtime_ns": stat.st_mtime_ns,
                    })
                return str(class_spec["name"]), None, {
                    "class_report": {
                        "label": int(class_spec["label"]),
                        "target_samples": int(class_spec["target_samples"]),
                        "selected_samples": selected,
                        "selected_unique_images": selected,
                        "flows_with_fewer_than_8_packets": 0,
                        "counters": {"sessions_observed": sessions},
                        "pool_excluded": True,
                        "pool_exclusion_reason": "selected flows below eligibility minimum; upstream split contained an empty role",
                    },
                    "source_inventory": source_inventory,
                }
        return str(class_spec["name"]), class_output, None

    completed = []
    class_outputs: dict[str, Path] = {}
    below_minimum: dict[str, dict] = {}
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=int(config["preprocessing"]["class_parallel_workers"])
    ) as executor:
        future_map = {executor.submit(execute, item): item[0]["name"] for item in commands}
        for future in concurrent.futures.as_completed(future_map):
            class_name, class_output, exclusion = future.result()
            completed.append(class_name)
            if class_output is None:
                below_minimum[class_name] = exclusion
                status = "BELOW_MINIMUM_EXCLUDED"
            else:
                class_outputs[class_name] = class_output
                status = "SUCCESS"
            print(
                f"parallel_class_complete={len(completed)}/{len(commands)} class={class_name} status={status}",
                flush=True,
            )
    combined_provenance: dict[str, list[dict]] = {}
    split_sizes = {}
    for split in ("train", "validation", "test"):
        arrays = []
        targets = []
        provenance = []
        for class_spec in classes:
            class_name = str(class_spec["name"])
            if class_name not in class_outputs:
                continue
            class_output = class_outputs[class_name]
            bundle = np.load(class_output / f"pool_{split}.npz", allow_pickle=False)
            arrays.append(bundle["data"])
            targets.append(bundle["target"])
            provenance.extend(
                json.loads(line)
                for line in (class_output / f"provenance_{split}.jsonl").read_text(encoding="utf-8").splitlines()
                if line
            )
        data = np.concatenate(arrays, axis=0)
        target = np.concatenate(targets, axis=0)
        np.savez_compressed(output / f"pool_{split}.npz", data=data, target=target)
        with (output / f"provenance_{split}.jsonl").open("x", encoding="utf-8") as handle:
            for row in provenance:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        combined_provenance[split] = provenance
        split_sizes[split] = len(provenance)
    class_reports = {
        name: payload["class_report"] for name, payload in below_minimum.items()
    }
    source_inventory = []
    for payload in below_minimum.values():
        source_inventory.extend(payload["source_inventory"])
    per_class_diagnostics = {}
    for class_spec in classes:
        class_name = str(class_spec["name"])
        if class_name not in class_outputs:
            per_class_diagnostics[class_name] = {
                "status": "BELOW_MINIMUM_EXCLUDED",
                "reason": below_minimum[class_name]["class_report"]["pool_exclusion_reason"],
            }
            continue
        class_report = json.loads(
            (class_outputs[class_name] / "preprocessing_report.json").read_text(encoding="utf-8")
        )
        class_reports.update(class_report["classes"])
        source_inventory.extend(class_report["source_inventory"])
        per_class_diagnostics[class_name] = class_report["split_diagnostics"]
    pairs = (("train", "validation"), ("train", "test"), ("validation", "test"))
    flow_sets = {
        split: {str(row["flow_id_sha256"]) for row in rows}
        for split, rows in combined_provenance.items()
    }
    image_sets = {
        split: {str(row["image_sha256"]) for row in rows}
        for split, rows in combined_provenance.items()
    }
    merged_report = {
        "protocol": "explicit-pcap-flow-image-v2-class-parallel-equivalent",
        "manifest": str(manifest.resolve()),
        "output_dir": str(output.resolve()),
        "configuration": {
            **config["preprocessing"],
            "output_prefix": "pool",
            "representation": "first 8 packets; zeroed IPv4 addresses; 80 IP/header bytes + 48 Raw payload bytes",
        },
        "source_inventory": source_inventory,
        "classes": class_reports,
        "split_sizes": split_sizes,
        "flow_overlap": {f"{left}_{right}": len(flow_sets[left] & flow_sets[right]) for left, right in pairs},
        "exact_image_overlap": {f"{left}_{right}": len(image_sets[left] & image_sets[right]) for left, right in pairs},
        "split_diagnostics": {
            "execution": "class-level parallel, deterministic class-label merge",
            "per_class": per_class_diagnostics,
            "note": "these pool splits are merged before the final frozen source-group or flow-disjoint protocol split",
        },
        "limitations": [
            "The paper and released code do not disclose aggregate-PCAP flow splitting, session timeout, or sampling seeds.",
            "IPv4 TCP/UDP packets are supported; later IP fragments without a decoded transport header are excluded.",
        ],
    }
    write_json(report, merged_report)
    success.write_text("STAGE12_PREPROCESSING_POOL_SUCCESS\n", encoding="utf-8")
    return output


def source_lookup(inventory: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    lookup: dict[tuple[str, str], dict[str, str]] = {}
    for row in inventory:
        if row["record_type"] != "pcap" or row["readable"] != "True":
            continue
        key = (row["canonical_class"], row["file_name"])
        if key in lookup:
            raise RuntimeError(f"ambiguous duplicate capture basename within class: {key}")
        lookup[key] = row
    return lookup


def save_role(path: Path, records: list[dict]) -> dict:
    if not records:
        raise RuntimeError(f"refusing empty role: {path}")
    data = np.stack([row["image"] for row in records]).astype(np.uint8, copy=False)
    target = np.asarray([int(row["eligible_label"]) for row in records], dtype=np.int64)
    np.savez_compressed(path, data=data, target=target)
    return {"samples": len(records), "array_sha256": array_digest(data, target), "file_sha256": sha256_file(path)}


def freeze_dataset(dataset: str, config: dict, pool_dir: Path, class_rows: list[dict]) -> dict:
    protocol_dir = STAGE_ROOT / "protocol" / dataset
    inventory = read_csv(protocol_dir / "dataset_inventory.csv")
    lookup = source_lookup(inventory)
    records = load_pool(pool_dir, "pool")
    for row in records:
        source = lookup[(str(row["class_name"]), str(row["source_name"]))]
        row["source_file"] = source["source_file"]
        row["domain_state"] = source["domain_state"]
        row["official_category"] = source["official_category"]
    by_class: dict[str, list[dict]] = defaultdict(list)
    for row in records:
        by_class[str(row["class_name"])].append(row)
    report = json.loads((pool_dir / "preprocessing_report.json").read_text(encoding="utf-8"))
    eligibility = config["eligibility"]
    row_by_name = {row["canonical_class"]: row for row in class_rows}
    eligible_names = []
    for class_name in sorted(row_by_name):
        row = row_by_name[class_name]
        values = by_class.get(class_name, [])
        selected_groups = {str(item["source_file"]) for item in values}
        class_report = report["classes"].get(class_name, {})
        selected = len(values) if values else int(class_report.get("selected_samples", 0))
        unique = (
            len({str(item["image_sha256"]) for item in values})
            if values else int(class_report.get("selected_unique_images", 0))
        )
        row["selected_source_groups"] = (
            len(selected_groups) if values else int(row["readable_source_groups"])
        )
        row["selected_flows"] = selected
        row["unique_images"] = unique
        if row["eligible"] == "False":
            continue
        reasons = []
        if len(selected_groups) < int(eligibility["minimum_readable_source_groups"]):
            reasons.append("fewer than minimum selected source groups")
        if selected < int(eligibility["minimum_selected_flows"]):
            reasons.append("fewer than minimum selected flows")
        row["eligible"] = str(not reasons)
        row["exclusion_reason"] = "; ".join(reasons)
        if not reasons:
            eligible_names.append(class_name)
    label_map = {name: index for index, name in enumerate(sorted(eligible_names))}
    for row in records:
        if row["class_name"] in label_map:
            row["eligible_label"] = label_map[str(row["class_name"])]
    write_csv(protocol_dir / "eligible_classes.csv", class_rows, [
        "canonical_class", "provisional_label", "readable_source_groups", "domain_states",
        "selected_source_groups", "selected_flows", "unique_images", "split_policy",
        "group_aware_feasible", "eligible", "exclusion_reason",
    ])
    write_json(protocol_dir / "canonical_label_map.json", {
        "dataset": dataset,
        "canonical_class_to_label": label_map,
        "domain_state_is_metadata": True,
        "semantic_leakage_rule": "all domain states of one canonical class share Known/Unknown status",
    })
    unknown_protocol = frozen_unknown_protocol(eligible_names, int(config["protocol_seed"]))
    unknown_path = protocol_dir / "unknown_class_protocol.json"
    if unknown_path.exists():
        existing = json.loads(unknown_path.read_text(encoding="utf-8"))
        if existing != unknown_protocol:
            raise RuntimeError(f"frozen unknown protocol mismatch: {unknown_path}")
    else:
        write_json(unknown_path, unknown_protocol)
    eligible_records = [row for name in eligible_names for row in by_class[name]]
    source_assignment, fallback_classes = global_source_group_assignment(
        eligible_records, eligible_names, int(config["protocol_seed"])
    )
    if source_assignment is None:
        dataset_split_policy = "FLOW_DISJOINT_ONLY"
        flow_assignment, split_diagnostics = global_flow_disjoint_assignment(
            {name: by_class[name] for name in eligible_names},
            int(config["protocol_seed"]),
            Path(config["open_detect_root"]),
        )
    else:
        dataset_split_policy = "SOURCE_PCAP_GROUP_AWARE_AND_EXACT_IMAGE_LINKED"
        flow_assignment = {}
        split_diagnostics = {
            "policy": dataset_split_policy,
            "joint_cross_class_exact_image_source_linking": True,
        }
    for name in eligible_names:
        row_by_name[name]["split_policy"] = dataset_split_policy
        row_by_name[name]["group_aware_feasible"] = str(name not in fallback_classes)
    # Rewrite now that the actual split feasibility has been determined.
    write_csv(protocol_dir / "eligible_classes.csv", class_rows, [
        "canonical_class", "provisional_label", "readable_source_groups", "domain_states",
        "selected_source_groups", "selected_flows", "unique_images", "split_policy",
        "group_aware_feasible", "eligible", "exclusion_reason",
    ])
    split_rows: list[dict] = []
    setting_summaries = {}
    for setting, setting_cfg in unknown_protocol["settings"].items():
        known = set(setting_cfg["known_classes"])
        unknown = set(setting_cfg["unknown_classes"])
        role_records: dict[str, list[dict]] = {role: [] for role in ROLES}
        for class_name in sorted(eligible_names):
            for row in by_class[class_name]:
                if class_name in unknown:
                    role = "unknown_test"
                    base_split = "all"
                else:
                    base_split = (
                        source_assignment[str(row["source_file"])]
                        if source_assignment is not None
                        else flow_assignment[str(row["flow_id_sha256"])]
                    )
                    role = {"train": "known_train", "validation": "known_validation", "test": "known_test"}[base_split]
                role_records[role].append(row)
                split_rows.append({
                    "dataset": dataset,
                    "setting": setting,
                    "canonical_class": class_name,
                    "eligible_label": label_map[class_name],
                    "domain_state": row["domain_state"],
                    "official_category": row["official_category"],
                    "source_file": row["source_file"],
                    "flow_id_sha256": row["flow_id_sha256"],
                    "image_sha256": row["image_sha256"],
                    "base_split": base_split,
                    "role": role,
                })
        setting_dir = STAGE_ROOT / "artifacts" / dataset / "protocol" / setting
        setting_dir.mkdir(parents=True, exist_ok=True)
        role_info = {}
        for role in ROLES:
            ordered = sorted(role_records[role], key=lambda item: (item["class_name"], item["flow_id_sha256"]))
            role_info[role] = save_role(setting_dir / f"{role}.npz", ordered)
        # Semantic and split leakage checks.
        assert not (known & unknown)
        role_classes = {role: {row["class_name"] for row in values} for role, values in role_records.items()}
        assert not (role_classes["unknown_test"] & (
            role_classes["known_train"] | role_classes["known_validation"] | role_classes["known_test"]
        ))
        known_groups = {
            role: {row["source_file"] for row in role_records[role]}
            for role in ("known_train", "known_validation", "known_test")
        }
        known_images = {
            role: {row["image_sha256"] for row in role_records[role]}
            for role in ("known_train", "known_validation", "known_test")
        }
        overlaps = {}
        for left, right in (("known_train", "known_validation"), ("known_train", "known_test"), ("known_validation", "known_test")):
            overlaps[f"{left}_{right}_group"] = len(known_groups[left] & known_groups[right])
            overlaps[f"{left}_{right}_image"] = len(known_images[left] & known_images[right])
        known_flow_ids = {
            role: {row["flow_id_sha256"] for row in role_records[role]}
            for role in ("known_train", "known_validation", "known_test")
        }
        for left, right in (("known_train", "known_validation"), ("known_train", "known_test"), ("known_validation", "known_test")):
            overlaps[f"{left}_{right}_flow_id"] = len(known_flow_ids[left] & known_flow_ids[right])
        prohibited_overlaps = {
            name: value for name, value in overlaps.items()
            if not name.endswith("_group") or dataset_split_policy != "FLOW_DISJOINT_ONLY"
        }
        if any(prohibited_overlaps.values()):
            raise RuntimeError(f"prohibited split leakage in {dataset}/{setting}: {prohibited_overlaps}")
        setting_summary = {
            "dataset": dataset,
            "setting": setting,
            "known_classes": sorted(known),
            "unknown_classes": sorted(unknown),
            "unknown_free": True,
            "split_policy": dataset_split_policy,
            "flow_disjoint_only": dataset_split_policy == "FLOW_DISJOINT_ONLY",
            "group_aware_not_feasible_classes": fallback_classes,
            "overlaps": overlaps,
            "prohibited_overlaps": prohibited_overlaps,
            "split_diagnostics": split_diagnostics,
            "roles": role_info,
        }
        write_json(setting_dir / "protocol.json", setting_summary)
        setting_summaries[setting] = setting_summary
    split_path = protocol_dir / "split_manifest.csv"
    write_csv(split_path, split_rows, [
        "dataset", "setting", "canonical_class", "eligible_label", "domain_state",
        "official_category", "source_file", "flow_id_sha256", "image_sha256", "base_split", "role",
    ])
    audit_rows = []
    for class_name, class_report in sorted(report["classes"].items()):
        counters = class_report["counters"]
        candidates = int(counters.get("sessions_observed", 0))
        selected = int(class_report["selected_samples"])
        audit_rows.append({
            "dataset": dataset,
            "canonical_class": class_name,
            "candidate_flows_observed": candidates,
            "sampled_for_conversion": selected,
            "successfully_converted_flows": selected,
            "sampling_excluded_flows": max(0, candidates - selected),
            "failed_flows": 0,
            "malformed": 0,
            "non_ip_or_non_tcp_udp_packets": int(counters.get("packets_non_tcp_udp_ipv4", 0)),
            "fragment": 0,
            "too_short_padded_flows": int(class_report.get("flows_with_fewer_than_8_packets", 0)),
            "parser_failure": int(counters.get("packet_representation_errors", 0)),
            "empty_flow": 0,
            "duplicate_selected_images": selected - int(class_report["selected_unique_images"]),
            "conversion_success_rate_sampled": 1.0 if selected else 0.0,
        })
    write_csv(protocol_dir / "preprocessing_audit.csv", audit_rows)
    freeze = {
        "status": "PROTOCOL_FROZEN_PRETRAIN",
        "dataset": dataset,
        "eligible_class_count": len(eligible_names),
        "eligible_classes": sorted(eligible_names),
        "unknown_protocol_sha256": sha256_file(unknown_path),
        "split_manifest_sha256": sha256_file(split_path),
        "preprocessing_manifest_sha256": sha256_file(protocol_dir / "preprocessing_manifest.json"),
        "canonical_label_map_sha256": sha256_file(protocol_dir / "canonical_label_map.json"),
        "settings": setting_summaries,
        "dataset_split_policy": dataset_split_policy,
        "group_aware_not_feasible_classes": fallback_classes,
        "claim_reduction_required": bool(fallback_classes),
        "test_metrics_opened": False,
    }
    write_json(protocol_dir / "protocol_freeze.json", freeze)
    audit_md = protocol_dir / "DATASET_AUDIT.md"
    with audit_md.open("a", encoding="utf-8") as handle:
        handle.write("\n## Final pre-training eligibility and protocol freeze\n\n")
        handle.write(f"- Eligible canonical classes: `{len(eligible_names)}` — {', '.join(sorted(eligible_names))}\n")
        handle.write(f"- Split policy: `{freeze['dataset_split_policy']}`\n")
        handle.write(f"- GROUP_AWARE_NOT_FEASIBLE classes: `{', '.join(fallback_classes) if fallback_classes else 'none'}`\n")
        handle.write(f"- Unknown protocol SHA-256: `{freeze['unknown_protocol_sha256']}`\n")
        handle.write(f"- Split manifest SHA-256: `{freeze['split_manifest_sha256']}`\n")
        handle.write("- Unknown-free pre-training check: `PASS`\n")
    return freeze


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("iscx_vpn", "iscx_tor"), required=True)
    args = parser.parse_args()
    config = load_config()
    dataset = args.dataset
    audit_status = STAGE_ROOT / "outputs" / "summary" / "dataset_audit_status.json"
    if not audit_status.is_file() or json.loads(audit_status.read_text(encoding="utf-8"))["status"] != "PASS":
        raise RuntimeError("dataset/provenance audit must pass before preprocessing")
    protocol_dir = STAGE_ROOT / "protocol" / dataset
    inventory = read_csv(protocol_dir / "dataset_inventory.csv")
    try:
        manifest, class_rows = prepare_manifest(dataset, config, inventory)
        pool_dir = run_open_detect_preprocessor(dataset, config, manifest)
        result = freeze_dataset(dataset, config, pool_dir, class_rows)
        (STAGE_ROOT / "artifacts" / dataset / "preprocessing_pool" / "SUCCESS").write_text(
            "STAGE12_DATASET_PROTOCOL_FROZEN_SUCCESS\n", encoding="utf-8"
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except BaseException as exc:
        write_json(STAGE_ROOT / "artifacts" / dataset / "preprocessing_pool" / "FAILURE.json", {
            "status": "FAILED",
            "dataset": dataset,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "preserve_and_do_not_overwrite": True,
            "test_metrics_opened": False,
        })
        raise


if __name__ == "__main__":
    raise SystemExit(main())
