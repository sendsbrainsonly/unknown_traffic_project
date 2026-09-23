#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


OUT = Path(__file__).resolve().parents[1]
PROJECT = OUT.parent
CONFIG_PATH = OUT / "config.json"
ROLES = ("known_train", "known_validation", "known_test")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_key(*parts: object) -> str:
    return hashlib.sha256("\0".join(map(str, parts)).encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def capture_filename(row: dict) -> str:
    refs = row.get("packet_refs") or []
    if not refs:
        raise ValueError(f"flow has no packet_refs: {row.get('flow_id_sha256')}")
    captures = {str(ref["capture"]) for ref in refs}
    if len(captures) != 1:
        raise ValueError(f"flow spans multiple capture identifiers: {captures}")
    token = next(iter(captures))
    if "|name=" not in token:
        raise ValueError(f"unsupported capture identifier: {token}")
    return token.rsplit("|name=", 1)[1]


def inventory_mapping(path: Path, normalization: dict[str, str]) -> dict[tuple[str, str], dict]:
    mapping: dict[tuple[str, str], dict] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["record_type"] != "pcap" or row["readable"] != "True":
                continue
            service = normalization.get(row["official_category"], row["official_category"])
            key = (row["canonical_class"], row["file_name"])
            value = {
                "service_label": service,
                "official_category": row["official_category"],
                "domain_state": row["domain_state"],
                "source_file": row["source_file"],
                "capture_group": row["group_id"],
            }
            if key in mapping and mapping[key] != value:
                raise ValueError(f"ambiguous inventory mapping: {key}")
            mapping[key] = value
    return mapping


def load_candidates(dataset: str, spec: dict, seed: int) -> tuple[list[dict], dict]:
    pool = (OUT / spec["source_pool"]).resolve()
    inventory_path = (OUT / spec["inventory_csv"]).resolve()
    mapping = inventory_mapping(inventory_path, spec["category_normalization"])
    candidates: list[dict] = []
    source_hashes = {str(inventory_path): sha256_file(inventory_path)}
    seen_flows: set[str] = set()
    for source_role in ("train", "validation", "test"):
        provenance_path = pool / f"provenance_{source_role}.jsonl"
        source_hashes[str(provenance_path)] = sha256_file(provenance_path)
        for source_index, row in enumerate(read_jsonl(provenance_path)):
            flow_id = str(row["flow_id_sha256"])
            if flow_id in seen_flows:
                raise ValueError(f"duplicate flow id in source pool: {dataset}/{flow_id}")
            seen_flows.add(flow_id)
            file_name = capture_filename(row)
            key = (str(row["class_name"]), file_name)
            if key not in mapping:
                raise KeyError(f"missing inventory mapping: {dataset}/{key}")
            meta = mapping[key]
            service = meta["service_label"]
            if service not in spec["services"]:
                raise ValueError(f"unexpected normalized service: {dataset}/{service}")
            candidates.append({
                "dataset": dataset,
                "flow_id": flow_id,
                "image_sha256": str(row["image_sha256"]),
                "fine_application": str(row["class_name"]),
                "service_label": service,
                "official_category": meta["official_category"],
                "domain_state": meta["domain_state"],
                "capture_group": meta["capture_group"],
                "source_file": meta["source_file"],
                "source_pool": str(pool),
                "source_role": source_role,
                "source_index": source_index,
                "selection_key": stable_key(seed, dataset, service, flow_id),
            })
    return candidates, source_hashes


def select_and_split(
    candidates: list[dict], services: list[str], cap: int, seed: int
) -> tuple[list[dict], dict]:
    selected: list[dict] = []
    image_services: dict[str, set[str]] = defaultdict(set)
    for row in candidates:
        image_services[row["image_sha256"]].add(row["service_label"])
    conflicting_images = {
        image_sha256 for image_sha256, labels in image_services.items() if len(labels) > 1
    }
    conflict_rows = [row for row in candidates if row["image_sha256"] in conflicting_images]
    candidates = [row for row in candidates if row["image_sha256"] not in conflicting_images]
    by_service: dict[str, list[dict]] = defaultdict(list)
    for row in candidates:
        by_service[row["service_label"]].append(row)
    if set(by_service) != set(services):
        raise ValueError(f"service coverage mismatch: got={sorted(by_service)} expected={services}")

    for service in services:
        rows = sorted(by_service[service], key=lambda row: row["selection_key"])[:cap]
        groups: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            groups[row["image_sha256"]].append(row)
        ordered_groups = sorted(
            groups.items(), key=lambda item: stable_key(seed, "split", service, item[0])
        )
        total = sum(len(group_rows) for _, group_rows in ordered_groups)
        train_target = round(total * 0.8)
        validation_target = round(total * 0.1)
        counts = Counter()
        for _, group_rows in ordered_groups:
            if counts["known_train"] < train_target:
                role = "known_train"
            elif counts["known_validation"] < validation_target:
                role = "known_validation"
            else:
                role = "known_test"
            for row in group_rows:
                materialized = dict(row)
                materialized["closed_role"] = role
                selected.append(materialized)
            counts[role] += len(group_rows)
        if any(counts[role] == 0 for role in ROLES):
            raise ValueError(f"empty role after split: {service}/{dict(counts)}")
    return selected, {
        "cross_service_duplicate_image_groups_excluded": len(conflicting_images),
        "cross_service_duplicate_rows_excluded": len(conflict_rows),
        "excluded_rows_by_service": dict(sorted(Counter(
            row["service_label"] for row in conflict_rows
        ).items())),
    }


def protocol_rows(dataset: str, selected: list[dict], services: list[str]) -> list[dict]:
    result: list[dict] = []
    label_map = {service: index for index, service in enumerate(services)}
    for unknown_service in services:
        protocol_id = f"{dataset}_loso_{unknown_service.lower().replace('-', '_')}"
        known_services = [service for service in services if service != unknown_service]
        known_label_map = {service: index for index, service in enumerate(known_services)}
        for row in selected:
            is_unknown = row["service_label"] == unknown_service
            result.append({
                "protocol_id": protocol_id,
                "dataset": dataset,
                "unknown_service": unknown_service,
                "known_services": "|".join(known_services),
                "role": "unknown_test" if is_unknown else row["closed_role"],
                "class_role": "unknown" if is_unknown else "known",
                "flow_id": row["flow_id"],
                "image_sha256": row["image_sha256"],
                "service_label": row["service_label"],
                "service_label_id": label_map[row["service_label"]],
                "known_label_id": "" if is_unknown else known_label_map[row["service_label"]],
                "fine_application": row["fine_application"],
                "official_category": row["official_category"],
                "domain_state": row["domain_state"],
                "capture_group": row["capture_group"],
                "source_file": row["source_file"],
                "source_pool": row["source_pool"],
                "source_role": row["source_role"],
                "source_index": row["source_index"],
                "label_scope": "WEAK_CAPTURE_LABEL",
                "split_scope": "FLOW_AND_EXACT_IMAGE_DISJOINT_NOT_CAPTURE_DISJOINT",
            })
    return result


def audit_protocols(rows: list[dict]) -> dict:
    protocols: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        protocols[row["protocol_id"]].append(row)
    summaries = []
    failures = []
    for protocol_id, protocol in sorted(protocols.items()):
        by_role = {role: [row for row in protocol if row["role"] == role] for role in (*ROLES, "unknown_test")}
        flow_sets = {role: {row["flow_id"] for row in values} for role, values in by_role.items()}
        image_sets = {role: {row["image_sha256"] for row in values} for role, values in by_role.items()}
        flow_overlaps = {}
        image_overlaps = {}
        roles = list(by_role)
        for index, left in enumerate(roles):
            for right in roles[index + 1:]:
                flow_overlaps[f"{left}__{right}"] = len(flow_sets[left] & flow_sets[right])
                image_overlaps[f"{left}__{right}"] = len(image_sets[left] & image_sets[right])
        unknown_in_fitting = sum(
            row["class_role"] == "unknown" and row["role"] != "unknown_test" for row in protocol
        )
        summary = {
            "protocol_id": protocol_id,
            "dataset": protocol[0]["dataset"],
            "unknown_service": protocol[0]["unknown_service"],
            "role_counts": {role: len(values) for role, values in by_role.items()},
            "known_service_counts": dict(sorted(Counter(
                row["service_label"] for row in protocol if row["class_role"] == "known"
            ).items())),
            "unknown_count": len(by_role["unknown_test"]),
            "unknown_in_fitting": unknown_in_fitting,
            "flow_overlaps": flow_overlaps,
            "image_overlaps": image_overlaps,
        }
        if unknown_in_fitting or any(flow_overlaps.values()) or any(image_overlaps.values()):
            failures.append(protocol_id)
        if any(len(by_role[role]) == 0 for role in by_role):
            failures.append(protocol_id)
        summaries.append(summary)
    return {
        "status": "PASS" if not failures else "FAIL",
        "protocol_count": len(protocols),
        "failed_protocols": sorted(set(failures)),
        "protocols": summaries,
    }


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    all_manifest_rows: list[dict] = []
    closed_rows: list[dict] = []
    source_hashes: dict[str, str] = {str(CONFIG_PATH): sha256_file(CONFIG_PATH)}
    dataset_summaries = {}

    for dataset, spec in config["datasets"].items():
        candidates, hashes = load_candidates(dataset, spec, int(config["protocol_seed"]))
        source_hashes.update(hashes)
        selected, conflict_audit = select_and_split(
            candidates,
            list(spec["services"]),
            int(config["max_flows_per_service"]),
            int(config["protocol_seed"]),
        )
        for row in selected:
            closed_rows.append({key: value for key, value in row.items() if key != "selection_key"})
        manifest = protocol_rows(dataset, selected, list(spec["services"]))
        all_manifest_rows.extend(manifest)
        dataset_summaries[dataset] = {
            "candidate_flows": len(candidates),
            "selected_flows": len(selected),
            "services": list(spec["services"]),
            "candidate_by_service": dict(sorted(Counter(row["service_label"] for row in candidates).items())),
            "selected_by_service": dict(sorted(Counter(row["service_label"] for row in selected).items())),
            "closed_role_counts": dict(sorted(Counter(row["closed_role"] for row in selected).items())),
            "captures_by_service": {
                service: len({row["capture_group"] for row in selected if row["service_label"] == service})
                for service in spec["services"]
            },
            "cross_service_duplicate_audit": conflict_audit,
        }

    closed_fields = [
        "dataset", "flow_id", "image_sha256", "fine_application", "service_label",
        "official_category", "domain_state", "capture_group", "source_file", "source_pool",
        "source_role", "source_index", "closed_role",
    ]
    manifest_fields = [
        "protocol_id", "dataset", "unknown_service", "known_services", "role", "class_role",
        "flow_id", "image_sha256", "service_label", "service_label_id", "known_label_id",
        "fine_application", "official_category", "domain_state", "capture_group", "source_file",
        "source_pool", "source_role", "source_index", "label_scope", "split_scope",
    ]
    write_csv(OUT / "closed_service_manifest.csv", closed_rows, closed_fields)
    write_csv(OUT / "loso_service_protocol_manifest.csv", all_manifest_rows, manifest_fields)
    audit = audit_protocols(all_manifest_rows)
    if audit["status"] != "PASS":
        write_json(OUT / "protocol_audit.json", audit)
        raise RuntimeError(f"protocol audit failed: {audit['failed_protocols']}")
    summary = {
        "experiment_id": config["experiment_id"],
        "status": "PROTOCOL_FROZEN_NOT_TRAINED",
        "task_decision": config["task_decision"],
        "dataset_summaries": dataset_summaries,
        "closed_manifest_rows": len(closed_rows),
        "loso_manifest_rows": len(all_manifest_rows),
        "protocol_count": audit["protocol_count"],
        "strict_unknown_free": True,
        "unknown_fitting_samples": 0,
        "source_hashes": dict(sorted(source_hashes.items())),
        "output_hashes": {},
        "limitations": config["limitations"],
    }
    write_json(OUT / "protocol_audit.json", audit)
    write_json(OUT / "protocol_summary.json", summary)
    for name in ("closed_service_manifest.csv", "loso_service_protocol_manifest.csv", "protocol_audit.json"):
        summary["output_hashes"][name] = sha256_file(OUT / name)
    write_json(OUT / "protocol_summary.json", summary)
    (OUT / "SUCCESS").write_text("PROTOCOL_FROZEN_NOT_TRAINED\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
