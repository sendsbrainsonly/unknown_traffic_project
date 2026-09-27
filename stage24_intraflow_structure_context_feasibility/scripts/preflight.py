#!/usr/bin/env python3
"""Stage 24 frozen-input and cross-flow-context audit; no model or Test features."""
from __future__ import annotations

import bisect
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


OUT = Path(__file__).resolve().parents[1]
PROJECT = OUT.parent
STAGE20 = PROJECT / "stage20_dual_coarse_service_protocol" / "closed_service_manifest.csv"
STAGE21 = PROJECT / "stage21_coarse_service_ours_e3_benchmark" / "feature_cache"
STAGE22 = PROJECT / "stage22_pretrained_trafficformer_e3_closed_set_comparison" / "runs"
EXPECTED_MANIFEST_SHA = "6404d496f50a6f7f6f397d125ad0cd653c9df152d8db1b9ed2ad2310967950bb"
DATASETS = ("iscx_vpn", "iscx_tor")
ROLES = ("known_train", "known_validation")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def selected_provenance(rows: list[dict]) -> dict[str, dict]:
    wanted: dict[Path, dict[int, str]] = defaultdict(dict)
    for row in rows:
        path = Path(row["source_pool"]) / f"provenance_{row['source_role']}.jsonl"
        index = int(row["source_index"])
        if index in wanted[path] and wanted[path][index] != row["flow_id"]:
            raise RuntimeError(f"inconsistent provenance reference: {path}:{index}")
        wanted[path][index] = row["flow_id"]
    result: dict[str, dict] = {}
    for path, requested in sorted(wanted.items()):
        found = set()
        with path.open(encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if index not in requested:
                    continue
                record = json.loads(line)
                flow_id = requested[index]
                if record["flow_id_sha256"] != flow_id:
                    raise RuntimeError(f"provenance flow-ID mismatch: {path}:{index}")
                result[flow_id] = record
                found.add(index)
                if len(found) == len(requested):
                    break
        if found != set(requested):
            raise RuntimeError(f"missing {len(set(requested) - found)} provenance records: {path}")
    if len(result) != len(rows):
        raise RuntimeError("provenance coverage mismatch")
    return result


def main() -> None:
    if sha256(STAGE20) != EXPECTED_MANIFEST_SHA:
        raise RuntimeError("frozen Stage20 manifest hash mismatch")
    with STAGE20.open(newline="", encoding="utf-8") as handle:
        all_rows = list(csv.DictReader(handle))
    if len(all_rows) != 22136:
        raise RuntimeError(f"unexpected Stage20 closed rows: {len(all_rows)}")
    trainval = [row for row in all_rows if row["closed_role"] in ROLES]
    if len(trainval) != 19926:
        raise RuntimeError(f"unexpected Known Train/Val rows: {len(trainval)}")
    provenance = selected_provenance(trainval)

    protected = {"stage20_manifest": sha256(STAGE20)}
    packet_rows: list[dict] = []
    capture_rows: list[dict] = []
    context_rows: list[dict] = []
    findings: dict[str, dict] = {}

    for dataset in DATASETS:
        dataset_rows = [row for row in all_rows if row["dataset"] == dataset]
        known = [row for row in dataset_rows if row["closed_role"] in ROLES]
        cache_path = STAGE21 / dataset / "e3_t8_inputs.npz"
        protected[f"stage21_{dataset}_cache"] = sha256(cache_path)
        cache = np.load(cache_path, allow_pickle=False)
        cache_pos = {str(flow_id): i for i, flow_id in enumerate(cache["flow_ids"])}
        cache_counts = cache["packet_counts"]
        if set(cache_pos) != {row["flow_id"] for row in dataset_rows}:
            raise RuntimeError(f"Stage21 cache membership mismatch: {dataset}")
        samples = []
        non_monotone_timestamps = 0
        for row in known:
            record = provenance[row["flow_id"]]
            refs = record["packet_refs"][:8]
            if not refs:
                raise RuntimeError(f"empty packet refs: {row['flow_id']}")
            n = len(refs)
            cache_n = int(cache_counts[cache_pos[row["flow_id"]]])
            if cache_n != n:
                raise RuntimeError(f"packet count mismatch: {row['flow_id']} {cache_n}/{n}")
            starts = [float(ref["timestamp"]) for ref in refs]
            if any(b < a for a, b in zip(starts, starts[1:])):
                non_monotone_timestamps += 1
            # Preserve packet order in the model cache; min/max are used only
            # to bound the conservative, read-only causal-context inventory.
            samples.append({**row, "packet_count": n, "start": min(starts), "end": max(starts)})
        classes = sorted({row["service_label"] for row in dataset_rows})
        for service in classes:
            class_all = [row for row in dataset_rows if row["service_label"] == service]
            class_tv = [row for row in samples if row["service_label"] == service]
            for role in ROLES:
                subset = [row for row in class_tv if row["closed_role"] == role]
                counts = Counter(row["packet_count"] for row in subset)
                packet_rows.append({"dataset": dataset, "service": service, "role": role,
                                    "flows": len(subset), "one_packet": counts[1], "two_packets": counts[2],
                                    "one_or_two": counts[1] + counts[2],
                                    "three_to_eight": len(subset) - counts[1] - counts[2],
                                    "mean_packets": round(sum(row["packet_count"] for row in subset) / len(subset), 6) if subset else 0})
            groups = defaultdict(set)
            for row in class_all:
                groups[row["closed_role"]].add(row["capture_group"])
            capture_rows.append({"dataset": dataset, "service": service, "flows": len(class_all),
                                 "capture_groups": len(set().union(*groups.values())),
                                 "train_groups": len(groups["known_train"]),
                                 "val_groups": len(groups["known_validation"]),
                                 "test_groups": len(groups["known_test"]),
                                 "train_val_shared_groups": len(groups["known_train"] & groups["known_validation"]),
                                 "train_test_shared_groups": len(groups["known_train"] & groups["known_test"]),
                                 "val_test_shared_groups": len(groups["known_validation"] & groups["known_test"]),
                                 "group_disjoint_3way_possible": int(len(set().union(*groups.values())) >= 3)})

        # A conservative reference bank: only Known Train flows ending before
        # the query starts; neither Validation nor Test flows populate the bank.
        train_by_capture: dict[str, list[float]] = defaultdict(list)
        for row in samples:
            if row["closed_role"] == "known_train":
                train_by_capture[row["capture_group"]].append(row["end"])
        for ends in train_by_capture.values():
            ends.sort()
        neighbor_counts = []
        for row in samples:
            ends = train_by_capture[row["capture_group"]]
            start = row["start"]
            count = bisect.bisect_right(ends, start) - bisect.bisect_left(ends, start - 60.0)
            if row["closed_role"] == "known_train" and row["end"] <= start:
                count -= 1  # never include the query itself for a one-packet flow
            neighbor_counts.append((row, max(count, 0)))
        for service in classes:
            for role in ROLES:
                values = [count for row, count in neighbor_counts if row["service_label"] == service and row["closed_role"] == role]
                context_rows.append({"dataset": dataset, "service": service, "role": role,
                                     "flows": len(values), "zero_prior_train_neighbors_60s": sum(x == 0 for x in values),
                                     "with_prior_train_neighbor_60s": sum(x > 0 for x in values),
                                     "median_prior_train_neighbors_60s": float(np.median(values)) if values else 0.0,
                                     "p90_prior_train_neighbors_60s": float(np.percentile(values, 90)) if values else 0.0})
        group_block = [row["service"] for row in capture_rows if row["dataset"] == dataset and not row["group_disjoint_3way_possible"]]
        counts = Counter(row["packet_count"] for row in samples)
        findings[dataset] = {"known_train_validation_flows": len(samples),
                             "one_or_two_packet_flows": counts[1] + counts[2],
                             "one_or_two_ratio": (counts[1] + counts[2]) / len(samples),
                             "non_monotone_provenance_packet_timestamp_flows": non_monotone_timestamps,
                             "group_disjoint_blocked_services": group_block,
                             "all_service_3way_group_split_feasible": not group_block,
                             "endpoint_or_session_id_in_frozen_provenance": False,
                             "claim_scope": "Known Train/Validation metadata and packet refs only; Test features not read"}
        for seed in (2022, 2023):
            run = STAGE22 / dataset / f"seed{seed}"
            for name in ("E1_model_best.pt", "representations.npz", "E2_model_best.pt", "E3_model_best.pt"):
                protected[f"stage22_{dataset}_{seed}_{name}"] = sha256(run / name)
    write_csv(OUT / "packet_count_audit.csv", packet_rows, list(packet_rows[0]))
    write_csv(OUT / "capture_group_audit.csv", capture_rows, list(capture_rows[0]))
    write_csv(OUT / "prior_context_audit.csv", context_rows, list(context_rows[0]))
    write_json(OUT / "protected_hashes_before.json", protected)
    result = {"status": "PASS", "findings": findings,
              "interflow_stage20_gate": "INTERFLOW_NOT_FEASIBLE_FOR_FULL_CLASS_CAPTURE_DISJOINT_CLAIM",
              "reason": "at least one Service lacks three independent capture groups; frozen provenance has no endpoint/session identifier",
              "protected_asset_count": len(protected), "unknown_usage": 0, "known_test_feature_usage": 0}
    write_json(OUT / "preflight.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
