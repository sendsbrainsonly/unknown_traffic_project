#!/usr/bin/env python3
"""Build Open-Detect images for the immutable historical Stage-1 flow IDs."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap


AUDIT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = AUDIT_ROOT.parent
sys.path.insert(0, str(AUDIT_ROOT))

from adapters.opendetect_preprocessing import encode_flow  # noqa: E402


SPLITS = ("train", "val", "test")


def sha256(path: Path, block_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def load_label_map(path: Path) -> tuple[dict[str, int], dict[int, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "class_to_id" in payload:
        forward = {str(k): int(v) for k, v in payload["class_to_id"].items()}
    elif "label_to_id" in payload:
        forward = {str(k): int(v) for k, v in payload["label_to_id"].items()}
    else:
        forward = {str(k): int(v) for k, v in payload.items()}
    reverse = {value: key for key, value in forward.items()}
    if len(forward) != 20 or len(reverse) != 20:
        raise ValueError("historical label map is not a bijective 20-class map")
    return forward, reverse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flows-dir", type=Path, default=PROJECT_ROOT / "data/flows")
    parser.add_argument(
        "--embedding-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/stage1/modelA/embeddings",
    )
    parser.add_argument(
        "--label-map",
        type=Path,
        default=PROJECT_ROOT / "data/fig_graph/all_flows/label_map.json",
    )
    parser.add_argument("--artifacts-dir", type=Path, default=AUDIT_ROOT / "artifacts")
    parser.add_argument(
        "--manifest", type=Path, default=AUDIT_ROOT / "outputs/input_alignment_manifest.csv"
    )
    parser.add_argument(
        "--summary", type=Path, default=AUDIT_ROOT / "outputs/input_alignment_summary.json"
    )
    args = parser.parse_args()

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    class_to_id, id_to_class = load_label_map(args.label_map)

    ids: dict[str, np.ndarray] = {}
    labels: dict[str, np.ndarray] = {}
    rows: dict[str, tuple[str, int, int]] = {}
    arrays: dict[str, np.memmap] = {}
    status: dict[str, dict[str, object]] = {}

    for split in SPLITS:
        ids[split] = np.load(args.embedding_dir / f"flow_ids_{split}.npy", allow_pickle=False)
        labels[split] = np.load(args.embedding_dir / f"labels_{split}.npy", allow_pickle=False)
        if len(ids[split]) != len(labels[split]):
            raise ValueError(f"{split}: flow IDs and labels differ in length")
        arrays[split] = open_memmap(
            args.artifacts_dir / f"{split}_images.npy",
            mode="w+",
            dtype=np.uint8,
            shape=(len(ids[split]), 32, 32),
        )
        for index, (flow_id, label) in enumerate(zip(ids[split].tolist(), labels[split].tolist())):
            flow_id = str(flow_id)
            if flow_id in rows:
                raise ValueError(f"duplicate flow ID across splits: {flow_id}")
            label = int(label)
            rows[flow_id] = (split, index, label)
            status[flow_id] = {
                "flow_id": flow_id,
                "class_name": id_to_class[label],
                "original_split": split,
                "source_file": "",
                "opendetect_input_id": f"{split}:{index}",
                "input_valid": False,
                "packets_found": 0,
                "packets_used": 0,
                "failure_reason": "not_found_in_stage0_pkls",
            }

    requested = len(rows)
    seen = set()
    counters = Counter()
    by_group: dict[tuple[str, str], Counter] = defaultdict(Counter)

    for pkl_path in sorted(args.flows_dir.glob("*.pkl")):
        print(f"loading {pkl_path.name}", flush=True)
        with pkl_path.open("rb") as handle:
            payload = pickle.load(handle)
        class_name = str(payload["class_name"])
        source_file = str(payload["source_file"])
        if class_name not in class_to_id:
            raise ValueError(f"unexpected Stage-0 class {class_name!r} in {pkl_path}")
        for flow_id, packets in payload["packets"].items():
            if flow_id not in rows:
                counters["unrequested_stage0_flow"] += 1
                continue
            if flow_id in seen:
                raise ValueError(f"duplicate Stage-0 flow ID: {flow_id}")
            seen.add(flow_id)
            split, index, expected_label = rows[flow_id]
            expected_class = id_to_class[expected_label]
            record = status[flow_id]
            record["source_file"] = source_file
            record["packets_found"] = len(packets)
            if expected_class != class_name:
                record["failure_reason"] = (
                    f"class_mismatch:stage1={expected_class};stage0={class_name}"
                )
                counters["failed"] += 1
                by_group[(expected_class, split)]["failed"] += 1
                continue
            try:
                encoded = encode_flow(packets)
                arrays[split][index] = encoded.image
                record["input_valid"] = True
                record["packets_found"] = encoded.packets_found
                record["packets_used"] = encoded.packets_used
                record["failure_reason"] = ""
                counters["success"] += 1
                by_group[(class_name, split)]["success"] += 1
            except Exception as exc:  # surfaced in the manifest; formal gate will fail
                record["failure_reason"] = f"{type(exc).__name__}:{exc}"
                counters["failed"] += 1
                by_group[(class_name, split)]["failed"] += 1
        del payload
        gc.collect()

    missing = set(rows) - seen
    counters["failed"] += len(missing)
    for flow_id in missing:
        split, _, label = rows[flow_id]
        by_group[(id_to_class[label], split)]["failed"] += 1

    for array in arrays.values():
        array.flush()

    fields = [
        "flow_id",
        "class_name",
        "original_split",
        "source_file",
        "opendetect_input_id",
        "input_valid",
        "packets_found",
        "packets_used",
        "failure_reason",
    ]
    with args.manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for split in SPLITS:
            for flow_id in ids[split].tolist():
                writer.writerow(status[str(flow_id)])

    group_rows = []
    for class_name in sorted(class_to_id, key=class_to_id.get):
        for split in SPLITS:
            group = by_group[(class_name, split)]
            count = int(np.sum(labels[split] == class_to_id[class_name]))
            success = int(group["success"])
            failed = count - success
            group_rows.append(
                {
                    "class_name": class_name,
                    "split": split,
                    "requested": count,
                    "success": success,
                    "failed": failed,
                    "success_rate": success / count if count else None,
                }
            )

    success = int(counters["success"])
    failed = requested - success
    summary = {
        "requested": requested,
        "success": success,
        "failed": failed,
        "success_rate": success / requested,
        "all_inputs_valid": failed == 0,
        "class_split_statistics": group_rows,
        "label_map": class_to_id,
        "source_paths": {
            "flows_dir": str(args.flows_dir.resolve()),
            "embedding_dir": str(args.embedding_dir.resolve()),
            "label_map": str(args.label_map.resolve()),
        },
        "artifacts": {
            split: {
                "images": str((args.artifacts_dir / f"{split}_images.npy").resolve()),
                "flow_ids": str((args.embedding_dir / f"flow_ids_{split}.npy").resolve()),
                "labels": str((args.embedding_dir / f"labels_{split}.npy").resolve()),
            }
            for split in SPLITS
        },
        "manifest": str(args.manifest.resolve()),
    }
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("requested", "success", "failed", "success_rate", "all_inputs_valid")}, indent=2))
    if failed:
        raise SystemExit("alignment gate failed; formal training is forbidden")


if __name__ == "__main__":
    main()
