#!/usr/bin/env python3
"""Build exact Stage20 RoNeTC views, reusing Stage19 and recovering 137 Tor flows.

Raw PCAP and all earlier-stage assets are read-only. Output is new Stage23 data.
The packet-to-view definition is imported from the existing Stage19 adapter.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
from scapy.utils import RawPcapReader


OUT = Path(__file__).resolve().parents[1]
PROJECT = OUT.parent
STAGE19 = PROJECT / "stage19_ronetc_four_dataset_comparison"
STAGE12 = PROJECT / "stage12_dual_external_validation"
STAGE20 = PROJECT / "stage20_dual_coarse_service_protocol"
FROZEN_HASH = "6404d496f50a6f7f6f397d125ad0cd653c9df152d8db1b9ed2ad2310967950bb"
ROOT = OUT / "ronetc_stage20_cache"
sys.path.insert(0, str(STAGE19 / "scripts"))
from build_byte_cache import packet_views  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def refs_for_missing(rows: list[dict[str, str]]) -> dict[str, list[dict]]:
    targets = {row["flow_id"]: row for row in rows}
    records = {}
    for pool in sorted({Path(row["source_pool"]) for row in rows}):
        for path in sorted(pool.glob("provenance_*.jsonl")):
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    record = json.loads(line)
                    uid = record["flow_id_sha256"]
                    if uid not in targets:
                        continue
                    if uid in records or record["image_sha256"] != targets[uid]["image_sha256"]:
                        raise RuntimeError(f"duplicate or inconsistent provenance: {uid}")
                    refs = record["packet_refs"][:8]
                    if not refs or any(int(ref["packet_number"]) <= 0 for ref in refs):
                        raise RuntimeError(f"empty/invalid packet refs: {uid}")
                    records[uid] = refs
    if set(records) != set(targets):
        raise RuntimeError(f"missing provenance: {len(set(targets) - set(records))}")
    return records


def build_dataset(dataset: str, target: list[dict[str, str]], pcap_root: Path) -> dict:
    output = ROOT / dataset
    if output.exists():
        raise FileExistsError(f"preserve existing cache, refuse overwrite: {output}")
    prior = STAGE19 / "byte_cache" / dataset
    audit = json.loads((prior / "cache_audit.json").read_text(encoding="utf-8"))
    if audit["status"] != "PASS" or sha256(prior / "flow_uids.npy") != audit["flow_uids_sha256"] or sha256(prior / "views.npy") != audit["views_sha256"]:
        raise RuntimeError(f"Stage19 cache hash mismatch: {dataset}")
    old_ids = [str(value) for value in np.load(prior / "flow_uids.npy", mmap_mode="r", allow_pickle=False)]
    old_index = {uid: index for index, uid in enumerate(old_ids)}
    if len(old_index) != len(old_ids):
        raise RuntimeError("duplicate Stage19 flow IDs")
    old_views = np.load(prior / "views.npy", mmap_mode="r", allow_pickle=False)
    old_counts = np.load(prior / "packet_counts.npy", mmap_mode="r", allow_pickle=False)
    if old_views.shape != (len(old_ids), 3, 8, 64):
        raise RuntimeError("Stage19 view tensor shape changed")
    sorted_rows = sorted(target, key=lambda row: row["flow_id"])
    ids = [row["flow_id"] for row in sorted_rows]
    if len(set(ids)) != len(ids):
        raise RuntimeError("duplicate Stage20 flow IDs")
    missing_rows = [row for row in sorted_rows if row["flow_id"] not in old_index]
    refs = refs_for_missing(missing_rows)
    by_pcap: dict[Path, dict[int, list[tuple[str, int]]]] = defaultdict(lambda: defaultdict(list))
    for row in missing_rows:
        pcap = pcap_root / row["source_file"]
        if not pcap.is_file():
            raise FileNotFoundError(pcap)
        for slot, ref in enumerate(refs[row["flow_id"]]):
            by_pcap[pcap][int(ref["packet_number"])].append((row["flow_id"], slot))

    output.mkdir(parents=True)
    try:
        np.save(output / "flow_uids.npy", np.asarray(ids, dtype="U80"), allow_pickle=False)
        views = np.lib.format.open_memmap(output / "views.npy", mode="w+", dtype=np.uint8, shape=(len(ids), 3, 8, 64))
        counts = np.lib.format.open_memmap(output / "packet_counts.npy", mode="w+", dtype=np.uint32, shape=(len(ids),))
        new_index = {uid: index for index, uid in enumerate(ids)}
        for row in sorted_rows:
            uid = row["flow_id"]
            if uid in old_index:
                index = old_index[uid]
                views[new_index[uid]] = old_views[index]
                counts[new_index[uid]] = old_counts[index]
            else:
                counts[new_index[uid]] = len(refs[uid])
        recovered = defaultdict(int)
        capture_audit = []
        for pcap, selected in sorted(by_pcap.items(), key=lambda pair: str(pair[0])):
            max_target = max(selected)
            scanned = 0
            reader = RawPcapReader(str(pcap))
            try:
                for packet_number, (raw_bytes, _meta) in enumerate(reader, start=1):
                    scanned = packet_number
                    for uid, slot in selected.get(packet_number, ()):
                        for view, content in enumerate(packet_views(raw_bytes, 64)):
                            views[new_index[uid], view, slot] = content
                        recovered[uid] += 1
                    if packet_number >= max_target:
                        break
            finally:
                reader.close()
            capture_audit.append({
                "pcap": str(pcap), "pcap_sha256": sha256(pcap),
                "packets_scanned": scanned, "last_required_packet": max_target,
                "selected_packet_numbers": len(selected),
                "selected_packet_references": sum(map(len, selected.values())),
            })
            print(json.dumps({"event": "pcap", "dataset": dataset, "name": pcap.name,
                              "packets_scanned": scanned, "references_found": sum(map(len, selected.values()))}), flush=True)
        failed = [uid for uid, sequence in refs.items() if recovered[uid] != len(sequence)]
        if failed:
            raise RuntimeError(f"{len(failed)} missing packet references after PCAP scan: {failed[:5]}")
        views.flush()
        counts.flush()
        if any(counts == 0):
            raise RuntimeError("zero packet flow in Stage20 cache")
        for uid in old_index.keys() & new_index.keys():
            if not np.array_equal(views[new_index[uid]], old_views[old_index[uid]]):
                raise RuntimeError(f"cached view parity failed: {uid}")
        result = {
            "status": "PASS", "dataset": dataset, "stage20_flows": len(ids),
            "copied_from_stage19": len(ids) - len(missing_rows),
            "recovered_from_exact_packet_refs": len(missing_rows),
            "view_shape": list(views.shape), "zero_packet_flows": 0,
            "stage20_manifest_sha256": FROZEN_HASH,
            "stage19_flow_uids_sha256": audit["flow_uids_sha256"],
            "stage19_views_sha256": audit["views_sha256"],
            "flow_uids_sha256": sha256(output / "flow_uids.npy"),
            "views_sha256": sha256(output / "views.npy"),
            "packet_counts_sha256": sha256(output / "packet_counts.npy"),
            "capture_audit": capture_audit, "raw_pcap_modified": False,
        }
        write_json(output / "cache_audit.json", result)
        (output / "SUCCESS").write_text("SUCCESS\n", encoding="utf-8")
        return result
    except BaseException as exc:
        write_json(output / "FAILURE.json", {
            "status": "FAILED", "error_type": type(exc).__name__,
            "error": str(exc), "traceback": traceback.format_exc(),
            "preserve_partial_artifacts": True,
        })
        raise


def main() -> None:
    manifest = STAGE20 / "closed_service_manifest.csv"
    if sha256(manifest) != FROZEN_HASH:
        raise RuntimeError("Stage20 manifest hash changed")
    prior_audit = json.loads((OUT / "ronetc_input_audit.json").read_text(encoding="utf-8"))
    if prior_audit["status"] not in ("RECOVERABLE_GAP", "PASS_FULL_COVERAGE"):
        raise RuntimeError("RoNeTC input audit not passed")
    if ROOT.exists():
        raise FileExistsError(f"cache root exists; refuse overwrite: {ROOT}")
    rows = csv_rows(manifest)
    config = json.loads((STAGE12 / "configs" / "stage12_config.json").read_text(encoding="utf-8"))
    ROOT.mkdir()
    results = []
    for dataset in ("iscx_vpn", "iscx_tor"):
        selected = [row for row in rows if row["dataset"] == dataset]
        results.append(build_dataset(dataset, selected, Path(config["datasets"][dataset]["pcap_root"])))
    if sha256(manifest) != FROZEN_HASH:
        raise RuntimeError("Stage20 manifest changed during view build")
    write_json(ROOT / "completion_verification.json", {
        "status": "PASS", "stage20_manifest_sha256": FROZEN_HASH,
        "datasets": results, "source_assets_modified": False,
    })
    print(json.dumps({"status": "PASS", "datasets": [
        {"dataset": result["dataset"], "flows": result["stage20_flows"],
         "recovered": result["recovered_from_exact_packet_refs"]} for result in results
    ]}, indent=2))


if __name__ == "__main__":
    main()
