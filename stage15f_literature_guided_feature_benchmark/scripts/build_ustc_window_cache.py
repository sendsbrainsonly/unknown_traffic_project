#!/usr/bin/env python3
"""Recover Stage 15F packet-window metrics from the preserved USTC Stage-0 PKLs."""

from __future__ import annotations

import csv
import gc
import json
import pickle
from pathlib import Path

from common import FLOW_WINDOW_FIELDS, PROJECT_ROOT, ROOT, WindowAccumulator, sha256_file, write_csv, write_json


def target_metadata() -> dict[str, dict[str, str]]:
    targets: dict[str, dict[str, str]] = {}
    for protocol in ("A-1", "A-2", "A-3"):
        path = PROJECT_ROOT / "stage3_unknown_utility" / "outputs" / protocol / "data_manifest.csv"
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row["known_or_unknown"].lower() != "known":
                    continue
                if row["used_encoder_train"].lower() != "true" and row["used_encoder_val"].lower() != "true":
                    continue
                prior = targets.setdefault(row["flow_id"], row)
                if prior["class_name"] != row["class_name"]:
                    raise RuntimeError(f"inconsistent USTC class: {row['flow_id']}")
    return targets


def packet_fields(frame: bytes) -> tuple[int | None, int | None, str]:
    if len(frame) < 34 or frame[12:14] != b"\x08\x00":
        return None, None, ""
    ihl = (frame[14] & 0x0F) * 4
    ip_total = int.from_bytes(frame[16:18], "big")
    proto = frame[23]
    l4 = 14 + ihl
    payload = None
    token = "ipv4"
    if proto == 6 and len(frame) >= l4 + 20:
        tcp_hlen = ((frame[l4 + 12] >> 4) & 0x0F) * 4
        payload = max(0, ip_total - ihl - tcp_hlen)
        token = "ipv4:tcp"
    elif proto == 17 and len(frame) >= l4 + 8:
        payload = max(0, ip_total - ihl - 8)
        token = "ipv4:udp"
    return ip_total, payload, token


def main() -> None:
    output = ROOT / "window_cache" / "ustc"
    if output.exists():
        raise RuntimeError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    targets = target_metadata()
    accumulators: dict[str, WindowAccumulator] = {}
    pkl_rows: list[dict[str, object]] = []
    for pkl_path in sorted((PROJECT_ROOT / "data" / "flows").glob("*.pkl")):
        with pkl_path.open("rb") as handle:
            payload = pickle.load(handle)
        matched = 0
        for uid, packets in payload["packets"].items():
            meta = targets.get(uid)
            if meta is None:
                continue
            matched += 1
            acc = WindowAccumulator(
                uid,
                meta["class_name"],
                str(payload["source_file"]),
                "Malware" if uid.startswith("Malware__") else "Benign",
            )
            for timestamp, caplen, _wirelen, direction, raw_frame in packets:
                frame = bytes(raw_frame)
                ip_bytes, transport_payload, tokens = packet_fields(frame)
                source = ("initiator", "") if int(direction) == 1 else ("peer", "")
                destination = ("peer", "") if int(direction) == 1 else ("initiator", "")
                acc.add(float(timestamp), int(caplen), ip_bytes, transport_payload, source, destination, tokens)
            accumulators[uid] = acc
        pkl_rows.append(
            {
                "pkl_path": str(pkl_path.resolve()),
                "pkl_sha256": sha256_file(pkl_path),
                "class_name": str(payload["class_name"]),
                "flows_in_pkl": len(payload["packets"]),
                "matched_known_train_validation_flows": matched,
            }
        )
        print(json.dumps({"event": "pkl_complete", "file": pkl_path.name, "matched": matched}), flush=True)
        del payload
        gc.collect()
    missing = sorted(set(targets) - set(accumulators))
    if missing:
        write_json(output / "failure.json", {"missing_count": len(missing), "preview": missing[:100]})
        raise RuntimeError(f"missing {len(missing)} USTC targets")
    rows = [accumulators[uid].row("ustc") for uid in sorted(accumulators)]
    metrics = output / "flow_window_metrics.csv.gz"
    write_csv(metrics, rows, FLOW_WINDOW_FIELDS)
    write_csv(output / "pkl_audit.csv", pkl_rows, list(pkl_rows[0]))
    write_json(
        output / "cache_audit.json",
        {
            "status": "PASS",
            "dataset": "ustc",
            "scope": "union of Known Train/Validation rows in frozen A-1/A-2/A-3 manifests",
            "target_flows": len(targets),
            "recovered_flows": len(rows),
            "known_test_values_stored": 0,
            "unknown_test_values_stored": 0,
            "packet_windows": [8, 16, 32, 64],
            "byte_coverage_semantics": "captured frame bytes",
            "time_coverage_semantics": "elapsed time from first through Nth packet divided by full flow duration",
            "source_semantics": "Stage-0 bidirectional five-tuple PKLs; stored order; no timeout/session split",
            "metrics_sha256": sha256_file(metrics),
        },
    )


if __name__ == "__main__":
    main()
