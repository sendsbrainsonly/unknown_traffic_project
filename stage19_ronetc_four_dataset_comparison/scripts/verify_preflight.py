#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import Counter

import numpy as np

from common import PROTOCOL_MANIFEST, ROOT, cache_dir, config, read_csv, read_json, sha256_file, write_json


def main() -> None:
    rows = read_csv(PROTOCOL_MANIFEST)
    checks = []
    for spec in config()["protocols"]:
        selected = [row for row in rows if row["dataset"] == spec["dataset"] and row["protocol_id"] == spec["protocol_id"]]
        roles = Counter(row["role"] for row in selected)
        train_uids = {row["flow_uid"] for row in selected if row["role"] == "known_train"}
        val_uids = {row["flow_uid"] for row in selected if row["role"] == "known_validation"}
        test_uids = {row["flow_uid"] for row in selected if row["role"] in {"known_test", "unknown_test"}}
        assert train_uids.isdisjoint(val_uids | test_uids) and val_uids.isdisjoint(test_uids)
        assert all(row["is_unknown"] == "0" for row in selected if row["role"] in {"known_train", "known_validation"})
        cache = cache_dir(spec["dataset"])
        audit = read_json(cache / "cache_audit.json")
        assert audit["status"] == "PASS" and audit["missing_flows"] == 0
        flow_ids = np.load(cache / "flow_uids.npy", allow_pickle=False)
        views = np.load(cache / "views.npy", mmap_mode="r", allow_pickle=False)
        assert len(flow_ids) == len(views) and views.shape[1:] == (3, 8, 64)
        assert {row["flow_uid"] for row in selected} <= set(map(str, flow_ids))
        checks.append({"dataset": spec["dataset"], "protocol_id": spec["protocol_id"], **roles, "cache_flows": len(flow_ids), "views_sha256": sha256_file(cache / "views.npy"), "status": "PASS"})
    payload = {"status": "PASS", "protocols": checks, "manifest_sha256": sha256_file(PROTOCOL_MANIFEST), "unknown_train_rows": 0, "unknown_validation_rows": 0, "test_used_for_tuning": False}
    write_json(ROOT / "preflight_verification.json", payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
