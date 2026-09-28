#!/usr/bin/env python3
"""Freeze score-blind Known Test membership for the predeclared 1:1 Bot view."""
from __future__ import annotations

import csv
import hashlib
import json

from stage42w_common import MANIFEST, PROJECT, PROTOCOL, ROOT, digest, write_json_new

KNOWN_ROLES = PROJECT / "stage40_ustc_cic_open_set/unknown_slowhttptest_roles.csv"
SELECTED = ROOT / "balanced_known_manifest.csv"
AMENDMENT = ROOT / "balanced_view_implementation_audit.json"


def main() -> None:
    if SELECTED.exists() or AMENDMENT.exists():
        raise FileExistsError("balanced Known Test membership already frozen")
    if (ROOT / "unknown_bot/input_caches/ustc/A-2/tf_fig_test/flow_ids.npy").exists():
        raise RuntimeError("Bot packet features were already opened")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if digest(MANIFEST) != protocol["candidate_manifest_sha256"]:
        raise RuntimeError("Bot candidate manifest drift")
    if digest(KNOWN_ROLES) != protocol["known_role_manifest_sha256"]:
        raise RuntimeError("Known role manifest drift")
    with KNOWN_ROLES.open(newline="", encoding="utf-8") as handle:
        known_ids = [row["flow_id"] for row in csv.DictReader(handle) if row["role"] == "known_test"]
    if len(known_ids) != 2272 or len(set(known_ids)) != 2272:
        raise RuntimeError("Known Test population drift")
    size = min(len(known_ids), protocol["unknown_test"])
    chosen = sorted(known_ids, key=lambda flow_id: (
        hashlib.sha256(f"2022|balanced|{flow_id}".encode()).digest(), flow_id))[:size]
    with SELECTED.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["flow_id"])
        writer.writerows((flow_id,) for flow_id in sorted(chosen))
    audit = {
        "status": "PASS", "candidate_manifest_sha256": protocol["candidate_manifest_sha256"],
        "known_role_manifest_sha256": protocol["known_role_manifest_sha256"],
        "known_test_population": 2272, "unknown_test_population": protocol["unknown_test"],
        "balanced_known_selected": size, "balanced_unknown_selected": size,
        "selection_rule": "lowest SHA256(2022|balanced|flow_id), score-blind; Known Test IDs only",
        "balanced_known_manifest_sha256": digest(SELECTED),
        "fix_code_sha256": digest(ROOT / "fix_balanced_view.py"),
        "reason": "Earlier evaluator keeps all Known Test when Unknown Test < Known Test; this isolated implementation correction enforces the already-frozen 1:1 view without changing candidate IDs, natural scores, thresholds, or model.",
        "frozen_before_bot_packet_feature_access": True,
    }
    write_json_new(AMENDMENT, audit)
    print(json.dumps(audit, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
