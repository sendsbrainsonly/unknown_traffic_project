#!/usr/bin/env python3
"""Recover the 1,960 Stage14B medium-2025 Known-Test inputs once.

Stage31 already materialized Known Train/Validation and Stage38 materialized
Unknown Test. Stage36 provides a parity-verified recovery function for the same
six inputs; this wrapper supplies only the Stage44-frozen missing IDs.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
STAGE36 = PROJECT / "stage36_vnat_sixclass_open_set_pilot"


def main() -> None:
    sys.path.insert(0, str(STAGE36))
    import run_pilot as pilot

    manifest = PROJECT / "stage14b_vnat_protocol_freeze" / "vnat_split_manifest.csv"
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = [
            row for row in csv.DictReader(handle)
            if row["protocol_id"] == "medium_seed2025"
            and row["class_role"] == "known"
            and row["split"] == "test"
        ]
    rows.sort(key=lambda row: row["flow_uid"])
    if len(rows) != 1960 or len({row["flow_uid"] for row in rows}) != 1960:
        raise RuntimeError("missing Known-Test role is not exactly 1,960 unique flows")
    pilot.recover(rows, ROOT / "input_caches" / "missing_known_test")


if __name__ == "__main__":
    main()
