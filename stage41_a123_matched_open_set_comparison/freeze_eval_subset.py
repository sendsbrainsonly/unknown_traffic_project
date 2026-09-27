#!/usr/bin/env python3
"""Freeze class-blind 1:1 Known/Unknown evaluation IDs before score inspection."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from freeze_matched_protocol import ROOT, digest


def main() -> None:
    protocol = json.loads((ROOT / "matched_protocol.json").read_text())
    output = ROOT / "balanced_test_ids.csv"
    if output.exists():
        raise FileExistsError(output)
    selected = []
    for setting, unit in protocol["units"].items():
        path = Path(unit["role_manifest"])
        if digest(path) != unit["role_manifest_sha256"]:
            raise RuntimeError("role manifest drift")
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        known = [r for r in rows if r["role"] == "known_test"]
        unknown = [r for r in rows if r["role"] == "unknown_test"]
        if len(known) < len(unknown) or not unknown:
            raise RuntimeError(f"cannot construct 1:1 evaluation for {setting}")
        rank = lambda r: (hashlib.sha256(f"2021|balanced|{setting}|{r['flow_id']}".encode()).digest(),
                          r["flow_id"])
        for row in sorted(known, key=rank)[:len(unknown)] + unknown:
            selected.append({"setting": setting, "flow_id": row["flow_id"],
                             "class_name": row["class_name"], "role": row["role"]})
    selected.sort(key=lambda r: (r["setting"], r["flow_id"]))
    with output.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
    print(json.dumps({"status": "PASS", "file": str(output), "sha256": digest(output),
                      "rows": len(selected)}))


if __name__ == "__main__":
    main()
