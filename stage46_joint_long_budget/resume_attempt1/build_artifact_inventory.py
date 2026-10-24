#!/usr/bin/env python3
"""Create a hash inventory for every internal Stage46 continuation artifact."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "artifact_inventory.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    entries = []
    for path in sorted(p for p in HERE.rglob("*") if p.is_file() and p != OUT and p.name != "manifest.json"):
        entries.append({"path": path.relative_to(HERE).as_posix(),
                        "size_bytes": path.stat().st_size,
                        "sha256": sha256(path)})
    payload = {"status": "PASS", "experiment_id": "stage46-resume-epoch075-attempt1",
               "file_count": len(entries), "total_bytes": sum(item["size_bytes"] for item in entries),
               "files": entries}
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "file_count": payload["file_count"],
                      "total_bytes": payload["total_bytes"], "inventory": OUT.name}, sort_keys=True))


if __name__ == "__main__":
    main()
