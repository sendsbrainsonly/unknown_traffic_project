"""Isolated paths and evidence helpers for the fifth CIC candidate."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
UNIT = "unknown_bot"
MANIFEST = ROOT / "candidate_unknown_manifest.csv"
PROTOCOL = ROOT / "candidate_protocol.json"
PROGRESS = ROOT / "progress.json"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def write_json_new(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def replace_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def progress(status: str, phase: str, **extra) -> None:
    value = {"status": status, "phase": phase,
             "updated_at_utc": datetime.now(timezone.utc).isoformat(), **extra}
    replace_json(PROGRESS, value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def write_manifest_new(rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError("candidate has no matched flows")
    with MANIFEST.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
