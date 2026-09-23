#!/usr/bin/env python3
"""Refresh the Stage 19 artifact inventory while preserving runtime sockets.

The workspace-wide refresher assumes every non-directory path is a readable
regular file. PyTorch multiprocessing leaves Unix-domain listener sockets in
project-local runtime directories, so this experiment-specific refresher
records those paths as special runtime artifacts without trying to read them.
Files that are still being updated by the active formal runs are inventoried
without a transient hash; terminal refreshes can hash them after training.
"""
from __future__ import annotations

import hashlib
import json
import os
import stat
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifest.json"
METADATA_FILES = {"RESULTS.md", "manifest.json"}
ACTIVE_PREFIXES = (
    "runs/vnat/medium_seed2026/",
    "runs/ustc/A-2/",
    "queue_logs/vnat_medium2026_independent_single_gpu_20260921.log",
    "queue_logs/ustc_A2_independent_single_gpu_20260921.log",
    "queue_logs/ustc_A2_ustc_three_gpu_final_20260921.log",
    "queue_status.json",
    "progress.txt",
    "takeover_independent_single_gpu_20260921.json",
    "takeover_ustc_three_gpu_final_20260921.json",
)
HASH_MAX_BYTES = 512 * 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def infer_role(relative: Path) -> str:
    name = relative.name.lower()
    parts = {part.lower() for part in relative.parts}
    if relative.suffix.lower() in {".pt", ".pth", ".ckpt"}:
        return "checkpoint"
    if "prediction" in name or "predictions" in parts:
        return "predictions"
    if relative.suffix.lower() in {".json", ".csv", ".parquet"}:
        return "metrics-or-data"
    if relative.suffix.lower() in {".log", ".out", ".err"}:
        return "log"
    return "artifact"


def is_active(relative: str) -> bool:
    return any(relative == prefix or relative.startswith(prefix) for prefix in ACTIVE_PREFIXES)


def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    external = [item for item in manifest.get("artifacts", []) if item.get("scope") == "external"]
    bundled: list[dict] = []
    for path in sorted(ROOT.rglob("*")):
        relative = path.relative_to(ROOT)
        relative_text = relative.as_posix()
        if relative_text in METADATA_FILES or path.is_dir():
            continue
        if path.is_symlink():
            bundled.append({
                "path": relative_text,
                "scope": "bundle",
                "role": "symlink",
                "size_bytes": None,
                "sha256": None,
                "hash_status": "symlink-not-followed",
                "target": os.readlink(path),
            })
            continue
        mode = path.lstat().st_mode
        if not stat.S_ISREG(mode):
            bundled.append({
                "path": relative_text,
                "scope": "bundle",
                "role": "runtime-socket" if stat.S_ISSOCK(mode) else "special-file",
                "size_bytes": path.lstat().st_size,
                "sha256": None,
                "hash_status": "skipped-special",
            })
            continue
        size = path.stat().st_size
        if is_active(relative_text):
            digest = None
            hash_status = "skipped-active"
        elif size > HASH_MAX_BYTES:
            digest = None
            hash_status = "skipped-large"
        else:
            digest = sha256_file(path)
            hash_status = "verified"
        bundled.append({
            "path": relative_text,
            "scope": "bundle",
            "role": infer_role(relative),
            "size_bytes": size,
            "sha256": digest,
            "hash_status": hash_status,
        })
    manifest["artifacts"] = external + bundled
    manifest["updated_at_utc"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    temporary = MANIFEST.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, MANIFEST)
    print(f"refreshed manifest: bundle_files={len(bundled)} external_refs={len(external)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
