"""Build a read-only 20-class view of the extracted USTC-TFC2016 PCAPs.

The local official extraction stores ``SMB-1.pcap`` and ``SMB-2.pcap`` as two
flat files.  The canonical Stage 0 class discovery treats flat PCAP stems as
class names, so that layout would incorrectly create 21 classes.  This module
creates a project-local symlink view in which those two shards live under one
``Benign/SMB/`` class directory.  Source PCAPs are never copied or modified.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List

from src.preprocessing.flow_split import discover_classes


EXPECTED_CLASSES = {
    "Benign": {
        "BitTorrent", "FTP", "Facetime", "Gmail", "MySQL", "Outlook",
        "SMB", "Skype", "Weibo", "WorldOfWarcraft",
    },
    "Malware": {
        "Cridex", "Geodo", "Htbot", "Miuref", "Neris", "Nsis-ay",
        "Shifu", "Tinba", "Virut", "Zeus",
    },
}


def _class_name(split: str, pcap: Path) -> str:
    if split == "Benign" and pcap.stem in {"SMB-1", "SMB-2"}:
        return "SMB"
    if split == "Benign" and pcap.parent.name == "Weibo":
        return "Weibo"
    return pcap.stem


def _source_pcaps(source_root: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for split in ("Benign", "Malware"):
        split_dir = source_root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"missing source class directory: {split_dir}")
        for pcap in sorted(split_dir.rglob("*.pcap")):
            if pcap.name.startswith("._"):
                continue
            rows.append({
                "split": split,
                "class_name": _class_name(split, pcap),
                "source": pcap.resolve(),
                "size_bytes": pcap.stat().st_size,
            })
    return rows


def build_ustc20_view(source_root: Path, output_root: Path) -> Dict[str, object]:
    """Create an atomic symlink view and return its manifest payload."""
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(
            f"output already exists: {output_root}; preserve it or choose a new path"
        )

    rows = _source_pcaps(source_root)
    actual = {
        split: {str(row["class_name"]) for row in rows if row["split"] == split}
        for split in EXPECTED_CLASSES
    }
    if actual != EXPECTED_CLASSES:
        raise ValueError(
            "USTC-TFC2016 class coverage mismatch: "
            f"expected={EXPECTED_CLASSES}, actual={actual}"
        )

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = output_root.with_name(f".{output_root.name}.staging-{os.getpid()}")
    if staging.exists():
        raise FileExistsError(f"staging path already exists: {staging}")

    try:
        for row in rows:
            split = str(row["split"])
            class_name = str(row["class_name"])
            source = Path(row["source"])
            if class_name in {"SMB", "Weibo"}:
                destination = staging / split / class_name / source.name
            else:
                destination = staging / split / source.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.symlink_to(source)
            row["view_path"] = str(destination.relative_to(staging))

        discovered = discover_classes(str(staging))
        discovered_names = {split: set() for split in EXPECTED_CLASSES}
        for split, class_name, _pcaps in discovered:
            discovered_names[split].add(class_name)
        if discovered_names != EXPECTED_CLASSES or len(discovered) != 20:
            raise AssertionError(
                f"view discovery mismatch: count={len(discovered)}, "
                f"classes={discovered_names}"
            )

        manifest = {
            "source_root": str(source_root),
            "output_root": str(output_root),
            "source_pcap_count": len(rows),
            "class_count": len(discovered),
            "classes": {
                split: sorted(names) for split, names in discovered_names.items()
            },
            "files": [
                {
                    key: (str(value) if isinstance(value, Path) else value)
                    for key, value in row.items()
                }
                for row in rows
            ],
        }
        (staging / "view_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        staging.rename(output_root)
        return manifest
    except Exception:
        if staging.exists():
            for path in sorted(staging.rglob("*"), reverse=True):
                if path.is_symlink() or path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
            staging.rmdir()
        raise
