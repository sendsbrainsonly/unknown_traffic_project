#!/usr/bin/env python3
"""Capture the prior manuscript and validate its integrated P/R table."""
import csv
import hashlib
import json
import re
import shutil
import sys
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
BUNDLE = ROOT.parent
PROJECT = BUNDLE.parent
CHAPTER = PROJECT / "paper/md/ch4/4(1).md"
METHODS = [
    ("TrafficFormer（预训练）", "TrafficFormer-pretrained"),
    ("YaTC（预训练）", "YaTC-official-pretrained-Stage20"),
    ("Open-Detect（corrected-paper）", "Open-Detect corrected-paper"),
    ("RoNeTC（重建版）", "RoNeTC-runnable-reconstruction"),
    ("TFE-GNN-8-UDP-short（适配版）", "TFE-GNN-8-UDP-short"),
    ("Trident-early8-86D（适配版）", "Trident-early8-86D"),
    ("E3（早期双视图）", "OURS-E3-T8-pretrained"),
    ("本方法（三视图）", "OURS-current-threeview-equal"),
]
METRICS = ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


class BodyCells(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows = []
        self.row = None
        self.cell = None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.row = []
        if tag == "td":
            self.cell = []

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag):
        if tag == "td":
            self.row.append("".join(self.cell).strip())
            self.cell = None
        if tag == "tr" and self.row:
            self.rows.append(self.row)


def snapshot():
    for source, name in ((CHAPTER, "chapter4.before.md"), (PROJECT / "README.md", "project_readme.before.md"),
                         (BUNDLE / "manifest.json", "manifest.before-publication.json")):
        destination = ROOT / name
        if destination.exists():
            raise FileExistsError(destination)
        shutil.copy2(source, destination)
    write_json(ROOT / "snapshot.json", {
        "timestamp_shanghai": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "chapter_path": str(CHAPTER.relative_to(PROJECT)), "chapter_before_sha256": sha(CHAPTER),
        "readme_before_sha256": sha(PROJECT / "README.md"),
        "metric_summary_sha256": sha(BUNDLE / "method_summary.csv"),
        "source_hashes_before_sha256": sha(BUNDLE / "source_hashes_before.json"),
    })
    print("SNAPSHOT_PASS prior chapter, readme and manifest retained")


def outside_allowed_edits(text):
    text = re.sub(r"已知类别分类采用.*?(?=未知类别检测采用)", "KNOWN_METRIC_DEFINITIONS\n", text, flags=re.S)
    return re.sub(r"## 4.2 已知类别识别性能.*?(?=为进一步考察本方法在不同标签粒度)",
                  "KNOWN_COMPARISON_TABLE_AND_DISCUSSION\n", text, flags=re.S)


def verify():
    before = (ROOT / "chapter4.before.md").read_bytes().decode("utf-8")
    after = CHAPTER.read_bytes().decode("utf-8")
    assert outside_allowed_edits(before) == outside_allowed_edits(after), "unrelated manuscript change"
    assert after.count("\r\n") == before.count("\r\n") + after.count("\n") - before.count("\n"), "line ending convention changed"
    section = after.split("## 4.2 已知类别识别性能", 1)[1].split("为进一步考察本方法在不同标签粒度", 1)[0]
    parser = BodyCells()
    parser.feed(section)
    assert len(parser.rows) == 8 and all(len(row) == 11 for row in parser.rows)
    with (BUNDLE / "method_summary.csv").open(newline="", encoding="utf-8") as stream:
        summary = {(r["dataset"], r["method"]): r for r in csv.DictReader(stream)}
    checks = 0
    for cells, (label, method) in zip(parser.rows, METHODS, strict=True):
        assert cells[0] == label
        expected = []
        for dataset in ("iscx_vpn", "iscx_tor"):
            row = summary[(dataset, method)]
            expected.extend(f"{100*float(row[key + '_mean']):.2f} ± {100*float(row[key + '_population_sd']):.2f}" for key in METRICS)
        assert cells[1:] == expected, (method, cells[1:], expected)
        checks += len(expected)
    assert "0.66和3.89个百分点" in section
    assert "3.42和4.41个百分点" in section
    snapshot_data = json.loads((ROOT / "snapshot.json").read_text())
    assert snapshot_data["metric_summary_sha256"] == sha(BUNDLE / "method_summary.csv")
    source_hashes = json.loads((BUNDLE / "source_hashes_before.json").read_text())
    assert all(sha(PROJECT / path) == digest for path, digest in source_hashes.items())
    record = {"status": "PASS", "timestamp_shanghai": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "display_cells_verified": checks, "methods": 8, "datasets": ["VPN6", "Tor7"], "seeds": [2022, 2023],
        "source_files_unchanged": len(source_hashes), "unrelated_manuscript_sections_unchanged": True,
        "line_endings_preserved": True, "chapter_sha256_after": sha(CHAPTER),
        "readme_sha256_after": sha(PROJECT / "README.md"), "metrics_changed": False,
        "frozen_experiments_changed": False, "models_or_data_uploaded": False}
    write_json(ROOT / "verification.json", record)
    manifest = json.loads((BUNDLE / "manifest.json").read_text(encoding="utf-8"))
    manifest["publication"] = {
        "date_shanghai": "2026-11-04", "chapter": str(CHAPTER.relative_to(PROJECT)),
        "chapter_sha256": record["chapter_sha256_after"], "verified_display_cells": checks,
        "verification": "publication_20261104/verification.json",
        "revision_note": "paper/md/ch4/修订记录_20261104_Precision_Recall补充.md",
        "scope": "manuscript, code, small metric/summary/per-class tables and verification evidence",
        "excluded": ["weights", "caches", "per-sample predictions", "runtime logs", "unrelated Stage48 outputs"],
    }
    write_json(BUNDLE / "manifest.json", manifest)
    print(json.dumps(record, ensure_ascii=False))


if __name__ == "__main__":
    {"snapshot": snapshot, "verify": verify}[sys.argv[1]]()
