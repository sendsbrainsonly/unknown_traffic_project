#!/usr/bin/env python3
"""Read-only dataset inventory and pre-training provenance audit for Stage 12."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import traceback
from collections import Counter, defaultdict
from pathlib import Path

from stage12_common import (
    CONFIG_PATH,
    PROJECT_ROOT,
    STAGE_ROOT,
    classify_capture,
    discover_pcaps,
    git_identity,
    load_config,
    sha256_file,
    write_csv,
    write_json,
)


INVENTORY_FIELDS = [
    "dataset", "record_type", "file_name", "source_file", "absolute_path",
    "size_bytes", "file_type", "official_label", "official_category",
    "application_label", "canonical_class", "domain_state", "group_id",
    "packet_count", "first_packet_timestamp", "last_packet_timestamp",
    "readable", "corruption_reason", "label_source",
]


def capinfos(path: Path) -> dict[str, object]:
    completed = subprocess.run(
        ["capinfos", "-TmQ", "-c", "-a", "-e", str(path)],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    row: dict[str, object] = {
        "readable": completed.returncode == 0,
        "packet_count": "",
        "first_packet_timestamp": "",
        "last_packet_timestamp": "",
        "corruption_reason": completed.stderr.strip().replace("\n", " | "),
    }
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if len(lines) >= 2:
        parsed = list(csv.DictReader(lines))
        if parsed:
            values = parsed[0]
            row.update(
                packet_count=values.get("Number of packets", ""),
                first_packet_timestamp=values.get("Start time", ""),
                last_packet_timestamp=values.get("End time", ""),
            )
    if not row["packet_count"]:
        match = re.search(r"after reading ([0-9]+) packets", completed.stderr)
        if match:
            row["packet_count"] = match.group(1)
    return row


def supporting_row(dataset: str, root: Path, path: Path) -> dict[str, object]:
    suffix = path.suffix.lower()
    kind = {
        ".csv": "flow_csv",
        ".zip": "archive",
        ".json": "metadata",
        ".md": "documentation",
        ".txt": "text",
    }.get(suffix, "supporting_file")
    return {
        "dataset": dataset,
        "record_type": kind,
        "file_name": path.name,
        "source_file": path.relative_to(root).as_posix(),
        "absolute_path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "file_type": suffix.lstrip(".") or "none",
        "official_label": "",
        "official_category": "",
        "application_label": "",
        "canonical_class": "",
        "domain_state": "",
        "group_id": "",
        "packet_count": "",
        "first_packet_timestamp": "",
        "last_packet_timestamp": "",
        "readable": path.is_file(),
        "corruption_reason": "",
        "label_source": "",
    }


def inventory_dataset(dataset: str, config: dict) -> tuple[list[dict], dict]:
    dataset_cfg = config["datasets"][dataset]
    root = Path(dataset_cfg["root"]).resolve()
    pcap_root = Path(dataset_cfg["pcap_root"]).resolve()
    if not root.is_dir() or not pcap_root.is_dir():
        raise FileNotFoundError(f"missing dataset path: {root} / {pcap_root}")
    pcaps = discover_pcaps(pcap_root)
    pcap_set = set(pcaps)
    rows: list[dict] = []
    mapping_errors: list[dict] = []
    for index, path in enumerate(pcaps, start=1):
        try:
            label = classify_capture(dataset, path, pcap_root)
            mapped = {
                "official_label": label.official_joint_label,
                "official_category": label.official_category,
                "application_label": label.application_label,
                "canonical_class": label.canonical_class,
                "domain_state": label.domain_state,
                "label_source": label.label_source,
            }
        except Exception as exc:
            mapped = {key: "" for key in (
                "official_label", "official_category", "application_label",
                "canonical_class", "domain_state", "label_source",
            )}
            mapping_errors.append({"path": str(path), "error": str(exc)})
        info = capinfos(path)
        rows.append(
            {
                "dataset": dataset,
                "record_type": "pcap",
                "file_name": path.name,
                "source_file": path.relative_to(pcap_root).as_posix(),
                "absolute_path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "file_type": path.suffix.lower().lstrip("."),
                **mapped,
                "group_id": path.relative_to(pcap_root).as_posix(),
                **info,
            }
        )
        print(
            f"dataset={dataset} capinfos={index}/{len(pcaps)} readable={info['readable']} "
            f"source={path.relative_to(pcap_root).as_posix()}",
            flush=True,
        )
    for path in sorted(item for item in root.rglob("*") if item.is_file() and item not in pcap_set):
        rows.append(supporting_row(dataset, root, path))
    readable_pcaps = [row for row in rows if row["record_type"] == "pcap" and row["readable"]]
    class_groups: dict[str, set[str]] = defaultdict(set)
    class_domains: dict[str, set[str]] = defaultdict(set)
    class_packets: Counter[str] = Counter()
    for row in readable_pcaps:
        if row["canonical_class"]:
            class_groups[str(row["canonical_class"])].add(str(row["group_id"]))
            class_domains[str(row["canonical_class"])].add(str(row["domain_state"]))
            if row["packet_count"] != "":
                class_packets[str(row["canonical_class"])] += int(row["packet_count"])
    summary = {
        "dataset": dataset,
        "root": str(root),
        "pcap_root": str(pcap_root),
        "total_files": sum(1 for path in root.rglob("*") if path.is_file()),
        "total_size_bytes": sum(path.stat().st_size for path in root.rglob("*") if path.is_file()),
        "pcap_files": len(pcaps),
        "pcap_size_bytes": sum(path.stat().st_size for path in pcaps),
        "readable_pcaps": len(readable_pcaps),
        "unreadable_pcaps": len(pcaps) - len(readable_pcaps),
        "mapping_errors": mapping_errors,
        "canonical_classes_before_flow_eligibility": sorted(class_groups),
        "class_readable_group_counts": {name: len(class_groups[name]) for name in sorted(class_groups)},
        "class_domain_states": {name: sorted(class_domains[name]) for name in sorted(class_domains)},
        "class_packet_counts": {name: class_packets[name] for name in sorted(class_packets)},
        "flow_count_status": "pending Stage12 Open-Detect preprocessing",
    }
    return rows, summary


def write_dataset_audit(dataset: str, rows: list[dict], summary: dict) -> None:
    protocol_dir = STAGE_ROOT / "protocol" / dataset
    write_csv(protocol_dir / "dataset_inventory.csv", rows, INVENTORY_FIELDS)
    pcap_rows = [row for row in rows if row["record_type"] == "pcap"]
    csv_rows = [row for row in rows if row["record_type"] == "flow_csv"]
    archive_rows = [row for row in rows if row["record_type"] == "archive"]
    lines = [
        f"# {dataset} Dataset Audit",
        "",
        "## Status before preprocessing",
        "",
        f"- Total files: `{summary['total_files']}`",
        f"- Total size: `{summary['total_size_bytes']}` bytes",
        f"- PCAP/PCAPNG: `{summary['pcap_files']}` files / `{summary['pcap_size_bytes']}` bytes",
        f"- Readable PCAP: `{summary['readable_pcaps']}`; unreadable/corrupt: `{summary['unreadable_pcaps']}`",
        f"- Flow CSV: `{len(csv_rows)}`; archive: `{len(archive_rows)}`",
        f"- Filename-label mapping errors: `{len(summary['mapping_errors'])}`",
        "- Final class eligibility remains pending until Open-Detect flow conversion counts are available.",
        "",
        "## Canonical label rule",
        "",
        "The canonical class is the application/service identity derived from the official capture filename and official application list. The official coarse traffic category and tunnel state are retained separately. Tunnel state is metadata, not a class.",
        "",
        "## Readable source groups before flow eligibility",
        "",
        "| Canonical class | Readable source groups | Domain states | Packet count |",
        "|---|---:|---|---:|",
    ]
    for name in summary["canonical_classes_before_flow_eligibility"]:
        lines.append(
            f"| {name} | {summary['class_readable_group_counts'][name]} | "
            f"{', '.join(summary['class_domain_states'][name])} | "
            f"{summary['class_packet_counts'].get(name, 0)} |"
        )
    corrupt = [row for row in pcap_rows if not row["readable"]]
    lines += ["", "## Unreadable or corrupt captures", ""]
    if corrupt:
        for row in corrupt:
            lines.append(f"- `{row['source_file']}`: {row['corruption_reason']}")
    else:
        lines.append("None detected by `capinfos`.")
    if summary["mapping_errors"]:
        lines += ["", "## Mapping errors", ""]
        lines.extend(f"- `{item['path']}`: {item['error']}" for item in summary["mapping_errors"])
    lines += [
        "",
        "## Leakage boundary",
        "",
        "The same canonical application across VPN/non-VPN or Tor/non-Tor will be assigned wholly to Known or wholly to Unknown. Source-PCAP groups will not cross Known Train/Validation/Test.",
    ]
    (protocol_dir / "DATASET_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def provenance_audit(config: dict) -> dict:
    od_root = Path(config["open_detect_root"])
    v6 = od_root / "artifacts" / "paper-reproduction" / "v6-local-pcap-fivefold-corrected-v1"
    stage_requirements = {
        "stage10a": PROJECT_ROOT / "stage10a_opendetect_protocol_audit",
        "stage10b": PROJECT_ROOT / "stage10b_opendetect_score_decomposition",
        "stage11a": PROJECT_ROOT / "stage11a_data_anchored_prototype",
        "stage11b": PROJECT_ROOT / "stage11b_decoupled_support_readout",
        "stage11c": PROJECT_ROOT / "stage11c_known_only_support_complexity",
    }
    stages = {}
    for name, root in stage_requirements.items():
        required = [root / "README.md"]
        if name != "stage10a":
            required += [root / "RESULTS.md", root / "manifest.json"]
        files = [path for path in required if path.is_file()]
        stages[name] = {
            "root": str(root.resolve()),
            "required_present": len(files) == len(required),
            "required": [str(path.resolve()) for path in required],
            "sha256": {path.name: sha256_file(path) for path in files},
        }
    success_runs = sorted(v6.glob("a*/fold*_seed*/SUCCESS"))
    v6_ok = (v6 / "CAMPAIGN_SUCCESS").is_file() and len(success_runs) == 15
    frozen_mentions = []
    for root in stage_requirements.values():
        for path in (root / "README.md", root / "RESULTS.md"):
            if path.is_file():
                text = path.read_text(encoding="utf-8", errors="replace")
                if "ISCX" in text or "ISCXTor" in text:
                    frozen_mentions.append(str(path.resolve()))
    status = "PASS" if v6_ok and all(item["required_present"] for item in stages.values()) and not frozen_mentions else "FAIL"
    return {
        "status": status,
        "open_detect_v6": {
            "root": str(v6.resolve()),
            "campaign_success": (v6 / "CAMPAIGN_SUCCESS").is_file(),
            "successful_runs": len(success_runs),
            "expected_runs": 15,
            "git": git_identity(od_root),
        },
        "frozen_stages": stages,
        "iscx_mentions_in_frozen_stage_readmes_or_results": frozen_mentions,
        "independent_status": "METHOD_UNTOUCHED" if not frozen_mentions else "REQUIRES_REVIEW",
        "config_sha256": sha256_file(CONFIG_PATH),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("iscx_vpn", "iscx_tor", "all"), default="all")
    args = parser.parse_args()
    config = load_config()
    datasets = tuple(config["datasets"]) if args.dataset == "all" else (args.dataset,)
    try:
        summaries = {}
        for dataset in datasets:
            rows, summary = inventory_dataset(dataset, config)
            write_dataset_audit(dataset, rows, summary)
            write_json(STAGE_ROOT / "protocol" / dataset / "dataset_inventory_summary.json", summary)
            summaries[dataset] = summary
        audit = provenance_audit(config)
        write_json(STAGE_ROOT / "outputs" / "summary" / "provenance_audit.json", audit)
        write_json(STAGE_ROOT / "outputs" / "summary" / "provenance_verification.json", audit)
        if audit["status"] != "PASS":
            raise RuntimeError(f"frozen provenance audit failed: {audit}")
        write_json(STAGE_ROOT / "outputs" / "summary" / "dataset_audit_status.json", {
            "status": "PASS",
            "datasets": summaries,
            "training_started": False,
            "test_opened": False,
        })
        print(json.dumps({"status": "PASS", "datasets": summaries, "provenance": audit}, indent=2, ensure_ascii=False))
        return 0
    except BaseException as exc:
        write_json(STAGE_ROOT / "outputs" / "summary" / "dataset_audit_failure.json", {
            "status": "FAILED",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "training_started": False,
            "test_opened": False,
        })
        raise


if __name__ == "__main__":
    raise SystemExit(main())
