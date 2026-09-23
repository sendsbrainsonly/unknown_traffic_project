#!/usr/bin/env python3
"""Build auditable, non-destructive label manifests for CipherSpectrum PCAPs.

The raw dataset is treated as read-only.  The script derives class labels from
the dataset's class directories (the release's SNI-derived organization), not
from the visited/page domain embedded in each filename.  It emits both:

* an official-compatible 40-class view, based on directory labels common to
  all four capture groups; and
* a complete local 41-class audit view, preserving the extra getpocket.com
  material and explicitly recording two capture-group packaging anomalies.

No train/validation/test split is created.  ``split_group_id`` is exported so
future splits can keep flows from the same browser capture together.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Iterable, Mapping, Sequence


DEFAULT_DATASET_ROOT = Path(
    "/home/birkenwald/data/SEU-WXY/WXY-SEU/TrafficClassifier/"
    "Dataset/ipherSpectrum(sok)"
)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "cipherspectrum_labeling" / "outputs"

CAPTURE_GROUPS = (
    "aes-128-gcm",
    "aes-256-gcm",
    "chacha20-poly1305",
    "mix",
)
EXPECTED_CIPHERS = {
    "aes-128-gcm": "aes-128",
    "aes-256-gcm": "aes-256",
    "chacha20-poly1305": "chacha20",
}
PCAP_MAGIC = {
    b"\xd4\xc3\xb2\xa1",
    b"\xa1\xb2\xc3\xd4",
    b"\x4d\x3c\xb2\xa1",
    b"\xa1\xb2\x3c\x4d",
    b"\x0a\x0d\x0d\x0a",  # pcapng, retained for defensive checking
}

FILENAME_RE = re.compile(
    r"^(?P<capture_id>traffic_"
    r"(?P<capture_date>\d{4}-\d{2}-\d{2})_"
    r"(?P<visited_domain>.+?)_"
    r"(?P<observed_cipher>aes-128|aes-256|chacha20)_"
    r"(?P<browser>chromium|firefox)_"
    r"(?P<iteration>\d+))"
    r"\.pcap\."
    r"(?P<transport>TCP|UDP)_"
    r"(?P<src_ip>[0-9-]+)_(?P<src_port>\d+)_"
    r"(?P<dst_ip>[0-9-]+)_(?P<dst_port>\d+)\.pcap$"
)

MANIFEST_FIELDS = (
    "flow_id",
    "relative_path",
    "size_bytes",
    "pcap_magic_valid",
    "capture_group_directory",
    "observed_cipher",
    "capture_group_status",
    "directory_label_raw",
    "canonical_label",
    "label_id_official40",
    "label_id_local41",
    "label_source",
    "label_status",
    "official40_eligible",
    "local41_candidate",
    "manual_review_required",
    "capture_id",
    "split_group_id",
    "capture_date",
    "visited_domain",
    "browser",
    "iteration",
    "transport",
    "src_ip",
    "src_port",
    "dst_ip",
    "dst_port",
    "parse_status",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="CipherSpectrum root or its nested payload root",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Project-local output directory",
    )
    return parser.parse_args(argv)


def locate_payload_root(dataset_root: Path) -> Path:
    """Return the directory whose immediate children are capture groups."""
    root = dataset_root.resolve()
    if all((root / group).is_dir() for group in CAPTURE_GROUPS):
        return root
    candidates = [
        child
        for child in root.iterdir()
        if child.is_dir()
        and all((child / group).is_dir() for group in CAPTURE_GROUPS)
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"Cannot uniquely locate CipherSpectrum payload root below {root}; "
            f"found {len(candidates)} candidates"
        )
    return candidates[0].resolve()


def resolve_group_root(payload_root: Path, group: str) -> Path:
    """Handle the release's repeated ``group/group/class`` layout."""
    outer = payload_root / group
    repeated = outer / group
    root = repeated if repeated.is_dir() else outer
    if not root.is_dir():
        raise RuntimeError(f"Missing capture group directory: {root}")
    return root


def class_directories(group_root: Path) -> set[str]:
    return {
        path.name
        for path in group_root.iterdir()
        if path.is_dir() and not path.name.startswith("._")
    }


def derive_label_sets(group_roots: Mapping[str, Path]) -> tuple[list[str], list[str]]:
    raw_sets = [class_directories(group_roots[group]) for group in CAPTURE_GROUPS]
    official40 = sorted(set.intersection(*raw_sets))
    normalized = set().union(*raw_sets)
    # This release contains a directory literally named ``chacha20`` in the
    # aes-256 group.  Its 1000 filenames all identify getpocket.com, so it is
    # normalized only in the audit view and remains excluded from official40.
    normalized.discard("chacha20")
    normalized.add("getpocket.com")
    return official40, sorted(normalized)


def parse_filename(name: str) -> dict[str, str]:
    match = FILENAME_RE.match(name)
    if match is None:
        return {
            "parse_status": "FILENAME_PARSE_ERROR",
            "capture_id": "",
            "split_group_id": "",
            "capture_date": "",
            "visited_domain": "",
            "observed_cipher": "",
            "browser": "",
            "iteration": "",
            "transport": "",
            "src_ip": "",
            "src_port": "",
            "dst_ip": "",
            "dst_port": "",
        }
    values = match.groupdict()
    values["src_ip"] = values["src_ip"].replace("-", ".")
    values["dst_ip"] = values["dst_ip"].replace("-", ".")
    values["split_group_id"] = values["capture_id"]
    values["parse_status"] = "OK"
    return values


def canonicalize_label(capture_group: str, raw_label: str) -> tuple[str, str, str]:
    if capture_group == "aes-256-gcm" and raw_label == "chacha20":
        return (
            "getpocket.com",
            "ANOMALOUS_DIRECTORY_NORMALIZATION",
            "LOCAL_EXTRA_NORMALIZED_DIRECTORY",
        )
    return raw_label, "DIRECTORY_NAME", "UNCLASSIFIED"


def capture_group_status(capture_group: str, observed_cipher: str) -> str:
    if capture_group == "mix":
        return "MIX_EXPECTED"
    expected = EXPECTED_CIPHERS[capture_group]
    return "CONSISTENT" if observed_cipher == expected else "MISPLACED_GROUP"


def stable_flow_id(relative_path: str) -> str:
    digest = hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:32]
    return f"cs_{digest}"


def valid_pcap_magic(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            return handle.read(4) in PCAP_MAGIC
    except OSError:
        return False


def atomic_csv_writers(
    output_dir: Path,
) -> tuple[dict[str, tuple[Path, object, csv.DictWriter]], list[object]]:
    specs = {
        "all": output_dir / "cipherspectrum_label_manifest.csv",
        "official40": output_dir / "cipherspectrum_official40_manifest.csv",
        "review": output_dir / "manual_review_inventory.csv",
    }
    writers: dict[str, tuple[Path, object, csv.DictWriter]] = {}
    handles: list[object] = []
    for key, final_path in specs.items():
        temp_path = final_path.with_suffix(final_path.suffix + ".tmp")
        handle = temp_path.open("w", encoding="utf-8", newline="")
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writers[key] = (final_path, handle, writer)
        handles.append(handle)
    return writers, handles


def finalize_atomic_csvs(writers: Mapping[str, tuple[Path, object, csv.DictWriter]]) -> None:
    for final_path, handle, _writer in writers.values():
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        final_path.with_suffix(final_path.suffix + ".tmp").replace(final_path)


def write_csv_atomic(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    temp_path.replace(path)


def write_json_atomic(path: Path, payload: object) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temp_path.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def build_manifests(dataset_root: Path, output_dir: Path) -> dict[str, object]:
    payload_root = locate_payload_root(dataset_root)
    output_dir = output_dir.resolve()
    if output_dir == payload_root or payload_root in output_dir.parents:
        raise RuntimeError("Output directory must not be inside the raw dataset")
    output_dir.mkdir(parents=True, exist_ok=True)

    group_roots = {
        group: resolve_group_root(payload_root, group) for group in CAPTURE_GROUPS
    }
    official40, local41 = derive_label_sets(group_roots)
    if len(official40) != 40:
        raise RuntimeError(f"Expected 40 common labels, found {len(official40)}")
    if len(local41) != 41:
        raise RuntimeError(f"Expected 41 normalized local labels, found {len(local41)}")
    official_ids = {label: index for index, label in enumerate(official40)}
    local_ids = {label: index for index, label in enumerate(local41)}

    writers, handles = atomic_csv_writers(output_dir)
    all_writer = writers["all"][2]
    official_writer = writers["official40"][2]
    review_writer = writers["review"][2]

    total_files = 0
    total_bytes = 0
    official_count = 0
    local_count = 0
    review_count = 0
    parse_errors = 0
    bad_magic = 0
    group_mismatches = 0
    official40_group_mismatches = 0
    visited_domain_label_differences = 0
    flow_ids: set[str] = set()
    class_counts: Counter[str] = Counter()
    class_group_counts: Counter[tuple[str, str]] = Counter()
    raw_label_counts: Counter[str] = Counter()
    label_status_counts: Counter[str] = Counter()
    mismatch_details: Counter[tuple[str, str, str, str]] = Counter()
    group_capture_ids: dict[str, set[str]] = defaultdict(set)
    capture_to_groups: dict[str, set[str]] = defaultdict(set)
    fingerprint = hashlib.sha256()

    try:
        for capture_group in CAPTURE_GROUPS:
            group_root = group_roots[capture_group]
            paths = sorted(
                path
                for path in group_root.glob("*/*.pcap")
                if path.is_file() and not path.name.startswith("._")
            )
            for path in paths:
                relative_path = path.relative_to(payload_root).as_posix()
                stat = path.stat()
                fingerprint.update(relative_path.encode("utf-8"))
                fingerprint.update(b"\0")
                fingerprint.update(str(stat.st_size).encode("ascii"))
                fingerprint.update(b"\0")
                fingerprint.update(str(stat.st_mtime_ns).encode("ascii"))
                fingerprint.update(b"\n")

                parsed = parse_filename(path.name)
                raw_label = path.parent.name
                canonical_label, label_source, initial_status = canonicalize_label(
                    capture_group, raw_label
                )
                magic_ok = valid_pcap_magic(path)
                group_status = capture_group_status(
                    capture_group, parsed["observed_cipher"]
                )
                parse_ok = parsed["parse_status"] == "OK"
                official_eligible = (
                    canonical_label in official_ids and parse_ok and magic_ok
                )
                local_candidate = canonical_label in local_ids and parse_ok and magic_ok

                if canonical_label in official_ids:
                    label_status = "OFFICIAL40_COMPATIBLE"
                elif initial_status == "LOCAL_EXTRA_NORMALIZED_DIRECTORY":
                    label_status = initial_status
                else:
                    label_status = "LOCAL_EXTRA_GETPOCKET"
                manual_review = (
                    not official_eligible
                    or group_status == "MISPLACED_GROUP"
                    or not parse_ok
                    or not magic_ok
                )

                flow_id = stable_flow_id(relative_path)
                if flow_id in flow_ids:
                    raise RuntimeError(f"Stable flow_id collision: {flow_id}")
                flow_ids.add(flow_id)

                row = {
                    "flow_id": flow_id,
                    "relative_path": relative_path,
                    "size_bytes": stat.st_size,
                    "pcap_magic_valid": bool_text(magic_ok),
                    "capture_group_directory": capture_group,
                    "observed_cipher": parsed["observed_cipher"],
                    "capture_group_status": group_status,
                    "directory_label_raw": raw_label,
                    "canonical_label": canonical_label,
                    "label_id_official40": official_ids.get(canonical_label, ""),
                    "label_id_local41": local_ids.get(canonical_label, ""),
                    "label_source": label_source,
                    "label_status": label_status,
                    "official40_eligible": bool_text(official_eligible),
                    "local41_candidate": bool_text(local_candidate),
                    "manual_review_required": bool_text(manual_review),
                    "capture_id": parsed["capture_id"],
                    "split_group_id": parsed["split_group_id"],
                    "capture_date": parsed["capture_date"],
                    "visited_domain": parsed["visited_domain"],
                    "browser": parsed["browser"],
                    "iteration": parsed["iteration"],
                    "transport": parsed["transport"],
                    "src_ip": parsed["src_ip"],
                    "src_port": parsed["src_port"],
                    "dst_ip": parsed["dst_ip"],
                    "dst_port": parsed["dst_port"],
                    "parse_status": parsed["parse_status"],
                }
                all_writer.writerow(row)
                if official_eligible:
                    official_writer.writerow(row)
                    official_count += 1
                if manual_review:
                    review_writer.writerow(row)
                    review_count += 1

                total_files += 1
                total_bytes += stat.st_size
                local_count += int(local_candidate)
                parse_errors += int(not parse_ok)
                bad_magic += int(not magic_ok)
                group_mismatches += int(group_status == "MISPLACED_GROUP")
                official40_group_mismatches += int(
                    group_status == "MISPLACED_GROUP" and official_eligible
                )
                visited_domain_label_differences += int(
                    parsed["visited_domain"] != canonical_label
                )
                class_counts[canonical_label] += 1
                class_group_counts[(canonical_label, capture_group)] += 1
                raw_label_counts[raw_label] += 1
                label_status_counts[label_status] += 1
                if group_status == "MISPLACED_GROUP":
                    mismatch_details[
                        (
                            capture_group,
                            raw_label,
                            canonical_label,
                            parsed["observed_cipher"],
                        )
                    ] += 1
                if parsed["capture_id"]:
                    group_capture_ids[capture_group].add(parsed["capture_id"])
                    capture_to_groups[parsed["capture_id"]].add(capture_group)
    except BaseException:
        for handle in handles:
            if not handle.closed:
                handle.close()
        raise
    finalize_atomic_csvs(writers)

    class_rows = []
    for label in local41:
        class_rows.append(
            {
                "canonical_label": label,
                "label_id_official40": official_ids.get(label, ""),
                "label_id_local41": local_ids[label],
                "official40_label": bool_text(label in official_ids),
                "total_files": class_counts[label],
                **{
                    f"{group}_files": class_group_counts[(label, group)]
                    for group in CAPTURE_GROUPS
                },
            }
        )
    class_fields = (
        "canonical_label",
        "label_id_official40",
        "label_id_local41",
        "official40_label",
        "total_files",
        *(f"{group}_files" for group in CAPTURE_GROUPS),
    )
    write_csv_atomic(output_dir / "class_summary.csv", class_fields, class_rows)

    overlap_rows: list[dict[str, object]] = []
    for group in CAPTURE_GROUPS:
        overlap_rows.append(
            {
                "scope": "single_group",
                "group_a": group,
                "group_b": "",
                "capture_id_count": len(group_capture_ids[group]),
            }
        )
    for group_a, group_b in combinations(CAPTURE_GROUPS, 2):
        overlap_rows.append(
            {
                "scope": "pairwise_overlap",
                "group_a": group_a,
                "group_b": group_b,
                "capture_id_count": len(
                    group_capture_ids[group_a] & group_capture_ids[group_b]
                ),
            }
        )
    overlap_rows.extend(
        [
            {
                "scope": "all_unique",
                "group_a": "ALL",
                "group_b": "",
                "capture_id_count": len(capture_to_groups),
            },
            {
                "scope": "multi_group",
                "group_a": "ALL",
                "group_b": "",
                "capture_id_count": sum(
                    len(groups) > 1 for groups in capture_to_groups.values()
                ),
            },
        ]
    )
    write_csv_atomic(
        output_dir / "capture_overlap_summary.csv",
        ("scope", "group_a", "group_b", "capture_id_count"),
        overlap_rows,
    )

    mismatch_rows = [
        {
            "capture_group_directory": group,
            "directory_label_raw": raw_label,
            "canonical_label": canonical_label,
            "observed_cipher": observed_cipher,
            "flow_count": count,
        }
        for (group, raw_label, canonical_label, observed_cipher), count in sorted(
            mismatch_details.items()
        )
    ]
    write_csv_atomic(
        output_dir / "packaging_anomaly_summary.csv",
        (
            "capture_group_directory",
            "directory_label_raw",
            "canonical_label",
            "observed_cipher",
            "flow_count",
        ),
        mismatch_rows,
    )

    label_maps = {
        "official_compatible_40": {
            "definition": "raw class-directory labels common to all four capture groups",
            "class_count": len(official40),
            "label_to_id": official_ids,
        },
        "local_audit_41": {
            "definition": (
                "local normalized view; getpocket.com retained and the anomalous "
                "aes-256-gcm/chacha20 directory normalized to getpocket.com"
            ),
            "class_count": len(local41),
            "label_to_id": local_ids,
        },
    }
    write_json_atomic(output_dir / "label_maps.json", label_maps)

    summary_path = output_dir / "labeling_summary.md"
    summary_text = f"""# CipherSpectrum Labeling Summary

Generated: {datetime.now(timezone.utc).isoformat()}

## Outcome

- Raw PCAP files scanned read-only: **{total_files:,}** ({total_bytes:,} bytes).
- Official-compatible view: **{len(official40)} classes / {official_count:,} flows**.
- Complete local audit view: **{len(local41)} normalized classes / {local_count:,} candidate flows**.
- Rows requiring manual/release-compatibility review: **{review_count:,}**.
- Filename parse errors: **{parse_errors:,}**; invalid PCAP magic values: **{bad_magic:,}**.
- Capture-group/cipher mismatches: **{group_mismatches:,}**.
- Of those mismatches, **{official40_group_mismatches:,}** retain an
  official-compatible class label; the mismatch concerns cipher packaging,
  not the SNI/directory class assignment.
- Filename `visited_domain` differs from the canonical class label for
  **{visited_domain_label_differences:,}** flows.  This is expected when a page
  contacts third-party TLS services and is why that field is not the label.

## Label policy

The class label is derived from the immediate class directory, which is the
release's SNI-oriented grouping.  The visited/page domain inside a PCAP filename
is retained as `visited_domain` metadata and is **not** used as the class label.

The official-compatible 40-class manifest contains only labels appearing as
class directories in every capture group.  This avoids silently treating a
local release discrepancy as an official class.

The full manifest additionally preserves `getpocket.com`.  The literal
`aes-256-gcm/aes-256-gcm/chacha20/` directory is normalized to
`getpocket.com` only in this audit view; those rows remain marked for review.
The `chacha20-poly1305/getpocket.com` material also reports an observed
`aes-256` cipher, so both cross-group packaging mismatches stay explicit.

## Primary references

- Official CipherSpectrum dataset page: https://cgi.cse.unsw.edu.au/~cspectrum/
- CipherSpectrum paper, including the traffic-labeling description:
  https://arxiv.org/html/2503.20093v4

## Leakage safeguard

`capture_id` and the identical `split_group_id` are exported for every flow.
Future train/validation/test creation must group on this field.  Flow-level
random splitting would place related split sessions from the same browser
capture in different partitions because capture IDs overlap across cipher/mix
groups.

## Recommended use

1. Use `cipherspectrum_official40_manifest.csv` for the conservative first
   class-label experiment.
2. If an experiment also assumes cipher-group purity, filter rows with
   `manual_review_required=true`; some official-class rows have a packaging
   mismatch even though their class label remains usable.
3. Treat the 41-class view as a local-release extension until its extra class
   and two capture-group anomalies receive a separate SNI/package audit.
4. Do not infer authoritative semantics beyond website/SNI class membership;
   these labels do not identify attack families or independent endpoints.
"""
    temp_summary = summary_path.with_suffix(".md.tmp")
    temp_summary.write_text(summary_text, encoding="utf-8")
    temp_summary.replace(summary_path)

    output_files = [
        output_dir / "cipherspectrum_label_manifest.csv",
        output_dir / "cipherspectrum_official40_manifest.csv",
        output_dir / "manual_review_inventory.csv",
        output_dir / "class_summary.csv",
        output_dir / "capture_overlap_summary.csv",
        output_dir / "packaging_anomaly_summary.csv",
        output_dir / "label_maps.json",
        summary_path,
    ]
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_argument": str(dataset_root.resolve()),
        "payload_root": str(payload_root),
        "output_dir": str(output_dir),
        "raw_access_mode": "read_only",
        "dataset_fingerprint_algorithm": "sha256(relative_path\\0size\\0mtime_ns\\n)",
        "dataset_fingerprint_sha256": fingerprint.hexdigest(),
        "total_files": total_files,
        "total_bytes": total_bytes,
        "official40_classes": len(official40),
        "official40_flows": official_count,
        "local41_classes": len(local41),
        "local41_candidate_flows": local_count,
        "manual_review_rows": review_count,
        "filename_parse_errors": parse_errors,
        "invalid_pcap_magic": bad_magic,
        "capture_group_mismatches": group_mismatches,
        "official40_capture_group_mismatches": official40_group_mismatches,
        "visited_domain_label_differences": visited_domain_label_differences,
        "raw_directory_label_counts": dict(sorted(raw_label_counts.items())),
        "label_status_counts": dict(sorted(label_status_counts.items())),
        "output_sha256": {
            path.name: sha256_file(path) for path in output_files
        },
    }
    write_json_atomic(output_dir / "run_metadata.json", metadata)
    return metadata


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        metadata = build_manifests(args.dataset_root, args.output_dir)
    except Exception as exc:  # keep CLI failure concise and non-partial
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
