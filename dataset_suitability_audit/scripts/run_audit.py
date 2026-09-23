#!/usr/bin/env python3
"""Read-only suitability audit for three local traffic datasets.

The script never writes below a dataset root. PCAP validation reads only the
global four-byte magic for every capture and runs capinfos/tshark on a small,
deterministic sample. CSVs are header/sample-read only; authoritative CICDDoS
row counts and labels are recovered from its existing local audit index.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow.parquet as pq


AUDIT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = AUDIT_ROOT.parent
DEFAULT_OUTPUT = AUDIT_ROOT / "outputs"

DATASETS = {
    "CipherSpectrum": Path(
        "/home/birkenwald/data/SEU-WXY/WXY-SEU/TrafficClassifier/"
        "Dataset/ipherSpectrum(sok)"
    ),
    "CSTNET-TLS1.3": Path(
        "/home/birkenwald/data/SEU-WXY/WXY-SEU/TrafficClassifier/"
        "Dataset/CSTNET-TLS1.3"
    ),
    "CICDDoS2019": Path(
        "/home/birkenwald/data/SEU-WXY/WXY-SEU/TrafficClassifier/"
        "Dataset/CICDDoS2019"
    ),
}

PCAP_MAGIC = {
    b"\xd4\xc3\xb2\xa1": "classic_pcap_le_us",
    b"\xa1\xb2\xc3\xd4": "classic_pcap_be_us",
    b"\x4d\x3c\xb2\xa1": "classic_pcap_le_ns",
    b"\xa1\xb2\x3c\x4d": "classic_pcap_be_ns",
    b"\x0a\x0d\x0d\x0a": "pcapng",
}

TABLE_EXTENSIONS = {".csv", ".parquet", ".feather", ".h5", ".hdf5", ".npy", ".npz"}
ARCHIVE_EXTENSIONS = {".zip", ".rar", ".7z"}
CIPHER_GROUPS = {"aes-128-gcm", "aes-256-gcm", "chacha20-poly1305", "mix"}
CIPHER_NAME_RE = re.compile(
    r"^traffic_(?P<date>\d{4}-\d{2}-\d{2})_(?P<domain>.+?)_"
    r"(?P<cipher>aes-128|aes-256|chacha20|mix)_(?:chromium|firefox)_",
    re.IGNORECASE,
)


def extension_of(path: Path) -> str:
    """Return a stable extension, retaining malformed lock-file suffixes."""
    name = path.name.lower()
    if name.endswith(".csv#"):
        return ".csv#"
    return path.suffix.lower() or "[no_extension]"


def is_macos_sidecar(path: Path) -> bool:
    return "__MACOSX" in path.parts or path.name.startswith("._") or path.name == ".DS_Store"


def classify_file(path: Path, ext: str) -> tuple[str, str]:
    if is_macos_sidecar(path):
        return "macOS metadata sidecar", "metadata_sidecar_not_data"
    if ext in {".pcap", ".pcapng"}:
        return "packet capture", "raw_or_pre_split_packet_capture"
    if ext == ".csv":
        return "CSV table", "flow_features_labels_or_manifest"
    if ext == ".parquet":
        return "Parquet table", "derived_flow_or_label_table"
    if ext == ".npy":
        return "NumPy array", "derived_packet_or_flow_representation"
    if ext == ".npz":
        return "NumPy archive", "derived_representation"
    if ext in {".h5", ".hdf5"}:
        return "HDF5", "derived_table_or_representation"
    if ext in ARCHIVE_EXTENSIONS:
        return "archive", "archive_not_extracted_by_audit"
    if ext in {".md", ".txt", ".json", ".log"}:
        return "text/metadata", "documentation_or_audit_metadata"
    return "other", "unknown"


def pcap_magic(path: Path) -> tuple[bool, str, str]:
    try:
        with path.open("rb") as handle:
            magic = handle.read(4)
    except OSError as exc:
        return False, "UNREADABLE", str(exc)
    fmt = PCAP_MAGIC.get(magic)
    if fmt:
        return True, fmt, ""
    return False, "INVALID_MAGIC", magic.hex()


def cipher_metadata(path: Path) -> dict[str, str]:
    match = CIPHER_NAME_RE.match(path.name)
    group = next((part for part in path.parts if part in CIPHER_GROUPS), "UNKNOWN")
    if not match:
        return {"capture_group": group, "domain": "UNPARSED", "date": "UNKNOWN", "cipher": "UNKNOWN"}
    return {
        "capture_group": group,
        "domain": match.group("domain").lower(),
        "date": match.group("date"),
        "cipher": match.group("cipher").lower(),
    }


def npy_header(path: Path) -> tuple[tuple[int, ...], str, bool]:
    from numpy.lib import format as npfmt

    with path.open("rb") as handle:
        version = npfmt.read_magic(handle)
        if version == (1, 0):
            shape, fortran, dtype = npfmt.read_array_header_1_0(handle)
        elif version in {(2, 0), (3, 0)}:
            shape, fortran, dtype = npfmt.read_array_header_2_0(handle)
        else:
            raise ValueError(f"unsupported npy version: {version}")
    return tuple(shape), str(dtype), bool(fortran)


def atomic_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)


def atomic_text(path: Path, text: str) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(text, encoding="utf-8")
    os.replace(temp, path)


def walk_dicts(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from walk_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_dicts(child)


def load_cic_csv_cache(root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    audit_path = root / "labeling_work/audit/dataset_audit.json"
    if not audit_path.exists():
        return {}
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for item in walk_dicts(payload):
        if not {"columns_raw", "row_count", "label_counts"}.issubset(item):
            continue
        source = str(item.get("path", "")).replace("\\", "/")
        day = "01-12" if "CSV-01-12" in source else "03-11" if "CSV-03-11" in source else "UNKNOWN"
        result[(day, Path(source).name)] = item
    return result


def local_cic_day(path: Path) -> str:
    text = path.as_posix()
    return "01-12" if "CSV-01-12" in text else "03-11" if "CSV-03-11" in text else "UNKNOWN"


def schema_flags(columns: list[str], path: Path) -> dict[str, bool]:
    normal = {c.strip().lower().replace("_", " ") for c in columns}

    def has(*names: str) -> bool:
        return any(name in normal for name in names)

    lower_name = path.name.lower()
    return {
        "has_label": has("label", "class", "target") or lower_name.startswith("y_"),
        "has_flow_id": has("flow id", "flow id"),
        "has_src_ip": has("source ip", "src ip"),
        "has_dst_ip": has("destination ip", "dst ip"),
        "has_src_port": has("source port", "src port"),
        "has_dst_port": has("destination port", "dst port"),
        "has_protocol": has("protocol"),
        "has_timestamp": has("timestamp", "start time", "start time"),
        "has_duration": has("flow duration", "duration us", "duration"),
        "has_packet_count": any("packet" in c and ("count" in c or "total" in c) for c in normal),
        "has_byte_count": any("byte" in c or "length of" in c for c in normal),
        "has_packet_sequence": any("sequence" in c or "direction" in c or "datagram" in c for c in normal),
        "has_packet_size_sequence": any("len" in c or "length sequence" in c for c in normal),
        "has_packet_direction": any("direction" in c for c in normal),
        "has_raw_or_payload_bytes": any("raw" in c or "payload" in c or "datagram" in c for c in normal),
    }


def inventory_all() -> tuple[list[dict[str, Any]], dict[str, list[Path]], dict[str, dict[str, int]]]:
    rows: list[dict[str, Any]] = []
    files_by_dataset: dict[str, list[Path]] = {}
    totals: dict[str, dict[str, int]] = {}
    for dataset, root in DATASETS.items():
        if not root.is_dir():
            raise FileNotFoundError(root)
        files = sorted(path for path in root.rglob("*") if path.is_file())
        files_by_dataset[dataset] = files
        logical = 0
        allocated = 0
        for path in files:
            stat = path.stat()
            logical += stat.st_size
            allocated += getattr(stat, "st_blocks", 0) * 512
            ext = extension_of(path)
            file_type, role = classify_file(path, ext)
            rows.append(
                {
                    "dataset": dataset,
                    "relative_path": path.relative_to(root).as_posix(),
                    "extension": ext,
                    "size_bytes": stat.st_size,
                    "size_gib": f"{stat.st_size / 2**30:.9f}",
                    "file_type": file_type,
                    "suspected_role": role,
                }
            )
        totals[dataset] = {"files": len(files), "logical_bytes": logical, "allocated_bytes": allocated}
    return rows, files_by_dataset, totals


def audit_pcaps(files_by_dataset: dict[str, list[Path]]) -> tuple[list[dict[str, Any]], dict[str, list[Path]]]:
    rows: list[dict[str, Any]] = []
    valid: dict[str, list[Path]] = defaultdict(list)
    for dataset, files in files_by_dataset.items():
        root = DATASETS[dataset]
        for path in files:
            if extension_of(path) not in {".pcap", ".pcapng"}:
                continue
            readable, fmt, detail = pcap_magic(path)
            sidecar = is_macos_sidecar(path)
            if readable and not sidecar:
                valid[dataset].append(path)
            rows.append(
                {
                    "dataset": dataset,
                    "relative_path": path.relative_to(root).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "macos_sidecar": sidecar,
                    "readable_magic": readable,
                    "capture_format": fmt,
                    "detail": detail,
                }
            )
    return rows, valid


def select_pcap_samples(dataset: str, paths: list[Path], maximum: int = 16) -> list[Path]:
    if not paths:
        return []
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        if dataset == "CipherSpectrum":
            label = cipher_metadata(path)["capture_group"]
        elif dataset == "CSTNET-TLS1.3":
            label = path.parent.name
        else:
            label = "all"
        groups[label].append(path)
    labels = sorted(groups)
    if len(labels) > maximum:
        indices = sorted({round(i * (len(labels) - 1) / (maximum - 1)) for i in range(maximum)})
        labels = [labels[i] for i in indices]
    return [sorted(groups[label])[0] for label in labels]


def parse_capinfos(output: str) -> tuple[str, str, str]:
    packet_count = ""
    first = ""
    last = ""
    for line in output.splitlines():
        key, _, value = line.partition(":")
        key = key.strip().lower()
        if key == "number of packets":
            packet_count = value.strip().split()[0]
        elif key == "first packet time":
            first = value.strip()
        elif key == "last packet time":
            last = value.strip()
    return packet_count, first, last


def pcap_sample_audit(valid: dict[str, list[Path]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset, paths in valid.items():
        root = DATASETS[dataset]
        for path in select_pcap_samples(dataset, paths):
            cap = subprocess.run(
                ["capinfos", "-M", "-c", "-a", "-e", "-S", str(path)],
                text=True,
                capture_output=True,
                timeout=30,
                check=False,
            )
            packet_count, first, last = parse_capinfos(cap.stdout)
            tshark = subprocess.run(
                [
                    "tshark", "-n", "-r", str(path), "-c", "500", "-Y", "tls",
                    "-T", "fields", "-e", "tls.handshake.extensions.supported_version",
                    "-e", "tls.handshake.version",
                ],
                text=True,
                capture_output=True,
                timeout=30,
                check=False,
            )
            tls_lines = [line.strip() for line in tshark.stdout.splitlines() if line.strip()]
            tokens = sorted({token for line in tls_lines for token in re.split(r"[\t,]+", line) if token})
            rows.append(
                {
                    "dataset": dataset,
                    "relative_path": path.relative_to(root).as_posix(),
                    "capinfos_readable": cap.returncode == 0,
                    "packet_count": packet_count,
                    "first_packet_timestamp_epoch": first,
                    "last_packet_timestamp_epoch": last,
                    "tshark_returncode": tshark.returncode,
                    "tls_filtered_lines": len(tls_lines),
                    "tls_version_fields": json.dumps(tokens),
                    "sample_only": True,
                }
            )
    return rows


def read_csv_header_sample(path: Path, sample_n: int = 3) -> tuple[list[str], list[list[str]], str]:
    try:
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader)
            samples = []
            for _ in range(sample_n):
                try:
                    samples.append(next(reader))
                except StopIteration:
                    break
        return header, samples, ""
    except Exception as exc:  # recorded rather than silently skipped
        return [], [], f"{type(exc).__name__}: {exc}"


def small_parquet_sample(path: Path, columns: list[str]) -> str:
    try:
        batch = next(pq.ParquetFile(path).iter_batches(batch_size=3, columns=columns[:12]), None)
        if batch is None:
            return "[]"
        return json.dumps(batch.to_pylist(), default=str, ensure_ascii=False)
    except Exception as exc:
        return f"SAMPLE_ERROR:{type(exc).__name__}:{exc}"


def audit_tables(files_by_dataset: dict[str, list[Path]]) -> tuple[list[dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    cic_cache = load_cic_csv_cache(DATASETS["CICDDoS2019"])
    sampled_parquet_schema: set[tuple[str, ...]] = set()
    for dataset, files in files_by_dataset.items():
        root = DATASETS[dataset]
        for path in files:
            ext = extension_of(path)
            if ext not in TABLE_EXTENSIONS:
                continue
            base: dict[str, Any] = {
                "dataset": dataset,
                "relative_path": path.relative_to(root).as_posix(),
                "extension": ext,
                "size_bytes": path.stat().st_size,
                "rows": "UNKNOWN",
                "shape": "",
                "dtype": "",
                "columns": "[]",
                "label_column": "",
                "sample_rows": "[]",
                "read_status": "OK",
                "row_count_source": "",
            }
            columns: list[str] = []
            if ext == ".csv":
                columns, samples, error = read_csv_header_sample(path)
                base["columns"] = json.dumps(columns, ensure_ascii=False)
                base["sample_rows"] = json.dumps(samples, ensure_ascii=False)
                if error:
                    base["read_status"] = error
                cache = cic_cache.get((local_cic_day(path), path.name))
                if cache:
                    base["rows"] = int(cache["row_count"])
                    base["row_count_source"] = "existing_local_streaming_audit"
                    base["label_column"] = next((c for c in columns if c.strip().lower() == "label"), "")
                elif path.stat().st_size <= 64 * 2**20:
                    with path.open("rb") as handle:
                        base["rows"] = max(0, sum(chunk.count(b"\n") for chunk in iter(lambda: handle.read(2**20), b"")) - 1)
                    base["row_count_source"] = "streamed_newline_count"
            elif ext == ".parquet":
                try:
                    parquet = pq.ParquetFile(path)
                    columns = list(parquet.schema_arrow.names)
                    base["rows"] = parquet.metadata.num_rows
                    base["row_count_source"] = "parquet_footer"
                    base["columns"] = json.dumps(columns, ensure_ascii=False)
                    base["dtype"] = str(parquet.schema_arrow)
                    key = tuple(columns)
                    if key not in sampled_parquet_schema:
                        base["sample_rows"] = small_parquet_sample(path, columns)
                        sampled_parquet_schema.add(key)
                    base["label_column"] = next((c for c in columns if c.strip().lower() == "label"), "")
                except Exception as exc:
                    base["read_status"] = f"{type(exc).__name__}: {exc}"
            elif ext == ".npy":
                try:
                    shape, dtype, fortran = npy_header(path)
                    base["rows"] = shape[0] if shape else 1
                    base["row_count_source"] = "npy_header"
                    base["shape"] = json.dumps(shape)
                    base["dtype"] = dtype
                    base["columns"] = json.dumps([path.stem])
                    columns = [path.stem]
                    if path.name.startswith("y_"):
                        base["label_column"] = path.stem
                    if "object" not in dtype:
                        array = np.load(path, mmap_mode="r")
                        preview = array[: min(3, len(array))].tolist()
                        base["sample_rows"] = json.dumps(preview, default=str, ensure_ascii=False)[:12000]
                    if fortran:
                        base["read_status"] = "OK_FORTRAN_ORDER"
                except Exception as exc:
                    base["read_status"] = f"{type(exc).__name__}: {exc}"
            flags = schema_flags(columns, path)
            base.update(flags)
            rows.append(base)
    return rows, cic_cache


def audit_archives(files_by_dataset: dict[str, list[Path]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset, files in files_by_dataset.items():
        root = DATASETS[dataset]
        for path in files:
            ext = extension_of(path)
            if ext not in ARCHIVE_EXTENSIONS:
                continue
            row: dict[str, Any] = {
                "dataset": dataset,
                "relative_path": path.relative_to(root).as_posix(),
                "extension": ext,
                "size_bytes": path.stat().st_size,
                "valid_archive": False,
                "member_count": "UNKNOWN",
                "uncompressed_bytes": "UNKNOWN",
                "member_extensions": "{}",
                "contains_pcap": "UNKNOWN",
                "contains_tables": "UNKNOWN",
                "read_status": "UNSUPPORTED_OR_INVALID",
            }
            if ext == ".zip" and zipfile.is_zipfile(path):
                try:
                    with zipfile.ZipFile(path) as archive:
                        infos = archive.infolist()
                    counts = Counter(Path(info.filename).suffix.lower() or "[no_extension]" for info in infos if not info.is_dir())
                    row.update(
                        {
                            "valid_archive": True,
                            "member_count": len(infos),
                            "uncompressed_bytes": sum(info.file_size for info in infos),
                            "member_extensions": json.dumps(dict(sorted(counts.items()))),
                            "contains_pcap": any(extn in {".pcap", ".pcapng"} for extn in counts),
                            "contains_tables": any(extn in TABLE_EXTENSIONS or extn == ".tsv" for extn in counts),
                            "read_status": "LISTED_WITHOUT_EXTRACTION",
                        }
                    )
                except Exception as exc:
                    row["read_status"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
    return rows


def cic_csv_class_counts(cache: dict[tuple[str, str], dict[str, Any]]) -> tuple[Counter, Counter, dict[str, set[str]]]:
    counts: Counter = Counter()
    source_files: Counter = Counter()
    days: dict[str, set[str]] = defaultdict(set)
    for (day, filename), item in cache.items():
        for label, count in item.get("label_counts", {}).items():
            count = int(count)
            counts[label] += count
            if count:
                source_files[label] += 1
                days[label].add(day)
    return counts, source_files, days


def cic_capture_class_counts(root: Path) -> tuple[Counter, dict[str, set[str]], Counter, dict[str, Counter]]:
    capture_files: Counter = Counter()
    days: dict[str, set[str]] = defaultdict(set)
    matched_counts: Counter = Counter()
    capture_distributions: dict[str, Counter] = defaultdict(Counter)
    for path in sorted((root / "labeling_work/labeled_flows").rglob("*.summary.json")):
        if path.name.endswith(".extract_summary.json"):
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        day = path.parent.name
        capture = path.name.removesuffix(".summary.json")
        for label, count in payload.get("label_counts", {}).items():
            if label in {"UNMATCHED", "AMBIGUOUS"}:
                continue
            count = int(count)
            if count:
                capture_files[label] += 1
                days[label].add(day)
                matched_counts[label] += count
                capture_distributions[label][capture] += count
    return capture_files, days, matched_counts, capture_distributions


def make_class_outputs(valid_pcaps: dict[str, list[Path]], cic_cache: dict[tuple[str, str], dict[str, Any]]):
    class_rows: list[dict[str, Any]] = []
    leakage_rows: list[dict[str, Any]] = []

    # CipherSpectrum: the on-disk hierarchy provides capture group and website labels.
    cipher_domain: Counter = Counter()
    cipher_group: Counter = Counter()
    cipher_domain_days: dict[str, set[str]] = defaultdict(set)
    unparsed = 0
    for path in valid_pcaps.get("CipherSpectrum", []):
        meta = cipher_metadata(path)
        cipher_group[meta["capture_group"]] += 1
        cipher_domain[meta["domain"]] += 1
        cipher_domain_days[meta["domain"]].add(meta["date"])
        unparsed += meta["domain"] == "UNPARSED"
    for level, counts, semantics in (
        ("capture_group", cipher_group, "cipher-specific or mixed capture group"),
        ("website_domain", cipher_domain, "website/domain represented by per-flow PCAP filenames"),
    ):
        for label, count in sorted(counts.items()):
            class_rows.append(
                {
                    "dataset": "CipherSpectrum", "label_level": level, "class_name": label,
                    "sample_count": count, "source_file_count": count, "source_capture_count": count,
                    "label_semantics": semantics,
                    "label_provenance": "LABEL_PROVENANCE_UNCLEAR: on-disk directory/filename organization; no standalone authoritative label map/readme found",
                }
            )
    for label, count in sorted(cipher_domain.items()):
        leakage_rows.append(
            {
                "dataset": "CipherSpectrum", "class_name": label,
                "num_source_files": count, "num_pcaps": count,
                "num_days": len(cipher_domain_days[label] - {"UNKNOWN"}),
                "capture_class_entanglement": "HIGH_DIRECTORY_BOUND",
                "time_class_entanglement": "LOWER_RISK_MULTIPLE_DATES" if len(cipher_domain_days[label]) > 1 else "UNKNOWN",
                "endpoint_class_entanglement": "HIGH_RISK_DOMAIN_IS_LABEL",
                "dominant_source_ratio": "UNKNOWN_NOT_FLOW_PARSED", "risk_level": "HIGH",
            }
        )

    # CSTNET: domain directory is the raw-PCAP class; numeric IDs have no local name map.
    cst_site = Counter(path.parent.name for path in valid_pcaps.get("CSTNET-TLS1.3", []))
    for label, count in sorted(cst_site.items()):
        class_rows.append(
            {
                "dataset": "CSTNET-TLS1.3", "label_level": "website_domain_pcap", "class_name": label,
                "sample_count": count, "source_file_count": count, "source_capture_count": count,
                "label_semantics": "TLS website/service domain",
                "label_provenance": "official-style directory organization plus local readme declaring 120 classes",
            }
        )
        leakage_rows.append(
            {
                "dataset": "CSTNET-TLS1.3", "class_name": label,
                "num_source_files": count, "num_pcaps": count, "num_days": "UNKNOWN",
                "capture_class_entanglement": "HIGH_DIRECTORY_BOUND",
                "time_class_entanglement": "UNKNOWN_NO_CAPTURE_DATE_MANIFEST",
                "endpoint_class_entanglement": "UNKNOWN_NOT_FLOW_PARSED",
                "dominant_source_ratio": "UNKNOWN_NOT_FLOW_PARSED", "risk_level": "MEDIUM_HIGH",
            }
        )
    cst_root = DATASETS["CSTNET-TLS1.3"]
    for level, folder in (("numeric_flow_label", "flow_500"), ("numeric_packet_label", "packet_5000")):
        aggregate: Counter = Counter()
        source_n = 0
        for path in sorted(cst_root.rglob(f"{folder}/y_*.npy")):
            array = np.load(path, mmap_mode="r")
            aggregate.update(Counter(int(value) for value in array.tolist()))
            source_n += 1
        for label, count in sorted(aggregate.items()):
            class_rows.append(
                {
                    "dataset": "CSTNET-TLS1.3", "label_level": level, "class_name": str(label),
                    "sample_count": count, "source_file_count": source_n, "source_capture_count": "UNKNOWN",
                    "label_semantics": "numeric website/service class ID",
                    "label_provenance": "y_*.npy arrays; numeric-ID-to-domain map not found locally",
                }
            )

    # CICDDoS: official row-level Label is authoritative; filenames are not labels.
    cic_counts, cic_sources, cic_days = cic_csv_class_counts(cic_cache)
    cap_sources, cap_days, matched_counts, cap_distributions = cic_capture_class_counts(DATASETS["CICDDoS2019"])
    for label, count in sorted(cic_counts.items()):
        class_rows.append(
            {
                "dataset": "CICDDoS2019", "label_level": "official_flow_label", "class_name": label,
                "sample_count": count, "source_file_count": cic_sources[label],
                "source_capture_count": cap_sources[label],
                "label_semantics": "BENIGN or DDoS attack family/subtype",
                "label_provenance": "row-level official CSV Label recovered by existing local streaming audit",
            }
        )
        distribution = cap_distributions[label]
        dominant = max(distribution.values()) / sum(distribution.values()) if distribution else None
        one_day = len(cic_days[label]) == 1
        leakage_rows.append(
            {
                "dataset": "CICDDoS2019", "class_name": label,
                "num_source_files": cic_sources[label], "num_pcaps": cap_sources[label],
                "num_days": len(cic_days[label]),
                "capture_class_entanglement": "HIGH" if cic_sources[label] <= 2 and label != "BENIGN" else "MEDIUM",
                "time_class_entanglement": "HIGH_SINGLE_DAY" if one_day and label != "BENIGN" else "MEDIUM_MULTI_DAY",
                "endpoint_class_entanglement": "UNKNOWN_NOT_RESCANNED",
                "dominant_source_ratio": f"{dominant:.9f}" if dominant is not None else "UNKNOWN",
                "risk_level": "HIGH" if label != "BENIGN" and (one_day or cic_sources[label] <= 2) else "MEDIUM",
                "derived_matched_flow_count": matched_counts[label],
                "capture_day_count_from_derived": len(cap_days[label]),
            }
        )
    return class_rows, leakage_rows, unparsed


def primary_class_rows(class_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    primary_level = {
        "CipherSpectrum": "website_domain",
        "CSTNET-TLS1.3": "website_domain_pcap",
        "CICDDoS2019": "official_flow_label",
    }
    return [row for row in class_rows if row["label_level"] == primary_level[row["dataset"]]]


def class_balance_rows(class_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    primary = primary_class_rows(class_rows)
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in primary:
        by_dataset[row["dataset"]].append(row)
    result = []
    for dataset, rows in by_dataset.items():
        counts = [int(row["sample_count"]) for row in rows]
        for row in rows:
            count = int(row["sample_count"])
            result.append(
                {
                    "dataset": dataset, "label_level": row["label_level"], "class_name": row["class_name"],
                    "sample_count": count, "lt_100": count < 100, "lt_500": count < 500,
                    "lt_1000": count < 1000, "lt_5000": count < 5000,
                    "dataset_num_classes": len(rows), "dataset_min": min(counts),
                    "dataset_median": statistics.median(counts), "dataset_max": max(counts),
                }
            )
    return result


def fixed_suitability_rows(primary: list[dict[str, Any]], valid_pcaps: dict[str, list[Path]]):
    class_count = Counter(row["dataset"] for row in primary)
    open_set = [
        {
            "dataset": "CipherSpectrum", "num_candidate_classes": class_count["CipherSpectrum"],
            "class_semantics": "website/domain (preferred); capture group is only four classes",
            "can_hold_out_classes": "YES_WITH_HIGH_SHORTCUT_RISK",
            "reason": f"{class_count['CipherSpectrum']} domain classes and many per-flow PCAPs, but domain/endpoints and directory/capture are intrinsically entangled",
        },
        {
            "dataset": "CSTNET-TLS1.3", "num_candidate_classes": class_count["CSTNET-TLS1.3"],
            "class_semantics": "TLS website/service domain",
            "can_hold_out_classes": "YES",
            "reason": "120 classes with raw per-flow captures; severe tail imbalance requires predeclared eligibility rules",
        },
        {
            "dataset": "CICDDoS2019", "num_candidate_classes": class_count["CICDDoS2019"],
            "class_semantics": "BENIGN plus DDoS attack family/subtype",
            "can_hold_out_classes": "YES_FEATURE_LEVEL_ONLY_ON_CURRENT_DISK",
            "reason": "19 official labels support class hold-out, but current directory lacks raw captures and attack labels are day/capture entangled",
        },
    ]
    opendetect = [
        {
            "dataset": "CipherSpectrum", "status": "A. DIRECTLY_READY",
            "pcap_available": bool(valid_pcaps.get("CipherSpectrum")), "precise_flow_label": True,
            "packet_order": True, "packet_bytes": True, "flow_packet_sequence": True,
            "reason": "each valid file is already a per-flow PCAP under capture-group/domain organization; no CSV join is required",
        },
        {
            "dataset": "CSTNET-TLS1.3", "status": "A. DIRECTLY_READY",
            "pcap_available": bool(valid_pcaps.get("CSTNET-TLS1.3")), "precise_flow_label": True,
            "packet_order": True, "packet_bytes": True, "flow_packet_sequence": True,
            "reason": "each valid PCAP is a flow-like capture under a domain class; raw bytes and order remain available",
        },
        {
            "dataset": "CICDDoS2019", "status": "C. PARTIALLY_READY",
            "pcap_available": False, "precise_flow_label": True,
            "packet_order": False, "packet_bytes": False, "flow_packet_sequence": False,
            "reason": "official flow CSV and derived aggregate Parquet exist, but current local directory contains no PCAP/PCAPNG and no packet-byte sequence",
        },
    ]
    mapping = [
        {
            "dataset": "CipherSpectrum", "mapping_keys_available": "per-flow PCAP path + directory + filename metadata",
            "timestamp_precision": "PCAP precision retained", "flow_id_available": "file path acts as flow identity",
            "bidirectional_or_unidirectional": "requires packet-level semantic confirmation",
            "expected_mapping_difficulty": "LOW_NO_EXTERNAL_JOIN",
        },
        {
            "dataset": "CSTNET-TLS1.3", "mapping_keys_available": "per-flow PCAP parent domain",
            "timestamp_precision": "PCAP precision retained", "flow_id_available": "relative PCAP path",
            "bidirectional_or_unidirectional": "requires packet-level semantic confirmation",
            "expected_mapping_difficulty": "LOW_NO_EXTERNAL_JOIN",
        },
        {
            "dataset": "CICDDoS2019", "mapping_keys_available": "CSV 5-tuple + timestamp + duration + Flow ID; raw capture absent now",
            "timestamp_precision": "microseconds; existing audit found day-specific +4h/+3h offsets",
            "flow_id_available": "official CSV Flow ID and derived flow_id",
            "bidirectional_or_unidirectional": "existing matcher supports direct and reverse tuples",
            "expected_mapping_difficulty": "HIGH_IF_RAW_PCAP_RESTORED; IMPOSSIBLE_ON_CURRENT_RAW-FREE_DIRECTORY",
        },
    ]
    unknown_free = [
        {
            "dataset": "CipherSpectrum", "candidate_classes": class_count["CipherSpectrum"],
            "quantity_grade": "GOOD", "strict_unknown_free_feasible": "YES_TECHNICALLY",
            "scientific_caveat": "domain/endpoints are the label; class hold-out risks endpoint novelty shortcut",
        },
        {
            "dataset": "CSTNET-TLS1.3", "candidate_classes": class_count["CSTNET-TLS1.3"],
            "quantity_grade": "GOOD", "strict_unknown_free_feasible": "YES",
            "scientific_caveat": "predeclare minimum class support; numeric ID-to-domain map is absent",
        },
        {
            "dataset": "CICDDoS2019", "candidate_classes": class_count["CICDDoS2019"],
            "quantity_grade": "GOOD", "strict_unknown_free_feasible": "YES_FEATURE_LEVEL; NO_RAW-BYTE_OPEN-DETECT_NOW",
            "scientific_caveat": "attack/day/capture entanglement makes random-flow splitting invalid for scientific claims",
        },
    ]
    openness = [
        {"dataset": "CipherSpectrum", "candidate_classes": class_count["CipherSpectrum"], "low_unknown_ratio": "84/1", "medium_unknown_ratio": "77/8", "high_unknown_ratio": "64/21", "actual_split_created": False},
        {"dataset": "CSTNET-TLS1.3", "candidate_classes": class_count["CSTNET-TLS1.3"], "low_unknown_ratio": "119/1", "medium_unknown_ratio": "108/12", "high_unknown_ratio": "90/30", "actual_split_created": False},
        {"dataset": "CICDDoS2019", "candidate_classes": class_count["CICDDoS2019"], "low_unknown_ratio": "18/1", "medium_unknown_ratio": "16/3", "high_unknown_ratio": "14/5", "actual_split_created": False},
    ]
    return open_set, opendetect, mapping, unknown_free, openness


SCORES = {
    "CipherSpectrum": {
        "raw_packet_availability": 2, "label_quality": 1, "number_of_classes": 2,
        "per_class_sample_size": 2, "flow_pcap_alignment": 2,
        "strict_unknown_free_feasibility": 2, "capture_session_leakage_risk": 0,
        "opendetect_compatibility": 2, "encrypted_traffic_relevance": 2,
        "cross_dataset_scientific_value": 1,
    },
    "CSTNET-TLS1.3": {
        "raw_packet_availability": 2, "label_quality": 2, "number_of_classes": 2,
        "per_class_sample_size": 1, "flow_pcap_alignment": 2,
        "strict_unknown_free_feasibility": 2, "capture_session_leakage_risk": 1,
        "opendetect_compatibility": 2, "encrypted_traffic_relevance": 2,
        "cross_dataset_scientific_value": 2,
    },
    "CICDDoS2019": {
        "raw_packet_availability": 0, "label_quality": 2, "number_of_classes": 2,
        "per_class_sample_size": 1, "flow_pcap_alignment": 1,
        "strict_unknown_free_feasibility": 2, "capture_session_leakage_risk": 0,
        "opendetect_compatibility": 0, "encrypted_traffic_relevance": 0,
        "cross_dataset_scientific_value": 1,
    },
}
TIERS = {"CSTNET-TLS1.3": "Tier A — PRIMARY EXTERNAL DATASET", "CipherSpectrum": "Tier B — SECONDARY VALIDATION", "CICDDoS2019": "Tier C — DIAGNOSTIC ONLY"}


def ranking_rows() -> list[dict[str, Any]]:
    rows = []
    for dataset, scores in SCORES.items():
        total = sum(scores.values())
        rows.append({"dataset": dataset, **scores, "total_score": total, "tier": TIERS[dataset]})
    return sorted(rows, key=lambda row: (-row["total_score"], row["dataset"]))


def inventory_markdown(dataset: str, rows: list[dict[str, Any]], totals: dict[str, int], pcap_rows: list[dict[str, Any]]) -> str:
    ext_counts: Counter = Counter()
    ext_bytes: Counter = Counter()
    for row in rows:
        ext_counts[row["extension"]] += 1
        ext_bytes[row["extension"]] += int(row["size_bytes"])
    pcaps = [row for row in pcap_rows if row["dataset"] == dataset and not row["macos_sidecar"]]
    valid = sum(bool(row["readable_magic"]) for row in pcaps)
    lines = [
        f"# {dataset} Inventory Summary", "",
        f"- Root: `{DATASETS[dataset]}`",
        f"- Total files: {totals['files']:,}",
        f"- Logical size: {totals['logical_bytes']:,} bytes ({totals['logical_bytes']/2**30:.3f} GiB)",
        f"- Allocated size from stat blocks: {totals['allocated_bytes']:,} bytes ({totals['allocated_bytes']/2**30:.3f} GiB)",
        f"- Non-sidecar PCAP/PCAPNG: {len(pcaps):,}; valid capture magic: {valid:,}", "",
        "| extension | files | logical bytes | GiB |", "|---|---:|---:|---:|",
    ]
    for ext in sorted(ext_counts, key=lambda key: (-ext_bytes[key], key)):
        lines.append(f"| {ext} | {ext_counts[ext]:,} | {ext_bytes[ext]:,} | {ext_bytes[ext]/2**30:.3f} |")
    return "\n".join(lines) + "\n"


def render_audit_summary(totals, valid, primary, balance, rankings, sample_rows, archive_rows, unparsed) -> str:
    balances = {dataset: [row for row in balance if row["dataset"] == dataset] for dataset in DATASETS}
    pcap_sizes = {dataset: sum(path.stat().st_size for path in valid.get(dataset, [])) for dataset in DATASETS}
    sampled_tls = defaultdict(list)
    for row in sample_rows:
        sampled_tls[row["dataset"]].append(row)
    lines = [
        "# Three-Dataset Open-Set Suitability Audit", "",
        "## Scope", "",
        "Read-only inventory/schema/label/protocol audit. No training, extraction, PCA/GMM, Unknown Detection, label rewriting, split creation, archive extraction, or bulk PCAP parsing was performed.", "",
        "## Core inventory", "",
        "| dataset | files | logical GiB | allocated GiB | valid PCAP | PCAP GiB | primary candidate classes | balance min / median / max |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for dataset in DATASETS:
        b = balances[dataset]
        min_med_max = "NA" if not b else f"{b[0]['dataset_min']} / {b[0]['dataset_median']} / {b[0]['dataset_max']}"
        lines.append(
            f"| {dataset} | {totals[dataset]['files']:,} | {totals[dataset]['logical_bytes']/2**30:.3f} | "
            f"{totals[dataset]['allocated_bytes']/2**30:.3f} | {len(valid.get(dataset, [])):,} | "
            f"{pcap_sizes[dataset]/2**30:.3f} | {sum(row['dataset']==dataset for row in primary)} | {min_med_max} |"
        )
    lines.extend(["", "## Dataset decisions", ""])
    lines.extend(
        [
            "### CSTNET-TLS1.3", "",
            "- Best first external dataset after USTC-TFC2016.",
            "- 120 domain/service classes, valid per-flow PCAPs, packet order and bytes available.",
            "- Local readme identifies both flow-level and packet-level 120-class arrays. The numeric ID-to-domain map was not found.",
            "- The very small tail class and possible endpoint/domain shortcut require predeclared eligibility and grouped/session-aware evaluation.",
            "- Claim that every capture is TLS 1.3 is supported by dataset documentation and only sampled packet inspection here; it is not a full-PCAP proof.", "",
            "### CipherSpectrum", "",
            "- Raw per-flow PCAPs are directly usable; on-disk capture groups and filenames expose cipher and website/domain levels.",
            "- Domain is the scientifically useful held-out level; the four capture groups alone are too few.",
            "- LABEL_PROVENANCE_UNCLEAR: no standalone authoritative label map/readme was found; website labels are strongly entangled with endpoint/domain identity, so this is secondary rather than primary validation.",
            f"- Unparsed valid-PCAP filenames: {unparsed}.", "",
            "### CICDDoS2019", "",
            "- Current disk contains authoritative official CSV labels and completed derived aggregate flow Parquets, but no PCAP/PCAPNG/archive containing PCAP.",
            "- Therefore raw-byte Open-Detect cannot be constructed from the current directory. Feature-level held-out detection remains possible.",
            "- Official labels are row-level; filenames are not label rules. Nineteen labels include BENIGN and fine-grained DDoS types.",
            "- Training/testing-day and attack/capture binding is a high scientific shortcut risk.", "",
            "## Ranking", "",
            "| rank | dataset | score / 20 | tier |", "|---:|---|---:|---|",
        ]
    )
    for rank, row in enumerate(rankings, start=1):
        lines.append(f"| {rank} | {row['dataset']} | {row['total_score']} | {row['tier']} |")
    lines.extend(
        [
            "", "## Recommended Experimental Role", "",
            "1. **CSTNET-TLS1.3** — first full encrypted-traffic external validation candidate.",
            "2. **CipherSpectrum** — secondary raw-byte/domain-held-out validation with explicit endpoint-shortcut controls.",
            "3. **CICDDoS2019** — diagnostic/feature-level validation unless original PCAPs are restored; even then use day/capture-aware protocols.", "",
            "Technical runnability is not treated as scientific suitability. Exact evidence tables are stored in the CSV outputs.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    if AUDIT_ROOT not in output.parents and output != DEFAULT_OUTPUT.resolve():
        raise ValueError("output must stay inside dataset_suitability_audit")
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()

    print("[1/7] Complete recursive inventory")
    inventory, files_by_dataset, totals = inventory_all()
    print("[2/7] Low-cost PCAP magic and deterministic packet-tool samples")
    pcap_rows, valid_pcaps = audit_pcaps(files_by_dataset)
    sample_rows = pcap_sample_audit(valid_pcaps)
    print("[3/7] Table headers/schemas/samples and archive listings")
    table_rows, cic_cache = audit_tables(files_by_dataset)
    archive_rows = audit_archives(files_by_dataset)
    print("[4/7] Label, balance and leakage inventories")
    class_rows, leakage_rows, unparsed = make_class_outputs(valid_pcaps, cic_cache)
    primary = primary_class_rows(class_rows)
    balance = class_balance_rows(class_rows)
    print("[5/7] Open-set/Open-Detect suitability and fixed rubric")
    open_set, opendetect, mapping, unknown_free, openness = fixed_suitability_rows(primary, valid_pcaps)
    rankings = ranking_rows()

    inventory_fields = ["dataset", "relative_path", "extension", "size_bytes", "size_gib", "file_type", "suspected_role"]
    atomic_csv(output / "dataset_inventory.csv", inventory, inventory_fields)
    atomic_csv(output / "pcap_inventory.csv", pcap_rows, ["dataset", "relative_path", "size_bytes", "macos_sidecar", "readable_magic", "capture_format", "detail"])
    atomic_csv(output / "pcap_sample_audit.csv", sample_rows, ["dataset", "relative_path", "capinfos_readable", "packet_count", "first_packet_timestamp_epoch", "last_packet_timestamp_epoch", "tshark_returncode", "tls_filtered_lines", "tls_version_fields", "sample_only"])
    table_fields = ["dataset", "relative_path", "extension", "size_bytes", "rows", "shape", "dtype", "columns", "label_column", "sample_rows", "read_status", "row_count_source"] + list(schema_flags([], Path("x")).keys())
    atomic_csv(output / "table_schema_audit.csv", table_rows, table_fields)
    atomic_csv(output / "archive_inventory.csv", archive_rows, ["dataset", "relative_path", "extension", "size_bytes", "valid_archive", "member_count", "uncompressed_bytes", "member_extensions", "contains_pcap", "contains_tables", "read_status"])
    atomic_csv(output / "class_inventory.csv", class_rows, ["dataset", "label_level", "class_name", "sample_count", "source_file_count", "source_capture_count", "label_semantics", "label_provenance"])
    atomic_csv(output / "class_balance.csv", balance, ["dataset", "label_level", "class_name", "sample_count", "lt_100", "lt_500", "lt_1000", "lt_5000", "dataset_num_classes", "dataset_min", "dataset_median", "dataset_max"])
    atomic_csv(output / "open_set_class_suitability.csv", open_set, ["dataset", "num_candidate_classes", "class_semantics", "can_hold_out_classes", "reason"])
    atomic_csv(output / "opendetect_input_suitability.csv", opendetect, ["dataset", "status", "pcap_available", "precise_flow_label", "packet_order", "packet_bytes", "flow_packet_sequence", "reason"])
    atomic_csv(output / "flow_pcap_mapping_audit.csv", mapping, ["dataset", "mapping_keys_available", "timestamp_precision", "flow_id_available", "bidirectional_or_unidirectional", "expected_mapping_difficulty"])
    leakage_fields = ["dataset", "class_name", "num_source_files", "num_pcaps", "num_days", "capture_class_entanglement", "time_class_entanglement", "endpoint_class_entanglement", "dominant_source_ratio", "risk_level", "derived_matched_flow_count", "capture_day_count_from_derived"]
    atomic_csv(output / "leakage_risk_audit.csv", leakage_rows, leakage_fields)
    atomic_csv(output / "unknown_free_protocol_suitability.csv", unknown_free, ["dataset", "candidate_classes", "quantity_grade", "strict_unknown_free_feasible", "scientific_caveat"])
    atomic_csv(output / "openness_suitability.csv", openness, ["dataset", "candidate_classes", "low_unknown_ratio", "medium_unknown_ratio", "high_unknown_ratio", "actual_split_created"])
    score_fields = ["dataset"] + list(next(iter(SCORES.values())).keys()) + ["total_score", "tier"]
    atomic_csv(output / "dataset_ranking.csv", rankings, score_fields)

    print("[6/7] Markdown summaries")
    for dataset in DATASETS:
        subset = [row for row in inventory if row["dataset"] == dataset]
        summary = inventory_markdown(dataset, subset, totals[dataset], pcap_rows)
        atomic_text(output / f"{dataset.replace('.', '_')}_inventory_summary.md", summary)
    combined_inventory = "# Dataset Inventory Summaries\n\n" + "\n".join(
        inventory_markdown(dataset, [row for row in inventory if row["dataset"] == dataset], totals[dataset], pcap_rows)
        for dataset in DATASETS
    )
    atomic_text(output / "inventory_summary.md", combined_inventory)
    summary_text = render_audit_summary(totals, valid_pcaps, primary, balance, rankings, sample_rows, archive_rows, unparsed)
    atomic_text(output / "audit_summary.md", summary_text)
    metadata = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "actual_command": "python scripts/run_audit.py",
        "dataset_roots": {key: str(value) for key, value in DATASETS.items()},
        "read_only_dataset_audit": True,
        "bulk_pcap_parsing": False,
        "pcap_validation": "all capture magic bytes plus deterministic capinfos/tshark sample",
        "models_trained": False,
        "unknown_detection_run": False,
        "splits_created": False,
        "archives_extracted": False,
        "totals": totals,
        "valid_pcap_counts": {key: len(valid_pcaps.get(key, [])) for key in DATASETS},
        "sampled_pcap_counts": dict(Counter(row["dataset"] for row in sample_rows)),
        "primary_candidate_class_counts": dict(Counter(row["dataset"] for row in primary)),
        "cipher_unparsed_filenames": unparsed,
        "ranking": rankings,
        "elapsed_seconds": time.time() - started,
        "output_files": sorted(path.name for path in output.iterdir() if path.is_file()),
        "tool_versions": {"python": sys.version, "capinfos": shutil.which("capinfos"), "tshark": shutil.which("tshark")},
    }
    atomic_text(output / "run_metadata.json", json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print("[7/7] PASS", {key: totals[key]["files"] for key in DATASETS})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
