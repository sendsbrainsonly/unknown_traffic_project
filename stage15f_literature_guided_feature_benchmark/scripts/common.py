#!/usr/bin/env python3
"""Shared helpers for the Stage 15F-0 read-only feature audit."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Sequence


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
WINDOWS = (8, 16, 32, 64)


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(block_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_csv(path: Path) -> Iterator[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle)


def write_csv(path: Path, rows: Iterable[dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


@dataclass
class WindowAccumulator:
    """Streaming packet-window statistics for one frozen flow.

    The accumulator stores no packet payload.  It retains only aggregate totals
    and the cumulative captured-frame bytes/time at the preregistered windows.
    """

    flow_uid: str
    class_name: str
    source_file: str
    domain_state: str
    packet_count: int = 0
    total_frame_bytes: int = 0
    total_ip_bytes: int = 0
    total_transport_payload_bytes: int = 0
    first_timestamp: float | None = None
    last_timestamp: float | None = None
    first_endpoint: tuple[str, str] | None = None
    forward_packets: int = 0
    reverse_packets: int = 0
    protocol_tokens: set[str] = field(default_factory=set)
    window_frame_bytes: dict[int, int] = field(default_factory=dict)
    window_timestamps: dict[int, float] = field(default_factory=dict)

    def add(
        self,
        timestamp: float,
        frame_bytes: int,
        ip_bytes: int | None,
        transport_payload_bytes: int | None,
        source_endpoint: tuple[str, str] | None,
        destination_endpoint: tuple[str, str] | None,
        protocol_tokens: str = "",
    ) -> None:
        if self.first_timestamp is None:
            self.first_timestamp = timestamp
        self.last_timestamp = timestamp
        if self.first_endpoint is None and source_endpoint is not None:
            self.first_endpoint = source_endpoint
        if source_endpoint is not None and destination_endpoint is not None and self.first_endpoint is not None:
            if source_endpoint == self.first_endpoint:
                self.forward_packets += 1
            elif destination_endpoint == self.first_endpoint:
                self.reverse_packets += 1
        self.packet_count += 1
        self.total_frame_bytes += max(int(frame_bytes), 0)
        if ip_bytes is not None:
            self.total_ip_bytes += max(int(ip_bytes), 0)
        if transport_payload_bytes is not None:
            self.total_transport_payload_bytes += max(int(transport_payload_bytes), 0)
        self.protocol_tokens.update(token.lower() for token in protocol_tokens.split(":") if token)
        if self.packet_count in WINDOWS:
            self.window_frame_bytes[self.packet_count] = self.total_frame_bytes
            self.window_timestamps[self.packet_count] = timestamp

    def row(self, dataset: str) -> dict[str, object]:
        if self.packet_count <= 0 or self.first_timestamp is None or self.last_timestamp is None:
            raise RuntimeError(f"empty target flow: {self.flow_uid}")
        duration = max(0.0, self.last_timestamp - self.first_timestamp)
        row: dict[str, object] = {
            "dataset": dataset,
            "flow_uid": self.flow_uid,
            "class_name": self.class_name,
            "source_file": self.source_file,
            "domain_state": self.domain_state,
            "packet_count": self.packet_count,
            "total_frame_bytes": self.total_frame_bytes,
            "total_ip_bytes": self.total_ip_bytes,
            "total_transport_payload_bytes": self.total_transport_payload_bytes,
            "duration_seconds": duration,
            "forward_packets": self.forward_packets,
            "reverse_packets": self.reverse_packets,
            "protocol_tokens": ":".join(sorted(self.protocol_tokens)),
        }
        for window in WINDOWS:
            used = min(window, self.packet_count)
            if self.packet_count <= window:
                frame_bytes = self.total_frame_bytes
                window_seconds = duration
            else:
                frame_bytes = self.window_frame_bytes[window]
                window_seconds = max(0.0, self.window_timestamps[window] - self.first_timestamp)
            row[f"packets_used_{window}"] = used
            row[f"frame_bytes_first_{window}"] = frame_bytes
            row[f"time_seconds_first_{window}"] = window_seconds
            row[f"byte_coverage_first_{window}"] = (
                frame_bytes / self.total_frame_bytes if self.total_frame_bytes > 0 else ""
            )
            row[f"time_coverage_first_{window}"] = (
                window_seconds / duration if duration > 0 else (1.0 if self.packet_count <= window else "")
            )
            row[f"complete_first_{window}"] = self.packet_count <= window
            row[f"truncated_first_{window}"] = self.packet_count > window
            row[f"padding_ratio_first_{window}"] = max(0, window - self.packet_count) / window
        return row


FLOW_WINDOW_FIELDS = [
    "dataset",
    "flow_uid",
    "class_name",
    "source_file",
    "domain_state",
    "packet_count",
    "total_frame_bytes",
    "total_ip_bytes",
    "total_transport_payload_bytes",
    "duration_seconds",
    "forward_packets",
    "reverse_packets",
    "protocol_tokens",
]
for _window in WINDOWS:
    FLOW_WINDOW_FIELDS.extend(
        [
            f"packets_used_{_window}",
            f"frame_bytes_first_{_window}",
            f"time_seconds_first_{_window}",
            f"byte_coverage_first_{_window}",
            f"time_coverage_first_{_window}",
            f"complete_first_{_window}",
            f"truncated_first_{_window}",
            f"padding_ratio_first_{_window}",
        ]
    )
