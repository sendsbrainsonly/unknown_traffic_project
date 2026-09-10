# -*- coding: utf-8 -*-
"""FEC-OSL Flow Interaction Graph (FIG) input generation from Stage 0 PKLs.

This module is intentionally additive, mirroring Task 0.3: it never modifies the
canonical Stage 0 flow splitter and never drops a flow.  Every flow_id retained
by Task 0.3 ``compatible_min1`` (i.e. every Stage 0 flow) is represented here
with the same ``flow_id``, so Task 0.5 cross-validation can pair each graph with
its TrafficFormer text sample.

FIG construction follows:

- Yang et al., "End-to-End Open-Set Semi-Supervised Learning for Fine-Grained
  Encrypted Traffic Classification" (FEC-OSL), IEEE TIFS 2026.  Each biflow is a
  graph of at most 30 packet nodes; node features are direction, length,
  timestamp, burst packet count, burst byte count, and the packet/byte count
  ratios to the preceding burst.
- Shen et al., "Accurate Decentralized Application Identification via Encrypted
  Traffic Analysis Using Graph Neural Networks", IEEE TIFS, vol. 16, 2021
  (the seminal reference the FEC-OSL procedure follows).  Its Algorithm 1
  segments bursts by packet direction only -- no time threshold -- and connects
  consecutive nodes inside a burst plus the first-to-first / last-to-last nodes
  of adjacent bursts.  FEC-OSL's prose "predeﬁned burst threshold" inherits the
  classic "short time interval" wording, but the referenced procedure is the
  direction-only Algorithm 1; we follow the procedure.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import pickle
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

FECOSL_REFERENCE = {
    "title": "End-to-End Open-Set Semi-Supervised Learning for Fine-Grained "
    "Encrypted Traffic Classification",
    "venue": "IEEE Transactions on Information Forensics and Security, 2026",
    "role": "FIG node features, edge rules, Nv = 30",
}
SEMINAL_BURST_REFERENCE = {
    "authors": "M. Shen, J. Zhang, L. Zhu, K. Xu, and X. Du",
    "title": "Accurate Decentralized Application Identification via Encrypted "
    "Traffic Analysis Using Graph Neural Networks",
    "venue": "IEEE Transactions on Information Forensics and Security, vol. 16, "
    "pp. 2367-2380, 2021",
    "doi": "10.1109/TIFS.2021.3050608",
    "role": "burst segmentation (direction only, Algorithm 1), edge construction",
}

# Feature order per node, following FEC-OSL's seven-feature list.
FEATURE_NAMES = (
    "direction",          # +1 initiator / -1 responder
    "length",             # packet length (caplen by default)
    "timestamp_delta",    # seconds since the first packet of the flow
    "burst_packet_count",
    "burst_byte_count",
    "prev_burst_packet_ratio",
    "prev_burst_byte_ratio",
)


@dataclass(frozen=True)
class FigFormat:
    """Construction controls for one FIG sample."""

    policy: str = "all_flows"
    min_packets: int = 1
    max_packets: int = 30
    burst_method: str = "direction"
    length_field: str = "caplen"
    timestamp_mode: str = "relative_delta"
    initiator_direction_value: int = 1
    first_burst_ratio_fill: float = 0.0

    def validate(self) -> None:
        if self.min_packets < 1:
            raise ValueError("min_packets must be >= 1; empty flows are invalid")
        if self.max_packets < self.min_packets:
            raise ValueError("max_packets must be >= min_packets")
        if self.burst_method != "direction":
            raise ValueError(
                "burst_method must be 'direction' (Shen 2021 Algorithm 1); "
                "no time-threshold variant is implemented"
            )
        if self.length_field not in ("caplen", "wirelen"):
            raise ValueError("length_field must be 'caplen' or 'wirelen'")
        if self.timestamp_mode != "relative_delta":
            raise ValueError("timestamp_mode must be 'relative_delta'")
        if self.initiator_direction_value not in (0, 1):
            raise ValueError("initiator_direction_value must be 0 or 1")


@dataclass(frozen=True)
class FlowGraph:
    """One FIG: node features plus undirected edges over node indices."""

    flow_id: str
    packet_count: int
    used_packet_count: int
    node_count: int
    burst_count: int
    features: Tuple[Tuple[float, ...], ...]  # node_count x 7
    edges: Tuple[Tuple[int, int], ...]  # sorted (min, max) pairs, deduplicated


def _direction_sign(direction: int, fmt: FigFormat) -> float:
    if direction == fmt.initiator_direction_value:
        return 1.0
    return -1.0


def _bursts(directions: Sequence[float]) -> List[Tuple[int, int]]:
    """Return (start, end) inclusive index ranges of direction runs.

    Shen 2021 Algorithm 1: a burst is a maximal run of consecutive packets
    transmitted along the same direction; no time threshold is applied.
    """
    if not directions:
        return []
    bursts = []
    start = 0
    for index in range(1, len(directions)):
        if directions[index] != directions[start]:
            bursts.append((start, index - 1))
            start = index
    bursts.append((start, len(directions) - 1))
    return bursts


def _build_edges(bursts: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Shen 2021 Algorithm 1 edge set.

    Intra-burst: consecutive nodes in a burst are chained.  Inter-burst: the
    first node of each burst is connected to the first node of the preceding
    burst, and the last node to the last node of the preceding burst.  When
    both bursts are singletons, first == last and the duplicated pair
    collapses, so at most one edge is added per pair.
    """
    edge_set = set()
    for start, end in bursts:
        for index in range(start, end):
            edge_set.add((index, index + 1))
    for previous, current in zip(bursts, bursts[1:]):
        edge_set.add((previous[0], current[0]))
        edge_set.add((previous[1], current[1]))
    return sorted(edge_set)


def build_flow_graph(
    flow_id: str,
    packets: Sequence[Sequence[Any]],
    fmt: FigFormat,
) -> FlowGraph:
    """Build one FIG from stored bidirectional flow packets.

    Stage 0 packets have the schema
    ``(timestamp, caplen, wirelen, direction, raw_frame)``, with direction 1
    for the initiator and 0 for the responder.  Packets are sorted by
    timestamp (stable); Stage 0 stores them in timestamp order, so this is a
    no-op on canonical data.  At most ``max_packets`` nodes are used; shorter
    flows keep their natural node count and are never padded.
    """

    fmt.validate()
    packet_count = len(packets)
    if packet_count < fmt.min_packets:
        raise ValueError(f"{flow_id}: {packet_count} packets < min_packets")

    length_index = 1 if fmt.length_field == "caplen" else 2
    selected = sorted(packets[:], key=lambda packet: packet[0])[: fmt.max_packets]
    for packet_index, packet in enumerate(selected):
        if len(packet) < 5:
            raise ValueError(
                f"{flow_id}: packet {packet_index} does not match "
                "(timestamp, caplen, wirelen, direction, raw_frame)"
            )
        if packet[3] not in (0, 1):
            raise ValueError(f"{flow_id}: packet {packet_index} direction must be 0 or 1")
        if packet[0] < 0 or packet[length_index] < 0:
            raise ValueError(f"{flow_id}: packet {packet_index} has negative ts/length")

    directions = [_direction_sign(packet[3], fmt) for packet in selected]
    lengths = [int(packet[length_index]) for packet in selected]
    timestamps = [float(packet[0]) for packet in selected]
    first_ts = timestamps[0]

    bursts = _bursts(directions)
    burst_of_node = [0] * len(selected)
    for burst_index, (start, end) in enumerate(bursts):
        for node_index in range(start, end + 1):
            burst_of_node[node_index] = burst_index

    burst_stats = [
        (end - start + 1, sum(lengths[start : end + 1])) for start, end in bursts
    ]

    features = []
    for node_index in range(len(selected)):
        burst_index = burst_of_node[node_index]
        burst_packets, burst_bytes = burst_stats[burst_index]
        if burst_index == 0:
            packet_ratio = fmt.first_burst_ratio_fill
            byte_ratio = fmt.first_burst_ratio_fill
        else:
            prev_packets, prev_bytes = burst_stats[burst_index - 1]
            packet_ratio = burst_packets / prev_packets
            byte_ratio = burst_bytes / prev_bytes if prev_bytes else fmt.first_burst_ratio_fill
        features.append(
            (
                directions[node_index],
                float(lengths[node_index]),
                timestamps[node_index] - first_ts,
                float(burst_packets),
                float(burst_bytes),
                packet_ratio,
                byte_ratio,
            )
        )

    return FlowGraph(
        flow_id=flow_id,
        packet_count=packet_count,
        used_packet_count=len(selected),
        node_count=len(selected),
        burst_count=len(bursts),
        features=tuple(features),
        edges=tuple(_build_edges(bursts)),
    )


def _compact_number(value: float) -> Any:
    """Round floats for storage; collapse whole floats to integers."""
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        return round(value, 6)
    return value


def graph_to_json_record(flow_id: str, graph: FlowGraph) -> str:
    """Serialize one FIG as a compact single-line JSON record."""
    record = {
        "flow_id": flow_id,
        "node_count": graph.node_count,
        "burst_count": graph.burst_count,
        "features": [
            [_compact_number(value) for value in node] for node in graph.features
        ],
        "edges": [list(edge) for edge in graph.edges],
    }
    return json.dumps(record, ensure_ascii=False, separators=(",", ":"))


def packet_count_bucket(packet_count: int) -> str:
    if packet_count <= 0:
        return "empty"
    if packet_count == 1:
        return "1"
    if packet_count == 2:
        return "2"
    if packet_count <= 4:
        return "3-4"
    return ">=5"


def _node_bucket(node_count: int) -> str:
    if node_count <= 1:
        return "1"
    if node_count == 2:
        return "2"
    if node_count <= 4:
        return "3-4"
    if node_count <= 10:
        return "5-10"
    if node_count <= 20:
        return "11-20"
    return "21-30"


def _parse_generated_pkl_name(path: Path) -> Tuple[str, str, str]:
    parts = path.stem.split("__", 2)
    if len(parts) != 3:
        raise ValueError(
            f"unexpected Stage 0 PKL name {path.name!r}; expected split__class__stem.pkl"
        )
    return parts[0], parts[1], parts[2]


def _discover_pkl_files(flows_dir: Path) -> Sequence[Path]:
    paths = sorted(flows_dir.glob("*.pkl"))
    if not paths:
        raise FileNotFoundError(f"no Stage 0 flow PKLs found under {flows_dir}")
    return paths


def _new_class_stats() -> Dict[str, Any]:
    return {
        "total_flows": 0,
        "total_nodes": 0,
        "total_edges": 0,
        "total_bursts": 0,
        "max_nodes": 0,
        "max_edges": 0,
        "max_bursts": 0,
    }


def _validate_payload(
    path: Path,
    payload: Mapping[str, Any],
    expected_split: str,
    expected_class: str,
) -> Mapping[str, Sequence[Sequence[Any]]]:
    if payload.get("split") != expected_split:
        raise ValueError(
            f"{path.name}: split metadata {payload.get('split')!r} != {expected_split!r}"
        )
    if payload.get("class_name") != expected_class:
        raise ValueError(
            f"{path.name}: class metadata {payload.get('class_name')!r} != {expected_class!r}"
        )
    packets = payload.get("packets")
    if not isinstance(packets, Mapping):
        raise ValueError(f"{path.name}: missing mapping field 'packets'")
    return packets


def generate_dataset(
    flows_dir: Path,
    output_dir: Path,
    fmt: FigFormat,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Generate FIG JSONL plus an auditable flow index.

    The output files are staged under ``output_dir`` and installed with atomic
    file replacement.  Existing outputs are preserved unless ``overwrite`` is
    explicitly true.  Every Stage 0 flow is retained; the label ids are built
    from the sorted class names exactly like Task 0.3, so both mappings agree.
    """

    fmt.validate()
    flows_dir = Path(flows_dir)
    output_dir = Path(output_dir)
    pkl_paths = _discover_pkl_files(flows_dir)

    parsed_names = {path: _parse_generated_pkl_name(path) for path in pkl_paths}
    class_names = sorted({class_name for _, class_name, _ in parsed_names.values()})
    class_to_id = {class_name: index for index, class_name in enumerate(class_names)}

    output_dir.mkdir(parents=True, exist_ok=True)
    filenames = {
        "jsonl": "fig_all.jsonl",
        "fig_index": "fig_index.csv",
        "label_map": "label_map.json",
        "summary": "generation_summary.json",
        "stats": "stats_by_class.csv",
    }
    existing = [output_dir / name for name in filenames.values() if (output_dir / name).exists()]
    if existing and not overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"refusing to overwrite existing outputs: {joined}")

    class_stats = {class_name: _new_class_stats() for class_name in class_names}
    seen_flow_ids = set()
    total_flows = 0
    total_nodes = 0
    total_edges = 0
    total_bursts = 0
    node_bucket_counts: Dict[str, int] = {}

    with tempfile.TemporaryDirectory(prefix=".fig-stage-", dir=output_dir) as stage:
        stage_dir = Path(stage)
        jsonl_path = stage_dir / filenames["jsonl"]
        index_path = stage_dir / filenames["fig_index"]

        index_fields = [
            "sample_index",
            "jsonl_line_number",
            "flow_id",
            "byte_offset",
            "label_id",
            "split",
            "class_name",
            "source_file",
            "source_pkl",
            "packet_count",
            "used_packet_count",
            "packet_count_bucket",
            "node_count",
            "burst_count",
            "edge_count",
            "policy",
            "graph_sha256",
        ]

        with jsonl_path.open("w", encoding="utf-8") as jsonl_fh, index_path.open(
            "w", newline="", encoding="utf-8"
        ) as index_fh:
            index_writer = csv.DictWriter(index_fh, fieldnames=index_fields)
            index_writer.writeheader()

            for pkl_path in pkl_paths:
                expected_split, expected_class, _ = parsed_names[pkl_path]
                with pkl_path.open("rb") as fh:
                    payload = pickle.load(fh)
                if not isinstance(payload, Mapping):
                    raise ValueError(f"{pkl_path.name}: expected a mapping payload")
                packets_by_flow = _validate_payload(
                    pkl_path, payload, expected_split, expected_class
                )
                source_file = str(payload.get("source_file", ""))

                for flow_id in sorted(packets_by_flow):
                    if flow_id in seen_flow_ids:
                        raise ValueError(f"duplicate flow_id across PKLs: {flow_id}")
                    seen_flow_ids.add(flow_id)
                    packets = packets_by_flow[flow_id]
                    if not isinstance(packets, Sequence):
                        raise ValueError(f"{pkl_path.name}/{flow_id}: packets must be a sequence")

                    graph = build_flow_graph(flow_id, packets, fmt)
                    record = graph_to_json_record(flow_id, graph)
                    offset = jsonl_fh.tell()
                    jsonl_fh.write(record + "\n")

                    stats = class_stats[expected_class]
                    stats["total_flows"] += 1
                    stats["total_nodes"] += graph.node_count
                    stats["total_edges"] += len(graph.edges)
                    stats["total_bursts"] += graph.burst_count
                    stats["max_nodes"] = max(stats["max_nodes"], graph.node_count)
                    stats["max_edges"] = max(stats["max_edges"], len(graph.edges))
                    stats["max_bursts"] = max(stats["max_bursts"], graph.burst_count)

                    total_flows += 1
                    total_nodes += graph.node_count
                    total_edges += len(graph.edges)
                    total_bursts += graph.burst_count
                    bucket = _node_bucket(graph.node_count)
                    node_bucket_counts[bucket] = node_bucket_counts.get(bucket, 0) + 1

                    index_writer.writerow(
                        {
                            "sample_index": total_flows - 1,
                            "jsonl_line_number": total_flows,
                            "flow_id": flow_id,
                            "byte_offset": offset,
                            "label_id": class_to_id[expected_class],
                            "split": expected_split,
                            "class_name": expected_class,
                            "source_file": source_file,
                            "source_pkl": pkl_path.name,
                            "packet_count": graph.packet_count,
                            "used_packet_count": graph.used_packet_count,
                            "packet_count_bucket": packet_count_bucket(graph.packet_count),
                            "node_count": graph.node_count,
                            "burst_count": graph.burst_count,
                            "edge_count": len(graph.edges),
                            "policy": fmt.policy,
                            "graph_sha256": hashlib.sha256(record.encode("utf-8")).hexdigest(),
                        }
                    )

        label_payload = {
            "class_to_id": class_to_id,
            "id_to_class": {str(value): key for key, value in class_to_id.items()},
        }
        (stage_dir / filenames["label_map"]).write_text(
            json.dumps(label_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        summary = {
            "policy": fmt.policy,
            "format": asdict(fmt),
            "feature_order": list(FEATURE_NAMES),
            "references": {
                "fecosl": FECOSL_REFERENCE,
                "seminal_burst": SEMINAL_BURST_REFERENCE,
            },
            "source_pkl_count": len(pkl_paths),
            "class_count": len(class_names),
            "total_flows": total_flows,
            "retained_flows": total_flows,
            "dropped_flows": 0,
            "retention_ratio": 1.0 if total_flows else 0.0,
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "total_bursts": total_bursts,
            "mean_nodes_per_flow": total_nodes / total_flows if total_flows else 0.0,
            "mean_edges_per_flow": total_edges / total_flows if total_flows else 0.0,
            "node_bucket_counts": dict(sorted(node_bucket_counts.items())),
            "padding": "none; node count is min(max_packets, packet_count)",
            "aligns_with": "Task 0.3 compatible_min1 flow_id set",
        }
        (stage_dir / filenames["summary"]).write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        stats_fields = [
            "class_name",
            "label_id",
            "total_flows",
            "total_nodes",
            "total_edges",
            "total_bursts",
            "mean_nodes",
            "mean_edges",
            "mean_bursts",
            "max_nodes",
            "max_edges",
            "max_bursts",
        ]
        with (stage_dir / filenames["stats"]).open(
            "w", newline="", encoding="utf-8"
        ) as fh:
            writer = csv.DictWriter(fh, fieldnames=stats_fields)
            writer.writeheader()
            for class_name in class_names:
                stats = class_stats[class_name]
                total = stats["total_flows"]
                writer.writerow(
                    {
                        "class_name": class_name,
                        "label_id": class_to_id[class_name],
                        **stats,
                        "mean_nodes": round(stats["total_nodes"] / total, 4) if total else 0.0,
                        "mean_edges": round(stats["total_edges"] / total, 4) if total else 0.0,
                        "mean_bursts": round(stats["total_bursts"] / total, 4) if total else 0.0,
                    }
                )

        for name in filenames.values():
            os.replace(stage_dir / name, output_dir / name)

    return summary


def format_from_config(config: Mapping[str, Any], policy: Optional[str] = None) -> FigFormat:
    """Build and validate a format object from the additive YAML config."""

    format_cfg = dict(config.get("format", {}))
    policies = config.get("policies", {})
    policy = policy or config.get("default_policy")
    if policy not in policies:
        raise KeyError(f"unknown policy {policy!r}; available: {sorted(policies)}")
    policy_cfg = dict(policies[policy])
    fmt = FigFormat(
        policy=policy,
        min_packets=int(policy_cfg["min_packets"]),
        max_packets=int(format_cfg.get("max_packets", 30)),
        burst_method=str(format_cfg.get("burst_method", "direction")),
        length_field=str(format_cfg.get("length_field", "caplen")),
        timestamp_mode=str(format_cfg.get("timestamp_mode", "relative_delta")),
        initiator_direction_value=int(format_cfg.get("initiator_direction_value", 1)),
        first_burst_ratio_fill=float(format_cfg.get("first_burst_ratio_fill", 0.0)),
    )
    fmt.validate()
    return fmt
