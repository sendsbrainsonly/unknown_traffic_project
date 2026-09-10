# -*- coding: utf-8 -*-
import csv
import json
import pickle
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.preprocessing.fig_graph import (
    FEATURE_NAMES,
    FigFormat,
    build_flow_graph,
    format_from_config,
    generate_dataset,
    graph_to_json_record,
)


def packet(timestamp=1.0, caplen=100, direction=1):
    frame = b"\x00" * caplen
    return (timestamp, caplen, caplen, direction, frame)


class FigConstructionTests(unittest.TestCase):
    def test_bursts_are_direction_runs(self):
        packets = [
            packet(timestamp=0.0, direction=1),
            packet(timestamp=0.1, direction=1),
            packet(timestamp=0.2, direction=0),
            packet(timestamp=0.3, direction=1),
            packet(timestamp=0.4, direction=0),
            packet(timestamp=0.5, direction=0),
        ]
        graph = build_flow_graph("f", packets, FigFormat())
        self.assertEqual(graph.burst_count, 4)

    def test_single_packet_flow_has_one_node_and_no_edges(self):
        graph = build_flow_graph("f", [packet()], FigFormat())
        self.assertEqual(graph.node_count, 1)
        self.assertEqual(graph.burst_count, 1)
        self.assertEqual(graph.edges, ())
        # First burst has no predecessor: both ratio features use the fill.
        self.assertEqual(graph.features[0][5:7], (0.0, 0.0))

    def test_two_opposite_packets_form_two_singleton_bursts_one_edge(self):
        graph = build_flow_graph(
            "f",
            [packet(timestamp=0.0, direction=1), packet(timestamp=0.1, direction=0)],
            FigFormat(),
        )
        # Shen 2021: two singleton bursts collapse to a single inter-burst edge.
        self.assertEqual(graph.burst_count, 2)
        self.assertEqual(graph.edges, ((0, 1),))

    def test_two_same_direction_packets_form_one_intra_edge(self):
        graph = build_flow_graph(
            "f",
            [packet(timestamp=0.0, direction=1), packet(timestamp=0.1, direction=1)],
            FigFormat(),
        )
        self.assertEqual(graph.burst_count, 1)
        self.assertEqual(graph.edges, ((0, 1),))

    def test_golden_features_and_edges(self):
        packets = [
            (10.0, 100, 100, 1, b"x"),
            (10.5, 50, 50, 1, b"x"),
            (11.0, 200, 200, 0, b"x"),
        ]
        graph = build_flow_graph("golden", packets, FigFormat())
        # direction, length, ts delta, burst packets, burst bytes, ratios
        self.assertEqual(
            graph.features,
            (
                (1.0, 100.0, 0.0, 2.0, 150.0, 0.0, 0.0),
                (1.0, 50.0, 0.5, 2.0, 150.0, 0.0, 0.0),
                (-1.0, 200.0, 1.0, 1.0, 200.0, 0.5, 200.0 / 150.0),
            ),
        )
        # Intra chain (0,1) plus inter-burst first-first (0,2) and last-last (1,2).
        self.assertEqual(graph.edges, ((0, 1), (0, 2), (1, 2)))

    def test_node_count_is_capped_at_30(self):
        packets = [packet(timestamp=float(i), direction=i % 2) for i in range(35)]
        graph = build_flow_graph("f", packets, FigFormat())
        self.assertEqual(graph.packet_count, 35)
        self.assertEqual(graph.used_packet_count, 30)
        self.assertEqual(graph.node_count, 30)

    def test_packets_are_sorted_by_timestamp(self):
        packets = [
            (2.0, 10, 10, 1, b"x"),
            (1.0, 10, 10, 0, b"x"),
            (1.5, 10, 10, 1, b"x"),
        ]
        graph = build_flow_graph("f", packets, FigFormat())
        # After sorting: responder, initiator, initiator -> two bursts.
        self.assertEqual(graph.burst_count, 2)
        self.assertEqual(graph.features[0][0], -1.0)
        self.assertEqual(graph.features[0][2], 0.0)

    def test_direction_is_encoded_as_signed(self):
        graph = build_flow_graph(
            "f",
            [packet(direction=1), packet(direction=0)],
            FigFormat(),
        )
        self.assertEqual(graph.features[0][0], 1.0)
        self.assertEqual(graph.features[1][0], -1.0)

    def test_invalid_direction_is_rejected(self):
        with self.assertRaises(ValueError):
            build_flow_graph("f", [(0.0, 10, 10, 2, b"x")], FigFormat())

    def test_wirelen_can_be_selected_as_length(self):
        packets = [(0.0, 100, 140, 1, b"x"), (0.1, 100, 140, 1, b"x")]
        graph = build_flow_graph("f", packets, FigFormat(length_field="wirelen"))
        self.assertEqual(graph.features[0][1], 140.0)

    def test_zero_byte_predecessor_burst_uses_fill_for_ratio(self):
        # Second burst has zero bytes; the byte ratio must not divide by zero.
        packets = [
            (0.0, 0, 0, 1, b""),
            (0.1, 0, 0, 0, b""),
        ]
        graph = build_flow_graph("f", packets, FigFormat())
        self.assertEqual(graph.features[1][6], 0.0)
        self.assertEqual(graph.features[1][5], 1.0)


class FigSerializationTests(unittest.TestCase):
    def test_json_record_golden(self):
        packets = [
            (10.0, 100, 100, 1, b"x"),
            (10.5, 50, 50, 1, b"x"),
            (11.0, 200, 200, 0, b"x"),
        ]
        graph = build_flow_graph("golden", packets, FigFormat())
        record = graph_to_json_record("golden", graph)
        self.assertEqual(
            record,
            '{"flow_id":"golden","node_count":3,"burst_count":2,'
            '"features":[[1,100,0,2,150,0,0],[1,50,0.5,2,150,0,0],'
            '[-1,200,1,1,200,0.5,1.333333]],"edges":[[0,1],[0,2],[1,2]]}',
        )
        parsed = json.loads(record)
        self.assertEqual(len(parsed["features"][0]), len(FEATURE_NAMES))


class FigGenerationTests(unittest.TestCase):
    def test_end_to_end_outputs_and_alignment_fields(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            flows_dir = root / "flows"
            output_dir = root / "output"
            flows_dir.mkdir()
            payloads = {
                "Benign__Alpha__a.pkl": {
                    "split": "Benign",
                    "class_name": "Alpha",
                    "source_file": "a.pcap",
                    "packets": {
                        "Alpha__000001": [packet(timestamp=0.0)],
                        "Alpha__000002": [
                            packet(timestamp=0.0, direction=1),
                            packet(timestamp=0.1, direction=0),
                        ],
                    },
                },
                "Malware__Beta__b.pkl": {
                    "split": "Malware",
                    "class_name": "Beta",
                    "source_file": "b.pcap",
                    "packets": {"Beta__000001": [packet(timestamp=0.0, caplen=64)]},
                },
            }
            for name, payload in payloads.items():
                with (flows_dir / name).open("wb") as fh:
                    pickle.dump(payload, fh)

            summary = generate_dataset(flows_dir, output_dir, FigFormat())
            self.assertEqual(summary["total_flows"], 3)
            self.assertEqual(summary["retained_flows"], 3)
            self.assertEqual(summary["dropped_flows"], 0)
            self.assertEqual(summary["total_nodes"], 4)
            self.assertEqual(summary["total_edges"], 1)

            jsonl_path = output_dir / "fig_all.jsonl"
            index_path = output_dir / "fig_index.csv"
            self.assertTrue(jsonl_path.is_file())
            self.assertTrue(index_path.is_file())

            with index_path.open(newline="", encoding="utf-8") as fh:
                mappings = list(csv.DictReader(fh))
            # Classes iterate in pkl-name order; flow_ids are sorted per pkl.
            self.assertEqual(
                [row["flow_id"] for row in mappings],
                ["Alpha__000001", "Alpha__000002", "Beta__000001"],
            )
            # Sorted class names match Task 0.3's label_id construction.
            self.assertEqual(
                [row["label_id"] for row in mappings], ["0", "0", "1"]
            )
            self.assertEqual(mappings[0]["jsonl_line_number"], "1")
            self.assertEqual(mappings[0]["byte_offset"], "0")
            self.assertEqual(mappings[0]["node_count"], "1")
            self.assertEqual(mappings[1]["burst_count"], "2")
            self.assertEqual(mappings[1]["edge_count"], "1")

            # byte_offset supports random access into the JSONL.
            with jsonl_path.open(encoding="utf-8") as fh:
                fh.seek(int(mappings[1]["byte_offset"]))
                row = json.loads(fh.readline())
            self.assertEqual(row["flow_id"], "Alpha__000002")
            self.assertEqual(row["edges"], [[0, 1]])

            label_map = json.loads(
                (output_dir / "label_map.json").read_text(encoding="utf-8")
            )
            self.assertEqual(label_map["class_to_id"], {"Alpha": 0, "Beta": 1})

            with (output_dir / "stats_by_class.csv").open(
                newline="", encoding="utf-8"
            ) as fh:
                stats = {row["class_name"]: row for row in csv.DictReader(fh)}
            self.assertEqual(stats["Alpha"]["max_nodes"], "2")
            self.assertEqual(stats["Beta"]["total_flows"], "1")

    def test_existing_outputs_are_not_overwritten_by_default(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            flows_dir = root / "flows"
            output_dir = root / "output"
            flows_dir.mkdir()
            output_dir.mkdir()
            (output_dir / "fig_all.jsonl").write_text("preserve\n", encoding="utf-8")
            payload = {
                "split": "Benign",
                "class_name": "Demo",
                "source_file": "demo.pcap",
                "packets": {"Demo__000001": [packet()]},
            }
            with (flows_dir / "Benign__Demo__demo.pkl").open("wb") as fh:
                pickle.dump(payload, fh)

            with self.assertRaises(FileExistsError):
                generate_dataset(flows_dir, output_dir, FigFormat())
            self.assertEqual(
                (output_dir / "fig_all.jsonl").read_text(encoding="utf-8"),
                "preserve\n",
            )

    def test_format_from_config(self):
        fmt = format_from_config(
            {
                "default_policy": "all_flows",
                "policies": {"all_flows": {"min_packets": 1}},
                "format": {"max_packets": 30, "burst_method": "direction"},
            }
        )
        self.assertEqual(fmt.max_packets, 30)
        self.assertEqual(fmt.policy, "all_flows")
        with self.assertRaises(ValueError):
            format_from_config(
                {
                    "default_policy": "all_flows",
                    "policies": {"all_flows": {"min_packets": 1}},
                    "format": {"burst_method": "time_threshold"},
                }
            )

    def test_cli_generates_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            flows_dir = root / "flows"
            output_dir = root / "cli-output"
            flows_dir.mkdir()
            payload = {
                "split": "Malware",
                "class_name": "CliDemo",
                "source_file": "cli-demo.pcap",
                "packets": {"CliDemo__000001": [packet()]},
            }
            with (flows_dir / "Malware__CliDemo__cli-demo.pkl").open("wb") as fh:
                pickle.dump(payload, fh)

            project_root = Path(__file__).resolve().parents[1]
            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/task04_generate_fig_graph.py",
                    "--flows-dir",
                    str(flows_dir),
                    "--output-dir",
                    str(output_dir),
                ],
                cwd=project_root,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue((output_dir / "fig_all.jsonl").is_file())
            summary = json.loads(
                (output_dir / "generation_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["policy"], "all_flows")
            self.assertEqual(summary["retained_flows"], 1)


if __name__ == "__main__":
    unittest.main()
