# -*- coding: utf-8 -*-
import csv
import json
import pickle
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.preprocessing.trafficformer_input import (
    TrafficFormerFormat,
    encode_flow,
    generate_dataset,
    official_bigram_generation,
)


def packet(payload, timestamp=1.0, direction=1):
    frame = b"\x00" * 14 + bytes(payload)
    return (timestamp, len(frame), len(frame), direction, frame)


def reference_official_bigram(packet_hex):
    # Literal behavior of the upstream cut(..., 1) + adjacent merge for even hex.
    octets = [packet_hex[i : i + 2] for i in range(0, len(packet_hex), 2)]
    return "".join(
        octets[index] + octets[index + 1] + " "
        for index in range(max(0, len(octets) - 1))
    )


class TrafficFormerEncodingTests(unittest.TestCase):
    def test_official_bigram_golden_vector(self):
        self.assertEqual(official_bigram_generation("4500003c"), "4500 0000 003c ")

    def test_bigram_matches_reference_for_even_hex(self):
        for packet_hex in ("", "45", "4500", "450000", "0011223344556677"):
            self.assertEqual(
                official_bigram_generation(packet_hex),
                reference_official_bigram(packet_hex),
            )

    def test_one_and_two_packet_flows_are_retained_by_primary_policy(self):
        fmt = TrafficFormerFormat(min_packets=1)
        one = encode_flow([packet(b"\x45\x00\x00\x3c")], fmt)
        two = encode_flow(
            [packet(b"\x45\x00"), packet(b"\x17\x03", timestamp=2.0)], fmt
        )
        self.assertIsNotNone(one)
        self.assertIsNotNone(two)
        self.assertEqual(one.used_packet_count, 1)
        self.assertEqual(two.used_packet_count, 2)
        self.assertEqual(one.text.count("[SEP]"), 1)
        self.assertEqual(two.text.count("[SEP]"), 2)

    def test_strict_policy_filters_two_packets(self):
        fmt = TrafficFormerFormat(policy="strict_min3", min_packets=3)
        self.assertIsNone(encode_flow([packet(b"\x45\x00"), packet(b"\x17\x03")], fmt))

    def test_first_five_packets_are_used_without_synthetic_padding(self):
        fmt = TrafficFormerFormat(min_packets=1, max_packets=5)
        packets = [packet(bytes([index, index + 1]), timestamp=float(index)) for index in range(6)]
        encoded = encode_flow(packets, fmt)
        self.assertEqual(encoded.packet_count, 6)
        self.assertEqual(encoded.used_packet_count, 5)
        self.assertEqual(encoded.text.count("[SEP]"), 5)

    def test_primary_and_strict_text_match_on_common_cohort(self):
        packets = [packet(b"\x45\x00\x00\x3c", timestamp=float(index)) for index in range(3)]
        primary = encode_flow(packets, TrafficFormerFormat(min_packets=1))
        strict = encode_flow(
            packets,
            TrafficFormerFormat(policy="strict_min3", min_packets=3),
        )
        self.assertEqual(primary.text, strict.text)

    def test_short_captured_frame_is_not_extended(self):
        encoded = encode_flow([packet(b"\x45")], TrafficFormerFormat(min_packets=1))
        self.assertEqual(encoded.text, "[SEP] ")
        self.assertEqual(encoded.token_count, 0)


class TrafficFormerGenerationTests(unittest.TestCase):
    def test_end_to_end_outputs_keep_flow_mapping(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            flows_dir = root / "flows"
            output_dir = root / "output"
            flows_dir.mkdir()
            payload = {
                "split": "Benign",
                "class_name": "Demo",
                "source_file": "demo.pcap",
                "n_flows": 2,
                "n_packets": 3,
                "skipped_packets": 0,
                "packets": {
                    "Demo__000001": [packet(b"\x45\x00\x00\x3c")],
                    "Demo__000002": [
                        packet(b"\x45\x00"),
                        packet(b"\x17\x03", timestamp=2.0),
                    ],
                },
            }
            with (flows_dir / "Benign__Demo__demo.pkl").open("wb") as fh:
                pickle.dump(payload, fh)

            summary = generate_dataset(
                flows_dir,
                output_dir,
                TrafficFormerFormat(min_packets=1),
            )
            self.assertEqual(summary["total_flows"], 2)
            self.assertEqual(summary["retained_flows"], 2)

            with (output_dir / "trafficformer_all.tsv").open(
                newline="", encoding="utf-8"
            ) as fh:
                rows = list(csv.reader(fh, delimiter="\t"))
            self.assertEqual(rows[0], ["label", "text_a"])
            self.assertEqual(len(rows), 3)

            with (output_dir / "flow_map.csv").open(
                newline="", encoding="utf-8"
            ) as fh:
                mappings = list(csv.DictReader(fh))
            self.assertEqual([row["flow_id"] for row in mappings], [
                "Demo__000001",
                "Demo__000002",
            ])
            self.assertEqual([row["tsv_line_number"] for row in mappings], ["2", "3"])

            label_map = json.loads((output_dir / "label_map.json").read_text(encoding="utf-8"))
            self.assertEqual(label_map["class_to_id"], {"Demo": 0})

    def test_existing_outputs_are_not_overwritten_by_default(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            flows_dir = root / "flows"
            output_dir = root / "output"
            flows_dir.mkdir()
            output_dir.mkdir()
            (output_dir / "trafficformer_all.tsv").write_text("preserve\n", encoding="utf-8")
            payload = {
                "split": "Benign",
                "class_name": "Demo",
                "source_file": "demo.pcap",
                "packets": {"flow": [packet(b"\x45\x00")]},
            }
            with (flows_dir / "Benign__Demo__demo.pkl").open("wb") as fh:
                pickle.dump(payload, fh)

            with self.assertRaises(FileExistsError):
                generate_dataset(
                    flows_dir,
                    output_dir,
                    TrafficFormerFormat(min_packets=1),
                )
            self.assertEqual(
                (output_dir / "trafficformer_all.tsv").read_text(encoding="utf-8"),
                "preserve\n",
            )

    def test_cli_generates_primary_policy_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            flows_dir = root / "flows"
            output_dir = root / "cli-output"
            flows_dir.mkdir()
            payload = {
                "split": "Malware",
                "class_name": "CliDemo",
                "source_file": "cli-demo.pcap",
                "packets": {"cli-flow": [packet(b"\x45\x00\x00\x3c")]},
            }
            with (flows_dir / "Malware__CliDemo__cli-demo.pkl").open("wb") as fh:
                pickle.dump(payload, fh)

            project_root = Path(__file__).resolve().parents[1]
            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/task03_generate_trafficformer_input.py",
                    "--policy",
                    "compatible_min1",
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
            self.assertTrue((output_dir / "trafficformer_all.tsv").is_file())
            self.assertTrue((output_dir / "flow_map.csv").is_file())
            summary = json.loads(
                (output_dir / "generation_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["policy"], "compatible_min1")
            self.assertEqual(summary["retained_flows"], 1)


if __name__ == "__main__":
    unittest.main()
