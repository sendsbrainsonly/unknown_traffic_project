# -*- coding: utf-8 -*-
"""Focused tests for Stage 2.6 component / shortcut audit helpers."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.task10g_component_shortcut_audit import (
    _metadata_from_packets,
    _safe_output_dir,
    add_relative_time,
    epsilon_squared_kruskal,
    fixed_k_for_classes,
    port_category,
)


def ipv4_tcp_frame(src_port=1234, dst_port=443):
    frame = bytearray(14 + 20 + 20)
    frame[12:14] = b"\x08\x00"
    frame[14] = 0x45
    frame[23] = 6
    frame[26:30] = bytes([10, 0, 0, 1])
    frame[30:34] = bytes([10, 0, 0, 2])
    frame[34:36] = int(src_port).to_bytes(2, "big")
    frame[36:38] = int(dst_port).to_bytes(2, "big")
    return bytes(frame)


class Stage26ProtocolTests(unittest.TestCase):
    def test_fixed_k_is_resolved_by_name_not_label_id(self):
        result = fixed_k_for_classes([
            ("Outlook", 99), ("FTP", 12), ("Cridex", 3), ("Miuref", 8)
        ])
        self.assertEqual(result, {99: 5, 12: 5, 3: 4, 8: 5})

    def test_port_category_boundaries(self):
        self.assertEqual(port_category(0), "well-known")
        self.assertEqual(port_category(1023), "well-known")
        self.assertEqual(port_category(1024), "registered")
        self.assertEqual(port_category(49151), "registered")
        self.assertEqual(port_category(49152), "dynamic")
        self.assertEqual(port_category(65535), "dynamic")
        self.assertIsNone(port_category(65536))

    def test_metadata_is_recovered_from_existing_packet_tuple(self):
        packets = [
            (10.0, 60, 64, 1, ipv4_tcp_frame(2222, 443)),
            (10.5, 70, 74, 0, ipv4_tcp_frame(443, 2222)),
        ]
        result = _metadata_from_packets(packets, "Demo.pcap")
        self.assertEqual(result["source_pcap"], "Demo.pcap")
        self.assertEqual(result["packet_count"], 2)
        self.assertEqual(result["total_bytes"], 130)
        self.assertEqual(result["forward_packet_count"], 1)
        self.assertEqual(result["backward_packet_count"], 1)
        self.assertEqual(result["src_port"], 2222)
        self.assertEqual(result["dst_port"], 443)
        self.assertAlmostEqual(result["duration"], 0.5)
        self.assertAlmostEqual(result["start_time"], 10.0)

    def test_empty_metadata_stays_missing_not_fabricated(self):
        result = _metadata_from_packets([], "Demo.pcap")
        self.assertEqual(result["source_pcap"], "Demo.pcap")
        self.assertIsNone(result["packet_count"])
        self.assertIsNone(result["src_port"])
        self.assertIsNone(result["start_time"])

    def test_relative_time_ranges_are_fit_on_train_only(self):
        rows = [
            {"split": "train", "source_pcap": "a", "start_time": 10.0},
            {"split": "train", "source_pcap": "a", "start_time": 20.0},
            {"split": "val", "source_pcap": "a", "start_time": 30.0},
        ]
        ranges = add_relative_time(rows)
        self.assertEqual(ranges["a"]["train_max"], 20.0)
        self.assertAlmostEqual(rows[2]["relative_time"], 2.0)
        self.assertEqual(rows[2]["relative_time_bin"], 9)

    def test_epsilon_squared_formula(self):
        self.assertAlmostEqual(epsilon_squared_kruskal(12.0, 100, 4), 9 / 96)
        self.assertEqual(epsilon_squared_kruskal(1.0, 100, 4), 0.0)
        self.assertTrue(np.isnan(epsilon_squared_kruskal(1.0, 4, 4)))

    def test_stage2_and_stage25_are_protected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for path in (
                "outputs/stage2",
                "outputs/stage2/sub",
                "outputs/stage2_5_covariance_diagnosis",
                "outputs/stage2_5_covariance_diagnosis/sub",
            ):
                with self.assertRaises(ValueError):
                    _safe_output_dir(root, path)
            allowed = _safe_output_dir(root, "outputs/stage2_6_component_shortcut_audit")
            self.assertEqual(allowed, (root / "outputs/stage2_6_component_shortcut_audit").resolve())


if __name__ == "__main__":
    unittest.main()
