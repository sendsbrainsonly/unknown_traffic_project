#!/usr/bin/env python3
"""Small deterministic checks for Stage 15F-0 audit semantics."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from common import WINDOWS, WindowAccumulator  # noqa: E402


def test_complete_short_flow() -> None:
    acc = WindowAccumulator("f", "c", "s", "d")
    for index, length in enumerate((100, 200, 300)):
        acc.add(float(index), length, length - 14, length - 54, ("a", "1"), ("b", "2"), "ip:tcp")
    row = acc.row("x")
    assert row["packet_count"] == 3
    assert row["frame_bytes_first_8"] == 600
    assert row["byte_coverage_first_8"] == 1.0
    assert row["time_coverage_first_8"] == 1.0
    assert row["padding_ratio_first_8"] == 5 / 8
    assert row["complete_first_8"] is True


def test_early_window_long_flow() -> None:
    acc = WindowAccumulator("f", "c", "s", "d")
    for index in range(65):
        acc.add(float(index), 10, 8, 2, ("a", "1"), ("b", "2"), "ip:udp")
    row = acc.row("x")
    for window in WINDOWS:
        assert row[f"frame_bytes_first_{window}"] == window * 10
        assert row[f"byte_coverage_first_{window}"] == window / 65
        assert row[f"time_coverage_first_{window}"] == (window - 1) / 64
        assert row[f"truncated_first_{window}"] is True
        assert row[f"padding_ratio_first_{window}"] == 0


def test_feature_registry_ids() -> None:
    import json

    payload = json.loads((ROOT / "feature_group_definitions.json").read_text(encoding="utf-8"))
    assert [item["id"] for item in payload["feature_groups"]] == [
        "B0", "B1", "B2", "B3", "B4", "T1", "T2", "S1", "S2", "M1", "M2", "M3", "P1"
    ]


if __name__ == "__main__":
    test_complete_short_flow()
    test_early_window_long_flow()
    test_feature_registry_ids()
    print("3/3 PASS")
