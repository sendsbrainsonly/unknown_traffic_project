from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from behavior_features import BehaviorAccumulator  # noqa: E402
from common import feature_names  # noqa: E402


def as_dict(acc: BehaviorAccumulator, max_packets=None):
    return dict(zip(feature_names(), acc.vector(max_packets)))


def test_single_packet_missingness_is_not_fabricated_zero():
    acc = BehaviorAccumulator("one")
    acc.add(1.0, 100, 40, ("a", "1"), ("b", "2"))
    values = as_dict(acc)
    assert values["packet_count"] == 1
    assert values["duration_seconds"] == 0
    assert np.isnan(values["packets_per_second"])
    assert np.isnan(values["iat_mean_seconds"])
    assert np.isnan(values["inter_burst_iat_mean"])
    assert values["burst_count"] == 1
    assert values["burst_duration_mean"] == 0


def test_burst_definition_is_contiguous_direction_without_time_threshold():
    acc = BehaviorAccumulator("bursts")
    packets = [
        (0.0, 10, 1), (1.0, 20, 1), (100.0, 30, -1),
        (101.0, 40, -1), (102.0, 50, 1),
    ]
    for timestamp, length, direction in packets:
        acc.add(timestamp, length, length, None, None, explicit_direction=direction)
    values = as_dict(acc)
    assert values["burst_count"] == 3
    assert values["forward_burst_count"] == 2
    assert values["reverse_burst_count"] == 1
    assert values["burst_packet_count_max"] == 2
    assert values["direction_changes"] == 2
    assert values["inter_burst_iat_mean"] == 50.0


def test_early16_uses_only_first_16_packets():
    acc = BehaviorAccumulator("early")
    for index in range(20):
        acc.add(float(index), index + 1, index, None, None, explicit_direction=1 if index < 10 else -1)
    full = as_dict(acc)
    early = as_dict(acc, 16)
    assert full["packet_count"] == 20
    assert early["packet_count"] == 16
    assert full["total_bytes"] == sum(range(1, 21))
    assert early["total_bytes"] == sum(range(1, 17))
    assert early["duration_seconds"] == 15
    assert early["forward_packet_count"] == 10
    assert early["reverse_packet_count"] == 6
