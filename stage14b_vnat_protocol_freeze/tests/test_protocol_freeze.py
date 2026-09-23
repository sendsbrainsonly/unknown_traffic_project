from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from freeze_vnat_protocol import APPLICATIONS, balanced_nested_unknown_schedule, known_split_map


def test_unknown_schedule_is_nested_and_balanced() -> None:
    _, schedule = balanced_nested_unknown_schedule(APPLICATIONS)
    assert len(schedule) == 15
    by_seed = {}
    for row in schedule:
        by_seed.setdefault(row["seed"], {})[row["setting"]] = set(row["unknown_applications"])
    for settings in by_seed.values():
        assert settings["Low"] < settings["Medium"] < settings["High"]
    low_counts = Counter(app for row in schedule if row["setting"] == "Low" for app in row["unknown_applications"])
    high_counts = Counter(app for row in schedule if row["setting"] == "High" for app in row["unknown_applications"])
    assert set(low_counts.values()) == {1}
    assert set(high_counts.values()) == {2}


def test_known_split_is_nonempty_and_deterministic() -> None:
    flows = [
        {"application": "example", "flow_uid": f"flow-{index}"}
        for index in range(44)
    ]
    first = known_split_map(flows, 2022)
    second = known_split_map(flows, 2022)
    assert first == second
    counts = Counter(first.values())
    assert counts == {"train": 36, "validation": 4, "test": 4}
