from __future__ import annotations

import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run_stage14a5 import exhaustive_maximin, quarantine_rows


def group(name: str, count: int) -> dict[str, object]:
    return {"group_id": name, "flow_count": count}


def test_cross_application_quarantine_is_exactly_four_rows() -> None:
    keys, rows = quarantine_rows()
    assert len(keys) == len(rows) == 4
    assert {row["application"] for row in rows} == {"rsync", "sftp"}


def test_exact_maximin_partition() -> None:
    partition, evaluated = exhaustive_maximin([
        group("g1", 8), group("g2", 7), group("g3", 6), group("g4", 3), group("g5", 1)
    ])
    counts = {
        split: sum(int(item["flow_count"]) for item in bucket)
        for split, bucket in partition.items()
    }
    assert counts == {"train": 9, "val": 8, "test": 8}
    assert evaluated > 0
