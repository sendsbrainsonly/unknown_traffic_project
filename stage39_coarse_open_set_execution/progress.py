"""Read-only aggregate progress for the frozen Stage39 experiment."""
from __future__ import annotations

import csv
import json

from freeze_protocols import ROOT


def state(name):
    path = ROOT / f"{name}_queue_progress.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"status": "NOT_STARTED"}


with (ROOT / "protocol_index.csv").open(newline="", encoding="utf-8") as stream:
    rows = [row for row in csv.DictReader(stream) if row["new_training_required"] == "True"]
trained, evaluated = 0, 0
for row in rows:
    folder = ROOT / "settings" / row["dataset"] / row["slug"]
    protocol = json.loads((folder / "protocol.json").read_text(encoding="utf-8"))
    name = "vnat" if row["dataset"] == "VNAT" else row["dataset"]
    trained += (folder / "runs" / name / protocol["protocol_id"] / "T0_equal/SUCCESS").is_file()
    evaluated += (folder / "detection/deployment_verification.json").is_file()
print(json.dumps({"frozen_new_training_settings": len(rows), "trained_settings": trained,
                  "verified_detection_settings": evaluated,
                  "training": state("training"), "evaluation": state("evaluation"),
                  "aggregate": state("aggregate")}, indent=2, ensure_ascii=False))
