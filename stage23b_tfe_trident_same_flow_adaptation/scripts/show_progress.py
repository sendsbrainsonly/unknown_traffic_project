#!/usr/bin/env python3
"""Read-only aggregate progress for Stage23B."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
path = root / "progress.json"
if not path.is_file():
    print("Stage23B queue has not written progress.json yet")
else:
    progress = json.loads(path.read_text(encoding="utf-8"))
    print(f"Stage23B {progress['stage']} at {progress['utc']}")
    print(f"Trident validation selections: {progress['trident_selected']}/4")
    print("TFE dataset-seed           epochs  selected  test  exit")
    for name, data in sorted(progress["tfe"].items()):
        print(f"{name:25} {data['completed_epochs']:6}  "
              f"{str(data['selected']):8}  {str(data['test_complete']):4}  "
              f"{data['exit_code']}")
    print("Test inputs:", progress["stage20_test_materialized"])
    if (root / "queue_failure.json").is_file():
        failure = json.loads((root / "queue_failure.json").read_text(encoding="utf-8"))
        print("QUEUE FAILED:", failure["error"])
    elif (root / "QUEUE_COMPLETE").is_file():
        print("QUEUE COMPLETE: see completion_verification.json")
