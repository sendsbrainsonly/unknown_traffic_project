#!/usr/bin/env python3
"""Independent integrity checks for the Stage46 continuation closeout."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
STAGE = HERE.parent


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load(path: Path):
    return json.loads(path.read_text())


def main() -> None:
    inventory = load(HERE / "artifact_inventory.json")
    if inventory["status"] != "PASS":
        raise RuntimeError("artifact inventory status is not PASS")
    for item in inventory["files"]:
        path = HERE / item["path"]
        if not path.is_file() or path.stat().st_size != item["size_bytes"]:
            raise RuntimeError(f"missing/size-mismatched artifact: {item['path']}")
        if digest(path) != item["sha256"]:
            raise RuntimeError(f"SHA256 mismatch: {item['path']}")

    preflight = load(HERE / "preflight.json")
    if preflight["status"] != "PASS" or preflight["tests"] != 4:
        raise RuntimeError("CPU recovery preflight did not pass 4/4")
    for dataset in ("iscx_vpn", "iscx_tor"):
        smoke = load(HERE / "smoke" / f"{dataset}.json")
        if smoke["status"] != "PASS" or smoke["known_test_used"] or smoke["unknown_used"]:
            raise RuntimeError(f"GPU restore smoke invalid for {dataset}")
        summary = load(HERE / "runs" / dataset / "resume_summary.json")
        if summary["status"] != "PASS" or summary["completed_epochs_total"] != 100:
            raise RuntimeError(f"resume summary invalid for {dataset}")
        if summary["known_test_used"] or summary["unknown_used"]:
            raise RuntimeError(f"Test/Unknown leaked into training for {dataset}")
        history = [json.loads(row) for row in (HERE / "runs" / dataset / "training_history.jsonl").read_text().splitlines()]
        if [int(row["epoch"]) for row in history] != list(range(76, 101)):
            raise RuntimeError(f"resumed epoch history incomplete for {dataset}")
        cp = HERE / "runs" / dataset / "prefix100_best.pt"
        if digest(cp) != summary["checkpoint"]["checkpoint_sha256"]:
            raise RuntimeError(f"selected checkpoint hash mismatch for {dataset}")
        combined = [json.loads(row) for row in (HERE / "combined" / f"{dataset}_training_history.jsonl").read_text().splitlines()]
        if [int(row["epoch"]) for row in combined] != list(range(1, 101)):
            raise RuntimeError(f"combined trajectory incomplete for {dataset}")

    selected = load(STAGE / "selected_before_test.json")
    if selected["status"] != "PASS" or len(selected["selections"]) != 6 or selected["test_used_for_selection"] or selected["unknown_used"]:
        raise RuntimeError("six-checkpoint freeze barrier failed")
    verification = load(STAGE / "completion_verification.json")
    if verification["status"] != "PASS" or verification["checkpoints"] != 6 or verification["replayed_metric_rows"] != 12 or verification["test_used_for_selection"] or verification["unknown_used"]:
        raise RuntimeError("parent completion verification failed")
    with (STAGE / "run_results.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 12 or sum(row["role"] == "known_test" for row in rows) != 6:
        raise RuntimeError("expected 12 rows with 6 post-freeze Known Test rows")
    print(json.dumps({"status": "PASS", "verified_artifacts": inventory["file_count"],
                      "inventory_bytes": inventory["total_bytes"], "resumed_runs": 2,
                      "epochs_per_run": 100, "selected_checkpoints": 6,
                      "metric_rows": len(rows), "known_test_used_for_selection": 0,
                      "unknown_used": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
