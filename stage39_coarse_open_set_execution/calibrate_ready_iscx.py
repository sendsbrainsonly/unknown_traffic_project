"""Calibrate completed ISCX service holdouts from Known Train/Validation only."""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

from freeze_protocols import PROJECT, ROOT, sha


def progress(status: str, completed: list[str], **extra) -> None:
    record = {"status": status, "updated_at_utc": datetime.now(timezone.utc).isoformat(),
              "completed": completed, "known_only": True, **extra}
    temporary = ROOT / "iscx_calibration_progress.json.tmp"
    temporary.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, ROOT / "iscx_calibration_progress.json")
    print(json.dumps(record, ensure_ascii=False), flush=True)


def main() -> None:
    with (ROOT / "protocol_index.csv").open(newline="", encoding="utf-8") as stream:
        planned = [row for row in csv.DictReader(stream)
                   if row["dataset"] in ("iscx_vpn", "iscx_tor") and
                   row["new_training_required"] == "True"]
    if len(planned) != 8:
        raise RuntimeError(f"expected eight frozen ISCX settings, got {len(planned)}")
    completed: list[str] = []
    try:
        for item in planned:
            dataset, slug = item["dataset"], item["slug"]
            fold = ROOT / "settings" / dataset / slug
            protocol = json.loads((fold / "protocol.json").read_text(encoding="utf-8"))
            if sha(fold / "role_manifest.csv") != protocol["role_manifest_sha256"]:
                raise RuntimeError(f"frozen role manifest changed: {dataset}/{slug}")
            run = fold / "runs" / dataset / protocol["protocol_id"]
            if not (run / "T0_equal/SUCCESS").is_file():
                raise RuntimeError(f"Known-only fusion missing: {dataset}/{slug}")
            marker = fold / "detection/calibration_verification.json"
            if marker.is_file():
                if json.loads(marker.read_text(encoding="utf-8"))["status"] != "PASS":
                    raise RuntimeError(f"calibration marker invalid: {dataset}/{slug}")
            else:
                progress("CALIBRATING", completed, current=f"{dataset}/{slug}")
                subprocess.run([sys.executable, str(ROOT / "detect_iscx_fold.py"),
                                "calibrate", "--dataset", dataset, "--fold", slug],
                               cwd=PROJECT, check=True)
                if not marker.is_file() or \
                   json.loads(marker.read_text(encoding="utf-8"))["status"] != "PASS":
                    raise RuntimeError(f"calibration did not verify: {dataset}/{slug}")
            completed.append(f"{dataset}/{slug}")
            progress("RUNNING", completed)
        progress("ISCX_KNOWN_CALIBRATION_COMPLETE", completed,
                 unknown_feature_values_loaded=0, test_feature_values_loaded=0)
    except BaseException as exc:
        progress("FAILED", completed, error=repr(exc))
        raise


if __name__ == "__main__":
    main()
