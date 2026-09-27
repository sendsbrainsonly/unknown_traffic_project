#!/usr/bin/env python3
"""Finalize the preserved Stage23B evidence after the unattended queue exits."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

OUT = Path(__file__).resolve().parents[1]
PROJECT = OUT.parent
WORKSPACE = PROJECT.parent.parent
PRESERVE = WORKSPACE / "skills/experiment-data-preservation/scripts"
STAGE20 = PROJECT / "stage20_dual_coarse_service_protocol/closed_service_manifest.csv"
STAGE23 = PROJECT / "stage23_closed_set_method_table/stage23_full_run_results.csv"
EXPECTED = {
    STAGE20: "6404d496f50a6f7f6f397d125ad0cd653c9df152d8db1b9ed2ad2310967950bb",
    STAGE23: "b10cbf5ccead40312768301c5379027ce983f4bcb8ec3cb31338f5b80ced0bdf",
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    while not (OUT / "QUEUE_COMPLETE").is_file():
        if (OUT / "queue_failure.json").is_file():
            raise RuntimeError("Stage23B queue failed; inspect queue_failure.json")
        time.sleep(30)
    for path, expected in EXPECTED.items():
        actual = digest(path)
        if actual != expected:
            raise RuntimeError(f"protected source changed: {path}, {actual}")
    for script, extra in (
        ("refresh_artifact_manifest.py", []),
        ("validate_experiment_bundle.py", ["--verify-hashes"]),
    ):
        subprocess.run([sys.executable, "-B", str(PRESERVE / script), str(OUT), *extra], check=True)
    verification = json.loads((OUT / "completion_verification.json").read_text(encoding="utf-8"))
    if verification["status"] != "PASS" or verification["runs"] != 8:
        raise RuntimeError("independent Test replay did not pass")
    (OUT / "POST_VERIFICATION_COMPLETE").write_text("PASS\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "runs": 8,
                      "protected_hashes": {str(path): digest(path) for path in EXPECTED}}), flush=True)


if __name__ == "__main__":
    main()
