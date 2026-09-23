#!/usr/bin/env python3
"""Run the four pre-registered Stage 14C-4 jobs sequentially."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "train_cleaned.py"
TASKS = (
    ("medium_seed2025", "protocol"),
    ("medium_seed2026", "protocol"),
    ("medium_seed2025", "fixed2022"),
    ("medium_seed2026", "fixed2022"),
)


def main() -> None:
    records: list[dict[str, object]] = []
    for protocol_id, seed_mode in TASKS:
        run_id = f"{protocol_id}_{seed_mode}"
        print(f"SERIAL_START {run_id}", flush=True)
        process = subprocess.run(
            [sys.executable, str(SCRIPT), "--protocol-id", protocol_id, "--seed-mode", seed_mode],
            cwd=ROOT.parent,
            check=False,
        )
        records.append({"run_id": run_id, "exit_code": process.returncode, "status": "success" if process.returncode == 0 else "failed"})
        (ROOT / "run_queue_status.json").write_text(json.dumps({"records": records}, indent=2) + "\n", encoding="utf-8")
        print(f"SERIAL_END {run_id} exit={process.returncode}", flush=True)
        if process.returncode != 0:
            raise SystemExit(process.returncode)
    (ROOT / "run_queue_status.json").write_text(json.dumps({"records": records, "status": "PASS"}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
