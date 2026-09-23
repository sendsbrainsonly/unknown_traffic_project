#!/usr/bin/env python3
"""Run the two exact-config telemetry replays serially on one selected GPU."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().with_name("replay_with_telemetry.py")


def main() -> None:
    for protocol_id in ("medium_seed2025", "medium_seed2026"):
        print(f"SERIAL_START {protocol_id}", flush=True)
        completed = subprocess.run(
            [sys.executable, str(SCRIPT), "--protocol-id", protocol_id],
            check=False,
        )
        print(f"SERIAL_END {protocol_id} exit={completed.returncode}", flush=True)
        if completed.returncode != 0:
            raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
