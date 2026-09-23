#!/usr/bin/env python3
"""Sequentially execute assigned frozen protocols on one visible GPU."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("protocol_ids", nargs="+")
    args = parser.parse_args()
    script = Path(__file__).with_name("train_one.py")
    failures: list[str] = []
    for protocol_id in args.protocol_ids:
        print(f"WORKER_START {protocol_id}", flush=True)
        result = subprocess.run(
            [sys.executable, str(script), "--protocol-id", protocol_id, "--device", "cuda:0"],
            check=False,
        )
        print(f"WORKER_END {protocol_id} exit={result.returncode}", flush=True)
        if result.returncode != 0:
            failures.append(protocol_id)
    if failures:
        raise SystemExit(f"failed protocols: {failures}")


if __name__ == "__main__":
    main()

