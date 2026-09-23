#!/usr/bin/env python3
"""Sequentially execute explicit pre-registered ablation variant/protocol pairs."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pairs", nargs="+", help="VARIANT:PROTOCOL")
    args = parser.parse_args()
    script = Path(__file__).with_name("run_ablation.py")
    failures: list[str] = []
    for pair in args.pairs:
        variant, protocol = pair.split(":", 1)
        print(f"WORKER_START {variant} {protocol}", flush=True)
        result = subprocess.run(
            [sys.executable, str(script), "--variant", variant, "--protocol-id", protocol],
            check=False,
        )
        print(f"WORKER_END {variant} {protocol} exit={result.returncode}", flush=True)
        if result.returncode:
            failures.append(pair)
    if failures:
        raise SystemExit(f"failed pairs: {failures}")


if __name__ == "__main__":
    main()
