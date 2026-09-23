#!/usr/bin/env python3
"""Run the three frozen seeds sequentially for one DQ-4 model."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("M-B", "M-C", "M-D"), required=True)
    args = parser.parse_args()
    trainer = Path(__file__).with_name("train_dq4.py")
    for seed in (2022, 2023, 2024):
        subprocess.run(
            [sys.executable, str(trainer), "--model", args.model, "--seed", str(seed)],
            check=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
