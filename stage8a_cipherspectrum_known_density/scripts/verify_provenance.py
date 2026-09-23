#!/usr/bin/env python3
"""Verify frozen Stage 6/7 assets before any Stage 8A representation access."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import STAGE8_ROOT, verify_all_provenance, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=STAGE8_ROOT / "outputs/summary/provenance_verification.json",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite provenance record: {args.output}")
    result = verify_all_provenance()
    write_json(args.output, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
