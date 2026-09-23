#!/usr/bin/env python3
"""Create the project-local 20-class USTC-TFC2016 symlink view."""

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.ustc20_view import build_ustc20_view  # noqa: E402


DEFAULT_SOURCE = Path(
    "/home/birkenwald/data/SEU-WXY/WXY-SEU/TrafficClassifier/Dataset/"
    "ustc-tfc2016/USTC-TFC2016（whole）/extracted/V1/"
    "1.DataSet(USTC-TFC2016)"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a read-only USTC-TFC2016 20-class PCAP view"
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw_views" / "ustc_tfc2016_20class",
    )
    args = parser.parse_args()
    manifest = build_ustc20_view(args.source_root, args.output_root)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
