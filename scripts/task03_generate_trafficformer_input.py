# -*- coding: utf-8 -*-
"""Generate TrafficFormer-format TSV and flow_id mapping from Stage 0 PKLs."""
import argparse
import json
import os
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.trafficformer_input import (  # noqa: E402
    format_from_config,
    generate_dataset,
)


def _project_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main():
    parser = argparse.ArgumentParser(
        description="Task 0.3: Stage 0 PKLs -> TrafficFormer TSV + flow_id map"
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "encoder" / "trafficformer_input.yaml"),
    )
    parser.add_argument(
        "--policy",
        default=None,
        help="Policy from config (default: config.default_policy)",
    )
    parser.add_argument("--flows-dir", default=None, help="Override inputs.flows_dir")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Exact output directory; default is outputs.root_dir/POLICY",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Atomically replace this generator's existing output files",
    )
    args = parser.parse_args()

    config_path = _project_path(args.config)
    with config_path.open(encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    if not isinstance(config, dict):
        raise ValueError(f"configuration must be a mapping: {config_path}")

    policy = args.policy or config["default_policy"]
    fmt = format_from_config(config, policy)
    flows_dir = _project_path(args.flows_dir or config["inputs"]["flows_dir"])
    if args.output_dir:
        output_dir = _project_path(args.output_dir)
    else:
        output_dir = _project_path(config["outputs"]["root_dir"]) / policy

    summary = generate_dataset(
        flows_dir=flows_dir,
        output_dir=output_dir,
        fmt=fmt,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    print("output_dir:", output_dir)


if __name__ == "__main__":
    main()
