#!/usr/bin/env python3
"""Run the Stage 5 method/protocol freeze without any Unknown evaluation."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

import freeze_cstnet_protocol
import run_known_only_sanity


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "tests"))
from verify_stage5 import verify  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    started = utc_now()
    known = run_known_only_sanity.run(output_root)
    protocol = freeze_cstnet_protocol.run(output_root)
    result = verify(output_root)
    metadata = {
        "run_id": "stage5_dual_gate_boundary_freeze_v1",
        "started_at_utc": started,
        "finished_at_utc": utc_now(),
        "status": "completed",
        "final_gate": result["gate"],
        "claim_scope": "METHOD_FREEZE_PLUS_PROTOCOL_FREEZE_NO_UNKNOWN_EVALUATION",
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip(),
        "environment": {
            "python": platform.python_version(),
            "prefix": sys.prefix,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "dgsb_rule_sha256": known["rule_sha256"],
        "ustc_unknown_inputs_accessed": 0,
        "cstnet_training_executed": False,
        "cstnet_unknown_scores_accessed": False,
        "created_before_unknown_evaluation": True,
        "dataset_inventory_fingerprint_sha256": protocol["dataset"]["inventory_fingerprint_sha256"],
        "verification": result,
    }
    path = output_root / "stage5_run_metadata.json"
    content = json.dumps(metadata, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != content:
        raise RuntimeError(f"refusing to modify completed Stage 5 metadata: {path}")
    if not path.exists():
        path.write_text(content, encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
