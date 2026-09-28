#!/usr/bin/env python3
"""Run unchanged Stage42-S extraction/scoring/replay against GoldenEye paths."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys

sys.dont_write_bytecode = True  # importing frozen Stage42-S code must not mutate its cache files

from stage42t_common import MANIFEST, PROJECT, PROTOCOL, ROOT, UNIT, digest, progress

BASE = PROJECT / "stage42s_cic_favorable_open_set"
PHASE_MODULE = {
    "cache": "build_unknown_cache.py",
    "evaluate": "evaluate_candidate.py",
    "verify": "verify_candidate.py",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=PHASE_MODULE)
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol["status"] != "PASS" or digest(MANIFEST) != protocol["candidate_manifest_sha256"]:
        raise RuntimeError("GoldenEye candidate freeze changed")
    for filename, expected in protocol["reused_stage42s_code_sha256"].items():
        if digest(BASE / filename) != expected:
            raise RuntimeError(f"reused Stage42-S code changed: {filename}")
    source = BASE / PHASE_MODULE[args.phase]
    sys.path.insert(0, str(BASE))
    spec = importlib.util.spec_from_file_location(f"stage42t_reused_{args.phase}", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load frozen Stage42-S code: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name, value in {
        "ROOT": ROOT, "UNIT": UNIT, "MANIFEST": MANIFEST,
        "PROTOCOL": PROTOCOL, "progress": progress,
    }.items():
        setattr(module, name, value)
    if args.phase == "evaluate":
        module.OUT = ROOT / UNIT / "detection"
    module.main()


if __name__ == "__main__":
    main()
