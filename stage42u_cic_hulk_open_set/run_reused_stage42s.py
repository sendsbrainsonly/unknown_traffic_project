#!/usr/bin/env python3
"""Reuse frozen Stage42-T path adapter with Hulk paths and unchanged scoring."""
from __future__ import annotations

import importlib.util
import json
import sys

sys.dont_write_bytecode = True

from stage42u_common import MANIFEST, PROJECT, PROTOCOL, ROOT, UNIT, digest, progress

RUNNER = PROJECT / "stage42t_cic_goldeneye_open_set/run_reused_stage42s.py"


def main() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if digest(MANIFEST) != protocol["candidate_manifest_sha256"]:
        raise RuntimeError("Hulk candidate manifest changed")
    if digest(RUNNER) != protocol["reused_stage42t_runner_sha256"]:
        raise RuntimeError("reused Stage42-T runner changed")
    sys.path.insert(0, str(RUNNER.parent))
    spec = importlib.util.spec_from_file_location("stage42u_reused_stage42t_runner", RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Stage42-T runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name, value in {
        "ROOT": ROOT, "UNIT": UNIT, "MANIFEST": MANIFEST,
        "PROTOCOL": PROTOCOL, "progress": progress,
    }.items():
        setattr(module, name, value)
    module.main()


if __name__ == "__main__":
    main()
