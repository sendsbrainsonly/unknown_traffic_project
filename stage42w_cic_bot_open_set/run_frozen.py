#!/usr/bin/env python3
"""Reuse hash-pinned Stage42-V runner with isolated Bot paths."""
from __future__ import annotations

import importlib.util
import json
import sys

sys.dont_write_bytecode = True

from stage42w_common import MANIFEST, PROJECT, PROTOCOL, ROOT, UNIT, digest, progress

SOURCE = PROJECT / "stage42v_cic_ddos_open_set/run_frozen.py"


def main() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if digest(MANIFEST) != protocol["candidate_manifest_sha256"]:
        raise RuntimeError("Bot candidate manifest changed")
    if digest(SOURCE) != protocol["reused_stage42v_runner_sha256"]:
        raise RuntimeError("reused Stage42-V runner changed")
    sys.path.insert(0, str(SOURCE.parent))
    spec = importlib.util.spec_from_file_location("stage42w_reused_stage42v_runner", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Stage42-V runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # The reused runner imports its cache module by name; keep this bundle first.
    sys.path.insert(0, str(ROOT))
    for name, value in {"ROOT": ROOT, "UNIT": UNIT, "MANIFEST": MANIFEST,
                        "PROTOCOL": PROTOCOL, "progress": progress}.items():
        setattr(module, name, value)
    module.main()


if __name__ == "__main__":
    main()
