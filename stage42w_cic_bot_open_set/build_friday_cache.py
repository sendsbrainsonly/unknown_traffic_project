#!/usr/bin/env python3
"""Route the frozen Friday packet-view builder to isolated Bot outputs."""
from __future__ import annotations

import importlib.util
import json
import sys

sys.dont_write_bytecode = True

from stage42w_common import MANIFEST, PROJECT, PROTOCOL, ROOT, UNIT, digest, progress

SOURCE = PROJECT / "stage42v_cic_ddos_open_set/build_friday_cache.py"


def main() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if digest(SOURCE) != protocol["reused_stage42v_cache_sha256"]:
        raise RuntimeError("reused Friday cache builder changed")
    sys.path.insert(0, str(SOURCE.parent))
    spec = importlib.util.spec_from_file_location("stage42w_reused_friday_cache", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen Friday cache builder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name, value in {"ROOT": ROOT, "UNIT": UNIT, "MANIFEST": MANIFEST,
                        "PROTOCOL": PROTOCOL, "progress": progress}.items():
        setattr(module, name, value)
    module.main()


if __name__ == "__main__":
    main()
