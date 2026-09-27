#!/usr/bin/env python3
"""Run Stage34's read-only PCAP extractor for the frozen balanced Test IDs."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from prepare_balanced import ROOT, MANIFEST, sha, verified_rows
from run_balanced import CIC, frozen_head_gate

SOURCE = ROOT.parent / "stage34_ustc_cic_closed_set"
sys.path.insert(0, str(SOURCE))
import cic_build_inputs as builder  # noqa: E402


def main() -> None:
    verified_rows()
    gate = json.loads((ROOT / "cicids2017_protocol_v2_audit.json").read_text())
    if gate["v2_manifest_sha256"] != sha(MANIFEST) or gate["actual_protocol"] != "Stage34B-balanced-1to1":
        raise RuntimeError("balanced builder compatibility hash mismatch")
    frozen_head_gate()
    builder.ROOT = ROOT
    builder.CIC = CIC
    builder.MANIFEST = MANIFEST
    builder.frozen_head_gate = frozen_head_gate
    old = sys.argv
    sys.argv = [str(builder.__file__), "--phase", "test"]
    try:
        builder.main()
    finally:
        sys.argv = old


if __name__ == "__main__":
    main()
