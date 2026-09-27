"""Locked Stage32 Test input gate and frozen checkpoint verification."""
from __future__ import annotations

import json
import sys

from common import (DATASETS, ROOT, S31, VNAT_PROTOCOL, freeze_sources,
                    sha256, write_json)


def ensure_three_heads_frozen() -> dict:
    progress = json.loads((ROOT / "progress.json").read_text())
    allowed = {"THREE_HEADS_FROZEN_TEST_NOT_OPENED", "TEST_INPUT_RECOVERY",
               "TEST_INPUTS_READY", "TEST_EVALUATING", "TEST_COMPLETE"}
    if progress["status"] not in allowed or progress["completed_heads"] != 3:
        raise RuntimeError("all three Stage32 heads are not frozen; Test remains locked")
    frozen = json.loads((ROOT / "frozen_source_hashes_before.json").read_text())
    if freeze_sources() != frozen:
        raise RuntimeError("Stage32 frozen source hash changed before Test")
    checkpoints = {}
    for dataset in DATASETS:
        run = ROOT / "runs" / dataset
        if not (run / "SUCCESS").is_file():
            raise RuntimeError(f"incomplete Stage32 head: {dataset}")
        info = json.loads((run / "known_validation_metrics.json").read_text())
        hashes = info["checkpoint_hashes"]
        for name in ("adapters_best.pt", "T0_equal_best.pt"):
            if sha256(run / name) != hashes[name]:
                raise RuntimeError(f"frozen Stage32 checkpoint mismatch: {dataset}/{name}")
        checkpoints[dataset] = hashes
    return {"source_hashes": frozen, "head_hashes": checkpoints,
            "test_parameter_selection": 0, "unknown_features_loaded": 0}


def prepare_vnat_test() -> dict:
    """Reuse Stage31 packet encoding functions, redirecting all writes to Stage32."""
    locked = ensure_three_heads_frozen()
    if not (ROOT / "selected_heads_before_test.json").exists():
        write_json(ROOT / "selected_heads_before_test.json", locked)
    else:
        if json.loads((ROOT / "selected_heads_before_test.json").read_text()) != locked:
            raise RuntimeError("selected head/source evidence changed")
    sys.path.insert(0, str(S31))
    import build_vnat_mfr  # noqa: E402
    import build_vnat_tf_fig  # noqa: E402

    for module in (build_vnat_tf_fig, build_vnat_mfr):
        module.OUT = ROOT
        module.ensure_all_heads_frozen = ensure_three_heads_frozen
    for module in (build_vnat_tf_fig, build_vnat_mfr):
        saved = sys.argv
        try:
            sys.argv = [str(module.__file__), "--protocol", VNAT_PROTOCOL, "--phase", "test"]
            module.main()
        finally:
            sys.argv = saved
    root = ROOT / "input_caches" / "vnat" / VNAT_PROTOCOL
    audits = {}
    for name in ("tf_fig_test", "yatc_mfr_test"):
        audit = json.loads((root / name / "cache_audit.json").read_text())
        if audit["status"] != "PASS" or audit["known_test_features_materialized"] != 1960:
            raise RuntimeError(f"VNAT Known Test cache invalid: {name}")
        audits[name] = audit
    return {"status": "PASS", "vn_test_flows": 1960,
            "cache_names": list(audits), "unknown_features_loaded": 0}
