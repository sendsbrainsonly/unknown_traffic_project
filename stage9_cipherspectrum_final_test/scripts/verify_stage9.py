#!/usr/bin/env python3
"""Independent recheck of frozen Stage 9 results, decisions, metrics, and hashes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from stage9_common import METHOD_ORDER, STAGE9_ROOT, VALID_SETTINGS, load_json, require, sha256_file, verify_all_provenance


def verify_hash_file(path: Path) -> int:
    count = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split(maxsplit=1)
        target = STAGE9_ROOT / relative
        require(target.is_file(), f"missing Stage 9 hash target: {target}")
        require(sha256_file(target) == expected, f"Stage 9 hash mismatch: {target}")
        count += 1
    return count


def main() -> None:
    provenance = verify_all_provenance()
    require(provenance["status"] == "PASS", "upstream provenance replay failed")
    test_open = load_json(STAGE9_ROOT / "outputs/test_open_provenance.json")
    require(test_open["created_before_any_test_result"] is True, "Test-open provenance invalid")
    setting_results = []
    for setting in VALID_SETTINGS:
        artifact = STAGE9_ROOT / "artifacts" / setting
        output = STAGE9_ROOT / "outputs" / setting
        require(load_json(artifact / "test_input_integrity.json")["status"] == "PASS", f"{setting}: input gate failed")
        require(load_json(artifact / "test_representation_integrity.json")["status"] == "PASS", f"{setting}: representation gate failed")
        known_manifest = pd.read_csv(artifact / "known_test_manifest.csv")
        unknown_manifest = pd.read_csv(artifact / "unknown_test_manifest.csv")
        known_details = pd.read_parquet(output / "known_sample_details.parquet")
        unknown_details = pd.read_parquet(output / "unknown_sample_details.parquet")
        known_metrics = pd.read_csv(output / "known_test_metrics.csv")
        unknown_metrics = pd.read_csv(output / "unknown_test_metrics.csv")
        require(set(known_metrics["method"]) == set(METHOD_ORDER), f"{setting}: missing Known methods")
        require(set(unknown_metrics["method"]) == set(METHOD_ORDER), f"{setting}: missing Unknown methods")
        require(len(known_details) == len(known_manifest) * len(METHOD_ORDER), f"{setting}: Known sample details length mismatch")
        require(len(unknown_details) == len(unknown_manifest) * len(METHOD_ORDER), f"{setting}: Unknown sample details length mismatch")
        for method in METHOD_ORDER:
            kd = known_details[known_details["method"] == method]
            ud = unknown_details[unknown_details["method"] == method]
            km = known_metrics[known_metrics["method"] == method].iloc[0]
            um = unknown_metrics[unknown_metrics["method"] == method].iloc[0]
            require(np.isclose(kd["final_accept"].mean(), km["known_acceptance"], rtol=0, atol=1e-15), f"{setting}/{method}: Known acceptance mismatch")
            require(np.isclose(ud["final_accept"].mean(), um["UFAR"], rtol=0, atol=1e-15), f"{setting}/{method}: UFAR mismatch")
            require(int(ud["final_accept"].sum()) == int(um["unknown_accepted_as_known_count"]), f"{setting}/{method}: Unknown count mismatch")
        dgsb_k = known_details[known_details["method"] == "DGSB-v2"]
        dgsb_u = unknown_details[unknown_details["method"] == "DGSB-v2"]
        for frame, role in ((dgsb_k, "Known"), (dgsb_u, "Unknown")):
            replay = frame["global_pass"].astype(bool) & frame["local_pass"].astype(bool)
            require(np.array_equal(replay.to_numpy(), frame["final_accept"].astype(bool).to_numpy()), f"{setting}: DGSB AND replay failed on {role}")
            replay_score = np.minimum(frame["global_margin"].to_numpy(), frame["local_margin"].to_numpy())
            require(np.allclose(replay_score, frame["detector_score"].to_numpy(), rtol=0, atol=1e-12), f"{setting}: DGSB score replay failed on {role}")
        bootstrap = pd.read_csv(output / "paired_bootstrap_primary.csv")
        require(set(bootstrap["bootstrap_iterations"]) == {1000} and set(bootstrap["seed"]) == {0}, f"{setting}: bootstrap freeze changed")
        hash_count = verify_hash_file(artifact / "test_bundle_hashes.sha256")
        setting_results.append({"setting": setting, "status": "PASS", "known_n": len(known_manifest), "unknown_n": len(unknown_manifest), "bundle_hashes_verified": hash_count})
    final_count = verify_hash_file(STAGE9_ROOT / "outputs/summary/stage9_final_hashes.sha256")
    final_gate = load_json(STAGE9_ROOT / "outputs/summary/final_gate.json")
    require(final_gate["post_test_tuning_performed"] is False, "post-Test tuning recorded")
    require(final_gate["methods_modified_after_test_open"] is False, "methods changed after Test open")
    payload = {
        "status": "PASS",
        "upstream_provenance": "PASS",
        "settings": setting_results,
        "final_hashes_verified": final_count,
        "final_gate": final_gate["gate"],
        "post_test_tuning_performed": False,
        "methods_frozen": list(METHOD_ORDER),
        "bootstrap_iterations": 1000,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
