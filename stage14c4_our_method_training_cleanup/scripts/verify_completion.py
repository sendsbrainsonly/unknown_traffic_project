#!/usr/bin/env python3
"""Verify Stage 14C-4 artifacts against the frozen diagnostic contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "stage14c4_our_method_training_cleanup"
EXPECTED_FREEZE = "c904daef04d2c8e79469191121325c0a6ceeaeb3e595ef9c4c7d7a62c980ae2f"
RUN_IDS = (
    "medium_seed2025_protocol",
    "medium_seed2026_protocol",
    "medium_seed2025_fixed2022",
    "medium_seed2026_fixed2022",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    failures: list[str] = []
    results: list[dict] = []

    for run_id in RUN_IDS:
        result_path = OUT / "runs" / run_id / "result.json"
        if not result_path.is_file():
            failures.append(f"missing result: {run_id}")
            continue
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        results.append(payload)
        if payload.get("status") != "success":
            failures.append(f"non-success result: {run_id}")
        if payload.get("epochs_completed") != 100 or payload.get("epochs_configured") != 100:
            failures.append(f"not 100/100 epochs: {run_id}")
        if payload.get("nan_or_crash") is not False:
            failures.append(f"NaN/crash flag: {run_id}")
        if payload.get("known_test_samples_used") != 0:
            failures.append(f"Known Test used: {run_id}")
        if payload.get("unknown_test_samples_used") != 0:
            failures.append(f"Unknown Test used: {run_id}")
        if payload.get("des_executed") is not False:
            failures.append(f"DES executed: {run_id}")
        if payload.get("freeze_hash") != EXPECTED_FREEZE:
            failures.append(f"freeze hash mismatch: {run_id}")

        checkpoint = Path(payload["checkpoint_path"])
        if not checkpoint.is_file() or sha256(checkpoint) != payload.get("checkpoint_sha256"):
            failures.append(f"checkpoint hash mismatch: {run_id}")

        resets = payload.get("prototype_reset_audits", [])
        if [item.get("epoch") for item in resets] != [51, 81]:
            failures.append(f"prototype reset epochs mismatch: {run_id}")
        for item in resets:
            if not all(
                item.get(key) is True
                for key in (
                    "prototype_optimizer_link_preserved",
                    "prototype_optimizer_state_cleared",
                    "prototype_parameter_id_preserved",
                )
            ):
                failures.append(f"prototype reset integrity failed: {run_id}")

    dynamics = pd.read_csv(OUT / "training_dynamics.csv")
    deterministic = pd.read_csv(OUT / "deterministic_split_metrics.csv")
    comparison = pd.read_csv(OUT / "cleanup_comparison.csv")
    summary = json.loads((OUT / "aggregate_summary.json").read_text(encoding="utf-8"))
    optimizer_audit = json.loads(
        (OUT / "prototype_optimizer_link_audit.json").read_text(encoding="utf-8")
    )
    manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))

    if len(results) != 4 or len(dynamics) != 400:
        failures.append("expected four results and 400 epoch rows")
    dynamics_keys = {
        "medium_seed2025_protocol": ("medium_seed2025", "protocol"),
        "medium_seed2026_protocol": ("medium_seed2026", "protocol"),
        "medium_seed2025_fixed2022": ("medium_seed2025", "fixed2022"),
        "medium_seed2026_fixed2022": ("medium_seed2026", "fixed2022"),
    }
    for run_id, (protocol_id, seed_mode) in dynamics_keys.items():
        rows = dynamics[
            (dynamics["protocol_id"] == protocol_id)
            & (dynamics["seed_mode"] == seed_mode)
        ]
        if len(rows) != 100 or rows["epoch"].tolist() != list(range(1, 101)):
            failures.append(f"epoch metrics incomplete: {run_id}")
    if len(deterministic) != 12:
        failures.append("expected 12 deterministic Known split rows")
    if deterministic[["known_test_samples_used", "unknown_test_samples_used"]].to_numpy().sum() != 0:
        failures.append("deterministic evaluator used Test/Unknown")
    if comparison["des_executed"].astype(bool).any():
        failures.append("comparison records DES execution")
    if summary.get("known_test_samples_used") != 0 or summary.get("unknown_test_samples_used") != 0:
        failures.append("aggregate records Test/Unknown use")
    if summary.get("des_executed") is not False:
        failures.append("aggregate records DES execution")
    if summary.get("conclusion") != "NEEDS_FURTHER_DIAGNOSIS":
        failures.append("unexpected conclusion")
    if manifest.get("status") != "partial":
        failures.append("manifest terminal status is not partial")
    if optimizer_audit.get("parameter_replacement_links_new_parameter") is not False:
        failures.append("old prototype-replacement defect not reproduced")
    if optimizer_audit.get("in_place_reset_preserves_parameter_id") is not True:
        failures.append("in-place prototype reset identity check failed")
    if optimizer_audit.get("in_place_reset_preserves_optimizer_link") is not True:
        failures.append("in-place prototype optimizer-link check failed")

    report = {
        "verification": "PASS" if not failures else "FAIL",
        "run_results": len(results),
        "epoch_rows": len(dynamics),
        "deterministic_rows": len(deterministic),
        "checkpoints_hash_verified": len(results),
        "known_test_samples_used": 0,
        "unknown_test_samples_used": 0,
        "des_executed": False,
        "freeze_hash": EXPECTED_FREEZE,
        "conclusion": summary.get("conclusion"),
        "failures": failures,
    }
    (OUT / "completion_verification.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
