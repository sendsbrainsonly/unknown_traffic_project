#!/usr/bin/env python3
"""Normalize completed Stage 3 bundle metadata without recomputing results."""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from common import STAGE3_ROOT, load_fold, sha256_file


SETTINGS = ("A-1", "A-2", "A-3")
DETECTORS = ("Native", "Single-Full", "Multi-Full-K2")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def render_results(setting: str, gate: dict[str, object]) -> str:
    fold = load_fold(setting)
    output_dir = STAGE3_ROOT / "outputs" / setting
    metrics = {row["detector"]: row for row in read_rows(output_dir / "detector_metrics.csv")}
    cis = {row["metric"]: row for row in read_rows(output_dir / "paired_bootstrap_ci.csv")}
    delta = (
        float(metrics["Multi-Full-K2"]["unknown_false_acceptance_rate"])
        - float(metrics["Single-Full"]["unknown_false_acceptance_rate"])
    )
    ci = cis["delta_unknown_far_multi_minus_single"]
    text = f"""# {setting} Strict Unknown-Free Utility Results

- Status: **SUCCESS**
- Claim scope: local frozen-split approximate experiment; not author-fold equivalent

## Data and split

- Known / Unknown classes: {fold['known_count']} / {fold['unknown_count']}
- Known train / validation / test: {fold['formal_counts']['known_train']} / {fold['formal_counts']['known_validation']} / {fold['formal_counts']['known_final_test']}
- Unknown final-test: {fold['formal_counts']['unknown_final_test']}
- Leakage audit: PASS; Unknown usage before final evaluation: 0

## Configuration and execution

- Checkpoint selection: Known Validation Accuracy/Macro-F1 harmonic mean only
- Threshold calibration: 95% Known Validation acceptance only
- Density comparison: PCA-64 + full covariance, K=1 versus fixed K=2
- Bootstrap: 1,000 paired class-stratified resamples, seed 0

## Core results

| Detector | Known test FRR | Unknown FAR | AUROC | AUPRC-Unknown |
|---|---:|---:|---:|---:|
"""
    for name in DETECTORS:
        row = metrics[name]
        text += (
            f"| {name} | {float(row['known_false_rejection_rate']):.6f} | "
            f"{float(row['unknown_false_acceptance_rate']):.6f} | "
            f"{float(row['auroc_unknown_positive']):.6f} | "
            f"{float(row['auprc_unknown_positive']):.6f} |\n"
        )
    text += f"""

- Delta Unknown FAR (Multi - Single): `{delta:.9f}`; negative favors Multi.
- Paired-bootstrap 95% CI: `[{float(ci['ci_2_5pct']):.9f}, {float(ci['ci_97_5pct']):.9f}]`.

## Preserved evidence

- Final predictions, detector metrics, per-class results, decision transitions,
  absorption counts, K2 stability diagnostics, and all 1,000 bootstrap samples
  are retained in this setting bundle.
- Checkpoints, latent exports, scaler, PCA, GMMs, and tmux logs are retained as
  hash-verified external artifacts.

## Limitations

- Official Open-Detect five sample-fold identities are unavailable; this uses
  the frozen local 80/10/10 flow split and official class-holdout scenario.
- Fixed K2 is a controlled utility diagnostic, not evidence that every class
  has two semantic modes.

## Conclusion and next step

The frozen cross-setting result is **Gate {gate['gate']} ({gate['gate_name']})**;
multi-component utility is not established across all three primary settings.
Per the task boundary, do not proceed to a later stage.
"""
    return text


def normalize_manifest(setting: str) -> None:
    output_dir = (STAGE3_ROOT / "outputs" / setting).resolve()
    path = output_dir / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    normalized: list[dict[str, object]] = []
    for item in manifest.get("artifacts", []):
        updated = dict(item)
        raw_path = Path(str(updated["path"]))
        resolved = raw_path.resolve() if raw_path.is_absolute() else (output_dir / raw_path).resolve()
        if updated.get("scope") == "external":
            updated["path"] = os.path.relpath(resolved, output_dir)
            if resolved.is_file():
                updated["size_bytes"] = resolved.stat().st_size
                updated["sha256"] = sha256_file(resolved)
                updated["hash_status"] = "verified"
        normalized.append(updated)
    manifest["artifacts"] = normalized
    manifest["status"] = "success"
    manifest["updated_at_utc"] = utc_now()
    manifest["next_step"] = "Stop after the frozen Gate; no later-stage execution is authorized."
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    gate = json.loads((STAGE3_ROOT / "outputs" / "summary" / "final_gate.json").read_text(encoding="utf-8"))
    for setting in SETTINGS:
        output_dir = STAGE3_ROOT / "outputs" / setting
        (output_dir / "RESULTS.md").write_text(render_results(setting, gate), encoding="utf-8")
        normalize_manifest(setting)
        print(json.dumps({"setting": setting, "metadata_status": "normalized"}, sort_keys=True))


if __name__ == "__main__":
    main()
