#!/usr/bin/env python3
"""Independently verify Stage 10A outputs and frozen upstream integrity."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
AUDIT = PROJECT / "stage10a_opendetect_protocol_audit"
SUMMARY = AUDIT / "outputs/summary"
OPENDETECT_PROJECT = PROJECT.parent / "Open-Detect"
OFFICIAL = OPENDETECT_PROJECT / "code"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def verify_hash_file(path: Path) -> int:
    count = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        expected, relative = line.split(maxsplit=1)
        candidate = Path(relative)
        target = candidate if candidate.is_absolute() else (path.parent / candidate).resolve()
        require(target.is_file(), f"missing hash target: {target}")
        require(sha256(target) == expected, f"hash mismatch: {target}")
        count += 1
    return count


required_csv = {
    "outputs/paper/paper_protocol.csv": {"item", "paper_value", "page", "section", "equation/table", "evidence", "confidence"},
    "outputs/official_code/official_implementation_map.csv": {"function", "file", "line_start", "line_end", "behavior", "paper_consistent", "notes"},
    "outputs/ustc_reproduction/ustc_reproduction_protocol.csv": {"item", "value", "evidence"},
    "outputs/ustc_reproduction/ustc_open_set_reproduction.csv": {
        "setting", "fold", "seed", "reproduction_status", "training_protocol",
        "known_classes", "unknown_classes", "checkpoint", "checkpoint_sha256",
        "best_validation_accuracy", "validation_macro_f1", "native_score",
        "native_threshold", "threshold_method", "open_accuracy_binary",
        "open_f1_binary_unknown_positive", "open_auroc_unknown_positive",
        "paper_exact_fold",
    },
    "outputs/summary/paper_vs_ustc.csv": {"item", "paper", "ustc_reproduction", "alignment", "reason"},
    "outputs/cipherspectrum/native_implementation_audit.csv": {"item", "official", "stage9_native", "alignment", "evidence"},
    "outputs/summary/preprocessing_comparison.csv": {"aspect", "paper", "official_code", "ustc_reproduction", "cipherspectrum", "alignment"},
    "outputs/summary/result_definition_alignment.csv": {"dataset", "known classes", "unknown classes", "classification metric", "classification value", "open-set metric", "F1 definition", "split type"},
    "outputs/summary/failure_decomposition.csv": {"factor", "evidence_level", "evidence"},
    "outputs/summary/latent_mode_audit.csv": {"layer", "train_latent_mode", "inference_classification_latent_mode", "inference_unknown_detection_latent_mode", "evidence"},
    "outputs/summary/threshold_comparison.csv": {"layer", "scope", "fit_data", "method"},
    "outputs/summary/leakage_risk_audit.csv": {"feature_or_shortcut", "paper", "official_code", "ustc_reproduction", "cipherspectrum"},
    "outputs/summary/same_encoder_detector_comparison.csv": {"setting", "method", "same_stage7_encoder", "same_mu_x", "known_macro_f1", "UFAR", "AUROC"},
    "outputs/cipherspectrum/stage7_encoder_audit.csv": {"setting", "known_classes", "train_n", "val_n", "known_test_n", "checkpoint_sha256", "val_accuracy", "val_macro_f1", "architecture_alignment"},
}

csv_counts: dict[str, int] = {}
for relative, required_fields in required_csv.items():
    path = AUDIT / relative
    require(path.is_file(), f"missing required CSV: {relative}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = set(reader.fieldnames or [])
    require(required_fields <= fields, f"schema mismatch in {relative}: {sorted(required_fields - fields)}")
    require(rows, f"empty CSV: {relative}")
    csv_counts[relative] = len(rows)

for relative in (
    "README.md",
    "AUDIT_REPORT.md",
    "sources/open_detect_paper.txt",
    "sources/source_manifest.json",
    "sources/source_manifest.md",
    "outputs/paper/metric_definition_audit.md",
    "outputs/official_code/prototype_definition.md",
):
    path = AUDIT / relative
    require(path.is_file() and path.stat().st_size > 0, f"missing/empty required file: {relative}")

report = (AUDIT / "AUDIT_REPORT.md").read_text(encoding="utf-8")
for heading in (
    "Executive Summary", "Paper Protocol", "Official Code", "USTC Reproduction",
    "CipherSpectrum Protocol", "Latent Mode", "Prototype Definition", "Native Score",
    "Threshold Calibration", "Metric Definitions", "Data Split", "Preprocessing",
    "Same-Encoder Detector Comparison", "Failure Decomposition", "What Can Be Claimed",
    "What Cannot Be Claimed", "Final Gate",
):
    require(f"## {heading}" in report, f"missing report heading: {heading}")
require("Gate B — HIGH-PRIORITY PROTOCOL MISMATCH" in report, "final Gate B not recorded")
require("EXTERNAL_RESULT_NOT_YET_ATTRIBUTABLE" in report, "attribution downgrade missing")
require("sibling method project" in report, "sibling Open-Detect project boundary missing")
require("zero paper-exact reproductions" in report, "sibling reproduction exactness boundary missing")
require("old nested 98.6158%/98.7371%" in report, "legacy nested-result correction missing")

with (AUDIT / "outputs/ustc_reproduction/ustc_open_set_reproduction.csv").open(
    newline="", encoding="utf-8-sig"
) as handle:
    ustc_rows = list(csv.DictReader(handle))
require(len(ustc_rows) == 15, f"expected 15 sibling USTC v6 runs, got {len(ustc_rows)}")
require(
    {row["setting"] for row in ustc_rows} == {"A-1", "A-2", "A-3"},
    "sibling USTC scenarios are incomplete",
)
for setting in ("A-1", "A-2", "A-3"):
    rows = [row for row in ustc_rows if row["setting"] == setting]
    require(len(rows) == 5, f"{setting} does not contain five local repeats")
    require({int(row["seed"]) for row in rows} == set(range(2022, 2027)), f"{setting} seeds differ")
for row in ustc_rows:
    checkpoint = Path(row["checkpoint"])
    require(checkpoint.is_file(), f"missing sibling checkpoint: {checkpoint}")
    require(sha256(checkpoint) == row["checkpoint_sha256"], f"sibling checkpoint hash mismatch: {checkpoint}")
    require(row["reproduction_status"] == "LOCAL_APPROXIMATION", "paper-exact claim found")
    require(row["training_protocol"] == "corrected-paper", "unexpected sibling training protocol")
    require(row["validation_macro_f1"] == "NOT_RECORDED", "invented validation Macro-F1 found")
    require(row["paper_exact_fold"].lower() == "false", "author-exact fold claim found")

source_manifest = json.loads((AUDIT / "sources/source_manifest.json").read_text(encoding="utf-8"))
require(
    source_manifest["opendetect_project_relation"] == "sibling project under Projects/; read-only source",
    "wrong Open-Detect project relationship",
)
require(
    source_manifest["source_boundaries"]["paper_level_exact_reproduction_count"] == 0,
    "sibling final matrix no longer reports zero exact reproductions",
)
for item in source_manifest["files"] + source_manifest["v6_checkpoints"]:
    source = Path(item["path"])
    require(source.is_file(), f"missing source-manifest target: {source}")
    require(source.stat().st_size == item["size_bytes"], f"source size mismatch: {source}")
    require(sha256(source) == item["sha256"], f"source hash mismatch: {source}")

official_source_diff = subprocess.run(
    ["git", "-C", str(OFFICIAL), "diff", "--name-only", "--", "*.py"],
    check=True,
    text=True,
    capture_output=True,
).stdout.splitlines()
require(not official_source_diff, f"tracked official Python source changed: {official_source_diff}")

# Sentinel hashes guard against verifier-side mutation.  Internal stage
# verifiers additionally replay the larger checkpoint/bundle hash sets.
sentinels = [
    PROJECT / "stage6_cipherspectrum_protocol/outputs/protocol/protocol_hashes.sha256",
    PROJECT / "stage6_cipherspectrum_protocol/outputs/corpus/canonical120k.sha256",
    PROJECT / "stage8a_cipherspectrum_known_density/manifest.json",
    PROJECT / "stage8b_cipherspectrum_dgsbv2/manifest.json",
    PROJECT / "stage9_cipherspectrum_final_test/outputs/summary/stage9_final_hashes.sha256",
    PROJECT / "stage9_cipherspectrum_final_test/outputs/summary/stage9_final_manifest.json",
]
for setting in ("low", "medium", "high"):
    run = next((PROJECT / "stage7_cipherspectrum_known_training/runs").glob(f"formal-{setting}-*"))
    sentinels.extend([
        run / "training_checkpoint_selection.json",
        run / "manifest.json",
        PROJECT / f"stage8a_cipherspectrum_known_density/artifacts/{setting}/bundle_manifest.json",
        PROJECT / f"stage8a_cipherspectrum_known_density/artifacts/{setting}/native_threshold.json",
    ])
before = {str(path.relative_to(PROJECT)): sha256(path) for path in sentinels}

stage6_hashes = {
    "protocol_hashes_verified": verify_hash_file(PROJECT / "stage6_cipherspectrum_protocol/outputs/protocol/protocol_hashes.sha256"),
    "corpus_hashes_verified": verify_hash_file(PROJECT / "stage6_cipherspectrum_protocol/outputs/corpus/canonical120k.sha256"),
}

verifiers = [
    ("stage7", PROJECT / "stage7_cipherspectrum_known_training/scripts/verify_stage7.py"),
    ("stage8a", PROJECT / "stage8a_cipherspectrum_known_density/scripts/verify_stage8a.py"),
    ("stage8b", PROJECT / "stage8b_cipherspectrum_dgsbv2/scripts/verify_stage8b.py"),
    ("stage9", PROJECT / "stage9_cipherspectrum_final_test/scripts/verify_stage9.py"),
]
env = dict(os.environ)
env["PYTHONDONTWRITEBYTECODE"] = "1"
verifier_results = []
for name, script in verifiers:
    result = subprocess.run(
        [sys.executable, "-B", str(script)],
        cwd=PROJECT,
        env=env,
        text=True,
        capture_output=True,
    )
    verifier_results.append({
        "stage": name,
        "command": f"{sys.executable} -B {script.relative_to(PROJECT)}",
        "returncode": result.returncode,
        "status": "PASS" if result.returncode == 0 else "FAIL",
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    })
    require(result.returncode == 0, f"{name} verifier failed: {result.stderr or result.stdout}")

after = {str(path.relative_to(PROJECT)): sha256(path) for path in sentinels}
require(before == after, "a frozen sentinel changed during verification")

integrity = {
    "status": "PASS",
    "audit_mode": "read-only static audit; no training/inference/refit/test rerun",
    "stage6": {"status": "PASS", **stage6_hashes},
    "stage7_to_stage9_verifiers": verifier_results,
    "opendetect_sibling": {
        "status": "PASS",
        "relationship": "sibling read-only source",
        "v6_runs_verified": len(ustc_rows),
        "checkpoint_hashes_verified": len(ustc_rows),
        "paper_exact_reproduction_count": 0,
        "tracked_official_python_source_diff": official_source_diff,
        "note": "nested official checkout has pre-existing tracked pyc changes; no tracked Python source diff",
    },
    "frozen_sentinel_hashes_before": before,
    "frozen_sentinel_hashes_after": after,
    "frozen_sentinels_unchanged": True,
    "frozen_experiment_modified": False,
    "opendetect_source_modified_by_audit": False,
}
SUMMARY.mkdir(parents=True, exist_ok=True)
(SUMMARY / "upstream_integrity.json").write_text(json.dumps(integrity, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

validation = {
    "status": "PASS",
    "csv_files_verified": csv_counts,
    "required_documents_verified": True,
    "required_report_headings_verified": True,
    "final_gate": "B",
    "claim_status": "EXTERNAL_RESULT_NOT_YET_ATTRIBUTABLE",
    "upstream_integrity": "PASS",
    "frozen_experiment_modified": False,
    "opendetect_sibling_integrity": "PASS",
    "opendetect_source_modified_by_audit": False,
}
(SUMMARY / "audit_validation.json").write_text(json.dumps(validation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps(validation, indent=2, ensure_ascii=False))
