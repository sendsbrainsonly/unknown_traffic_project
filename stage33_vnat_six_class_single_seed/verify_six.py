#!/usr/bin/env python3
"""Independent file-level replay of the frozen VNAT six-class diagnostic."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from run_six import CLASSES, LABEL_MAP, PROJECT, ROOT, prior, rows


def main() -> None:
    run = ROOT / "runs" / "vnat"
    result = run / "known_test_evaluation"
    report = json.loads((result / "results.json").read_text())
    with (result / "sample_predictions.csv").open(newline="", encoding="utf-8") as handle:
        predictions = list(csv.DictReader(handle))
    frozen_rows = rows("known_test")
    if len(predictions) != 1960 or len(frozen_rows) != 1960:
        raise RuntimeError("Known Test row count mismatch")
    if [(r["flow_id"], r["true_application"]) for r in predictions] != frozen_rows:
        raise RuntimeError("prediction/frozen-manifest ID or application mismatch")
    true_names = [LABEL_MAP[fine] for _, fine in frozen_rows]
    if [r["true_class"] for r in predictions] != true_names:
        raise RuntimeError("saved six-class truth does not match fixed mapping")
    logits = np.load(result / "logits.npy", allow_pickle=False)
    if logits.shape != (1960, 6) or not np.isfinite(logits).all():
        raise RuntimeError("invalid saved Test logits")
    truth = np.asarray([CLASSES.index(name) for name in true_names], dtype=np.int64)
    chosen = logits.argmax(1)
    if [r["predicted_class"] for r in predictions] != [CLASSES[int(i)] for i in chosen]:
        raise RuntimeError("saved predictions are not argmax of logits")
    if [int(r["correct"]) for r in predictions] != (truth == chosen).astype(int).tolist():
        raise RuntimeError("saved correctness bits mismatch")
    recomputed = {
        "accuracy": float(accuracy_score(truth, chosen)),
        "macro_f1": float(f1_score(truth, chosen, labels=range(6), average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(truth, chosen, labels=range(6), average="weighted", zero_division=0)),
    }
    for name, value in recomputed.items():
        if not np.isclose(value, report["metrics"][name], rtol=0, atol=1e-12):
            raise RuntimeError(f"independent replay mismatch: {name}")
    matrix = confusion_matrix(truth, chosen, labels=range(6))
    if matrix.tolist() != report["confusion_matrix"]:
        raise RuntimeError("confusion matrix mismatch")
    with (result / "confusion_matrix.csv").open(newline="", encoding="utf-8") as handle:
        table = list(csv.reader(handle))
    if table[0] != ["true/pred", *CLASSES] or [r[0] for r in table[1:]] != list(CLASSES):
        raise RuntimeError("confusion CSV class order mismatch")
    if [[int(v) for v in r[1:]] for r in table[1:]] != matrix.tolist():
        raise RuntimeError("confusion CSV values mismatch")
    for name, digest in report["checkpoint_hashes"].items():
        if prior.sha256(run / name) != digest:
            raise RuntimeError(f"checkpoint hash mismatch: {name}")
    frozen = json.loads((ROOT / "frozen_source_hashes_before.json").read_text())
    if prior.freeze_sources() != frozen:
        raise RuntimeError("Stage32/31 sources changed")
    cache_root = PROJECT / "stage32_three_dataset_coarse_single_seed" / "input_caches" / "vnat" / prior.VNAT_PROTOCOL
    for relative, digest in report["stage32_test_cache_hashes"].items():
        if prior.sha256(cache_root / relative) != digest:
            raise RuntimeError(f"Stage32 Test cache changed: {relative}")
    freeze = PROJECT / "stage14b_vnat_protocol_freeze"
    expected_freeze = json.loads((freeze / "outputs" / "freeze_hashes.json").read_text())
    protected = {
        "vnat_open_set_protocol_sha256": freeze / "vnat_open_set_protocol.json",
        "vnat_split_manifest_sha256": freeze / "vnat_split_manifest.csv",
        "vnat_final_class_audit_sha256": freeze / "vnat_final_class_audit.csv",
        "stage14b_protocol_freeze_sha256": freeze / "stage14b_protocol_freeze.md",
    }
    for key, path in protected.items():
        if prior.sha256(path) != expected_freeze[key]:
            raise RuntimeError(f"Stage14B freeze input changed: {key}")
    summary = {
        "status": "PASS", "test_rows_replayed": len(predictions), "test_labels_from_frozen_manifest": True,
        "unknown_feature_values_loaded": 0, "saved_logit_argmax_parity": True,
        "metrics_recomputed_independently": recomputed, "confusion_matrix_parity": True,
        "selected_checkpoint_hashes_unchanged": True, "stage32_source_and_cache_hashes_unchanged": True,
        "stage14b_freeze_hash": expected_freeze["freeze_hash"], "stage14b_inputs_unchanged": True,
    }
    prior.write_json(ROOT / "completion_verification.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
