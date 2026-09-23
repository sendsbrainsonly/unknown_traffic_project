#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss, precision_recall_fscore_support

from common import (
    PROJECT, ROOT, STAGE15R, array_digest, pilot_specs, protocol_rows,
    read_csv, read_json, run_dir, sha256_file, write_csv, write_json,
)


sys.path.insert(0, str(STAGE15R / "vendor"))
import lightgbm as lgb  # noqa: E402


S12_NAMES = [
    "packet_count", "total_bytes", "duration_seconds", "packet_length_mean",
    "packet_length_std", "iat_mean_seconds", "iat_std_seconds",
    "forward_packet_count", "reverse_packet_count", "forward_reverse_packet_ratio",
    "direction_changes", "payload_bytes",
]
LEGACY_COLUMNS = [
    "packet_count", "total_bytes", "duration_seconds", "packet_length_mean",
    "packet_length_std", "iat_mean_seconds", "iat_std_seconds",
    "forward_packets", "reverse_packets", "forward_reverse_ratio",
    "direction_changes", "payload_bytes",
]


def scalar(value) -> float:
    return float("nan") if value in (None, "") else float(value)


def legacy_mapping(dataset: str) -> tuple[dict[str, np.ndarray], dict]:
    if dataset in {"ustc", "iscx_vpn", "iscx_tor"}:
        path = STAGE15R / "feature_cache" / dataset / "flow_statistics.csv"
        rows = read_csv(path)
        return {
            row["flow_uid"]: np.asarray([scalar(row.get(name)) for name in LEGACY_COLUMNS], dtype=np.float32)
            for row in rows
        }, {"path": str(path.resolve()), "sha256": sha256_file(path)}
    if dataset == "vnat":
        cache = PROJECT / "stage14c5_feature_representation_audit" / "raw_feature_cache"
        uids = np.load(cache / "flow_uids.npy", allow_pickle=False)
        stats = np.load(cache / "flow_statistics.npy", mmap_mode="r", allow_pickle=False)
        directions = np.load(cache / "packet_directions.npy", mmap_mode="r", allow_pickle=False)
        masks = np.load(cache / "packet_mask.npy", mmap_mode="r", allow_pickle=False)
        mapping = {}
        for index, uid in enumerate(uids):
            packet_count, total_bytes, duration, length_mean, length_std, iat_mean, iat_std, ratio = map(float, stats[index])
            sequence = directions[index][masks[index].astype(bool)]
            sequence = sequence[sequence != 0]
            changes = float(np.sum(sequence[1:] != sequence[:-1])) if len(sequence) > 1 else 0.0
            mapping[str(uid)] = np.asarray([
                packet_count, total_bytes, duration, length_mean, length_std, iat_mean, iat_std,
                np.nan, np.nan, ratio, changes, np.nan,
            ], dtype=np.float32)
        return mapping, {"path": str(cache.resolve()), "sha256": sha256_file(cache / "cache_audit.json")}
    raise KeyError(dataset)


def matrix(rows: list[dict], mapping: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids = np.asarray([row["sample_id"] for row in rows])
    missing = [uid for uid in ids if uid not in mapping]
    if missing:
        raise RuntimeError(f"missing {len(missing)} S12 features: {missing[:3]}")
    values = np.vstack([mapping[uid] for uid in ids])
    labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
    return values, labels, ids


def main() -> None:
    summaries = []
    for spec in pilot_specs():
        dataset, protocol_id = spec["dataset"], spec["protocol_id"]
        output = run_dir("FULL_FLOW", "S12", dataset, protocol_id)
        if output.exists():
            raise RuntimeError(f"refusing to overwrite S12 parity run: {output}")
        output.mkdir(parents=True)
        classes, train_rows, val_rows = protocol_rows(dataset, protocol_id)
        mapping, source = legacy_mapping(dataset)
        train_x, train_y, train_ids = matrix(train_rows, mapping)
        val_x, val_y, val_ids = matrix(val_rows, mapping)
        del mapping
        historical = STAGE15R / "pilot_runs" / "e3" / dataset / protocol_id
        reference = read_json(historical / "result.json")
        checkpoint = Path(reference["checkpoint_path"])
        if sha256_file(checkpoint) != reference["checkpoint_sha256"]:
            raise RuntimeError(f"historical E3 checkpoint hash mismatch: {checkpoint}")
        booster = lgb.Booster(model_file=str(checkpoint))
        probability = np.asarray(booster.predict(val_x, num_iteration=booster.best_iteration), dtype=np.float64)
        prediction = probability.argmax(axis=1).astype(np.int64)
        precision, recall, f1, support = precision_recall_fscore_support(
            val_y, prediction, labels=np.arange(len(classes)), zero_division=0,
        )
        confusion = confusion_matrix(val_y, prediction, labels=np.arange(len(classes)))
        metrics = {
            "validation_loss": float(log_loss(val_y, probability, labels=np.arange(len(classes)))),
            "validation_accuracy": float(accuracy_score(val_y, prediction)),
            "validation_macro_f1": float(f1.mean()),
            "validation_weighted_f1": float(np.average(f1, weights=support)),
        }
        reference_confusion = np.load(historical / "validation_confusion_matrix.npy", allow_pickle=False)
        metric_match = all(abs(metrics[name] - float(reference[name])) <= (1e-9 if name == "validation_loss" else 1e-12) for name in metrics)
        confusion_match = np.array_equal(confusion, reference_confusion)
        status = "PASS" if metric_match and confusion_match else "FAIL"
        np.save(output / "validation_confusion_matrix.npy", confusion, allow_pickle=False)
        np.savez_compressed(output / "validation_predictions.npz", sample_ids=val_ids, true_labels=val_y, predicted_labels=prediction, probabilities=probability.astype(np.float32))
        per_class = [
            {"scenario": "FULL_FLOW", "feature_set": "S12", "dataset": dataset, "protocol_id": protocol_id,
             "class_name": name, "precision": float(precision[i]), "recall": float(recall[i]),
             "f1": float(f1[i]), "support": int(support[i])}
            for i, name in enumerate(classes)
        ]
        write_csv(output / "per_class_results.csv", per_class)
        importance = booster.feature_importance(importance_type="gain")
        write_csv(output / "feature_importance.csv", [
            {"feature": name, "gain": float(importance[i]), "normalized_gain": float(importance[i] / importance.sum()) if importance.sum() else 0.0}
            for i, name in enumerate(S12_NAMES)
        ])
        history = read_csv(historical / "history.csv")
        write_csv(output / "history.csv", history)
        result = {
            "status": status, "scenario": "FULL_FLOW", "feature_set": "S12", "dataset": dataset,
            "protocol_id": protocol_id, "class_names": classes, "known_train_samples": len(train_y),
            "known_validation_samples": len(val_y), "known_test_samples_used": 0, "unknown_test_samples_used": 0,
            "feature_names": S12_NAMES, "feature_count": len(S12_NAMES),
            "train_missing_values": int(np.isnan(train_x).sum()), "validation_missing_values": int(np.isnan(val_x).sum()),
            "input_hash": array_digest(train_ids, train_x, train_y, val_ids, val_x, val_y),
            **metrics, "best_epoch": int(reference["best_epoch"]), "epochs_completed": int(reference["epochs_completed"]),
            "runtime_seconds": 0.0, "peak_memory_bytes": 0, "model_reused": True,
            "checkpoint_path": str(checkpoint), "checkpoint_sha256": reference["checkpoint_sha256"],
            "lightgbm_version": lgb.__version__, "source_audit": source,
            "parity": {"metrics_match": metric_match, "confusion_exact_match": confusion_match, "reference_result": str((historical / "result.json").resolve())},
        }
        write_json(output / "result.json", result)
        write_json(output / "run_config.json", {"reuse": "Stage15R E3 exact checkpoint", "visible_roles": ["known_train", "known_validation"], "strict_unknown_free": True})
        summaries.append(result)
        if status != "PASS":
            write_json(ROOT / "s12_parity_summary.json", {"status": "FAIL", "runs": summaries})
            raise RuntimeError(f"S12 parity failed for {dataset}/{protocol_id}: metrics={metric_match}, confusion={confusion_match}")
    write_json(ROOT / "s12_parity_summary.json", {"status": "PASS", "runs": summaries})
    lines = ["# S12 / Stage 15R E3 parity audit", "", "Status: `PASS`", "", "All five FULL_FLOW S12 runs reuse the exact Stage 15R E3 LightGBM checkpoint and reconstruct the original feature matrix/order. Validation metrics and confusion matrices match exactly within the preregistered numerical tolerance.", "", "| Dataset | Protocol | Accuracy | Macro-F1 | Weighted-F1 | Metrics | Confusion |", "|---|---|---:|---:|---:|---|---|"]
    for row in summaries:
        lines.append(f"| {row['dataset']} | {row['protocol_id']} | {row['validation_accuracy']:.6f} | {row['validation_macro_f1']:.6f} | {row['validation_weighted_f1']:.6f} | PASS | PASS |")
    lines += ["", "The parity baseline does not use the newly extracted FULL_FLOW behavior cache. Consequently, data repairs, restored VNAT direction fields, or new feature definitions cannot be counted as S12 gain.", ""]
    (ROOT / "feature_parity_audit.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": "PASS", "runs": len(summaries)}, sort_keys=True))


if __name__ == "__main__":
    main()
