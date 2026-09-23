#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss, precision_recall_fscore_support

from common import (
    CONFIG_PATH, STAGE15R, array_digest, cache_dir, feature_names,
    feature_set_names, group_by_feature, pilot_specs, protocol_rows, read_json,
    run_dir, sha256_file, write_csv, write_json,
)


sys.path.insert(0, str(STAGE15R / "vendor"))
import lightgbm as lgb  # noqa: E402


S12_NAMES = [
    "packet_count", "total_bytes", "duration_seconds", "packet_length_mean", "packet_length_std",
    "iat_mean_seconds", "iat_std_seconds", "forward_packet_count", "reverse_packet_count",
    "forward_reverse_packet_ratio", "direction_changes", "payload_bytes",
]
DIRECTIONAL_S12 = {"forward_packet_count", "reverse_packet_count", "forward_reverse_packet_ratio", "direction_changes"}


def assemble(dataset: str, scenario: str, feature_set: str, rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    cache = cache_dir(dataset)
    uids = np.load(cache / "flow_uids.npy", allow_pickle=False).astype(str)
    uid_index = {uid: index for index, uid in enumerate(uids)}
    source = np.load(cache / ("full_flow_features.npy" if scenario == "FULL_FLOW" else "early16_features.npy"), mmap_mode="r", allow_pickle=False)
    names = feature_names()
    columns = {name: index for index, name in enumerate(names)}
    selected_names = feature_set_names(feature_set)
    row_indices = []
    missing = []
    for row in rows:
        index = uid_index.get(row["sample_id"])
        if index is None:
            missing.append(row["sample_id"])
        else:
            row_indices.append(index)
    if missing:
        raise RuntimeError(f"behavior cache misses {len(missing)} sample IDs: {missing[:3]}")
    # Stage 15R E3 trained on float32 matrices. Preserve that dtype so an
    # incremental comparison is not confounded by LightGBM binning precision.
    matrix = np.asarray(source[np.asarray(row_indices)][:, [columns[name] for name in selected_names]], dtype=np.float32)
    if scenario == "FULL_FLOW":
        legacy = np.load(cache / "legacy_s12_features.npy", mmap_mode="r", allow_pickle=False)
        legacy_rows = np.asarray(legacy[np.asarray(row_indices)], dtype=np.float32)
        legacy_col = {name: index for index, name in enumerate(S12_NAMES)}
        selected_col = {name: index for index, name in enumerate(selected_names)}
        preserve_all = feature_set in {"S12", "S-A", "S-AB"}
        for name in S12_NAMES:
            if name in selected_col and (preserve_all or name not in DIRECTIONAL_S12):
                matrix[:, selected_col[name]] = legacy_rows[:, legacy_col[name]]
    elif dataset in {"ustc", "vnat"} and "payload_bytes" in selected_names:
        matrix[:, selected_names.index("payload_bytes")] = np.nan
    labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
    sample_ids = np.asarray([row["sample_id"] for row in rows])
    return matrix, labels, sample_ids, {
        "feature_names": selected_names,
        "missing_values": int(np.isnan(matrix).sum()),
        "missing_ratio": float(np.isnan(matrix).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("ustc", "vnat", "iscx_vpn", "iscx_tor"), required=True)
    parser.add_argument("--protocol-id", required=True)
    parser.add_argument("--scenario", choices=("FULL_FLOW", "EARLY_16"), required=True)
    parser.add_argument("--feature-set", choices=("S-A", "S-AB", "S-ABC", "S-ABCD", "S-Burst", "S12"), required=True)
    args = parser.parse_args()
    config = read_json(CONFIG_PATH)
    allowed_specs = {(row["dataset"], row["protocol_id"]) for row in pilot_specs()}
    if (args.dataset, args.protocol_id) not in allowed_specs:
        raise RuntimeError("protocol is not preregistered")
    allowed_sets = config["observation_scenarios"][args.scenario]
    if args.feature_set not in allowed_sets:
        raise RuntimeError("feature set is not preregistered for this scenario")
    if args.scenario == "FULL_FLOW" and args.feature_set == "S12":
        raise RuntimeError("FULL_FLOW S12 must use the parity/reuse path")
    output = run_dir(args.scenario, args.feature_set, args.dataset, args.protocol_id)
    if output.exists():
        raise RuntimeError(f"refusing to overwrite preserved run: {output}")
    output.mkdir(parents=True)
    started = time.time()
    classes, train_rows, val_rows = protocol_rows(args.dataset, args.protocol_id)
    train_x, train_y, train_ids, train_audit = assemble(args.dataset, args.scenario, args.feature_set, train_rows)
    val_x, val_y, val_ids, val_audit = assemble(args.dataset, args.scenario, args.feature_set, val_rows)
    if set(np.unique(train_y)) != set(range(len(classes))) or set(np.unique(val_y)) != set(range(len(classes))):
        raise RuntimeError("Known class coverage mismatch")
    parameters = dict(config["model"])
    parameters.pop("implementation")
    parameters.pop("objective")
    early_stopping = int(parameters.pop("early_stopping_rounds"))
    parameters.pop("checkpoint_criterion")
    model = lgb.LGBMClassifier(objective="multiclass", num_class=len(classes), verbosity=-1, **parameters)
    model.fit(
        train_x, train_y, eval_set=[(val_x, val_y)], eval_metric="multi_logloss",
        callbacks=[lgb.early_stopping(early_stopping, verbose=False), lgb.log_evaluation(0)],
    )
    probability = np.asarray(model.predict_proba(val_x), dtype=np.float64)
    prediction = probability.argmax(axis=1).astype(np.int64)
    precision, recall, f1, support = precision_recall_fscore_support(val_y, prediction, labels=np.arange(len(classes)), zero_division=0)
    confusion = confusion_matrix(val_y, prediction, labels=np.arange(len(classes)))
    np.save(output / "validation_confusion_matrix.npy", confusion, allow_pickle=False)
    np.savez_compressed(output / "validation_predictions.npz", sample_ids=val_ids, true_labels=val_y, predicted_labels=prediction, probabilities=probability.astype(np.float32))
    write_csv(output / "per_class_results.csv", [
        {"scenario": args.scenario, "feature_set": args.feature_set, "dataset": args.dataset, "protocol_id": args.protocol_id,
         "class_name": name, "precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i]), "support": int(support[i])}
        for i, name in enumerate(classes)
    ])
    history = model.evals_result_["valid_0"]["multi_logloss"]
    write_csv(output / "history.csv", [
        {"iteration": index + 1, "validation_multiclass_logloss": float(value), "is_best": int(index + 1 == model.best_iteration_)}
        for index, value in enumerate(history)
    ])
    checkpoint = output / "model_best.txt"
    model.booster_.save_model(str(checkpoint), num_iteration=model.best_iteration_)
    gain = model.booster_.feature_importance(importance_type="gain")
    groups = group_by_feature()
    write_csv(output / "feature_importance.csv", [
        {"feature": name, "feature_group": groups[name], "gain": float(gain[i]), "normalized_gain": float(gain[i] / gain.sum()) if gain.sum() else 0.0}
        for i, name in enumerate(train_audit["feature_names"])
    ])
    runtime = time.time() - started
    peak_memory = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    result = {
        "status": "PASS", "scenario": args.scenario, "feature_set": args.feature_set,
        "dataset": args.dataset, "protocol_id": args.protocol_id, "class_names": classes,
        "known_train_samples": len(train_y), "known_validation_samples": len(val_y),
        "known_test_samples_used": 0, "unknown_test_samples_used": 0,
        "feature_names": train_audit["feature_names"], "feature_count": len(train_audit["feature_names"]),
        "train_missing_values": train_audit["missing_values"], "train_missing_ratio": train_audit["missing_ratio"],
        "validation_missing_values": val_audit["missing_values"], "validation_missing_ratio": val_audit["missing_ratio"],
        "input_hash": array_digest(train_ids, train_x, train_y, val_ids, val_x, val_y),
        "best_epoch": int(model.best_iteration_), "epochs_completed": len(history),
        "validation_loss": float(log_loss(val_y, probability, labels=np.arange(len(classes)))),
        "validation_accuracy": float(accuracy_score(val_y, prediction)),
        "validation_macro_f1": float(f1.mean()), "validation_weighted_f1": float(np.average(f1, weights=support)),
        "runtime_seconds": runtime, "peak_memory_bytes": peak_memory, "model_reused": False,
        "checkpoint_path": str(checkpoint.resolve()), "checkpoint_sha256": sha256_file(checkpoint),
        "lightgbm_version": lgb.__version__, "cache_audit": read_json(cache_dir(args.dataset) / "cache_audit.json"),
    }
    write_json(output / "result.json", result)
    write_json(output / "run_config.json", {"stage15f1b_config": config, "invocation": vars(args), "visible_roles": ["known_train", "known_validation"], "strict_unknown_free": True})
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
