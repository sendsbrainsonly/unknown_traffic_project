#!/usr/bin/env python3
"""Run one preregistered LightGBM flow-statistics pilot on Known Train/Val."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss, precision_recall_fscore_support

from common import (
    CONFIG_PATH, PROJECT_ROOT, ROOT, STAGE12_ROOT, STAGE14C5_ROOT,
    STAGE14C_INPUT_ROOT, STAGE3_ROOT, read_csv, read_json, sha256_file,
    write_csv, write_json,
)

sys.path.insert(0, str(ROOT / "vendor"))
import lightgbm as lgb  # noqa: E402


FEATURES = (
    "packet_count", "total_bytes", "duration_seconds", "packet_length_mean",
    "packet_length_std", "iat_mean_seconds", "iat_std_seconds",
    "forward_packets", "reverse_packets", "forward_reverse_ratio",
    "direction_changes", "payload_bytes",
)


def scalar(value) -> float:
    if value in (None, ""):
        return float("nan")
    return float(value)


def selected_ids(dataset: str, protocol_id: str, setting: str):
    if dataset.startswith("iscx_"):
        protocol = read_json(STAGE12_ROOT / "artifacts" / dataset / "protocol" / setting / "protocol.json")
        known = list(protocol["known_classes"])
        local = {name: index for index, name in enumerate(known)}
        manifest = [row for row in read_csv(STAGE12_ROOT / "protocol" / dataset / "split_manifest.csv") if row["setting"] == setting]
        output = []
        for role in ("known_train", "known_validation"):
            rows = [row for row in manifest if row["role"] == role]
            output.append(([row["flow_id_sha256"] for row in rows], np.asarray([local[row["canonical_class"]] for row in rows], dtype=np.int64)))
        return output[0], output[1], known
    if dataset == "vnat":
        base = STAGE14C_INPUT_ROOT / "runs" / protocol_id / "inputs"
        label_map = read_json(base / "label_map.json")
        known = [label_map["local_to_class"][str(i)] for i in range(len(label_map["local_to_class"]))]
        manifest = read_csv(base / "input_manifest.csv")
        output = []
        for role in ("train", "validation"):
            rows = [row for row in manifest if row["split"] == role]
            output.append(([row["flow_uid"] for row in rows], np.asarray([int(row["local_label"]) for row in rows], dtype=np.int64)))
        return output[0], output[1], known
    if dataset == "ustc":
        config = read_json(STAGE3_ROOT / "outputs" / "A-2" / "training_config.json")
        known = list(config["known_classes"])
        class_to_id = read_json(PROJECT_ROOT / "data" / "trafficformer_input" / "compatible_min1" / "label_map.json")["class_to_id"]
        original_to_local = {int(class_to_id[name]): index for index, name in enumerate(known)}
        output = []
        for role in ("train", "val"):
            ids = np.load(PROJECT_ROOT / "outputs" / "stage1" / "modelA" / "embeddings" / f"flow_ids_{role}.npy", allow_pickle=False)
            labels = np.load(PROJECT_ROOT / "outputs" / "stage1" / "modelA" / "embeddings" / f"labels_{role}.npy", allow_pickle=False)
            keep = np.isin(labels, list(original_to_local))
            output.append(([str(value) for value in ids[keep]], np.asarray([original_to_local[int(value)] for value in labels[keep]], dtype=np.int64)))
        return output[0], output[1], known
    raise KeyError(dataset)


def feature_map(dataset: str) -> tuple[dict[str, np.ndarray], dict]:
    if dataset in {"ustc", "iscx_vpn", "iscx_tor"}:
        cache = ROOT / "feature_cache" / dataset
        rows = read_csv(cache / "flow_statistics.csv")
        mapping = {row["flow_uid"]: np.asarray([scalar(row.get(name)) for name in FEATURES], dtype=np.float32) for row in rows}
        return mapping, {"source": str((cache / "flow_statistics.csv").resolve()), "source_sha256": sha256_file(cache / "flow_statistics.csv"), "missing_feature_semantics": "NaN retained for unavailable payload fields"}
    if dataset == "vnat":
        cache = STAGE14C5_ROOT / "raw_feature_cache"
        uids = np.load(cache / "flow_uids.npy", allow_pickle=False)
        stats = np.load(cache / "flow_statistics.npy", mmap_mode="r", allow_pickle=False)
        directions = np.load(cache / "packet_directions.npy", mmap_mode="r", allow_pickle=False)
        masks = np.load(cache / "packet_mask.npy", mmap_mode="r", allow_pickle=False)
        mapping = {}
        for i, uid in enumerate(uids):
            packet_count, total_bytes, duration, length_mean, length_std, iat_mean, iat_std, ratio = map(float, stats[i])
            seq = directions[i][masks[i].astype(bool)]
            seq = seq[seq != 0]
            changes = float(np.sum(seq[1:] != seq[:-1])) if len(seq) > 1 else 0.0
            mapping[str(uid)] = np.asarray([packet_count, total_bytes, duration, length_mean, length_std, iat_mean, iat_std, np.nan, np.nan, ratio, changes, np.nan], dtype=np.float32)
        return mapping, {"source": str(cache.resolve()), "source_sha256": sha256_file(cache / "cache_audit.json"), "missing_feature_semantics": "VNAT full-flow forward/reverse counts and payload bytes unavailable; retained as NaN; direction_changes uses first 8 packets"}
    raise KeyError(dataset)


def matrix(ids: list[str], mapping: dict[str, np.ndarray]) -> np.ndarray:
    missing = [uid for uid in ids if uid not in mapping]
    if missing:
        raise RuntimeError(f"missing {len(missing)} frozen flow features; preview={missing[:3]}")
    return np.vstack([mapping[uid] for uid in ids])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("iscx_vpn", "iscx_tor", "vnat", "ustc"), required=True)
    parser.add_argument("--protocol-id", required=True)
    parser.add_argument("--setting", required=True)
    args = parser.parse_args()
    config = read_json(CONFIG_PATH)
    allowed = {(row["dataset"], row["protocol_id"], row["setting"]) for row in config["pilot_protocols"]}
    if (args.dataset, args.protocol_id, args.setting) not in allowed:
        raise RuntimeError("pilot protocol is not preregistered")
    output = ROOT / "pilot_runs" / "e3" / args.dataset / args.protocol_id
    if output.exists():
        raise RuntimeError(f"refusing to overwrite preserved run: {output}")
    output.mkdir(parents=True)
    started = time.time()
    (train_ids, train_y), (val_ids, val_y), class_names = selected_ids(args.dataset, args.protocol_id, args.setting)
    mapping, source_audit = feature_map(args.dataset)
    train_x, val_x = matrix(train_ids, mapping), matrix(val_ids, mapping)
    del mapping
    if set(np.unique(train_y)) != set(range(len(class_names))) or set(np.unique(val_y)) != set(range(len(class_names))):
        raise RuntimeError("Known class coverage mismatch")
    parameters = dict(config["e3"]["parameters"])
    early_stopping_rounds = int(parameters.pop("early_stopping_rounds"))
    parameters.pop("checkpoint_criterion")
    model = lgb.LGBMClassifier(objective="multiclass", num_class=len(class_names), verbosity=-1, **parameters)
    model.fit(train_x, train_y, eval_set=[(val_x, val_y)], eval_metric="multi_logloss", callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False), lgb.log_evaluation(0)])
    prediction = model.predict(val_x).astype(np.int64)
    probability = model.predict_proba(val_x)
    precision, recall, f1, support = precision_recall_fscore_support(val_y, prediction, labels=np.arange(len(class_names)), zero_division=0)
    confusion = confusion_matrix(val_y, prediction, labels=np.arange(len(class_names)))
    np.save(output / "validation_confusion_matrix.npy", confusion, allow_pickle=False)
    per_class = [{"experiment": "E3", "dataset": args.dataset, "protocol_id": args.protocol_id, "class_name": name, "precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i]), "support": int(support[i])} for i, name in enumerate(class_names)]
    write_csv(output / "per_class_results.csv", per_class)
    history = model.evals_result_["valid_0"]["multi_logloss"]
    write_csv(output / "history.csv", [{"iteration": i + 1, "validation_multiclass_logloss": value, "is_best": int(i + 1 == model.best_iteration_)} for i, value in enumerate(history)])
    model.booster_.save_model(str(output / "model_best.txt"), num_iteration=model.best_iteration_)
    result = {
        "status": "PASS", "experiment": "E3", "dataset": args.dataset, "protocol_id": args.protocol_id,
        "setting": args.setting, "known_train_samples": len(train_y), "known_validation_samples": len(val_y),
        "known_test_samples_used": 0, "unknown_test_samples_used": 0, "class_names": class_names,
        "best_epoch": int(model.best_iteration_), "epochs_completed": len(history), "train_loss": "NOT_RECORDED_BY_FIXED_EVAL_SET",
        "validation_loss": float(log_loss(val_y, probability, labels=np.arange(len(class_names)))),
        "validation_accuracy": float(accuracy_score(val_y, prediction)), "validation_macro_f1": float(f1.mean()),
        "validation_weighted_f1": float(np.average(f1, weights=support)), "runtime_seconds": time.time() - started,
        "peak_gpu_memory_bytes": 0, "checkpoint_path": str((output / "model_best.txt").resolve()),
        "checkpoint_sha256": sha256_file(output / "model_best.txt"), "lightgbm_version": lgb.__version__,
        "features": list(FEATURES), "source_audit": source_audit,
    }
    write_json(output / "result.json", result)
    write_json(output / "run_config.json", {"stage15r_config": config, "invocation": vars(args), "strict_unknown_free": True, "visible_roles": ["known_train", "known_validation"]})
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
