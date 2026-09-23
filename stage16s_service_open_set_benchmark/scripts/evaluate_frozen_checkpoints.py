#!/usr/bin/env python3
"""Canonical inference for all already-trained Stage 16S checkpoints.

No training or parameter update occurs. Results go to a separate tree so the
17 original successes and one preserved post-training failure stay unchanged.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from stage16s_common import (
    CANONICAL, CONFIG, METHODS, OUT, RUNS, SEEDS, SERVICES, known_classes,
    load_role, protocol_id, protocol_rows, read_json, sha256_file, write_csv,
    write_json,
)
from train_evaluate_run import (
    TrafficImages, closed_metrics, detection_metrics, extract, import_module,
    import_open_detect,
)


def evaluate(selected_protocol: str, seed: int) -> dict:
    source = RUNS / selected_protocol / f"seed{seed}"
    target = CANONICAL / selected_protocol / f"seed{seed}"
    if (target / "SUCCESS").is_file():
        return read_json(target / "result.json")
    if target.exists():
        raise FileExistsError(f"preserved partial canonical evaluation: {target}")
    target.mkdir(parents=True)

    checkpoint_path = source / "model_best.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    config = read_json(CONFIG)
    training = config["training"]
    roles = {
        role: load_role(selected_protocol, role)
        for role in ("known_train", "known_validation", "known_test", "unknown_test")
    }
    classes = known_classes(selected_protocol)
    unknown_service = next(row["unknown_service"] for row in protocol_rows(selected_protocol))
    source_config = read_json(source / "config.json")

    CorrectedOpenDetectNet, _, run_epoch, _ = import_open_detect()
    device = torch.device("cuda:0")
    model = CorrectedOpenDetectNet(
        training["architecture"], int(training["channels"]),
        int(training["latent_dim"]), len(classes), 1, 1,
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    started = time.time()

    stable_train_loader = DataLoader(
        TrafficImages(roles["known_train"]["data"], roles["known_train"]["labels"], False),
        batch_size=int(training["eval_batch_size"]), shuffle=False,
        num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        TrafficImages(roles["known_validation"]["data"], roles["known_validation"]["labels"], False),
        batch_size=int(training["eval_batch_size"]), shuffle=False,
        num_workers=0, pin_memory=True,
    )
    stable_train = run_epoch(model, stable_train_loader, device, float(training["lambda"]), None)
    stable_val = run_epoch(model, val_loader, device, float(training["lambda"]), None)
    latent = {
        role: extract(model, value, device, int(training["eval_batch_size"]))
        for role, value in roles.items()
    }
    native_val_metrics = closed_metrics(
        roles["known_validation"]["labels"], latent["known_validation"]["native_prediction"]
    )
    recorded_accuracy = float(checkpoint["validation_accuracy"])
    if abs(native_val_metrics["accuracy"] - recorded_accuracy) > 1e-12:
        raise RuntimeError(
            f"canonical checkpoint parity failed: {native_val_metrics['accuracy']} != {recorded_accuracy}"
        )

    stage14d = import_module(
        "stage16s_canonical_stage14d_common",
        OUT.parent / "stage14d_vnat_frozen_open_set_evaluation" / "scripts" / "stage14d_common.py",
    )
    stage15b = import_module(
        "stage16s_canonical_stage15b",
        OUT.parent / "stage15b_known_only_hybrid_detector" / "scripts" / "run_stage15b.py",
    )
    centroids = stage14d.empirical_centroids(
        latent["known_train"]["mu"], roles["known_train"]["labels"]
    )
    des0, centroid_predictions, local = {}, {}, {}
    for role in ("known_validation", "known_test", "unknown_test"):
        des0[role], centroid_predictions[role] = stage14d.centroid_scores(latent[role]["mu"], centroids)
        local[role] = stage14d.local_knn10_scores(
            latent[role]["mu"], centroid_predictions[role],
            latent["known_train"]["mu"], roles["known_train"]["labels"],
        )
    epsilon = float(config["des_v1"]["epsilon"])
    global_parameters = stage14d.robust_parameters(des0["known_validation"], epsilon)
    local_parameters = stage14d.robust_parameters(local["known_validation"], epsilon)
    des1 = {
        role: 0.5 * stage14d.robust_normalize(des0[role], global_parameters)
        + 0.5 * stage14d.robust_normalize(local[role], local_parameters)
        for role in ("known_validation", "known_test", "unknown_test")
    }
    od = {role: latent[role]["od_score"] for role in ("known_validation", "known_test", "unknown_test")}
    od_percentile = {
        role: stage15b.empirical_percentile(od["known_validation"], od[role]) for role in od
    }
    des0_percentile = {
        role: stage15b.empirical_percentile(des0["known_validation"], des0[role]) for role in des0
    }
    h1 = {role: np.maximum(od_percentile[role], des0_percentile[role]) for role in od}
    method_scores = {"OD-Native": od, "DES-v0": des0, "DES-v1": des1, "H1": h1}

    known_closed = closed_metrics(roles["known_test"]["labels"], latent["known_test"]["native_prediction"])
    method_metrics, thresholds = {}, {}
    for method, scores in method_scores.items():
        metric = detection_metrics(
            scores["known_validation"], scores["known_test"], scores["unknown_test"],
            roles["known_test"]["labels"], latent["known_test"]["native_prediction"],
            latent["unknown_test"]["native_prediction"], len(classes), stage14d.threshold_p95,
        )
        method_metrics[method] = {
            **metric, **{f"known_closed_{key}": value for key, value in known_closed.items()}
        }
        thresholds[method] = {
            "threshold": metric["threshold"], "source": "Known Validation P95 only",
            "numpy_method": "higher", "unknown_if": "score >= threshold",
            "validation_samples": len(scores["known_validation"]),
            "validation_rejection_rate": metric["validation_frr"],
            "score_direction": "larger_is_more_unknown",
        }

    np.savez_compressed(
        target / "latent_outputs.npz",
        **{f"{role}_{name}": values[name] for role, values in latent.items()
           for name in ("mu", "logvar", "native_prediction")},
        **{f"{role}_labels": roles[role]["labels"] for role in roles},
        **{f"{role}_flow_ids": np.asarray([row["flow_id"] for row in roles[role]["metadata"]]).astype(str)
           for role in roles},
    )
    np.savez_compressed(
        target / "score_arrays.npz",
        **{f"{role}_{method.lower().replace('-', '_')}": scores[role]
           for method, scores in method_scores.items() for role in scores},
        **{f"threshold_{method.lower().replace('-', '_')}": np.asarray(thresholds[method]["threshold"])
           for method in METHODS},
        validation_A_OD=od_percentile["known_validation"],
        validation_A_DES0=des0_percentile["known_validation"],
        known_test_A_OD=od_percentile["known_test"],
        known_test_A_DES0=des0_percentile["known_test"],
        unknown_test_A_OD=od_percentile["unknown_test"],
        unknown_test_A_DES0=des0_percentile["unknown_test"],
    )

    score_rows = []
    for role in ("known_validation", "known_test", "unknown_test"):
        for index, meta in enumerate(roles[role]["metadata"]):
            row = {
                "protocol_id": selected_protocol, "unknown_service": unknown_service,
                "seed": seed, "role": role, "flow_id": meta["flow_id"],
                "true_service": meta["service_label"],
                "application_label": meta["application_label"], "capture_id": meta["capture_id"],
                "local_true_label": int(roles[role]["labels"][index]),
                "native_predicted_index": int(latent[role]["native_prediction"][index]),
                "native_predicted_service": classes[int(latent[role]["native_prediction"][index])],
            }
            for method in METHODS:
                key = method.lower().replace("-", "_")
                score = float(method_scores[method][role][index])
                row[f"{key}_score"] = score
                row[f"{key}_rejected"] = int(score >= thresholds[method]["threshold"])
            score_rows.append(row)
    write_csv(target / "sample_scores.csv", score_rows)
    write_json(target / "thresholds.json", {
        "methods": thresholds, "des_v1_global": global_parameters, "des_v1_local": local_parameters,
        "h1_percentile_fit": "Known Validation only; searchsorted side=right",
        "unknown_calibration_samples": 0, "test_calibration_samples": 0,
    })
    write_json(target / "metrics.json", {
        "protocol_id": selected_protocol, "unknown_service": unknown_service, "seed": seed,
        "known_services": classes, "methods": method_metrics,
        "known_validation_closed": native_val_metrics, "known_test_closed": known_closed,
    })

    with (source / "training_log.csv").open(encoding="utf-8", newline="") as handle:
        training_rows = list(csv.DictReader(handle))
    source_result = read_json(source / "result.json") if (source / "result.json").is_file() else {}
    checkpoint_sha = sha256_file(checkpoint_path)
    source_failure = read_json(source / "FAILURE.json") if (source / "FAILURE.json").is_file() else None
    result = {
        "status": "SUCCESS", "evaluation_status": "CANONICAL_FROZEN_CHECKPOINT_INFERENCE",
        "protocol_id": selected_protocol, "unknown_service": unknown_service, "seed": seed,
        "known_services": classes, "role_counts": source_config["role_counts"],
        "train_counts": source_config["train_counts"], "best_epoch": int(checkpoint["epoch"]),
        "completed_epochs": len(training_rows),
        "stopped_early": len(training_rows) < int(training["epochs"]),
        "train_loss": stable_train["total"], "validation_loss": stable_val["total"],
        "validation_accuracy": native_val_metrics["accuracy"],
        "validation_macro_f1": native_val_metrics["macro_f1"],
        "validation_weighted_f1": native_val_metrics["weighted_f1"],
        "known_test_closed": known_closed, "methods": method_metrics,
        "runtime_seconds": source_result.get("runtime_seconds"),
        "peak_gpu_memory_bytes": source_result.get("peak_gpu_memory_bytes"),
        "canonical_inference_seconds": time.time() - started,
        "checkpoint_path": str(checkpoint_path.resolve()), "checkpoint_sha256": checkpoint_sha,
        "score_file_sha256": sha256_file(target / "sample_scores.csv"),
        "source_run_status": "FAILED_POST_TRAINING_PARITY_ASSERTION" if source_failure else "SUCCESS",
        "source_failure_preserved": bool(source_failure),
        "evaluation_preprocessing": "PIL grayscale -> torchvision ToTensor; exact checkpoint-selection path",
        "unknown_training_samples": 0, "unknown_validation_samples": 0,
        "unknown_support_samples": 0, "unknown_normalization_samples": 0,
        "unknown_threshold_samples": 0, "known_test_selection_samples": 0,
    }
    write_json(target / "result.json", result)
    write_json(target / "lineage.json", {
        "source_run": str(source.resolve()), "source_checkpoint": str(checkpoint_path.resolve()),
        "source_checkpoint_sha256": checkpoint_sha, "source_failure": source_failure,
        "training_performed": False, "parameters_updated": False,
    })
    (target / "SUCCESS").write_text("SUCCESS\n", encoding="utf-8")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(json.dumps({
        "protocol_id": selected_protocol, "seed": seed,
        "source_run_status": result["source_run_status"],
        "validation_accuracy": result["validation_accuracy"],
        "known_test_macro_f1": known_closed["macro_f1"],
    }))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-id")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if (args.protocol_id is None) != (args.seed is None):
        parser.error("--protocol-id and --seed must be provided together")
    if args.protocol_id is not None:
        evaluate(args.protocol_id, args.seed)
    else:
        for service in SERVICES:
            for seed in SEEDS:
                evaluate(protocol_id(service), seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
