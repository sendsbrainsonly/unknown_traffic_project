#!/usr/bin/env python3
"""Evaluate M0/M1/M2 for one frozen VNAT protocol without training."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from stage14d_common import (
    ARTIFACT_ROOT,
    CACHE_ROOT,
    CONFIG_PATH,
    HARD_CLASSES,
    METHODS,
    OPENDETECT_ROOT,
    ROOT,
    canonical_array_sha256,
    centroid_scores,
    classification_metrics,
    detection_metrics,
    empirical_centroids,
    load_manifest_rows,
    load_protocols,
    local_knn10_scores,
    per_class_classification,
    read_json,
    robust_normalize,
    robust_parameters,
    sha256_file,
    verify_checkpoint_freeze,
    verify_stage14b,
    write_csv,
    write_json,
)

sys.path.insert(0, str(OPENDETECT_ROOT / "code"))
from model import OpenDetectNet  # noqa: E402


def extract_latent(
    model: torch.nn.Module,
    images: np.ndarray,
    indices: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mus: list[np.ndarray] = []
    logvars: list[np.ndarray] = []
    native_scores: list[np.ndarray] = []
    native_predictions: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            selected = indices[start : start + batch_size]
            batch = np.asarray(images[selected]).reshape(-1, 32, 32)
            tensor = torch.from_numpy(batch.copy()).unsqueeze(1).to(device=device, dtype=torch.float32).div_(255.0)
            mu, logvar, _ = model.encoder(tensor)
            distances = model.distance(mu, model.prototypes)
            kl = model.kl_div_to_prototypes(mu, logvar, model.prototypes)
            mus.append(mu.cpu().numpy())
            logvars.append(logvar.cpu().numpy())
            native_scores.append(kl.min(dim=1).values.cpu().numpy())
            native_predictions.append(distances.argmin(dim=1).cpu().numpy())
    return (
        np.concatenate(mus),
        np.concatenate(logvars),
        np.concatenate(native_scores).astype(np.float64),
        np.concatenate(native_predictions).astype(np.int64),
    )


def build_role_arrays(protocol_id: str, known_classes: list[str]) -> dict[str, dict[str, np.ndarray]]:
    rows = load_manifest_rows(protocol_id)
    flow_uids = np.load(CACHE_ROOT / "flow_uids.npy", allow_pickle=False)
    uid_to_index = {str(uid): index for index, uid in enumerate(flow_uids.tolist())}
    class_to_local = {name: index for index, name in enumerate(known_classes)}
    roles: dict[str, dict[str, np.ndarray]] = {}
    role_specs = {
        "train": ("known", "train"),
        "validation": ("known", "validation"),
        "known_test": ("known", "test"),
        "unknown_test": ("unknown", "unknown_test"),
    }
    for role, (class_role, split) in role_specs.items():
        selected = [row for row in rows if row["class_role"] == class_role and row["split"] == split]
        if not selected:
            raise RuntimeError(f"{protocol_id}: empty role {role}")
        applications = np.asarray([row["application"] for row in selected])
        labels = (
            np.asarray([class_to_local[name] for name in applications], dtype=np.int64)
            if role != "unknown_test"
            else np.full(len(selected), -1, dtype=np.int64)
        )
        roles[role] = {
            "indices": np.asarray([uid_to_index[row["flow_uid"]] for row in selected], dtype=np.int64),
            "flow_uids": np.asarray([row["flow_uid"] for row in selected]),
            "applications": applications,
            "labels": labels,
        }
    return roles


def unknown_rows(
    protocol_id: str,
    setting: str,
    seed: int,
    method: str,
    applications: np.ndarray,
    scores: np.ndarray,
    predictions: np.ndarray,
    threshold: float,
    known_classes: list[str],
) -> list[dict[str, object]]:
    rows = []
    for application in sorted(np.unique(applications).tolist()):
        mask = applications == application
        accepted = scores[mask] < threshold
        predicted = predictions[mask]
        counts = np.bincount(predicted, minlength=len(known_classes))
        top_index = int(np.argmax(counts))
        rows.append({
            "protocol_id": protocol_id,
            "setting": setting,
            "seed": seed,
            "method": method,
            "analysis_type": "unknown_true_class",
            "application": application,
            "support": int(mask.sum()),
            "precision": "",
            "recall": "",
            "f1": "",
            "known_frr": "",
            "unknown_recall": float(np.mean(~accepted)),
            "ufar": float(np.mean(accepted)),
            "score_mean": float(scores[mask].mean()),
            "score_median": float(np.median(scores[mask])),
            "top_absorbing_known_class": known_classes[top_index],
            "top_absorbed_count": int(counts[top_index]),
            "hard_class": application in HARD_CLASSES,
        })
    return rows


def known_rows(
    protocol_id: str,
    setting: str,
    seed: int,
    method: str,
    labels: np.ndarray,
    predictions: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    known_classes: list[str],
) -> list[dict[str, object]]:
    rows = []
    for local, metric in enumerate(per_class_classification(labels, predictions, known_classes)):
        mask = labels == local
        rows.append({
            "protocol_id": protocol_id,
            "setting": setting,
            "seed": seed,
            "method": method,
            "analysis_type": "known_true_class",
            **metric,
            "known_frr": float(np.mean(scores[mask] >= threshold)),
            "unknown_recall": "",
            "ufar": "",
            "score_mean": float(scores[mask].mean()),
            "score_median": float(np.median(scores[mask])),
            "top_absorbing_known_class": "",
            "top_absorbed_count": "",
            "hard_class": metric["application"] in HARD_CLASSES,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-id", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    started = time.time()
    output = ARTIFACT_ROOT / args.protocol_id
    output.mkdir(parents=True, exist_ok=True)
    if (output / "SUCCESS").is_file():
        print(json.dumps({"status": "SKIP_SUCCESS", "protocol_id": args.protocol_id}))
        return
    if (output / "FAILURE.json").exists():
        raise RuntimeError(f"refusing to overwrite preserved failure: {output}")
    try:
        stage14b_before = verify_stage14b()
        frozen_checkpoints = verify_checkpoint_freeze()
        protocols = load_protocols()
        if args.protocol_id not in protocols:
            raise KeyError(args.protocol_id)
        protocol = protocols[args.protocol_id]
        checkpoint_record = frozen_checkpoints[args.protocol_id]
        config = read_json(CONFIG_PATH)
        if args.batch_size != int(config["evaluation"]["inference_batch_size"]):
            raise RuntimeError("formal inference batch size must match the frozen Native validation batch size")
        known_classes = list(map(str, protocol["known_applications"]))
        unknown_classes = list(map(str, protocol["unknown_applications"]))
        if checkpoint_record["known_classes"] != known_classes or checkpoint_record["unknown_classes"] != unknown_classes:
            raise RuntimeError("checkpoint/protocol class-list mismatch")
        roles = build_role_arrays(args.protocol_id, known_classes)
        train_counts = np.bincount(roles["train"]["labels"], minlength=len(known_classes))
        if np.any(train_counts < 10):
            raise RuntimeError(f"kNN-10 support gate failed: {train_counts.tolist()}")

        device = torch.device(args.device)
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("Stage 14D formal inference requires assigned CUDA")
        model = OpenDetectNet("resnet18", 1, 128, len(known_classes), 1.0, 1.0).to(device)
        checkpoint = torch.load(checkpoint_record["checkpoint_path"], map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        images = np.load(CACHE_ROOT / "images.npy", mmap_mode="r", allow_pickle=False)

        latent: dict[str, dict[str, np.ndarray]] = {}
        for role in ("train", "validation", "known_test", "unknown_test"):
            mu, logvar, native_scores, native_predictions = extract_latent(
                model, images, roles[role]["indices"], device, args.batch_size
            )
            latent[role] = {
                "mu": mu,
                "logvar": logvar,
                "native_scores": native_scores,
                "native_predictions": native_predictions,
            }

        validation_native_class = classification_metrics(
            roles["validation"]["labels"], latent["validation"]["native_predictions"]
        )
        if abs(validation_native_class["known_accuracy"] - float(checkpoint_record["validation_accuracy"])) > 1e-12:
            raise RuntimeError("Native validation accuracy parity failed")
        if abs(validation_native_class["known_macro_f1"] - float(checkpoint_record["validation_macro_f1"])) > 1e-12:
            raise RuntimeError("Native validation Macro-F1 parity failed")

        centroids = empirical_centroids(latent["train"]["mu"], roles["train"]["labels"])
        global_scores: dict[str, np.ndarray] = {}
        centroid_predictions: dict[str, np.ndarray] = {}
        local_scores: dict[str, np.ndarray] = {}
        for role in ("validation", "known_test", "unknown_test"):
            global_scores[role], centroid_predictions[role] = centroid_scores(latent[role]["mu"], centroids)
            local_scores[role] = local_knn10_scores(
                latent[role]["mu"], centroid_predictions[role], latent["train"]["mu"], roles["train"]["labels"]
            )
        epsilon = float(config["des_v1"]["epsilon"])
        global_params = robust_parameters(global_scores["validation"], epsilon)
        local_params = robust_parameters(local_scores["validation"], epsilon)
        global_z = {
            role: robust_normalize(global_scores[role], global_params)
            for role in ("validation", "known_test", "unknown_test")
        }
        local_z = {
            role: robust_normalize(local_scores[role], local_params)
            for role in ("validation", "known_test", "unknown_test")
        }
        fused_scores = {
            role: 0.5 * global_z[role] + 0.5 * local_z[role]
            for role in ("validation", "known_test", "unknown_test")
        }
        method_scores = {
            "M0": {role: latent[role]["native_scores"] for role in ("validation", "known_test", "unknown_test")},
            "M1": global_scores,
            "M2": fused_scores,
        }
        method_predictions = {
            "M0": {role: latent[role]["native_predictions"] for role in ("validation", "known_test", "unknown_test")},
            "M1": centroid_predictions,
            "M2": centroid_predictions,
        }
        metrics: dict[str, dict[str, object]] = {}
        per_class_rows: list[dict[str, object]] = []
        score_rows: list[dict[str, object]] = []
        for method in METHODS:
            detection = detection_metrics(
                method_scores[method]["validation"],
                method_scores[method]["known_test"],
                method_scores[method]["unknown_test"],
            )
            classification = classification_metrics(
                roles["known_test"]["labels"], method_predictions[method]["known_test"]
            )
            metrics[method] = {**detection, **classification}
            threshold = float(detection["threshold"])
            per_class_rows.extend(known_rows(
                args.protocol_id, str(protocol["setting"]), int(protocol["seed"]), method,
                roles["known_test"]["labels"], method_predictions[method]["known_test"],
                method_scores[method]["known_test"], threshold, known_classes,
            ))
            per_class_rows.extend(unknown_rows(
                args.protocol_id, str(protocol["setting"]), int(protocol["seed"]), method,
                roles["unknown_test"]["applications"], method_scores[method]["unknown_test"],
                method_predictions[method]["unknown_test"], threshold, known_classes,
            ))
            for role in ("validation", "known_test", "unknown_test"):
                is_unknown = role == "unknown_test"
                predictions = method_predictions[method][role]
                scores = method_scores[method][role]
                for index in range(len(scores)):
                    score_rows.append({
                        "protocol_id": args.protocol_id,
                        "setting": protocol["setting"],
                        "seed": int(protocol["seed"]),
                        "role": role,
                        "flow_uid": str(roles[role]["flow_uids"][index]),
                        "true_application": str(roles[role]["applications"][index]),
                        "is_unknown": int(is_unknown),
                        "method": method,
                        "score": float(scores[index]),
                        "threshold": threshold,
                        "predicted_unknown": int(scores[index] >= threshold),
                        "predicted_known_application": known_classes[int(predictions[index])],
                        "global_score": float(global_scores[role][index]) if method == "M2" else "",
                        "local_score": float(local_scores[role][index]) if method == "M2" else "",
                        "global_z": float(global_z[role][index]) if method == "M2" else "",
                        "local_z": float(local_z[role][index]) if method == "M2" else "",
                    })

        np.savez_compressed(
            output / "latent_outputs.npz",
            train_mu=latent["train"]["mu"],
            train_labels=roles["train"]["labels"],
            train_flow_uids=roles["train"]["flow_uids"],
            validation_mu=latent["validation"]["mu"],
            validation_logvar=latent["validation"]["logvar"],
            validation_labels=roles["validation"]["labels"],
            validation_flow_uids=roles["validation"]["flow_uids"],
            known_test_mu=latent["known_test"]["mu"],
            known_test_logvar=latent["known_test"]["logvar"],
            known_test_labels=roles["known_test"]["labels"],
            known_test_flow_uids=roles["known_test"]["flow_uids"],
            unknown_test_mu=latent["unknown_test"]["mu"],
            unknown_test_logvar=latent["unknown_test"]["logvar"],
            unknown_test_applications=roles["unknown_test"]["applications"],
            unknown_test_flow_uids=roles["unknown_test"]["flow_uids"],
        )
        np.savez_compressed(
            output / "score_arrays.npz",
            validation_m0=method_scores["M0"]["validation"],
            known_test_m0=method_scores["M0"]["known_test"],
            unknown_test_m0=method_scores["M0"]["unknown_test"],
            validation_m1=method_scores["M1"]["validation"],
            known_test_m1=method_scores["M1"]["known_test"],
            unknown_test_m1=method_scores["M1"]["unknown_test"],
            validation_m2=method_scores["M2"]["validation"],
            known_test_m2=method_scores["M2"]["known_test"],
            unknown_test_m2=method_scores["M2"]["unknown_test"],
            validation_global=global_scores["validation"],
            validation_local=local_scores["validation"],
            known_test_global=global_scores["known_test"],
            known_test_local=local_scores["known_test"],
            unknown_test_global=global_scores["unknown_test"],
            unknown_test_local=local_scores["unknown_test"],
        )
        write_csv(output / "sample_scores.csv", score_rows)
        write_csv(output / "per_class_analysis.csv", per_class_rows)
        stage14b_after = verify_stage14b()
        checkpoint_after = sha256_file(Path(checkpoint_record["checkpoint_path"]))
        if checkpoint_after != checkpoint_record["checkpoint_sha256"]:
            raise RuntimeError("checkpoint changed during evaluation")
        result = {
            "status": "SUCCESS",
            "protocol_id": args.protocol_id,
            "setting": protocol["setting"],
            "seed": int(protocol["seed"]),
            "known_classes": known_classes,
            "unknown_classes": unknown_classes,
            "hard_known_classes": sorted(HARD_CLASSES & set(known_classes)),
            "hard_unknown_classes": sorted(HARD_CLASSES & set(unknown_classes)),
            "samples": {role: int(len(values["indices"])) for role, values in roles.items()},
            "metrics": metrics,
            "normalization": {"M2_global": global_params, "M2_local": local_params},
            "checkpoint": checkpoint_record,
            "checkpoint_sha256_after": checkpoint_after,
            "stage14b_before": stage14b_before,
            "stage14b_after": stage14b_after,
            "native_validation_parity": {
                "accuracy": validation_native_class["known_accuracy"],
                "macro_f1": validation_native_class["known_macro_f1"],
                "status": "PASS",
            },
            "array_hashes": {
                "train_mu": canonical_array_sha256(latent["train"]["mu"]),
                "validation_mu": canonical_array_sha256(latent["validation"]["mu"]),
                "known_test_mu": canonical_array_sha256(latent["known_test"]["mu"]),
                "unknown_test_mu": canonical_array_sha256(latent["unknown_test"]["mu"]),
            },
            "operations": {
                "new_encoder_training": False,
                "optimizer": None,
                "backward": False,
                "unknown_support_samples": 0,
                "unknown_normalization_samples": 0,
                "unknown_threshold_samples": 0,
                "test_parameter_selection": False,
                "k": 10,
                "fusion_weights": [0.5, 0.5],
                "threshold": "Known Validation P95",
                "inference_batch_size": args.batch_size,
                "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "UNSET"),
            },
            "duration_seconds": time.time() - started,
        }
        write_json(output / "result.json", result)
        (output / "SUCCESS").write_text("success\n", encoding="utf-8")
        print(json.dumps({"status": "SUCCESS", "protocol_id": args.protocol_id, "metrics": metrics}, indent=2))
    except Exception as error:
        write_json(output / "FAILURE.json", {
            "status": "FAILURE",
            "protocol_id": args.protocol_id,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "duration_seconds": time.time() - started,
        })
        raise


if __name__ == "__main__":
    main()
