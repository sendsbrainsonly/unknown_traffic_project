#!/usr/bin/env python3
"""Score the frozen native Open-Detect checkpoint on matched Stage 44 flows."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader

from matched_common import FOLDS, IMAGE_CACHE, PROJECT, ROOT, STAGE44, balanced_indices, image_index, load_protocols, read_csv, role_rows, sha256, verify_freeze, write_csv


sys.path.insert(0, str(PROJECT.parent / "Open-Detect" / "code"))
from model import OpenDetectNet  # noqa: E402
sys.path.insert(0, str(PROJECT.parent / "Open-Detect" / "reproduction"))
from run_reproduction import TrafficImages  # noqa: E402


def native_outputs(model: torch.nn.Module, images: np.ndarray, positions: np.ndarray,
                   device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Use Native PIL/ToTensor preprocessing, KL score and prototype prediction."""
    scores: list[np.ndarray] = []
    predicted: list[np.ndarray] = []
    selected_images = np.asarray(images[positions]).reshape(-1, 32, 32)
    dataset = TrafficImages(selected_images, np.zeros(len(positions), dtype=np.int64), [0], train=False, reindex=True)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    model.eval()
    with torch.no_grad():
        for tensor, _ in loader:
            tensor = tensor.to(device, non_blocking=True)
            mu, logvar, _ = model.encoder(tensor)
            distances = model.distance(mu, model.prototypes)
            kl = model.kl_div_to_prototypes(mu, logvar, model.prototypes)
            scores.append(kl.min(dim=1).values.cpu().numpy())
            predicted.append(distances.argmin(dim=1).cpu().numpy())
    return np.concatenate(scores).astype(np.float64), np.concatenate(predicted).astype(np.int64)


def metrics(truth: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, float]:
    decision = score > threshold  # Stage 44 operating-point convention for both methods.
    known, unknown = truth == 0, truth == 1
    return {
        "auroc": float(roc_auc_score(truth, score)),
        "auprc": float(average_precision_score(truth, score)),
        "binary_accuracy": float(accuracy_score(truth, decision)),
        "binary_precision": float(precision_score(truth, decision, zero_division=0)),
        "binary_recall": float(recall_score(truth, decision, zero_division=0)),
        "binary_f1": float(f1_score(truth, decision, zero_division=0)),
        "ufar": float(np.mean(~decision[unknown])),
        "known_frr": float(np.mean(decision[known])),
        "known_acceptance": float(np.mean(~decision[known])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", choices=FOLDS, required=True)
    args = parser.parse_args()
    fold = args.fold
    frozen_before = verify_freeze()
    protocol = load_protocols()[fold]
    roles = role_rows(fold)
    result = json.loads((ROOT / "runs" / fold / "result.json").read_text(encoding="utf-8"))
    checkpoint_path = ROOT / "runs" / fold / "best_checkpoint.pt"
    if result["status"] != "success" or not (ROOT / "runs" / fold / "COMPLETED").is_file():
        raise RuntimeError(f"{fold}: Native training is not complete")
    checkpoint_before = sha256(checkpoint_path)
    if checkpoint_before != result["checkpoint_sha256"]:
        raise RuntimeError(f"{fold}: checkpoint hash mismatch")
    if result["known_classes"] != protocol["known_applications"] or result["unknown_classes"] != [fold]:
        raise RuntimeError(f"{fold}: trained class list differs from frozen fold")
    if result["train_samples"] != len(roles["known_train"]) or result["validation_samples"] != len(roles["known_validation"]):
        raise RuntimeError(f"{fold}: trained role count differs from frozen fold")
    input_audit = json.loads((ROOT / "inputs" / fold / "input_audit.json").read_text(encoding="utf-8"))
    if input_audit["status"] != "PASS" or input_audit["frozen_inputs"] != frozen_before:
        raise RuntimeError(f"{fold}: Known-only input audit mismatch")
    if not torch.cuda.is_available():
        raise RuntimeError("Native matched evaluation requires the assigned GPU")
    torch.set_num_threads(4)
    device = torch.device("cuda:0")
    model = OpenDetectNet("resnet18", 1, 128, len(protocol["known_applications"]), 1.0, 1.0).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    uid_index = image_index()
    images = np.load(IMAGE_CACHE / "images.npy", mmap_mode="r", allow_pickle=False)

    output = ROOT / "evaluation" / fold
    output.mkdir(parents=True, exist_ok=False)
    score: dict[str, np.ndarray] = {}
    predicted: dict[str, np.ndarray] = {}
    for role in ("known_validation", "known_test", "unknown_test"):
        positions = np.asarray([uid_index[row["flow_uid"]] for row in roles[role]], dtype=np.int64)
        score[role], predicted[role] = native_outputs(model, images, positions, device)
        if not np.isfinite(score[role]).all():
            raise RuntimeError(f"{fold}: non-finite Native score in {role}")

    known_classes = list(protocol["known_applications"])
    val_y = np.asarray([known_classes.index(row["service"]) for row in roles["known_validation"]])
    val_accuracy = float(accuracy_score(val_y, predicted["known_validation"]))
    val_macro_f1 = float(f1_score(val_y, predicted["known_validation"], average="macro", zero_division=0))
    if abs(val_accuracy - result["validation_accuracy"]) > 1e-12 or abs(val_macro_f1 - result["validation_macro_f1"]) > 1e-12:
        raise RuntimeError(f"{fold}: Native checkpoint/validation parity failed")
    test_y = np.asarray([known_classes.index(row["service"]) for row in roles["known_test"]])
    closed = {
        "accuracy": float(accuracy_score(test_y, predicted["known_test"])),
        "macro_f1": float(f1_score(test_y, predicted["known_test"], average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(test_y, predicted["known_test"], average="weighted", zero_division=0)),
    }
    (output / "closed_set_metrics.json").write_text(json.dumps(closed, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    thresholds = {percentile: float(np.percentile(score["known_validation"], percentile, method="higher"))
                  for percentile in (90, 95, 99)}
    (output / "calibration.json").write_text(json.dumps({
        "source": "Known Validation only", "thresholds": thresholds,
        "score": "Open-Detect Native minimum prototype KL", "decision": "score > threshold",
        "unknown_used": 0, "test_used": 0,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    known_ids = [row["flow_uid"] for row in roles["known_test"]]
    unknown_ids = [row["flow_uid"] for row in roles["unknown_test"]]
    count = min(len(known_ids), len(unknown_ids))
    known_pick = balanced_indices(known_ids, count, f"{fold}:known")
    unknown_pick = balanced_indices(unknown_ids, count, f"{fold}:unknown")
    result_rows: list[dict[str, object]] = []
    for view, known_scores, unknown_scores in (
        ("natural", score["known_test"], score["unknown_test"]),
        ("balanced_1to1", score["known_test"][known_pick], score["unknown_test"][unknown_pick]),
    ):
        truth = np.r_[np.zeros(len(known_scores), dtype=np.int64), np.ones(len(unknown_scores), dtype=np.int64)]
        combined = np.r_[known_scores, unknown_scores]
        for percentile, threshold in thresholds.items():
            result_rows.append({
                "fold": fold, "unknown_service": fold, "method": "od_native", "view": view,
                "threshold_percentile": percentile, "threshold": threshold,
                "known_samples": len(known_scores), "unknown_samples": len(unknown_scores),
                "known_closed_accuracy": closed["accuracy"],
                "known_closed_macro_f1": closed["macro_f1"],
                "known_closed_weighted_f1": closed["weighted_f1"],
                **metrics(truth, combined, threshold),
            })
    write_csv(output / "open_set_metrics.csv", result_rows)

    sample_rows: list[dict[str, object]] = []
    for role in ("known_validation", "known_test", "unknown_test"):
        for row, value, predicted_class in zip(roles[role], score[role], predicted[role]):
            sample_rows.append({
                "fold": fold, "flow_uid": row["flow_uid"], "role": role,
                "service": row["service"], "application": row["application"],
                "vpn_status": row["vpn_status"], "is_unknown": int(role == "unknown_test"),
                "score_od_native": float(value),
                "predicted_known": known_classes[int(predicted_class)],
                "prediction_od_native_p95": int(value > thresholds[95]),
            })
    write_csv(output / "sample_scores.csv", sample_rows)
    application_rows = []
    for application in sorted({row["application"] for row in roles["unknown_test"]}):
        selected = np.asarray([i for i, row in enumerate(roles["unknown_test"]) if row["application"] == application])
        truth = np.r_[np.zeros(len(known_ids), dtype=np.int64), np.ones(len(selected), dtype=np.int64)]
        combined = np.r_[score["known_test"], score["unknown_test"][selected]]
        application_rows.append({
            "fold": fold, "unknown_application": application, "method": "od_native",
            "unknown_samples": len(selected), **metrics(truth, combined, thresholds[95]),
        })
    write_csv(output / "per_unknown_application.csv", application_rows)

    baseline = read_csv(STAGE44 / "protocols" / fold / "evaluation/sample_scores.csv")
    for role in ("known_validation", "known_test", "unknown_test"):
        if [row["flow_uid"] for row in sample_rows if row["role"] == role] != [row["flow_uid"] for row in baseline if row["role"] == role]:
            raise RuntimeError(f"{fold}: Open-Detect and three-view sample IDs differ for {role}")
    if sha256(checkpoint_path) != checkpoint_before or verify_freeze() != frozen_before:
        raise RuntimeError(f"{fold}: checkpoint or protocol changed during evaluation")
    verification = {
        "status": "PASS", "fold": fold, "role_manifest_sha256": sha256(STAGE44 / "protocols" / fold / "role_manifest.csv"),
        "checkpoint_sha256_before": checkpoint_before, "checkpoint_sha256_after": sha256(checkpoint_path),
        "source_freeze_before": frozen_before, "source_freeze_after": verify_freeze(),
        "validation_parity_accuracy": val_accuracy, "validation_parity_macro_f1": val_macro_f1,
        "metric_rows": len(result_rows), "sample_score_rows": len(sample_rows),
        "unknown_train_samples": 0, "unknown_validation_samples": 0,
        "unknown_threshold_samples": 0, "test_threshold_samples": 0,
        "matched_test_flow_ids": True,
    }
    (output / "evaluation_verification.json").write_text(json.dumps(verification, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"fold": fold, "status": "PASS", "closed": closed, "metric_rows": len(result_rows)}), flush=True)


if __name__ == "__main__":
    main()
