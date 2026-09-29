#!/usr/bin/env python3
"""Read-only diagnosis of the failed Native checkpoint/validation parity gate."""
from __future__ import annotations

import json
import sys

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from matched_common import IMAGE_CACHE, PROJECT, ROOT, image_index, load_protocols, role_rows

sys.path.insert(0, str(PROJECT.parent / "Open-Detect" / "code"))
from model import OpenDetectNet  # noqa: E402
sys.path.insert(0, str(PROJECT / "stage14c_native_opendetect_vnat" / "scripts"))
from train_one import collect_predictions  # noqa: E402
sys.path.insert(0, str(PROJECT.parent / "Open-Detect" / "reproduction"))
from run_reproduction import TrafficImages  # noqa: E402


def main() -> None:
    fold = "remote_access"
    torch.set_num_threads(4)
    device = torch.device("cuda:0")
    classes = list(load_protocols()[fold]["known_applications"])
    model = OpenDetectNet("resnet18", 1, 128, len(classes), 1.0, 1.0).to(device)
    checkpoint = torch.load(ROOT / "runs" / fold / "best_checkpoint.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    expected = json.loads((ROOT / "runs" / fold / "result.json").read_text())
    rows = role_rows(fold)["known_validation"]
    positions = np.asarray([image_index()[row["flow_uid"]] for row in rows], dtype=np.int64)
    cache_images = np.load(IMAGE_CACHE / "images.npy", mmap_mode="r", allow_pickle=False)
    prepared_images = np.load(ROOT / "inputs" / fold / "validation_images.npy", allow_pickle=False)
    prepared_labels = np.load(ROOT / "inputs" / fold / "validation_labels.npy", allow_pickle=False)
    truth = np.asarray([classes.index(row["service"]) for row in rows], dtype=np.int64)
    same_image = bool(np.array_equal(prepared_images, np.asarray(cache_images[positions]).reshape(-1, 32, 32)))
    same_label = bool(np.array_equal(truth, prepared_labels))
    predictions = []
    losses_predictions = []
    with torch.no_grad():
        for start in range(0, len(truth), 64):
            batch = torch.from_numpy(prepared_images[start:start + 64].copy()).unsqueeze(1).to(device=device, dtype=torch.float32).div_(255.0)
            labels = torch.as_tensor(truth[start:start + 64], device=device)
            mu, _, _ = model.encoder(batch)
            predictions.extend(model.distance(mu, model.prototypes).argmin(dim=1).cpu().tolist())
            losses_predictions.extend(model.loss(batch, labels)[2].cpu().tolist())
    direct = np.asarray(predictions)
    loss = np.asarray(losses_predictions)
    dataset = TrafficImages(prepared_images, prepared_labels, list(range(len(classes))), train=False, reindex=True)
    native_truth, native_prediction = collect_predictions(model, DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0), device)
    saved_matrix = np.load(ROOT / "runs" / fold / "validation_confusion_matrix.npy", allow_pickle=False)
    native_matrix = confusion_matrix(native_truth, native_prediction, labels=list(range(len(classes))))
    direct_matrix = confusion_matrix(truth, direct, labels=list(range(len(classes))))
    print(json.dumps({
        "fold": fold,
        "expected_validation_accuracy": expected["validation_accuracy"],
        "expected_validation_macro_f1": expected["validation_macro_f1"],
        "direct_accuracy": float(accuracy_score(truth, direct)),
        "direct_macro_f1": float(f1_score(truth, direct, average="macro", zero_division=0)),
        "loss_accuracy": float(accuracy_score(truth, loss)),
        "loss_macro_f1": float(f1_score(truth, loss, average="macro", zero_division=0)),
        "direct_loss_prediction_mismatch": int(np.sum(direct != loss)),
        "prepared_cache_images_equal": same_image,
        "prepared_role_labels_equal": same_label,
        "native_loader_macro_f1": float(f1_score(native_truth, native_prediction, average="macro", zero_division=0)),
        "native_loader_direct_prediction_mismatch": int(np.sum(native_prediction != direct)),
        "native_saved_confusion_equal": bool(np.array_equal(native_matrix, saved_matrix)),
        "direct_saved_confusion_equal": bool(np.array_equal(direct_matrix, saved_matrix)),
        "saved_confusion": saved_matrix.tolist(),
        "native_confusion": native_matrix.tolist(),
        "direct_confusion": direct_matrix.tolist(),
        "classes": classes,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
