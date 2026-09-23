#!/usr/bin/env python3
"""Shared, auditable Stage 11A implementation.

Only USTC v6 development splits are loaded here.  No CipherSpectrum sample
path is defined or accepted by this module.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader


STAGE_ROOT = Path(__file__).resolve().parents[1]
UNKNOWN_ROOT = STAGE_ROOT.parent
PROJECTS_ROOT = UNKNOWN_ROOT.parent
OPEN_DETECT_ROOT = PROJECTS_ROOT / "Open-Detect"
VENDOR_ROOT = OPEN_DETECT_ROOT / "code"
REPRODUCTION_ROOT = OPEN_DETECT_ROOT / "reproduction"
for _path in (VENDOR_ROOT, REPRODUCTION_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from corrected_model import CorrectedOpenDetectNet  # noqa: E402
from data.splits import get_splits  # noqa: E402
from run_reproduction import TrafficImages, array_digest, seed_everything  # noqa: E402
from utils import weight_init  # noqa: E402


CONFIG_PATH = STAGE_ROOT / "configs" / "stage11a_config.json"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows and fieldnames is None:
        raise ValueError(f"Cannot infer CSV schema for empty rows: {path}")
    names = fieldnames or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def append_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


class DataAnchoredOpenDetectNet(CorrectedOpenDetectNet):
    """Corrected Open-Detect with prototypes stored as a non-gradient buffer."""

    def __init__(
        self,
        arch: str = "resnet18",
        channel: int = 1,
        latent_dim: int = 128,
        n_classes: int = 10,
        temp_inter: float = 1,
        temp_intra: float = 1,
        init: bool = True,
    ) -> None:
        super().__init__(arch, channel, latent_dim, n_classes, temp_inter, temp_intra, init)
        initial = self.prototypes.detach().clone()
        del self._parameters["prototypes"]
        self.register_buffer("prototypes", initial, persistent=True)


def assert_prototype_is_buffer(model: nn.Module, optimizer: optim.Optimizer | None = None) -> None:
    parameter_names = dict(model.named_parameters())
    buffer_names = dict(model.named_buffers())
    if "prototypes" in parameter_names:
        raise AssertionError("DAP prototypes remain an optimizer-owned Parameter")
    if "prototypes" not in buffer_names:
        raise AssertionError("DAP prototypes are not a registered buffer")
    if model.prototypes.requires_grad:
        raise AssertionError("DAP prototype buffer requires gradients")
    if optimizer is not None:
        optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
        if id(model.prototypes) in optimizer_ids:
            raise AssertionError("DAP prototypes appear in optimizer param groups")


@dataclass
class LoaderBundle:
    train: DataLoader
    anchor_train: DataLoader
    validation: DataLoader
    known_test: DataLoader
    unknown_test: DataLoader
    known_classes: list[int]
    unknown_classes: list[int]
    arrays: dict[str, np.ndarray]


def load_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        return loaded["data"], loaded["target"]


def build_loaders(
    data_dir: Path,
    split: int,
    seed: int,
    batch_size: int,
    eval_batch_size: int,
    workers: int,
) -> LoaderBundle:
    train_x, train_y = load_npz(data_dir / "ustc_train.npz")
    val_x, val_y = load_npz(data_dir / "ustc_validation.npz")
    test_x, test_y = load_npz(data_dir / "ustc_test.npz")
    known_classes, unknown_classes, _, _ = get_splits("USTC", split)
    known_classes = list(map(int, known_classes))
    unknown_classes = list(map(int, unknown_classes))
    train = TrafficImages(train_x, train_y, known_classes, train=True, reindex=True)
    anchor_train = TrafficImages(train_x, train_y, known_classes, train=False, reindex=True)
    validation = TrafficImages(val_x, val_y, known_classes, train=False, reindex=True)
    known_test = TrafficImages(test_x, test_y, known_classes, train=False, reindex=True)
    unknown_test = TrafficImages(test_x, test_y, unknown_classes, train=False, reindex=False)
    kwargs = {
        "num_workers": workers,
        "pin_memory": True,
        "persistent_workers": workers > 0,
    }
    generator = torch.Generator().manual_seed(seed)
    return LoaderBundle(
        train=DataLoader(
            train,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
            drop_last=False,
            **kwargs,
        ),
        anchor_train=DataLoader(anchor_train, batch_size=eval_batch_size, shuffle=False, **kwargs),
        validation=DataLoader(validation, batch_size=eval_batch_size, shuffle=False, **kwargs),
        known_test=DataLoader(known_test, batch_size=eval_batch_size, shuffle=False, **kwargs),
        unknown_test=DataLoader(unknown_test, batch_size=eval_batch_size, shuffle=False, **kwargs),
        known_classes=known_classes,
        unknown_classes=unknown_classes,
        arrays={
            "train_x": train_x,
            "train_y": train_y,
            "val_x": val_x,
            "val_y": val_y,
            "test_x": test_x,
            "test_y": test_y,
        },
    )


def model_for_method(method: str, n_classes: int, latent_dim: int, device: torch.device) -> nn.Module:
    cls = DataAnchoredOpenDetectNet if method == "DAP" else CorrectedOpenDetectNet
    if method not in {"DAP", "B0"}:
        raise ValueError(f"Unsupported method: {method}")
    return cls("resnet18", 1, latent_dim, n_classes, 1, 1).to(device)


def initialised_dap_model(
    n_classes: int, latent_dim: int, device: torch.device
) -> DataAnchoredOpenDetectNet:
    model = model_for_method("DAP", n_classes, latent_dim, device)
    model.apply(weight_init)
    return model


def deterministic_mu(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    previous_training = model.training
    model.eval()
    with torch.no_grad():
        for images, batch_labels in loader:
            mu, _, _ = model.encoder(images.to(device, non_blocking=True))
            features.append(mu.cpu().numpy())
            labels.append(batch_labels.numpy())
    model.train(previous_training)
    return np.concatenate(features), np.concatenate(labels).astype(np.int64, copy=False)


DRIFT_FIELDS = [
    "epoch",
    "phase",
    "class_index",
    "original_class_id",
    "train_samples",
    "before_anchor_gap",
    "after_anchor_gap",
    "median_within_class_radius",
    "normalized_gap",
    "after_normalized_gap",
    "prototype_source",
]


def anchor_prototypes(
    model: DataAnchoredOpenDetectNet,
    loader: DataLoader,
    device: torch.device,
    original_class_ids: list[int],
    epoch: int,
    phase: str,
) -> list[dict[str, object]]:
    """Replace every prototype with a Known-Train deterministic-mu centroid."""
    features, labels = deterministic_mu(model, loader, device)
    if sorted(np.unique(labels).tolist()) != list(range(len(original_class_ids))):
        raise AssertionError("Known Train does not cover each reindexed class exactly as expected")
    before = model.prototypes.detach().cpu().numpy().astype(np.float64, copy=True)
    centers = np.vstack([features[labels == index].mean(axis=0) for index in range(len(original_class_ids))])
    with torch.no_grad():
        model.prototypes.copy_(torch.from_numpy(centers).to(device=device, dtype=model.prototypes.dtype))
    after = model.prototypes.detach().cpu().numpy().astype(np.float64, copy=True)
    rows: list[dict[str, object]] = []
    epsilon = np.finfo(np.float64).eps
    for index, original_class_id in enumerate(original_class_ids):
        values = np.asarray(features[labels == index], dtype=np.float64)
        center = np.asarray(centers[index], dtype=np.float64)
        radius = float(np.median(np.linalg.norm(values - center, axis=1)))
        before_gap = float(np.linalg.norm(before[index] - center))
        after_gap = float(np.linalg.norm(after[index] - center))
        rows.append(
            {
                "epoch": epoch,
                "phase": phase,
                "class_index": index,
                "original_class_id": original_class_id,
                "train_samples": len(values),
                "before_anchor_gap": before_gap,
                "after_anchor_gap": after_gap,
                "median_within_class_radius": radius,
                "normalized_gap": before_gap / (radius + epsilon),
                "after_normalized_gap": after_gap / (radius + epsilon),
                "prototype_source": "KNOWN_TRAIN_DETERMINISTIC_MU_ONLY",
            }
        )
    return rows


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    lamda: float,
    optimizer: optim.Optimizer | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"total": 0.0, "rec": 0.0, "kld": 0.0, "ent": 0.0, "dis": 0.0}
    seen = 0
    labels_all: list[np.ndarray] = []
    predictions_all: list[np.ndarray] = []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            _, _, predictions, losses = model.loss(images, labels)
            total_loss = lamda * (losses["rec"] + losses["kld"] + losses["ent"]) + (1 - lamda) * losses["dis"]
            if training:
                total_loss.backward()
                optimizer.step()
            batch = len(labels)
            seen += batch
            totals["total"] += float(total_loss.item()) * batch
            for name in ("rec", "kld", "ent", "dis"):
                totals[name] += float(losses[name].item()) * batch
            labels_all.append(labels.detach().cpu().numpy())
            predictions_all.append(predictions.detach().cpu().numpy())
    labels_np = np.concatenate(labels_all)
    predictions_np = np.concatenate(predictions_all)
    return {
        "accuracy": float(accuracy_score(labels_np, predictions_np)),
        "macro_f1": float(f1_score(labels_np, predictions_np, average="macro", zero_division=0)),
        **{name: value / seen for name, value in totals.items()},
    }


@dataclass
class LatentOutputs:
    mu: np.ndarray
    logvar: np.ndarray
    labels: np.ndarray
    native_scores: np.ndarray
    predictions: np.ndarray


def extract_outputs(model: nn.Module, loader: DataLoader, device: torch.device) -> LatentOutputs:
    mus: list[np.ndarray] = []
    logvars: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for images, batch_labels in loader:
            mean, raw_logvar, _ = model.encoder(images.to(device, non_blocking=True))
            logvar = torch.clamp(raw_logvar, min=model.min_logvar, max=model.max_logvar)
            class_kl = 0.5 * model.distance(mean, model.prototypes)
            variance_kl = 0.5 * torch.sum(logvar.exp() - logvar - 1, dim=1, keepdim=True)
            kl = class_kl + variance_kl
            min_scores, min_predictions = kl.min(dim=1)
            mus.append(mean.cpu().numpy())
            logvars.append(logvar.cpu().numpy())
            labels.append(batch_labels.numpy())
            scores.append(min_scores.cpu().numpy())
            predictions.append(min_predictions.cpu().numpy())
    return LatentOutputs(
        mu=np.concatenate(mus),
        logvar=np.concatenate(logvars),
        labels=np.concatenate(labels).astype(np.int64, copy=False),
        native_scores=np.concatenate(scores),
        predictions=np.concatenate(predictions).astype(np.int64, copy=False),
    )


def classification_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
    }


def detection_metrics(
    validation_known_scores: np.ndarray,
    test_known_scores: np.ndarray,
    test_unknown_scores: np.ndarray,
    eval_seed: int,
) -> dict[str, object]:
    threshold = float(np.quantile(validation_known_scores, 0.95, method="higher"))
    count = min(len(test_known_scores), len(test_unknown_scores))
    rng = np.random.default_rng(eval_seed)
    known_idx = rng.choice(len(test_known_scores), size=count, replace=False)
    unknown_idx = rng.choice(len(test_unknown_scores), size=count, replace=False)
    known = test_known_scores[known_idx]
    unknown = test_unknown_scores[unknown_idx]
    y_true = np.concatenate([np.zeros(count, dtype=np.int64), np.ones(count, dtype=np.int64)])
    y_score = np.concatenate([known, unknown])
    y_pred = (y_score >= threshold).astype(np.int64)
    return {
        "evaluation_label": "DEVELOPMENT_RESULT",
        "external_validation_label": "NOT_INDEPENDENT_EXTERNAL_VALIDATION",
        "protocol": "symmetric-balanced-1to1-v2",
        "threshold": threshold,
        "threshold_source": "Known Validation KL/NLL P95 only; numpy method=higher",
        "validation_known_acceptance": float(np.mean(validation_known_scores < threshold)),
        "sample_count_known": count,
        "sample_count_unknown": count,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "binary_f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auroc": float(roc_auc_score(y_true, y_score)),
        "auprc": float(average_precision_score(y_true, y_score)),
        "known_frr": float(np.mean(known >= threshold)),
        "ufar": float(np.mean(unknown < threshold)),
        "all_known_test_frr": float(np.mean(test_known_scores >= threshold)),
        "all_unknown_test_ufar": float(np.mean(test_unknown_scores < threshold)),
    }


def prototype_alignment(
    model: nn.Module,
    train_outputs: LatentOutputs,
    original_class_ids: list[int],
) -> tuple[list[dict[str, object]], dict[str, float]]:
    prototype = model.prototypes.detach().cpu().numpy().astype(np.float64)
    rows: list[dict[str, object]] = []
    epsilon = np.finfo(np.float64).eps
    for index, original_class_id in enumerate(original_class_ids):
        values = np.asarray(train_outputs.mu[train_outputs.labels == index], dtype=np.float64)
        center = values.mean(axis=0)
        gap = float(np.linalg.norm(prototype[index] - center))
        radius = float(np.median(np.linalg.norm(values - center, axis=1)))
        rows.append(
            {
                "class_index": index,
                "original_class_id": original_class_id,
                "train_samples": len(values),
                "gap": gap,
                "median_within_class_radius": radius,
                "normalized_gap": gap / (radius + epsilon),
                "centroid_source": "KNOWN_TRAIN_DETERMINISTIC_MU_ONLY",
            }
        )
    values = np.asarray([row["normalized_gap"] for row in rows], dtype=np.float64)
    return rows, {
        "normalized_gap_mean": float(values.mean()),
        "normalized_gap_median": float(np.median(values)),
        "normalized_gap_max": float(values.max()),
    }


def _fit_gmm(values: np.ndarray, components: int) -> tuple[GaussianMixture, list[dict[str, str]]]:
    caught_rows: list[dict[str, str]] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = GaussianMixture(
            n_components=components,
            covariance_type="full",
            reg_covar=1e-3,
            n_init=3,
            max_iter=300,
            random_state=0,
        ).fit(values)
    for item in caught:
        caught_rows.append(
            {"warning_category": item.category.__name__, "warning_message": str(item.message)}
        )
    return model, caught_rows


def density_diagnostics(
    train_outputs: LatentOutputs,
    validation_outputs: LatentOutputs,
    known_test_outputs: LatentOutputs,
    unknown_test_outputs: LatentOutputs,
    eval_seed: int,
) -> tuple[dict[str, object], dict[str, object]]:
    scaler = StandardScaler(copy=True).fit(np.asarray(train_outputs.mu, dtype=np.float64))
    train_scaled = scaler.transform(train_outputs.mu)
    pca = PCA(n_components=64, svd_solver="randomized", random_state=0).fit(train_scaled)
    transformed = {
        "train": pca.transform(train_scaled),
        "validation": pca.transform(scaler.transform(validation_outputs.mu)),
        "known_test": pca.transform(scaler.transform(known_test_outputs.mu)),
        "unknown_test": pca.transform(scaler.transform(unknown_test_outputs.mu)),
    }
    results: dict[str, object] = {
        "representation": "deterministic mu -> train-only StandardScaler -> train-only randomized PCA64",
        "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
        "methods": {},
        "warnings": [],
    }
    serializable_models: dict[str, object] = {"scaler": scaler, "pca": pca, "models": {}}
    n_classes = int(train_outputs.labels.max()) + 1
    for components, method in ((1, "K1"), (2, "K2")):
        models: list[GaussianMixture] = []
        for class_index in range(n_classes):
            model, caught = _fit_gmm(transformed["train"][train_outputs.labels == class_index], components)
            models.append(model)
            for item in caught:
                results["warnings"].append({"method": method, "class_index": class_index, **item})
        method_scores: dict[str, np.ndarray] = {}
        method_predictions: dict[str, np.ndarray] = {}
        for role, values in transformed.items():
            nll = np.column_stack([-model.score_samples(values) for model in models])
            method_scores[role] = nll.min(axis=1)
            method_predictions[role] = nll.argmin(axis=1)
        results["methods"][method] = {
            "components": components,
            "covariance_type": "full",
            "reg_covar": 1e-3,
            "n_init": 3,
            "max_iter": 300,
            "random_state": 0,
            "all_converged": bool(all(model.converged_ for model in models)),
            "validation_known": classification_metrics(
                validation_outputs.labels, method_predictions["validation"]
            ),
            "known_test": classification_metrics(
                known_test_outputs.labels, method_predictions["known_test"]
            ),
            "detection": detection_metrics(
                method_scores["validation"],
                method_scores["known_test"],
                method_scores["unknown_test"],
                eval_seed,
            ),
        }
        serializable_models["models"][method] = models
    return results, serializable_models


def evaluate_checkpoint(
    model: nn.Module,
    loaders: LoaderBundle,
    device: torch.device,
    method: str,
    scenario: str,
    seed: int,
    fold: int,
    density_model_path: Path,
) -> dict[str, object]:
    train = extract_outputs(model, loaders.anchor_train, device)
    validation = extract_outputs(model, loaders.validation, device)
    known_test = extract_outputs(model, loaders.known_test, device)
    unknown_test = extract_outputs(model, loaders.unknown_test, device)
    alignment_rows, alignment_summary = prototype_alignment(model, train, loaders.known_classes)
    native = {
        "validation_known": classification_metrics(validation.labels, validation.predictions),
        "known_test": classification_metrics(known_test.labels, known_test.predictions),
        "detection": detection_metrics(
            validation.native_scores,
            known_test.native_scores,
            unknown_test.native_scores,
            read_json(CONFIG_PATH)["training"]["eval_seed"],
        ),
    }
    density, models = density_diagnostics(
        train,
        validation,
        known_test,
        unknown_test,
        read_json(CONFIG_PATH)["training"]["eval_seed"],
    )
    density_model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(models, density_model_path, compress=3)
    return {
        "evaluation_label": "DEVELOPMENT_RESULT",
        "external_validation_label": "NOT_INDEPENDENT_EXTERNAL_VALIDATION",
        "scenario": scenario,
        "seed": seed,
        "fold": fold,
        "method": method,
        "data_roles": {
            "prototype_alignment": "Known Train deterministic mu only",
            "density_fit": "Known Train deterministic mu only",
            "threshold": "Known Validation score P95 only",
            "checkpoint_selection": "Known Validation Accuracy only",
            "test": "Known Test and USTC Unknown Test evaluation only",
        },
        "sample_counts": {
            "train_known": len(train.labels),
            "validation_known": len(validation.labels),
            "test_known": len(known_test.labels),
            "test_unknown": len(unknown_test.labels),
        },
        "prototype_alignment": alignment_summary,
        "prototype_alignment_by_class": alignment_rows,
        "native": native,
        "density_diagnostics": density,
        "density_models_sha256": sha256_file(density_model_path),
    }


def load_checkpoint(model: nn.Module, checkpoint_path: Path, device: torch.device) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return checkpoint


def array_hashes(loaders: LoaderBundle) -> dict[str, str]:
    arrays = loaders.arrays
    return {
        "train": array_digest(arrays["train_x"], arrays["train_y"]),
        "validation": array_digest(arrays["val_x"], arrays["val_y"]),
        "test": array_digest(arrays["test_x"], arrays["test_y"]),
    }


def set_reproducible(seed: int) -> None:
    seed_everything(seed)
    random.seed(seed)

