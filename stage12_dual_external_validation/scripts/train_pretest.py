#!/usr/bin/env python3
"""Train one frozen Stage 12 encoder and calibrate M0/M1 without test access."""

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
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from stage12_common import (
    CONFIG_PATH,
    STAGE_ROOT,
    array_digest,
    git_identity,
    load_config,
    quantile_higher,
    seed_everything,
    sha256_file,
    write_csv,
    write_json,
)


class TrafficImages(Dataset):
    def __init__(self, data: np.ndarray, targets: np.ndarray, known_labels: list[int], train: bool) -> None:
        mapping = {label: index for index, label in enumerate(known_labels)}
        if any(int(value) not in mapping for value in targets):
            raise ValueError("non-known label entered a train/validation dataset")
        self.data = data
        self.targets = np.asarray([mapping[int(value)] for value in targets], dtype=np.int64)
        self.transform = transforms.Compose(
            [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor()]
            if train else [transforms.ToTensor()]
        )

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int):
        return self.transform(Image.fromarray(self.data[index], mode="L")), int(self.targets[index])


def load_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    bundle = np.load(path, allow_pickle=False)
    data, target = bundle["data"], bundle["target"]
    if data.ndim != 3 or data.shape[1:] != (32, 32) or data.dtype != np.uint8:
        raise ValueError(f"invalid image array: {path} {data.shape} {data.dtype}")
    return data, target.astype(np.int64, copy=False)


def import_open_detect(config: dict):
    root = Path(config["open_detect_root"])
    vendor = root / "code"
    reproduction = root / "reproduction"
    for path in (vendor, reproduction):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from corrected_model import CorrectedOpenDetectNet
    from run_reproduction import reset_prototypes_in_place, run_epoch
    from utils import weight_init
    return CorrectedOpenDetectNet, reset_prototypes_in_place, run_epoch, weight_init


def collect_mu_logvar(model, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means, logvars, labels = [], [], []
    model.eval()
    with torch.no_grad():
        for images, target in loader:
            mean, raw_logvar, _ = model.encoder(images.to(device, non_blocking=True))
            means.append(mean.cpu().numpy())
            logvars.append(torch.clamp(raw_logvar, -30.0, 20.0).cpu().numpy())
            labels.append(target.numpy())
    return np.concatenate(means), np.concatenate(logvars), np.concatenate(labels)


def native_scores(model, mu: np.ndarray, logvar: np.ndarray, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    values = torch.from_numpy(mu).to(device=device, dtype=torch.float32)
    logs = torch.from_numpy(logvar).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        class_kl = 0.5 * model.distance(values, model.prototypes)
        variance_kl = 0.5 * torch.sum(logs.exp() - logs - 1, dim=1, keepdim=True)
        distance = class_kl + variance_kl
        scores, predictions = torch.min(distance, dim=1)
    return scores.cpu().numpy().astype(np.float64), predictions.cpu().numpy().astype(np.int64)


def empirical_centroids(mu: np.ndarray, labels: np.ndarray, n_classes: int) -> np.ndarray:
    return np.vstack([mu[labels == index].mean(axis=0) for index in range(n_classes)]).astype(np.float32)


def des_scores(mu: np.ndarray, centroids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    distances = ((mu[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    predictions = np.argmin(distances, axis=1)
    return distances[np.arange(len(mu)), predictions].astype(np.float64), predictions.astype(np.int64)


def code_hash(paths: list[Path]) -> str:
    import hashlib

    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode())
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def run(dataset: str, setting: str, seed: int) -> dict:
    config = load_config()
    training = config["training"]
    protocol_dir = STAGE_ROOT / "protocol" / dataset
    freeze = json.loads((protocol_dir / "protocol_freeze.json").read_text(encoding="utf-8"))
    if freeze["status"] != "PROTOCOL_FROZEN_PRETRAIN" or freeze["test_metrics_opened"]:
        raise RuntimeError("dataset protocol is not in unopened frozen state")
    setting_data = STAGE_ROOT / "artifacts" / dataset / "protocol" / setting
    protocol = json.loads((setting_data / "protocol.json").read_text(encoding="utf-8"))
    if not protocol["unknown_free"] or any(protocol["prohibited_overlaps"].values()):
        raise RuntimeError("split protocol failed Unknown-free/leakage checks")
    output = STAGE_ROOT / "runs" / dataset / setting / f"seed{seed}"
    ready = output / "READY_FOR_ONE_SHOT_TEST"
    if ready.is_file():
        return {"status": "SKIP_READY", "output": str(output)}
    if output.exists():
        raise FileExistsError(f"preserved incomplete run; refusing overwrite: {output}")
    output.mkdir(parents=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Stage12 training")

    with (protocol_dir / "split_manifest.csv").open(encoding="utf-8", newline="") as handle:
        split_rows = [row for row in csv.DictReader(handle) if row["setting"] == setting]
    manifest_files = {
        "known_train": "train_manifest.csv",
        "known_validation": "val_manifest.csv",
        "known_test": "known_test_manifest.csv",
        "unknown_test": "unknown_test_manifest.csv",
    }
    manifest_hashes = {}
    for role, name in manifest_files.items():
        role_rows = [row for row in split_rows if row["role"] == role]
        if not role_rows:
            raise RuntimeError(f"empty frozen manifest role: {dataset}/{setting}/{role}")
        write_csv(output / name, role_rows)
        manifest_hashes[role] = sha256_file(output / name)

    # Strict boundary: only these two files are loaded before test opening.
    train_path = setting_data / "known_train.npz"
    val_path = setting_data / "known_validation.npz"
    train_x, train_y = load_npz(train_path)
    val_x, val_y = load_npz(val_path)
    known_names = list(protocol["known_classes"])
    label_map = json.loads((protocol_dir / "canonical_label_map.json").read_text(encoding="utf-8"))[
        "canonical_class_to_label"
    ]
    known_labels = [int(label_map[name]) for name in known_names]
    if set(map(int, np.unique(train_y))) != set(known_labels) or set(map(int, np.unique(val_y))) != set(known_labels):
        raise RuntimeError("Known Train/Validation class set mismatch")

    seed_everything(seed)
    generator = torch.Generator().manual_seed(seed)
    train_set = TrafficImages(train_x, train_y, known_labels, train=True)
    val_set = TrafficImages(val_x, val_y, known_labels, train=False)
    loader_args = {"num_workers": int(training["workers"]), "pin_memory": True, "persistent_workers": False}
    train_loader = DataLoader(
        train_set, batch_size=int(training["batch_size"]), shuffle=True,
        generator=generator, drop_last=False, **loader_args,
    )
    stable_train_loader = DataLoader(
        TrafficImages(train_x, train_y, known_labels, train=False),
        batch_size=int(training["eval_batch_size"]), shuffle=False, **loader_args,
    )
    val_loader = DataLoader(
        val_set, batch_size=int(training["eval_batch_size"]), shuffle=False, **loader_args,
    )
    CorrectedOpenDetectNet, reset_prototypes, run_epoch, weight_init = import_open_detect(config)
    device = torch.device("cuda:0")
    model = CorrectedOpenDetectNet(
        training["architecture"], int(training["channels"]), int(training["latent_dim"]),
        len(known_labels), 1, 1,
    ).to(device)
    model.apply(weight_init)
    optimizer = optim.Adam(model.parameters(), lr=float(training["learning_rate"]), betas=tuple(training["betas"]))
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=list(map(int, training["scheduler_milestones"])),
        gamma=float(training["scheduler_gamma"]),
    )
    run_config = {
        "dataset": dataset,
        "setting": setting,
        "seed": seed,
        "known_classes": known_names,
        "unknown_classes": protocol["unknown_classes"],
        "known_labels_original": known_labels,
        "train_samples": len(train_set),
        "validation_samples": len(val_set),
        "train_array_sha256": array_digest(train_x, train_y),
        "validation_array_sha256": array_digest(val_x, val_y),
        "training": training,
        "threshold": config["threshold"],
        "methods": config["methods"],
        "gpu": torch.cuda.get_device_name(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "open_detect_git": git_identity(Path(config["open_detect_root"])),
        "stage12_git": git_identity(STAGE_ROOT.parent),
        "test_data_loaded": False,
        "unknown_data_loaded": False,
    }
    write_json(output / "config.json", run_config)
    best_accuracy = -1.0
    best_epoch = -1
    patience_reference = -1.0
    epochs_without_improvement = 0
    completed_epochs = 0
    stopped_early = False
    started = time.time()
    csv_fields = [
        "epoch", "lr", "elapsed_seconds", "train_accuracy", "train_total", "train_rec",
        "train_kld", "train_ent", "train_dis", "validation_accuracy", "validation_total",
        "validation_rec", "validation_kld", "validation_ent", "validation_dis", "best_epoch",
        "best_validation_accuracy", "prototype_reset", "epochs_without_improvement",
    ]
    with (
        (output / "training_log.jsonl").open("x", encoding="utf-8") as handle,
        (output / "training_log.csv").open("x", encoding="utf-8", newline="") as csv_handle,
    ):
        csv_writer = csv.DictWriter(csv_handle, fieldnames=csv_fields)
        csv_writer.writeheader()
        for epoch in range(int(training["epochs"])):
            epoch_started = time.time()
            train_metrics = run_epoch(model, train_loader, device, float(training["lambda"]), optimizer)
            reset = epoch in set(map(int, training["prototype_reset_zero_based_epochs"]))
            if reset:
                reset_prototypes(model, train_loader, device)
            val_metrics = run_epoch(model, val_loader, device, float(training["lambda"]), None)
            scheduler.step()
            if val_metrics["accuracy"] > best_accuracy:
                best_accuracy = float(val_metrics["accuracy"])
                best_epoch = epoch + 1
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "n_classes": len(known_labels),
                    "latent_dim": int(training["latent_dim"]),
                    "epoch": best_epoch,
                    "validation_accuracy": best_accuracy,
                    "known_classes": known_names,
                    "known_labels_original": known_labels,
                }, output / "model_best.pt")
            if val_metrics["accuracy"] > patience_reference + float(training["early_stop_min_delta"]):
                patience_reference = float(val_metrics["accuracy"])
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if reset:
                patience_reference = float(val_metrics["accuracy"])
                epochs_without_improvement = 0
            completed_epochs = epoch + 1
            record = {
                "epoch": completed_epochs,
                "lr": optimizer.param_groups[0]["lr"],
                "elapsed_seconds": time.time() - epoch_started,
                "train": train_metrics,
                "validation": val_metrics,
                "best_epoch": best_epoch,
                "best_validation_accuracy": best_accuracy,
                "prototype_reset": reset,
                "epochs_without_improvement": epochs_without_improvement,
            }
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            csv_writer.writerow({
                "epoch": completed_epochs,
                "lr": record["lr"],
                "elapsed_seconds": record["elapsed_seconds"],
                **{f"train_{name}": train_metrics[name] for name in ("accuracy", "total", "rec", "kld", "ent", "dis")},
                **{f"validation_{name}": val_metrics[name] for name in ("accuracy", "total", "rec", "kld", "ent", "dis")},
                "best_epoch": best_epoch,
                "best_validation_accuracy": best_accuracy,
                "prototype_reset": reset,
                "epochs_without_improvement": epochs_without_improvement,
            })
            csv_handle.flush()
            print(
                f"dataset={dataset} setting={setting} seed={seed} epoch={completed_epochs:03d}/"
                f"{training['epochs']} train_acc={train_metrics['accuracy']:.6f} "
                f"val_acc={val_metrics['accuracy']:.6f} seconds={record['elapsed_seconds']:.1f}", flush=True,
            )
            if (
                completed_epochs >= int(training["early_stop_min_epoch"])
                and epochs_without_improvement >= int(training["early_stop_patience"])
            ):
                stopped_early = True
                break

    checkpoint = torch.load(output / "model_best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    train_mu, train_logvar, train_labels = collect_mu_logvar(model, stable_train_loader, device)
    val_mu, val_logvar, val_labels = collect_mu_logvar(model, val_loader, device)
    m0_val_scores, m0_val_predictions = native_scores(model, val_mu, val_logvar, device)
    centroids = empirical_centroids(train_mu, train_labels, len(known_labels))
    m1_val_scores, m1_val_predictions = des_scores(val_mu, centroids)
    thresholds = {
        "M0_OPEN_DETECT_NATIVE": {
            "threshold": quantile_higher(m0_val_scores),
            "source": "Known Validation only",
            "numpy_method": "higher",
        },
        "M1_DES_V0": {
            "threshold": quantile_higher(m1_val_scores),
            "source": "Known Validation only",
            "numpy_method": "higher",
        },
    }
    write_json(output / "thresholds.json", thresholds)
    write_json(output / "native_threshold.json", thresholds["M0_OPEN_DETECT_NATIVE"])
    write_json(output / "des_threshold.json", thresholds["M1_DES_V0"])
    np.save(output / "des_centroids.npy", centroids, allow_pickle=False)
    checkpoint_sha256 = sha256_file(output / "model_best.pt")
    centroids_sha256 = sha256_file(output / "des_centroids.npy")
    (output / "checkpoint.sha256").write_text(checkpoint_sha256 + "  model_best.pt\n", encoding="utf-8")
    (output / "des_centroids.sha256").write_text(centroids_sha256 + "  des_centroids.npy\n", encoding="utf-8")
    np.savez_compressed(
        output / "known_train_validation_features.npz",
        train_mu=train_mu, train_logvar=train_logvar, train_labels_reindexed=train_labels,
        validation_mu=val_mu, validation_logvar=val_logvar, validation_labels_reindexed=val_labels,
        validation_m0_scores=m0_val_scores, validation_m0_predictions=m0_val_predictions,
        validation_m1_scores=m1_val_scores, validation_m1_predictions=m1_val_predictions,
    )
    code_files = [Path(__file__), Path(__file__).with_name("stage12_common.py"), CONFIG_PATH]
    pretest = {
        "status": "READY_FOR_ONE_SHOT_TEST",
        "dataset": dataset,
        "setting": setting,
        "seed": seed,
        "known_classes": known_names,
        "unknown_classes": protocol["unknown_classes"],
        "train_manifest_hash": manifest_hashes["known_train"],
        "validation_manifest_hash": manifest_hashes["known_validation"],
        "known_test_manifest_hash": manifest_hashes["known_test"],
        "unknown_test_manifest_hash": manifest_hashes["unknown_test"],
        "train_array_sha256": run_config["train_array_sha256"],
        "validation_array_sha256": run_config["validation_array_sha256"],
        "known_test_array_sha256_frozen_not_opened": protocol["roles"]["known_test"]["array_sha256"],
        "unknown_test_array_sha256_frozen_not_opened": protocol["roles"]["unknown_test"]["array_sha256"],
        "checkpoint_sha256": checkpoint_sha256,
        "open_detect_threshold": thresholds["M0_OPEN_DETECT_NATIVE"]["threshold"],
        "des_centroids_sha256": centroids_sha256,
        "des_threshold": thresholds["M1_DES_V0"]["threshold"],
        "method_definitions": {
            "M0": "min_y KL[N(mu_x,diag(exp(logvar_x))) || N(p_y,I)]",
            "M1": "min_y squared_Euclidean(mu_x, Known-Train empirical centroid_y)",
        },
        "code_hash": code_hash(code_files),
        "config_hash": sha256_file(CONFIG_PATH),
        "protocol_hash": sha256_file(setting_data / "protocol.json"),
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_accuracy,
        "completed_epochs": completed_epochs,
        "stopped_early": stopped_early,
        "test_data_loaded": False,
        "unknown_data_loaded": False,
        "test_metrics_computed": False,
    }
    write_json(output / "pretest_freeze.json", pretest)
    (output / "RESULTS.md").write_text(
        f"# Stage 12 pre-test run: {dataset}/{setting}/seed{seed}\n\n"
        f"- Status: `READY_FOR_ONE_SHOT_TEST`\n"
        f"- Known classes: `{len(known_names)}`; Unknown classes: `{len(protocol['unknown_classes'])}`\n"
        f"- Best epoch: `{best_epoch}`; Known-Validation accuracy: `{best_accuracy:.6f}`\n"
        "- M0 and M1 share this exact checkpoint.\n"
        "- DES centroids use Known Train only; both thresholds use Known Validation P95 only.\n"
        "- Known Test and Unknown Test were not loaded and no test metric was computed.\n",
        encoding="utf-8",
    )
    write_json(output / "manifest.json", {
        "experiment_id": f"stage12-{dataset}-{setting}-seed{seed}",
        "status": "ready_for_one_shot_test",
        "claim_scope": "independent-external-confirmation-pretest",
        "dataset": dataset,
        "setting": setting,
        "seed": seed,
        "configuration": run_config,
        "pretest_freeze": pretest,
        "artifacts": [
            {"path": name, "sha256": sha256_file(output / name)}
            for name in (
                "config.json", "train_manifest.csv", "val_manifest.csv", "known_test_manifest.csv",
                "unknown_test_manifest.csv", "training_log.csv", "training_log.jsonl", "model_best.pt",
                "checkpoint.sha256", "native_threshold.json", "des_threshold.json", "thresholds.json",
                "des_centroids.npy", "des_centroids.sha256", "known_train_validation_features.npz",
                "pretest_freeze.json", "RESULTS.md",
            )
        ],
        "elapsed_seconds": time.time() - started,
        "limitations": ["test remains unopened", "application labels inherit controlled-capture filenames"],
    })
    ready.write_text("READY_FOR_ONE_SHOT_TEST\n", encoding="utf-8")
    return {"status": "READY_FOR_ONE_SHOT_TEST", "output": str(output), "best_epoch": best_epoch}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("iscx_vpn", "iscx_tor"), required=True)
    parser.add_argument("--setting", choices=("low", "medium", "high"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    output = STAGE_ROOT / "runs" / args.dataset / args.setting / f"seed{args.seed}"
    try:
        result = run(args.dataset, args.setting, args.seed)
    except BaseException as exc:
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "FAILURE.json", {
            "status": "FAILED_PRETEST",
            "dataset": args.dataset,
            "setting": args.setting,
            "seed": args.seed,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "preserve_and_do_not_overwrite": True,
            "test_metrics_computed": False,
        })
        raise
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
