#!/usr/bin/env python3
"""Stage20-matched RoNeTC closed-set runner using the frozen Stage19 recipe.

Only Known Train/Validation are touched before checkpoint selection. This is
the existing project's runnable reconstruction, not an author-exact claim.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


OUT = Path(__file__).resolve().parents[1]
PROJECT = OUT.parent
STAGE19 = PROJECT / "stage19_ronetc_four_dataset_comparison"
STAGE20 = PROJECT / "stage20_dual_coarse_service_protocol"
STAGE22 = PROJECT / "stage22_pretrained_trafficformer_e3_closed_set_comparison"
MANIFEST = STAGE20 / "closed_service_manifest.csv"
CACHE = OUT / "ronetc_stage20_cache"
FROZEN_HASH = "6404d496f50a6f7f6f397d125ad0cd653c9df152d8db1b9ed2ad2310967950bb"
sys.path.insert(0, str(STAGE19 / "scripts"))
from train_eval import (  # noqa: E402
    DenseRoNeTCAdapter, FrozenByteDataset, MobileVitClassifier,
    classification_metrics, install_project_local_short_tmp, parity_check,
    predict, train_epoch,
)
from common import RONETC, seed_everything  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def manifest_rows(dataset: str, role: str) -> list[dict[str, str]]:
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["dataset"] == dataset and row["closed_role"] == role]
    rows.sort(key=lambda row: row["flow_id"])
    if not rows or len({row["flow_id"] for row in rows}) != len(rows):
        raise RuntimeError(f"empty or duplicate Stage20 role: {dataset}/{role}")
    return rows


def prepare_role(dataset: str, role: str, services: list[str], index: dict[str, int],
                 batch_size: int, workers: int, seed: int) -> tuple[DataLoader, list[dict[str, str]]]:
    rows = manifest_rows(dataset, role)
    if {row["service_label"] for row in rows} != set(services):
        raise RuntimeError(f"class coverage mismatch: {dataset}/{role}")
    if not all(row["flow_id"] in index for row in rows):
        raise RuntimeError(f"RoNeTC cache does not cover all {dataset}/{role} flows")
    positions = np.asarray([index[row["flow_id"]] for row in rows], dtype=np.int64)
    labels = np.asarray([services.index(row["service_label"]) for row in rows], dtype=np.int64)
    dataset_obj = FrozenByteDataset(CACHE / dataset / "views.npy", positions, labels,
                                    [row["flow_id"] for row in rows])
    kwargs = {
        "batch_size": batch_size, "shuffle": role == "known_train",
        "num_workers": workers, "pin_memory": True, "persistent_workers": workers > 0,
    }
    if role == "known_train":
        kwargs["generator"] = torch.Generator().manual_seed(seed)
    return DataLoader(dataset_obj, **kwargs), rows


def write_predictions(path: Path, dataset: str, seed: int, services: list[str],
                      by_role: list[tuple[list[dict[str, str]], dict]]) -> None:
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "dataset", "seed", "role", "method", "flow_id", "true_service",
            "predicted_service", "correct",
        ])
        writer.writeheader()
        for rows, result in by_role:
            if result["flow_ids"] != [row["flow_id"] for row in rows]:
                raise RuntimeError("prediction order/flow ID mismatch")
            for uid, label, predicted in zip(result["flow_ids"], result["labels"], result["predictions"]):
                writer.writerow({
                    "dataset": dataset, "seed": seed, "role": result["role"],
                    "method": "RoNeTC-runnable-reconstruction", "flow_id": uid,
                    "true_service": services[int(label)],
                    "predicted_service": services[int(predicted)],
                    "correct": int(label == predicted),
                })


def execute(dataset: str, seed: int, smoke: bool, run_tag: str) -> None:
    if sha256(MANIFEST) != FROZEN_HASH:
        raise RuntimeError("Stage20 manifest hash changed")
    frozen = read_json(STAGE22 / "config.json")
    if dataset not in frozen["datasets"] or seed not in frozen["seeds"]:
        raise RuntimeError("dataset/seed outside paired Stage22 scope")
    services = frozen["datasets"][dataset]["services"]
    prior = read_json(STAGE19 / "config.json")
    cfg = prior["training"]
    view_audit = read_json(CACHE / dataset / "cache_audit.json")
    if view_audit["status"] != "PASS" or view_audit["stage20_manifest_sha256"] != FROZEN_HASH:
        raise RuntimeError("Stage23 exact RoNeTC cache audit failed")
    if sha256(CACHE / dataset / "views.npy") != view_audit["views_sha256"]:
        raise RuntimeError("Stage23 RoNeTC view cache hash changed")
    uid_path = CACHE / dataset / "flow_uids.npy"
    if sha256(uid_path) != view_audit["flow_uids_sha256"]:
        raise RuntimeError("Stage23 RoNeTC ID cache hash changed")
    ids = [str(value) for value in np.load(uid_path, mmap_mode="r", allow_pickle=False)]
    index = {uid: offset for offset, uid in enumerate(ids)}
    if len(index) != len(ids) or len(index) != frozen["datasets"][dataset]["expected_flows"]:
        raise RuntimeError("RoNeTC cache membership invalid")
    if not torch.cuda.is_available():
        raise RuntimeError("formal RoNeTC requires CUDA")
    label = "smoke" if smoke else "formal"
    if run_tag and not all(char.isalnum() or char in "_-" for char in run_tag):
        raise ValueError("run-tag must contain only alphanumeric, underscore or hyphen")
    suffix = f"_{run_tag}" if run_tag else ""
    run = OUT / "runs" / "ronetc_stage20" / dataset / f"seed{seed}_{label}{suffix}"
    if run.exists():
        raise FileExistsError(f"preserve existing run: {run}")
    run.mkdir(parents=True)
    started = time.time()
    try:
        temp_alias, temp_directory_fd = install_project_local_short_tmp(run)
        seed_everything(seed)
        torch.set_num_threads(4)
        device = torch.device("cuda:0")
        train_loader, train_rows = prepare_role(dataset, "known_train", services, index,
                                                int(cfg["batch_size"]), int(cfg["workers"]), seed)
        val_loader, val_rows = prepare_role(dataset, "known_validation", services, index,
                                            int(cfg["eval_batch_size"]), int(cfg["workers"]), seed)
        expected = frozen["datasets"][dataset]
        if len(train_rows) != expected["expected_train"] or len(val_rows) != expected["expected_validation"]:
            raise RuntimeError("Stage20 Train/Validation sample count mismatch")
        write_json(run / "input_preflight.json", {
            "status": "PASS", "dataset": dataset, "seed": seed,
            "known_train": len(train_rows), "known_validation": len(val_rows),
            "known_test_features_loaded_before_selection": 0,
            "unknown_features_loaded": 0, "stage20_manifest_sha256": FROZEN_HASH,
            "view_cache_sha256": view_audit["views_sha256"],
            "project_local_tmp_alias": temp_alias,
            "project_local_tmp_directory_fd": temp_directory_fd,
        })
        source_model = Path(RONETC) / "model.py"
        write_json(run / "config.json", {
            "dataset": dataset, "seed": seed, "method": "RoNeTC-runnable-reconstruction",
            "services": services, "source_stage19_config_sha256": sha256(STAGE19 / "config.json"),
            "source_model_sha256": sha256(source_model), "training": cfg,
            "checkpoint_selection": "Known Validation accuracy only",
            "stage20_manifest_sha256": FROZEN_HASH,
            "unknown_training_samples": 0, "unknown_validation_samples": 0,
            "test_selection_samples": 0,
        })
        core = MobileVitClassifier(
            len(services), 3, int(cfg["lambda_epochs"]), packet_num=int(prior["input"]["packet_num"])
        ).to(device)
        adapter_parity = parity_check(core, device, 8, 64)
        write_json(run / "adapter_parity.json", adapter_parity)
        model = DenseRoNeTCAdapter(core).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=float(cfg["learning_rate"]),
            betas=tuple(cfg["betas"]), weight_decay=float(cfg["weight_decay"])
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=list(cfg["scheduler_milestones"]),
            gamma=float(cfg["scheduler_gamma"])
        )
        torch.cuda.reset_peak_memory_stats(device)
        if smoke:
            values, labels, _uid = next(iter(train_loader))
            values, labels = values.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            fused, uncertainty, loss_terms = model(values, labels, 1)
            loss = loss_terms.mean()
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite RoNeTC smoke loss")
            loss.backward()
            optimizer.step()
            write_json(run / "smoke_result.json", {
                "status": "PASS", "dataset": dataset, "seed": seed,
                "batch": len(labels), "loss": float(loss.item()),
                "finite_fused": bool(torch.isfinite(fused).all()),
                "finite_uncertainty": bool(torch.isfinite(uncertainty).all()),
                "adapter_parity": adapter_parity,
                "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
                "known_test_features_loaded": 0,
            })
            (run / "SUCCESS").write_text("SUCCESS\n", encoding="utf-8")
            print(json.dumps({"status": "PASS_SMOKE", "dataset": dataset, "seed": seed}), flush=True)
            return

        best_acc, best_epoch = -1.0, -1
        checkpoint = run / "model_best.pt"
        with (run / "training_history.jsonl").open("x", encoding="utf-8") as handle:
            for epoch in range(1, int(cfg["epochs"]) + 1):
                train_loss, train_acc = train_epoch(model, train_loader, optimizer, device, epoch)
                val_output = predict(model, val_loader, device, "known_validation")
                val_metrics = classification_metrics(val_output["labels"], val_output["predictions"], len(services))
                lr = float(optimizer.param_groups[0]["lr"])
                if val_metrics["accuracy"] > best_acc:
                    best_acc, best_epoch = float(val_metrics["accuracy"]), epoch
                    torch.save({
                        "model_state_dict": core.state_dict(), "epoch": epoch,
                        "validation_accuracy": best_acc, "services": services,
                        "seed": seed, "source_model_sha256": sha256(source_model),
                    }, checkpoint)
                handle.write(json.dumps({
                    "epoch": epoch, "lr": lr, "train_loss": train_loss,
                    "train_accuracy": train_acc, "validation_accuracy": val_metrics["accuracy"],
                    "validation_macro_f1": val_metrics["macro_f1"],
                    "validation_weighted_f1": val_metrics["weighted_f1"],
                    "best_epoch": best_epoch, "best_validation_accuracy": best_acc,
                    "elapsed_seconds": time.time() - started,
                }) + "\n")
                handle.flush()
                print(f"{dataset} seed={seed} epoch={epoch} val_acc={val_metrics['accuracy']:.6f} best={best_acc:.6f}", flush=True)
                scheduler.step()
        checkpoint_payload = torch.load(checkpoint, map_location=device, weights_only=False)
        core.load_state_dict(checkpoint_payload["model_state_dict"])
        val_output = predict(model, val_loader, device, "known_validation")
        val_metrics = classification_metrics(val_output["labels"], val_output["predictions"], len(services))
        if abs(val_metrics["accuracy"] - best_acc) > 1e-12:
            raise RuntimeError("best checkpoint validation accuracy parity failed")

        # Load Known Test only after the checkpoint has been selected and restored.
        test_loader, test_rows = prepare_role(dataset, "known_test", services, index,
                                              int(cfg["eval_batch_size"]), int(cfg["workers"]), seed)
        if len(test_rows) != expected["expected_test"]:
            raise RuntimeError("Stage20 Known Test count mismatch")
        test_output = predict(model, test_loader, device, "known_test")
        test_metrics = classification_metrics(test_output["labels"], test_output["predictions"], len(services))
        prediction_path = run / "predictions.csv"
        write_predictions(prediction_path, dataset, seed, services,
                          [(val_rows, val_output), (test_rows, test_output)])
        write_json(run / "metrics.json", {
            "dataset": dataset, "seed": seed, "method": "RoNeTC-runnable-reconstruction",
            "best_epoch": best_epoch, "completed_epochs": int(cfg["epochs"]),
            "validation": val_metrics, "test": test_metrics,
            "checkpoint_sha256": sha256(checkpoint),
            "predictions_sha256": sha256(prediction_path),
            "source_stage19_config_sha256": sha256(STAGE19 / "config.json"),
            "source_model_sha256": sha256(source_model),
            "stage20_manifest_sha256": FROZEN_HASH,
            "view_cache_sha256": view_audit["views_sha256"],
            "runtime_seconds": time.time() - started,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
            "unknown_training_samples": 0, "unknown_validation_samples": 0,
            "test_selection_samples": 0,
        })
        if sha256(MANIFEST) != FROZEN_HASH:
            raise RuntimeError("Stage20 manifest changed during training")
        (run / "SUCCESS").write_text("SUCCESS\n", encoding="utf-8")
        print(json.dumps({"status": "SUCCESS", "dataset": dataset, "seed": seed,
                          "test": {key: test_metrics[key] for key in ("accuracy", "macro_f1", "weighted_f1")}}), flush=True)
    except BaseException as exc:
        write_json(run / "FAILURE.json", {
            "status": "FAILED", "error_type": type(exc).__name__,
            "error": str(exc), "traceback": traceback.format_exc(),
            "preserve_partial_artifacts": True,
        })
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("iscx_vpn", "iscx_tor"), required=True)
    parser.add_argument("--seed", type=int, choices=(2022, 2023), required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--run-tag", default="")
    args = parser.parse_args()
    execute(args.dataset, args.seed, args.smoke, args.run_tag)
