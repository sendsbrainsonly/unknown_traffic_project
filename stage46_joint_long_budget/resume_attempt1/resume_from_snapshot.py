#!/usr/bin/env python3
"""Continue a Stage46 run from a verified epoch-boundary recovery snapshot.

The frozen Stage46 training script is imported unchanged. Outputs live in this
new attempt directory; original interrupted files are never overwritten.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
STAGE = HERE.parent
PROJECT = STAGE.parent
sys.path.insert(0, str(STAGE))
import experiment as exp  # noqa: E402


def atomic_checkpoint(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".partial")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def recovery_payload(model, opts, scheduler, scaler, loader, epoch, step,
                     best_epoch, best_value, best_state, config, source_snapshot):
    return {
        "state_dict": model.state_dict(),
        "epoch": epoch,
        "step": step,
        "optimizers": {name: opt.state_dict() for name, opt in opts.items()},
        "scheduler": scheduler.state_dict(),
        "grad_scaler": scaler.state_dict(),
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "cpu_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state(),
        "loader_generator": loader.generator.get_state(),
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_value,
        "best_state_dict": best_state,
        "config": config,
        "source_recovery_checkpoint": str(source_snapshot.relative_to(PROJECT)),
        "resume_boundary": "completed_epoch",
    }


def build(dataset: str, device: torch.device):
    exp.assert_locked()
    train_n, val_n, _ = exp.source.EXPECTED[dataset]
    scale = np.load(STAGE / "inputs" / dataset / "graph_scaler.npz", allow_pickle=False)
    train_ds = exp.dataset_for(dataset, "known_train", scale["mean"], scale["std"])
    val_ds = exp.dataset_for(dataset, "known_validation", scale["mean"], scale["std"])
    if (len(train_ds), len(val_ds)) != (train_n, val_n):
        raise RuntimeError("Known Train/Val count changed")
    if set(train_ds.ids) & set(val_ds.ids):
        raise RuntimeError("Known Train/Val flow IDs overlap")

    total_steps = math.ceil(len(train_ds) / exp.BATCH) * exp.EPOCHS
    model = exp.joint.JointThreeView(exp.source.CLASSES[dataset], total_steps).to(device)
    tf_opt, tf_scheduler = exp.joint.tf_branch.source.load_tf_module().build_optimizer(model.tf_args, model.tf)
    yatc_groups = exp.joint.yatc_branch.source.lrd.param_groups_lrd(
        model.yatc, .05, no_weight_decay_list=model.yatc.no_weight_decay(), layer_decay=.75)
    fusion_module = nn.ModuleList([model.adapters, model.head])
    modules = {"trafficformer": model.tf, "graph": model.graph, "yatc": model.yatc, "fusion": fusion_module}
    opts = {
        "trafficformer": tf_opt,
        "graph": torch.optim.Adam(model.graph.parameters(), lr=1e-3),
        "yatc": torch.optim.AdamW(yatc_groups, lr=5e-4),
        "fusion": torch.optim.Adam(fusion_module.parameters(), lr=1e-3),
    }
    scaler = torch.cuda.amp.GradScaler()
    train_loader = exp.joint.load_batches(exp.Indexed(train_ds), exp.MICRO, True)
    val_loader = exp.joint.load_batches(val_ds, exp.MICRO, False)
    return train_ds, val_ds, model, opts, tf_scheduler, scaler, train_loader, val_loader, modules, total_steps


def restore(dataset: str, snapshot_path: Path, device: torch.device):
    snapshot = torch.load(snapshot_path, map_location="cpu", weights_only=False, mmap=True)
    built = build(dataset, device)
    train_ds, val_ds, model, opts, scheduler, scaler, loader, val_loader, modules, total_steps = built
    expected = {
        "dataset": dataset, "epochs": exp.EPOCHS, "prefixes": exp.PREFIXES,
        "effective_batch": exp.BATCH, "microbatch": exp.MICRO,
        "total_optimizer_steps": total_steps,
        "train_samples": len(train_ds), "val_samples": len(val_ds),
        "classes": exp.source.CLASSES[dataset],
    }
    for key, value in expected.items():
        if snapshot["config"].get(key) != value:
            raise RuntimeError(f"snapshot configuration mismatch for {key}")
    expected_step = int(snapshot["epoch"]) * math.ceil(len(train_ds) / exp.BATCH)
    if int(snapshot["step"]) != expected_step:
        raise RuntimeError("snapshot optimizer-step count does not match its completed epoch")
    model.load_state_dict(snapshot["state_dict"], strict=True)
    for name, optimizer in opts.items():
        optimizer.load_state_dict(snapshot["optimizers"][name])
    scheduler.load_state_dict(snapshot["scheduler"])
    scaler.load_state_dict(snapshot["grad_scaler"])
    loader.generator.set_state(snapshot["loader_generator"])
    random.setstate(snapshot["python_rng"])
    np.random.set_state(snapshot["numpy_rng"])
    torch.set_rng_state(snapshot["cpu_rng"])
    torch.cuda.set_rng_state(snapshot["cuda_rng"], device=device)
    return snapshot, built


def smoke(dataset: str, snapshot_path: Path) -> None:
    exp.assert_locked()
    if os.environ.get("CONDA_PREFIX") != exp.ENV or sys.prefix != exp.ENV:
        raise RuntimeError("resume smoke is outside the fixed project environment")
    exp.joint.seed_all()
    device = torch.device("cuda:0")
    snapshot, built = restore(dataset, snapshot_path, device)
    train_ds, _, model, opts, scheduler, scaler, loader, _, modules, _ = built
    group = []
    for batch, _ in loader:
        group.append(exp.joint.to_device(batch, device))
        if len(group) == 2:
            break
    if len(group) != 2:
        raise RuntimeError("could not obtain the two microbatches required for one atomic update")
    step = int(snapshot["step"])
    exp.joint.set_yatc_lr(opts["yatc"], step, int(snapshot["config"]["total_optimizer_steps"]))
    value, size, truth, pred, norms = exp.support.atomic_step(
        model, group, True, opts, modules, scaler,
        lambda event: None)
    scheduler.step()
    if not np.isfinite(value) or size <= 0 or not all(np.isfinite(list(norms.values()))):
        raise RuntimeError("resumed one-step smoke produced a nonfinite result")
    target = HERE / "smoke" / f"{dataset}.json"
    if target.exists():
        raise FileExistsError(target)
    exp.write(target, {
        "status": "PASS", "dataset": dataset, "device": str(device),
        "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "source_checkpoint": str(snapshot_path.relative_to(PROJECT)),
        "start_epoch": int(snapshot["epoch"]), "start_step": step,
        "one_atomic_update_completed": True, "batch_samples": int(size),
        "loss": float(value) / int(size), "gradient_norms": norms,
        "known_train_samples_only": int(size), "known_validation_used": 0,
        "known_test_used": 0, "unknown_used": 0,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device),
    })
    print(f"RESUME_LOAD_SMOKE_PASS {dataset} update_from_step={step}", flush=True)


def train(dataset: str, snapshot_path: Path, output_dir: Path) -> None:
    exp.assert_locked()
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse continuation output directory: {output_dir}")
    if not snapshot_path.is_file():
        raise FileNotFoundError(snapshot_path)
    output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda:0")
    model = opts = scheduler = scaler = loader = None
    epoch = step = 0
    try:
        exp.joint.seed_all()
        snapshot = torch.load(snapshot_path, map_location="cpu", weights_only=False, mmap=True)
        expected_config = {
            "dataset": dataset, "epochs": exp.EPOCHS, "prefixes": exp.PREFIXES,
            "effective_batch": exp.BATCH, "microbatch": exp.MICRO,
        }
        for key, value in expected_config.items():
            if snapshot["config"].get(key) != value:
                raise RuntimeError(f"snapshot configuration mismatch for {key}")
        start_epoch = int(snapshot["epoch"])
        expected_step = start_epoch * math.ceil(len(exp.source.rows_for(dataset, "known_train")) / exp.BATCH)
        if start_epoch not in (75, 85, 95) or int(snapshot["step"]) != expected_step:
            raise RuntimeError(f"snapshot is not a supported complete-epoch boundary: epoch={start_epoch}, step={snapshot['step']}")

        (train_ds, val_ds, model, opts, scheduler, scaler, loader, val_loader,
         modules, total_steps) = build(dataset, device)
        model.load_state_dict(snapshot["state_dict"], strict=True)
        for name, optimizer in opts.items():
            optimizer.load_state_dict(snapshot["optimizers"][name])
        scheduler.load_state_dict(snapshot["scheduler"])
        scaler.load_state_dict(snapshot["grad_scaler"])
        loader.generator.set_state(snapshot["loader_generator"])
        step = int(snapshot["step"])
        best_epoch = int(snapshot["best_epoch"])
        best_value = float(snapshot["best_val_macro_f1"])
        best_state = snapshot["best_state_dict"]
        if best_epoch < 1 or best_epoch > start_epoch or best_state is None:
            raise RuntimeError("snapshot lacks a valid Known-Val best checkpoint state")

        # Restore RNG only after model/optimizer/data-loader construction.
        random.setstate(snapshot["python_rng"])
        np.random.set_state(snapshot["numpy_rng"])
        torch.set_rng_state(snapshot["cpu_rng"])
        torch.cuda.set_rng_state(snapshot["cuda_rng"], device=device)

        config = dict(snapshot["config"])
        config.update({"resume_from_epoch": start_epoch,
                       "resume_source_checkpoint": str(snapshot_path.relative_to(PROJECT)),
                       "resume_attempt": "stage46_resume_attempt1"})
        (output_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")
        history_path = output_dir / "training_history.jsonl"
        numeric_path = output_dir / "numerical_events.jsonl"
        started = time.time()
        events = []
        torch.cuda.reset_peak_memory_stats(device)
        with history_path.open("x") as history, numeric_path.open("x") as numeric:
            for epoch in range(start_epoch + 1, exp.EPOCHS + 1):
                t0 = time.time()
                model.train()
                seen_indices, group, online_y, online_p = [], [], [], []
                loss_sum = seen = 0
                norm_sums = {key: 0.0 for key in modules}
                n_groups = 0

                def numeric_event(event):
                    numeric.write(json.dumps({"epoch": epoch, "step": step, **event}) + "\n")
                    numeric.flush()

                for i, (batch, indices) in enumerate(loader):
                    seen_indices.extend(indices.tolist())
                    group.append(exp.joint.to_device(batch, device))
                    if len(group) != 2 and i + 1 != len(loader):
                        continue
                    exp.joint.set_yatc_lr(opts["yatc"], step, total_steps)
                    value, size, truth, pred, norms = exp.support.atomic_step(
                        model, group, True, opts, modules, scaler, numeric_event)
                    if not all(np.isfinite(list(norms.values()))):
                        raise RuntimeError("nonfinite gradient norm in resumed epoch")
                    scheduler.step()
                    step += 1
                    loss_sum += value
                    seen += size
                    online_y.extend(truth)
                    online_p.extend(pred)
                    for key, norm in norms.items():
                        norm_sums[key] += norm
                    n_groups += 1
                    group.clear()
                    if n_groups % 25 == 0:
                        exp.write(output_dir / "batch_progress.json", {
                            "dataset": dataset, "epoch": epoch, "seen": seen,
                            "train_samples": len(train_ds), "optimizer_steps": step,
                            "updated_at_utc": exp.now(),
                        })
                if seen != len(train_ds) or sorted(seen_indices) != list(range(len(train_ds))):
                    raise RuntimeError("resumed epoch missing/duplicate KnownTrain identities")

                order_path = output_dir / f"order_epoch{epoch:03}.npy"
                np.save(order_path, np.asarray(seen_indices, dtype=np.int64), allow_pickle=False)
                reference_order = exp.run_dir(dataset) / f"order_epoch{epoch:03}.npy"
                order_replay = None
                if reference_order.is_file():
                    expected_order = np.load(reference_order, allow_pickle=False)
                    order_replay = bool(np.array_equal(np.asarray(seen_indices), expected_order))
                    if not order_replay:
                        raise RuntimeError(f"replayed sample order differs at epoch {epoch}")

                logits, truth = exp.joint.predict(model, val_loader, device)
                if not np.isfinite(logits).all():
                    raise RuntimeError("nonfinite KnownVal logits")
                val_dir = output_dir / "validation" / f"epoch{epoch:03}"
                val = exp.support.save_predictions(val_dir, dataset, "known_validation", val_ds.ids, truth, logits)
                old_val = exp.run_dir(dataset) / "validation" / f"epoch{epoch:03}"
                replay = {"available": False}
                if old_val.is_dir() and (old_val / "logits.npy").is_file():
                    old_logits = np.load(old_val / "logits.npy", allow_pickle=False)
                    max_abs = float(np.max(np.abs(logits - old_logits)))
                    replay = {"available": True, "max_abs_logit_difference": max_abs,
                              "allclose_atol1e-4_rtol1e-4": bool(np.allclose(logits, old_logits, atol=1e-4, rtol=1e-4))}

                online = exp.joint.metrics(np.asarray(online_y), np.asarray(online_p), exp.source.CLASSES[dataset])
                if val["macro_f1"] > best_value:
                    best_value, best_epoch, best_state = val["macro_f1"], epoch, exp.cpu_state(model)
                event = {
                    "dataset": dataset, "epoch": epoch, "train_loss": loss_sum / seen,
                    **{f"train_online_{key}": online[key] for key in ("accuracy", "macro_f1", "weighted_f1")},
                    **{f"val_{key}": val[key] for key in ("loss", "accuracy", "macro_f1", "weighted_f1")},
                    "optimizer_steps": step, "train_seen": seen,
                    "order_sha256": exp.sha(order_path), "replayed_reference_order_exact": order_replay,
                    "reference_validation_replay": replay,
                    "gradient_norm_means": {key: value / n_groups for key, value in norm_sums.items()},
                    "lr_groups_last_step": {key: [float(g["lr"]) for g in opt.param_groups] for key, opt in opts.items()},
                    "grad_scale": float(scaler.get_scale()), "epoch_seconds": time.time() - t0,
                    "elapsed_seconds": time.time() - started,
                    "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device),
                }
                event.update(best_epoch=best_epoch, best_val_macro_f1=best_value)
                events.append(event)
                history.write(json.dumps(event) + "\n")
                history.flush()
                exp.write(output_dir / "progress.json", event)
                print(json.dumps({key: event[key] for key in (
                    "dataset", "epoch", "train_loss", "val_loss", "val_macro_f1", "best_epoch", "elapsed_seconds")}), flush=True)

                if epoch in (85, 95, 100):
                    payload = recovery_payload(model, opts, scheduler, scaler, loader, epoch, step,
                                               best_epoch, best_value, best_state, config, snapshot_path)
                    exp.save_torch(output_dir / f"recovery_epoch{epoch:03}.pt", payload)

        if step != total_steps:
            raise RuntimeError(f"resumed optimizer-step total mismatch {step} != {total_steps}")
        if not events or events[-1]["epoch"] != exp.EPOCHS:
            raise RuntimeError("resumed training did not reach epoch100")

        checkpoint_path = output_dir / "prefix100_best.pt"
        exp.save_torch(checkpoint_path, {"state_dict": best_state, "config": config,
                                        "budget_epochs": 100, "best_epoch": best_epoch,
                                        "best_val_macro_f1": best_value})
        prefix = {"dataset": dataset, "budget_epochs": 100, "best_epoch": best_epoch,
                  "best_val_macro_f1": best_value,
                  "checkpoint_path": str(checkpoint_path.relative_to(STAGE)),
                  "checkpoint_sha256": exp.sha(checkpoint_path)}
        exp.write(output_dir / "prefix100_selection.json", prefix)
        exp.write(output_dir / "resume_summary.json", {
            "status": "PASS", "dataset": dataset, "completed_epochs_total": 100,
            "resumed_epochs": [start_epoch + 1, exp.EPOCHS], "optimizer_steps_total": step,
            "checkpoint": prefix, "source_recovery_checkpoint": str(snapshot_path.relative_to(PROJECT)),
            "known_test_used": 0, "unknown_used": 0,
            "sample_order_parity_through_existing_epochs": all(e["replayed_reference_order_exact"] for e in events if e["replayed_reference_order_exact"] is not None),
            "reference_validation_logit_parity": [e["reference_validation_replay"] for e in events if e["reference_validation_replay"]["available"]],
        })

        # Existing 20/50 checkpoints are retained; create the standard Stage46
        # summary only after this resumed 100-epoch prefix has completed.
        old20 = exp.read(exp.run_dir(dataset) / "prefix020_selection.json")
        old50 = exp.read(exp.run_dir(dataset) / "prefix050_selection.json")
        for old in (old20, old50):
            checkpoint = STAGE / old["checkpoint_path"]
            if exp.sha(checkpoint) != old["checkpoint_sha256"]:
                raise RuntimeError("existing Stage46 20/50 checkpoint hash changed")
        summary_path = exp.run_dir(dataset) / "train_summary.json"
        if summary_path.exists():
            raise FileExistsError(summary_path)
        exp.write(summary_path, {
            "status": "PASS", "completed_epochs": 100, "optimizer_steps": step,
            "prefixes": [old20, old50, prefix],
            "resume_provenance": str(output_dir.relative_to(STAGE)),
            "source_snapshot": str(snapshot_path.relative_to(STAGE)),
            "known_test_used_during_train": 0, "unknown_used": 0,
        })
        exp.assert_locked()
        print(f"RESUME_TRAIN_PASS {dataset} completed_epochs=100 best_epoch={best_epoch} best_val_macro_f1={best_value:.6f}", flush=True)
    except BaseException as exc:
        exp.write(output_dir / "FAILURE.json", {"error": repr(exc), "traceback": traceback.format_exc(),
                                                 "epoch": epoch, "optimizer_steps": step,
                                                 "safe_resume_boundary": str(sorted(output_dir.glob("recovery_epoch*.pt"))[-1])
                                                 if list(output_dir.glob("recovery_epoch*.pt")) else str(snapshot_path)})
        if model is not None and opts is not None and scheduler is not None and scaler is not None and loader is not None:
            try:
                failure = recovery_payload(model, opts, scheduler, scaler, loader, epoch, step,
                                           int(snapshot.get("best_epoch", 0)),
                                           float(snapshot.get("best_val_macro_f1", -1)),
                                           snapshot.get("best_state_dict"), snapshot.get("config", {}), snapshot_path)
                exp.save_torch(output_dir / "failure_state.pt", failure)
            except BaseException as save_exc:
                exp.write(output_dir / "FAILURE_SAVE_ERROR.json", {"error": repr(save_exc)})
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=exp.DATASETS, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    snapshot_path = args.snapshot.resolve()
    if args.smoke_only:
        smoke(args.dataset, snapshot_path)
    else:
        if args.output_dir is None:
            parser.error("--output-dir required unless --smoke-only is used")
        train(args.dataset, snapshot_path, args.output_dir.resolve())


if __name__ == "__main__":
    main()
