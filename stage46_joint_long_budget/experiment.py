#!/usr/bin/env python3
"""Predeclared long-horizon joint training; historical sources remain read-only."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
ENV = "/home/birkenwald/data/SEU-WXY/conda_envs/2025-10-8-WXY-dgl_py310"
EPOCHS, SEED, MICRO, BATCH = 100, 2022, 32, 64
PREFIXES = (20, 50, 100)
DATASETS = ("iscx_vpn", "iscx_tor")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


source = load_module("stage46_original_inputs", PROJECT / "stage37_joint_vpn_tor_single_seed/run_joint.py")
support = load_module("stage46_atomic_support", PROJECT / "stage45_staged_vs_joint_finetuning/runner.py")
joint = source.joint
read, write, csv_write, sha = support.read, support.write, support.csv_write, support.sha


def now():
    return support.now()


def run_dir(dataset):
    return ROOT / "runs" / dataset


def historical_run(dataset):
    return source.HERE / "runs" / dataset / "seed2022_joint_e2e20"


def select_best(events, budget):
    eligible = [e for e in events if 1 <= e["epoch"] <= budget]
    if not eligible:
        raise ValueError("no eligible trained epoch")
    return max(eligible, key=lambda e: (e["val_macro_f1"], -e["epoch"]))


def assert_locked():
    lock = read(ROOT / "frozen_sources.json")
    for rel, expected in lock["sha256"].items():
        if sha(PROJECT / rel) != expected:
            raise RuntimeError(f"source/code drift: {rel}")
    return lock


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def save_torch(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(value, temporary)
    temporary.replace(path)


def dataset_for(dataset, role, mean, std):
    if role == "known_test":
        assert_test_gate()
    elif role not in ("known_train", "known_validation"):
        raise ValueError("Unknown/unregistered role forbidden")
    return source.Flows(dataset, role, mean, std)


def prepare():
    if (ROOT / "preflight.json").exists():
        raise FileExistsError("preflight already registered")
    if os.environ.get("CONDA_PREFIX") != ENV or sys.prefix != ENV:
        raise RuntimeError("wrong Python environment")
    paths = set()
    audits = []
    # Reuse original Known-only data and hash validation, not an alternative split.
    for dataset in DATASETS:
        hashes = source.guards(dataset)
        old = read(historical_run(dataset) / "config.json")
        if hashes != old["source_hashes"]:
            raise RuntimeError(f"historical Stage37 source mismatch: {dataset}")
        mean, std = source.graph_scale(dataset)
        train = dataset_for(dataset, "known_train", mean, std)
        val = dataset_for(dataset, "known_validation", mean, std)
        if (len(train), len(val)) != source.EXPECTED[dataset][:2] or set(train.ids) & set(val.ids):
            raise RuntimeError("Known Train/Val count or identity failed")
        folder = ROOT / "inputs" / dataset
        folder.mkdir(parents=True, exist_ok=False)
        np.savez(folder / "graph_scaler.npz", mean=mean, std=std)
        membership = [{"flow_id": uid, "role": role, "class": source.CLASSES[dataset][int(y)]}
                      for role, values in (("known_train", train), ("known_validation", val))
                      for uid, y in zip(values.ids, values.labels, strict=True)]
        csv_write(folder / "train_val_membership.csv", membership)
        args = joint.tf_branch.source.make_tf_args(len(source.CLASSES[dataset]), math.ceil(len(train) / BATCH) * EPOCHS)
        if (args.optimizer, args.scheduler, args.warmup, args.learning_rate) != ("adamw", "linear", .1, 6e-5):
            raise RuntimeError("TF actual optimizer/schedule differs from registered plan")
        audits.append({"dataset": dataset, "samples": {"known_train": len(train), "known_validation": len(val)},
                       "class_counts": {role: dict(Counter(source.CLASSES[dataset][int(y)] for y in values.labels))
                                        for role, values in (("known_train", train), ("known_validation", val))},
                       "source_hashes": hashes, "tf_args": {key: getattr(args, key) for key in
                            ("optimizer", "scheduler", "warmup", "learning_rate", "train_steps")},
                       "membership_sha256": sha(folder / "train_val_membership.csv"),
                       "test_values_used": 0, "unknown_used": 0})
        paths.update([source.MANIFEST, source.INPUTS / dataset / "e3_t8_inputs.npz",
                      source.MFR / dataset / "cache_audit_trainval.json", historical_run(dataset) / "config.json",
                      historical_run(dataset) / "train_summary.json", historical_run(dataset) / "known_train_graph_scaler.npz",
                      historical_run(dataset) / "known_train_initial_feature_scalers.npz"])
        for role in ("known_train", "known_validation"):
            paths.update(source.MFR / dataset / f"{role}_{suffix}.npy" for suffix in ("mfr", "flow_ids"))
        paths.update(folder.iterdir())
    paths.update([joint.tf_branch.source.PRETRAINED_MODEL, joint.yatc_branch.source.WEIGHT])
    paths.update(PROJECT / rel for rel in (
        "stage35_joint_vs_staged_training/joint_ustc.py", "stage37_joint_vpn_tor_single_seed/run_joint.py",
        "stage45_staged_vs_joint_finetuning/runner.py", "stage32_three_dataset_coarse_single_seed/common.py",
        "stage31_four_dataset_three_view_equal/train_equal_fusion.py", "stage31_four_dataset_three_view_equal/train_tf_fig_branch.py",
        "stage31_four_dataset_three_view_equal/train_yatc_branch.py", "stage22_pretrained_trafficformer_e3_closed_set_comparison/scripts/train_closed_run.py",
        "stage17_encoder_recovery_and_open_set_pilot/scripts/train_pilot_run.py", "src/stage1/tagcn.py",
        "stage23_closed_set_method_table/scripts/train_yatc_closed.py", "tf_runtime/code/fine-tuning/run_classifier.py"))
    paths.update(ROOT / name for name in ("PLAN.md", "experiment.py", "run_queue.py", "progress.py", "tests.py"))
    write(ROOT / "frozen_sources.json", {"created_at_utc": now(), "sha256": {
        os.path.relpath(p, PROJECT): sha(p) for p in sorted(paths)}})
    write(ROOT / "preflight.json", {"status": "PASS", "datasets": audits, "epochs": EPOCHS,
                                    "prefixes": PREFIXES, "seed": SEED, "environment": ENV})
    manifest = read(ROOT / "manifest.json")
    manifest.update(status="running", inputs=audits, configuration={"files": ["PLAN.md", "preflight.json"],
        "parameters": {"epochs": EPOCHS, "prefixes": PREFIXES, "effective_batch": BATCH, "microbatch": MICRO}, "seeds": [SEED]},
        limitations=["Single seed; previously exposed Test;100epoch warmup/decay horizon differs from historical20.",
                     "Atomic mixed-precision update repair differs from historical per-optimizer skipping; not pure epoch/equal-FLOPs attribution."],
        next_step="Train both100epoch runs, freeze all six selections, evaluate and replay.")
    manifest["execution"].update(command="python stage46_joint_long_budget/run_queue.py", environment=ENV,
                                  tmux_session="stage46_train_20261023")
    write(ROOT / "manifest.json", manifest)
    assert_locked()
    print(json.dumps({"status": "PREPARE_PASS", "datasets": audits}), flush=True)


class Indexed(Dataset):
    """Identity tracing without changing the original RandomSampler RNG policy."""
    def __init__(self, original):
        self.original = original

    def __len__(self):
        return len(self.original)

    def __getitem__(self, index):
        return self.original[index], index


def train(dataset):
    assert_locked()
    run = run_dir(dataset)
    run.mkdir(parents=True, exist_ok=False)
    started = time.time()
    model = opts = scaler = tf_scheduler = train_loader = None
    epoch = step = 0
    try:
        joint.seed_all()
        scales = np.load(ROOT / "inputs" / dataset / "graph_scaler.npz", allow_pickle=False)
        mean, std = scales["mean"], scales["std"]
        train_ds = dataset_for(dataset, "known_train", mean, std)
        val_ds = dataset_for(dataset, "known_validation", mean, std)
        total_steps = math.ceil(len(train_ds) / BATCH) * EPOCHS
        device = torch.device("cuda:0")
        model = joint.JointThreeView(source.CLASSES[dataset], total_steps).to(device)
        initial = cpu_state(model)
        initial_hash = support.state_hash(initial)
        save_torch(run / "initial_model.pt", {"state_dict": initial, "state_sha256": initial_hash,
                    "seed": SEED, "initialization": "same official-pretrained/random policy as Stage37"})
        del initial
        means, stds, count = joint.fit_initial_feature_scalers(model, joint.load_batches(train_ds, MICRO, False), device)
        actual_scales = {f"{name}_{kind}": v for name, m, s in zip(joint.VIEWS, means, stds, strict=True)
                         for kind, v in (("mean", m), ("std", s))}
        np.savez(run / "initial_feature_scalers.npz", **actual_scales)
        with np.load(historical_run(dataset) / "known_train_initial_feature_scalers.npz", allow_pickle=False) as old:
            scaler_diff = {key: {"max_abs_difference": float(np.max(np.abs(value - old[key]))),
                                "allclose_atol2e-4_rtol2e-3": bool(np.allclose(value, old[key], atol=2e-4, rtol=2e-3))}
                           for key, value in actual_scales.items()}
        with np.load(historical_run(dataset) / "known_train_graph_scaler.npz", allow_pickle=False) as old:
            np.testing.assert_array_equal(mean, old["mean"])
            np.testing.assert_array_equal(std, old["std"])
        write(run / "initialization_audit.json", {"initial_state_sha256": initial_hash, "scaler_samples": count,
              "scaler_role": "known_train", "historical_embedding_scaler_comparison": scaler_diff,
              "graph_scaler_exact_parity": True, "tf_pretrained_load": model.tf_load, "yatc_pretrained_load": model.yatc_load})
        if count != len(train_ds) or not all(item["allclose_atol2e-4_rtol2e-3"] for item in scaler_diff.values()):
            raise RuntimeError("initial KnownTrain feature scalers do not reproduce historical initialization")
        tf_opt, tf_scheduler = joint.tf_branch.source.load_tf_module().build_optimizer(model.tf_args, model.tf)
        yg = joint.yatc_branch.source.lrd.param_groups_lrd(model.yatc, .05,
                    no_weight_decay_list=model.yatc.no_weight_decay(), layer_decay=.75)
        fusion_module = nn.ModuleList([model.adapters, model.head])
        modules = {"trafficformer": model.tf, "graph": model.graph, "yatc": model.yatc, "fusion": fusion_module}
        opts = {"trafficformer": tf_opt, "graph": torch.optim.Adam(model.graph.parameters(), lr=1e-3),
                "yatc": torch.optim.AdamW(yg, lr=5e-4), "fusion": torch.optim.Adam(fusion_module.parameters(), lr=1e-3)}
        scaler = torch.cuda.amp.GradScaler()
        train_loader = joint.load_batches(Indexed(train_ds), MICRO, True)
        val_loader = joint.load_batches(val_ds, MICRO, False)
        config = {"dataset": dataset, "seed": SEED, "epochs": EPOCHS, "prefixes": PREFIXES,
                  "total_optimizer_steps": total_steps, "effective_batch": BATCH, "microbatch": MICRO,
                  "initial_state_sha256": initial_hash, "classes": source.CLASSES[dataset],
                  "train_samples": len(train_ds), "val_samples": len(val_ds), "known_test_used_during_train": 0,
                  "unknown_used": 0, "checkpoint_selection": "prefix Known-Val Macro-F1; earliest tie; trainedepochs only",
                  "tf_optimizer": model.tf_args.optimizer, "tf_scheduler": model.tf_args.scheduler,
                  "tf_warmup_fraction": model.tf_args.warmup, "yatc_warmup_fraction": .1,
                  "base_learning_rates": {"trafficformer": 6e-5, "graph": 1e-3, "yatc": 5e-4, "fusion": 1e-3},
                  "grad_scaler": "Stage45 atomic same-batch RNG replay max8backoffs", "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
                  "frozen_sources_sha256": sha(ROOT / "frozen_sources.json")}
        write(run / "config.json", config)
        # Construction/normalization complete before train-loader iteration, just as Stage37.
        best, best_epoch, best_state = -1.0, 0, None
        events, prefixes = [], []
        torch.cuda.reset_peak_memory_stats(device)
        with (run / "training_history.jsonl").open("x") as history, (run / "numerical_events.jsonl").open("x") as numeric:
            for epoch in range(1, EPOCHS + 1):
                t0 = time.time()
                model.train()
                seen_indices, group, online_y, online_p = [], [], [], []
                loss_sum = seen = 0
                norm_sums = {k: 0.0 for k in modules}
                n_groups = 0

                def numeric_event(event):
                    numeric.write(json.dumps({"epoch": epoch, "step": step, **event}) + "\n")
                    numeric.flush()

                for i, (batch, indices) in enumerate(train_loader):
                    seen_indices.extend(indices.tolist())
                    group.append(joint.to_device(batch, device))
                    if len(group) != 2 and i + 1 != len(train_loader):
                        continue
                    joint.set_yatc_lr(opts["yatc"], step, total_steps)
                    value, size, truth, pred, norms = support.atomic_step(model, group, True, opts, modules, scaler, numeric_event)
                    if step == 0 and any(norm <= 0 for norm in norms.values()):
                        raise RuntimeError("first accepted update has a disconnected trainable branch")
                    tf_scheduler.step()
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
                        write(run / "batch_progress.json", {"epoch": epoch, "seen": seen, "train_samples": len(train_ds),
                            "optimizer_steps": step, "elapsed_seconds": time.time() - started, "updated_at_utc": now()})
                if seen != len(train_ds) or sorted(seen_indices) != list(range(len(train_ds))):
                    raise RuntimeError("epoch missing/duplicate KnownTrain identities")
                np.save(run / f"order_epoch{epoch:03}.npy", np.asarray(seen_indices, dtype=np.int64), allow_pickle=False)
                logits, truth = joint.predict(model, val_loader, device)
                if not np.isfinite(logits).all():
                    raise RuntimeError("nonfinite KnownVal logits")
                val = support.save_predictions(run / "validation" / f"epoch{epoch:03}", dataset,
                                               "known_validation", val_ds.ids, truth, logits)
                online = joint.metrics(np.asarray(online_y), np.asarray(online_p), source.CLASSES[dataset])
                event = {"dataset": dataset, "epoch": epoch, "train_loss": loss_sum / seen,
                         **{f"train_online_{key}": online[key] for key in ("accuracy", "macro_f1", "weighted_f1")},
                         **{f"val_{key}": val[key] for key in ("loss", "accuracy", "macro_f1", "weighted_f1")},
                         "optimizer_steps": step, "train_seen": seen, "order_sha256": sha(run / f"order_epoch{epoch:03}.npy"),
                         "gradient_norm_means": {key: v / n_groups for key, v in norm_sums.items()},
                         "lr_groups_last_step": {key: [float(g["lr"]) for g in opt.param_groups] for key, opt in opts.items()},
                         "grad_scale": float(scaler.get_scale()), "epoch_seconds": time.time() - t0,
                         "elapsed_seconds": time.time() - started, "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device)}
                if val["macro_f1"] > best:
                    best, best_epoch, best_state = val["macro_f1"], epoch, cpu_state(model)
                event.update(best_epoch=best_epoch, best_val_macro_f1=best)
                events.append(event)
                history.write(json.dumps(event) + "\n")
                history.flush()
                write(run / "progress.json", event)
                print(json.dumps({key: event[key] for key in ("dataset", "epoch", "train_loss", "val_loss", "val_macro_f1",
                                                             "best_epoch", "elapsed_seconds", "peak_gpu_memory_bytes")}), flush=True)
                if epoch in PREFIXES:
                    if select_best(events, epoch)["epoch"] != best_epoch:
                        raise RuntimeError("prefix checkpoint selection mismatch")
                    path = run / f"prefix{epoch:03}_best.pt"
                    save_torch(path, {"state_dict": best_state, "config": config, "budget_epochs": epoch,
                                     "best_epoch": best_epoch, "best_val_macro_f1": best})
                    prefix = {"dataset": dataset, "budget_epochs": epoch, "best_epoch": best_epoch,
                              "best_val_macro_f1": best, "checkpoint_path": str(path.relative_to(ROOT)), "checkpoint_sha256": sha(path)}
                    prefixes.append(prefix)
                    write(run / f"prefix{epoch:03}_selection.json", prefix)
                    zs, ys = joint.predict(model, joint.load_batches(train_ds, MICRO, False), device)
                    support.save_predictions(run / "train_eval" / f"epoch{epoch:03}", dataset, "known_train", train_ds.ids, ys, zs)
                if epoch in (20, 50, 75, 100):
                    save_torch(run / f"recovery_epoch{epoch:03}.pt", {"state_dict": model.state_dict(), "epoch": epoch, "step": step,
                        "optimizers": {key: opt.state_dict() for key, opt in opts.items()}, "scheduler": tf_scheduler.state_dict(),
                        "grad_scaler": scaler.state_dict(), "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                        "cpu_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(),
                        "loader_generator": train_loader.generator.get_state(), "best_epoch": best_epoch,
                        "best_val_macro_f1": best, "best_state_dict": best_state, "config": config})
        if step != total_steps or len(events) != EPOCHS:
            raise RuntimeError("incomplete update/epoch budget")
        assert_locked()
        write(run / "train_summary.json", {"status": "PASS", "completed_epochs": EPOCHS, "optimizer_steps": step,
              "prefixes": prefixes, "elapsed_seconds": time.time() - started, "known_test_used_during_train": 0, "unknown_used": 0})
        print(f"TRAIN_PASS {dataset} epochs=100 steps={step}", flush=True)
    except BaseException as exc:
        write(run / "TRAIN_FAILURE.json", {"error": repr(exc), "traceback": traceback.format_exc(), "epoch": epoch, "step": step})
        if model is not None:
            try:
                save_torch(run / "failure_state.pt", {"state_dict": model.state_dict(), "epoch": epoch, "step": step,
                    "optimizers": {key: opt.state_dict() for key, opt in (opts or {}).items()},
                    "scheduler": tf_scheduler.state_dict() if tf_scheduler else None,
                    "grad_scaler": scaler.state_dict() if scaler else None, "error": repr(exc)})
            except BaseException as save_exc:
                write(run / "FAILURE_SAVE_ERROR.json", {"error": repr(save_exc)})
        raise


def freeze():
    assert_locked()
    if (ROOT / "selected_before_test.json").exists():
        raise FileExistsError("selections already frozen")
    selections = []
    for dataset in DATASETS:
        summary = read(run_dir(dataset) / "train_summary.json")
        if summary["status"] != "PASS" or summary["completed_epochs"] != EPOCHS:
            raise RuntimeError("both100epoch runs must complete before Test")
        selections.extend(summary["prefixes"])
    if len(selections) != 6 or {r["budget_epochs"] for r in selections} != set(PREFIXES):
        raise RuntimeError("missing budget checkpoint")
    for item in selections:
        if sha(ROOT / item["checkpoint_path"]) != item["checkpoint_sha256"]:
            raise RuntimeError("checkpoint changed before freeze")
    write(ROOT / "selected_before_test.json", {"created_at_utc": now(), "status": "PASS", "selections": selections,
                                              "test_used_for_selection": 0, "unknown_used": 0})
    print("SIX_CHECKPOINT_SELECTIONS_FROZEN", flush=True)


def assert_test_gate():
    gate = read(ROOT / "selected_before_test.json")
    if gate["status"] != "PASS" or len(gate["selections"]) != 6:
        raise RuntimeError("Test barrier not satisfied")
    for item in gate["selections"]:
        if sha(ROOT / item["checkpoint_path"]) != item["checkpoint_sha256"]:
            raise RuntimeError("selected checkpoint changed")
    return gate


def evaluate(dataset):
    assert_locked()
    gate = assert_test_gate()
    joint.seed_all()
    run = run_dir(dataset)
    scale = np.load(ROOT / "inputs" / dataset / "graph_scaler.npz", allow_pickle=False)
    values = {role: dataset_for(dataset, role, scale["mean"], scale["std"])
              for role in ("known_validation", "known_test")}
    train_ids = {r["flow_id"] for r in source.rows_for(dataset, "known_train")}
    if len(values["known_test"]) != source.EXPECTED[dataset][2] or set(values["known_test"].ids) & (train_ids | set(values["known_validation"].ids)):
        raise RuntimeError("KnownTest identity/count overlap failure")
    device = torch.device("cuda:0")
    model = joint.JointThreeView(source.CLASSES[dataset], math.ceil(source.EXPECTED[dataset][0] / BATCH) * EPOCHS).to(device)
    output_rows = []
    for item in [r for r in gate["selections"] if r["dataset"] == dataset]:
        checkpoint = torch.load(ROOT / item["checkpoint_path"], map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        for role, data in values.items():
            logits, truth = joint.predict(model, joint.load_batches(data, MICRO, False), device)
            output = run / "evaluation" / f"prefix{item['budget_epochs']:03}" / role
            met = support.save_predictions(output, dataset, role, data.ids, truth, logits)
            output_rows.append({**item, "role": role, "samples": len(data),
                                **{key: met[key] for key in ("accuracy", "macro_f1", "weighted_f1", "loss")}})
            if role == "known_validation":
                expected = read(run / "validation" / f"epoch{item['best_epoch']:03}" / "metrics.json")
                if any(abs(met[key] - expected[key]) > 1e-6 for key in ("accuracy", "macro_f1", "weighted_f1", "loss")):
                    raise RuntimeError("selected checkpoint validation replay mismatch")
        del checkpoint
    csv_write(run / "evaluation_results.csv", output_rows)
    assert_locked()
    assert_test_gate()
    write(run / "evaluation_complete.json", {"status": "PASS", "rows": len(output_rows), "completed_at_utc": now()})
    print(f"EVALUATE_PASS {dataset}", flush=True)


def finalize():
    assert_locked()
    gate = assert_test_gate()
    rows, paired, perclass = [], [], []
    for dataset in DATASETS:
        if read(run_dir(dataset) / "evaluation_complete.json")["status"] != "PASS":
            raise RuntimeError("evaluation missing")
        with (run_dir(dataset) / "evaluation_results.csv").open() as handle:
            results = list(csv.DictReader(handle))
        for row in results:
            row.update({key: float(row[key]) for key in ("accuracy", "macro_f1", "weighted_f1", "loss")})
            row["budget_epochs"], row["best_epoch"] = int(row["budget_epochs"]), int(row["best_epoch"])
            output = run_dir(dataset) / "evaluation" / f"prefix{row['budget_epochs']:03}" / row["role"]
            logits = np.load(output / "logits.npy", allow_pickle=False)
            with (output / "predictions.csv").open() as handle:
                predictions = list(csv.DictReader(handle))
            expected = source.rows_for(dataset, row["role"])
            if [x["flow_id"] for x in predictions] != [r["flow_id"] for r in expected]:
                raise RuntimeError("independent identity replay failed")
            labels = np.asarray([source.CLASSES[dataset].index(source.label_of(r["service_label"])) for r in expected])
            pred = logits.argmax(1)
            replay = {"accuracy": accuracy_score(labels, pred), "macro_f1": f1_score(labels, pred, average="macro", zero_division=0),
                      "weighted_f1": f1_score(labels, pred, average="weighted", zero_division=0)}
            if any(abs(row[key] - replay[key]) > 1e-12 for key in replay):
                raise RuntimeError("independent metric replay failed")
            if any(x["true_class"] != source.CLASSES[dataset][int(y)] or x["predicted_class"] != source.CLASSES[dataset][int(p)]
                   for x, y, p in zip(predictions, labels, pred, strict=True)):
                raise RuntimeError("prediction label replay failed")
            np.testing.assert_array_equal(confusion_matrix(labels, pred, labels=np.arange(len(source.CLASSES[dataset]))),
                                          np.load(output / "confusion_matrix.npy", allow_pickle=False))
            with (output / "per_class.csv").open() as handle:
                perclass.extend({"dataset": dataset, "budget_epochs": row["budget_epochs"], "role": row["role"], **r}
                                for r in csv.DictReader(handle))
            rows.append(row)
        for role in ("known_validation", "known_test"):
            indexed = {r["budget_epochs"]: r for r in rows if r["dataset"] == dataset and r["role"] == role}
            for a, b in ((20, 50), (20, 100), (50, 100)):
                paired.append({"dataset": dataset, "role": role, "comparison": f"{b}-{a}",
                    **{f"delta_{key}": indexed[b][key] - indexed[a][key] for key in ("accuracy", "macro_f1", "weighted_f1")}})
    csv_write(ROOT / "run_results.csv", rows)
    csv_write(ROOT / "paired_comparison.csv", paired)
    csv_write(ROOT / "per_class_results.csv", perclass)
    historical = []
    for dataset in DATASETS:
        for name, path, key in (("Stage37_original_J20", historical_run(dataset) / "known_test_evaluation/results.json", "known_test"),
                                ("Stage32_staged", PROJECT / "stage32_three_dataset_coarse_single_seed/runs" / dataset / "known_test_evaluation/results.json", "metrics")):
            obj = read(path)
            with (path.parent / "sample_predictions.csv").open() as handle:
                old_predictions = list(csv.DictReader(handle))
            expected = source.rows_for(dataset, "known_test")
            if [r["flow_id"] for r in old_predictions] != [r["flow_id"] for r in expected]:
                raise RuntimeError(f"historical Test sample identities differ: {path}")
            historical.append({"dataset": dataset, "method": name, "result_path": str(path.relative_to(PROJECT)),
                               "source_sha256": sha(path), "same_test_flow_ids": True,
                               **{k: obj[key][k] for k in ("accuracy", "macro_f1", "weighted_f1")}})
    csv_write(ROOT / "historical_references.csv", historical)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figures = ROOT / "figures"
    figures.mkdir(exist_ok=False)
    for dataset in DATASETS:
        history = [json.loads(line) for line in (run_dir(dataset) / "training_history.jsonl").read_text().splitlines()]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for key in ("train_loss", "val_loss"):
            axes[0].plot([r["epoch"] for r in history], [r[key] for r in history], label=key)
        for key in ("train_online_macro_f1", "val_macro_f1"):
            axes[1].plot([r["epoch"] for r in history], [r[key] for r in history], label=key)
        for ax in axes:
            ax.set_xlabel("Epoch"); ax.legend()
            for n in PREFIXES:
                ax.axvline(n, alpha=.2, color="grey")
        fig.suptitle(dataset + " fixed100-horizon joint training")
        fig.tight_layout(); fig.savefig(figures / f"{dataset}_learning_curves.png", dpi=160); plt.close(fig)
    test_deltas = [r for r in paired if r["role"] == "known_test" and r["comparison"] == "100-20"]
    conclusion = "LONGER_BUDGET_IMPROVES_BOTH" if all(r["delta_macro_f1"] > 0 for r in test_deltas) else "LONGER_BUDGET_NOT_CONSISTENTLY_BETTER"
    lines = ["# Stage46 — 联合训练100epoch预算结果", "", "Status: complete / INDEPENDENT_REPLAY_PASS", "",
             "同一100epoch调度轨迹的20/50/100预算；Known-Val选模，六选点冻结后才Test。单seed/已暴露Test，不能宣称等FLOPs或纯epoch因果效应。", "",
             "| Dataset | Budget | Best epoch | Role | Accuracy | Macro-F1 | Weighted-F1 |", "|---|---:|---:|---|---:|---:|---:|"]
    lines.extend(f"| {r['dataset']} | {r['budget_epochs']} | {r['best_epoch']} | {r['role']} | {r['accuracy']:.6f} | {r['macro_f1']:.6f} | {r['weighted_f1']:.6f} |" for r in rows)
    lines += ["", "## 历史同流参照（训练预算/调度不同）", ""]
    lines.extend(f"- {r['dataset']} {r['method']}: Test Accuracy/Macro/Weighted-F1={r['accuracy']:.6f}/{r['macro_f1']:.6f}/{r['weighted_f1']:.6f}" for r in historical)
    lines += ["", "## 限制与结论", "", "100epoch warmup/decay比历史20epoch拉长，且使用修复后的原子GradScaler；不能将历史差异全部归因于轮数。Train在线指标含训练模式dropout，不与eval指标直接等同。", "",
              f"结论：`{conclusion}`。只回答当前固定配方；无多seed稳定性结论。不自动启动下一实验。"]
    report = "\n".join(lines) + "\n"
    (ROOT / "stage46_report.md").write_text(report)
    (ROOT / "RESULTS.md").write_text(report)
    write(ROOT / "completion_verification.json", {"status": "PASS", "conclusion": conclusion, "runs_complete": 2,
        "epochs_each": EPOCHS, "checkpoints": len(gate["selections"]), "replayed_metric_rows": len(rows),
        "source_hashes_unchanged": True, "test_used_for_selection": 0, "unknown_used": 0,
        "completed_at_utc": now(), "selection_file_sha256": sha(ROOT / "selected_before_test.json")})
    manifest = read(ROOT / "manifest.json")
    manifest.update(status="complete", updated_at_utc=now(), core_results=rows, next_step="Stop; report budget sensitivity; no automatic new training.")
    write(ROOT / "manifest.json", manifest)
    print("STAGE46_METRIC_REPLAY_PASS", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "train", "freeze", "evaluate", "finalize"))
    parser.add_argument("--dataset", choices=DATASETS)
    args = parser.parse_args()
    if args.action in ("train", "evaluate") and args.dataset is None:
        parser.error("--dataset required")
    if args.action == "train":
        train(args.dataset)
    elif args.action == "evaluate":
        evaluate(args.dataset)
    else:
        globals()[args.action]()
