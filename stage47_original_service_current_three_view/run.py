#!/usr/bin/env python3
"""Original-service matched benchmark of the actual current staged three-view model."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
S32 = PROJECT / "stage32_three_dataset_coarse_single_seed"
S31 = PROJECT / "stage31_four_dataset_three_view_equal"
S30 = PROJECT / "stage30_three_view_entropy_fusion"
S23B = PROJECT / "stage23b_tfe_trident_same_flow_adaptation"
sys.path.insert(0, str(S32))
spec = importlib.util.spec_from_file_location("stage47_current_training", S32 / "train_one.py")
current = importlib.util.module_from_spec(spec)
spec.loader.exec_module(current)
import common as inputs  # noqa: E402; Stage32 frozen-input routes

loader = inputs.stage30()
VIEWS = current.VIEWS
DATASETS = ("iscx_vpn", "iscx_tor")
SEEDS = (2022, 2023)
EPOCHS, BATCH = current.EPOCHS, current.BATCH
DEVICE = torch.device("cpu")
METHOD = "OURS-current-threeview-equal"
METRICS = ("accuracy", "macro_f1", "weighted_f1")
COUNTS = {"iscx_vpn": (8764, 1098, 1093), "iscx_tor": (8946, 1118, 1117)}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def csv_read(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def csv_write(path, rows):
    if not rows:
        raise RuntimeError(f"no rows: {path}")
    names = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader(); writer.writerows(rows)


def sha(path):
    return inputs.sha256(Path(path))


def run_dir(dataset, seed):
    return ROOT / "runs" / dataset / f"seed{seed}"


def metadata(dataset, role):
    # Membership only, never creates new splits or merges semantic labels.
    return sorted((r["flow_id"], r["service_label"]) for r in csv_read(inputs.S20)
                  if r["dataset"] == dataset and r["closed_role"] == role)


def lock_check():
    frozen = read(ROOT / "source_hashes_before.json")
    for path, expected in frozen.items():
        if sha(PROJECT / path) != expected:
            raise RuntimeError(f"protected source drift: {path}")
    return frozen


def save_predictions(folder, dataset, seed, role, values, logits, classes):
    folder.mkdir(parents=True, exist_ok=False)
    labels = values["labels"]
    prediction = logits.argmax(1)
    if not np.isfinite(logits).all():
        raise RuntimeError("nonfinite inference")
    scores, pc = current.metrics(labels, prediction, len(classes))
    scores["loss"] = float(F.cross_entropy(torch.from_numpy(logits), torch.as_tensor(labels)))
    np.save(folder / "logits.npy", logits, allow_pickle=False)
    np.save(folder / "confusion_matrix.npy", confusion_matrix(labels, prediction, labels=np.arange(len(classes))), allow_pickle=False)
    csv_write(folder / "predictions.csv", [{"flow_id": str(uid), "true_class": classes[int(y)],
        "predicted_class": classes[int(p)], "correct": int(y == p), "dataset": dataset, "seed": seed, "role": role}
        for uid, y, p in zip(values["flow_ids"], labels, prediction, strict=True)])
    csv_write(folder / "per_class.csv", [{"class": name, "precision": float(pc[0][i]), "recall": float(pc[1][i]),
                                         "f1": float(pc[2][i]), "support": int(pc[3][i])} for i, name in enumerate(classes)])
    write(folder / "metrics.json", scores)
    return scores


def prepare():
    if (ROOT / "preflight.json").exists():
        raise FileExistsError("preflight already exists")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_available():
        raise RuntimeError("registered CPU-only task must have CUDA_VISIBLE_DEVICES empty")
    if sys.prefix != "/home/birkenwald/data/SEU-WXY/conda_envs/2025-10-8-WXY-dgl_py310":
        raise RuntimeError("wrong environment")
    if sha(inputs.S20) != loader.EXPECTED_S20:
        raise RuntimeError("Stage20 freeze hash mismatch")
    protected = {inputs.S20, ROOT / "PLAN.md", ROOT / "run.py", ROOT / "progress.py", ROOT / "tests.py",
                 S32 / "train_one.py", S32 / "common.py", S31 / "train_equal_fusion.py", S30 / "preflight.py",
                 PROJECT / "stage28_er_cmgi_entropy_fusion/preflight.py", S23B / "stage23b_all_methods_run_level.csv"}
    audit = []
    known_labels = loader.known_labels()
    for dataset in DATASETS:
        for seed in SEEDS:
            paths = loader.sources(dataset, seed)
            old_hashes = read(S30 / "frozen_input_hashes_before.json")[f"{dataset}/{seed}"]
            actual_hashes = {name: sha(path) for name, path in paths.items()}
            if actual_hashes != old_hashes:
                raise RuntimeError(f"upstream frozen features/checkpoints drift: {dataset}/{seed}")
            protected.update(paths.values())
            pair = loader.load_three(dataset, seed, known_labels[dataset])
            classes = pair["services"]
            if len(classes) != (6 if dataset == "iscx_vpn" else 7):
                raise RuntimeError("wrong original-service label space")
            train, val = [pair["roles"][role] for role in ("known_train", "known_validation")]
            stats = current.normalize(train, val)
            folder = ROOT / "inputs" / dataset / f"seed{seed}"
            folder.mkdir(parents=True, exist_ok=False)
            np.savez(folder / "known_train_view_scalers.npz", **{f"{view}_{kind}": v for view, (m, s) in stats.items()
                                                               for kind, v in (("mean", m), ("std", s))})
            write(folder / "classes.json", classes)
            membership = []
            for index, (role, values) in enumerate((("known_train", train), ("known_validation", val))):
                expected = metadata(dataset, role)
                if len(values["labels"]) != COUNTS[dataset][index] or values["flow_ids"].astype(str).tolist() != [uid for uid, _ in expected]:
                    raise RuntimeError("frozen sample-ID/count mismatch")
                if [classes[int(y)] for y in values["labels"]] != [label for _, label in expected]:
                    raise RuntimeError("original service truth mismatch")
                np.savez(folder / f"{role}.npz", **values)
                membership.extend({"role": role, "flow_id": uid, "service": label} for uid, label in expected)
            csv_write(folder / "train_val_membership.csv", membership)
            # Actual current model forward parity, before any new Test access.
            parity = None
            if seed == 2022:
                ref = S32 / "runs" / dataset
                adapter = current.Adapters(len(inputs.COARSE_CLASSES[dataset]))
                adapter.load_state_dict(torch.load(ref / "adapters_best.pt", map_location="cpu", weights_only=False)["state_dict"], strict=True)
                head = current.EqualFusion(len(inputs.COARSE_CLASSES[dataset]), np.zeros(3), np.ones(3))
                head.load_state_dict(torch.load(ref / "T0_equal_best.pt", map_location="cpu", weights_only=False)["state_dict"], strict=True)
                logits = current.infer(head, current.encode(adapter, val, DEVICE), DEVICE)
                expected = np.load(ref / "known_validation_logits.npy", allow_pickle=False)
                np.testing.assert_allclose(logits, expected, atol=1e-5, rtol=1e-5)
                if not np.array_equal(logits.argmax(1), expected.argmax(1)):
                    raise RuntimeError("Stage32 current-model prediction parity failed")
                parity = {"status": "PASS", "max_abs_logit_error": float(np.max(np.abs(logits - expected))),
                          "predictions_identical": True, "role": "known_validation"}
                protected.update(ref / name for name in ("adapters_best.pt", "T0_equal_best.pt", "known_validation_logits.npy"))
            protected.update(p for p in folder.iterdir() if p.is_file())
            audit.append({"dataset": dataset, "seed": seed, "classes": classes, "counts": COUNTS[dataset],
                          "train_val_class_support": {role: {label: int(np.sum(values["labels"] == i)) for i, label in enumerate(classes)}
                                                       for role, values in (("known_train", train), ("known_validation", val))},
                          "current_coarse_model_parity": parity, "test_values_used": 0, "unknown_used": 0})
    write(ROOT / "source_hashes_before.json", {os.path.relpath(p, PROJECT): sha(p) for p in sorted(protected)})
    write(ROOT / "preflight.json", {"status": "PASS", "rows": audit, "test_values_used": 0, "unknown_used": 0})
    print("PREFLIGHT_PASS current model parity/input hashes/original6+7labels", flush=True)


def load_known(dataset, seed, role):
    if role not in ("known_train", "known_validation"):
        raise ValueError("Known-only cache loader forbids Test/Unknown")
    with np.load(ROOT / "inputs" / dataset / f"seed{seed}" / f"{role}.npz", allow_pickle=False) as values:
        return {key: values[key].copy() for key in values.files}


def train(dataset, seed):
    lock_check()
    started = time.monotonic()
    run = run_dir(dataset, seed)
    run.mkdir(parents=True, exist_ok=False)
    classes = read(ROOT / "inputs" / dataset / f"seed{seed}" / "classes.json")
    train_values, val = [load_known(dataset, seed, role) for role in ("known_train", "known_validation")]
    config = {"dataset": dataset, "encoder_seed": seed, "training_seed": seed, "classes": classes, "method": METHOD,
              "architecture_source": str((S31 / "train_equal_fusion.py").relative_to(PROJECT)),
              "normalization_source": str((S32 / "train_one.py").relative_to(PROJECT)),
              "adapter_epochs": EPOCHS, "head_epochs": EPOCHS, "batch_size": BATCH,
              "optimizer": "Adam(lr=1e-3)", "adapter_loss": "mean CE +0.05mean KL", "head_loss": "CE",
              "device": "cpu", "threads": 4, "encoder_updates": 0, "test_selection_samples": 0, "unknown_used": 0}
    write(run / "config.json", config)
    curves, best_epochs = [], {}
    current.seeded(seed)
    adapter = current.Adapters(len(classes)).to(DEVICE)
    optimizer = torch.optim.Adam(adapter.parameters(), lr=1e-3)
    tensors = [torch.from_numpy(train_values[name]) for name in VIEWS] + [torch.from_numpy(train_values["labels"])]
    batches = DataLoader(TensorDataset(*tensors), batch_size=BATCH, shuffle=True,
                         generator=torch.Generator().manual_seed(seed), num_workers=0)
    best, state = -1.0, None
    for epoch in range(1, EPOCHS + 1):
        adapter.train(); loss_sum = 0.0; seen = 0
        for batch in batches:
            optimizer.zero_grad(set_to_none=True)
            loss, _, _ = adapter.training_loss(list(batch[:3]), batch[-1])
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite adapter loss")
            loss.backward(); optimizer.step()
            loss_sum += float(loss.detach()) * len(batch[-1]); seen += len(batch[-1])
        _, _, pred = current.encode(adapter, val, DEVICE)
        f1 = [current.metrics(val["labels"], pred[:, i], len(classes))[0]["macro_f1"] for i in range(3)]
        score = float(np.mean(f1))
        event = {"dataset": dataset, "seed": seed, "phase": "adapters", "epoch": epoch, "train_samples": seen,
                 "train_loss": loss_sum / seen, "val_mean_unimodal_macro_f1": score,
                 **{f"val_{view}_macro_f1": v for view, v in zip(VIEWS, f1, strict=True)}}
        if score > best:
            best, best_epochs["adapter"] = score, epoch
            state = {k: v.detach().clone() for k, v in adapter.state_dict().items()}
        curves.append(event); write(run / "progress.json", event)
        with (run / "training_history.jsonl").open("a") as handle:
            handle.write(json.dumps(event) + "\n")
        if epoch == 1 or epoch % 5 == 0:
            print(json.dumps(event), flush=True)
    adapter.load_state_dict(state)
    torch.save({"state_dict": state, "best_epoch": best_epochs["adapter"], "epochs_completed": EPOCHS, "config": config}, run / "adapters_best.pt")
    etrain, eval_ = current.encode(adapter, train_values, DEVICE), current.encode(adapter, val, DEVICE)
    current.seeded(seed)
    center = np.median(etrain[1], axis=0)
    mad = np.median(np.abs(etrain[1] - center), axis=0)
    head = current.EqualFusion(len(classes), center, mad).to(DEVICE)
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)
    batches = DataLoader(TensorDataset(torch.from_numpy(etrain[0]), torch.from_numpy(etrain[1]), torch.from_numpy(train_values["labels"])),
                         batch_size=BATCH, shuffle=True, generator=torch.Generator().manual_seed(seed), num_workers=0)
    best, state = -1.0, None
    for epoch in range(1, EPOCHS + 1):
        head.train(); loss_sum = 0.0; seen = 0
        for mu, entropy, y in batches:
            optimizer.zero_grad(set_to_none=True)
            logits, _ = head(mu, entropy)
            loss = F.cross_entropy(logits, y)
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite head loss")
            loss.backward(); optimizer.step()
            loss_sum += float(loss.detach()) * len(y); seen += len(y)
        logits = current.infer(head, eval_, DEVICE)
        scores = current.metrics(val["labels"], logits.argmax(1), len(classes))[0]
        event = {"dataset": dataset, "seed": seed, "phase": "T0_equal", "epoch": epoch, "train_samples": seen,
                 "train_loss": loss_sum / seen, **{f"val_{k}": v for k, v in scores.items()}}
        if scores["macro_f1"] > best:
            best, best_epochs["head"] = scores["macro_f1"], epoch
            state = {k: v.detach().clone() for k, v in head.state_dict().items()}
        curves.append(event); write(run / "progress.json", event)
        with (run / "training_history.jsonl").open("a") as handle:
            handle.write(json.dumps(event) + "\n")
        if epoch == 1 or epoch % 5 == 0:
            print(json.dumps(event), flush=True)
    head.load_state_dict(state)
    torch.save({"state_dict": state, "best_epoch": best_epochs["head"], "epochs_completed": EPOCHS, "config": config}, run / "T0_equal_best.pt")
    logits = current.infer(head, eval_, DEVICE)
    scores = save_predictions(run / "known_validation", dataset, seed, "known_validation", val, logits, classes)
    csv_write(run / "training_history.csv", curves)
    lock_check()
    result = {"status": "PASS", "best_epochs": best_epochs, "validation": scores, "elapsed_seconds": time.monotonic() - started,
              "encoder_updates": 0, "test_selection_samples": 0, "unknown_used": 0,
              "checkpoint_sha256": {name: sha(run / name) for name in ("adapters_best.pt", "T0_equal_best.pt")}}
    write(run / "train_summary.json", result)
    print(f"TRAIN_PASS {dataset}/{seed} valMacroF1={scores['macro_f1']:.6f}", flush=True)


def freeze():
    selections = []
    for dataset in DATASETS:
        for seed in SEEDS:
            run = run_dir(dataset, seed)
            info = read(run / "train_summary.json")
            if info["status"] != "PASS":
                raise RuntimeError("four-run freeze barrier incomplete")
            for name, digest in info["checkpoint_sha256"].items():
                if sha(run / name) != digest:
                    raise RuntimeError("checkpoint changed before freeze")
                selections.append({"dataset": dataset, "seed": seed, "path": str((run / name).relative_to(ROOT)), "sha256": digest})
    write(ROOT / "selected_before_test.json", {"status": "PASS", "checkpoints": selections, "test_used_for_selection": 0})
    print("ALL_FOUR_RUNS_EIGHT_CHECKPOINTS_FROZEN", flush=True)


def test_gate():
    gate = read(ROOT / "selected_before_test.json")
    if gate["status"] != "PASS" or len(gate["checkpoints"]) != 8:
        raise RuntimeError("Test blocked before all four selected models freeze")
    for row in gate["checkpoints"]:
        if sha(ROOT / row["path"]) != row["sha256"]:
            raise RuntimeError("selected checkpoint changed")


def test_values(dataset, seed, classes):
    test_gate()
    expected = metadata(dataset, "known_test")
    folder = ROOT / "inputs" / dataset / f"seed{seed}"
    with np.load(loader.sources(dataset, seed)["stage22_representations"], allow_pickle=False) as e3, \
         np.load(loader.sources(dataset, seed)["stage27_yatc_features"], allow_pickle=False) as yt:
        ids, yids = e3["known_test_flow_ids"].astype(str).tolist(), yt["known_test_flow_ids"].astype(str).tolist()
        if ids != yids or ids != [uid for uid, _ in expected] or len(ids) != COUNTS[dataset][2]:
            raise RuntimeError("Test ID order/coverage drift")
        z = e3["known_test_e3_z"].astype(np.float32)
        y = (yt["known_test_yatc"].astype(np.float32) - yt["y_mean"]) / yt["y_std"]
        result = {"flow_ids": np.asarray(ids), "labels": np.asarray([classes.index(c) for _, c in expected], dtype=np.int64),
                  "trafficformer": z[:, :768].copy(), "graph": z[:, 768:].copy(), "yatc": y.astype(np.float32)}
    with np.load(folder / "known_train_view_scalers.npz", allow_pickle=False) as stats:
        for view, dim in zip(VIEWS, (768, 128, 192), strict=True):
            result[view] = ((result[view] - stats[f"{view}_mean"]) / stats[f"{view}_std"]).astype(np.float32)
            if result[view].shape != (len(ids), dim) or not np.isfinite(result[view]).all():
                raise RuntimeError("invalid Test features")
    all_train_val_ids = set(load_known(dataset, seed, "known_train")["flow_ids"]) | set(load_known(dataset, seed, "known_validation")["flow_ids"])
    if set(ids) & all_train_val_ids:
        raise RuntimeError("Train/Val/Test identity overlap")
    return result


def evaluate(dataset, seed):
    test_gate(); lock_check()
    run = run_dir(dataset, seed)
    classes = read(ROOT / "inputs" / dataset / f"seed{seed}" / "classes.json")
    adapter = current.Adapters(len(classes))
    adapter.load_state_dict(torch.load(run / "adapters_best.pt", map_location="cpu", weights_only=False)["state_dict"], strict=True)
    head = current.EqualFusion(len(classes), np.zeros(3), np.ones(3))
    head.load_state_dict(torch.load(run / "T0_equal_best.pt", map_location="cpu", weights_only=False)["state_dict"], strict=True)
    val = load_known(dataset, seed, "known_validation")
    val_logits = current.infer(head, current.encode(adapter, val, DEVICE), DEVICE)
    np.testing.assert_allclose(val_logits, np.load(run / "known_validation/logits.npy", allow_pickle=False), atol=1e-6, rtol=1e-6)
    values = test_values(dataset, seed, classes)
    logits = current.infer(head, current.encode(adapter, values, DEVICE), DEVICE)
    scores = save_predictions(run / "known_test", dataset, seed, "known_test", values, logits, classes)
    # Independent metric and checkpoint replay from fixed source arrays, in reverse order.
    reversed_values = {key: value[::-1].copy() for key, value in values.items()}
    replay = current.infer(head, current.encode(adapter, reversed_values, DEVICE), DEVICE)[::-1]
    np.testing.assert_allclose(logits, replay, atol=1e-5, rtol=1e-5)
    if not np.array_equal(logits.argmax(1), replay.argmax(1)):
        raise RuntimeError("batch/permutation prediction replay failed")
    truth, prediction = values["labels"], logits.argmax(1)
    independent = {"accuracy": accuracy_score(truth, prediction), "macro_f1": f1_score(truth, prediction, average="macro", zero_division=0),
                   "weighted_f1": f1_score(truth, prediction, average="weighted", zero_division=0)}
    if any(abs(scores[key] - independent[key]) > 1e-12 for key in METRICS):
        raise RuntimeError("independent metric replay failed")
    for role in ("known_validation", "known_test"):
        folder = run / role
        predictions = csv_read(folder / "predictions.csv")
        expected = metadata(dataset, role)
        if [(p["flow_id"], p["true_class"]) for p in predictions] != expected:
            raise RuntimeError("saved identity/truth replay mismatch")
        actual = np.load(folder / "logits.npy", allow_pickle=False).argmax(1)
        labels = np.asarray([classes.index(c) for _, c in expected])
        met = read(folder / "metrics.json")
        p, r, f, s = precision_recall_fscore_support(labels, actual, labels=np.arange(len(classes)), zero_division=0)
        for key, value in (("accuracy", float(np.mean(labels == actual))), ("macro_f1", float(f.mean())), ("weighted_f1", float(np.average(f, weights=s)))):
            if abs(met[key] - value) > 1e-12:
                raise RuntimeError("saved metric replay failed")
        np.testing.assert_array_equal(np.load(folder / "confusion_matrix.npy", allow_pickle=False),
                                     confusion_matrix(labels, actual, labels=np.arange(len(classes))))
    write(run / "evaluation_verification.json", {"status": "PASS", "independent_metric_replay": independent,
        "checkpoint_logit_reverse_order_max_error": float(np.max(np.abs(logits - replay))), "source_hashes_unchanged": True,
        "test_parameter_selection": 0, "unknown_used": 0})
    print(f"TEST_PASS {dataset}/{seed} {scores}", flush=True)


def aggregate():
    test_gate()
    new, per_class, paired = [], [], []
    old = csv_read(S23B / "stage23b_all_methods_run_level.csv")
    for row in old:
        row.update({key: float(row[key]) for key in METRICS})
        row["seed"] = int(row["seed"])
    for dataset in DATASETS:
        for seed in SEEDS:
            run = run_dir(dataset, seed)
            if read(run / "evaluation_verification.json")["status"] != "PASS":
                raise RuntimeError("unverified Test")
            scores = read(run / "known_test/metrics.json")
            summary = read(run / "train_summary.json")
            row = {"dataset": dataset, "seed": seed, "method": METHOD, **{k: scores[k] for k in METRICS},
                   "adapter_best_epoch": summary["best_epochs"]["adapter"], "head_best_epoch": summary["best_epochs"]["head"],
                   "train_flows": COUNTS[dataset][0], "validation_flows": COUNTS[dataset][1], "test_flows": COUNTS[dataset][2],
                   "sample_parity": "PASS", "method_scope": "current three-view architecture, original fine service, CPU adapter/head retrain"}
            new.append(row)
            per_class.extend({"dataset": dataset, "seed": seed, **r} for r in csv_read(run / "known_test/per_class.csv"))
            for baseline in [r for r in old if r["dataset"] == dataset and r["seed"] == seed]:
                if tuple(int(baseline[k]) for k in ("train_flows", "validation_flows", "test_flows")) != COUNTS[dataset] or baseline["sample_parity"] != "PASS":
                    raise RuntimeError("historical sample-count/parity mismatch")
                paired.append({"dataset": dataset, "seed": seed, "candidate": METHOD, "baseline": baseline["method"],
                               **{f"delta_{k}": row[k] - baseline[k] for k in METRICS}})
    csv_write(ROOT / "run_results.csv", new)
    csv_write(ROOT / "all_methods_run_results.csv", old + new)
    csv_write(ROOT / "per_class_results.csv", per_class)
    csv_write(ROOT / "paired_comparison.csv", paired)
    methods = list(dict.fromkeys(r["method"] for r in old)) + [METHOD]
    means = []
    for dataset in DATASETS:
        for method in methods:
            rows = [r for r in old + new if r["dataset"] == dataset and r["method"] == method]
            if sorted(r["seed"] for r in rows) != list(SEEDS):
                raise RuntimeError("two-seed aggregation identity mismatch")
            means.append({"dataset": dataset, "method": method, "num_seeds": 2,
                          **{f"{key}_{stat}": float(op([r[key] for r in rows])) for key in METRICS
                             for stat, op in (("mean", np.mean), ("population_sd", lambda x: np.std(x, ddof=0)))}})
    csv_write(ROOT / "method_summary.csv", means)
    titles = {"iscx_vpn": "ISCX-VPN（六类）", "iscx_tor": "ISCXTor2016（七类）"}
    text = ["# 表4-2补充：当前三视图的原服务标签同流闭集对照", "", "Status: complete / INDEPENDENT_REPLAY_PASS", "",
            "2022、2023两个encoder/training seed，均值±总体标准差(ddof=0)，单位%。保留七条旧配置并新增当前三视图，共八配置。", ""]
    for dataset in DATASETS:
        rows = [r for r in means if r["dataset"] == dataset]
        maxima = {key: max(r[f"{key}_mean"] for r in rows) for key in METRICS}
        text += ["## " + titles[dataset], "", "| 方法 | Accuracy | Macro-F1 | Weighted-F1 |", "|---|---:|---:|---:|"]
        for row in rows:
            values = []
            for key in METRICS:
                cell = f"{100*row[key+'_mean']:.2f} ± {100*row[key+'_population_sd']:.2f}"
                values.append(f"**{cell}**" if row[key+'_mean'] == maxima[key] else cell)
            text.append("| " + row["method"] + " | " + " | ".join(values) + " |")
        fresh = next(r for r in rows if r["method"] == METHOD)
        text += ["", "当前方法减基线的双seed平均Macro-F1差值（百分点）："]
        for baseline in rows:
            if baseline["method"] != METHOD:
                text.append(f"- {baseline['method']}: {100*(fresh['macro_f1_mean']-baseline['macro_f1_mean']):+.3f}")
        text.append("")
    text += ["## 执行与限制", "", "复用原fine-trained TF/FIG/YaTC encoder及冻结表示，重新训练当前Gaussian适配器与等权拼接头，非Stage27其他融合头、非Stage30旧分数、非Stage46联合结果。每run30+30epoch，Known-Val选模，全部四run冻结后Test。", "",
             "CPU4线程串行轻量模块训练，与历史GPU训练可能有浮点/随机数实现差异；模型、loss、epoch、batch、数据角色及标签规则未改。两个seed不足以充分证明稳定性；旧Test已暴露，只属于development对照。该闭集结果不能证明开集拒识改进。", "",
             "原表及历史图未覆盖。table4_2_supplement.md是待整合正文的新表，保留本次独立实验来源。无Unknown访问/encoder更新/Test选参，不自动开始下一阶段。"]
    report = "\n".join(text) + "\n"
    (ROOT / "table4_2_supplement.md").write_text(report)
    (ROOT / "RESULTS.md").write_text(report)
    lock_check()
    write(ROOT / "source_hashes_after.json", read(ROOT / "source_hashes_before.json"))
    write(ROOT / "completion_verification.json", {"status": "PASS", "completed_runs": 4, "adapter_epochs_each": EPOCHS,
        "head_epochs_each": EPOCHS, "frozen_checkpoints": 8, "metric_replays": 8, "permutation_checkpoint_replays": 4,
        "all_method_run_rows": len(old + new), "summary_rows": len(means), "paired_rows": len(paired),
        "source_hashes_unchanged": True, "encoder_updates": 0, "unknown_used": 0, "test_parameter_selection": 0})
    manifest = read(ROOT / "manifest.json")
    manifest.update(status="complete", core_results=[r for r in means if r["method"] == METHOD], next_step="Stop; present matched8configuration results, no additional experiments.")
    manifest["execution"]["exit_code"] = 0
    write(ROOT / "manifest.json", manifest)
    print("STAGE47_COMPLETE_INDEPENDENT_REPLAY_PASS", flush=True)


def main():
    torch.set_num_threads(4)
    manifest = read(ROOT / "manifest.json")
    manifest.update(status="running", inputs=[{"path": str(inputs.S20), "sha256": sha(inputs.S20), "role": "frozen split"}],
        configuration={"files": ["PLAN.md"], "parameters": {"adapter_epochs": EPOCHS, "head_epochs": EPOCHS, "batch_size": BATCH, "device": "cpu", "threads": 4}, "seeds": list(SEEDS)},
        limitations=["Previously exposed Test/development; two seeds only; CPU/GPU numerical RNG differences disclosed."],
        next_step="Known-only preparation and training, four-run freeze, Test and independent replay.")
    manifest["execution"].update(tmux_session="stage47_table42_fill_20261023", environment=sys.prefix,
                                 command="env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python stage47_original_service_current_three_view/run.py", physical_gpu_ids=[])
    manifest["artifacts"].append({"path": "../.tmux-task/stage47_table42_fill_20261023/output.log", "scope": "external", "role": "execution-log", "sha256": None, "hash_status": "live-log"})
    write(ROOT / "manifest.json", manifest)
    try:
        prepare()
        for dataset in DATASETS:
            for seed in SEEDS:
                write(ROOT / "progress.json", {"phase": "train", "dataset": dataset, "seed": seed})
                train(dataset, seed)
        freeze()
        for dataset in DATASETS:
            for seed in SEEDS:
                write(ROOT / "progress.json", {"phase": "test", "dataset": dataset, "seed": seed})
                evaluate(dataset, seed)
        aggregate()
        write(ROOT / "progress.json", {"status": "complete", "runs": 4})
    except BaseException as exc:
        write(ROOT / "FAILURE.json", {"error": repr(exc), "traceback": traceback.format_exc()})
        manifest = read(ROOT / "manifest.json")
        manifest.update(status="failed", next_step="Inspect preserved failure; no Test-barrier bypass.")
        manifest["execution"]["exit_code"] = 1
        write(ROOT / "manifest.json", manifest)
        with (ROOT / "RESULTS.md").open("a") as handle:
            handle.write(f"\nExecution failure, partial evidence preserved: {exc!r}\n")
        raise


if __name__ == "__main__":
    main()
