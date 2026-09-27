#!/usr/bin/env python3
"""Stage32 one-shot Known Test evaluation; fixed checkpoints and coarse labels."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback

import numpy as np
import torch

from common import (COARSE_CLASSES, DATASETS, F2_INPUT, F2_RUN, ROOT, S22, S25,
                    S27, S31, VNAT_PROTOCOL, VIEWS, coarse, expected_rows,
                    freeze_sources, sha256, write_json)
from test_inputs import ensure_three_heads_frozen

sys.path.insert(0, str(S31))
import evaluate_known_test as stage31_eval  # noqa: E402
from train_equal_fusion import Adapters, EqualFusion, encode, infer, metrics  # noqa: E402
from train_yatc_branch import protocol_rows  # noqa: E402


def save_csv(path, rows):
    if not rows or path.exists():
        raise RuntimeError(f"empty/existing CSV: {path}")
    names = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def load_iscx_test(dataset: str, rows: list[tuple[str, str]]) -> dict:
    """Use the same frozen Stage22/27 representations as Stage30 Train/Val."""
    p22 = S22 / "runs" / dataset / "seed2022" / "representations.npz"
    p27 = S27 / "runs" / dataset / "seed2022" / "features.npz"
    with np.load(p22, allow_pickle=False) as e3, np.load(p27, allow_pickle=False) as yatc:
        ids = e3["known_test_flow_ids"].astype(str).tolist()
        yids = yatc["known_test_flow_ids"].astype(str).tolist()
        expected = [uid for uid, _ in rows]
        if ids != yids or len(ids) != len(set(ids)) or set(ids) != set(expected):
            raise RuntimeError(f"{dataset}: Stage22/27 frozen Known Test flow-ID mismatch")
        reorder = np.asarray([ids.index(uid) for uid in expected], dtype=np.int64)
        z = e3["known_test_e3_z"].astype(np.float32)[reorder]
        y = yatc["known_test_yatc"].astype(np.float32)[reorder]
        mean = yatc["y_mean"].astype(np.float32)
        std = yatc["y_std"].astype(np.float32)
        if z.shape != (len(rows), 896) or y.shape != (len(rows), 192) or not (std > 0).all():
            raise RuntimeError(f"{dataset}: invalid historical Test representation shape")
        values = {"trafficformer": z[:, :768].copy(), "graph": z[:, 768:].copy(),
                  "yatc": ((y - mean) / std).astype(np.float32)}
    return values


def load_vnat_test(rows: list[tuple[str, str]], device: torch.device) -> dict:
    cache_root = ROOT / "input_caches" / "vnat" / VNAT_PROTOCOL
    for name in ("tf_fig_test", "yatc_mfr_test"):
        audit = json.loads((cache_root / name / "cache_audit.json").read_text())
        if audit["status"] != "PASS" or audit["known_test_features_materialized"] != len(rows):
            raise RuntimeError(f"VNAT Known Test cache failed: {name}")
    _, fine_classes, _ = protocol_rows("vnat", VNAT_PROTOCOL)
    stage31_eval.OUT = ROOT  # reuse read-only extraction logic with Stage32-owned Test caches
    values, _ = stage31_eval.extract_branches("vnat", VNAT_PROTOCOL, fine_classes,
        rows, device, S31 / "runs" / "vnat" / VNAT_PROTOCOL)
    return values


def classification(truth: np.ndarray, prediction: np.ndarray, classes: tuple[str, ...]) -> tuple[dict, list[dict]]:
    score, pc = metrics(truth, prediction, len(classes))
    rows = [{"class": name, "precision": float(pc[0][i]),
             "recall": float(pc[1][i]), "f1": float(pc[2][i]),
             "support": int(pc[3][i])} for i, name in enumerate(classes)]
    return score, rows


def bootstrap_macro_delta(truth, fresh, old, nclasses: int, repetitions: int = 1000):
    rng = np.random.default_rng(2022)
    samples = []
    for _ in range(repetitions):
        idx = rng.integers(0, len(truth), len(truth))
        new = metrics(truth[idx], fresh[idx], nclasses)[0]["macro_f1"]
        previous = metrics(truth[idx], old[idx], nclasses)[0]["macro_f1"]
        samples.append(new - previous)
    return [float(v) for v in np.quantile(samples, [0.025, 0.975])]


def baseline_iscx(dataset, rows, truth, fresh, classes):
    expected = dict(rows)
    output = []
    for name, path, field in (
        ("E3_hard_remap", S25 / "remap_predictions.csv", "method"),
        ("E3_coarse_head", S25 / "coarse_head_predictions.csv", "encoder"),
        ("YaTC_hard_remap_context", S25 / "remap_predictions.csv", "method"),
    ):
        method = "YaTC" if name.startswith("YaTC") else "E3"
        with path.open(newline="", encoding="utf-8") as f:
            entries = [r for r in csv.DictReader(f) if r["dataset"] == dataset
                and r["seed"] == "2022" and r[field] == method and r["role"] == "known_test"]
        lookup = {r["flow_id"]: r for r in entries}
        if len(lookup) != len(rows) or set(lookup) != set(expected):
            raise RuntimeError(f"{dataset}/{name} historical exact-ID mismatch")
        if any(lookup[uid]["true_fine"] != fine or
               lookup[uid]["true_coarse"] != coarse(dataset, fine) for uid, fine in rows):
            raise RuntimeError(f"{dataset}/{name} historical truth mismatch")
        old = np.asarray([classes.index(lookup[uid]["pred_coarse"]) for uid, _ in rows], dtype=np.int64)
        prior = metrics(truth, old, len(classes))[0]
        current = metrics(truth, fresh, len(classes))[0]
        ci = bootstrap_macro_delta(truth, fresh, old, len(classes))
        output.append({"dataset": dataset, "role": "known_test", "baseline": name,
            "same_flow_ids": True, "samples": len(rows),
            "baseline_accuracy": prior["accuracy"], "new_accuracy": current["accuracy"],
            "delta_accuracy": current["accuracy"] - prior["accuracy"],
            "baseline_macro_f1": prior["macro_f1"], "new_macro_f1": current["macro_f1"],
            "delta_macro_f1": current["macro_f1"] - prior["macro_f1"],
            "baseline_weighted_f1": prior["weighted_f1"], "new_weighted_f1": current["weighted_f1"],
            "delta_weighted_f1": current["weighted_f1"] - prior["weighted_f1"],
            "bootstrap_delta_macro_f1_low": ci[0], "bootstrap_delta_macro_f1_high": ci[1],
            "comparison_note": "same frozen flow/split/coarse label; historical model mechanism differs"})
    return output


def baseline_vnat_validation(classes):
    with (F2_INPUT / "input_manifest.csv").open(newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "validation"]
    rows.sort(key=lambda r: int(r["local_index"]))
    history_ids = [r["flow_uid"] for r in rows]
    config = json.loads((F2_RUN / "training_config.json").read_text())
    original = config["known_classes"]
    prediction = np.load(F2_RUN / "validation_predictions.npy", allow_pickle=False)
    old_by_id = {uid: classes.index(coarse("vnat", original[int(pred)]))
                 for uid, pred in zip(history_ids, prediction, strict=True)}
    path = ROOT / "runs" / "vnat" / "known_validation_predictions.csv"
    with path.open(newline="", encoding="utf-8") as f:
        current_rows = list(csv.DictReader(f))
    if len(current_rows) != len(old_by_id) or set(r["flow_id"] for r in current_rows) != set(old_by_id):
        raise RuntimeError("VNAT F2/T0 validation pairing mismatch")
    truth = np.asarray([classes.index(r["true_coarse"]) for r in current_rows], dtype=np.int64)
    fresh = np.asarray([classes.index(r["pred_coarse"]) for r in current_rows], dtype=np.int64)
    old = np.asarray([old_by_id[r["flow_id"]] for r in current_rows], dtype=np.int64)
    prior = metrics(truth, old, len(classes))[0]
    current = metrics(truth, fresh, len(classes))[0]
    ci = bootstrap_macro_delta(truth, fresh, old, len(classes))
    return {"dataset": "vnat", "role": "known_validation", "baseline": "historical_F2_hard_remap",
        "same_flow_ids": True, "samples": len(truth),
        "baseline_accuracy": prior["accuracy"], "new_accuracy": current["accuracy"],
        "delta_accuracy": current["accuracy"] - prior["accuracy"],
        "baseline_macro_f1": prior["macro_f1"], "new_macro_f1": current["macro_f1"],
        "delta_macro_f1": current["macro_f1"] - prior["macro_f1"],
        "baseline_weighted_f1": prior["weighted_f1"], "new_weighted_f1": current["weighted_f1"],
        "delta_weighted_f1": current["weighted_f1"] - prior["weighted_f1"],
        "bootstrap_delta_macro_f1_low": ci[0], "bootstrap_delta_macro_f1_high": ci[1],
        "comparison_note": "same frozen flow/split/coarse label; historical F2 training seed=2025 vs Stage32=2022; Val not Test"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("selected GPU required for frozen Test inference")
    locked = ensure_three_heads_frozen()
    run = ROOT / "runs" / args.dataset
    result = run / "known_test_evaluation"
    if result.exists():
        raise FileExistsError(result)
    result.mkdir(parents=True)
    try:
        torch.set_num_threads(4)
        device = torch.device("cuda:0")
        classes = COARSE_CLASSES[args.dataset]
        rows = expected_rows(args.dataset, "known_test")
        features = (load_vnat_test(rows, device) if args.dataset == "vnat"
                    else load_iscx_test(args.dataset, rows))
        scaler = np.load(run / "known_train_view_scalers.npz", allow_pickle=False)
        for view in VIEWS:
            features[view] = ((features[view] - scaler[f"{view}_mean"]) /
                              scaler[f"{view}_std"]).astype(np.float32)
            if not np.isfinite(features[view]).all():
                raise RuntimeError(f"nonfinite {args.dataset} Test view: {view}")
        truth = np.asarray([classes.index(coarse(args.dataset, fine)) for _, fine in rows], dtype=np.int64)
        bundle = {"labels": truth, **features}
        adapter = Adapters(len(classes)).to(device)
        adapter.load_state_dict(torch.load(run / "adapters_best.pt", map_location="cpu", weights_only=False)["state_dict"])
        encoded = encode(adapter, bundle, device)
        head = EqualFusion(len(classes), np.zeros(3), np.ones(3)).to(device)
        head.load_state_dict(torch.load(run / "T0_equal_best.pt", map_location="cpu", weights_only=False)["state_dict"])
        logits = infer(head, encoded, device)
        prediction = logits.argmax(1)
        probabilities = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
        score, per_class = classification(truth, prediction, classes)
        save_csv(result / "sample_predictions.csv", [
            {"flow_id": uid, "true_fine": fine, "true_coarse": classes[int(y)],
             "pred_coarse": classes[int(p)], "correct": int(y == p),
             "max_probability": float(prob.max())}
            for (uid, fine), y, p, prob in zip(rows, truth, prediction, probabilities, strict=True)])
        np.save(result / "logits.npy", logits.astype(np.float32), allow_pickle=False)
        save_csv(result / "per_class.csv", per_class)
        if args.dataset == "vnat":
            comparisons = [baseline_vnat_validation(classes)]
        else:
            comparisons = baseline_iscx(args.dataset, rows, truth, prediction, classes)
        save_csv(result / "paired_comparison.csv", comparisons)
        source_after = freeze_sources()
        if source_after != locked["source_hashes"]:
            raise RuntimeError("frozen source changed during Test evaluation")
        for dataset in DATASETS:
            info = json.loads((ROOT / "runs" / dataset / "known_validation_metrics.json").read_text())
            if info["checkpoint_hashes"] != locked["head_hashes"][dataset]:
                raise RuntimeError("frozen head changed during Test evaluation")
        write_json(result / "results.json", {"status": "PASS", "dataset": args.dataset,
            "known_test_samples": len(rows), "coarse_classes": classes, "metrics": score,
            "per_class": per_class, "comparisons": comparisons,
            "training_seed": 2022, "unknown_feature_values_loaded": 0,
            "test_parameter_selection": 0, "checkpoint_hashes": locked["head_hashes"][args.dataset],
            "comparison_limitation": "VNAT Test lacks frozen F2 same-flow replay" if args.dataset == "vnat" else
                "Historical Test previously exposed; development evidence only"})
        (result / "SUCCESS").write_text("SUCCESS\n")
        print(json.dumps({"status": "PASS", "dataset": args.dataset, "test": score,
                          "comparisons": comparisons}), flush=True)
    except BaseException as exc:
        write_json(result / "FAILURE.json", {"error": repr(exc), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
