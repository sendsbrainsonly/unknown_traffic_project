#!/usr/bin/env python3
"""VNAT six-class diagnostic using Stage32's frozen inputs and T0 training code.

The six-class taxonomy is an exploratory response to Stage32's exposed four-class
Test score, not a pre-Test label choice or independent validation protocol.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
S32 = PROJECT / "stage32_three_dataset_coarse_single_seed"
S31 = PROJECT / "stage31_four_dataset_three_view_equal"
sys.path.insert(0, str(S32))
import common as prior  # noqa: E402

CLASSES = ("streaming", "rdp", "rsync", "scp", "skype", "ssh")
LABEL_MAP = {
    "netflix": "streaming", "youtube": "streaming",
    "rdp": "rdp", "rsync": "rsync", "scp": "scp",
    "skype": "skype", "ssh": "ssh",
}
ROLES = ("known_train", "known_validation", "known_test")
EXPECTED_COUNTS = {"known_train": 15704, "known_validation": 1960, "known_test": 1960}


def write_json(path: Path, value: object) -> None:
    prior.write_json(path, value)


def rows(role: str) -> list[tuple[str, str]]:
    if role not in ROLES:
        raise ValueError(role)
    source = prior.expected_rows("vnat", role)
    if len(source) != EXPECTED_COUNTS[role] or set(fine for _, fine in source) != set(LABEL_MAP):
        raise RuntimeError(f"unexpected frozen {role} membership or applications")
    if set(LABEL_MAP[fine] for _, fine in source) != set(CLASSES):
        raise RuntimeError(f"six-class {role} has an empty class")
    return source


def load_known(dataset: str, role: str) -> dict:
    if dataset != "vnat" or role not in ROLES[:2]:
        raise ValueError("six-class fit may only load VNAT Known Train/Validation")
    source = prior.load_known("vnat", role)
    expected = rows(role)
    if source["flow_ids"].astype(str).tolist() != [uid for uid, _ in expected]:
        raise RuntimeError(f"{role} flow-ID order changed")
    source["labels"] = np.asarray([CLASSES.index(LABEL_MAP[fine]) for _, fine in expected], dtype=np.int64)
    return source


def preflight() -> None:
    if (ROOT / "preflight.json").exists():
        raise FileExistsError(ROOT / "preflight.json")
    frozen = prior.freeze_sources()
    details = {}
    for role in ROLES:
        source = rows(role)  # Test: manifest IDs/labels only; no feature values.
        details[role] = {
            "flows": len(source),
            "application_counts": dict(sorted(Counter(fine for _, fine in source).items())),
            "six_class_counts": dict(sorted(Counter(LABEL_MAP[fine] for _, fine in source).items())),
        }
    train, val = load_known("vnat", "known_train"), load_known("vnat", "known_validation")
    if set(train["flow_ids"].tolist()) & set(val["flow_ids"].tolist()):
        raise RuntimeError("Known Train/Validation ID overlap")
    if prior.freeze_sources() != frozen:
        raise RuntimeError("frozen sources changed in preflight")
    write_json(ROOT / "frozen_source_hashes_before.json", frozen)
    write_json(ROOT / "preflight.json", {
        "status": "PASS", "protocol": prior.VNAT_PROTOCOL, "training_seed": prior.TRAINING_SEED,
        "classes": CLASSES, "application_to_class": LABEL_MAP, "split_support": details,
        "known_train_features_loaded": len(train["labels"]),
        "known_validation_features_loaded": len(val["labels"]),
        "test_feature_values_loaded": 0, "unknown_feature_values_loaded": 0,
        "stage32_test_seen_before_taxonomy_choice": True,
    })
    print(json.dumps({"phase": "preflight", "status": "PASS", "support": details}), flush=True)


def train() -> None:
    if not (ROOT / "preflight.json").exists():
        raise RuntimeError("run preflight first")
    import train_one as stage32_train  # noqa: E402

    # Reuse the exact optimizer, T0 architecture, normalization, epochs and
    # checkpoint-selection loops without altering any historical Stage32 file.
    stage32_train.ROOT = ROOT
    stage32_train.DATASETS = ("vnat",)
    stage32_train.COARSE_CLASSES = {"vnat": CLASSES}
    stage32_train.load_known = load_known
    stage32_train.freeze_sources = prior.freeze_sources
    saved = sys.argv
    try:
        sys.argv = [str(stage32_train.__file__), "--dataset", "vnat"]
        stage32_train.main()
    finally:
        sys.argv = saved


def evaluate() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("selected CUDA GPU required for frozen Test inference")
    preflight_report = json.loads((ROOT / "preflight.json").read_text())
    if preflight_report["status"] != "PASS" or preflight_report["test_feature_values_loaded"] != 0:
        raise RuntimeError("Known-only preflight failed")
    source_hashes = json.loads((ROOT / "frozen_source_hashes_before.json").read_text())
    if prior.freeze_sources() != source_hashes:
        raise RuntimeError("frozen source changed before Test")
    run = ROOT / "runs" / "vnat"
    if not (run / "SUCCESS").is_file():
        raise RuntimeError("six-class training incomplete")
    selected = json.loads((run / "known_validation_metrics.json").read_text())
    checkpoint_hashes = selected["checkpoint_hashes"]
    for name, digest in checkpoint_hashes.items():
        if prior.sha256(run / name) != digest:
            raise RuntimeError(f"selected checkpoint changed: {name}")
    result = run / "known_test_evaluation"
    if result.exists():
        raise FileExistsError(result)
    # Persist the selected checkpoint evidence before opening Test feature values.
    write_json(ROOT / "selected_head_before_test.json", {
        "checkpoint_hashes": checkpoint_hashes, "source_hashes": source_hashes,
        "head_selected_using": "Known Validation six-class Macro-F1",
        "test_parameter_selection": 0,
    })
    result.mkdir()
    torch.set_num_threads(4)
    device = torch.device("cuda:0")
    source = rows("known_test")
    sys.path.insert(0, str(S31))
    import evaluate_one as stage32_eval  # noqa: E402
    from train_equal_fusion import Adapters, EqualFusion, encode, infer, metrics  # noqa: E402

    cache_root = S32 / "input_caches" / "vnat" / prior.VNAT_PROTOCOL
    cache_files = sorted(p for p in cache_root.rglob("*") if p.is_file())
    cache_before = {str(p.relative_to(cache_root)): prior.sha256(p) for p in cache_files}
    features = stage32_eval.load_vnat_test(source, device)
    with np.load(run / "known_train_view_scalers.npz", allow_pickle=False) as scaler:
        for view in prior.VIEWS:
            features[view] = ((features[view] - scaler[f"{view}_mean"]) /
                              scaler[f"{view}_std"]).astype(np.float32)
            if not np.isfinite(features[view]).all():
                raise RuntimeError(f"nonfinite Test view {view}")
    truth = np.asarray([CLASSES.index(LABEL_MAP[fine]) for _, fine in source], dtype=np.int64)
    adapter = Adapters(len(CLASSES)).to(device)
    adapter.load_state_dict(torch.load(run / "adapters_best.pt", map_location="cpu", weights_only=False)["state_dict"])
    encoded = encode(adapter, {"labels": truth, **features}, device)
    head = EqualFusion(len(CLASSES), np.zeros(3), np.ones(3)).to(device)
    head.load_state_dict(torch.load(run / "T0_equal_best.pt", map_location="cpu", weights_only=False)["state_dict"])
    logits = infer(head, encoded, device)
    prediction = logits.argmax(1)
    score, pc = metrics(truth, prediction, len(CLASSES))
    probability = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    with (result / "sample_predictions.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("flow_id", "true_application", "true_class",
            "predicted_class", "correct", "max_probability"))
        writer.writeheader()
        for (uid, fine), y, pred, prob in zip(source, truth, prediction, probability, strict=True):
            writer.writerow({"flow_id": uid, "true_application": fine,
                "true_class": CLASSES[int(y)], "predicted_class": CLASSES[int(pred)],
                "correct": int(y == pred), "max_probability": float(prob.max())})
    np.save(result / "logits.npy", logits.astype(np.float32), allow_pickle=False)
    per_class = [{"class": name, "precision": float(pc[0][i]), "recall": float(pc[1][i]),
                  "f1": float(pc[2][i]), "support": int(pc[3][i])}
                 for i, name in enumerate(CLASSES)]
    with (result / "per_class.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("class", "precision", "recall", "f1", "support"))
        writer.writeheader()
        writer.writerows(per_class)
    confusion = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64)
    np.add.at(confusion, (truth, prediction), 1)
    with (result / "confusion_matrix.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true/pred", *CLASSES])
        for name, line in zip(CLASSES, confusion, strict=True):
            writer.writerow([name, *line.tolist()])
    cache_after = {str(p.relative_to(cache_root)): prior.sha256(p) for p in cache_files}
    if cache_after != cache_before or prior.freeze_sources() != source_hashes:
        raise RuntimeError("frozen input changed during Test evaluation")
    if any(prior.sha256(run / name) != digest for name, digest in checkpoint_hashes.items()):
        raise RuntimeError("selected checkpoint changed during Test evaluation")
    write_json(result / "results.json", {
        "status": "PASS", "protocol": prior.VNAT_PROTOCOL, "classes": CLASSES,
        "known_test_samples": len(source), "metrics": score, "per_class": per_class,
        "confusion_matrix": confusion.tolist(), "checkpoint_hashes": checkpoint_hashes,
        "stage32_test_cache_hashes": cache_before, "frozen_source_hashes_unchanged": True,
        "unknown_feature_values_loaded": 0, "test_parameter_selection": 0,
        "claim_scope": "post-hoc exploratory six-class development; Test already exposed at four classes",
    })
    (result / "SUCCESS").write_text("SUCCESS\n")
    print(json.dumps({"phase": "known_test", "status": "PASS", "metrics": score,
                      "per_class": per_class}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("preflight", "train", "evaluate"))
    phase = parser.parse_args().phase
    {"preflight": preflight, "train": train, "evaluate": evaluate}[phase]()


if __name__ == "__main__":
    main()
