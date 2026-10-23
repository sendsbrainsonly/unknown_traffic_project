#!/usr/bin/env python3
"""Independent saved-score metric replay, calibration and sampled geometry checks."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
CFG = json.loads((ROOT / "config.json").read_text())
SOURCE = PROJECT / CFG["source"]


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def selected(ids, n, namespace):
    ordering = sorted(range(len(ids)), key=lambda i: hashlib.sha256(f"2022:{namespace}:{ids[i]}".encode()).hexdigest())
    return np.asarray(sorted(ordering[:n]), dtype=np.int64)


def main():
    torch.set_num_threads(4)
    max_error, rows_verified, samples_verified, geometry_samples = 0., 0, 0, 0
    for fold in CFG["folds"]:
        protocol = json.loads((SOURCE / "protocols" / fold / "protocol.json").read_text())
        source_rows = read_csv(SOURCE / "protocols" / fold / "role_manifest.csv")
        source_by_role = {role: sorted((r for r in source_rows if r["role"] == role), key=lambda r: r["flow_uid"])
                          for role in ("known_train", "known_validation", "known_test", "unknown_test")}
        classes = protocol["known_services"]
        init_hashes = set()
        for variant in CFG["variants"]:
            base = ROOT / "runs" / fold / variant
            meta = json.loads((base / "config.json").read_text())
            init_hashes.add(meta["initial_state_sha256"])
            assert meta["unknown_training_samples"] == meta["unknown_validation_samples"] == 0
            assert meta["active_mask"] == CFG["variants"][variant]
            assert meta["epochs"] == 30 and meta["batch_size"] == 256
            scores = read_csv(base / "sample_scores.csv")
            roles = {role: [r for r in scores if r["role"] == role]
                     for role in ("known_validation", "known_test", "unknown_test")}
            for role, rs in roles.items():
                expected = source_by_role[role]
                assert [r["flow_uid"] for r in rs] == [r["flow_uid"] for r in expected]
                assert [(r["service"], r["application"], r["vpn_status"]) for r in rs] == [
                    (r["service"], r["application"], r["vpn_status"]) for r in expected]
            samples_verified += len(scores)
            calibration = json.loads((base / "calibration.json").read_text())
            assert calibration["test_used"] == calibration["unknown_used"] == 0
            val = roles["known_validation"]
            for key, score_name in (("centroid", "score_centroid"), ("local", "score_local")):
                a = np.asarray([float(r[score_name]) for r in val])
                med = float(np.median(a))
                mad = float(np.median(np.abs(a - med)))
                assert med == calibration["median"][key] and mad == calibration["mad"][key]
            for role, rs in roles.items():
                dg = np.asarray([float(r["score_centroid"]) for r in rs])
                dl = np.asarray([float(r["score_local"]) for r in rs])
                recomputed = .5 * ((dg - calibration["median"]["centroid"]) / (calibration["mad"]["centroid"] + 1e-8)
                                  + (dl - calibration["median"]["local"]) / (calibration["mad"]["local"] + 1e-8))
                assert np.allclose(recomputed, [float(r["score_des_v1"]) for r in rs], rtol=1e-12, atol=1e-12)
            for method in CFG["detectors"]:
                threshold = float(np.percentile([float(r["score_" + method]) for r in val], 95, method="higher"))
                assert threshold == calibration["thresholds"][method]
                for row in scores:
                    assert int(row["prediction_" + method + "_p95"]) == int(float(row["score_" + method]) > threshold)
            truth = [classes.index(r["service"]) for r in roles["known_test"]]
            predicted = [classes.index(r["predicted_service"]) for r in roles["known_test"]]
            closed = json.loads((base / "closed_metrics.json").read_text())
            for key, value in {"accuracy": accuracy_score(truth, predicted),
                               "macro_f1": f1_score(truth, predicted, average="macro"),
                               "weighted_f1": f1_score(truth, predicted, average="weighted")}.items():
                assert abs(closed[key] - value) < 1e-12
            for row in read_csv(base / "open_metrics.csv"):
                a, b = roles["known_test"], roles["unknown_test"]
                if row["view"] == "balanced_1to1":
                    n = min(len(a), len(b))
                    ai = selected([r["flow_uid"] for r in a], n, fold + ":known")
                    bi = selected([r["flow_uid"] for r in b], n, fold + ":unknown")
                    a, b = [a[i] for i in ai], [b[i] for i in bi]
                y = np.r_[np.zeros(len(a), dtype=np.int64), np.ones(len(b), dtype=np.int64)]
                s = np.asarray([float(r["score_" + row["method"]]) for r in a + b])
                prediction = s > float(row["threshold"])
                values = {"auroc": roc_auc_score(y, s), "auprc": average_precision_score(y, s),
                          "binary_accuracy": accuracy_score(y, prediction),
                          "binary_precision": precision_score(y, prediction, zero_division=0),
                          "binary_recall": recall_score(y, prediction, zero_division=0),
                          "binary_f1": f1_score(y, prediction, zero_division=0),
                          "ufar": np.mean(~prediction[len(a):]), "known_frr": np.mean(prediction[:len(a)])}
                for key, value in values.items():
                    error = abs(float(row[key]) - float(value))
                    max_error = max(max_error, error)
                    assert error < 1e-12, (fold, variant, row["method"], key, error)
                rows_verified += 1
            # Check a fixed prefix of each role using CPU head replay, then exact
            # Euclidean arithmetic against the full frozen training support.
            checkpoint = torch.load(base / "head_best.pt", map_location="cpu", weights_only=False)
            state = checkpoint["state_dict"]
            fuser = torch.nn.Sequential(torch.nn.Linear(192, 128), torch.nn.ReLU(), torch.nn.Dropout(.1),
                                       torch.nn.Linear(128, 128), torch.nn.ReLU()).eval()
            fuser.load_state_dict({k.removeprefix("fuser."): v for k, v in state.items() if k.startswith("fuser.")})
            classifier = torch.nn.Linear(128, len(classes)).eval()
            classifier.load_state_dict({k.removeprefix("classifier."): v for k, v in state.items() if k.startswith("classifier.")})
            train_h = np.load(base / "known_train_representation.npz")["embedding"]
            labels = np.asarray([classes.index(r["service"]) for r in source_by_role["known_train"]])
            centroids = np.stack([train_h[labels == i].mean(0) for i in range(len(classes))])
            for role, rs in roles.items():
                n = min(8, len(rs))
                mu = np.load(ROOT / "latent_cache" / fold / f"{role}_mu.npy", allow_pickle=False)[:n].copy()
                mu *= np.asarray(CFG["variants"][variant], dtype=np.float32)[None, :, None]
                for i, enabled in enumerate(CFG["variants"][variant]):
                    if not enabled:
                        assert not np.any(mu[:, i])
                with torch.no_grad():
                    h = fuser(torch.from_numpy((mu / 3).reshape(n, 192))).numpy()
                    logits = classifier(torch.from_numpy(h)).numpy()
                with np.load(base / f"{role}_representation.npz") as saved:
                    assert np.allclose(h, saved["embedding"][:n], rtol=1e-4, atol=1e-4)
                    assert np.allclose(logits, saved["logits"][:n], rtol=1e-4, atol=1e-4)
                    saved_h = saved["embedding"][:n]
                for i in range(n):
                    distances = ((saved_h[i].astype(np.float64) - centroids) ** 2).sum(1)
                    label = int(distances.argmin())
                    local = np.sqrt(((train_h[labels == label].astype(np.float64) - saved_h[i]) ** 2).sum(1))
                    local = float(np.partition(local, 9)[:10].mean())
                    assert np.isclose(distances[label], float(rs[i]["score_centroid"]), rtol=1e-8, atol=1e-8)
                    assert np.isclose(local, float(rs[i]["score_local"]), rtol=1e-4, atol=1e-4)
                    geometry_samples += 1
        assert len(init_hashes) == 1, (fold, "different initial weights across variants")
    assert rows_verified == 128
    frozen = json.loads((ROOT / "HEADS_FROZEN.json").read_text())
    assert len(frozen["checkpoint_sha256"]) == 16
    assert all(digest(ROOT / path) == value for path, value in frozen["checkpoint_sha256"].items())
    result = {"status": "PASS", "metric_rows": rows_verified, "sample_score_rows": samples_verified,
              "max_metric_error": max_error, "sampled_head_and_geometry_replays": geometry_samples,
              "shared_initialization_each_fold": True, "known_only_calibration": True}
    (ROOT / "independent_replay_verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
