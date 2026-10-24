#!/usr/bin/env python3
"""CPU-only checks for safe, exact-boundary Stage46 continuation."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

import resume_from_snapshot as resume


def main() -> None:
    exp = resume.exp
    exp.assert_locked()
    attempt = Path(__file__).resolve().parent
    results = []
    for dataset in exp.DATASETS:
        run = exp.run_dir(dataset)
        snapshot_path = run / "recovery_epoch075.pt"
        snapshot = torch.load(snapshot_path, map_location="cpu", weights_only=False, mmap=True)
        train_rows = exp.source.rows_for(dataset, "known_train")
        step_per_epoch = math.ceil(len(train_rows) / exp.BATCH)
        assert snapshot["epoch"] == 75
        assert snapshot["step"] == 75 * step_per_epoch
        assert snapshot["config"]["dataset"] == dataset
        assert snapshot["config"]["epochs"] == exp.EPOCHS
        assert snapshot["config"]["total_optimizer_steps"] == exp.EPOCHS * step_per_epoch

        history = [json.loads(line) for line in (run / "training_history.jsonl").read_text().splitlines()]
        prefix75 = [event for event in history if event["epoch"] <= 75]
        assert [event["epoch"] for event in prefix75] == list(range(1, 76))
        best = max(prefix75, key=lambda event: (event["val_macro_f1"], -event["epoch"]))
        assert best["epoch"] == snapshot["best_epoch"]
        assert abs(best["val_macro_f1"] - snapshot["best_val_macro_f1"]) < 1e-12

        scale = np.load(exp.ROOT / "inputs" / dataset / "graph_scaler.npz", allow_pickle=False)
        train_ds = exp.dataset_for(dataset, "known_train", scale["mean"], scale["std"])
        val_ds = exp.dataset_for(dataset, "known_validation", scale["mean"], scale["std"])
        loader = exp.joint.load_batches(exp.Indexed(train_ds), exp.MICRO, True)
        loader.generator.set_state(snapshot["loader_generator"])
        order = []
        for _, indices in loader:
            order.extend(indices.tolist())
        expected = np.load(run / "order_epoch076.npy", allow_pickle=False)
        assert np.array_equal(np.asarray(order), expected), f"{dataset}: epoch076 sample order mismatch"
        assert len(set(train_ds.ids)) == len(train_ds) and not (set(train_ds.ids) & set(val_ds.ids))
        assert not (attempt / "runs" / dataset).exists(), "continuation output directory already exists"
        results.append({
            "dataset": dataset,
            "resume_checkpoint": str(snapshot_path.relative_to(exp.PROJECT)),
            "resume_checkpoint_sha256": exp.sha(snapshot_path),
            "resume_epoch": 75,
            "step": int(snapshot["step"]),
            "best_epoch": int(snapshot["best_epoch"]),
            "best_val_macro_f1": float(snapshot["best_val_macro_f1"]),
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
            "epoch076_sample_order_exact": True,
            "known_test_used": 0,
            "unknown_used": 0,
        })
        del snapshot, train_ds, val_ds, loader, order

    report = {"status": "PASS", "device": "CPU-only", "tests": 4,
              "test_names": ["source_lock", "epoch_boundary_and_optimizer_step", "best_checkpoint_parity",
                             "epoch076_dataloader_order_replay"], "datasets": results,
              "frozen_sources_sha256": exp.sha(exp.ROOT / "frozen_sources.json"),
              "resume_script_sha256": exp.sha(attempt / "resume_from_snapshot.py"),
              "known_test_used": 0, "unknown_used": 0}
    (attempt / "preflight.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
