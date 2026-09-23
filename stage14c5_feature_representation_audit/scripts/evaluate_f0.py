#!/usr/bin/env python3
"""Generate Known-Val diagnostics for frozen Stage 14C F0 checkpoints."""

from __future__ import annotations

import csv
import json

import torch

from common import RUNS_ROOT, STAGE14C_ROOT, json_dump, protocol_ids, source_input_dir, stage14c_common, verify_frozen_inputs
from model_utils import evaluate_predictions, make_model, write_diagnostics


def main() -> None:
    frozen = verify_frozen_inputs()
    with (STAGE14C_ROOT / "stage14c_training_summary.csv").open(encoding="utf-8", newline="") as handle:
        baseline = {f"{row['setting'].lower()}_seed{row['seed']}": row for row in csv.DictReader(handle)}
    if set(baseline) != set(protocol_ids()):
        raise RuntimeError("Stage 14C baseline/protocol grid mismatch")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda:0")
    for protocol_id in protocol_ids():
        run_dir = RUNS_ROOT / "f0" / protocol_id
        if run_dir.exists():
            raise RuntimeError(f"refusing to overwrite {run_dir}")
        run_dir.mkdir(parents=True)
        protocol = stage14c_common.load_protocol(protocol_id)
        row = baseline[protocol_id]
        checkpoint_path = STAGE14C_ROOT / "checkpoints" / f"{protocol_id}_best.pt"
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model = make_model(len(protocol["known_applications"]), device)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        input_dir = source_input_dir(protocol_id)
        truth, predictions = evaluate_predictions(model, input_dir, device)
        diagnostics = write_diagnostics("f0", protocol_id, run_dir, truth, predictions, input_dir)
        if abs(diagnostics["val_accuracy_recomputed"] - float(row["val_accuracy"])) > 1e-12 or abs(diagnostics["val_macro_f1_recomputed"] - float(row["val_macro_f1"])) > 1e-12:
            raise RuntimeError(f"{protocol_id}: F0 recomputation mismatch")
        result = {
            "feature": "f0", "protocol_id": protocol_id, "setting": protocol["setting"], "seed": int(protocol["seed"]),
            "selection_data": "Known Validation only", "unknown_samples_used": 0, "known_test_samples_used": 0,
            "best_epoch": int(row["best_epoch"]), "stop_epoch": int(row["stop_epoch"]),
            "train_loss": float(row["train_loss"]), "val_loss": float(row["val_loss"]),
            "val_accuracy": float(row["val_accuracy"]), "val_macro_f1": float(row["val_macro_f1"]),
            "checkpoint_path": str(checkpoint_path), "checkpoint_sha256": row["checkpoint_sha256"],
            "freeze_hash": frozen["freeze_hash"], "status": "success", "baseline_reused": True,
            **diagnostics,
        }
        json_dump(run_dir / "result.json", result)
        print(json.dumps({"event": "f0_diagnostics_complete", "protocol_id": protocol_id, "val_accuracy": result["val_accuracy"], "val_macro_f1": result["val_macro_f1"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
