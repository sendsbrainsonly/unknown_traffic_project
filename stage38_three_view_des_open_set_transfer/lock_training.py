#!/usr/bin/env python3
"""Record Stage 38B training recipe before observing Stage 38A Unknown scores."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
S31 = PROJECT / "stage31_four_dataset_three_view_equal"
S22 = PROJECT / "stage22_pretrained_trafficformer_e3_closed_set_comparison"
S23 = PROJECT / "stage23_closed_set_method_table"
TF_WEIGHT = PROJECT / "tf_runtime/code/models/pretrained_model.bin"
YATC_WEIGHT = PROJECT.parent / "YaTC/code/output_dir/pretrained-model.pth"
YATC_SHA = "66314aa57d4364bad0228835a181158f819d10c857936fbbdbae6dc483211fc6"


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for part in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(part)
    return value.hexdigest()


def main() -> None:
    audit = json.loads((ROOT / "stage38b_known_label_audit.json").read_text())
    if audit["status"] != "PASS" or audit["unknown_training_samples"] != 0:
        raise RuntimeError("Known-only label/cache audit not PASS")
    tf_config_path = S22 / "config.json"
    tf_config = json.loads(tf_config_path.read_text())
    tf_spec = tf_config["training"]["trafficformer"]
    if tf_spec["epochs"] != 20 or tf_spec["batch_size"] != 64 or tf_spec["learning_rate"] != 6e-5:
        raise RuntimeError("Stage22 TrafficFormer recipe changed")
    if sha(TF_WEIGHT) != tf_config["pretrained_model_sha256"]:
        raise RuntimeError("official TrafficFormer pretrained hash changed")
    # The Stage23 YaTC source records the author weight path/hash; hash the
    # actual file here before any new Known-only branch fitting.
    if not YATC_WEIGHT.is_file() or sha(YATC_WEIGHT) != YATC_SHA:
        raise RuntimeError("official YaTC pretrained hash/path changed")
    files = [S31 / name for name in ("train_tf_fig_branch.py", "train_yatc_branch.py",
                                      "train_equal_fusion.py", "preflight.py")]
    files += [S22 / "scripts/train_closed_run.py", S22 / "scripts/stage22_common.py",
              tf_config_path, S23 / "scripts/train_yatc_closed.py",
              PROJECT / "STAGE38_THREE_VIEW_DES_OPEN_SET_TRANSFER_PLAN.md"]
    value = {
        "status": "FROZEN_BEFORE_STAGE38A_UNKNOWN_RESULTS",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_seed": 2022,
        "known_label_audit": "stage38b_known_label_audit.json",
        "label_mapping": {"Chat": "Communication", "Email": "Communication",
                          "VoIP": "Communication", "Audio": "Streaming",
                          "Video": "Streaming", "Streaming": "Streaming",
                          "File-Transfer": "File-Transfer", "P2P": "P2P",
                          "Browsing": "Browsing"},
        "source_code_sha256": {str(path.relative_to(PROJECT)): sha(path) for path in files},
        "pretrained_sha256": {"trafficformer": sha(TF_WEIGHT), "yatc": YATC_SHA},
        "branch_recipes": {
            "trafficformer": {"epochs": 20, "batch_size": 64, "lr": 6e-5,
                              "initialization": "official_pretrained", "checkpoint": "Known-Val Macro-F1"},
            "graph": {"epochs": 50, "batch_size": 64, "lr": 1e-3,
                      "architecture": "TAGCN hidden128 k_hops2", "checkpoint": "Known-Val Macro-F1"},
            "yatc": {"epochs": 200, "batch_size": 64, "optimizer": "AdamW",
                     "effective_lr": 5e-4, "weight_decay": 0.05, "layer_decay": 0.75,
                     "warmup_epochs": 20, "label_smoothing": 0.1,
                     "checkpoint": "Known-Val Weighted-F1"},
            "adapters": {"epochs": 30, "batch_size": 256, "optimizer": "Adam lr=1e-3",
                         "loss": "mean CE + 0.05 mean KL",
                         "checkpoint": "Known-Val mean unimodal Macro-F1"},
            "fusion": {"epochs": 30, "batch_size": 256, "optimizer": "Adam lr=1e-3",
                       "weights": [1/3, 1/3, 1/3], "loss": "cross entropy",
                       "checkpoint": "Known-Val Macro-F1"},
        },
        "scaler_fit": "Known Train only", "unknown_training_samples": 0,
        "known_test_for_selection": 0, "unknown_for_selection": 0,
    }
    with (ROOT / "stage38b_training_config_lock.json").open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"status": "PASS", "locked_code_files": len(files),
                      "labels": [r["known_coarse_classes"] for r in audit["datasets"]]}), flush=True)


if __name__ == "__main__":
    main()
