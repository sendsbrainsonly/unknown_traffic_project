#!/usr/bin/env python3
"""Build the canonical Stage46 complete 1-100 epoch history and curves.

Uses original epochs 1-75 and the verified continuation epochs 76-100. It does
not read Test, Unknown, or prediction artifacts and never modifies parent logs.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
STAGE = HERE.parent


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    output = HERE / "combined"
    figures = HERE / "figures"
    output.mkdir(exist_ok=False)
    figures.mkdir(exist_ok=False)
    for dataset in ("iscx_vpn", "iscx_tor"):
        original = read_jsonl(STAGE / "runs" / dataset / "training_history.jsonl")
        resumed = read_jsonl(HERE / "runs" / dataset / "training_history.jsonl")
        prefix = [row for row in original if int(row["epoch"]) <= 75]
        if [int(row["epoch"]) for row in prefix] != list(range(1, 76)):
            raise RuntimeError(f"{dataset}: expected original epochs 1-75")
        if [int(row["epoch"]) for row in resumed] != list(range(76, 101)):
            raise RuntimeError(f"{dataset}: expected resumed epochs 76-100")
        rows = prefix + resumed
        history_path = output / f"{dataset}_training_history.jsonl"
        history_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        epochs = [int(row["epoch"]) for row in rows]
        for key in ("train_loss", "val_loss"):
            axes[0].plot(epochs, [row[key] for row in rows], label=key)
        for key in ("train_online_macro_f1", "val_macro_f1"):
            axes[1].plot(epochs, [row[key] for row in rows], label=key)
        for axis in axes:
            axis.set_xlabel("Epoch")
            axis.legend()
            for boundary in (20, 50, 100):
                axis.axvline(boundary, alpha=.2, color="grey")
        fig.suptitle(dataset + " Stage46 complete 100-epoch trajectory")
        fig.tight_layout()
        fig.savefig(figures / f"{dataset}_learning_curves.png", dpi=160)
        plt.close(fig)
        print(json.dumps({"dataset": dataset, "epochs": len(rows),
                          "first_epoch": epochs[0], "last_epoch": epochs[-1],
                          "history": str(history_path.relative_to(HERE)),
                          "figure": str((figures / f"{dataset}_learning_curves.png").relative_to(HERE))},
                         sort_keys=True))


if __name__ == "__main__":
    main()
