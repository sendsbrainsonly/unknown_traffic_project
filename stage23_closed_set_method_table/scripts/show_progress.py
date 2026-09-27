#!/usr/bin/env python3
"""Read-only CLI progress for Stage23 matched closed-set runs."""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1]
PLANNED = [(dataset, seed) for dataset in ("iscx_vpn", "iscx_tor") for seed in (2022, 2023)]


def main() -> None:
    complete = 0
    print("Stage23 Open-Detect corrected-paper (Stage20 matched)")
    print("dataset    seed  state       epochs  best_val_acc  best_epoch")
    for dataset, seed in PLANNED:
        run = OUT / "runs" / "opendetect_corrected_paper" / dataset / f"seed{seed}"
        if (run / "SUCCESS").is_file() and (run / "metrics.json").is_file():
            data = json.loads((run / "metrics.json").read_text())
            state = "SUCCESS"
            epochs = data["completed_epochs"]
            best = data["validation"]["accuracy"]
            best_epoch = data["best_epoch"]
            complete += 1
        elif (run / "FAILURE.json").is_file():
            state, epochs, best, best_epoch = "FAILED", 0, None, None
        elif (run / "training_history.jsonl").is_file():
            history = (run / "training_history.jsonl").read_text().splitlines()
            latest = json.loads(history[-1]) if history else {}
            state = "RUNNING_OR_INTERRUPTED"
            epochs = int(latest.get("epoch", 0))
            best = latest.get("best_validation_accuracy")
            best_epoch = latest.get("best_epoch")
        else:
            state, epochs, best, best_epoch = "NOT_STARTED", 0, None, None
        best_text = f"{best:.6f}" if isinstance(best, float) else "-"
        print(f"{dataset:<11}{seed:<6}{state:<24}{epochs:<8}{best_text:<14}{best_epoch or '-'}")
    print(f"formal completed: {complete}/{len(PLANNED)}; Stage22 verified E1/E3 baseline rows: 8")
    print()
    print("Stage23 RoNeTC runnable reconstruction (Stage20 matched)")
    print("dataset    seed  state       epochs  best_val_acc  best_epoch")
    ronet_complete = 0
    for dataset, seed in PLANNED:
        run = OUT / "runs" / "ronetc_stage20" / dataset / f"seed{seed}_formal"
        if (run / "SUCCESS").is_file() and (run / "metrics.json").is_file():
            data = json.loads((run / "metrics.json").read_text())
            state, epochs = "SUCCESS", data["completed_epochs"]
            best, best_epoch = data["validation"]["accuracy"], data["best_epoch"]
            ronet_complete += 1
        elif (run / "FAILURE.json").is_file():
            state, epochs, best, best_epoch = "FAILED", 0, None, None
        elif (run / "training_history.jsonl").is_file():
            history = (run / "training_history.jsonl").read_text().splitlines()
            latest = json.loads(history[-1]) if history else {}
            state, epochs = "RUNNING_OR_INTERRUPTED", int(latest.get("epoch", 0))
            best, best_epoch = latest.get("best_validation_accuracy"), latest.get("best_epoch")
        elif run.is_dir():
            state, epochs, best, best_epoch = "PREFLIGHT_OR_START", 0, None, None
        else:
            state, epochs, best, best_epoch = "NOT_STARTED", 0, None, None
        best_text = f"{best:.6f}" if isinstance(best, float) else "-"
        print(f"{dataset:<11}{seed:<6}{state:<24}{epochs:<8}{best_text:<14}{best_epoch or '-'}")
    print(f"formal completed: {ronet_complete}/{len(PLANNED)}")
    print()
    print("Stage23 YaTC official-pretrained (Stage20 matched)")
    print("dataset    seed  state       epochs  best_val_wf1  best_epoch  test")
    yatc_complete = 0
    for dataset, seed in PLANNED:
        run = OUT / "runs" / "yatc_stage20" / dataset / f"seed{seed}_formal"
        evaluation = run / "test_evaluation_cuda.json"
        if (run / "SUCCESS").is_file() and (run / "metrics.json").is_file():
            data = json.loads((run / "metrics.json").read_text())
            state, epochs = "SUCCESS", data["completed_epochs"]
            best, best_epoch = data["validation"]["weighted_f1"], data["best_epoch"]
            yatc_complete += 1
        elif (run / "FAILURE.json").is_file():
            state, epochs, best, best_epoch = "FAILED", 0, None, None
        elif (run / "training_history.jsonl").is_file():
            history = (run / "training_history.jsonl").read_text().splitlines()
            latest = json.loads(history[-1]) if history else {}
            state, epochs = "RUNNING_OR_INTERRUPTED", int(latest.get("epoch", 0))
            best, best_epoch = latest.get("best_validation_weighted_f1"), latest.get("best_epoch")
        elif run.is_dir():
            state, epochs, best, best_epoch = "PREFLIGHT_OR_START", 0, None, None
        else:
            state, epochs, best, best_epoch = "NOT_STARTED", 0, None, None
        best_text = f"{best:.6f}" if isinstance(best, float) else "-"
        test_state = "PASS" if evaluation.is_file() else "-"
        print(f"{dataset:<11}{seed:<6}{state:<24}{epochs:<8}{best_text:<14}{best_epoch or '-':<12}{test_state}")
    print(f"formal completed: {yatc_complete}/{len(PLANNED)}")
    print("RUNNING_OR_INTERRUPTED requires tmux status or exit.status to distinguish live from interrupted.")


if __name__ == "__main__":
    main()
