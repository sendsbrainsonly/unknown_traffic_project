#!/usr/bin/env python3
"""Read-only Stage46 total progress; standard library only, no GPU initialization."""
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    state_path = ROOT / "queue_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    print(f"Stage46 {state.get('status','preflight')} | components {state.get('completed_components',0)}/6 | physicalGPU={state.get('physical_gpu_ids',[])}")
    print("original pretrained initialization | joint100epochs | seed2022 | budgets20/50/100")
    for dataset in ("iscx_vpn", "iscx_tor"):
        run = ROOT / "runs" / dataset
        history = []
        path = run / "training_history.jsonl"
        if path.exists():
            for line in path.read_text().splitlines():
                try:
                    history.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        if history:
            last = history[-1]
            remaining = (100 - last["epoch"]) * statistics.median(r["epoch_seconds"] for r in history[-5:]) / 60
            print(f"{dataset}: {last['epoch']}/100 | ValAcc={last['val_accuracy']:.4f} Macro={last['val_macro_f1']:.4f} Weighted={last['val_weighted_f1']:.4f} | bestMacro={last['best_val_macro_f1']:.4f}@{last['best_epoch']} | trainETA~{remaining:.0f}min")
        else:
            print(f"{dataset}: normalization/initialization or waiting; no completed epoch yet")
        for marker in ("TRAIN_FAILURE.json", "evaluation_complete.json"):
            if (run / marker).exists():
                print(f"  {marker}: {(run / marker).read_text().strip()}")
    if (ROOT / "completion_verification.json").exists():
        print((ROOT / "completion_verification.json").read_text())
    if state.get("error"):
        print("ERROR:", state["error"])


if __name__ == "__main__":
    main()
