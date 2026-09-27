#!/usr/bin/env python3
"""Read-only aggregate progress for the Stage32 detached tmux run."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATASETS = ("iscx_vpn", "iscx_tor", "vnat")

LOG = ROOT.parent / ".tmux-task" / "codex_stage32_three_coarse_test_20260925" / "output.log"


def main() -> None:
    state = json.loads((ROOT / "progress.json").read_text())
    print(f"Stage32: {state['status']} | heads {state['completed_heads']}/{state['total_heads']} | "
          f"Test features opened {state['test_features_opened']}")
    if state["status"] == "TEST_INPUT_RECOVERY" and LOG.is_file():
        progress = []
        for line in LOG.read_text(errors="replace").splitlines():
            try:
                event = json.loads(line)
            except (ValueError, TypeError):
                continue
            if "capture" in event and event.get("protocol") == "medium_seed2025":
                progress.append(event)
        if progress:
            latest = progress[-1]
            phase = "TrafficFormer/FIG" if len(progress) <= latest["total_captures"] else "YaTC MFR"
            print(f"VNAT Test input recovery: {phase} capture {latest['capture']}/{latest['total_captures']} "
                  f"(completed capture events {len(progress)}/{2 * latest['total_captures']})")
    for dataset in DATASETS:
        train = state["datasets"][dataset]
        test = state.get("test_evaluation", {}).get(dataset, {})
        print(f"{dataset:9s} train={train['status']:7s} val_MacroF1={train.get('val_macro_f1')} "
              f"Test={test.get('status', 'LOCKED')} Test_MacroF1={test.get('macro_f1')}")
    print(f"updated: {state['updated_at_utc']}")


if __name__ == "__main__":
    main()
