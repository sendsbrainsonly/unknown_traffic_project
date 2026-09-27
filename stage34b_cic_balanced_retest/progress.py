#!/usr/bin/env python3
"""Print one read-only aggregate progress snapshot for the balanced CIC run."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
JOBS = (
    ("graph", "stage34b_graph_0926", 50, "E2"),
    ("trafficformer", "stage34b_trafficformer_0926", 20, "E1"),
    ("yatc", "stage34b_yatc_0926", 200, "yatc"),
    ("fusion", "stage34b_fusion_0926", 60, "fusion"),
    ("test_inputs", "stage34b_test_inputs_0926", None, None),
    ("test", "stage34b_test_0926", None, None),
    ("finalizer", "stage34b_finalize_0926", None, None),
)


def current_epoch(log: Path, kind: str | None) -> tuple[str, str]:
    if kind is None or not log.is_file():
        return "-", "-"
    latest = None
    for line in log.read_text(errors="replace").splitlines():
        if not line.startswith("{"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if kind == "fusion":
            if row.get("phase") in ("adapters", "T0_equal") and "epoch" in row:
                latest = row
        elif kind == "yatc":
            if row.get("branch") == "yatc" and "epoch" in row:
                latest = row
        elif row.get("encoder") == kind and "epoch" in row:
            latest = row
    if latest is None:
        return "-", "-"
    name = f"{latest.get('phase')}:" if kind == "fusion" else ""
    metric = latest.get("val_macro_f1", latest.get("val_mean_macro_f1"))
    return name + str(latest["epoch"]), f"{metric:.6f}" if isinstance(metric, (float, int)) else "-"


def main() -> None:
    progress = ROOT / "queue_progress.json"
    if progress.is_file():
        state = json.loads(progress.read_text())
        print(f"queue={state['status']} phase={state['phase']} updated={state['updated_at_utc']}")
    else:
        print("queue=NOT_STARTED")
    print("job              status       epoch/budget  val_macro_f1")
    for label, session, budget, kind in JOBS:
        folder = PROJECT / ".tmux-task" / session
        exit_file = folder / "exit.status"
        state = f"exit={exit_file.read_text().strip()}" if exit_file.is_file() else (
            "running" if folder.is_dir() else "pending")
        epoch, metric = current_epoch(folder / "output.log", kind)
        position = f"{epoch}/{budget}" if budget is not None else epoch
        print(f"{label:<16} {state:<12} {position:<13} {metric}")
    completion = ROOT / "completion_verification.json"
    if completion.is_file():
        outcome = json.loads(completion.read_text())
        print(f"completion={outcome['status']} test_macro_f1={outcome.get('macro_f1', '-')}")


if __name__ == "__main__":
    main()
