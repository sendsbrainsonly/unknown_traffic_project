#!/usr/bin/env python3
"""Read-only one-shot aggregate progress for Stage41, including all three OD runs."""
from __future__ import annotations

import json
from pathlib import Path

from freeze_matched_protocol import PROJECT, ROOT


def status(name: str) -> str:
    path = PROJECT / ".tmux-task" / name / "exit.status"
    if not (PROJECT / ".tmux-task" / name).exists():
        return "not_started"
    if not path.exists():
        return "running"
    code = int(path.read_text().strip())
    return "done" if code == 0 else f"failed({code})"


def line_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def main() -> None:
    queue = ROOT / "queue_progress.json"
    info = json.loads(queue.read_text()) if queue.exists() else {"phase": "not_started"}
    print(f"Stage41 queue: {info.get('status', 'UNKNOWN')} / {info.get('phase', '?')}")
    interrupted_a1 = line_count(ROOT / "A-1/od_run/train_log.jsonl")
    print(f"  A-1 OD initial GPU0 attempt: interrupted after {interrupted_a1}/100 epochs; evidence preserved")
    for setting, gpu in (("A-1", 7), ("A-2", 6), ("A-3", 7)):
        label = setting.replace("-", "").lower()
        run_dir = "od_run_2gpu" if setting == "A-1" else "od_run"
        session = "stage41_od_a1_retry_gpu7_0927" if setting == "A-1" else f"stage41_od_{label}_gpu{gpu}_0927"
        epochs = line_count(ROOT / setting / run_dir / "train_log.jsonl")
        print(f"  {setting} OD GPU{gpu}: {epochs}/100 epochs; "
              f"{status(session)}; "
              f"score extraction {status(f'stage41_od_extract_{label}_0927')}")
    for setting in ("A-1", "A-3"):
        label = setting.replace("-", "").lower()
        runs = ROOT / setting / "runs/ustc/A-2"
        pieces = []
        for branch in ("trafficformer", "graph", "yatc"):
            session = f"stage41_{label}_{branch}_0927"
            folder = runs / branch
            if branch == "yatc":
                epoch = line_count(folder / "training_history.jsonl")
                total = 200
            else:
                epoch = line_count(folder / "training_history.json") if (folder / "SUCCESS").exists() else 0
                total = 20 if branch == "trafficformer" else 50
            pieces.append(f"{branch}:{status(session)}" +
                          (f" {epoch}/{total}" if branch == "yatc" else ""))
        print(f"  {setting} three-view: " + ", ".join(pieces) +
              f"; fusion:{status(f'stage41_{label}_fusion_0927')}" +
              f"; evaluation:{status(f'stage41_{label}_eval_0927')}")
    print("  paired comparison:", status("stage41_comparison_0927"))
    if (ROOT / "comparison_report.md").exists():
        print("  result:", ROOT / "comparison_report.md")


if __name__ == "__main__":
    main()
