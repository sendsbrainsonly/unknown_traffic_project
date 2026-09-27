#!/usr/bin/env python3
"""Fail-closed unattended continuation under one named project-local tmux task."""
from __future__ import annotations

import json
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parents[1]
PROJECT = OUT.parent
WORKSPACE = PROJECT.parent.parent
HELPER = WORKSPACE / ".agents/skills/tmux-task-execution/scripts/tmux_task.sh"
SELECTOR = WORKSPACE / "skills/using-superpowers/scripts/select_gpu.py"
SESSIONS = {
    ("iscx_vpn", 2022): "codex_stage23b_tfe_vpn2022_20260924",
    ("iscx_vpn", 2023): "codex_stage23b_tfe_vpn2023_20260924",
    ("iscx_tor", 2022): "codex_stage23b_tfe_tor2022_20260924",
    ("iscx_tor", 2023): "codex_stage23b_tfe_tor2023_20260924",
}
WORKDIR = "Projects/unknown_traffic_project"


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def status(session: str):
    path = PROJECT / ".tmux-task" / session / "exit.status"
    return int(path.read_text().strip()) if path.is_file() else None


def run_task(session: str, command: list[str], *, background=False):
    action = "start" if background else "run"
    args = ["bash", str(HELPER), action, session, WORKDIR, "--", *command]
    print(json.dumps({"event": "task_launch", "session": session, "action": action,
                      "command": command, "utc": timestamp()}), flush=True)
    result = subprocess.run(args, cwd=WORKSPACE, check=False, text=True)
    if result.returncode:
        raise RuntimeError(f"{session} helper returned {result.returncode}")


def progress(stage: str):
    records = {}
    for (dataset, seed), session in SESSIONS.items():
        run = OUT / "runs" / "tfe" / dataset / f"seed{seed}"
        history = run / "training_history.jsonl"
        records[f"{dataset}-{seed}"] = {
            "session": session, "exit_code": status(session),
            "completed_epochs": sum(1 for _ in history.open(encoding="utf-8")) if history.is_file() else 0,
            "selected": (run / "SELECTION_COMPLETE").is_file(),
            "test_complete": (run / "SUCCESS").is_file(),
        }
    output = {"stage": stage, "utc": timestamp(), "tfe": records,
              "trident_selected": sum((OUT / "runs/trident" / d / f"seed{s}" / "SELECTION_COMPLETE").is_file()
                                      for d in ("iscx_vpn", "iscx_tor") for s in (2022, 2023)),
              "stage20_test_materialized": {d: (OUT / "inputs" / d / "test/input_audit.json").is_file()
                                             for d in ("iscx_vpn", "iscx_tor")}}
    (OUT / "progress.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "progress", **output}), flush=True)


def wait_for_training():
    final = ("iscx_tor", 2023)
    while True:
        completed = 0
        active = 0
        for (dataset, seed), session in SESSIONS.items():
            state = status(session)
            selected = (OUT / "runs/tfe" / dataset / f"seed{seed}" / "SELECTION_COMPLETE").is_file()
            if state is not None and (state != 0 or not selected):
                raise RuntimeError(f"TFE training failed or incomplete: {session} state={state}")
            if state == 0:
                completed += 1
            elif state is None and (PROJECT / ".tmux-task" / session).exists():
                active += 1
        if completed == 4:
            return
        if not (PROJECT / ".tmux-task" / SESSIONS[final]).exists() and active < 3:
            run_task(SESSIONS[final], ["python", str(SELECTOR), "--min-free-gb", "12",
                     "--count", "1", "--max-utilization", "30",
                     "--allowed", "0,1,2,3,4,5,6,7", "--",
                     "python", "-B", str(OUT / "scripts/run_tfe.py"),
                     "--dataset", "iscx_tor", "--seed", "2023", "--phase", "train"],
                     background=True)
        progress("training")
        time.sleep(30)


def main():
    for dataset in ("iscx_vpn", "iscx_tor"):
        for seed in (2022, 2023):
            if not (OUT / "runs/trident" / dataset / f"seed{seed}" / "SELECTION_COMPLETE").is_file():
                raise RuntimeError(f"Trident selection missing: {dataset}/{seed}")
    wait_for_training()
    progress("post_selection")
    for dataset in ("iscx_vpn", "iscx_tor"):
        run_task(f"codex_stage23b_extract_{dataset}_test_20260924",
                 ["python", "-B", str(OUT / "scripts/build_inputs.py"),
                  "--dataset", dataset, "--phase", "test"])
    for dataset in ("iscx_vpn", "iscx_tor"):
        for seed in (2022, 2023):
            run_task(f"codex_stage23b_trident_{dataset}_{seed}_test_20260924",
                     ["python", "-B", str(OUT / "scripts/run_trident.py"),
                      "--dataset", dataset, "--seed", str(seed), "--phase", "test"])
    for dataset in ("iscx_vpn", "iscx_tor"):
        for seed in (2022, 2023):
            run_task(f"codex_stage23b_tfe_{dataset}_{seed}_test_20260924",
                     ["python", str(SELECTOR), "--min-free-gb", "12", "--count", "1",
                      "--max-utilization", "30", "--allowed", "0,1,2,3,4,5,6,7",
                      "--", "python", "-B", str(OUT / "scripts/run_tfe.py"),
                      "--dataset", dataset, "--seed", str(seed), "--phase", "test"])
    progress("finalizing")
    run_task("codex_stage23b_final_replay_20260924",
             ["python", "-B", str(OUT / "scripts/finalize.py")])
    (OUT / "QUEUE_COMPLETE").write_text("PASS\n", encoding="utf-8")
    progress("complete")


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        (OUT / "queue_failure.json").write_text(json.dumps({
            "utc": timestamp(), "error_type": type(exc).__name__,
            "error": str(exc), "traceback": traceback.format_exc(),
            "partial_artifacts_preserved": True}, indent=2) + "\n", encoding="utf-8")
        raise
