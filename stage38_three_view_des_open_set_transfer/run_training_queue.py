#!/usr/bin/env python3
"""Finish the frozen Stage38B Known-only branches with at most three owned GPUs."""
from __future__ import annotations

import json
import os
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from run_iscx_known import ROOT, PROJECT, lock

WORKSPACE = PROJECT.parents[1]
HELPER = WORKSPACE / "skills/tmux-task-execution/scripts/tmux_task.sh"
SELECTOR = WORKSPACE / "skills/using-superpowers/scripts/select_gpu.py"
PYTHON = "/home/birkenwald/data/SEU-WXY/conda_envs/2025-10-8-WXY-dgl_py310/bin/python"
START = time.monotonic()
EXISTING = {
    "vpn_graph": "stage38b_vpn_graph_gpu2_v2_0926",
    "vpn_trafficformer": "stage38b_vpn_tf_gpu3_v2_0926",
    "vpn_yatc": "stage38b_vpn_yatc_gpu4_v2_0926",
    "tor_graph": "stage38b_tor_graph_gpu2_0926",
    "tor_yatc": "stage38b_tor_yatc_gpu2_0926",
}
NEW = {
    "tor_trafficformer": "stage38b_tor_tf_auto_0926",
    "vpn_fusion": "stage38b_vpn_fusion_auto_0926",
    "tor_fusion": "stage38b_tor_fusion_auto_0926",
}
SESSIONS = EXISTING | NEW


def update(status: str, phase: str, **details: object) -> None:
    payload = {"status": status, "phase": phase,
               "updated_at_utc": datetime.now(timezone.utc).isoformat(),
               "elapsed_seconds": round(time.monotonic() - START, 1),
               "max_parallel_owned_gpus": 3,
               "gpu2_release_deadline_utc": "2026-09-26T07:45:00+00:00",
               "sessions": SESSIONS, **details}
    tmp = ROOT / "training_queue_progress.json.tmp"
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp, ROOT / "training_queue_progress.json")
    print(json.dumps(payload), flush=True)


def status(key: str) -> int | None:
    path = PROJECT / ".tmux-task" / SESSIONS[key] / "exit.status"
    return int(path.read_text().strip()) if path.is_file() else None


def check_failures() -> None:
    for key in SESSIONS:
        code = status(key)
        if code is not None and code != 0:
            raise RuntimeError(f"Known-only worker {key} failed with exit {code}")


def wait_for(*keys: str) -> None:
    update("WAITING", "KNOWN_TRAINING", waiting_for=list(keys))
    while True:
        check_failures()
        if all(status(key) == 0 for key in keys):
            return
        time.sleep(30)


def launch(key: str, dataset: str, component: str, allowed: str) -> None:
    name = NEW[key]
    if (PROJECT / ".tmux-task" / name).exists():
        raise FileExistsError(f"refusing to reuse training session: {name}")
    command = [PYTHON, str(SELECTOR), "--min-free-gb", "25", "--allowed", allowed,
               "--max-utilization", "30", "--", PYTHON, str(ROOT / "run_iscx_known.py"),
               "train", "--dataset", dataset, "--component", component]
    update("STARTING", key, allowed_physical_gpus=allowed)
    subprocess.run([str(HELPER), "start", name, str(PROJECT), "--", *command],
                   cwd=PROJECT, check=True)
    update("RUNNING", key, session=name, allowed_physical_gpus=allowed)


def main() -> None:
    try:
        lock("iscx_vpn")
        lock("iscx_tor")
        wait_for("vpn_graph", "tor_graph", "vpn_yatc")
        launch("tor_trafficformer", "iscx_tor", "trafficformer", "4,7")
        wait_for("vpn_trafficformer", "vpn_yatc")
        launch("vpn_fusion", "iscx_vpn", "fusion", "3,7")
        wait_for("tor_trafficformer", "tor_yatc")
        launch("tor_fusion", "iscx_tor", "fusion", "4,7")
        wait_for("vpn_fusion", "tor_fusion")
        for dataset in ("iscx_vpn", "iscx_tor"):
            lock(dataset)
            for component in ("trafficformer", "graph", "yatc", "T0_equal"):
                path = ROOT / "runs" / dataset / "medium_seed2022" / component / "SUCCESS"
                if not path.is_file():
                    raise RuntimeError(f"missing successful Known-only model: {path}")
        update("KNOWN_TRAINING_COMPLETE", "STAGE38B", next_phase="frozen Test extraction")
    except BaseException as exc:
        (ROOT / "training_queue_failure.json").write_text(json.dumps({
            "status": "FAIL", "error": repr(exc), "traceback": traceback.format_exc(),
            "updated_at_utc": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n")
        update("FAILED", "STOPPED", error=repr(exc))
        raise


if __name__ == "__main__":
    main()
