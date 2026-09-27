"""Run the frozen Stage39 Known-only settings, three branch GPUs at most."""
from __future__ import annotations

import csv
import json
import os
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from freeze_protocols import PROJECT, ROOT, sha

WORKSPACE = PROJECT.parents[1]
HELPER = WORKSPACE / "skills/tmux-task-execution/scripts/tmux_task.sh"
SELECTOR = WORKSPACE / "skills/using-superpowers/scripts/select_gpu.py"
PYTHON = "/home/birkenwald/data/SEU-WXY/conda_envs/2025-10-8-WXY-dgl_py310/bin/python"
LANES = {"trafficformer": 0, "graph": 1, "yatc": 7, "fusion": 1}
START = time.monotonic()


def save(status: str, dataset: str = "", fold: str = "", phase: str = "", **extra):
    payload = {"status": status, "dataset": dataset, "fold": fold, "phase": phase,
               "updated_at_utc": datetime.now(timezone.utc).isoformat(),
               "elapsed_seconds": round(time.monotonic() - START, 1),
               "max_parallel_owned_gpus": 3, "allowed_physical_gpu_ids": [0, 1, 7], **extra}
    temp = ROOT / "training_queue_progress.json.tmp"
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temp, ROOT / "training_queue_progress.json")
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def session_name(dataset: str, fold: str, component: str) -> str:
    return f"stage39_{dataset}_{fold}_{component}_auto_0926"


def exit_code(session: str):
    path = PROJECT / ".tmux-task" / session / "exit.status"
    return int(path.read_text().strip()) if path.is_file() else None


def wait_for(dataset: str, fold: str, components: tuple[str, ...]):
    sessions = {component: session_name(dataset, fold, component) for component in components}
    while True:
        codes = {component: exit_code(session) for component, session in sessions.items()}
        if any(code is not None and code != 0 for code in codes.values()):
            raise RuntimeError(f"Stage39 training worker failed: {dataset}/{fold}/{codes}")
        if all(code == 0 for code in codes.values()):
            return
        save("RUNNING", dataset, fold, "training", worker_exit_codes=codes)
        time.sleep(30)


def launch(dataset: str, fold: str, component: str):
    name = session_name(dataset, fold, component)
    if (PROJECT / ".tmux-task" / name).exists():
        raise FileExistsError(f"refusing to reuse session: {name}")
    card = LANES[component]
    command = [PYTHON, str(SELECTOR), "--allowed", str(card), "--min-free-gb", "25",
               "--max-utilization", "30", "--", PYTHON, str(ROOT / "train_fold_component.py"),
               "--dataset", dataset, "--fold", fold, "--component", component]
    subprocess.run([str(HELPER), "start", name, str(PROJECT), "--", *command],
                   cwd=PROJECT, check=True)


def verify_model(dataset: str, fold: str):
    root = ROOT / "settings" / dataset / fold
    protocol = json.loads((root / "protocol.json").read_text())
    if protocol["status"] != "FROZEN_PRETRAIN" or sha(root / "role_manifest.csv") != protocol["role_manifest_sha256"]:
        raise RuntimeError(f"frozen protocol changed: {dataset}/{fold}")
    base = root / "runs" / ("vnat" if dataset == "VNAT" else dataset) / protocol["protocol_id"]
    for component in ("trafficformer", "graph", "yatc", "T0_equal"):
        if not (base / component / "SUCCESS").is_file():
            raise RuntimeError(f"model SUCCESS missing: {base / component}")
    return protocol


def main():
    try:
        with (ROOT / "protocol_index.csv").open(newline="", encoding="utf-8") as stream:
            planned = [r for r in csv.DictReader(stream) if r["new_training_required"] == "True"]
        if len(planned) != 12:
            raise RuntimeError(f"expected 12 frozen new-training settings, got {len(planned)}")
        completed = []
        for row in planned:
            dataset, fold = row["dataset"], row["slug"]
            root = ROOT / "settings" / dataset / fold
            protocol = json.loads((root / "protocol.json").read_text())
            if sha(root / "role_manifest.csv") != protocol["role_manifest_sha256"]:
                raise RuntimeError(f"role manifest hash changed: {dataset}/{fold}")
            if (root / "runs" / ("vnat" if dataset == "VNAT" else dataset) /
                protocol["protocol_id"] / "T0_equal" / "SUCCESS").is_file():
                verify_model(dataset, fold)
                completed.append(f"{dataset}/{fold}")
                save("RUNNING", dataset, fold, "already_completed", completed=completed)
                continue
            if not (root / "cache_subset_verification.json").is_file():
                save("RUNNING", dataset, fold, "preparing_known_cache", completed=completed)
                subprocess.run([PYTHON, str(ROOT / "build_fold_cache.py"), "--dataset", dataset,
                                "--fold", fold], cwd=PROJECT, check=True)
            cache = json.loads((root / "cache_subset_verification.json").read_text())
            if cache["status"] != "PASS" or cache["unknown_training_samples"] or cache["unknown_validation_samples"]:
                raise RuntimeError(f"Known-only cache audit failed: {dataset}/{fold}")
            save("RUNNING", dataset, fold, "launching_three_branches", completed=completed)
            for component in ("trafficformer", "graph", "yatc"):
                launch(dataset, fold, component)
            wait_for(dataset, fold, ("trafficformer", "graph", "yatc"))
            save("RUNNING", dataset, fold, "launching_fusion", completed=completed)
            launch(dataset, fold, "fusion")
            wait_for(dataset, fold, ("fusion",))
            verify_model(dataset, fold)
            completed.append(f"{dataset}/{fold}")
            save("RUNNING", dataset, fold, "setting_complete", completed=completed)
        save("KNOWN_TRAINING_COMPLETE", phase="all_frozen_settings", completed=completed,
             next_phase="Known-only detector calibration then single evaluation")
    except BaseException as exc:
        failure_path = ROOT / "training_queue_failure.json"
        if failure_path.exists():
            failure_path = ROOT / f"training_queue_failure_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
        failure_path.write_text(json.dumps({
            "status": "FAILED", "error": repr(exc), "traceback": traceback.format_exc(),
            "updated_at_utc": datetime.now(timezone.utc).isoformat()}, ensure_ascii=False, indent=2) + "\n")
        save("FAILED", phase="stopped", error=repr(exc))
        raise


if __name__ == "__main__":
    main()
