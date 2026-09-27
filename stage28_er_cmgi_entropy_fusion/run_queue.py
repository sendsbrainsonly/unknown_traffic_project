#!/usr/bin/env python3
"""Bounded three-GPU execution queue for the remaining Stage28 Known-only units."""
from __future__ import annotations

import json
import subprocess
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from preflight import OUT, PROJECT

HELPER = PROJECT / ".." / ".." / "skills" / "tmux-task-execution" / "scripts" / "tmux_task.sh"
SELECT = PROJECT / ".." / ".." / "skills" / "using-superpowers" / "scripts" / "select_gpu.py"
PRESERVE = PROJECT / ".." / ".." / "skills" / "experiment-data-preservation" / "scripts"
PROGRESS = OUT / "queue_progress.json"
LOCK = threading.Lock()
RUNS = [(dataset, encoder, train) for dataset in ("iscx_vpn", "iscx_tor")
        for encoder in (2022, 2023) for train in range(2022, 2027)
        if (dataset, encoder, train) != ("iscx_vpn", 2022, 2022)]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def identity(unit: tuple[str, int, int]) -> str:
    dataset, encoder, train = unit
    return f"{dataset}/encoder{encoder}_train{train}"


def save_progress(state: dict) -> None:
    with LOCK:
        state["updated_at_utc"] = now()
        temp = OUT / "queue_progress.json.next"
        temp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
        temp.replace(PROGRESS)


def call(session: str, args: list[str]) -> str:
    cmd = [str(HELPER), "run", session, str(PROJECT), "--", *args]
    process = subprocess.run(cmd, cwd=PROJECT, text=True, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, check=False)
    if process.returncode:
        raise RuntimeError(f"{session}: exit {process.returncode}; see .tmux-task/{session}/output.log; "
                           f"tail={process.stdout[-1000:]}")
    return process.stdout


def worker(gpu: int, units: list[tuple[str, int, int]], state: dict) -> None:
    for dataset, encoder, train in units:
        unit = (dataset, encoder, train)
        key = identity(unit)
        run = OUT / "runs" / dataset / f"encoder{encoder}_train{train}"
        suffix = f"{dataset}_{encoder}_{train}"
        base_args = ["--dataset", dataset, "--encoder-seed", str(encoder), "--training-seed", str(train)]
        try:
            state["units"][key].update({"state": "initializing", "gpu": gpu, "started_at_utc": now()})
            save_progress(state)
            call(f"codex_s28_init_{suffix}", ["python", str(PRESERVE / "init_experiment_bundle.py"),
                 str(run), "--id", f"stage28-{suffix}", "--objective",
                 "Known-only Stage28 P0-P3 ER-CMGI-inspired fusion diagnostic",
                 "--experiment-type", "ablation", "--claim-scope", "diagnostic", "--status", "running",
                 "--tmux-session", f"codex_s28_train_{suffix}"])
            for stage, script in (("primary", "run_one.py"), ("baselines", "baselines.py")):
                state["units"][key]["state"] = stage
                save_progress(state)
                call(f"codex_s28_{stage}_{suffix}", ["python", str(SELECT), "--allowed", str(gpu),
                     "--min-free-gb", "8", "--", "python", str(OUT / script), *base_args])
            state["units"][key]["state"] = "verifying"
            save_progress(state)
            call(f"codex_s28_finalize_{suffix}", ["python", str(OUT / "finalize_run.py"), *base_args])
            manifest_path = run / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["execution"]["physical_gpu_ids"] = [gpu]
            manifest["execution"]["command"] = f"run_one.py and baselines.py {dataset} {encoder} {train}"
            manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
            call(f"codex_s28_validate_{suffix}", ["bash", "-c",
                 f"python {PRESERVE / 'refresh_artifact_manifest.py'} {run} && "
                 f"python {PRESERVE / 'validate_experiment_bundle.py'} {run} --verify-hashes"])
            state["units"][key].update({"state": "success", "finished_at_utc": now()})
            print(json.dumps({"unit": key, "status": "success", "gpu": gpu}), flush=True)
        except BaseException as exc:
            state["units"][key].update({"state": "failed", "finished_at_utc": now(), "error": str(exc)})
            print(json.dumps({"unit": key, "status": "failed", "error": str(exc)}), flush=True)
            if run.is_dir() and (run / "manifest.json").is_file():
                (run / "queue_failure.txt").write_text(traceback.format_exc())
                manifest = json.loads((run / "manifest.json").read_text())
                manifest["status"] = "failed"
                manifest["execution"]["physical_gpu_ids"] = [gpu]
                manifest["limitations"] = [str(exc)]
                (run / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
                (run / "RESULTS.md").write_text(
                    f"# Stage28 {key}\n\n- Status: failed; all partial evidence retained.\n\n"
                    f"## Data and split\n\nKnown Train/Validation only.\n\n"
                    f"## Configuration and execution\n\nPhysical GPU {gpu}; see queue failure and tmux logs.\n\n"
                    f"## Core results\n\nIncomplete; no scientific conclusion.\n\n"
                    f"## Preserved evidence\n\nExisting files and queue_failure.txt.\n\n"
                    f"## Limitations\n\n{exc}\n\n"
                    f"## Conclusion and next step\n\nDiagnose before rerun under a new ID.\n")
        finally:
            save_progress(state)


def main() -> None:
    if PROGRESS.exists():
        raise RuntimeError("queue_progress.json already exists; refusing second queue")
    state = {"status": "running", "started_at_utc": now(), "max_parallel_gpu_jobs": 3,
             "planned_units": 20, "pilot_completed": 1,
             "units": {identity(unit): {"state": "pending"} for unit in RUNS}}
    save_progress(state)
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(worker, gpu, RUNS[gpu::3], state) for gpu in range(3)]
        for future in futures:
            future.result()
    success = sum(row["state"] == "success" for row in state["units"].values())
    state["status"] = "success" if success == len(RUNS) else "partial_failure"
    state["completed_remaining"] = success
    state["failed_remaining"] = len(RUNS) - success
    save_progress(state)
    print(json.dumps({"queue_status": state["status"], "completed_remaining": success,
                      "failed_remaining": len(RUNS) - success}), flush=True)


if __name__ == "__main__":
    main()
