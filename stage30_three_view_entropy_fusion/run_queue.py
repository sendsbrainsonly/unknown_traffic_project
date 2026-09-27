#!/usr/bin/env python3
"""Run remaining Stage30 units on at most three independently selected GPUs."""
from __future__ import annotations

import json
import subprocess
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from preflight import OUT, PROJECT

HELPER = PROJECT / ".." / ".." / "skills" / "tmux-task-execution" / "scripts" / "tmux_task.sh"
SELECT = PROJECT / ".." / ".." / "skills" / "using-superpowers" / "scripts" / "select_gpu.py"
PRESERVE = PROJECT / ".." / ".." / "skills" / "experiment-data-preservation" / "scripts"
PROGRESS = OUT / "queue_progress.json"
LOCK = threading.RLock()
UNITS = [(dataset, encoder, train) for dataset in ("iscx_vpn", "iscx_tor")
         for encoder in (2022, 2023) for train in range(2022, 2027)
         if (dataset, encoder, train) != ("iscx_vpn", 2022, 2022)]


def now():
    return datetime.now(timezone.utc).isoformat()


def key(unit):
    return f"{unit[0]}/encoder{unit[1]}_train{unit[2]}"


def record(progress, unit, **updates):
    with LOCK:
        progress["units"][key(unit)].update(updates)
        progress["updated_at_utc"] = now()
        path = OUT / "queue_progress.json.next"
        path.write_text(json.dumps(progress, indent=2, sort_keys=True) + "\n")
        path.replace(PROGRESS)


def call(session, command):
    process = subprocess.run([str(HELPER), "run", session, str(PROJECT), "--", *command],
                             cwd=PROJECT, text=True, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, check=False)
    if process.returncode:
        raise RuntimeError(f"{session} exit {process.returncode}; .tmux-task/{session}/output.log; "
                           f"tail={process.stdout[-800:]}")
    return process.stdout


def worker(gpu, units, progress):
    for unit in units:
        dataset, encoder, train = unit
        suffix = f"{dataset}_{encoder}_{train}"
        run = OUT / "runs" / dataset / f"encoder{encoder}_train{train}"
        args = ["--dataset", dataset, "--encoder-seed", str(encoder), "--training-seed", str(train)]
        try:
            record(progress, unit, state="initializing", gpu=gpu, started_at_utc=now())
            call(f"codex_s30_init_{suffix}", ["python", str(PRESERVE / "init_experiment_bundle.py"),
                 str(run), "--id", f"stage30-{suffix}", "--objective",
                 "Three-view Known-only equal versus entropy-fusion validation",
                 "--experiment-type", "ablation", "--claim-scope", "diagnostic", "--status", "running",
                 "--tmux-session", f"codex_s30_train_{suffix}"])
            record(progress, unit, state="training")
            call(f"codex_s30_train_{suffix}", ["python", str(SELECT), "--allowed", str(gpu),
                 "--min-free-gb", "8", "--", "python", str(OUT / "run_one.py"), *args])
            record(progress, unit, state="verifying")
            call(f"codex_s30_verify_{suffix}", ["python", str(OUT / "finalize_run.py"), *args])
            manifest_path = run / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["execution"]["physical_gpu_ids"] = [gpu]
            manifest["execution"]["command"] = f"run_one.py {dataset} {encoder} {train}"
            manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
            call(f"codex_s30_validate_{suffix}", ["bash", "-c",
                 f"python {PRESERVE / 'refresh_artifact_manifest.py'} {run} && "
                 f"python {PRESERVE / 'validate_experiment_bundle.py'} {run} --verify-hashes"])
            record(progress, unit, state="success", finished_at_utc=now())
            print(json.dumps({"unit": key(unit), "status": "success", "gpu": gpu}), flush=True)
        except BaseException as exc:
            record(progress, unit, state="failed", finished_at_utc=now(), error=str(exc))
            print(json.dumps({"unit": key(unit), "status": "failed", "error": str(exc)}), flush=True)
            if run.is_dir() and (run / "manifest.json").is_file():
                (run / "queue_failure.txt").write_text(traceback.format_exc())
                manifest = json.loads((run / "manifest.json").read_text())
                manifest["status"] = "failed"
                manifest["execution"]["physical_gpu_ids"] = [gpu]
                manifest["limitations"] = [str(exc)]
                (run / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
                (run / "RESULTS.md").write_text(
                    f"# Stage30 {key(unit)}\n\n- Status: failed; partial evidence retained.\n\n"
                    f"## Data and split\n\nKnown Train/Val only.\n\n"
                    f"## Configuration and execution\n\nGPU {gpu}, named tmux; see queue_failure.txt.\n\n"
                    f"## Core results\n\nIncomplete.\n\n## Preserved evidence\n\nPartial files and tmux logs.\n\n"
                    f"## Limitations\n\n{exc}\n\n## Conclusion and next step\n\nDiagnose before retry.\n")
                try:
                    call(f"codex_s30_failmanifest_{suffix}", ["python", str(PRESERVE / "refresh_artifact_manifest.py"), str(run)])
                except BaseException:
                    pass


def main():
    if PROGRESS.exists():
        raise RuntimeError("queue progress already exists")
    progress = {"status": "running", "started_at_utc": now(), "planned_units": 20,
                "pilot_completed": 1, "max_parallel_gpu_jobs": 3,
                "units": {key(unit): {"state": "pending"} for unit in UNITS}}
    PROGRESS.write_text(json.dumps(progress, indent=2, sort_keys=True) + "\n")
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(worker, gpu, UNITS[gpu::3], progress) for gpu in range(3)]
        for future in futures:
            future.result()
    completed = sum(value["state"] == "success" for value in progress["units"].values())
    with LOCK:
        progress.update({"status": "success" if completed == len(UNITS) else "partial_failure",
                         "completed_remaining": completed, "failed_remaining": len(UNITS)-completed,
                         "updated_at_utc": now()})
        PROGRESS.write_text(json.dumps(progress, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": progress["status"], "completed_remaining": completed,
                      "failed_remaining": len(UNITS)-completed}), flush=True)


if __name__ == "__main__":
    main()
