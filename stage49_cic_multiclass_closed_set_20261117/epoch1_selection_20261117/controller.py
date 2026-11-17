#!/usr/bin/env python3
"""Safely save the selected epoch1 and continue the existing frozen task."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ATTEMPT = Path(__file__).resolve().parent
ROOT = ATTEMPT.parent
PROJECT = ROOT.parent
sys.path.insert(0, str(ROOT))
import protocol

WORKSPACE = PROJECT.parents[1]
HELPER = WORKSPACE / "skills/tmux-task-execution/scripts/tmux_task.sh"
SELECTOR = WORKSPACE / "skills/using-superpowers/scripts/select_gpu.py"
TOOLS = WORKSPACE / "skills/experiment-data-preservation/scripts"
PYTHON = "/home/birkenwald/data/SEU-WXY/conda_envs/2025-10-8-WXY-dgl_py310/bin/python"
ORIGINAL_QUEUE = "stage49_cic_queue_20261117"
ORIGINAL_TF = "stage49_cic_trafficformer_20261117"
STATE = {}


def update(phase, status="RUNNING", **extra):
    STATE.update(status=status, phase=phase, updated_at_utc=datetime.now(timezone.utc).isoformat(),
                 active_controller="stage49_epoch1_controller_20261117", **extra)
    temporary = ROOT / "queue_progress.json.tmp"
    protocol.write_json(temporary, STATE)
    os.replace(temporary, ROOT / "queue_progress.json")
    print(json.dumps(STATE), flush=True)


def code(session):
    path = PROJECT / ".tmux-task" / session / "exit.status"
    return int(path.read_text().strip()) if path.exists() else None


def wait(session):
    while code(session) is None:
        time.sleep(15)
    if code(session) != 0:
        raise RuntimeError(f"child failed {session}: exit{code(session)}")


def capacity(exclude=()):
    result = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu",
                             "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
    eligible = []
    for line in result.stdout.splitlines():
        index, free, utilization = [int(x.strip()) for x in line.split(",")]
        if index not in exclude and free >= 30720 and utilization <= 30:
            eligible.append((utilization, -free, index))
    return [index for _, _, index in sorted(eligible)]


def launch(label, script, args=(), gpu=None):
    session = f"stage49_selected_{label}_20261117"
    if (PROJECT / ".tmux-task" / session).exists():
        raise FileExistsError(session)
    command = [PYTHON, str(script), *args]
    if gpu is not None:
        command = [PYTHON, str(SELECTOR), "--min-free-gb", "30", "--max-utilization", "30",
                   "--allowed", str(gpu), "--", *command]
    subprocess.run([str(HELPER), "start", session, str(PROJECT), "--", *command], check=True, cwd=PROJECT)
    STATE["sessions"].append(session)
    STATE["jobs"][label] = {"session": session, "state": "running", "physical_gpu_id": gpu}
    update(label)
    return session


def gpu_job(label, script, args=(), exclude=()):
    while not (ids := capacity(exclude)):
        update(f"GPU_CAPACITY_{label}", "WAITING", min_free_gib=30, max_utilization=30)
        time.sleep(30)
    session = launch(label, script, args, gpu=ids[0])
    wait(session)
    STATE["jobs"][label].update(state="success", exit_code=0)
    update(f"{label}_complete")


def seal_selection():
    verification = json.loads((ATTEMPT / "epoch1_replay_verification.json").read_text())
    metric = verification["parity"]
    (ATTEMPT / "RESULTS.md").write_text(f"""# Stage49 user-selected first TrafficFormer epoch

- Status: success / epoch1 replay and promotion verified

## Data and split

Same frozen Stage49 eleven-class seed2022 members; Train99922/Val8438/Test11688. No role/label/group changes.

## Configuration and execution

User requested epoch1. Original job held its best state only in memory, so an isolated one-epoch replay retained the original20-epoch scheduler horizon, all data ordering, initialization, optimizer and seed. Full-attempt logs/config/history are preserved. Original orchestrator stopped before replay; original TF worker stopped only after a verified first-epoch checkpoint/features existed. Original source files remain unchanged.

## Core results

Selected epoch1 Validation Accuracy={metric['val_accuracy']['replayed']:.8%}, Macro-F1={metric['val_macro_f1']['replayed']:.8%}. All stored original epoch1 metrics match within registered tolerances. This verifies metric parity of a deterministic replay; original in-memory parameters were not directly exported.

## Preserved evidence

PLAN.md, reference_epoch1.json, epoch1_replay_verification.json, source-code hashes, promotion.json, original full-attempt directory/history, original and replay tmux logs; canonical selected checkpoint/features in parent experiment.

## Limitations

Training-budget amendment is user-requested after viewing Known Validation. It is a development checkpoint choice; no Test results used. Original parameter-by-parameter equality cannot be verified against the in-memory-only original snapshot.

## Conclusion and next step

Continue the original adapter/head recipe, frozen Test extraction/evaluation and independent replay using the promoted epoch1 branch. Parent experiment remains running until its own verification completes.
""")
    manifest = json.loads((ATTEMPT / "manifest.json").read_text())
    manifest.update(status="success", core_results=[verification],
                    configuration={"files": ["PLAN.md"], "parameters": {"executed_epochs": 1, "scheduler_horizon_epochs": 20}, "seeds": [2022]},
                    next_step="Continue frozen fusion and Test in the parent Stage49 task",
                    limitations=["deterministic replay with metric parity, not direct in-memory weight export"],
                    execution={"tmux_session": "stage49_epoch1_controller_20261117", "command": "python epoch1_selection_20261117/controller.py",
                               "environment": str(Path(PYTHON).parent.parent)})
    protocol.write_json(ATTEMPT / "manifest.json", manifest)
    for script, args in (("refresh_artifact_manifest.py", ()), ("validate_experiment_bundle.py", ("--verify-hashes",))):
        subprocess.run([PYTHON, str(TOOLS / script), str(ATTEMPT), *args], check=True)


def main():
    global STATE
    if (ATTEMPT / "reference_epoch1.json").exists():
        raise FileExistsError("selection already started; inspect before resuming")
    protocol.verified_rows()
    protocol.verify_sources()
    STATE = json.loads((ROOT / "queue_progress.json").read_text())
    if STATE["phase"] != "BRANCHES":
        raise RuntimeError("original queue already left branch training; inspect before selection")
    protocol.write_json(ATTEMPT / "original_queue_state.json", STATE)
    original_events = []
    log = PROJECT / ".tmux-task" / ORIGINAL_TF / "output.log"
    for line in log.read_text(errors="replace").splitlines():
        if line.startswith("{"):
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if event.get("encoder") == "E1" and "epoch" in event:
                original_events.append(event)
    reference = next(event for event in original_events if event["epoch"] == 1)
    protocol.write_json(ATTEMPT / "reference_epoch1.json", {"original_epoch1": reference,
                        "user_request": "use the first TrafficFormer epoch", "test_used_for_selection": 0})
    hashes = {str(p): protocol.sha(p) for p in [*ATTEMPT.glob("*.py"), ATTEMPT / "PLAN.md"]}
    protocol.write_json(ATTEMPT / "execution_code_hashes.json", hashes)
    # Only the original controller is stopped now; the separate TF training
    # worker continues until the replay is safely stored and verified.
    subprocess.run([str(HELPER), "close", ORIGINAL_QUEUE], check=True)
    update("USER_SELECTED_EPOCH1_REPLAY", selected_epoch=1, original_tf_continues_until_verified=True)
    try:
        original_gpu = STATE["jobs"]["trafficformer"]["physical_gpu_id"]
        gpu_job("epoch1_replay", ATTEMPT / "replay_first.py", exclude=(original_gpu,))
        verification = json.loads((ATTEMPT / "epoch1_replay_verification.json").read_text())
        if verification["status"] != "PASS":
            raise RuntimeError("replayed first epoch not verified")
        source = ATTEMPT / "selected_trafficformer"
        if protocol.sha(source / "model_best.pt") != verification["checkpoint_sha256"] or not (source / "SUCCESS").is_file():
            raise RuntimeError("replay checkpoint not safely saved")
        subprocess.run([str(HELPER), "close", ORIGINAL_TF], check=True)
        # The helper terminates the exact selected worker session; verify its
        # known original PID no longer refers to this branch before moving data.
        for _ in range(20):
            command = Path("/proc/3101443/cmdline")
            if not command.exists() or b"stage49_cic_multiclass_closed_set_20261117/worker.py" not in command.read_bytes():
                break
            time.sleep(1)
        else:
            raise RuntimeError("original TF worker still alive; do not promote over it")
        canonical = ROOT / "cicids2017/runs/ustc/A-2/trafficformer"
        preserved = ATTEMPT / "original_trafficformer_full_budget_attempt"
        canonical.rename(preserved)
        source.rename(canonical)
        current_events = []
        for line in log.read_text(errors="replace").splitlines():
            if line.startswith("{"):
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if event.get("encoder") == "E1" and "epoch" in event:
                    current_events.append(event)
        protocol.write_json(preserved / "interruption_record.json", {"status": "interrupted_by_user_epoch1_selection",
                            "completed_epoch_events": current_events, "original_session": ORIGINAL_TF,
                            "replacement_checkpoint_verified": True, "original_best_was_memory_only": True})
        protocol.write_json(ATTEMPT / "promotion.json", {"status": "PASS", "selected_epoch": 1,
                            "canonical_path": str(canonical), "preserved_attempt": str(preserved),
                            "checkpoint_sha256": protocol.sha(canonical / "model_best.pt"),
                            "split_manifest_sha256": protocol.sha(protocol.MANIFEST)})
        STATE["jobs"]["trafficformer_original_attempt"] = STATE["jobs"]["trafficformer"].copy()
        STATE["jobs"]["trafficformer_original_attempt"]["state"] = "interrupted_by_user_epoch1_selection"
        STATE["jobs"]["trafficformer"] = {**STATE["jobs"]["epoch1_replay"], "state": "success", "selected_epoch": 1}
        update("EPOCH1_PROMOTED", original_tf_continues_until_verified=False)
        seal_selection()
        gpu_job("fusion", ROOT / "worker.py", ("fusion",))
        session = launch("test_inputs", ROOT / "worker.py", ("inputs", "--phase", "test"))
        wait(session)
        STATE["jobs"]["test_inputs"].update(state="success", exit_code=0)
        gpu_job("evaluate", ROOT / "worker.py", ("evaluate",))
        session = launch("finalize", ATTEMPT / "finalize_selected.py")
        wait(session)
    except BaseException as exc:
        protocol.write_json(ATTEMPT / "continuation_failure.json", {"error": repr(exc), "traceback": traceback.format_exc()})
        update("EPOCH1_CONTINUATION_FAILED", "FAIL", error=repr(exc))
        raise


if __name__ == "__main__":
    main()
