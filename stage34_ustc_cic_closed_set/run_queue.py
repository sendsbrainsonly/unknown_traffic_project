#!/usr/bin/env python3
"""Bounded, fail-closed Stage34 continuation; maximum three owned GPUs."""
from __future__ import annotations

import json
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import cic_protocol
import cic_runner
import ustc_runner

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
WORKSPACE = PROJECT.parents[1]
HELPER = WORKSPACE / "skills/tmux-task-execution/scripts/tmux_task.sh"
SELECTOR = WORKSPACE / "skills/using-superpowers/scripts/select_gpu.py"
PROGRESS = ROOT / "queue_progress.json"
STARTED = time.monotonic()
SESSIONS = []


def now():
    return datetime.now(timezone.utc).isoformat()


def update(stage, **more):
    payload = {"status": "RUNNING", "stage": stage, "updated_at_utc": now(),
               "elapsed_seconds": round(time.monotonic()-STARTED, 1),
               "max_parallel_owned_gpus": 3, "sessions": SESSIONS, **more}
    PROGRESS.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload), flush=True)


def status(session):
    file = PROJECT / ".tmux-task" / session / "exit.status"
    return int(file.read_text().strip()) if file.is_file() else None


def wait(sessions, label):
    update(label, waiting_for=sessions)
    while True:
        values = {name: status(name) for name in sessions}
        if all(code is not None for code in values.values()):
            if any(code != 0 for code in values.values()):
                raise RuntimeError(f"{label} failed: {values}; inspect named tmux logs")
            update(label + "_DONE", exit_codes=values)
            return
        time.sleep(20)


def launch(session, script, args=(), gpu=None):
    if (PROJECT / ".tmux-task" / session).exists():
        raise FileExistsError(f"refusing to reuse tmux evidence: {session}")
    command = ["python", str(script), *args]
    if gpu is not None:
        command = ["python", str(SELECTOR), "--min-free-gb", "20", "--allowed", str(gpu), "--", *command]
    subprocess.run([str(HELPER), "start", session, str(PROJECT), "--", *command],
                   check=True, cwd=PROJECT)
    SESSIONS.append(session)
    update("LAUNCHED", last_session=session, physical_gpu=gpu)


def check_final():
    ustc = ROOT / "runs/ustc/A-2/known_test_evaluation/results.json"
    cic = ROOT / "cicids2017/runs/ustc/A-2/known_test_evaluation/results.json"
    for file in (ustc, cic):
        if not file.is_file():
            raise FileNotFoundError(file)
    ua, ca = json.loads(ustc.read_text()), json.loads(cic.read_text())
    if ua["status"] != "PASS" or ca["status"] != "PASS":
        raise RuntimeError("Known Test result not PASS")
    if ustc_runner.sha(ustc_runner.SOURCE) != json.loads((ROOT / "ustc_sample_audit.json").read_text())["source_sha256"]:
        raise RuntimeError("USTC source alignment manifest changed")
    if ustc_runner.sha(ustc_runner.CONFIG) != json.loads((ROOT / "ustc_sample_audit.json").read_text())["a2_config_sha256"]:
        raise RuntimeError("USTC A-2 config changed")
    ci = json.loads((ROOT / "cicids2017_protocol_audit.json").read_text())
    for day, expected in ci["source_parquet_sha256"].items():
        if cic_protocol.digest(cic_protocol.PARQUETS / f"{day}_flows.parquet") != expected:
            raise RuntimeError(f"CIC mapping Parquet changed: {day}")
    final = {"status": "PASS", "ustc": ua, "cicids2017": ca,
             "ustc_source_unchanged": True, "cic_mapping_sources_unchanged": True,
             "unknown_samples_used": 0, "test_parameter_selection": 0,
             "claim_scope": "development diagnostic; CIC group-disjoint but capture/day confounded"}
    (ROOT / "completion_verification.json").write_text(json.dumps(final, indent=2) + "\n")
    update("COMPLETE", results={
        "USTC_A2": {key: ua[key] for key in ("accuracy", "macro_f1", "weighted_f1")},
        "CIC_strict3": {key: ca[key] for key in ("accuracy", "macro_f1", "weighted_f1")}})


def main():
    try:
        wait(["stage34_ustc_tf", "stage34_ustc_graph", "stage34_ustc_yatc"], "USTC_BRANCHES")
        launch("stage34_ustc_fusion", ROOT / "ustc_runner.py", ("fusion",), gpu=1)
        wait(["stage34_ustc_fusion"], "USTC_FUSION")
        launch("stage34_ustc_test_inputs", ROOT / "ustc_runner.py", ("test-inputs",))
        wait(["stage34_ustc_test_inputs"], "USTC_TEST_INPUTS")
        launch("stage34_ustc_test_eval", ROOT / "ustc_runner.py", ("test-evaluate",), gpu=1)
        wait(["stage34_ustc_test_eval"], "USTC_TEST_EVALUATION")
        wait(["stage34_cic_inputs_trainval"], "CIC_TRAINVAL_INPUTS")
        cic_runner.record_alias()
        for name, gpu in (("trafficformer", 0), ("graph", 1), ("yatc", 2)):
            launch(f"stage34_cic_{name}", ROOT / "cic_runner.py", ("branch", "--branch", name), gpu=gpu)
        wait(["stage34_cic_trafficformer", "stage34_cic_graph", "stage34_cic_yatc"], "CIC_BRANCHES")
        launch("stage34_cic_fusion", ROOT / "cic_runner.py", ("fusion",), gpu=1)
        wait(["stage34_cic_fusion"], "CIC_FUSION")
        launch("stage34_cic_test_inputs", ROOT / "cic_build_inputs.py", ("--phase", "test"))
        wait(["stage34_cic_test_inputs"], "CIC_TEST_INPUTS")
        launch("stage34_cic_test_eval", ROOT / "cic_runner.py", ("evaluate",), gpu=1)
        wait(["stage34_cic_test_eval"], "CIC_TEST_EVALUATION")
        check_final()
    except BaseException as exc:
        failure = {"status": "FAIL", "updated_at_utc": now(), "error": repr(exc),
                   "traceback": traceback.format_exc(), "sessions": SESSIONS}
        (ROOT / "queue_failure.json").write_text(json.dumps(failure, indent=2) + "\n")
        PROGRESS.write_text(json.dumps(failure, indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()
