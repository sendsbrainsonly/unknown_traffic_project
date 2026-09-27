#!/usr/bin/env python3
"""Resume Stage34 after the user released physical GPU 2.

The original three-GPU queue and its interrupted YaTC attempt remain in the
project as evidence. This queue only finishes the already-frozen CIC task;
it does not change its samples, model, optimizer, or checkpoint rule.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import cic_protocol
import ustc_runner

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
WORKSPACE = PROJECT.parents[1]
HELPER = WORKSPACE / "skills/tmux-task-execution/scripts/tmux_task.sh"
SELECTOR = WORKSPACE / "skills/using-superpowers/scripts/select_gpu.py"
PROGRESS = ROOT / "queue_progress.json"
STARTED = time.monotonic()
SESSIONS: list[str] = []


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def update(stage: str, **details: object) -> None:
    payload = {"status": "RUNNING", "stage": stage, "updated_at_utc": now(),
               "elapsed_seconds": round(time.monotonic() - STARTED, 1),
               "max_parallel_owned_gpus": 2, "allowed_physical_gpus": [0, 1],
               "sessions": SESSIONS, **details}
    temporary = PROGRESS.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, PROGRESS)
    print(json.dumps(payload), flush=True)


def exit_status(session: str) -> int | None:
    path = PROJECT / ".tmux-task" / session / "exit.status"
    return int(path.read_text().strip()) if path.is_file() else None


def wait(sessions: list[str], stage: str) -> None:
    update(stage, waiting_for=sessions)
    while True:
        values = {name: exit_status(name) for name in sessions}
        if any(code is not None and code != 0 for code in values.values()):
            raise RuntimeError(f"{stage} failed: {values}; inspect named tmux logs")
        if all(code == 0 for code in values.values()):
            update(stage + "_DONE", exit_codes=values)
            return
        time.sleep(20)


def launch(session: str, script: Path, args: tuple[str, ...], gpu: int | None) -> None:
    if (PROJECT / ".tmux-task" / session).exists():
        raise FileExistsError(f"refusing to reuse tmux evidence: {session}")
    command = ["python", str(script), *args]
    if gpu is not None:
        if gpu not in (0, 1):
            raise ValueError(f"GPU outside user limit: {gpu}")
        command = ["python", str(SELECTOR), "--min-free-gb", "20",
                   "--allowed", str(gpu), "--", *command]
    subprocess.run([str(HELPER), "start", session, str(PROJECT), "--", *command],
                   check=True, cwd=PROJECT)
    SESSIONS.append(session)
    update("LAUNCHED", last_session=session, physical_gpu=gpu)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def archive_interrupted_yatc() -> None:
    old = ROOT / "cicids2017/runs/ustc/A-2/yatc"
    archived = old.with_name("yatc_interrupted_gpu2_20260925")
    if exit_status("stage34_cic_yatc") in (None, 0):
        raise RuntimeError("original GPU2 YaTC attempt is not a finished interruption")
    if not old.is_dir() or (old / "SUCCESS").exists() or archived.exists():
        raise RuntimeError("unexpected YaTC attempt/archive state; no relocation performed")
    checkpoint = old / "model_best.pt"
    checkpoint_hash = sha256(checkpoint) if checkpoint.is_file() else None
    old.rename(archived)
    note = {"status": "INTERRUPTED_BY_USER_GPU_RELEASE", "at_utc": now(),
            "original_session": "stage34_cic_yatc",
            "original_exit_code": exit_status("stage34_cic_yatc"),
            "original_log": str(PROJECT / ".tmux-task/stage34_cic_yatc/output.log"),
            "archived_directory": str(archived),
            "best_checkpoint_sha256_if_present": checkpoint_hash,
            "resume_policy": "restart unchanged YaTC recipe from official initialization"}
    (archived / "INTERRUPTION.json").write_text(json.dumps(note, indent=2) + "\n")
    update("GPU2_ATTEMPT_ARCHIVED", interrupted_attempt=str(archived))


def verify_final() -> None:
    ustc_path = ROOT / "runs/ustc/A-2/known_test_evaluation/results.json"
    cic_path = ROOT / "cicids2017/runs/ustc/A-2/known_test_evaluation/results.json"
    ustc = json.loads(ustc_path.read_text())
    cic = json.loads(cic_path.read_text())
    if ustc["status"] != "PASS" or cic["status"] != "PASS":
        raise RuntimeError("Known Test result is not PASS")
    sample_audit = json.loads((ROOT / "ustc_sample_audit.json").read_text())
    if ustc_runner.sha(ustc_runner.SOURCE) != sample_audit["source_sha256"]:
        raise RuntimeError("USTC source alignment manifest changed")
    if ustc_runner.sha(ustc_runner.CONFIG) != sample_audit["a2_config_sha256"]:
        raise RuntimeError("USTC A-2 config changed")
    cic_audit = json.loads((ROOT / "cicids2017_protocol_audit.json").read_text())
    for day, expected in cic_audit["source_parquet_sha256"].items():
        if cic_protocol.digest(cic_protocol.PARQUETS / f"{day}_flows.parquet") != expected:
            raise RuntimeError(f"CIC mapping Parquet changed: {day}")
    verification = {"status": "PASS", "ustc": ustc, "cicids2017": cic,
                    "ustc_source_unchanged": True, "cic_mapping_sources_unchanged": True,
                    "unknown_samples_used": 0, "test_parameter_selection": 0,
                    "allowed_physical_gpus": [0, 1],
                    "claim_scope": "development diagnostic; CIC group-disjoint but capture/day confounded"}
    (ROOT / "completion_verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    update("COMPLETE", results={
        "USTC_A2": {key: ustc[key] for key in ("accuracy", "macro_f1", "weighted_f1")},
        "CIC_strict3": {key: cic[key] for key in ("accuracy", "macro_f1", "weighted_f1")}})


def main() -> None:
    try:
        if exit_status("stage34_bounded_queue") in (None, 0):
            raise RuntimeError("original three-GPU queue must be stopped before continuation")
        if exit_status("stage34_ustc_test_eval") != 0:
            raise RuntimeError("frozen USTC Known Test evaluation not complete")
        archive_interrupted_yatc()
        previous = ["stage34_cic_trafficformer", "stage34_cic_graph"]
        update("WAITING_FOR_FREE_GPU_0_OR_1", waiting_for=previous)
        while True:
            states = {name: exit_status(name) for name in previous}
            if any(code is not None and code != 0 for code in states.values()):
                raise RuntimeError(f"CIC existing branch failed: {states}")
            if states[previous[0]] == 0 or states[previous[1]] == 0:
                selected = 0 if states[previous[0]] == 0 else 1
                break
            time.sleep(20)
        launch("stage34_cic_yatc_two_gpu_retry", ROOT / "cic_runner.py",
               ("branch", "--branch", "yatc"), gpu=selected)
        wait([*previous, "stage34_cic_yatc_two_gpu_retry"], "CIC_BRANCHES")
        launch("stage34_cic_fusion_two_gpu", ROOT / "cic_runner.py", ("fusion",), gpu=1)
        wait(["stage34_cic_fusion_two_gpu"], "CIC_FUSION")
        launch("stage34_cic_test_inputs_two_gpu", ROOT / "cic_build_inputs.py",
               ("--phase", "test"), gpu=None)
        wait(["stage34_cic_test_inputs_two_gpu"], "CIC_TEST_INPUTS")
        launch("stage34_cic_test_eval_two_gpu", ROOT / "cic_runner.py", ("evaluate",), gpu=1)
        wait(["stage34_cic_test_eval_two_gpu"], "CIC_TEST_EVALUATION")
        verify_final()
    except BaseException as exc:
        failure = {"status": "FAIL", "updated_at_utc": now(), "error": repr(exc),
                   "traceback": traceback.format_exc(), "sessions": SESSIONS,
                   "allowed_physical_gpus": [0, 1]}
        (ROOT / "two_gpu_queue_failure.json").write_text(json.dumps(failure, indent=2) + "\n")
        PROGRESS.write_text(json.dumps(failure, indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()
