#!/usr/bin/env python3
"""Repair only the Stage49 final artifact metadata after the preserved exit 1."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ATTEMPT = Path(__file__).resolve().parent
ROOT = ATTEMPT.parent
PROJECT = ROOT.parent
WORKSPACE = PROJECT.parents[1]
TOOLS = WORKSPACE / "skills/experiment-data-preservation/scripts"
ORIGINAL_FINALIZE_SESSION = "stage49_selected_finalize_20261117"
ORIGINAL_CONTROLLER_SESSION = "stage49_epoch1_controller_20261117"
RECOVERY_SESSION = "stage49_finalize_recovery_20261117"
SUCCESS_SESSIONS = (
    "stage49_cic_protocol_20261117",
    "stage49_cic_preflight_20261117",
    "stage49_cic_trainval_inputs_20261117",
    "stage49_selected_epoch1_replay_20261117",
    "stage49_cic_yatc_20261117",
    "stage49_cic_graph_20261117",
    "stage49_selected_fusion_20261117",
    "stage49_selected_test_inputs_20261117",
    "stage49_selected_evaluate_20261117",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def session_file(session: str, name: str) -> Path:
    return PROJECT / ".tmux-task" / session / name


def assert_no_stage49_workers() -> list[dict[str, object]]:
    forbidden = ("/worker.py", "/replay_first.py", "/controller.py")
    matches: list[dict[str, object]] = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            command = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if str(ROOT) in command and any(marker in command for marker in forbidden):
            matches.append({"pid": int(proc.name), "command": command})
    if matches:
        raise RuntimeError(f"Stage49 worker/controller still alive: {matches}")
    return matches


def external_entry(path: Path, role: str) -> dict[str, object]:
    return {
        "path": os.path.relpath(path, ROOT),
        "scope": "external",
        "role": role,
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
        "hash_status": "verified",
    }


def validate_bundle() -> None:
    subprocess.run(
        [sys.executable, str(TOOLS / "refresh_artifact_manifest.py"), str(ROOT)],
        check=True,
    )
    subprocess.run(
        [sys.executable, str(TOOLS / "validate_experiment_bundle.py"), str(ROOT), "--verify-hashes"],
        check=True,
    )


def main() -> None:
    sys.path.insert(0, str(ROOT))
    import protocol

    completion = json.loads((ROOT / "completion_verification.json").read_text())
    if completion.get("status") != "PASS" or not completion.get("saved_logit_metric_replay"):
        raise RuntimeError("existing numerical completion verification is not PASS")
    source_hashes = protocol.verify_sources()
    before = json.loads((ROOT / "checkpoint_hashes_before_test.json").read_text())
    after = json.loads((ROOT / "checkpoint_hashes_after_test.json").read_text())
    if before != after or any(protocol.sha(ROOT / relative) != digest for relative, digest in after.items()):
        raise RuntimeError("checkpoint hashes changed after completed inference")

    session_statuses: dict[str, int] = {}
    for session in SUCCESS_SESSIONS:
        status_path = session_file(session, "exit.status")
        if not status_path.is_file():
            raise RuntimeError(f"missing successful session status: {session}")
        status = int(status_path.read_text().strip())
        if status != 0:
            raise RuntimeError(f"completed session is not exit0: {session}={status}")
        session_statuses[session] = status

    failed_status = session_file(ORIGINAL_FINALIZE_SESSION, "exit.status")
    failed_log = session_file(ORIGINAL_FINALIZE_SESSION, "output.log")
    if failed_status.read_text().strip() != "1":
        raise RuntimeError("original finalizer exit1 evidence is missing")
    failure_text = failed_log.read_text(errors="replace")
    if "artifact path must be relative" not in failure_text or "validation failed: errors=21" not in failure_text:
        raise RuntimeError("original finalizer failure signature changed")
    live_workers = assert_no_stage49_workers()

    failure_files = [failed_log, failed_status]
    controller_log = session_file(ORIGINAL_CONTROLLER_SESSION, "output.log")
    controller_status = session_file(ORIGINAL_CONTROLLER_SESSION, "exit.status")
    if controller_log.is_file():
        failure_files.append(controller_log)
    if controller_status.is_file():
        failure_files.append(controller_status)
    failure_evidence = {
        "status": "PRESERVED",
        "original_finalize_session": ORIGINAL_FINALIZE_SESSION,
        "original_exit_code": 1,
        "root_cause": "absolute external artifact paths rejected by bundle validator",
        "numerical_completion_preceded_failure": True,
        "files": [
            {
                "path": os.path.relpath(path, ROOT),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in failure_files
        ],
    }
    write_json(ATTEMPT / "original_failure_evidence.json", failure_evidence)

    manifest_path = ROOT / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    external: dict[str, dict[str, object]] = {}
    for item in manifest.get("artifacts", []):
        if item.get("scope") != "external":
            continue
        path = Path(str(item["path"]))
        if not path.is_absolute():
            path = (ROOT / path).resolve()
        if not path.is_file():
            raise RuntimeError(f"external artifact is missing: {path}")
        normalized = external_entry(path, str(item.get("role", "execution-log")))
        external[str(normalized["path"])] = normalized
    for path in failure_files:
        normalized = external_entry(path, "preserved-finalize-failure")
        external[str(normalized["path"])] = normalized
    manifest["artifacts"] = list(external.values())
    manifest["status"] = "success"
    manifest["updated_at_utc"] = now()
    manifest["finalization_recovery"] = {
        "status": "running",
        "attempt": "finalize_recovery_attempt1",
        "scope": "metadata-only",
        "original_finalize_exit_code": 1,
        "original_failure_preserved": True,
        "numerical_outputs_reused": True,
        "training_rerun": False,
        "feature_extraction_rerun": False,
        "gpu_inference_rerun": False,
        "external_artifact_paths_normalized": len(external),
    }
    write_json(manifest_path, manifest)

    queue_path = ROOT / "queue_progress.json"
    queue = json.loads(queue_path.read_text())
    queue.update(status="RECOVERING_FINALIZE_METADATA", phase="FINALIZE_RECOVERY", updated_at_utc=now())
    queue["jobs"]["finalize"].update(state="failed_exit1_preserved", exit_code=1)
    queue["jobs"]["finalize_recovery"] = {
        "session": RECOVERY_SESSION,
        "state": "running",
        "physical_gpu_id": None,
    }
    write_json(queue_path, queue)
    validate_bundle()

    verification = {
        "status": "PASS",
        "recovery_scope": "metadata-only",
        "completion_verification_reused": True,
        "completion_metrics": {
            key: completion[key]
            for key in (
                "accuracy",
                "macro_precision",
                "macro_recall",
                "macro_f1",
                "weighted_precision",
                "weighted_recall",
                "weighted_f1",
            )
        },
        "source_hashes_unchanged": len(source_hashes),
        "checkpoint_hashes_unchanged": len(after),
        "successful_completed_sessions": session_statuses,
        "original_finalize_exit1_preserved": True,
        "duplicate_stage49_workers": live_workers,
        "training_rerun": False,
        "feature_extraction_rerun": False,
        "gpu_inference_rerun": False,
    }
    write_json(ATTEMPT / "finalize_recovery_verification.json", verification)
    queue.update(status="SUCCESS_AFTER_FINALIZE_RECOVERY", phase="COMPLETE", updated_at_utc=now())
    queue["jobs"]["finalize_recovery"].update(state="verified", exit_code=0)
    write_json(queue_path, queue)
    manifest = json.loads(manifest_path.read_text())
    manifest["finalization_recovery"].update(status="PASS", completed_at_utc=now())
    write_json(manifest_path, manifest)
    validate_bundle()
    print(json.dumps(verification, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
