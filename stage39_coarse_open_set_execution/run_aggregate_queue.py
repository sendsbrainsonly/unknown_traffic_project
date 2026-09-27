"""Aggregate Stage39 only after every frozen setting has verified scores."""
from __future__ import annotations

import json
import os
import subprocess
import time
import traceback
from datetime import datetime, timezone

from freeze_protocols import PROJECT, ROOT

WORKSPACE = PROJECT.parents[1]
HELPER = WORKSPACE / "skills/tmux-task-execution/scripts/tmux_task.sh"
PYTHON = "/home/birkenwald/data/SEU-WXY/conda_envs/2025-10-8-WXY-dgl_py310/bin/python"
START = time.monotonic()


def save(status, **extra):
    value = {"status": status, "updated_at_utc": datetime.now(timezone.utc).isoformat(),
             "elapsed_seconds": round(time.monotonic() - START, 1), **extra}
    temporary = ROOT / "aggregate_queue_progress.json.tmp"
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, ROOT / "aggregate_queue_progress.json")
    print(json.dumps(value, ensure_ascii=False), flush=True)


def main():
    try:
        while True:
            path = ROOT / "evaluation_queue_progress.json"
            state = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
            if state.get("status") == "FAILED":
                raise RuntimeError("dependent evaluation queue failed")
            if state.get("status") in ("DETECTION_COMPLETE", "RESULTS_AGGREGATED"):
                break
            save("WAITING_FOR_EVALUATION", dependent_status=state.get("status"))
            time.sleep(30)
        if state["status"] == "RESULTS_AGGREGATED":
            result = json.loads((ROOT / "aggregate_verification.json").read_text())
            if result["status"] != "PASS":
                raise RuntimeError("aggregate marker is not PASS")
        else:
            save("AGGREGATING")
            session = "stage39_aggregate_0926"
            if (PROJECT / ".tmux-task" / session).exists():
                raise FileExistsError(f"preserve existing aggregate session: {session}")
            subprocess.run([str(HELPER), "run", session, str(PROJECT), "--", PYTHON,
                            str(ROOT / "aggregate_results.py")], cwd=PROJECT, check=True)
            result = json.loads((ROOT / "aggregate_verification.json").read_text())
            if result["status"] != "PASS":
                raise RuntimeError("aggregate verification failed")
        save("AGGREGATE_VERIFIED", settings=result["new_training_settings"],
             report="stage39_report.md", next_step="update final RESULTS.md/manifest/index and validate bundle")
    except BaseException as exc:
        failure_path = ROOT / "aggregate_queue_failure.json"
        if failure_path.exists():
            failure_path = ROOT / f"aggregate_queue_failure_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
        failure_path.write_text(json.dumps({
            "status": "FAILED", "error": repr(exc), "traceback": traceback.format_exc(),
            "updated_at_utc": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n")
        save("FAILED", error=repr(exc))
        raise


if __name__ == "__main__":
    main()
