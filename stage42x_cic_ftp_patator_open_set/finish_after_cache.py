#!/usr/bin/env python3
"""Finish frozen FTP-Patator scoring after the packet cache passes audit."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from stage42x_common import MANIFEST, PROTOCOL, ROOT, UNIT, digest

CACHE_EXIT = ROOT / ".tmux-task/stage42x-cache-20260928/exit.status"
QUEUE_STATE = ROOT / "queue_progress.json"
SELECTOR = Path("/home/birkenwald/data/SEU-WXY/WXY-SEU/TrafficClassifier/skills/using-superpowers/scripts/select_gpu.py")


def save(status: str, phase: str, **extra) -> None:
    value = {"status": status, "phase": phase,
             "updated_at_utc": datetime.now(timezone.utc).isoformat(), **extra}
    temporary = QUEUE_STATE.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, QUEUE_STATE)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def audit_cache(expected: int, manifest_hash: str) -> None:
    base = ROOT / UNIT / "input_caches/ustc/A-2"
    for name in ("tf_fig_test", "yatc_mfr_test"):
        path = base / name / "cache_audit.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        if (data["status"] != "PASS" or data["flows"] != expected
                or data["candidate_manifest_sha256"] != manifest_hash
                or data["unknown_samples_used_for_fitting"]
                or data["unknown_samples_used_for_calibration"]):
            raise RuntimeError(f"full FTP-Patator cache audit failed: {path}")


def run(args: list[str], phase: str) -> None:
    save("RUNNING", phase, command=args)
    subprocess.run(args, cwd=ROOT, check=True)


def main() -> None:
    try:
        protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        if (protocol["status"] != "PASS" or protocol["unknown_test"] != 3985
                or digest(MANIFEST) != protocol["candidate_manifest_sha256"]):
            raise RuntimeError("FTP-Patator freeze drift before queue completion")
        start = time.monotonic()
        while not CACHE_EXIT.exists():
            if time.monotonic() - start > 14400:
                raise TimeoutError("FTP-Patator packet cache exceeded four-hour wait")
            save("WAITING_FOR_CACHE", "full_packet_cache", elapsed_seconds=round(time.monotonic() - start))
            time.sleep(30)
        if CACHE_EXIT.read_text(encoding="utf-8").strip() != "0":
            raise RuntimeError("FTP-Patator packet-cache process failed")
        audit_cache(protocol["unknown_test"], protocol["candidate_manifest_sha256"])
        save("RUNNING", "cache_audit_pass", unknown_test=protocol["unknown_test"])
        run([sys.executable, str(SELECTOR), "--min-free-gb", "16", "--max-utilization", "30",
             "--", sys.executable, "run_frozen.py", "evaluate"], "frozen_gpu_evaluation")
        run([sys.executable, "run_frozen.py", "verify"], "independent_metric_replay")
        out = ROOT / UNIT / "detection"
        verification = json.loads((out / "independent_verification.json").read_text(encoding="utf-8"))
        evaluation = json.loads((out / "evaluation_audit.json").read_text(encoding="utf-8"))
        if (verification["status"] != "PASS" or verification["unknown_test"] != 3985
                or verification["known_test"] != 2272 or verification["metrics_replayed"] != 8
                or evaluation["status"] != "PASS" or not evaluation["checkpoint_hashes_unchanged"]
                or evaluation["unknown_fit_count"] or evaluation["test_fit_count"]
                or digest(MANIFEST) != protocol["candidate_manifest_sha256"]):
            raise RuntimeError("full FTP-Patator completion audit failed")
        save("COMPLETE", "independent_metric_replay_pass", verification=verification)
    except BaseException as exc:
        failure = {"status": "FAILED", "error": repr(exc), "traceback": traceback.format_exc(),
                   "updated_at_utc": datetime.now(timezone.utc).isoformat()}
        (ROOT / "queue_failure.json").write_text(json.dumps(failure, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        save("FAILED", "stopped", error=repr(exc))
        raise


if __name__ == "__main__":
    main()
