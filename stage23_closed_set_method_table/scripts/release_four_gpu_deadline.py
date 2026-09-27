#!/usr/bin/env python3
"""Safety guard: release Stage23-owned GPU0–3 sessions before Beijing 08:00.

Only four exact RoNeTC sessions started by this task are in scope. The guard
does not touch other users, other projects, or YaTC on GPU4/5.
"""

from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


OUT = Path(__file__).resolve().parents[1]
WORKSPACE = OUT.parent.parent.parent
HELPER = WORKSPACE / "skills" / "tmux-task-execution" / "scripts" / "tmux_task.sh"
DEADLINE = datetime(2026, 9, 23, 23, 55, tzinfo=timezone.utc)
TARGETS = (
    ("codex_stage23_ronetc_iscx_vpn_2022_gpu0_20260923", "iscx_vpn", 2022, 0),
    ("codex_stage23_ronetc_iscx_vpn_2023_gpu1_20260923", "iscx_vpn", 2023, 1),
    ("codex_stage23_ronetc_iscx_tor_2022_gpu2_20260923", "iscx_tor", 2022, 2),
    ("codex_stage23_ronetc_iscx_tor_2023_gpu3_20260923", "iscx_tor", 2023, 3),
)


def status(session: str) -> str:
    result = subprocess.run([str(HELPER), "status", session], text=True, capture_output=True, timeout=15)
    return (result.stdout + result.stderr).strip()


def main() -> None:
    print(json.dumps({
        "guard": "Stage23 RoNeTC GPU0-3 deadline release",
        "deadline_utc": DEADLINE.isoformat(), "targets": [item[0] for item in TARGETS],
    }), flush=True)
    while datetime.now(timezone.utc) < DEADLINE:
        remaining = (DEADLINE - datetime.now(timezone.utc)).total_seconds()
        time.sleep(min(60, max(1, remaining)))
    events = []
    for session, dataset, seed, gpu in TARGETS:
        before = status(session)
        action = "already_not_running"
        closed = None
        if "state=running" in before:
            closed = subprocess.run([str(HELPER), "close", session], text=True, capture_output=True, timeout=30)
            action = "closed_exact_owned_session"
        after = status(session)
        run = OUT / "runs" / "ronetc_stage20" / dataset / f"seed{seed}_formal"
        if action == "closed_exact_owned_session" and run.is_dir() and not (run / "SUCCESS").exists():
            history = run / "training_history.jsonl"
            lines = history.read_text(encoding="utf-8").splitlines() if history.is_file() else []
            last_epoch = json.loads(lines[-1])["epoch"] if lines else 0
            (run / "INTERRUPTED_BY_DEADLINE.json").write_text(json.dumps({
                "status": "INTERRUPTED_BY_USER_DEADLINE", "session": session,
                "physical_gpu": gpu, "last_completed_epoch": last_epoch,
                "deadline_utc": DEADLINE.isoformat(), "test_metrics_valid": False,
            }, indent=2) + "\n", encoding="utf-8")
        events.append({
            "session": session, "physical_gpu": gpu, "before": before, "after": after,
            "action": action, "close_exit_code": closed.returncode if closed else None,
            "close_output": (closed.stdout + closed.stderr).strip() if closed else None,
        })
    payload = {
        "deadline_utc": DEADLINE.isoformat(),
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "four exact Stage23-owned RoNeTC sessions on physical GPU0-3 only",
        "events": events,
        "all_target_sessions_not_running": all("state=running" not in item["after"] for item in events),
    }
    (OUT / "deadline_release_event.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload), flush=True)
    if not payload["all_target_sessions_not_running"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
