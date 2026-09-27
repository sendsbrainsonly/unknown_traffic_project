#!/usr/bin/env python3
"""Bounded three-physical-GPU queue for Stage31 branches and T0 heads only."""
from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from preflight import CELLS, OUT, PROJECT

TMUX = PROJECT.parents[1] / "skills" / "tmux-task-execution" / "scripts" / "tmux_task.sh"
SELECT = PROJECT.parents[1] / "skills" / "using-superpowers" / "scripts" / "select_gpu.py"
LANES = {0: "trafficformer", 1: "graph", 2: "yatc"}
INITIAL = {
    ("vnat", "medium_seed2025", "trafficformer"):
        "codex_stage31_vnat2025_tf_formal_20260925",
    ("ustc", "A-2", "graph"):
        "codex_stage31_ustc_graph_formal_20260925",
    ("vnat", "medium_seed2025", "yatc"):
        "codex_stage31_vnat2025_yatc_formal_20260925",
}
PRIORITY = (("iscx_vpn", "medium_seed2022"), ("iscx_tor", "medium_seed2022"),
            ("vnat", "medium_seed2026"), ("vnat", "medium_seed2025"), ("ustc", "A-2"))


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def save(state):
    state["updated_at_utc"] = timestamp()
    temporary = OUT / "queue_progress.json.tmp"
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    temporary.replace(OUT / "queue_progress.json")


def tmux_state(name):
    result = subprocess.run([str(TMUX), "status", name], cwd=PROJECT,
                            capture_output=True, text=True)
    output = result.stdout + result.stderr
    if "state=running" in output:
        return "running"
    if "state=finished" in output and "exit_code=0" in output:
        return "finished"
    if "state=finished" in output:
        return "failed"
    return "absent"


def cache_ready(dataset, protocol):
    root = OUT / "input_caches" / dataset
    if dataset == "vnat":
        root = root / protocol
        mfr = root / ("yatc_mfr_attempt3" if protocol == "medium_seed2025" else "yatc_mfr")
        roles = ("train", "validation")
    elif dataset == "ustc":
        root = root / protocol
        mfr = root / "yatc_mfr"
        roles = ("known_train", "known_validation")
    else:
        mfr = root / "yatc_mfr"
        roles = ("known_train", "known_validation")
    tf_fig = root / "tf_fig"
    if not (mfr / "cache_audit.json").is_file() or not (tf_fig / "cache_audit.json").is_file():
        return False
    ma = json.loads((mfr / "cache_audit.json").read_text())
    ta = json.loads((tf_fig / "cache_audit.json").read_text())
    if ma["status"] != ta["status"] or ma["status"] != "PASS":
        raise RuntimeError(f"input audit failed: {dataset}/{protocol}")
    a = set(np.load(tf_fig / "flow_ids.npy", allow_pickle=False).astype(str).tolist())
    b = set().union(*(set(np.load(mfr / f"{r}_flow_ids.npy", allow_pickle=False).astype(str).tolist())
                      for r in roles))
    if len(a) != len(b) or a != b:
        raise RuntimeError(f"same-flow three-input gate failed: {dataset}/{protocol}")
    return True


def start(dataset, protocol, branch, gpu):
    short = {"iscx_vpn": "vpn", "iscx_tor": "tor", "vnat": "vnat", "ustc": "ustc"}[dataset]
    tag = protocol.replace("medium_seed", "m").replace("A-2", "a2")
    session = f"codex_stage31_auto_{short}_{tag}_{branch}_20260925"
    if branch == "yatc":
        script = "train_yatc_branch.py"
        command = ["--dataset", dataset, "--protocol", protocol]
    elif branch == "T0_equal":
        script = "train_equal_fusion.py"
        command = ["--dataset", dataset, "--protocol", protocol]
    else:
        script = "train_tf_fig_branch.py"
        command = ["--dataset", dataset, "--protocol", protocol, "--branch", branch]
    payload = ["python", str(SELECT), "--allowed", str(gpu), "--min-free-gb", "16",
               "--", "python", str(OUT / script), *command]
    result = subprocess.run([str(TMUX), "start", session, str(PROJECT), "--", *payload],
                            cwd=PROJECT, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"failed to start {session}: {result.stdout} {result.stderr}")
    return session


def main():
    state = {"status": "RUNNING", "gpu_cap": 3, "physical_gpu_lanes": LANES,
             "active": {str(gpu): None for gpu in LANES}, "events": [],
             "test_features_opened": 0}
    for gpu, branch in LANES.items():
        for (dataset, protocol, name), session in INITIAL.items():
            if name == branch and tmux_state(session) == "running":
                state["active"][str(gpu)] = {"dataset": dataset, "protocol": protocol,
                                              "branch": branch, "session": session}
    save(state)
    try:
        while True:
            for gpu in LANES:
                job = state["active"][str(gpu)]
                if job is None:
                    continue
                current = tmux_state(job["session"])
                if current == "running":
                    continue
                run = OUT / "runs" / job["dataset"] / job["protocol"] / job["branch"]
                if current != "finished" or not (run / "SUCCESS").is_file():
                    raise RuntimeError(f"job failed or incomplete: {job}; tmux={current}")
                state["events"].append({"event": "COMPLETED", "at": timestamp(), **job})
                state["active"][str(gpu)] = None
            ready = {(d, p): cache_ready(d, p) for d, p in CELLS}
            for gpu, branch in LANES.items():
                if state["active"][str(gpu)] is not None:
                    continue
                for dataset, protocol in PRIORITY:
                    if not ready[(dataset, protocol)]:
                        continue
                    run = OUT / "runs" / dataset / protocol / branch
                    if (run / "SUCCESS").is_file():
                        continue
                    if run.exists():
                        raise RuntimeError(f"unresolved partial branch run: {run}")
                    session = start(dataset, protocol, branch, gpu)
                    job = {"dataset": dataset, "protocol": protocol,
                           "branch": branch, "session": session}
                    state["active"][str(gpu)] = job
                    state["events"].append({"event": "STARTED", "at": timestamp(), **job})
                    break
            if all((OUT / "runs" / d / p / branch / "SUCCESS").is_file()
                   for d, p in CELLS for branch in LANES.values()):
                # All three physical lanes are now free; use GPU 1 for lightweight T0 heads.
                if all(state["active"][str(gpu)] is None for gpu in LANES):
                    for dataset, protocol in CELLS:
                        run = OUT / "runs" / dataset / protocol / "T0_equal"
                        if (run / "SUCCESS").is_file():
                            continue
                        if run.exists():
                            raise RuntimeError(f"unresolved partial T0 run: {run}")
                        session = start(dataset, protocol, "T0_equal", 1)
                        job = {"dataset": dataset, "protocol": protocol,
                               "branch": "T0_equal", "session": session}
                        state["active"]["1"] = job
                        state["events"].append({"event": "STARTED", "at": timestamp(), **job})
                        break
                if all((OUT / "runs" / d / p / "T0_equal" / "SUCCESS").is_file() for d, p in CELLS):
                    state["status"] = "ALL_BRANCHES_AND_T0_COMPLETE_TEST_NOT_RUN"
                    save(state)
                    print(json.dumps({"status": state["status"], "events": len(state["events"])}), flush=True)
                    return
            save(state)
            print(json.dumps({"status": state["status"], "active": state["active"],
                              "input_ready": sum(ready.values())}), flush=True)
            time.sleep(30)
    except BaseException as exc:
        state["status"] = "FAILED"
        state["error"] = repr(exc)
        save(state)
        raise


if __name__ == "__main__":
    main()
