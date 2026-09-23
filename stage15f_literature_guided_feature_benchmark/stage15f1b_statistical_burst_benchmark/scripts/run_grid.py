#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys

from common import CONFIG_PATH, ROOT, pilot_specs, read_json, run_dir, write_json


def main() -> None:
    parity = read_json(ROOT / "s12_parity_summary.json")
    if parity["status"] != "PASS" or len(parity["runs"]) != 5:
        raise RuntimeError("formal grid is blocked until S12 parity passes 5/5")
    config = read_json(CONFIG_PATH)
    jobs = []
    for scenario, feature_sets in config["observation_scenarios"].items():
        for feature_set in feature_sets:
            if scenario == "FULL_FLOW" and feature_set == "S12":
                continue
            for spec in pilot_specs():
                jobs.append((scenario, feature_set, spec["dataset"], spec["protocol_id"]))
    completed, failed = [], []
    for job_index, (scenario, feature_set, dataset, protocol_id) in enumerate(jobs, start=1):
        output = run_dir(scenario, feature_set, dataset, protocol_id)
        result_path = output / "result.json"
        if result_path.exists() and read_json(result_path).get("status") == "PASS":
            completed.append({"scenario": scenario, "feature_set": feature_set, "dataset": dataset, "protocol_id": protocol_id, "status": "PASS", "resume": "existing"})
            continue
        command = [
            sys.executable, str(ROOT / "scripts" / "run_behavior_pilot.py"),
            "--dataset", dataset, "--protocol-id", protocol_id,
            "--scenario", scenario, "--feature-set", feature_set,
        ]
        print(json.dumps({"event": "job_start", "job": job_index, "total": len(jobs), "scenario": scenario, "feature_set": feature_set, "dataset": dataset, "protocol_id": protocol_id}), flush=True)
        process = subprocess.run(command, check=False)
        record = {"scenario": scenario, "feature_set": feature_set, "dataset": dataset, "protocol_id": protocol_id, "exit_code": process.returncode}
        if process.returncode == 0 and result_path.exists() and read_json(result_path).get("status") == "PASS":
            record["status"] = "PASS"
            completed.append(record)
        else:
            record["status"] = "FAIL"
            failed.append(record)
            write_json(ROOT / "grid_status.json", {"status": "FAIL", "expected_trained_jobs": len(jobs), "completed": completed, "failed": failed})
            raise SystemExit(f"formal grid stopped after failed job: {record}")
        write_json(ROOT / "grid_status.json", {"status": "RUNNING", "expected_trained_jobs": len(jobs), "completed": completed, "failed": failed})
    write_json(ROOT / "grid_status.json", {"status": "PASS", "expected_trained_jobs": len(jobs), "completed": completed, "failed": failed})
    print(json.dumps({"status": "PASS", "trained_or_resumed_jobs": len(completed), "failed": len(failed)}, sort_keys=True))


if __name__ == "__main__":
    main()
