#!/usr/bin/env python3
"""Initialize durable Stage 11B experiment bundles before inference."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from stage11b_common import CONFIG_PATH, STAGE_ROOT, read_json


WORKSPACE_ROOT = STAGE_ROOT.parents[2]
INIT_BUNDLE = (
    WORKSPACE_ROOT
    / ".agents"
    / "skills"
    / "experiment-data-preservation"
    / "scripts"
    / "init_experiment_bundle.py"
)


def initialise(path: Path, experiment_id: str, objective: str, session: str) -> None:
    if (path / "manifest.json").is_file():
        return
    command = [
        sys.executable,
        "-B",
        str(INIT_BUNDLE),
        str(path),
        "--id",
        experiment_id,
        "--objective",
        objective,
        "--experiment-type",
        "method-development-evaluation",
        "--claim-scope",
        "diagnostic",
        "--status",
        "planned",
        "--tmux-session",
        session,
    ]
    if path.exists() and any(path.iterdir()):
        command.append("--adopt-existing")
    subprocess.run(command, cwd=STAGE_ROOT.parent, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-smoke", action="store_true")
    args = parser.parse_args()
    config = read_json(CONFIG_PATH)
    initialise(
        STAGE_ROOT,
        "stage11b-decoupled-support-readout-20260915-v1",
        "Frozen USTC Open-Detect R0-R3 decoupled support readout study",
        "stage11b-campaign",
    )
    prepared: list[str] = []
    for scenario in ("a1", "a2", "a3"):
        for fold, seed in enumerate(config["seeds"]):
            run_dir = STAGE_ROOT / "artifacts" / scenario / f"fold{fold}_seed{seed}"
            initialise(
                run_dir,
                f"stage11b-{scenario}-fold{fold}-seed{seed}",
                f"Frozen encoder R0-R3 readout evaluation for {scenario.upper()} fold {fold} seed {seed}",
                f"stage11b-{scenario}-fold{fold}",
            )
            prepared.append(str(run_dir))
    smoke = None
    if args.include_smoke:
        smoke_dir = STAGE_ROOT / "artifacts" / "smoke" / "a1" / "fold0_seed2022"
        initialise(
            smoke_dir,
            "stage11b-smoke-a1-fold0-seed2022",
            "Full-path frozen inference smoke for Stage11B",
            "stage11b-smoke",
        )
        smoke = str(smoke_dir)
    print(json.dumps({"status": "PASS", "formal_runs_prepared": len(prepared), "smoke": smoke}, indent=2))


if __name__ == "__main__":
    main()
