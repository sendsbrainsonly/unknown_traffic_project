#!/usr/bin/env python3
"""Initialize durable Stage 11C root and run bundles."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from stage11c_common import SCENARIOS, SEEDS, STAGE_ROOT, output_run_dir


WORKSPACE_ROOT = STAGE_ROOT.parents[2]
INIT_BUNDLE = WORKSPACE_ROOT / ".agents" / "skills" / "experiment-data-preservation" / "scripts" / "init_experiment_bundle.py"


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
        "diagnostic",
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
    initialise(STAGE_ROOT, "stage11c-known-only-support-complexity-20260915-v1", "Known-only support-complexity diagnosis over 15 frozen Stage11B runs", "stage11c-campaign")
    prepared = 0
    for scenario in SCENARIOS:
        for fold, seed in enumerate(SEEDS):
            initialise(
                output_run_dir(scenario, fold, seed),
                f"stage11c-{scenario}-fold{fold}-seed{seed}",
                f"Known-only geometry and support diagnosis for {scenario.upper()} fold {fold} seed {seed}",
                f"stage11c-{scenario}-fold{fold}",
            )
            prepared += 1
    print(json.dumps({"status": "PASS", "formal_runs_prepared": prepared}, indent=2))


if __name__ == "__main__":
    main()
