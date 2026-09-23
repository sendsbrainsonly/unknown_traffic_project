#!/usr/bin/env python3
"""Create non-overwriting Stage 11A run bundles and frozen run configs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from stage11a_common import CONFIG_PATH, STAGE_ROOT, read_json, sha256_file, write_json


WORKSPACE_ROOT = STAGE_ROOT.parents[2]
INIT_BUNDLE = (
    WORKSPACE_ROOT
    / ".agents"
    / "skills"
    / "experiment-data-preservation"
    / "scripts"
    / "init_experiment_bundle.py"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adopt-existing", action="store_true")
    args = parser.parse_args()
    config = read_json(CONFIG_PATH)
    v6_root = Path(config["v6_frozen_root"])
    split_root = Path(config["v6_split_root"])
    source_manifest = read_json(
        STAGE_ROOT.parent / "stage10a_opendetect_protocol_audit" / "sources" / "source_manifest.json"
    )
    expected_checkpoints = {
        str(Path(row["path"]).resolve()): row["sha256"] for row in source_manifest["v6_checkpoints"]
    }
    prepared: list[str] = []
    for scenario, scenario_cfg in config["scenarios"].items():
        for fold, seed in enumerate(config["seeds"]):
            run_dir = STAGE_ROOT / "runs" / scenario / f"fold{fold}_seed{seed}"
            b0_dir = v6_root / scenario / f"fold{fold}_seed{seed}"
            b0_checkpoint = (b0_dir / "model_best.pt").resolve()
            data_dir = (split_root / f"fold{fold}_seed{seed}").resolve()
            if not (run_dir / "manifest.json").is_file():
                command = [
                    sys.executable,
                    "-B",
                    str(INIT_BUNDLE),
                    str(run_dir),
                    "--id",
                    f"stage11a-dap-{scenario}-fold{fold}-seed{seed}",
                    "--objective",
                    f"Formal DAP training and frozen-checkpoint diagnostics for {scenario.upper()} fold {fold} seed {seed}",
                    "--experiment-type",
                    "method-development-training",
                    "--claim-scope",
                    "diagnostic",
                    "--status",
                    "planned",
                    "--tmux-session",
                    f"stage11a-{scenario}-fold{fold}",
                ]
                if args.adopt_existing:
                    command.append("--adopt-existing")
                subprocess.run(command, cwd=STAGE_ROOT.parent, check=True)
            run_config = {
                "evaluation_label": config["status_label"],
                "external_validation_label": config["external_validation_label"],
                "scenario": scenario,
                "split": scenario_cfg["split"],
                "fold": fold,
                "seed": seed,
                "method": "DAP",
                "data_dir": str(data_dir),
                "b0_frozen_run_dir": str(b0_dir.resolve()),
                "b0_frozen_checkpoint": str(b0_checkpoint),
                "b0_frozen_checkpoint_expected_sha256": expected_checkpoints[str(b0_checkpoint)],
                "stage11a_config": str(CONFIG_PATH.resolve()),
                "stage11a_config_sha256": sha256_file(CONFIG_PATH),
                "training": config["training"],
                "density_diagnostics": config["density_diagnostics"],
                "unknown_use": "evaluation after frozen best checkpoint only",
                "checkpoint_selection": "Known Validation Accuracy only",
                "prototype_update_data": "Known Train deterministic mu only",
            }
            config_path = run_dir / "config.json"
            if config_path.exists() and read_json(config_path) != run_config:
                raise RuntimeError(f"Refusing to overwrite a different run config: {config_path}")
            if not config_path.exists():
                write_json(config_path, run_config)
            prepared.append(str(run_dir))
    print(json.dumps({"status": "PASS", "formal_runs_prepared": len(prepared), "runs": prepared}, indent=2))


if __name__ == "__main__":
    main()
