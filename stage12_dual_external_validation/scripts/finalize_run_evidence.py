#!/usr/bin/env python3
"""Write the canonical evidence bundle metadata for one Stage 12 run.

This script is deliberately documentation-only: it reads existing run metadata
and results, and never imports or evaluates model/test arrays.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


REQUIRED_HEADINGS = (
    "## Data and split",
    "## Configuration and execution",
    "## Core results",
    "## Preserved evidence",
    "## Limitations",
    "## Conclusion and next step",
)


def utc_from_timestamp(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def metric_rows(final_results: dict | None, pretest: dict) -> list[dict]:
    rows = [
        {"name": "best_known_validation_accuracy", "value": pretest["best_validation_accuracy"], "unit": "fraction"},
        {"name": "best_epoch", "value": pretest["best_epoch"], "unit": "epoch"},
        {"name": "completed_epochs", "value": pretest["completed_epochs"], "unit": "epoch"},
    ]
    if final_results is None:
        return rows
    for method, metrics in final_results["methods"].items():
        for name in ("auroc", "auprc", "ufar", "known_frr", "known_macro_f1", "known_accuracy"):
            rows.append({"name": f"{method}.{name}", "value": metrics[name], "unit": "fraction"})
    for name, value in final_results["delta_m1_minus_m0"].items():
        rows.append({"name": f"delta_m1_minus_m0.{name}", "value": value, "unit": "fraction"})
    return rows


def build_results(pretest: dict, run_config: dict, final_results: dict | None) -> str:
    dataset = pretest["dataset"]
    setting = pretest["setting"]
    seed = pretest["seed"]
    final = final_results is not None
    status = "success" if final else "partial"
    lines = [
        f"# Experiment results: stage12-{dataset}-{setting}-seed{seed}",
        "",
        f"- Status: `{status}`",
        "- Experiment type: `external-open-set-validation`",
        "- Claim scope: `independent-external-confirmation`",
        f"- Objective: Compare Open-Detect Native with DES-v0 on {dataset}/{setting} under the frozen Unknown-free protocol.",
        "",
        "## Data and split",
        "",
        f"- Known classes: `{len(pretest['known_classes'])}`; unknown classes: `{len(pretest['unknown_classes'])}`.",
        f"- Known class names: `{', '.join(pretest['known_classes'])}`.",
        f"- Unknown class names: `{', '.join(pretest['unknown_classes'])}`.",
        f"- Known Train / Known Validation samples: `{run_config['train_samples']}` / `{run_config['validation_samples']}`.",
        f"- Protocol hash: `{pretest['protocol_hash']}`.",
        "- Known Test and Unknown Test manifests were frozen before training.",
        "",
        "## Configuration and execution",
        "",
        f"- Seed: `{seed}`; architecture: `{run_config['training']['architecture']}`; latent dimension: `{run_config['training']['latent_dim']}`.",
        f"- Batch size: `{run_config['training']['batch_size']}`; maximum epochs: `{run_config['training']['epochs']}`.",
        f"- Selected physical GPU recorded by the run: `{run_config.get('cuda_visible_devices', 'unknown')}`.",
        f"- Checkpoint selection: `{run_config['training']['checkpoint_criterion']}`.",
        f"- Completed epochs: `{pretest['completed_epochs']}`; stopped early: `{pretest['stopped_early']}`.",
        "",
        "## Core results",
        "",
        f"- Best Known-Validation accuracy: `{pretest['best_validation_accuracy']:.6f}` at epoch `{pretest['best_epoch']}`.",
    ]
    if final:
        for method, metrics in final_results["methods"].items():
            lines.append(
                f"- {method}: AUROC `{metrics['auroc']:.6f}`, AUPRC `{metrics['auprc']:.6f}`, "
                f"UFAR `{metrics['ufar']:.6f}`, Known FRR `{metrics['known_frr']:.6f}`, "
                f"Known Macro-F1 `{metrics['known_macro_f1']:.6f}`."
            )
        delta = final_results["delta_m1_minus_m0"]
        lines.append(
            f"- M1-M0 delta: AUROC `{delta['auroc']:+.6f}`, UFAR `{delta['ufar']:+.6f}`, "
            f"Known FRR `{delta['known_frr']:+.6f}`, Known Macro-F1 `{delta['known_macro_f1']:+.6f}`."
        )
    else:
        lines.append("- Final Known/Unknown test metrics: not computed; the one-shot test remains unopened.")
    lines.extend([
        "",
        "## Preserved evidence",
        "",
        "- `model_best.pt`, checkpoint and centroid SHA-256 files, train/validation logs, frozen manifests, thresholds, and feature evidence.",
        "- `manifest.json` inventories every in-bundle file after the workspace refresh step.",
        "",
        "## Limitations",
        "",
        "- Application labels inherit the controlled-capture filename taxonomy.",
        "- The dataset protocol may require the recorded FLOW_DISJOINT_ONLY claim reduction when source-group splitting is infeasible.",
    ])
    if final:
        lines.extend([
            "- The final test was opened once under frozen code; method changes and reruns are prohibited.",
            "",
            "## Conclusion and next step",
            "",
            "- This run is complete. Preserve it unchanged for Stage 12 cross-seed and cross-dataset aggregation.",
        ])
    else:
        lines.extend([
            "- This is a pre-test partial result; no Known Test or Unknown Test array was loaded.",
            "",
            "## Conclusion and next step",
            "",
            "- Keep the run frozen until every expected encoder is READY_FOR_ONE_SHOT_TEST, then execute the single authorized final-test opening.",
        ])
    text = "\n".join(lines) + "\n"
    if not all(heading in text for heading in REQUIRED_HEADINGS):
        raise RuntimeError("generated RESULTS.md is missing a required heading")
    return text


def finalize(run_dir: Path, phase: str) -> None:
    run_dir = run_dir.resolve()
    pretest_path = run_dir / "pretest_freeze.json"
    config_path = run_dir / "config.json"
    if not pretest_path.is_file() or not config_path.is_file():
        raise FileNotFoundError("pretest_freeze.json and config.json are required")
    if not (run_dir / "READY_FOR_ONE_SHOT_TEST").is_file():
        raise RuntimeError("run is not READY_FOR_ONE_SHOT_TEST")
    pretest = json.loads(pretest_path.read_text(encoding="utf-8"))
    run_config = json.loads(config_path.read_text(encoding="utf-8"))
    final_path = run_dir / "final_results.json"
    final_results = json.loads(final_path.read_text(encoding="utf-8")) if final_path.is_file() else None
    if phase == "final" and final_results is None:
        raise RuntimeError("final phase requires final_results.json")
    if phase == "pretest" and final_results is not None:
        raise RuntimeError("pretest phase refuses a run that already has final results")

    old_manifest_path = run_dir / "manifest.json"
    old_manifest = json.loads(old_manifest_path.read_text(encoding="utf-8")) if old_manifest_path.is_file() else {}
    created_at = old_manifest.get("created_at_utc")
    if not created_at:
        created_at = utc_from_timestamp(min(path.stat().st_mtime for path in run_dir.iterdir() if path.is_file()))
    final = final_results is not None
    gpu_ids = [int(part) for part in str(run_config.get("cuda_visible_devices", "")).split(",") if part.strip().isdigit()]
    status = "success" if final else "partial"
    objective = (
        "Compare Open-Detect Native and DES-v0 on a frozen independent external dataset with Unknown-free "
        "training, validation, checkpoint selection, prototypes, and thresholds."
    )
    manifest = {
        "schema_version": 1,
        "experiment_id": f"stage12-{pretest['dataset']}-{pretest['setting']}-seed{pretest['seed']}",
        "created_at_utc": created_at,
        "updated_at_utc": utc_now(),
        "status": status,
        "experiment_type": "external-open-set-validation",
        "claim_scope": "independent-external-confirmation",
        "objective": objective,
        "inputs": [{
            "dataset": pretest["dataset"],
            "setting": pretest["setting"],
            "protocol_sha256": pretest["protocol_hash"],
            "train_array_sha256": pretest["train_array_sha256"],
            "validation_array_sha256": pretest["validation_array_sha256"],
            "known_test_array_sha256_frozen_not_opened": pretest["known_test_array_sha256_frozen_not_opened"],
            "unknown_test_array_sha256_frozen_not_opened": pretest["unknown_test_array_sha256_frozen_not_opened"],
        }],
        "code": {
            "revision": pretest["code_hash"],
            "dirty": bool(run_config.get("stage12_git", {}).get("status")),
            "changes": ["Workspace state is captured verbatim in config.json; evidence packaging does not alter method code."],
        },
        "execution": {
            "tmux_session": None,
            "command": f"python -B scripts/train_pretest.py --dataset {pretest['dataset']} --setting {pretest['setting']} --seed {pretest['seed']}",
            "exit_code": 0,
            "environment": "/home/birkenwald/data/SEU-WXY/conda_envs/2025-10-8-WXY-dgl_py310",
            "physical_gpu_ids": gpu_ids,
        },
        "configuration": {
            "files": ["config.json", "pretest_freeze.json"],
            "parameters": run_config["training"],
            "seeds": [pretest["seed"]],
        },
        "core_results": metric_rows(final_results, pretest),
        "artifacts": [item for item in old_manifest.get("artifacts", []) if item.get("scope") == "external"],
        "limitations": [
            "Application labels inherit controlled-capture filenames.",
            "The frozen protocol records any required FLOW_DISJOINT_ONLY claim reduction.",
        ] + ([] if final else ["Final Known Test and Unknown Test metrics remain unopened and uncomputed."]),
        "next_step": (
            "Preserve unchanged for cross-seed and cross-dataset aggregation."
            if final else
            "Wait until all expected runs are ready, then execute the single authorized final-test opening."
        ),
        "pretest_freeze": pretest,
    }
    if final:
        manifest["final_results"] = final_results
    (run_dir / "RESULTS.md").write_text(build_results(pretest, run_config, final_results), encoding="utf-8")
    old_manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "run_dir": str(run_dir), "phase": phase}, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--phase", choices=("pretest", "final"), required=True)
    args = parser.parse_args()
    finalize(args.run_dir, args.phase)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
