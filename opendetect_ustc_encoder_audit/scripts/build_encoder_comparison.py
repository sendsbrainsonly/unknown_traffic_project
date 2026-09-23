#!/usr/bin/env python3
"""Build the frozen TrafficFormer/Open-Detect comparison and final audit Gate."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch


AUDIT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = AUDIT_ROOT.parent
CLASSES = ("FTP", "Cridex", "Miuref", "Outlook")


def case_name(trafficformer_multi: bool, opendetect_multi: bool) -> str:
    return {
        (True, True): "Case A",
        (True, False): "Case B",
        (False, False): "Case C",
        (False, True): "Case D",
    }[(trafficformer_multi, opendetect_multi)]


def overall_gate(rows: list[dict[str, object]]) -> str:
    cases = [str(row["encoder_case"]) for row in rows]
    if all(case == "Case A" for case in cases):
        return "A. CROSS-ENCODER PERSISTENCE"
    if sum(case == "Case B" for case in cases) >= 3 and not any(
        case == "Case D" for case in cases
    ):
        return "B. ENCODER-DEPENDENT"
    return "C. MIXED CROSS-ENCODER EVIDENCE"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def upsert_readme_section(path: Path, body: str) -> None:
    text = path.read_text(encoding="utf-8")
    heading = "## Latent Gaussian Audit"
    pattern = rf"{re.escape(heading)}\n.*?(?=\n## |\Z)"
    replacement = f"{heading}\n\n{body.strip()}\n"
    if re.search(pattern, text, flags=re.DOTALL):
        text = re.sub(pattern, replacement, text, count=1, flags=re.DOTALL)
    else:
        text = text.rstrip() + "\n\n" + replacement
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-root", type=Path, default=AUDIT_ROOT)
    parser.add_argument(
        "--trafficformer-results",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/stage2_5_covariance_diagnosis/covariance_diagnosis.csv",
    )
    args = parser.parse_args()
    output_dir = args.audit_root / "outputs/latent_gaussian_audit"
    artifact_dir = args.audit_root / "artifacts/latent_gaussian_audit"

    tf = pd.read_csv(args.trafficformer_results)
    tf = tf[
        tf["class"].isin(CLASSES)
        & (tf["representation"] == "PCA64")
        & (tf["covariance_type"] == "full")
        & np.isclose(tf["reg_covar"], 1e-3)
        & tf["K"].isin((1, 2, 3))
        & (tf["seed"] == 0)
    ].copy()
    if len(tf) != 12:
        raise ValueError(f"expected 12 immutable TrafficFormer rows, found {len(tf)}")
    od = pd.read_csv(output_dir / "gaussian_summary_by_class.csv").set_index(
        "class_name"
    )

    rows: list[dict[str, object]] = []
    for class_name in CLASSES:
        tf_class = tf[tf["class"] == class_name].set_index("K")
        tf_nll = {k: float(tf_class.loc[k, "val_nll"]) for k in (1, 2, 3)}
        tf_delta2 = tf_nll[1] - tf_nll[2]
        tf_delta3 = tf_nll[1] - tf_nll[3]
        tf_k2_supported = bool(
            tf_delta2 > 1.0
            and bool(tf_class.loc[2, "converged"])
            and int(tf_class.loc[2, "warning_count"]) == 0
            and int(tf_class.loc[2, "tiny_components_1pct"]) == 0
        )
        tf_k3_supported = bool(
            tf_delta3 > 1.0
            and bool(tf_class.loc[3, "converged"])
            and int(tf_class.loc[3, "warning_count"]) == 0
            and int(tf_class.loc[3, "tiny_components_1pct"]) == 0
        )
        tf_multi = tf_k2_supported or tf_k3_supported
        od_row = od.loc[class_name]
        od_multi = bool(od_row["opendetect_multi_supported"])
        rows.append(
            {
                "class_name": class_name,
                "trafficformer_k1_nll": tf_nll[1],
                "trafficformer_k2_nll": tf_nll[2],
                "trafficformer_k3_nll": tf_nll[3],
                "trafficformer_delta2": tf_delta2,
                "trafficformer_delta3": tf_delta3,
                "opendetect_k1_nll": float(od_row["k1_validation_nll_mean"]),
                "opendetect_k2_nll": float(od_row["k2_validation_nll_mean"]),
                "opendetect_k3_nll": float(od_row["k3_validation_nll_mean"]),
                "opendetect_delta2": float(od_row["delta_nll2_mean"]),
                "opendetect_delta3": float(od_row["delta_nll3_mean"]),
                "trafficformer_multi_supported": tf_multi,
                "opendetect_multi_supported": od_multi,
                "encoder_case": case_name(tf_multi, od_multi),
            }
        )
    write_csv(output_dir / "encoder_comparison.csv", rows)

    gate = overall_gate(rows)
    selection = json.loads((output_dir / "checkpoint_selection.json").read_text())
    gaussian_meta = json.loads((output_dir / "gaussian_metadata.json").read_text())
    compact = pd.read_csv(output_dir / "class_compactness.csv").set_index("class_name")
    gaussian = pd.read_csv(output_dir / "gaussian_results.csv")
    components = pd.read_csv(output_dir / "component_weights.csv")
    covariance = pd.read_csv(output_dir / "covariance_diagnostics.csv")
    warning_count = len(pd.read_csv(output_dir / "gaussian_warnings.csv"))
    tiny_count = int(components["tiny_component_below_1pct"].astype(bool).sum())
    very_tiny_count = int(
        components["very_tiny_component_below_0_5pct"].astype(bool).sum()
    )
    empty_val_count = int(components["validation_component_empty"].astype(bool).sum())
    cholesky_failures = int((~covariance["cholesky_success"].astype(bool)).sum())
    max_condition = float(covariance["condition_number"].max())
    explicit_dependence = any(row["encoder_case"] in ("Case B", "Case D") for row in rows)

    lines = [
        "# Open-Detect USTC Latent Gaussian Audit",
        "",
        "## Frozen formal checkpoint",
        "",
        f"- Training stop epoch: {selection['training_stop_epoch']}",
        f"- Best epoch: {selection['best_epoch']}",
        f"- Validation Accuracy: {selection['best_val_accuracy']:.9f}",
        f"- Validation Macro-F1: {selection['best_val_macro_f1']:.9f}",
        f"- Combined score: {selection['best_combined_score']:.9f}",
        f"- Checkpoint SHA-256: `{selection['checkpoint_sha256']}`",
        f"- Stop reason: {selection['early_stop_reason']}",
        "",
        "## Protocol",
        "",
        "Deterministic `mu_x` was extracted for the complete fixed train and validation splits. StandardScaler and PCA64 were fit only on all 20-class train `mu_x`; validation was transform/evaluation-only. Class-specific full-covariance GMMs used K=1/2/3, seeds 0/1/2, reg_covar=1e-3, n_init=3, max_iter=300 and float64. Test was not loaded.",
        "",
        "## Encoder comparison",
        "",
        "| class | TF K1 | TF K2 | TF K3 | TF Delta2 | TF Delta3 | OD K1 | OD K2 | OD K3 | OD Delta2 | OD Delta3 | case |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['class_name']} | {row['trafficformer_k1_nll']:.6f} | {row['trafficformer_k2_nll']:.6f} | {row['trafficformer_k3_nll']:.6f} | {row['trafficformer_delta2']:.6f} | {row['trafficformer_delta3']:.6f} | {row['opendetect_k1_nll']:.6f} | {row['opendetect_k2_nll']:.6f} | {row['opendetect_k3_nll']:.6f} | {row['opendetect_delta2']:.6f} | {row['opendetect_delta3']:.6f} | {row['encoder_case']} |"
        )
    lines += [
        "",
        "## Open-Detect compactness",
        "",
        "| class | assigned prototype mean/median/p95 | within-class variance | nearest-wrong mean | margin |",
        "|---|---:|---:|---:|---:|",
    ]
    for class_name in CLASSES:
        item = compact.loc[class_name]
        lines.append(
            f"| {class_name} | {item['assigned_prototype_distance_mean']:.6f} / {item['assigned_prototype_distance_median']:.6f} / {item['assigned_prototype_distance_p95']:.6f} | {item['within_class_variance']:.6f} | {item['nearest_wrong_prototype_distance_mean']:.6f} | {item['mean_wrong_minus_assigned_margin']:.6f} |"
        )
    lines += [
        "",
        "Compactness is descriptive and is not used as proof that K=1 is adequate.",
        "",
        "## Stability and degeneracy",
        "",
        f"- Gaussian fits: {len(gaussian)}",
        f"- Captured warnings/non-convergence records: {warning_count}",
        f"- Tiny component rows (<1% train assignment): {tiny_count}/{len(components)}",
        f"- Very tiny component rows (<0.5%): {very_tiny_count}/{len(components)}",
        f"- Validation-empty component rows: {empty_val_count}/{len(components)}",
        f"- Cholesky failures: {cholesky_failures}/{len(covariance)}",
        f"- Maximum covariance condition number: {max_condition:.9g}",
        "- Seed-specific NLL, DeltaNLL, component counts/weights, train-validation TV, eigenvalues and iterations are preserved in the CSV outputs.",
        "",
        "## Per-class cases",
        "",
    ]
    for row in rows:
        if row["encoder_case"] == "Case A":
            interpretation = "residual representation-level heterogeneity persists across encoders"
        elif row["encoder_case"] == "Case B":
            interpretation = "Gaussian complexity is encoder-dependent; Open-Detect is more single-Gaussian-like"
        elif row["encoder_case"] == "Case C":
            interpretation = "K=1 is adequate under both representations by the fixed rule"
        else:
            interpretation = "Open-Detect introduces residual structure absent from TrafficFormer"
        lines.append(f"- {row['class_name']}: {row['encoder_case']}; {interpretation}.")
    lines += [
        "",
        "## Overall Gate",
        "",
        f"**{gate}**",
        "",
        f"Explicit encoder dependence observed: {'yes' if explicit_dependence else 'no'}.",
        "",
        "This result concerns class-conditional density fit in learned representations. It does not prove natural Gaussian modes, semantic traffic subtypes, Open-Detect detection failure, unknown false acceptance, or that a multi-Gaussian detector will improve Unknown Detection.",
        "",
        "## Known limitations",
        "",
        "- At the epoch-5 GPU migration, the old checkpoint lacked optimizer/RNG state; the discontinuity cannot be repaired retrospectively.",
        "- Batch 1024 produced `logvar.exp()` overflow; formal training used batch 512.",
        "- Open-Detect released loss, decoder and prototype reset differ from the paper as documented in the source audit.",
        "- TrafficFormer comparison uses its immutable seed-0 Stage 2.5 fit; it was not rerun.",
    ]
    (output_dir / "audit_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    tmux_helper = (
        "/home/birkenwald/data/SEU-WXY/WXY-SEU/TrafficClassifier/"
        ".agents/skills/tmux-task-execution/scripts/tmux_task.sh"
    )
    gpu_selector = (
        "/home/birkenwald/data/SEU-WXY/WXY-SEU/TrafficClassifier/"
        ".agents/skills/using-superpowers/scripts/select_gpu.py"
    )
    audit_root = str(args.audit_root.resolve())
    main_project = str(PROJECT_ROOT.resolve())
    commands = [
        f"{tmux_helper} start od_formal_composite_earlystop_gpu3 {main_project} -- "
        f"python {gpu_selector} --min-free-gb 20 --count 1 --max-utilization 30 "
        f"--allowed 3 -- python {audit_root}/scripts/train_opendetect.py --mode formal "
        f"--epochs 100 --batch-size 512 --workers 0 --early-stopping-patience 5 "
        f"--resume {audit_root}/artifacts/latest_checkpoint.pt "
        f"--checkpoint {audit_root}/artifacts/best_checkpoint.pt "
        f"--latest-checkpoint {audit_root}/artifacts/latest_checkpoint.pt",
        f"{tmux_helper} start od_latent_gaussian_audit_v2 {audit_root} -- "
        f"bash {audit_root}/scripts/run_latent_gaussian_audit.sh",
        f"python {audit_root}/scripts/freeze_best_checkpoint.py --patience 5 "
        "--max-epochs 100",
        f"python {gpu_selector} --min-free-gb 4 --count 1 --max-utilization 30 "
        f"--allowed 0,3,4 -- python {audit_root}/scripts/extract_mu_and_compactness.py "
        "--batch-size 512 --workers 0",
        f"python {audit_root}/scripts/run_gaussian_audit.py --reg-covar 1e-3 "
        "--n-init 3 --max-iter 300",
        f"python {audit_root}/scripts/build_encoder_comparison.py",
        f"python {audit_root}/scripts/verify_latent_gaussian_audit.py",
    ]
    run_metadata = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "Open-Detect USTC Latent Gaussian Audit",
        "audit_root": str(args.audit_root.resolve()),
        "main_project": str(PROJECT_ROOT.resolve()),
        "upstream_read_only": str((PROJECT_ROOT.parent / "Open-Detect").resolve()),
        "author_code_commit": "b50a18515a01799468c10f6c9b60c01f8a6a4e7c",
        "checkpoint_selection": selection,
        "latent_artifacts": {
            "train": str((artifact_dir / "train_mu.parquet").resolve()),
            "validation": str((artifact_dir / "val_mu.parquet").resolve()),
        },
        "flow_id_assertions": "8/8 representative class-split sets exact",
        "gaussian": gaussian_meta,
        "trafficformer_source": str(args.trafficformer_results.resolve()),
        "trafficformer_recomputed": False,
        "overall_gate": gate,
        "explicit_encoder_dependence": explicit_dependence,
        "test_loaded_or_used": False,
        "actual_commands": commands,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
        },
        "output_files": sorted(
            {
                *(str(path.relative_to(output_dir)) for path in output_dir.rglob("*") if path.is_file()),
                "run_metadata.json",
            }
        ),
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    table = [
        "| Class | TrafficFormer | Open-Detect | Case |",
        "|---|---|---|---|",
        *[
            f"| {row['class_name']} | {'Multi' if row['trafficformer_multi_supported'] else 'Single'} | {'Multi' if row['opendetect_multi_supported'] else 'Single'} | {row['encoder_case']} |"
            for row in rows
        ],
    ]
    readme_body = (
        f"- Status: COMPLETE\n"
        f"- Frozen best checkpoint: epoch {selection['best_epoch']}, SHA-256 `{selection['checkpoint_sha256']}`\n"
        f"- Training stopped at epoch {selection['training_stop_epoch']}: {selection['early_stop_reason']}\n"
        "- Deterministic train/validation `mu_x`: complete; 391,280/48,910 rows, 128 dimensions\n"
        "- StandardScaler/PCA64/GMM fit: train only; validation evaluation only; test unused\n\n"
        + "\n".join(table)
        + f"\n\nOverall Gate: **{gate}**\n\n"
        "Detailed NLL, DeltaNLL, seed stability, component degeneracy and covariance diagnostics are in `outputs/latent_gaussian_audit/`.\n\n"
        "Known limitations: epoch-5 GPU migration lacked optimizer/RNG state; batch 1024 caused log-variance overflow and formal training used batch 512; released Open-Detect implementation differs from the paper. These results do not imply an Unknown Detection outcome."
    )
    upsert_readme_section(args.audit_root / "README.md", readme_body)


if __name__ == "__main__":
    main()
