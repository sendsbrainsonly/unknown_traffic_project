#!/usr/bin/env python3
"""Held-out PCA64/full-covariance Gaussian audit for CIC-IDS-2017 z_t."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sklearn
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALL_CLASSES = (
    "BENIGN", "DDoS", "DoS Hulk", "PortScan", "DoS GoldenEye",
    "DoS Slowhttptest", "DoS slowloris",
)
SEEDS = (0, 1, 2)
K_VALUES = (1, 2, 3)
PCA_DIM = 64
PCA_SEED = 42
REG_COVAR = 1e-3
N_INIT = 3
MAX_ITER = 300
TINY_WEIGHT = 0.01
# Declared before looking at results.  A stable meaningful improvement requires
# every seed to improve and a mean absolute gain of >=0.10 nats/sample.
MEANINGFUL_DELTA = 0.10
GMM_INPUT_DTYPE = np.float64


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def finite_array(path: Path) -> np.ndarray:
    value = np.load(path, mmap_mode="r")
    for start in range(0, len(value), 50_000):
        if not np.isfinite(value[start:start + 50_000]).all():
            raise ValueError(f"non-finite values: {path}")
    return value


def fit_train_representation(
    x_train_raw: np.ndarray, x_val_raw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, StandardScaler, PCA]:
    """Fit scaler/PCA on train only and promote GMM inputs to float64."""
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train_raw)
    x_val_scaled = scaler.transform(x_val_raw)
    pca = PCA(n_components=PCA_DIM, svd_solver="randomized", random_state=PCA_SEED)
    x_train = np.asarray(pca.fit_transform(x_train_scaled), dtype=GMM_INPUT_DTYPE)
    x_val = np.asarray(pca.transform(x_val_scaled), dtype=GMM_INPUT_DTYPE)
    return x_train, x_val, scaler, pca


def summarize(results: list[dict], classes: tuple[str, ...]) -> list[dict]:
    output: list[dict] = []
    for class_name in classes:
        by_seed_k = {
            (int(row["seed"]), int(row["K"])): float(row["val_avg_nll"])
            for row in results if row["class_name"] == class_name
        }
        row: dict[str, object] = {"class_name": class_name}
        for k in K_VALUES:
            values = np.asarray([by_seed_k[(seed, k)] for seed in SEEDS])
            row[f"val_nll_k{k}_mean"] = float(values.mean())
            row[f"val_nll_k{k}_std"] = float(values.std(ddof=0))
        for k in (2, 3):
            deltas = np.asarray([
                by_seed_k[(seed, 1)] - by_seed_k[(seed, k)] for seed in SEEDS
            ])
            baseline = np.asarray([abs(by_seed_k[(seed, 1)]) for seed in SEEDS])
            relative = np.divide(
                deltas, baseline, out=np.full_like(deltas, np.nan), where=baseline > 0
            )
            row[f"delta_nll_{k}_mean"] = float(deltas.mean())
            row[f"delta_nll_{k}_std"] = float(deltas.std(ddof=0))
            row[f"delta_nll_{k}_min"] = float(deltas.min())
            row[f"relative_{k}_mean"] = float(np.nanmean(relative))
            row[f"all_seeds_improve_k{k}"] = bool(np.all(deltas > 0))
            row[f"stable_meaningful_k{k}"] = bool(
                np.all(deltas > 0) and deltas.mean() >= MEANINGFUL_DELTA
            )
        means = {k: float(row[f"val_nll_k{k}_mean"]) for k in K_VALUES}
        row["lowest_mean_validation_nll_K"] = min(means, key=means.get)
        row["supports_residual_components"] = bool(
            row["stable_meaningful_k2"] or row["stable_meaningful_k3"]
        )
        output.append(row)
    return output


def gate_for(summary: list[dict]) -> tuple[str, int]:
    attacks = [row for row in summary if row["class_name"] != "BENIGN"]
    supporting = sum(bool(row["supports_residual_components"]) for row in attacks)
    if supporting >= 4:
        return "A. CROSS-DATASET SUPPORT", supporting
    if supporting >= 1:
        return "B. MIXED EVIDENCE", supporting
    return "C. DATASET-SPECIFIC / NOT REPLICATED", supporting


def markdown_summary(
    summary: list[dict], results: list[dict], components: list[dict], gate: str,
    supporting: int,
) -> str:
    lines = [
        "# CIC-IDS-2017 Cross-Dataset Gaussian Audit", "",
        "## Protocol", "",
        "For each class independently, `StandardScaler` and randomized PCA-64 were fit on train only. "
        "Validation was transform/evaluation only. Full-covariance Gaussian/GMM models used "
        "K=1/2/3, `reg_covar=1e-3`, `n_init=3`, `max_iter=300`, and seeds 0/1/2. "
        "No BIC, test set, Unknown data, metadata feature, or adaptive K was used.", "",
        "The decision rule was fixed before inspecting results: a K>1 comparison is stable and "
        "meaningful only if all three paired seed deltas are positive and mean absolute "
        f"ΔNLL is at least {MEANINGFUL_DELTA:.2f} nats/sample. Gate A requires a majority "
        "(at least 4 of 6) attack classes; 1-3 gives B; zero gives C.", "",
        "## Held-out validation results", "",
        "| class | K1 NLL | K2 NLL | K3 NLL | ΔNLL2 | ΔNLL3 | stable K2 | stable K3 |",
        "|---|---:|---:|---:|---:|---:|:---:|:---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['class_name']} | {row['val_nll_k1_mean']:.6f} ± {row['val_nll_k1_std']:.6f} "
            f"| {row['val_nll_k2_mean']:.6f} ± {row['val_nll_k2_std']:.6f} "
            f"| {row['val_nll_k3_mean']:.6f} ± {row['val_nll_k3_std']:.6f} "
            f"| {row['delta_nll_2_mean']:.6f} ± {row['delta_nll_2_std']:.6f} "
            f"| {row['delta_nll_3_mean']:.6f} ± {row['delta_nll_3_std']:.6f} "
            f"| {row['stable_meaningful_k2']} | {row['stable_meaningful_k3']} |"
        )
    nonconverged = sum(not bool(row["converged"]) for row in results)
    warning_runs = sum(bool(row["warning_messages"]) for row in results)
    tiny = sum(bool(row["tiny_component"]) for row in components)
    component_total = len(components)
    benign = next(row for row in summary if row["class_name"] == "BENIGN")
    attacks = [row for row in summary if row["class_name"] != "BENIGN"]
    lines += [
        "", "## BENIGN (reported separately)", "",
        f"BENIGN ΔNLL2={benign['delta_nll_2_mean']:.6f} and "
        f"ΔNLL3={benign['delta_nll_3_mean']:.6f}. BENIGN is a broad normal-traffic "
        "collection, so this result is not used to establish a universal one-class/one-Gaussian claim.",
        "", "## Attack classes", "",
        f"{supporting}/6 attack classes met the predeclared stable-meaningful rule. "
        + "; ".join(
            f"{row['class_name']}: Δ2={row['delta_nll_2_mean']:.4f}, "
            f"Δ3={row['delta_nll_3_mean']:.4f}, support={row['supports_residual_components']}"
            for row in attacks
        ) + ".",
        "", "## Numerical diagnostics", "",
        f"- Non-converged fits: {nonconverged}/{len(results)}.",
        f"- Fits emitting captured warnings: {warning_runs}/{len(results)}.",
        f"- Tiny components (weight < {TINY_WEIGHT:.2f}): {tiny}/{component_total} components.",
        "- Components are distributional Gaussian components only; they are not interpreted as semantic attack subtypes.",
        "", "## Final Gate", "",
        f"**{gate}**", "",
    ]
    if gate.startswith("A"):
        lines.append(
            "A majority of evaluated attack classes retain stable held-out likelihood improvement "
            "for K=2 and/or K=3 after full covariance, providing second-dataset support for "
            "one-Gaussian insufficiency. This does not establish semantic submodes or a true K."
        )
    elif gate.startswith("B"):
        lines.append(
            "Only part of the attack-class set retains stable held-out improvement after full "
            "covariance. The evidence is class-dependent and does not justify a uniform mixture assumption."
        )
    else:
        lines.append(
            "The attack classes do not retain stable meaningful held-out improvement after full "
            "covariance, so the USTC residual component result is not replicated here."
        )
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    classes = tuple(item.strip() for item in args.classes.split(",") if item.strip())
    unknown = set(classes) - set(ALL_CLASSES)
    if unknown:
        raise ValueError(f"unapproved classes: {sorted(unknown)}")
    embedding_dir = args.embedding_dir.resolve()
    z_train = finite_array(embedding_dir / "embedding_train.npy")
    z_val = finite_array(embedding_dir / "embedding_val.npy")
    y_train = np.load(embedding_dir / "class_names_train.npy", mmap_mode="r")
    y_val = np.load(embedding_dir / "class_names_val.npy", mmap_mode="r")
    flow_train = np.load(embedding_dir / "flow_ids_train.npy", mmap_mode="r")
    flow_val = np.load(embedding_dir / "flow_ids_val.npy", mmap_mode="r")
    if z_train.shape[1] != 768 or z_val.shape[1] != 768:
        raise ValueError(f"z_t dimension mismatch: {z_train.shape}, {z_val.shape}")
    overlap = set(flow_train.tolist()) & set(flow_val.tolist())
    if overlap:
        raise ValueError(f"train/validation flow ID overlap: {len(overlap)}")

    results: list[dict] = []
    components: list[dict] = []
    warnings_rows: list[dict] = []
    pca_rows: list[dict] = []
    started = time.time()
    for class_name in classes:
        train_idx = np.flatnonzero(np.asarray(y_train) == class_name)
        val_idx = np.flatnonzero(np.asarray(y_val) == class_name)
        if len(train_idx) <= PCA_DIM or len(val_idx) == 0:
            raise ValueError(f"insufficient samples for {class_name}: {len(train_idx)}/{len(val_idx)}")
        x_train_raw = np.asarray(z_train[train_idx], dtype=np.float32)
        x_val_raw = np.asarray(z_val[val_idx], dtype=np.float32)
        x_train, x_val, scaler, pca = fit_train_representation(x_train_raw, x_val_raw)
        pca_rows.append({
            "class_name": class_name,
            "train_samples": len(train_idx),
            "val_samples": len(val_idx),
            "input_dim": z_train.shape[1],
            "pca_dim": PCA_DIM,
            "pca_seed": PCA_SEED,
            "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
            "scaler_fit_split": "train",
            "pca_fit_split": "train",
            "validation_usage": "transform_and_evaluate_only",
            "gmm_input_dtype": str(x_train.dtype),
        })
        del x_train_raw, x_val_raw

        for seed in SEEDS:
            for k in K_VALUES:
                with warnings.catch_warnings(record=True) as captured:
                    warnings.simplefilter("always")
                    model = GaussianMixture(
                        n_components=k,
                        covariance_type="full",
                        reg_covar=REG_COVAR,
                        n_init=N_INIT,
                        max_iter=MAX_ITER,
                        random_state=seed,
                    )
                    model.fit(x_train)
                messages = [f"{type(item.message).__name__}: {item.message}" for item in captured]
                train_nll = -float(model.score(x_train))
                val_nll = -float(model.score(x_val))
                assignments = model.predict(x_train)
                hard_counts = np.bincount(assignments, minlength=k)
                conditions: list[float] = []
                for component in range(k):
                    covariance = model.covariances_[component]
                    eigenvalues = np.linalg.eigvalsh(covariance)
                    minimum = float(eigenvalues.min())
                    maximum = float(eigenvalues.max())
                    condition = maximum / minimum if minimum > 0 else float("inf")
                    conditions.append(condition)
                    weight = float(model.weights_[component])
                    components.append({
                        "class_name": class_name,
                        "K": k,
                        "seed": seed,
                        "component": component,
                        "weight": weight,
                        "hard_train_count": int(hard_counts[component]),
                        "hard_train_ratio": float(hard_counts[component] / len(x_train)),
                        "covariance_min_eigenvalue": minimum,
                        "covariance_max_eigenvalue": maximum,
                        "covariance_condition_number": condition,
                        "tiny_component": weight < TINY_WEIGHT,
                        "tiny_weight_threshold": TINY_WEIGHT,
                    })
                result = {
                    "class_name": class_name,
                    "representation": "z_t",
                    "standardization": "StandardScaler_fit_train_only",
                    "pca_dim": PCA_DIM,
                    "pca_fit_split": "train",
                    "covariance_type": "full",
                    "reg_covar": REG_COVAR,
                    "K": k,
                    "seed": seed,
                    "n_init": N_INIT,
                    "max_iter": MAX_ITER,
                    "gmm_input_dtype": str(x_train.dtype),
                    "train_samples": len(x_train),
                    "val_samples": len(x_val),
                    "train_avg_nll": train_nll,
                    "val_avg_nll": val_nll,
                    "converged": bool(model.converged_),
                    "n_iter": int(model.n_iter_),
                    "min_component_weight": float(model.weights_.min()),
                    "max_covariance_condition_number": float(max(conditions)),
                    "warning_messages": " | ".join(messages),
                }
                results.append(result)
                for message in messages:
                    warnings_rows.append({
                        "class_name": class_name, "K": k, "seed": seed,
                        "warning": message,
                        "is_convergence_warning": "ConvergenceWarning" in message,
                    })
                print(
                    f"{class_name} K={k} seed={seed} val_nll={val_nll:.6f} "
                    f"converged={model.converged_} min_weight={model.weights_.min():.6f}",
                    flush=True,
                )
        del x_train, x_val

    result_fields = list(results[0])
    component_fields = list(components[0])
    write_csv(args.output_dir / "gaussian_results.csv", results, result_fields)
    write_csv(args.output_dir / "component_weights.csv", components, component_fields)
    write_csv(
        args.output_dir / "gaussian_warnings.csv", warnings_rows,
        ["class_name", "K", "seed", "warning", "is_convergence_warning"],
    )
    write_csv(args.output_dir / "pca_manifest.csv", pca_rows, list(pca_rows[0]))
    summary = summarize(results, classes)
    write_csv(args.output_dir / "gaussian_summary_by_class.csv", summary, list(summary[0]))

    smoke = len(classes) == 3
    if smoke:
        smoke_summary = {
            "passed": len(results) == 27 and all(np.isfinite(float(row["val_avg_nll"])) for row in results),
            "classes": list(classes),
            "fit_count": len(results),
            "expected_fit_count": 27,
            "pca_dim": PCA_DIM,
            "all_finite": all(np.isfinite(float(row["val_avg_nll"])) for row in results),
            "nonconverged": sum(not bool(row["converged"]) for row in results),
        }
        (args.output_dir / "smoke_summary.json").write_text(
            json.dumps(smoke_summary, indent=2) + "\n", encoding="utf-8"
        )
        if not smoke_summary["passed"]:
            raise RuntimeError(f"smoke gate failed: {smoke_summary}")
        print(json.dumps(smoke_summary, indent=2), flush=True)
        return

    if classes != ALL_CLASSES:
        raise ValueError("formal audit must use exactly the seven approved classes in fixed order")
    gate, supporting = gate_for(summary)
    (args.output_dir / "audit_summary.md").write_text(
        markdown_summary(summary, results, components, gate, supporting), encoding="utf-8"
    )
    training_config = json.loads((args.output_dir / "training_config.json").read_text(encoding="utf-8"))
    prep_metadata = json.loads(
        (args.output_dir / "artifacts/preparation_metadata.json").read_text(encoding="utf-8")
    )
    run_metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "audit": "CIC-IDS-2017 Cross-Dataset Gaussian Audit",
        "gate": gate,
        "supporting_attack_classes": supporting,
        "classes": list(classes),
        "no_test_split": True,
        "unknown_data_used": False,
        "features": "TrafficFormer z_t only",
        "metadata_used_as_features": False,
        "fit_protocol": {
            "scaler": "StandardScaler fit train only",
            "pca": f"PCA-{PCA_DIM} randomized fit train only, seed {PCA_SEED}",
            "gmm": {
                "covariance_type": "full", "K": list(K_VALUES),
                "reg_covar": REG_COVAR, "n_init": N_INIT,
                "max_iter": MAX_ITER, "seeds": list(SEEDS),
            },
            "validation": "transform/evaluation only",
            "bic_used": False,
            "numerical_precision": {
                "embedding_storage_dtype": "float32",
                "gmm_input_dtype": "float64",
                "reason": "avoid full-covariance Cholesky loss of positive definiteness observed with float32 PCA coordinates",
            },
        },
        "decision_rule": {
            "stable_meaningful": f"all seed deltas > 0 and mean delta >= {MEANINGFUL_DELTA}",
            "gate_A": ">=4 of 6 attacks support",
            "gate_B": "1-3 of 6 attacks support",
            "gate_C": "0 of 6 attacks support",
            "tiny_component_weight_threshold": TINY_WEIGHT,
        },
        "training": training_config,
        "preparation": prep_metadata,
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
        },
        "gaussian_command": " ".join(sys.argv),
        "elapsed_seconds": time.time() - started,
        "output_hashes": {
            name: sha256_file(args.output_dir / name)
            for name in (
                "trafficformer_protocol.md", "sampling_manifest.csv", "split_manifest.csv",
                "training_config.json", "training_metrics.csv", "embedding_manifest.csv",
                "gaussian_results.csv", "gaussian_summary_by_class.csv",
                "component_weights.csv", "audit_summary.md",
            )
        },
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"gate": gate, "supporting_attack_classes": supporting}, indent=2), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--classes", default=",".join(ALL_CLASSES))
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
