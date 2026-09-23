#!/usr/bin/env python3
"""Run the frozen train-only PCA64 full-GMM audit on deterministic mu_x."""

from __future__ import annotations

import argparse
import csv
import json
import math
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


AUDIT_ROOT = Path(__file__).resolve().parents[1]
CLASSES = ("FTP", "Cridex", "Miuref", "Outlook")
SEEDS = (0, 1, 2)
K_VALUES = (1, 2, 3)
EXPECTED_ROWS = {"train": 391_280, "val": 48_910}


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str] | None = None) -> None:
    fieldnames = fields if fields is not None else list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=AUDIT_ROOT / "artifacts/latent_gaussian_audit",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=AUDIT_ROOT / "outputs/latent_gaussian_audit",
    )
    parser.add_argument("--reg-covar", type=float, default=1e-3)
    parser.add_argument("--n-init", type=int, default=3)
    parser.add_argument("--max-iter", type=int, default=300)
    args = parser.parse_args()
    if args.max_iter < 300 or args.n_init < 3:
        raise ValueError("formal protocol requires max_iter>=300 and n_init>=3")
    if not math.isclose(args.reg_covar, 1e-3, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError("main audit reg_covar must remain 1e-3")

    frames: dict[str, pd.DataFrame] = {}
    for split in ("train", "val"):
        frame = pd.read_parquet(args.artifact_dir / f"{split}_mu.parquet")
        if len(frame) != EXPECTED_ROWS[split]:
            raise ValueError(f"{split} latent row count mismatch")
        if set(frame["split"].unique()) != {split}:
            raise ValueError(f"{split} parquet split metadata is invalid")
        if not frame["flow_id"].is_unique:
            raise ValueError(f"{split} flow IDs are not unique")
        frames[split] = frame
    if set(frames["train"]["flow_id"]).intersection(frames["val"]["flow_id"]):
        raise ValueError("train/validation flow-ID overlap")
    mu_columns = [column for column in frames["train"].columns if column.startswith("mu_")]
    mu_columns.sort(key=lambda name: int(name.split("_")[1]))
    if len(mu_columns) != 128:
        raise ValueError(f"expected 128 latent columns, found {len(mu_columns)}")
    train_mu = frames["train"][mu_columns].to_numpy(dtype=np.float64, copy=True)
    val_mu = frames["val"][mu_columns].to_numpy(dtype=np.float64, copy=True)
    if not np.isfinite(train_mu).all() or not np.isfinite(val_mu).all():
        raise ValueError("non-finite latent found before Gaussian audit")
    train_names = frames["train"]["class_name"].to_numpy()
    val_names = frames["val"]["class_name"].to_numpy()

    scaler = StandardScaler().fit(train_mu)
    train_scaled = scaler.transform(train_mu).astype(np.float64, copy=False)
    val_scaled = scaler.transform(val_mu).astype(np.float64, copy=False)
    actual_dim = min(64, train_scaled.shape[1])
    pca = PCA(n_components=actual_dim, svd_solver="randomized", random_state=0)
    train_pca = pca.fit_transform(train_scaled).astype(np.float64, copy=False)
    val_pca = pca.transform(val_scaled).astype(np.float64, copy=False)
    joblib.dump(scaler, args.artifact_dir / "standard_scaler.joblib")
    joblib.dump(pca, args.artifact_dir / "pca64.joblib")

    gmm_dir = args.artifact_dir / "gmms"
    gmm_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    weight_rows: list[dict[str, object]] = []
    covariance_rows: list[dict[str, object]] = []
    warning_rows: list[dict[str, object]] = []
    for class_name in CLASSES:
        x_train = train_pca[train_names == class_name]
        x_val = val_pca[val_names == class_name]
        if not len(x_train) or not len(x_val):
            raise ValueError(f"empty class split: {class_name}")
        for seed in SEEDS:
            for k in K_VALUES:
                model = GaussianMixture(
                    n_components=k,
                    covariance_type="full",
                    reg_covar=args.reg_covar,
                    n_init=args.n_init,
                    max_iter=args.max_iter,
                    random_state=seed,
                )
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    model.fit(x_train)
                train_assignments = model.predict(x_train)
                val_assignments = model.predict(x_val)
                train_counts = np.bincount(train_assignments, minlength=k)
                val_counts = np.bincount(val_assignments, minlength=k)
                train_ratios = train_counts / len(x_train)
                val_ratios = val_counts / len(x_val)
                tv_distance = float(0.5 * np.abs(train_ratios - val_ratios).sum())
                cholesky_failures = 0
                for component, covariance in enumerate(model.covariances_):
                    eigenvalues = np.linalg.eigvalsh(covariance)
                    cholesky_success = True
                    try:
                        np.linalg.cholesky(covariance)
                    except np.linalg.LinAlgError:
                        cholesky_success = False
                        cholesky_failures += 1
                    covariance_rows.append(
                        {
                            "class_name": class_name,
                            "seed": seed,
                            "K": k,
                            "component": component,
                            "min_eigenvalue": float(eigenvalues.min()),
                            "max_eigenvalue": float(eigenvalues.max()),
                            "condition_number": float(eigenvalues.max() / eigenvalues.min()),
                            "cholesky_success": cholesky_success,
                        }
                    )
                    weight_rows.append(
                        {
                            "class_name": class_name,
                            "seed": seed,
                            "K": k,
                            "component": component,
                            "model_weight": float(model.weights_[component]),
                            "train_count": int(train_counts[component]),
                            "train_empirical_ratio": float(train_ratios[component]),
                            "validation_count": int(val_counts[component]),
                            "validation_empirical_ratio": float(val_ratios[component]),
                            "tiny_component_below_1pct": bool(train_ratios[component] < 0.01),
                            "very_tiny_component_below_0_5pct": bool(
                                train_ratios[component] < 0.005
                            ),
                            "validation_component_empty": bool(val_counts[component] == 0),
                            "train_val_component_weight_tv": tv_distance,
                        }
                    )
                warning_messages = []
                for item in caught:
                    message = f"{item.category.__name__}: {item.message}"
                    warning_messages.append(message)
                    warning_rows.append(
                        {
                            "class_name": class_name,
                            "seed": seed,
                            "K": k,
                            "warning": message,
                        }
                    )
                if not model.converged_:
                    warning_messages.append("GaussianMixture.converged_=False")
                    warning_rows.append(
                        {
                            "class_name": class_name,
                            "seed": seed,
                            "K": k,
                            "warning": "GaussianMixture.converged_=False",
                        }
                    )
                model_path = gmm_dir / f"{class_name}_seed{seed}_k{k}.joblib"
                joblib.dump(model, model_path)
                rows.append(
                    {
                        "class_name": class_name,
                        "representation": "Open-Detect deterministic mu_x",
                        "pca_dim": actual_dim,
                        "covariance_type": "full",
                        "reg_covar": args.reg_covar,
                        "K": k,
                        "seed": seed,
                        "n_init": args.n_init,
                        "max_iter": args.max_iter,
                        "train_samples": len(x_train),
                        "validation_samples": len(x_val),
                        "train_avg_nll": -float(model.score(x_train)),
                        "validation_avg_nll": -float(model.score(x_val)),
                        "converged": bool(model.converged_),
                        "n_iter": int(model.n_iter_),
                        "component_weights": json.dumps(model.weights_.tolist()),
                        "min_component_weight": float(model.weights_.min()),
                        "min_train_component_ratio": float(train_ratios.min()),
                        "min_validation_component_ratio": float(val_ratios.min()),
                        "min_validation_component_count": int(val_counts.min()),
                        "train_val_component_weight_tv": tv_distance,
                        "tiny_component_count_below_1pct": int((train_ratios < 0.01).sum()),
                        "very_tiny_component_count_below_0_5pct": int(
                            (train_ratios < 0.005).sum()
                        ),
                        "cholesky_failures": cholesky_failures,
                        "warning": " | ".join(warning_messages),
                        "model_path": str(model_path.resolve()),
                    }
                )
                print(
                    f"{class_name} seed={seed} K={k} train_nll={rows[-1]['train_avg_nll']:.6f} val_nll={rows[-1]['validation_avg_nll']:.6f}",
                    flush=True,
                )

    by_key = {
        (str(row["class_name"]), int(row["seed"]), int(row["K"])): row for row in rows
    }
    for class_name in CLASSES:
        for seed in SEEDS:
            k1 = float(by_key[(class_name, seed, 1)]["validation_avg_nll"])
            for k in K_VALUES:
                row = by_key[(class_name, seed, k)]
                row["delta_nll_from_k1"] = k1 - float(row["validation_avg_nll"])

    summary_rows: list[dict[str, object]] = []
    for class_name in CLASSES:
        summary: dict[str, object] = {"class_name": class_name}
        eligible: list[int] = []
        for k in K_VALUES:
            selected = [by_key[(class_name, seed, k)] for seed in SEEDS]
            nll = np.asarray([float(row["validation_avg_nll"]) for row in selected])
            summary[f"k{k}_validation_nll_mean"] = float(nll.mean())
            summary[f"k{k}_validation_nll_std"] = float(nll.std(ddof=0))
            if k == 1:
                continue
            deltas = np.asarray([float(row["delta_nll_from_k1"]) for row in selected])
            all_seed_delta_gt_1 = bool(np.all(deltas > 1.0))
            min_ratio = min(float(row["min_train_component_ratio"]) for row in selected)
            min_val_count = min(int(row["min_validation_component_count"]) for row in selected)
            max_tv = max(float(row["train_val_component_weight_tv"]) for row in selected)
            numerical_ok = all(
                bool(row["converged"])
                and int(row["cholesky_failures"]) == 0
                and not str(row["warning"])
                for row in selected
            )
            nondegenerate = min_ratio >= 0.01 and min_val_count > 0 and max_tv <= 0.10
            if all_seed_delta_gt_1 and numerical_ok and nondegenerate:
                eligible.append(k)
            summary.update(
                {
                    f"delta_nll{k}_mean": float(deltas.mean()),
                    f"delta_nll{k}_min": float(deltas.min()),
                    f"delta_nll{k}_max": float(deltas.max()),
                    f"delta_nll{k}_std": float(deltas.std(ddof=0)),
                    f"k{k}_all_seed_delta_gt_1": all_seed_delta_gt_1,
                    f"k{k}_min_train_component_ratio": min_ratio,
                    f"k{k}_min_validation_component_count": min_val_count,
                    f"k{k}_max_train_val_tv": max_tv,
                    f"k{k}_numerically_stable": numerical_ok,
                    f"k{k}_nondegenerate": nondegenerate,
                }
            )
        summary["supported_multi_K"] = ",".join(map(str, eligible))
        summary["opendetect_multi_supported"] = bool(eligible)
        summary["opendetect_diagnosis"] = "Multi" if eligible else "Single"
        summary_rows.append(summary)

    write_csv(args.output_dir / "gaussian_results.csv", rows)
    write_csv(args.output_dir / "gaussian_summary_by_class.csv", summary_rows)
    write_csv(args.output_dir / "component_weights.csv", weight_rows)
    write_csv(args.output_dir / "covariance_diagnostics.csv", covariance_rows)
    write_csv(
        args.output_dir / "gaussian_warnings.csv",
        warning_rows,
        ["class_name", "seed", "K", "warning"],
    )
    metadata = {
        "loaded_splits": ["train", "val"],
        "test_loaded": False,
        "standard_scaler_fit": "all 20-class train mu_x only",
        "pca_fit": "all 20-class train mu_x only",
        "pca_dim_requested": 64,
        "pca_dim_actual": actual_dim,
        "pca_random_state": 0,
        "pca_explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
        "gmm_fit": "class-specific train only",
        "gmm_evaluation": "matching held-out validation only",
        "classes": list(CLASSES),
        "K": list(K_VALUES),
        "seeds": list(SEEDS),
        "covariance_type": "full",
        "reg_covar": args.reg_covar,
        "n_init": args.n_init,
        "max_iter": args.max_iter,
        "float_dtype": "float64",
        "tiny_component_threshold": "assigned train ratio < 0.01",
        "very_tiny_component_threshold": "assigned train ratio < 0.005",
        "multi_rule": "at least one of K=2/3 has DeltaNLL>1 nat/sample in all seeds, all fits converged/warning-free/Cholesky-valid, minimum train component ratio >=1%, validation component count >0, and max train/validation component-weight TV <=0.10",
        "reg_covar_sensitivity_run": False,
    }
    (args.output_dir / "gaussian_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
