# -*- coding: utf-8 -*-
"""Stage 2.5: diagnose GMM multimodality versus covariance misspecification.

This script is deliberately separate from Stage 2.  It reuses Stage 2's
train/validation alignment and train-only standardization, never loads test
embeddings, and refuses to write inside ``outputs/stage2``.

Experiments
-----------
1. Original standardized z_f + diagonal GMM, K=1..5.
2. Train-fitted PCA-64/PCA-128 + diagonal/full GMM, K=1..5.
3. PCA-64 diagonal/full sensitivity at reg_covar 1e-4/1e-3/1e-2.

All GMMs are fitted on Known train only.  Known validation is used only for
average log-likelihood/NLL evaluation.  The fixed protocol is seed=0,
n_init=3, max_iter=200.  A deterministic train sample cap may be supplied when
full covariance is infeasible; the same sampled rows are then used for every
setting so diagonal/full comparisons remain paired and auditable.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
import tempfile
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

# Reuse the Stage 2 loading/alignment and train-only standardization protocol.
from task10_stage2_audit import _load_aligned, _standardize  # noqa: E402


K_RANGE = (1, 2, 3, 4, 5)
PCA_DIMS = (64, 128)
SENSITIVITY_REGS = (1e-4, 1e-3, 1e-2)
DEFAULT_CLASSES = ("FTP", "Cridex", "Miuref", "Outlook")
SEED = 0
N_INIT = 3
MAX_ITER = 200


@dataclass(frozen=True)
class Setting:
    representation: str
    pca_dim: Optional[int]
    covariance_type: str
    reg_covar: float
    max_iter: int = MAX_ITER

    @property
    def matrix_key(self) -> str:
        return "original" if self.pca_dim is None else f"pca{self.pca_dim}"

    @property
    def label(self) -> str:
        if self.pca_dim is None:
            return "Original-diag"
        return f"PCA{self.pca_dim}-{self.covariance_type}"


MAIN_SETTINGS = (
    # Exact Stage 2 review protocol keeps its original max_iter=100.
    Setting("Original", None, "diag", 1e-3, max_iter=100),
    Setting("PCA64", 64, "diag", 1e-3),
    Setting("PCA64", 64, "full", 1e-3),
    Setting("PCA128", 128, "diag", 1e-3),
    Setting("PCA128", 128, "full", 1e-3),
)


DIAGNOSIS_FIELDS = (
    "fit_id", "class", "class_id", "representation", "pca_dim",
    "covariance_type", "reg_covar", "K", "seed", "n_init", "max_iter",
    "train_samples_available", "train_samples", "val_samples",
    "sampling_applied", "bic_train", "train_avg_loglik", "val_avg_loglik",
    "val_nll", "delta_val_nll_from_K1", "delta_avg_loglik_from_K1",
    "best_K_by_val_nll", "best_K_by_bic", "delta_val_nll_K1_Kbest",
    "delta_avg_loglik_K1_Kbest", "relative_to_K1", "min_component_size",
    "min_component_ratio", "tiny_components_0_5pct", "tiny_components_1pct",
    "tiny_components_2pct", "converged", "n_iter", "warning_count", "figures",
)


def _normalise_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def resolve_classes(
    class_to_id: Mapping[str, int], requested: Sequence[str]
) -> List[Tuple[str, int]]:
    """Match requested class names case/format-insensitively against label_map."""
    normalised: Dict[str, Tuple[str, int]] = {}
    for name, class_id in class_to_id.items():
        key = _normalise_name(name)
        if key in normalised:
            raise ValueError(f"ambiguous normalized class name in label map: {name!r}")
        normalised[key] = (name, int(class_id))
    matched = []
    missing = []
    for name in requested:
        item = normalised.get(_normalise_name(name))
        if item is None:
            missing.append(name)
        elif item not in matched:
            matched.append(item)
    if missing:
        raise ValueError(
            f"classes not found in label map: {missing}; available={sorted(class_to_id)}"
        )
    if len(matched) < 4:
        raise ValueError("Stage 2.5 first pass requires at least four representative classes")
    return matched


def fit_pca_train_only(
    train: np.ndarray, val: np.ndarray, n_components: int, seed: int = SEED
) -> Tuple[PCA, np.ndarray, np.ndarray]:
    """Fit PCA on train and only transform validation; never fit on validation."""
    if n_components > min(train.shape):
        raise ValueError(
            f"PCA-{n_components} requires at least {n_components} train rows/features; "
            f"got {train.shape}"
        )
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=seed)
    train_pca = pca.fit_transform(train)
    val_pca = pca.transform(val)
    return pca, train_pca.astype(np.float32), val_pca.astype(np.float32)


def fixed_sample_indices(
    indices: np.ndarray, cap: Optional[int], seed: int, class_id: int
) -> np.ndarray:
    """Return a deterministic sorted subset; no sampling occurs when cap is None."""
    if cap is None or len(indices) <= cap:
        return indices
    if cap < max(K_RANGE):
        raise ValueError(f"sample cap must be at least {max(K_RANGE)}")
    rng = np.random.default_rng(seed + 1009 * int(class_id))
    return np.sort(rng.choice(indices, size=cap, replace=False))


def _fit_id(class_id: int, setting: Setting, k: int) -> str:
    reg = f"{setting.reg_covar:.0e}".replace("+", "")
    return (
        f"class{class_id}__{setting.representation.lower()}__"
        f"{setting.covariance_type}__reg{reg}__k{k}"
    )


def fit_gmm(
    class_name: str,
    class_id: int,
    setting: Setting,
    k: int,
    x_train: np.ndarray,
    x_val: np.ndarray,
    train_samples_available: int,
    sampling_applied: bool,
) -> Tuple[dict, List[dict], List[dict]]:
    """Fit one GMM on train and evaluate train/validation with full warning capture."""
    fit_id = _fit_id(class_id, setting, k)
    model = GaussianMixture(
        n_components=k,
        covariance_type=setting.covariance_type,
        reg_covar=setting.reg_covar,
        n_init=N_INIT,
        max_iter=setting.max_iter,
        random_state=SEED,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(x_train)
    warning_rows = [
        {
            "fit_id": fit_id,
            "class": class_name,
            "class_id": class_id,
            "representation": setting.representation,
            "pca_dim": "" if setting.pca_dim is None else setting.pca_dim,
            "covariance_type": setting.covariance_type,
            "reg_covar": setting.reg_covar,
            "K": k,
            "warning_index": index,
            "warning_category": warning.category.__name__,
            "warning_message": str(warning.message),
        }
        for index, warning in enumerate(caught, start=1)
    ]
    if not model.converged_ and not any(
        row["warning_category"] == "ConvergenceWarning" for row in warning_rows
    ):
        warning_rows.append(
            {
                "fit_id": fit_id,
                "class": class_name,
                "class_id": class_id,
                "representation": setting.representation,
                "pca_dim": "" if setting.pca_dim is None else setting.pca_dim,
                "covariance_type": setting.covariance_type,
                "reg_covar": setting.reg_covar,
                "K": k,
                "warning_index": len(warning_rows) + 1,
                "warning_category": "ConvergenceStatus",
                "warning_message": "GaussianMixture.converged_ is False",
            }
        )

    assignments = model.predict(x_train)
    sizes = np.bincount(assignments, minlength=k)
    ratios = sizes / len(x_train)
    component_rows = []
    for component, (size, ratio, weight) in enumerate(
        zip(sizes, ratios, model.weights_)
    ):
        component_rows.append(
            {
                "fit_id": fit_id,
                "class": class_name,
                "class_id": class_id,
                "representation": setting.representation,
                "pca_dim": "" if setting.pca_dim is None else setting.pca_dim,
                "covariance_type": setting.covariance_type,
                "reg_covar": setting.reg_covar,
                "K": k,
                "component": component,
                "component_size": int(size),
                "component_ratio": float(ratio),
                "mixture_weight": float(weight),
                "tiny_below_0_5pct": int(ratio < 0.005),
                "tiny_below_1pct": int(ratio < 0.01),
                "tiny_below_2pct": int(ratio < 0.02),
            }
        )

    train_loglik = float(model.score(x_train))
    val_loglik = float(model.score(x_val))
    row = {
        "fit_id": fit_id,
        "class": class_name,
        "class_id": class_id,
        "representation": setting.representation,
        "pca_dim": "" if setting.pca_dim is None else setting.pca_dim,
        "covariance_type": setting.covariance_type,
        "reg_covar": setting.reg_covar,
        "K": k,
        "seed": SEED,
        "n_init": N_INIT,
        "max_iter": setting.max_iter,
        "train_samples_available": train_samples_available,
        "train_samples": len(x_train),
        "val_samples": len(x_val),
        "sampling_applied": int(sampling_applied),
        "bic_train": float(model.bic(x_train)),
        "train_avg_loglik": train_loglik,
        "val_avg_loglik": val_loglik,
        "val_nll": -val_loglik,
        "min_component_size": int(sizes.min()),
        "min_component_ratio": float(ratios.min()),
        "tiny_components_0_5pct": int((ratios < 0.005).sum()),
        "tiny_components_1pct": int((ratios < 0.01).sum()),
        "tiny_components_2pct": int((ratios < 0.02).sum()),
        "converged": int(model.converged_),
        "n_iter": int(model.n_iter_),
        "warning_count": len(warning_rows),
        "figures": "",
    }
    return row, component_rows, warning_rows


def decorate_group_metrics(rows: List[dict]) -> None:
    """Add K=1 deltas and setting-level best-K fields in place."""
    groups: Dict[Tuple, List[dict]] = {}
    for row in rows:
        key = (
            row["class_id"], row["representation"], row["pca_dim"],
            row["covariance_type"], row["reg_covar"],
        )
        groups.setdefault(key, []).append(row)
    for group in groups.values():
        by_k = {int(row["K"]): row for row in group}
        if set(by_k) != set(K_RANGE):
            raise ValueError(f"incomplete K range for {group[0]['fit_id']}")
        k1 = by_k[1]
        best_nll = min(group, key=lambda row: (row["val_nll"], row["K"]))
        best_bic = min(group, key=lambda row: (row["bic_train"], row["K"]))
        best_delta = k1["val_nll"] - best_nll["val_nll"]
        for row in group:
            delta = k1["val_nll"] - row["val_nll"]
            row["delta_val_nll_from_K1"] = delta
            row["delta_avg_loglik_from_K1"] = (
                row["val_avg_loglik"] - k1["val_avg_loglik"]
            )
            row["best_K_by_val_nll"] = int(best_nll["K"])
            row["best_K_by_bic"] = int(best_bic["K"])
            row["delta_val_nll_K1_Kbest"] = best_delta
            row["delta_avg_loglik_K1_Kbest"] = (
                best_nll["val_avg_loglik"] - k1["val_avg_loglik"]
            )
            row["relative_to_K1"] = delta / max(abs(k1["val_nll"]), 1e-12)


def _write_csv(path: Path, rows: Sequence[Mapping], fieldnames: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "class"


def make_figures(rows: List[dict], figures_dir: Path) -> Dict[int, str]:
    """Create the two required trend figures per class for the main settings."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir.mkdir(parents=True, exist_ok=True)
    output: Dict[int, str] = {}
    class_pairs = sorted({(int(row["class_id"]), row["class"]) for row in rows})
    for class_id, class_name in class_pairs:
        subset = [row for row in rows if int(row["class_id"]) == class_id]
        paths = []
        for metric, ylabel, suffix in (
            ("val_nll", "Validation NLL (lower is better)", "validation_nll"),
            ("bic_train", "Train BIC (trend only)", "bic"),
        ):
            fig, ax = plt.subplots(figsize=(8, 5.5))
            for setting in MAIN_SETTINGS:
                line = sorted(
                    [
                        row for row in subset
                        if row["representation"] == setting.representation
                        and row["covariance_type"] == setting.covariance_type
                        and np.isclose(row["reg_covar"], setting.reg_covar)
                    ],
                    key=lambda row: int(row["K"]),
                )
                ax.plot(
                    [row["K"] for row in line],
                    [row[metric] for row in line],
                    marker="o",
                    label=setting.label,
                )
            ax.set_xticks(K_RANGE)
            ax.set_xlabel("Number of GMM components (K)")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{class_name}: K vs {suffix.replace('_', ' ')}")
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
            fig.tight_layout()
            rel = f"figures/{_slug(class_name)}_{suffix}.png"
            fig.savefig(figures_dir.parent / rel, dpi=180)
            plt.close(fig)
            paths.append(rel)
        output[class_id] = ";".join(paths)
    for row in rows:
        row["figures"] = output[int(row["class_id"])]
    return output


def _groups(rows: Sequence[dict]) -> Dict[Tuple, List[dict]]:
    result: Dict[Tuple, List[dict]] = {}
    for row in rows:
        key = (
            int(row["class_id"]), row["class"], row["representation"],
            row["covariance_type"], float(row["reg_covar"]),
        )
        result.setdefault(key, []).append(row)
    return result


def build_summary(
    main_rows: Sequence[dict],
    sensitivity_rows: Sequence[dict],
    pca_metadata: Mapping[int, Mapping[str, float]],
    sample_cap: Optional[int],
    material_delta: float,
    warning_count: int,
) -> Tuple[str, str]:
    """Answer the six required questions and return (markdown, A/B/C sentence)."""
    main_groups = _groups(main_rows)
    sensitivity_groups = _groups(sensitivity_rows)

    def group(class_id, representation, covariance, reg=1e-3):
        for key, value in main_groups.items():
            if (
                key[0] == class_id and key[2] == representation
                and key[3] == covariance and np.isclose(key[4], reg)
            ):
                return sorted(value, key=lambda row: int(row["K"]))
        raise KeyError((class_id, representation, covariance, reg))

    class_pairs = sorted({(int(row["class_id"]), row["class"]) for row in main_rows})
    h1_rows = []
    h2_rows = []
    pca_consistency = []
    for class_id, class_name in class_pairs:
        for dim in PCA_DIMS:
            rep = f"PCA{dim}"
            diag = group(class_id, rep, "diag")
            full = group(class_id, rep, "full")
            diag_by_k = {int(row["K"]): row for row in diag}
            full_by_k = {int(row["K"]): row for row in full}
            diag_best45 = min((diag_by_k[4], diag_by_k[5]), key=lambda row: row["val_nll"])
            full_k1 = full_by_k[1]
            gap = full_k1["val_nll"] - diag_best45["val_nll"]
            diag_gain = diag_by_k[1]["val_nll"] - diag_best45["val_nll"]
            explained = (
                (diag_by_k[1]["val_nll"] - full_k1["val_nll"]) / diag_gain
                if diag_gain > 0 else float("nan")
            )
            h1_rows.append((class_name, dim, gap, diag_gain, explained))
            full_best = min(full, key=lambda row: (row["val_nll"], row["K"]))
            delta = full_k1["val_nll"] - full_best["val_nll"]
            h2_rows.append((class_name, dim, int(full_best["K"]), delta))
        for covariance in ("diag", "full"):
            k64 = int(min(group(class_id, "PCA64", covariance), key=lambda r: (r["val_nll"], r["K"]))["K"])
            k128 = int(min(group(class_id, "PCA128", covariance), key=lambda r: (r["val_nll"], r["K"]))["K"])
            pca_consistency.append((class_name, covariance, k64, k128))

    h1_substantial = [row for row in h1_rows if np.isfinite(row[4]) and row[4] >= 0.8]
    h2_material = [row for row in h2_rows if row[2] > 1 and row[3] >= material_delta]

    main_setting_groups = list(main_groups.values())
    val_hits = sum(
        int(min(rows, key=lambda row: (row["val_nll"], row["K"]))["K"] == 5)
        for rows in main_setting_groups
    )
    bic_hits = sum(
        int(min(rows, key=lambda row: (row["bic_train"], row["K"]))["K"] == 5)
        for rows in main_setting_groups
    )

    def ranking_disagreement(first: Sequence[int], second: Sequence[int]) -> float:
        """Normalized pairwise ordering disagreement in [0, 1]."""
        position_first = {value: index for index, value in enumerate(first)}
        position_second = {value: index for index, value in enumerate(second)}
        discordant = 0
        total = 0
        for left in K_RANGE:
            for right in K_RANGE:
                if left >= right:
                    continue
                total += 1
                order_first = position_first[left] < position_first[right]
                order_second = position_second[left] < position_second[right]
                discordant += int(order_first != order_second)
        return discordant / total

    sensitivity_by_pair: Dict[Tuple[int, str, str], List[List[dict]]] = {}
    for key, rows in sensitivity_groups.items():
        sensitivity_by_pair.setdefault((key[0], key[1], key[3]), []).append(rows)
    unstable_pairs = []
    sensitivity_lines = []
    for (class_id, class_name, covariance), groups_for_regs in sorted(sensitivity_by_pair.items()):
        details = []
        best_ks = []
        tiny_counts = []
        rankings = []
        for rows in sorted(groups_for_regs, key=lambda rs: float(rs[0]["reg_covar"])):
            best = min(rows, key=lambda row: (row["val_nll"], row["K"]))
            best_ks.append(int(best["K"]))
            tiny_counts.append(int(best["tiny_components_0_5pct"]))
            ranking = tuple(
                int(row["K"])
                for row in sorted(rows, key=lambda row: (row["val_nll"], row["K"]))
            )
            rankings.append(ranking)
            details.append(
                f"reg={float(best['reg_covar']):.0e}:K={best['K']},"
                f"delta={best['delta_val_nll_K1_Kbest']:.6g},"
                f"rank={','.join(map(str, ranking))},"
                f"tiny<0.5%={best['tiny_components_0_5pct']}"
            )
        max_rank_disagreement = max(
            (
                ranking_disagreement(rankings[i], rankings[j])
                for i in range(len(rankings))
                for j in range(i + 1, len(rankings))
            ),
            default=0.0,
        )
        tiny_range = max(tiny_counts) - min(tiny_counts)
        unstable = (
            max(best_ks) - min(best_ks) >= 2
            or max_rank_disagreement >= 0.5
            or tiny_range >= 2
        )
        if unstable:
            unstable_pairs.append((class_name, covariance))
        sensitivity_lines.append(
            f"| {class_name} | {covariance} | {'; '.join(details)} | "
            f"{max_rank_disagreement:.2f} | {tiny_range} | "
            f"{'yes' if unstable else 'no'} |"
        )

    pca_agree = sum(1 for _, _, k64, k128 in pca_consistency if k64 == k128)
    tiny_fits = sum(int(row["tiny_components_0_5pct"] > 0) for row in main_rows)
    unconverged = sum(int(not row["converged"]) for row in main_rows)
    numeric_unstable = (
        len(unstable_pairs) >= max(1, len(sensitivity_by_pair) // 2)
        or unconverged >= max(1, len(main_rows) // 10)
    )
    h1_supported = len(h1_substantial) >= max(1, len(h1_rows) // 2)
    h2_supported = len(h2_material) >= max(1, int(np.ceil(0.75 * len(h2_rows))))
    if numeric_unstable:
        conclusion = "C. Result is numerically unstable / inconclusive."
    elif h1_supported:
        conclusion = (
            "B. Evidence suggests covariance misspecification explains a substantial "
            "part of the apparent multimodality."
        )
    elif h2_supported:
        conclusion = "A. Evidence supports genuine multi-component structure."
    else:
        conclusion = "C. Result is numerically unstable / inconclusive."

    lines = [
        "# Stage 2.5 GMM Multimodality Validity Diagnosis",
        "",
        f"Generated UTC: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}",
        "",
        "## Protocol and leakage controls",
        "",
        "- Reused Stage 2 `_load_aligned` and `_standardize`; only train/val triplets are loaded.",
        "- PCA is fitted once on all Known train z_f and only transforms Known validation.",
        "- Every GMM is fitted only on class-specific Known train; validation is evaluation-only.",
        "- Test embeddings are neither accepted as arguments nor read by this script.",
        f"- GMM protocol: K=1..5, seed={SEED}, n_init={N_INIT}; Original-diag "
        "keeps Stage 2 max_iter=100, while PCA experiments use max_iter=200.",
        f"- Deterministic train sample cap: {sample_cap if sample_cap else 'none (full class train data)'}.",
        "- Absolute delta is primary: delta_NLL = NLL(K=1) - NLL(best); positive means K>1 improves validation.",
        "- Cross-dimensional absolute likelihood/BIC values are not compared; H1 comparisons are within the same PCA space.",
        "- BIC plots are trend diagnostics only and are not used to assert a true mode count.",
        "",
        "## PCA metadata",
        "",
        "| PCA dim | train rows | original dim | explained variance ratio sum |",
        "|---:|---:|---:|---:|",
    ]
    for dim in PCA_DIMS:
        meta = pca_metadata[dim]
        lines.append(
            f"| {dim} | {int(meta['train_rows'])} | {int(meta['original_dim'])} | "
            f"{meta['explained_variance_ratio_sum']:.6f} |"
        )
    lines += [
        "",
        "## Required questions",
        "",
        "### 1. Does K=1 become substantially stronger with full covariance?",
        "",
        "Comparison is PCA-full K=1 versus PCA-diag best of K=4/5 in the same PCA space. "
        "`gap <= 0` means full K=1 is at least as good; explained fraction >=0.8 is treated as substantial.",
        "",
        "| class | PCA dim | full-K1 NLL minus diag-best45 NLL | diag K1-to-best45 delta | explained fraction |",
        "|---|---:|---:|---:|---:|",
    ]
    for class_name, dim, gap, diag_gain, explained in h1_rows:
        exp_text = "nan" if not np.isfinite(explained) else f"{explained:.4f}"
        lines.append(f"| {class_name} | {dim} | {gap:.6f} | {diag_gain:.6f} | {exp_text} |")
    lines += [
        "",
        f"Substantial comparisons: {len(h1_substantial)}/{len(h1_rows)}.",
        "",
        "### 2. Under full covariance, does K>1 still improve held-out validation?",
        "",
        f"A material absolute delta is operationalized as >= {material_delta:g} average NLL units per sample.",
        "",
        "| class | PCA dim | best K by val NLL | delta NLL K1-to-best |",
        "|---|---:|---:|---:|",
    ]
    for class_name, dim, best_k, delta in h2_rows:
        lines.append(f"| {class_name} | {dim} | {best_k} | {delta:.6f} |")
    lines += [
        "",
        f"Material K>1 comparisons: {len(h2_material)}/{len(h2_rows)}.",
        "",
        "### 3. Does the best K still hit Kmax=5?",
        "",
        f"- Validation-NLL best K=5: {val_hits}/{len(main_setting_groups)} class-setting groups.",
        f"- Train-BIC best K=5: {bic_hits}/{len(main_setting_groups)} class-setting groups.",
        "",
        "### 4. Are PCA-64 and PCA-128 conclusions consistent?",
        "",
        "| class | covariance | PCA64 best K | PCA128 best K |",
        "|---|---|---:|---:|",
    ]
    for class_name, covariance, k64, k128 in pca_consistency:
        lines.append(f"| {class_name} | {covariance} | {k64} | {k128} |")
    lines += [
        "",
        f"Exact best-K agreement: {pca_agree}/{len(pca_consistency)} comparisons.",
        "",
        "### 5. Does reg_covar materially affect the conclusion?",
        "",
        "A setting is flagged unstable when the best-K range is at least 2, the maximum "
        "pairwise NLL-ranking disagreement is at least 0.5, or the best model's number "
        "of <0.5% components changes by at least 2.",
        "",
        "| class | covariance | sensitivity detail | max ranking disagreement | tiny-count range | unstable |",
        "|---|---|---|---:|---:|---|",
        *sensitivity_lines,
        "",
        f"Unstable class-covariance pairs: {len(unstable_pairs)}/{len(sensitivity_by_pair)}.",
        "",
        "### 6. Do tiny components remain common?",
        "",
        f"- Main-experiment fits containing at least one component below 0.5%: {tiny_fits}/{len(main_rows)}.",
        f"- Non-converged main fits: {unconverged}/{len(main_rows)}.",
        f"- Captured warning records across unique fits: {warning_count}.",
        "- Exact per-component sizes, ratios, weights and 0.5%/1%/2% flags are in `component_size_statistics.csv`.",
        "",
        "## Final diagnosis",
        "",
        conclusion,
        "",
        "This conclusion concerns evidence for multiple Gaussian components, not a claim that every class "
        "contains five semantically real subgroups. No K beyond 5 was fitted in Stage 2.5.",
    ]
    return "\n".join(lines) + "\n", conclusion


def _validate_input_files(zt_dir: Path, zg_dir: Path, label_map_path: Path) -> None:
    required = [label_map_path]
    for directory in (zt_dir, zg_dir):
        for part in ("train", "val"):
            required.extend(
                directory / f"{prefix}_{part}.npy"
                for prefix in ("embedding", "labels", "flow_ids")
            )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Stage 2.5 requires existing Stage 1 train/val artifacts; missing:\n  "
            + "\n  ".join(missing)
        )


def _safe_output_dir(root: Path, value: str) -> Path:
    output_dir = Path(value)
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir = output_dir.resolve()
    protected = (root / "outputs" / "stage2").resolve()
    if output_dir == protected or protected in output_dir.parents:
        raise ValueError("refusing to write to outputs/stage2 or any of its descendants")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty Stage 2.5 output: {output_dir}; "
            "choose a new --out directory"
        )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2.5 covariance diagnosis")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--zt-dir", default=None)
    parser.add_argument("--zg-dir", default=None)
    parser.add_argument(
        "--label-map", default=None,
        help="Default: <root>/data/fig_graph/all_flows/label_map.json",
    )
    parser.add_argument(
        "--out", default="outputs/stage2_5_covariance_diagnosis",
    )
    parser.add_argument("--classes", default=",".join(DEFAULT_CLASSES))
    parser.add_argument(
        "--max-train-per-class", type=int, default=None,
        help="Deterministic cap for infeasible full-covariance runs; applies to all settings",
    )
    parser.add_argument(
        "--material-delta-nll", type=float, default=0.01,
        help="Transparent per-sample absolute NLL threshold used by the summary heuristic",
    )
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    zt_dir = Path(args.zt_dir).resolve() if args.zt_dir else root / "outputs/stage1/modelA/embeddings"
    zg_dir = Path(args.zg_dir).resolve() if args.zg_dir else root / "outputs/stage1/modelB/seeds/seed2"
    label_map_path = Path(args.label_map).resolve() if args.label_map else root / "data/fig_graph/all_flows/label_map.json"
    output_dir = _safe_output_dir(root, args.out)
    _validate_input_files(zt_dir, zg_dir, label_map_path)

    label_payload = json.loads(label_map_path.read_text(encoding="utf-8"))
    class_pairs = resolve_classes(
        label_payload["class_to_id"],
        [item.strip() for item in args.classes.split(",") if item.strip()],
    )

    print("[1/6] Load train/val embeddings and reuse Stage 2 alignment/standardization")
    zt, labels, zg = _load_aligned(zt_dir, zg_dir)
    _zt_standardized, zf = _standardize(zt, zg)
    del zt, zg, _zt_standardized
    zf = {part: values.astype(np.float32, copy=False) for part, values in zf.items()}
    if set(labels) != {"train", "val"} or set(zf) != {"train", "val"}:
        raise AssertionError("only train/val data must be loaded")

    print("[2/6] Fit PCA on Known train only; transform train and validation")
    matrices = {"original": zf}
    pca_metadata = {}
    for dim in PCA_DIMS:
        pca, train_pca, val_pca = fit_pca_train_only(zf["train"], zf["val"], dim)
        matrices[f"pca{dim}"] = {"train": train_pca, "val": val_pca}
        pca_metadata[dim] = {
            "train_rows": len(zf["train"]),
            "val_rows": len(zf["val"]),
            "original_dim": zf["train"].shape[1],
            "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
            "fit_split": "train",
            "transform_splits": ["train", "val"],
            "random_state": SEED,
        }
        print(f"  PCA-{dim}: explained variance={pca.explained_variance_ratio_.sum():.6f}")

    sensitivity_settings = tuple(
        Setting("PCA64", 64, covariance, reg)
        for covariance in ("diag", "full")
        for reg in SENSITIVITY_REGS
    )
    unique_settings = list(dict.fromkeys(MAIN_SETTINGS + sensitivity_settings))

    print("[3/6] Fit GMMs on class-specific Known train; validation is evaluation-only")
    cache: Dict[Tuple[int, Setting, int], Tuple[dict, List[dict], List[dict]]] = {}
    all_component_rows: List[dict] = []
    all_warning_rows: List[dict] = []
    train_indices = {}
    val_indices = {}
    for class_name, class_id in class_pairs:
        available = np.flatnonzero(labels["train"] == class_id)
        val_idx = np.flatnonzero(labels["val"] == class_id)
        if len(available) < max(K_RANGE) or len(val_idx) == 0:
            raise ValueError(
                f"{class_name}: insufficient train/val rows ({len(available)}/{len(val_idx)})"
            )
        train_indices[class_id] = fixed_sample_indices(
            available, args.max_train_per_class, SEED, class_id
        )
        val_indices[class_id] = val_idx

    total = len(class_pairs) * len(unique_settings) * len(K_RANGE)
    completed = 0
    started = time.time()
    for class_name, class_id in class_pairs:
        for setting in unique_settings:
            data = matrices[setting.matrix_key]
            x_train = data["train"][train_indices[class_id]]
            x_val = data["val"][val_indices[class_id]]
            available = int((labels["train"] == class_id).sum())
            sampled = len(x_train) != available
            for k in K_RANGE:
                key = (class_id, setting, k)
                try:
                    fitted = fit_gmm(
                        class_name, class_id, setting, k, x_train, x_val,
                        available, sampled,
                    )
                except (MemoryError, np.linalg.LinAlgError) as exc:
                    raise RuntimeError(
                        f"full diagnosis failed at {class_name}/{setting.label}/K={k}: {exc}. "
                        "Rerun with a fixed --max-train-per-class value; the script will "
                        "record the cap and apply the same sample to all settings."
                    ) from exc
                cache[key] = fitted
                all_component_rows.extend(fitted[1])
                all_warning_rows.extend(fitted[2])
                completed += 1
                print(
                    f"  {completed}/{total} {class_name} {setting.label} "
                    f"reg={setting.reg_covar:.0e} K={k} "
                    f"val_nll={fitted[0]['val_nll']:.6f}"
                )

    main_rows = [
        dict(cache[(class_id, setting, k)][0])
        for class_name, class_id in class_pairs
        for setting in MAIN_SETTINGS
        for k in K_RANGE
    ]
    sensitivity_rows = [
        dict(cache[(class_id, setting, k)][0])
        for class_name, class_id in class_pairs
        for setting in sensitivity_settings
        for k in K_RANGE
    ]
    decorate_group_metrics(main_rows)
    decorate_group_metrics(sensitivity_rows)

    print("[4/6] Stage outputs in a temporary directory")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage_dir = Path(tempfile.mkdtemp(prefix=".stage2_5_covariance_", dir=output_dir.parent))
    try:
        figure_map = make_figures(main_rows, stage_dir / "figures")
        for row in sensitivity_rows:
            row["figures"] = figure_map[int(row["class_id"])]

        component_fields = (
            "fit_id", "class", "class_id", "representation", "pca_dim",
            "covariance_type", "reg_covar", "K", "component", "component_size",
            "component_ratio", "mixture_weight", "tiny_below_0_5pct",
            "tiny_below_1pct", "tiny_below_2pct",
        )
        warning_fields = (
            "fit_id", "class", "class_id", "representation", "pca_dim",
            "covariance_type", "reg_covar", "K", "warning_index",
            "warning_category", "warning_message",
        )
        _write_csv(stage_dir / "covariance_diagnosis.csv", main_rows, DIAGNOSIS_FIELDS)
        _write_csv(stage_dir / "reg_covar_sensitivity.csv", sensitivity_rows, DIAGNOSIS_FIELDS)
        _write_csv(
            stage_dir / "component_size_statistics.csv",
            all_component_rows,
            component_fields,
        )
        _write_csv(
            stage_dir / "convergence_warnings.csv",
            all_warning_rows,
            warning_fields,
        )

        summary_text, conclusion = build_summary(
            main_rows,
            sensitivity_rows,
            pca_metadata,
            args.max_train_per_class,
            args.material_delta_nll,
            len(all_warning_rows),
        )
        (stage_dir / "diagnosis_summary.md").write_text(summary_text, encoding="utf-8")
        metadata = {
            "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "project_root": str(root),
            "zt_dir": str(zt_dir),
            "zg_dir": str(zg_dir),
            "label_map": str(label_map_path),
            "classes": [{"class": name, "class_id": cid} for name, cid in class_pairs],
            "loaded_splits": ["train", "val"],
            "test_loaded": False,
            "standardization": "reused task10 _standardize; train statistics only",
            "pca": pca_metadata,
            "gmm": {
                "K": list(K_RANGE), "seed": SEED, "n_init": N_INIT,
                "max_iter": {"Original-diag": 100, "PCA": MAX_ITER},
            },
            "max_train_per_class": args.max_train_per_class,
            "elapsed_seconds": time.time() - started,
            "conclusion": conclusion,
            "output_files": sorted(
                str(path.relative_to(stage_dir))
                for path in stage_dir.rglob("*") if path.is_file()
            ),
        }
        (stage_dir / "run_metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        if output_dir.exists():
            output_dir.rmdir()  # only an existing empty directory is allowed
        os.replace(stage_dir, output_dir)
    except Exception:
        shutil.rmtree(stage_dir, ignore_errors=True)
        raise

    print("[5/6] Wrote auditable CSV, warning log, summary, metadata and figures")
    print(f"[6/6] Complete: {output_dir}")
    print(f"Conclusion: {conclusion}")


if __name__ == "__main__":
    main()
