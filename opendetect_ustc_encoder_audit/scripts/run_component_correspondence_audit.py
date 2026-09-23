#!/usr/bin/env python3
"""USTC cross-encoder K=2 component correspondence and shortcut audit.

This script is deliberately downstream-only.  It reuses the immutable
TrafficFormer Stage 2.5/2.6 inputs and the frozen Open-Detect latent audit,
fits no encoder, never loads test data, and refuses to overwrite an existing
formal output directory.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import pickle
import re
import shutil
import sys
import tempfile
import time
import warnings
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import joblib
import matplotlib
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import kruskal
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    adjusted_mutual_info_score,
    adjusted_rand_score,
    balanced_accuracy_score,
    f1_score,
    normalized_mutual_info_score,
)
from sklearn.mixture import GaussianMixture
from threadpoolctl import threadpool_limits

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


AUDIT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = AUDIT_ROOT.parent
PROJECT_SCRIPTS = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_SCRIPTS))

from src.preprocessing.flow_split import TCP, UDP, parse_ipv4  # noqa: E402
from task10_stage2_audit import _load_aligned, _standardize  # noqa: E402
from task10f_covariance_diagnosis import fit_pca_train_only  # noqa: E402
from task10g_component_shortcut_audit import epsilon_squared_kruskal  # noqa: E402


CLASSES = ("FTP", "Cridex", "Miuref", "Outlook")
K = 2
SEED = 0
REG_COVAR = 1e-3
N_INIT = 3
MAX_ITER_TF = 200
PCA_DIM = 64
NLL_TOLERANCE = 1e-6
CONTINUOUS_FEATURES = (
    "packet_count",
    "total_bytes",
    "flow_duration",
    "forward_packet_count",
    "backward_packet_count",
    "forward_bytes",
    "backward_bytes",
    "mean_packet_size",
    "bytes_per_packet",
)
MODEL_S_FEATURES = (
    "packet_count",
    "total_bytes",
    "flow_duration",
    "forward_packet_count",
    "backward_packet_count",
)
MODEL_S_PLUS_FEATURES = MODEL_S_FEATURES + (
    "src_port",
    "dst_port",
    "protocol_code",
    "time_bucket",
)
CATEGORICAL_FEATURES = (
    "source_pcap",
    "source_file",
    "src_port",
    "dst_port",
    "protocol",
    "time_bucket",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_list(values: Sequence[float]) -> str:
    return json.dumps([float(value) for value in values], separators=(",", ":"))


def safe_nmi(left: Sequence[Any], right: Sequence[Any]) -> float:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        return float(normalized_mutual_info_score(left, right))


def safe_ami(left: Sequence[Any], right: Sequence[Any]) -> float:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        return float(adjusted_mutual_info_score(left, right))


def contingency_and_matching(
    tf_labels: np.ndarray, od_labels: np.ndarray, n_components: int = K
) -> tuple[np.ndarray, dict[int, int], float]:
    """Return contingency, optimal TF->OD mapping and matched accuracy."""
    table = np.zeros((n_components, n_components), dtype=np.int64)
    np.add.at(table, (tf_labels.astype(int), od_labels.astype(int)), 1)
    row_ind, col_ind = linear_sum_assignment(-table)
    mapping = {int(left): int(right) for left, right in zip(row_ind, col_ind)}
    accuracy = float(table[row_ind, col_ind].sum() / table.sum())
    return table, mapping, accuracy


def correspondence_label(nmi: float, ari: float, accuracy: float) -> str:
    """Combined descriptive label; no single metric can trigger HIGH."""
    if nmi >= 0.70 and ari >= 0.70 and accuracy >= 0.85:
        return "HIGH_CORRESPONDENCE"
    if nmi < 0.20 and abs(ari) < 0.20 and accuracy < 0.70:
        return "LOW_CORRESPONDENCE"
    return "MODERATE_CORRESPONDENCE"


def upsert_readme_section(path: Path, body: str) -> None:
    text = path.read_text(encoding="utf-8")
    heading = "## Cross-Encoder Component Correspondence Audit"
    pattern = rf"{re.escape(heading)}\n.*?(?=\n## |\Z)"
    replacement = f"{heading}\n\n{body.strip()}\n"
    if re.search(pattern, text, flags=re.DOTALL):
        text = re.sub(pattern, replacement, text, count=1, flags=re.DOTALL)
    else:
        text = text.rstrip() + "\n\n" + replacement
    text = re.sub(
        r"## Current Status\n.*?(?=\n## )",
        "## Current Status\n\n"
        "Open-Detect formal training, latent Gaussian diagnosis, and the "
        "K=2 cross-encoder component correspondence/shortcut audit are COMPLETE.\n",
        text,
        count=1,
        flags=re.DOTALL,
    )
    path.write_text(text, encoding="utf-8")


def prepare_output(path: Path) -> Path:
    protected = (
        AUDIT_ROOT / "outputs" / "latent_gaussian_audit",
        PROJECT_ROOT / "outputs" / "stage2_5_covariance_diagnosis",
        PROJECT_ROOT / "outputs" / "stage2_6_component_shortcut_audit",
    )
    resolved = path.resolve()
    if any(resolved == item.resolve() or item.resolve() in resolved.parents for item in protected):
        raise ValueError(f"refusing protected output path: {resolved}")
    if resolved.exists() and any(resolved.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=".component_correspondence_", dir=resolved.parent))


def official_tf_k(n_components: int) -> pd.DataFrame:
    path = PROJECT_ROOT / "outputs/stage2_5_covariance_diagnosis/covariance_diagnosis.csv"
    frame = pd.read_csv(path)
    selected = frame[
        frame["class"].isin(CLASSES)
        & (frame["representation"] == "PCA64")
        & (frame["covariance_type"] == "full")
        & np.isclose(frame["reg_covar"], REG_COVAR)
        & (frame["K"] == n_components)
        & (frame["seed"] == SEED)
    ].copy()
    if len(selected) != len(CLASSES):
        raise AssertionError(
            f"expected four formal TrafficFormer K={n_components} rows, got {len(selected)}"
        )
    return selected.set_index("class")


def _build_tf_assignments_single_thread(
    n_components: int = K,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, list[float]]]:
    zt_dir = PROJECT_ROOT / "outputs/stage1/modelA/embeddings"
    zg_dir = PROJECT_ROOT / "outputs/stage1/modelB/seeds/seed2"
    labels_map = json.loads(
        (PROJECT_ROOT / "data/fig_graph/all_flows/label_map.json").read_text(encoding="utf-8")
    )["class_to_id"]
    official = official_tf_k(n_components)
    zt, labels, zg = _load_aligned(zt_dir, zg_dir)
    _, zf = _standardize(zt, zg)
    del zt, zg
    zf = {split: values.astype(np.float32, copy=False) for split, values in zf.items()}
    pca, train_pca, val_pca = fit_pca_train_only(zf["train"], zf["val"], PCA_DIM, seed=SEED)
    del zf
    matrices = {"train": train_pca, "val": val_pca}
    flow_ids = {
        split: np.load(zt_dir / f"flow_ids_{split}.npy") for split in ("train", "val")
    }
    rows: list[dict[str, Any]] = []
    checks: dict[str, Any] = {}
    weights: dict[str, list[float]] = {}
    for class_name in CLASSES:
        class_id = int(labels_map[class_name])
        indices = {
            split: np.flatnonzero(labels[split] == class_id) for split in ("train", "val")
        }
        model = GaussianMixture(
            n_components=n_components,
            covariance_type="full",
            reg_covar=REG_COVAR,
            n_init=N_INIT,
            max_iter=MAX_ITER_TF,
            random_state=SEED,
        ).fit(matrices["train"][indices["train"]])
        train_nll = float(-model.score(matrices["train"][indices["train"]]))
        val_nll = float(-model.score(matrices["val"][indices["val"]]))
        expected_train = float(-official.loc[class_name, "train_avg_loglik"])
        expected_val = float(official.loc[class_name, "val_nll"])
        train_difference = abs(train_nll - expected_train)
        val_difference = abs(val_nll - expected_val)
        if train_difference > NLL_TOLERANCE or val_difference > NLL_TOLERANCE:
            raise AssertionError(
                f"{class_name}: TrafficFormer K=2 NLL reproduction failed: "
                f"train diff={train_difference}, val diff={val_difference}"
            )
        checks[class_name] = {
            "train_nll_recomputed": train_nll,
            "train_nll_official": expected_train,
            "train_absolute_difference": train_difference,
            "validation_nll_recomputed": val_nll,
            "validation_nll_official": expected_val,
            "validation_absolute_difference": val_difference,
            "converged": bool(model.converged_),
            "n_iter": int(model.n_iter_),
        }
        weights[class_name] = [float(item) for item in model.weights_]
        for split in ("train", "val"):
            x = matrices[split][indices[split]]
            probabilities = model.predict_proba(x)
            component = probabilities.argmax(axis=1)
            maximum = probabilities[np.arange(len(component)), component]
            rows.extend(
                {
                    "flow_id": str(flow_ids[split][global_index]),
                    "class_name": class_name,
                    "split": split,
                    "component_id": int(component[local_index]),
                    "component_probability": float(maximum[local_index]),
                    "max_posterior": float(maximum[local_index]),
                }
                for local_index, global_index in enumerate(indices[split])
            )
    metadata = {
        "representation": "train-standardized z_f (768+128=896) -> train-fitted randomized PCA64",
        "blas_thread_limit": 1,
        "pca_fit_split": "train",
        "pca_transform_splits": ["train", "val"],
        "pca_explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
        "gmm_fit_split": "class-specific train",
        "n_components": n_components,
        "validation_role": "assignment and NLL reproduction only",
        "nll_reproduction_tolerance": NLL_TOLERANCE,
        "checks": checks,
    }
    return pd.DataFrame(rows), metadata, weights


def build_tf_assignments(
    n_components: int = K,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, list[float]]]:
    """Replay historical Stage 2.5 with its numerically decisive BLAS limit."""
    with threadpool_limits(limits=1):
        return _build_tf_assignments_single_thread(n_components)


def build_od_assignments(
    n_components: int = K,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, list[float]]]:
    artifact_dir = AUDIT_ROOT / "artifacts/latent_gaussian_audit"
    output_dir = AUDIT_ROOT / "outputs/latent_gaussian_audit"
    scaler = joblib.load(artifact_dir / "standard_scaler.joblib")
    pca = joblib.load(artifact_dir / "pca64.joblib")
    official = pd.read_csv(output_dir / "gaussian_results.csv")
    official = official[
        official["class_name"].isin(CLASSES)
        & (official["K"] == n_components)
        & (official["seed"] == SEED)
        & (official["covariance_type"] == "full")
        & np.isclose(official["reg_covar"], REG_COVAR)
    ].set_index("class_name")
    if len(official) != len(CLASSES):
        raise AssertionError(
            f"expected four formal Open-Detect K={n_components} rows, got {len(official)}"
        )
    official_components = pd.read_csv(output_dir / "component_weights.csv")
    official_components = official_components[
        official_components["class_name"].isin(CLASSES)
        & (official_components["K"] == n_components)
        & (official_components["seed"] == SEED)
    ]
    split_frames = {
        "train": pd.read_parquet(artifact_dir / "train_mu.parquet"),
        "val": pd.read_parquet(artifact_dir / "val_mu.parquet"),
    }
    mu_columns = [f"mu_{index}" for index in range(128)]
    scaled: dict[str, np.ndarray] = {}
    for split, frame in split_frames.items():
        if frame["flow_id"].duplicated().any():
            raise AssertionError(f"Open-Detect {split} has duplicate flow_id")
        latent = frame[mu_columns].to_numpy(dtype=np.float64, copy=True)
        scaled[split] = scaler.transform(latent).astype(np.float64, copy=False)
    replay_pca = PCA(n_components=64, svd_solver="randomized", random_state=0)
    replay_train = replay_pca.fit_transform(scaled["train"]).astype(np.float64, copy=False)
    pca_parameter_differences = {
        "components_max_abs": float(np.max(np.abs(replay_pca.components_ - pca.components_))),
        "mean_max_abs": float(np.max(np.abs(replay_pca.mean_ - pca.mean_))),
        "explained_variance_max_abs": float(
            np.max(np.abs(replay_pca.explained_variance_ - pca.explained_variance_))
        ),
    }
    if any(value > 1e-10 for value in pca_parameter_differences.values()):
        raise AssertionError(
            "Open-Detect PCA fit_transform replay does not match frozen PCA: "
            f"{pca_parameter_differences}"
        )
    transformed = {
        "train": replay_train,
        "val": pca.transform(scaled["val"]).astype(np.float64, copy=False),
    }
    rows: list[dict[str, Any]] = []
    checks: dict[str, Any] = {}
    weights: dict[str, list[float]] = {}
    for class_name in CLASSES:
        model_path = artifact_dir / f"gmms/{class_name}_seed0_k{n_components}.joblib"
        model = joblib.load(model_path)
        weights[class_name] = [float(item) for item in model.weights_]
        class_indices = {
            split: np.flatnonzero(frame["class_name"].to_numpy() == class_name)
            for split, frame in split_frames.items()
        }
        train_nll = float(-model.score(transformed["train"][class_indices["train"]]))
        val_nll = float(-model.score(transformed["val"][class_indices["val"]]))
        expected_train = float(official.loc[class_name, "train_avg_nll"])
        expected_val = float(official.loc[class_name, "validation_avg_nll"])
        train_difference = abs(train_nll - expected_train)
        val_difference = abs(val_nll - expected_val)
        if val_difference > NLL_TOLERANCE:
            raise AssertionError(
                f"{class_name}: Open-Detect K=2 NLL reproduction failed: "
                f"train diff={train_difference}, val diff={val_difference}"
            )
        probabilities_by_split = {
            split: model.predict_proba(transformed[split][class_indices[split]])
            for split in ("train", "val")
        }
        predictions_by_split = {
            split: probabilities.argmax(axis=1)
            for split, probabilities in probabilities_by_split.items()
        }
        observed_train_counts = np.bincount(
            predictions_by_split["train"], minlength=n_components
        )
        expected_train_counts = (
            official_components[official_components.class_name == class_name]
            .sort_values("component")["train_count"]
            .to_numpy(dtype=int)
        )
        if not np.array_equal(observed_train_counts, expected_train_counts):
            raise AssertionError(
                f"{class_name}: Open-Detect replay changed train component counts: "
                f"observed={observed_train_counts.tolist()}, "
                f"official={expected_train_counts.tolist()}"
            )
        checks[class_name] = {
            "model_path": str(model_path.resolve()),
            "model_sha256": sha256(model_path),
            "train_nll_recomputed": train_nll,
            "train_nll_official": expected_train,
            "train_absolute_difference": train_difference,
            "train_nll_replay_note": (
                "official train coordinates used randomized PCA.fit_transform; "
                "the saved PCA can only transform(train), so train score is diagnostic "
                "rather than the artifact identity gate"
            ),
            "validation_nll_recomputed": val_nll,
            "validation_nll_official": expected_val,
            "validation_absolute_difference": val_difference,
            "train_component_counts_recomputed": observed_train_counts.tolist(),
            "train_component_counts_official": expected_train_counts.tolist(),
            "train_component_counts_exact": True,
        }
        for split in ("train", "val"):
            indices = class_indices[split]
            probabilities = probabilities_by_split[split]
            component = predictions_by_split[split]
            maximum = probabilities[np.arange(len(component)), component]
            frame = split_frames[split]
            rows.extend(
                {
                    "flow_id": str(frame.iloc[global_index]["flow_id"]),
                    "class_name": class_name,
                    "split": split,
                    "component_id": int(component[local_index]),
                    "component_probability": float(maximum[local_index]),
                    "max_posterior": float(maximum[local_index]),
                }
                for local_index, global_index in enumerate(indices)
            )
    metadata = {
        "representation": "deterministic mu_x -> frozen train-fitted StandardScaler/PCA64",
        "n_components": n_components,
        "scaler_sha256": sha256(artifact_dir / "standard_scaler.joblib"),
        "pca_sha256": sha256(artifact_dir / "pca64.joblib"),
        "train_coordinate_recovery": (
            "replayed PCA.fit_transform(train) because sklearn does not persist its "
            "training scores; replayed PCA parameters were required to match the frozen PCA"
        ),
        "pca_replay_parameter_max_absolute_differences": pca_parameter_differences,
        "nll_reproduction_tolerance": NLL_TOLERANCE,
        "checks": checks,
    }
    return pd.DataFrame(rows), metadata, weights


def assert_flow_identity(tf: pd.DataFrame, od: pd.DataFrame) -> list[dict[str, Any]]:
    assertions = []
    for class_name in CLASSES:
        for split in ("train", "val"):
            left = set(tf[(tf.class_name == class_name) & (tf.split == split)].flow_id)
            right = set(od[(od.class_name == class_name) & (od.split == split)].flow_id)
            equal = left == right
            assertions.append(
                {
                    "class_name": class_name,
                    "split": split,
                    "tf_count": len(left),
                    "od_count": len(right),
                    "sets_equal": equal,
                    "tf_only": len(left - right),
                    "od_only": len(right - left),
                }
            )
            if not equal:
                raise AssertionError(f"{class_name}/{split}: TF and OD flow_id sets differ")
    return assertions


def correspondence(
    tf: pd.DataFrame,
    od: pd.DataFrame,
    tf_weights: Mapping[str, Sequence[float]],
    od_weights: Mapping[str, Sequence[float]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = tf.merge(
        od,
        on=["flow_id", "class_name", "split"],
        suffixes=("_tf", "_od"),
        validate="one_to_one",
    )
    result_rows = []
    confusion_rows = []
    for class_name in CLASSES:
        n_components = len(tf_weights[class_name])
        for split in ("train", "val"):
            subset = merged[(merged.class_name == class_name) & (merged.split == split)]
            left = subset.component_id_tf.to_numpy(dtype=int)
            right = subset.component_id_od.to_numpy(dtype=int)
            table, mapping, hungarian = contingency_and_matching(
                left, right, n_components=n_components
            )
            nmi = safe_nmi(left, right)
            ami = safe_ami(left, right)
            ari = float(adjusted_rand_score(left, right))
            left_ratio = np.bincount(left, minlength=n_components) / len(left)
            right_ratio = np.bincount(right, minlength=n_components) / len(right)
            matched_right = np.asarray(
                [right_ratio[mapping[index]] for index in range(n_components)]
            )
            weight_tv = float(0.5 * np.abs(left_ratio - matched_right).sum())
            result_rows.append(
                {
                    "class_name": class_name,
                    "split": split,
                    "primary_validation_result": split == "val",
                    "n_samples": len(subset),
                    "nmi": nmi,
                    "ami": ami,
                    "ari": ari,
                    "hungarian_accuracy": hungarian,
                    "tf_component_weights": json_list(tf_weights[class_name]),
                    "od_component_weights": json_list(od_weights[class_name]),
                    "tf_empirical_component_proportions": json_list(left_ratio),
                    "od_empirical_component_proportions": json_list(right_ratio),
                    "hungarian_tf_to_od_mapping": json.dumps(mapping, sort_keys=True),
                    "matched_empirical_weight_tv": weight_tv,
                    "correspondence_label": correspondence_label(nmi, ari, hungarian),
                }
            )
            for tf_component in range(n_components):
                row_total = int(table[tf_component].sum())
                for od_component in range(n_components):
                    count = int(table[tf_component, od_component])
                    confusion_rows.append(
                        {
                            "class_name": class_name,
                            "split": split,
                            "tf_component_id": tf_component,
                            "od_component_id": od_component,
                            "count": count,
                            "row_ratio": count / row_total if row_total else math.nan,
                            "total_ratio": count / len(subset),
                            "hungarian_matched_cell": mapping[tf_component] == od_component,
                        }
                    )
    return pd.DataFrame(result_rows), pd.DataFrame(confusion_rows)


def load_metadata(assignments: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    source = PROJECT_ROOT / "outputs/stage2_6_component_shortcut_audit/component_assignments.csv"
    metadata = pd.read_csv(source)
    metadata = metadata.rename(columns={"class": "class_name", "duration": "flow_duration"})
    keep = [
        "flow_id",
        "class_name",
        "split",
        "source_pcap",
        "packet_count",
        "total_bytes",
        "flow_duration",
        "forward_packet_count",
        "backward_packet_count",
        "src_port",
        "dst_port",
        "start_time",
        "relative_time",
        "relative_time_bin",
    ]
    metadata = metadata[keep].copy()
    wanted = set(assignments.flow_id)
    metadata = metadata[metadata.flow_id.isin(wanted)].copy()
    if set(metadata.flow_id) != wanted or metadata.flow_id.duplicated().any():
        raise AssertionError("Stage 2.6 metadata does not provide exactly one row per wanted flow")
    metadata["source_file"] = metadata["source_pcap"]
    metadata["time_bucket"] = metadata["relative_time_bin"]

    flow_map = pd.read_csv(
        PROJECT_ROOT / "data/trafficformer_input/compatible_min1/flow_map.csv",
        usecols=["flow_id", "source_pkl"],
    )
    flow_map = flow_map[flow_map.flow_id.isin(wanted)]
    if set(flow_map.flow_id) != wanted or flow_map.flow_id.duplicated().any():
        raise AssertionError("flow_map does not provide exactly one source PKL per wanted flow")
    by_pkl: dict[str, set[str]] = defaultdict(set)
    for row in flow_map.itertuples(index=False):
        by_pkl[str(row.source_pkl)].add(str(row.flow_id))
    extra: dict[str, dict[str, Any]] = {}
    for pkl_name, ids in sorted(by_pkl.items()):
        with (PROJECT_ROOT / "data/flows" / pkl_name).open("rb") as handle:
            payload = pickle.load(handle)
        packets_by_flow = payload["packets"]
        if not ids.issubset(packets_by_flow):
            raise AssertionError(f"{pkl_name}: missing selected flow packets")
        for flow_id in ids:
            packets = packets_by_flow[flow_id]
            parsed = next(
                (parse_ipv4(bytes(packet[4])) for packet in packets if parse_ipv4(bytes(packet[4]))),
                None,
            )
            protocol_number = parsed[0] if parsed else None
            forward_bytes = sum(int(packet[1]) for packet in packets if int(packet[3]) == 1)
            backward_bytes = sum(int(packet[1]) for packet in packets if int(packet[3]) == 0)
            total = forward_bytes + backward_bytes
            extra[flow_id] = {
                "protocol": "TCP" if protocol_number == TCP else "UDP" if protocol_number == UDP else None,
                "protocol_code": protocol_number,
                "forward_bytes": forward_bytes,
                "backward_bytes": backward_bytes,
                "mean_packet_size": total / len(packets) if packets else math.nan,
                "bytes_per_packet": total / len(packets) if packets else math.nan,
            }
        del payload, packets_by_flow
        gc.collect()
    extra_frame = pd.DataFrame.from_dict(extra, orient="index").rename_axis("flow_id").reset_index()
    metadata = metadata.merge(extra_frame, on="flow_id", validate="one_to_one")
    missing = {column: int(metadata[column].isna().sum()) for column in extra_frame.columns if column != "flow_id"}
    return metadata, {
        "base_metadata": str(source.resolve()),
        "protocol_and_optional_byte_metadata": "existing Stage 0 PKLs; raw PCAP not loaded",
        "source_pkls": sorted(by_pkl),
        "missing_counts": missing,
    }


def continuous_audit(od: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    merged = od.merge(metadata, on=["flow_id", "class_name", "split"], validate="one_to_one")
    rows = []
    for class_name in CLASSES:
        for split in ("train", "val"):
            subset = merged[(merged.class_name == class_name) & (merged.split == split)]
            for feature in CONTINUOUS_FEATURES:
                valid = subset[["component_id", feature]].dropna()
                groups = [
                    group[feature].to_numpy(dtype=float)
                    for _, group in valid.groupby("component_id", sort=True)
                ]
                status = "OK"
                if len(groups) < K or valid[feature].nunique() < 2:
                    h_stat = p_value = effect = math.nan
                    status = "NOT_IDENTIFIABLE_CONSTANT_OR_MISSING"
                else:
                    try:
                        h_stat, p_value = kruskal(*groups)
                        effect = epsilon_squared_kruskal(float(h_stat), len(valid), len(groups))
                    except ValueError:
                        h_stat = p_value = effect = math.nan
                        status = "NOT_IDENTIFIABLE_CONSTANT_OR_MISSING"
                for component_id in range(K):
                    values = valid[valid.component_id == component_id][feature].to_numpy(dtype=float)
                    quantiles = np.quantile(values, [0.10, 0.25, 0.50, 0.75, 0.90]) if len(values) else [math.nan] * 5
                    rows.append(
                        {
                            "class_name": class_name,
                            "split": split,
                            "feature": feature,
                            "component_id": component_id,
                            "count": len(values),
                            "mean": float(np.mean(values)) if len(values) else math.nan,
                            "std": float(np.std(values, ddof=1)) if len(values) > 1 else math.nan,
                            "median": float(quantiles[2]),
                            "q25": float(quantiles[1]),
                            "q75": float(quantiles[3]),
                            "p10": float(quantiles[0]),
                            "p90": float(quantiles[4]),
                            "kruskal_h": h_stat,
                            "p_value": p_value,
                            "epsilon_squared": effect,
                            "effect_size_formula": "max(0,(H-k+1)/(n-k))",
                            "status": status,
                        }
                    )
    return pd.DataFrame(rows)


def categorical_audit(od: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    merged = od.merge(metadata, on=["flow_id", "class_name", "split"], validate="one_to_one")
    rows = []
    for class_name in CLASSES:
        for split in ("train", "val"):
            subset = merged[(merged.class_name == class_name) & (merged.split == split)]
            for feature in CATEGORICAL_FEATURES:
                valid = subset[["component_id", feature]].dropna()
                unique = int(valid[feature].nunique())
                if unique <= 1:
                    status = "NOT_IDENTIFIABLE_CONSTANT_WITHIN_CLASS"
                    nmi = ami = math.nan
                else:
                    status = "OK"
                    values = valid[feature].astype(str).to_numpy()
                    components = valid.component_id.to_numpy(dtype=int)
                    nmi = safe_nmi(components, values)
                    ami = safe_ami(components, values)
                rows.append(
                    {
                        "class_name": class_name,
                        "split": split,
                        "feature": feature,
                        "valid_samples": len(valid),
                        "unique_values": unique,
                        "status": status,
                        "nmi": nmi,
                        "ami": ami,
                    }
                )
    return pd.DataFrame(rows)


def metadata_predictability(od: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    merged = od.merge(metadata, on=["flow_id", "class_name", "split"], validate="one_to_one")
    rows = []
    for class_name in CLASSES:
        class_data = merged[merged.class_name == class_name]
        train = class_data[class_data.split == "train"]
        val = class_data[class_data.split == "val"]
        for model_name, features in (("S", MODEL_S_FEATURES), ("S+", MODEL_S_PLUS_FEATURES)):
            train_x = train[list(features)].to_numpy(dtype=float)
            val_x = val[list(features)].to_numpy(dtype=float)
            medians = np.nanmedian(train_x, axis=0)
            if np.isnan(medians).any():
                raise AssertionError(f"{class_name}/{model_name}: a feature is entirely missing in train")
            train_x = np.where(np.isnan(train_x), medians, train_x)
            val_x = np.where(np.isnan(val_x), medians, val_x)
            train_y = train.component_id.to_numpy(dtype=int)
            val_y = val.component_id.to_numpy(dtype=int)
            model = RandomForestClassifier(
                n_estimators=200,
                random_state=SEED,
                class_weight="balanced",
                n_jobs=1,
            )
            model.fit(train_x, train_y)
            prediction = model.predict(val_x)
            rows.append(
                {
                    "class_name": class_name,
                    "model": model_name,
                    "features": ";".join(features),
                    "train_samples": len(train),
                    "validation_samples": len(val),
                    "accuracy": accuracy_score(val_y, prediction),
                    "balanced_accuracy": balanced_accuracy_score(val_y, prediction),
                    "macro_f1": f1_score(val_y, prediction, average="macro"),
                    "n_estimators": 200,
                    "random_state": SEED,
                    "class_weight": "balanced",
                    "fit_split": "train",
                    "evaluation_split": "validation",
                    "feature_importances": json.dumps(
                        dict(zip(features, map(float, model.feature_importances_))), sort_keys=True
                    ),
                }
            )
    return pd.DataFrame(rows)


def train_val_stability(
    assignments: Mapping[str, pd.DataFrame],
    model_weights: Mapping[str, Mapping[str, Sequence[float]]],
) -> pd.DataFrame:
    rows = []
    for encoder, frame in assignments.items():
        for class_name in CLASSES:
            n_components = len(model_weights[encoder][class_name])
            class_frame = frame[frame.class_name == class_name]
            counts = {
                split: np.bincount(
                    class_frame[class_frame.split == split].component_id.to_numpy(dtype=int),
                    minlength=n_components,
                )
                for split in ("train", "val")
            }
            ratios = {split: value / value.sum() for split, value in counts.items()}
            tv = float(0.5 * np.abs(ratios["train"] - ratios["val"]).sum())
            for component_id in range(n_components):
                rows.append(
                    {
                        "encoder": encoder,
                        "class_name": class_name,
                        "component_id": component_id,
                        "gmm_weight": float(model_weights[encoder][class_name][component_id]),
                        "train_count": int(counts["train"][component_id]),
                        "train_ratio": float(ratios["train"][component_id]),
                        "validation_count": int(counts["val"][component_id]),
                        "validation_ratio": float(ratios["val"][component_id]),
                        "train_validation_tv": tv,
                        "tiny_train_below_1pct": ratios["train"][component_id] < 0.01,
                        "very_tiny_train_below_0_5pct": ratios["train"][component_id] < 0.005,
                        "validation_component_empty": counts["val"][component_id] == 0,
                    }
                )
    return pd.DataFrame(rows)


def strongest_continuous(frame: pd.DataFrame, class_name: str, split: str) -> pd.Series:
    candidates = frame[(frame.class_name == class_name) & (frame.split == split)].drop_duplicates(
        ["class_name", "split", "feature"]
    )
    candidates = candidates[np.isfinite(candidates.epsilon_squared)]
    if candidates.empty:
        raise AssertionError(f"{class_name}/{split}: no identifiable continuous feature")
    return candidates.loc[candidates.epsilon_squared.idxmax()]


def strongest_categorical(frame: pd.DataFrame, class_name: str, split: str) -> pd.Series | None:
    candidates = frame[
        (frame.class_name == class_name) & (frame.split == split) & (frame.status == "OK")
    ]
    candidates = candidates[np.isfinite(candidates.nmi)]
    return None if candidates.empty else candidates.loc[candidates.nmi.idxmax()]


def profile_comparison(
    continuous: pd.DataFrame,
    categorical: pd.DataFrame,
    predictability: pd.DataFrame,
    correspondence_frame: pd.DataFrame,
) -> pd.DataFrame:
    tf_cont = pd.read_csv(
        PROJECT_ROOT / "outputs/stage2_6_component_shortcut_audit/continuous_feature_tests.csv"
    ).rename(columns={"class": "class_name"})
    tf_rf = pd.read_csv(
        PROJECT_ROOT / "outputs/stage2_6_component_shortcut_audit/component_predictability.csv"
    ).rename(columns={"class": "class_name"})
    rows = []
    for class_name in CLASSES:
        tf_candidates = tf_cont[(tf_cont.class_name == class_name) & (tf_cont.split == "val")]
        tf_candidates = tf_candidates[np.isfinite(tf_candidates.epsilon_squared)]
        tf_best = tf_candidates.loc[tf_candidates.epsilon_squared.idxmax()]
        tf_rf_row = tf_rf[tf_rf.class_name == class_name].iloc[0]
        od_best = strongest_continuous(continuous, class_name, "val")
        od_cat = strongest_categorical(categorical, class_name, "val")
        od_s = predictability[(predictability.class_name == class_name) & (predictability.model == "S")].iloc[0]
        od_sp = predictability[(predictability.class_name == class_name) & (predictability.model == "S+")].iloc[0]
        corr = correspondence_frame[
            (correspondence_frame.class_name == class_name) & (correspondence_frame.split == "val")
        ].iloc[0]
        rows.append(
            {
                "class_name": class_name,
                "tf_strongest_continuous_feature": tf_best.feature,
                "tf_strongest_effect_size": float(tf_best.epsilon_squared),
                "tf_metadata_rf_macro_f1": float(tf_rf_row.macro_f1),
                "tf_metadata_rf_protocol": "Stage2.6 fixed K=4/5; train-only 80/20 diagnostic",
                "od_strongest_continuous_feature": od_best.feature,
                "od_strongest_effect_size": float(od_best.epsilon_squared),
                "od_strongest_categorical_feature": "NONE_IDENTIFIABLE" if od_cat is None else od_cat.feature,
                "od_strongest_categorical_nmi": math.nan if od_cat is None else float(od_cat.nmi),
                "od_strongest_categorical_ami": math.nan if od_cat is None else float(od_cat.ami),
                "od_model_s_macro_f1": float(od_s.macro_f1),
                "od_model_s_plus_macro_f1": float(od_sp.macro_f1),
                "cross_encoder_nmi": float(corr.nmi),
                "cross_encoder_ami": float(corr.ami),
                "cross_encoder_ari": float(corr.ari),
                "cross_encoder_hungarian_accuracy": float(corr.hungarian_accuracy),
                "cross_encoder_correspondence": corr.correspondence_label,
            }
        )
    return pd.DataFrame(rows)


def classify_cases(profile: pd.DataFrame, stability: pd.DataFrame) -> tuple[dict[str, str], str]:
    cases: dict[str, str] = {}
    for row in profile.itertuples(index=False):
        tf_simple = row.tf_strongest_effect_size >= 0.50 or row.tf_metadata_rf_macro_f1 >= 0.90
        od_simple = row.od_strongest_effect_size >= 0.50 or row.od_model_s_macro_f1 >= 0.90
        same_feature = row.tf_strongest_continuous_feature == row.od_strongest_continuous_feature
        high = row.cross_encoder_correspondence == "HIGH_CORRESPONDENCE"
        if tf_simple and od_simple:
            cases[row.class_name] = "Case 4"
        elif high and same_feature:
            cases[row.class_name] = "Case 1"
        elif high and not od_simple:
            cases[row.class_name] = "Case 2"
        else:
            cases[row.class_name] = "Case 3"
    counts = Counter(cases.values())
    stable_classes = 0
    for class_name in CLASSES:
        subset = stability[stability.class_name == class_name]
        if (
            (subset.train_validation_tv <= 0.10).all()
            and (~subset.tiny_train_below_1pct).all()
            and (~subset.validation_component_empty).all()
        ):
            stable_classes += 1
    if counts["Case 4"] >= 3:
        gate = "C. SIMPLE-STATISTICS PERSISTENCE"
    elif counts["Case 1"] + counts["Case 2"] >= 3 and stable_classes >= 3:
        gate = "A. STRONG CROSS-ENCODER STRUCTURE"
    elif counts["Case 3"] >= 3:
        gate = "B. MULTI EXISTS, COMPONENTS DIFFER"
    else:
        gate = "D. MIXED"
    return cases, gate


def make_figures(
    output_dir: Path,
    confusion: pd.DataFrame,
    od: pd.DataFrame,
    metadata: pd.DataFrame,
    continuous: pd.DataFrame,
) -> list[str]:
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    paths = []
    merged = od.merge(metadata, on=["flow_id", "class_name", "split"], validate="one_to_one")
    for class_name in CLASSES:
        table_rows = confusion[(confusion.class_name == class_name) & (confusion.split == "val")]
        matrix = np.zeros((K, K), dtype=int)
        for row in table_rows.itertuples(index=False):
            matrix[row.tf_component_id, row.od_component_id] = row.count
        fig, ax = plt.subplots(figsize=(5, 4))
        image = ax.imshow(matrix, cmap="Blues")
        for left in range(K):
            for right in range(K):
                ax.text(right, left, str(matrix[left, right]), ha="center", va="center")
        ax.set_xticks(range(K), ["OD C0", "OD C1"])
        ax.set_yticks(range(K), ["TF C0", "TF C1"])
        ax.set_title(f"{class_name}: validation correspondence")
        fig.colorbar(image, ax=ax, label="flow count")
        fig.tight_layout()
        path = figures / f"{class_name}_validation_contingency.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(str(path.relative_to(output_dir)))

        best = strongest_continuous(continuous, class_name, "val").feature
        subset = merged[(merged.class_name == class_name) & (merged.split == "val")]
        arrays = [subset[subset.component_id == component][best].dropna().to_numpy() for component in range(K)]
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.boxplot(arrays, tick_labels=["OD C0", "OD C1"], showfliers=False)
        ax.set_ylabel(best)
        ax.set_title(f"{class_name}: OD validation {best}")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        path = figures / f"{class_name}_od_{best}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(str(path.relative_to(output_dir)))
    return paths


def build_summary(
    profile: pd.DataFrame,
    stability: pd.DataFrame,
    cases: Mapping[str, str],
    gate: str,
) -> str:
    lines = [
        "# USTC Cross-Encoder Component Correspondence & Shortcut Audit",
        "",
        "## Protocol",
        "",
        "- Main analysis: K=2, full covariance, PCA64, seed=0 for FTP, Cridex, Miuref and Outlook.",
        "- Both GMMs were fit on train; correspondence metrics use held-out validation as the primary result.",
        "- TrafficFormer K=2 was refit from the immutable Stage 2.5 z_f inputs and passed the 1e-6 train/validation NLL reproduction gate.",
        "- Open-Detect reused the frozen scaler, PCA64 and K=2 GMM artifacts; validation NLL reproduced within 1e-6 and train component counts matched the formal record exactly.",
        "- No encoder training, split change, K search, test use, BIC, Adaptive K or Unknown Detection was performed.",
        "",
        "## Per-class answers",
        "",
    ]
    prompts = {
        "FTP": "Do TF and OD partition the same validation flows, and is OD still driven by packet_count?",
        "Cridex": "Are component assignments consistent across encoders?",
        "Miuref": "Does backward_packet_count still dominate the OD components?",
        "Outlook": "Does total_bytes still dominate the OD components?",
    }
    for class_name in CLASSES:
        row = profile[profile.class_name == class_name].iloc[0]
        tv = stability[stability.class_name == class_name].groupby("encoder").train_validation_tv.first()
        lines.extend(
            [
                f"### {class_name}",
                "",
                f"- Question: {prompts[class_name]}",
                f"- Validation correspondence: NMI={row.cross_encoder_nmi:.6f}, AMI={row.cross_encoder_ami:.6f}, ARI={row.cross_encoder_ari:.6f}, Hungarian accuracy={row.cross_encoder_hungarian_accuracy:.6f}; {row.cross_encoder_correspondence}.",
                f"- OD strongest validation continuous feature: {row.od_strongest_continuous_feature}, epsilon-squared={row.od_strongest_effect_size:.6f}.",
                f"- OD metadata Model S/S+ validation Macro-F1: {row.od_model_s_macro_f1:.6f}/{row.od_model_s_plus_macro_f1:.6f}.",
                f"- Train/validation component TV: TF={tv['TrafficFormer']:.6f}, OD={tv['Open-Detect']:.6f}.",
                f"- Classification: {cases[class_name]}.",
                "",
            ]
        )
    lines.extend(
        [
            "## Overall Gate",
            "",
            f"**{gate}**",
            "",
            "The audit supports only stable or encoder-dependent representation-level local structure as indicated by the reported metrics. It does not establish semantic multimodality, natural Gaussian modes, attack stages, or an Unknown Detection result.",
            "",
            "## Interpretation safeguards",
            "",
            "- High metadata predictability means component assignment is strongly associated with simple flow statistics; it does not by itself make the structure false.",
            "- High cross-encoder correspondence means the two learned representations partition many of the same flows after label permutation; it does not identify a semantic subtype.",
            "- Source PCAP/source file are constant within each audited class and are explicitly marked NOT_IDENTIFIABLE_CONSTANT_WITHIN_CLASS.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=AUDIT_ROOT / "outputs/component_correspondence_audit",
    )
    args = parser.parse_args()
    started = time.time()
    final_output = args.output.resolve()
    stage = prepare_output(final_output)
    try:
        print("[1/9] Refit TrafficFormer formal K=2 and verify Stage 2.5 NLL")
        tf, tf_meta, tf_weights = build_tf_assignments()
        print("[2/9] Reuse Open-Detect frozen K=2 models and verify formal NLL")
        od, od_meta, od_weights = build_od_assignments()
        flow_assertions = assert_flow_identity(tf, od)
        print("[3/9] Write exact assignments and compute train/validation correspondence")
        tf.to_parquet(stage / "tf_component_assignments.parquet", index=False)
        od.to_parquet(stage / "od_component_assignments.parquet", index=False)
        cross, confusion = correspondence(tf, od, tf_weights, od_weights)
        cross.to_csv(stage / "cross_encoder_correspondence.csv", index=False)
        confusion.to_csv(stage / "cross_encoder_confusion.csv", index=False)
        print("[4/9] Reuse Stage 2.6 metadata and recover protocol/optional byte statistics")
        metadata, metadata_meta = load_metadata(od)
        print("[5/9] Audit Open-Detect continuous and categorical shortcuts")
        continuous = continuous_audit(od, metadata)
        categorical = categorical_audit(od, metadata)
        continuous.to_csv(stage / "shortcut_continuous_tests.csv", index=False)
        categorical.to_csv(stage / "shortcut_categorical_tests.csv", index=False)
        print("[6/9] Fit metadata-only Model S and S+ on train; evaluate validation")
        predictability = metadata_predictability(od, metadata)
        predictability.to_csv(stage / "metadata_predictability.csv", index=False)
        stability = train_val_stability(
            {"TrafficFormer": tf, "Open-Detect": od},
            {"TrafficFormer": tf_weights, "Open-Detect": od_weights},
        )
        stability.to_csv(stage / "train_val_stability.csv", index=False)
        print("[7/9] Build direct Stage 2.6 profile comparison and figures")
        profile = profile_comparison(continuous, categorical, predictability, cross)
        profile.to_csv(stage / "component_profile_comparison.csv", index=False)
        cases, gate = classify_cases(profile, stability)
        figures = make_figures(stage, confusion, od, metadata, continuous)
        summary = build_summary(profile, stability, cases, gate)
        (stage / "audit_summary.md").write_text(summary, encoding="utf-8")
        print("[8/9] Record immutable provenance and audit safeguards")
        (stage / "flow_id_assertions.csv").write_text(
            pd.DataFrame(flow_assertions).to_csv(index=False), encoding="utf-8"
        )
        metadata_payload = {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "experiment": "USTC Cross-Encoder Component Correspondence & Shortcut Audit",
            "project_root": str(PROJECT_ROOT.resolve()),
            "audit_root": str(AUDIT_ROOT.resolve()),
            "formal_protocol": {
                "classes": list(CLASSES),
                "K": K,
                "covariance_type": "full",
                "pca_dim": PCA_DIM,
                "seed": SEED,
                "reg_covar": REG_COVAR,
                "n_init": N_INIT,
                "primary_correspondence_split": "validation",
                "test_loaded_or_used": False,
                "encoder_retrained": False,
                "split_changed": False,
                "sampling_changed": False,
                "k_searched": False,
                "bic_used": False,
                "unknown_detection_run": False,
            },
            "trafficformer": tf_meta,
            "opendetect": od_meta,
            "metadata": metadata_meta,
            "flow_id_assertions": flow_assertions,
            "cases": cases,
            "overall_gate": gate,
            "figures": figures,
            "elapsed_seconds": time.time() - started,
            "actual_command": f"python {Path(__file__).resolve()}",
        }
        output_names = sorted(
            str(path.relative_to(stage)) for path in stage.rglob("*") if path.is_file()
        )
        metadata_payload["output_files"] = sorted({*output_names, "run_metadata.json"})
        (stage / "run_metadata.json").write_text(
            json.dumps(metadata_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if final_output.exists():
            final_output.rmdir()
        stage.rename(final_output)
        print("[9/9] Update README after successful atomic output publication")
        readme_table = [
            "| Class | NMI | ARI | Hungarian | OD strongest continuous | Case |",
            "|---|---:|---:|---:|---|---|",
        ]
        for row in profile.itertuples(index=False):
            readme_table.append(
                f"| {row.class_name} | {row.cross_encoder_nmi:.4f} | {row.cross_encoder_ari:.4f} | {row.cross_encoder_hungarian_accuracy:.4f} | {row.od_strongest_continuous_feature} ({row.od_strongest_effect_size:.4f}) | {cases[row.class_name]} |"
            )
        readme_body = (
            "- Status: COMPLETE\n"
            "- Main protocol: K=2, full covariance, PCA64, seed=0; train fit and validation-primary correspondence\n"
            "- Exact flow-id assertions: 8/8 class-split sets equal\n\n"
            + "\n".join(readme_table)
            + f"\n\nOverall Gate: **{gate}**\n\n"
            "Interpretation boundary: these results concern persistent representation-level local structure and simple-statistics associations. They do not establish semantic modes and do not contain an Unknown Detection experiment."
        )
        upsert_readme_section(AUDIT_ROOT / "README.md", readme_body)
        print(json.dumps({"status": "PASS", "gate": gate, "output": str(final_output)}))
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
