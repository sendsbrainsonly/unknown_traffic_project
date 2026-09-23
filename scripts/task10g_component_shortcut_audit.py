# -*- coding: utf-8 -*-
"""Stage 2.6: audit whether fixed residual GMM components track shortcuts.

The protocol is intentionally narrower than Stage 2.5.  It reuses the same
train-only z_f standardization and train-fitted PCA-64 representation, then
fits one fixed full-covariance GMM per representative class.  K is supplied by
the completed Stage 2.5 diagnosis and is never searched or reselected here.

Only train/validation embeddings are accepted.  Existing Stage 0 flow PKLs are
used as metadata sources; raw PCAPs are never reparsed.  Validation is used
only for component assignment and consistency diagnostics.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
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
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import kruskal
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    adjusted_mutual_info_score,
    balanced_accuracy_score,
    f1_score,
    normalized_mutual_info_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from task10_stage2_audit import _load_aligned, _standardize  # noqa: E402
from task10f_covariance_diagnosis import (  # noqa: E402
    fit_pca_train_only,
    resolve_classes,
)
from src.preprocessing.flow_split import parse_ipv4  # noqa: E402


SEED = 0
PCA_DIM = 64
REG_COVAR = 1e-3
N_INIT = 3
MAX_ITER = 200
FIXED_CLASS_K = {"FTP": 5, "Cridex": 4, "Miuref": 5, "Outlook": 5}
CONTINUOUS_FEATURES = (
    "packet_count",
    "total_bytes",
    "duration",
    "forward_packet_count",
    "backward_packet_count",
)
RF_FEATURES = CONTINUOUS_FEATURES + (
    "src_port_category_code",
    "dst_port_category_code",
    "relative_time",
)
NA = "NA"


def port_category(port: Optional[int]) -> Optional[str]:
    """Return the required IANA-style numeric port band."""
    if port is None:
        return None
    value = int(port)
    if not 0 <= value <= 65535:
        return None
    if value <= 1023:
        return "well-known"
    if value <= 49151:
        return "registered"
    return "dynamic"


def _port_category_code(value: Optional[str]) -> Optional[int]:
    return {"well-known": 0, "registered": 1, "dynamic": 2}.get(value)


def epsilon_squared_kruskal(h_statistic: float, n: int, groups: int) -> float:
    """Kruskal-Wallis epsilon squared: max(0, (H-k+1)/(n-k))."""
    if groups < 2 or n <= groups or not np.isfinite(h_statistic):
        return float("nan")
    return max(0.0, float((h_statistic - groups + 1) / (n - groups)))


def _normalise_class_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def fixed_k_for_classes(class_pairs: Sequence[Tuple[str, int]]) -> Dict[int, int]:
    """Resolve the prescribed class-name K mapping without hard-coded ids."""
    normalized = {_normalise_class_name(k): v for k, v in FIXED_CLASS_K.items()}
    resolved = {}
    for name, class_id in class_pairs:
        key = _normalise_class_name(name)
        if key not in normalized:
            raise ValueError(f"no fixed Stage 2.5 K prescribed for class {name!r}")
        resolved[int(class_id)] = int(normalized[key])
    return resolved


def _safe_output_dir(root: Path, value: str) -> Path:
    output = Path(value)
    if not output.is_absolute():
        output = root / output
    output = output.resolve()
    protected = [
        (root / "outputs" / "stage2").resolve(),
        (root / "outputs" / "stage2_5_covariance_diagnosis").resolve(),
    ]
    for path in protected:
        if output == path or path in output.parents:
            raise ValueError(f"refusing to write into protected result tree: {path}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
    return output


def _validate_inputs(
    zt_dir: Path,
    zg_dir: Path,
    label_map: Path,
    flow_map: Path,
    flows_dir: Path,
) -> None:
    required = [label_map, flow_map]
    for directory in (zt_dir, zg_dir):
        for split in ("train", "val"):
            required.extend(
                directory / f"{prefix}_{split}.npy"
                for prefix in ("embedding", "labels", "flow_ids")
            )
    missing = [str(path) for path in required if not path.is_file()]
    if not flows_dir.is_dir():
        missing.append(str(flows_dir))
    if missing:
        raise FileNotFoundError("missing Stage 2.6 inputs:\n  " + "\n  ".join(missing))


def _deterministic_subset(
    indices: np.ndarray, cap: Optional[int], class_id: int
) -> np.ndarray:
    if cap is None or len(indices) <= cap:
        return indices
    if cap < 20:
        raise ValueError("diagnostic row cap must be at least 20")
    rng = np.random.default_rng(SEED + 1009 * int(class_id))
    return np.sort(rng.choice(indices, size=cap, replace=False))


def load_fixed_assignments(
    zt_dir: Path,
    zg_dir: Path,
    class_pairs: Sequence[Tuple[str, int]],
    fixed_k: Mapping[int, int],
    row_cap: Optional[int] = None,
) -> Tuple[List[dict], dict, List[dict]]:
    """Fit train-only scaler/PCA/GMM and assign train/val rows.

    The returned rows contain only flow id, split, class and component fields;
    Stage 0 metadata is joined later by exact flow id.
    """
    zt, labels, zg = _load_aligned(zt_dir, zg_dir)
    _, zf = _standardize(zt, zg)
    flow_ids = {
        split: np.load(zt_dir / f"flow_ids_{split}.npy")
        for split in ("train", "val")
    }
    for split in ("train", "val"):
        if len(flow_ids[split]) != len(zf[split]):
            raise AssertionError(f"{split}: flow-id/embedding length mismatch")
    pca, train_pca, val_pca = fit_pca_train_only(
        zf["train"], zf["val"], PCA_DIM, seed=SEED
    )
    pca_data = {"train": train_pca, "val": val_pca}
    del zt, zg, zf

    rows: List[dict] = []
    gmm_meta = {}
    warning_rows = []
    for class_name, class_id in class_pairs:
        split_indices = {}
        for split in ("train", "val"):
            available = np.flatnonzero(labels[split] == class_id)
            split_indices[split] = _deterministic_subset(available, row_cap, class_id)
            if len(split_indices[split]) == 0:
                raise ValueError(f"{class_name}: no {split} rows")
        k = fixed_k[class_id]
        if len(split_indices["train"]) < k:
            raise ValueError(f"{class_name}: fewer train rows than fixed K={k}")

        model = GaussianMixture(
            n_components=k,
            covariance_type="full",
            reg_covar=REG_COVAR,
            n_init=N_INIT,
            max_iter=MAX_ITER,
            random_state=SEED,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(pca_data["train"][split_indices["train"]])
        for index, item in enumerate(caught, start=1):
            warning_rows.append({
                "class": class_name,
                "class_id": class_id,
                "warning_index": index,
                "warning_category": item.category.__name__,
                "warning_message": str(item.message),
            })
        if not model.converged_ and not caught:
            warning_rows.append({
                "class": class_name,
                "class_id": class_id,
                "warning_index": 1,
                "warning_category": "ConvergenceStatus",
                "warning_message": "GaussianMixture.converged_ is False",
            })

        for split in ("train", "val"):
            indices = split_indices[split]
            x = pca_data[split][indices]
            component = model.predict(x)
            posterior = model.predict_proba(x).max(axis=1)
            for local, global_index in enumerate(indices):
                rows.append({
                    "flow_id": str(flow_ids[split][global_index]),
                    "split": split,
                    "class": class_name,
                    "class_id": int(class_id),
                    "fixed_K": int(k),
                    "component_id": int(component[local]),
                    "component_posterior": float(posterior[local]),
                })
        gmm_meta[class_name] = {
            "class_id": int(class_id),
            "fixed_K": int(k),
            "train_rows": int(len(split_indices["train"])),
            "val_rows": int(len(split_indices["val"])),
            "converged": bool(model.converged_),
            "n_iter": int(model.n_iter_),
            "weights": [float(x) for x in model.weights_],
        }
    pca_meta = {
        "fit_split": "train",
        "transform_splits": ["train", "val"],
        "n_components": PCA_DIM,
        "random_state": SEED,
        "train_rows": int(len(train_pca)),
        "val_rows": int(len(val_pca)),
        "original_dim": int(pca.n_features_in_),
        "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
    }
    return rows, {"pca": pca_meta, "gmm": gmm_meta}, warning_rows


def _metadata_from_packets(
    packets: Sequence[Sequence[Any]], source_pcap: Optional[str]
) -> dict:
    """Recover flow metadata from the already-materialized Stage 0 packet list."""
    result = {
        "source_pcap": source_pcap,
        "packet_count": None,
        "total_bytes": None,
        "duration": None,
        "forward_packet_count": None,
        "backward_packet_count": None,
        "src_port": None,
        "dst_port": None,
        "start_time": None,
    }
    if not packets:
        return result
    valid = [packet for packet in packets if len(packet) >= 5]
    if not valid:
        return result
    timestamps = [float(packet[0]) for packet in valid]
    result.update({
        "packet_count": len(packets),
        "total_bytes": int(sum(int(packet[1]) for packet in valid)),
        "duration": float(max(timestamps) - min(timestamps)),
        "forward_packet_count": int(sum(int(packet[3]) == 1 for packet in valid)),
        "backward_packet_count": int(sum(int(packet[3]) == 0 for packet in valid)),
        "start_time": float(min(timestamps)),
    })
    first = min(valid, key=lambda packet: float(packet[0]))
    parsed = parse_ipv4(bytes(first[4]))
    if parsed is not None:
        _, _, _, src_port, dst_port = parsed
        result["src_port"] = int(src_port)
        result["dst_port"] = int(dst_port)
    return result


def join_stage0_metadata(
    assignments: List[dict], flow_map_path: Path, flows_dir: Path
) -> Tuple[List[dict], dict]:
    """Join selected rows to the canonical Stage 0 PKLs by exact flow_id."""
    wanted = {row["flow_id"] for row in assignments}
    index: Dict[str, dict] = {}
    with flow_map_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"flow_id", "class_name", "source_file", "source_pkl"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"flow map lacks fields: {sorted(required)}")
        for row in reader:
            if row["flow_id"] in wanted:
                if row["flow_id"] in index:
                    raise ValueError(f"duplicate flow_id in flow map: {row['flow_id']}")
                index[row["flow_id"]] = row
    missing_index = wanted - set(index)
    if missing_index:
        raise ValueError(f"{len(missing_index)} assigned flow ids absent from flow_map")

    by_pkl: Dict[str, set] = defaultdict(set)
    for flow_id, row in index.items():
        by_pkl[row["source_pkl"]].add(flow_id)
    metadata: Dict[str, dict] = {}
    pkl_inventory = []
    for pkl_name, selected_ids in sorted(by_pkl.items()):
        path = flows_dir / pkl_name
        if not path.is_file():
            raise FileNotFoundError(f"indexed Stage 0 PKL not found: {path}")
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        packet_map = payload.get("packets")
        if not isinstance(packet_map, Mapping):
            raise ValueError(f"{pkl_name}: missing packets mapping")
        missing = selected_ids - set(packet_map)
        if missing:
            raise ValueError(f"{pkl_name}: {len(missing)} selected flow ids missing")
        for flow_id in selected_ids:
            meta = _metadata_from_packets(packet_map[flow_id], payload.get("source_file"))
            if meta["source_pcap"] != index[flow_id]["source_file"]:
                raise ValueError(f"{flow_id}: source PCAP mismatch between index and PKL")
            metadata[flow_id] = meta
        pkl_inventory.append({
            "source_pkl": pkl_name,
            "source_pcap": payload.get("source_file"),
            "selected_flows": len(selected_ids),
            "payload_flows": int(payload.get("n_flows", len(packet_map))),
        })

    missing_fields = Counter()
    for row in assignments:
        row.update(metadata[row["flow_id"]])
        row["src_port_category"] = port_category(row["src_port"])
        row["dst_port_category"] = port_category(row["dst_port"])
        row["src_port_category_code"] = _port_category_code(row["src_port_category"])
        row["dst_port_category_code"] = _port_category_code(row["dst_port_category"])
        for field in (
            "source_pcap", *CONTINUOUS_FEATURES, "src_port", "dst_port", "start_time"
        ):
            if row.get(field) is None:
                missing_fields[field] += 1
    return assignments, {
        "metadata_source": "existing Stage 0 flow PKLs; no raw PCAP reparsing",
        "pkl_inventory": pkl_inventory,
        "missing_field_counts": dict(sorted(missing_fields.items())),
    }


def add_relative_time(rows: List[dict]) -> dict:
    """Fit source-PCAP time ranges on train and apply them to train/validation."""
    train_times: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        if row["split"] == "train" and row.get("source_pcap") is not None and row.get("start_time") is not None:
            train_times[row["source_pcap"]].append(float(row["start_time"]))
    ranges = {
        pcap: (min(values), max(values)) for pcap, values in train_times.items()
    }
    for row in rows:
        pcap = row.get("source_pcap")
        start = row.get("start_time")
        bounds = ranges.get(pcap)
        if start is None or bounds is None or bounds[1] <= bounds[0]:
            row["relative_time"] = None
            row["relative_time_bin"] = None
            continue
        relative = (float(start) - bounds[0]) / (bounds[1] - bounds[0])
        row["relative_time"] = float(relative)
        row["relative_time_bin"] = int(np.clip(math.floor(relative * 10), 0, 9))
    return {
        pcap: {"train_min": low, "train_max": high, "fit_split": "train"}
        for pcap, (low, high) in sorted(ranges.items())
    }


def _safe_nmi(left: Sequence[Any], right: Sequence[Any]) -> float:
    if len(left) == 0 or len(left) != len(right):
        return float("nan")
    # Raw ports can legitimately have almost one unique value per flow.  The
    # sklearn target-type warning is irrelevant for this clustering metric.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The number of unique classes is greater than 50%.*",
            category=UserWarning,
        )
        return float(normalized_mutual_info_score(left, right))


def _safe_ami(left: Sequence[Any], right: Sequence[Any]) -> float:
    if len(left) == 0 or len(left) != len(right):
        return float("nan")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The number of unique classes is greater than 50%.*",
            category=UserWarning,
        )
        return float(adjusted_mutual_info_score(left, right))


def _entropy(values: Sequence[Any]) -> float:
    if not values:
        return float("nan")
    counts = np.asarray(list(Counter(values).values()), dtype=float)
    probs = counts / counts.sum()
    return float(-(probs * np.log2(probs)).sum())


def _subsets(rows: Sequence[dict]) -> Iterable[Tuple[str, str, List[dict]]]:
    classes = sorted({row["class"] for row in rows})
    for class_name in classes:
        for split in ("train", "val"):
            yield class_name, split, [
                row for row in rows if row["class"] == class_name and row["split"] == split
            ]


def pcap_component_audit(rows: Sequence[dict]) -> List[dict]:
    output = []
    for class_name, split, subset in _subsets(rows):
        valid = [row for row in subset if row.get("source_pcap") is not None]
        pcaps = sorted({row["source_pcap"] for row in valid})
        components = sorted({int(row["component_id"]) for row in subset})
        nmi = _safe_nmi([row["component_id"] for row in valid], [row["source_pcap"] for row in valid])
        ami = _safe_ami([row["component_id"] for row in valid], [row["source_pcap"] for row in valid])
        identifiable = len(pcaps) >= 2
        for component in components:
            comp_rows = [row for row in valid if int(row["component_id"]) == component]
            counts = Counter(row["source_pcap"] for row in comp_rows)
            dominant, dominant_count = counts.most_common(1)[0] if counts else (None, 0)
            dominant_ratio = dominant_count / len(comp_rows) if comp_rows else float("nan")
            for pcap in pcaps or [None]:
                count = counts.get(pcap, 0)
                output.append({
                    "class": class_name,
                    "split": split,
                    "component_id": component,
                    "source_pcap": pcap,
                    "count": count,
                    "component_total": len(comp_rows),
                    "within_component_ratio": count / len(comp_rows) if comp_rows else float("nan"),
                    "source_pcap_count_in_class": len(pcaps),
                    "source_pcap_identifiable": int(identifiable),
                    "nmi_component_source_pcap": nmi,
                    "ami_component_source_pcap": ami,
                    "dominant_pcap": dominant,
                    "dominant_pcap_ratio": dominant_ratio,
                    "pcap_entropy_bits": _entropy([row["source_pcap"] for row in comp_rows]),
                    "possible_pcap_shortcut": int(np.isfinite(dominant_ratio) and dominant_ratio >= 0.90),
                    "diagnostic_caveat": (
                        "single source PCAP in class; dominance is tautological and not identifiable"
                        if not identifiable else "ratio flag is diagnostic, not a paper-level conclusion"
                    ),
                })
    return output


def continuous_audit(rows: Sequence[dict]) -> Tuple[List[dict], List[dict]]:
    statistics, tests = [], []
    for class_name, split, subset in _subsets(rows):
        components = sorted({int(row["component_id"]) for row in subset})
        for feature in CONTINUOUS_FEATURES:
            groups = []
            for component in components:
                values = np.asarray([
                    float(row[feature]) for row in subset
                    if int(row["component_id"]) == component and row.get(feature) is not None
                ])
                if len(values):
                    groups.append(values)
                    q25, median, q75 = np.quantile(values, [0.25, 0.5, 0.75])
                    statistics.append({
                        "class": class_name, "split": split,
                        "component_id": component, "feature": feature,
                        "count": len(values), "mean": values.mean(),
                        "std": values.std(ddof=1) if len(values) > 1 else 0.0,
                        "median": median, "q25": q25, "q75": q75,
                        "min": values.min(), "max": values.max(),
                    })
            valid_n = sum(len(group) for group in groups)
            if len(groups) >= 2 and any(np.ptp(group) > 0 for group in groups):
                try:
                    h_stat, p_value = kruskal(*groups)
                    note = ""
                except ValueError as exc:
                    h_stat, p_value, note = float("nan"), float("nan"), str(exc)
            else:
                h_stat, p_value, note = float("nan"), float("nan"), "insufficient or constant groups"
            tests.append({
                "class": class_name, "split": split, "feature": feature,
                "valid_samples": valid_n, "groups_tested": len(groups),
                "kruskal_H": h_stat, "p_value": p_value,
                "epsilon_squared": epsilon_squared_kruskal(h_stat, valid_n, len(groups)),
                "effect_size_formula": "max(0,(H-k+1)/(n-k))",
                "note": note,
            })
    return statistics, tests


def _top_values(values: Sequence[Any], limit: int = 5) -> str:
    counts = Counter(values)
    total = sum(counts.values())
    result = [
        {"value": value, "count": count, "ratio": count / total}
        for value, count in counts.most_common(limit)
    ]
    return json.dumps(result, ensure_ascii=False, separators=(",", ":"))


def port_audit(rows: Sequence[dict]) -> List[dict]:
    output = []
    for class_name, split, subset in _subsets(rows):
        components = sorted({int(row["component_id"]) for row in subset})
        metrics = {}
        for field in ("src_port", "dst_port", "src_port_category", "dst_port_category"):
            valid = [row for row in subset if row.get(field) is not None]
            metrics[field] = _safe_nmi(
                [row["component_id"] for row in valid], [str(row[field]) for row in valid]
            )
        for component in components:
            comp = [row for row in subset if int(row["component_id"]) == component]
            src = [int(row["src_port"]) for row in comp if row.get("src_port") is not None]
            dst = [int(row["dst_port"]) for row in comp if row.get("dst_port") is not None]
            src_cat = [row["src_port_category"] for row in comp if row.get("src_port_category") is not None]
            dst_cat = [row["dst_port_category"] for row in comp if row.get("dst_port_category") is not None]
            output.append({
                "class": class_name, "split": split, "component_id": component,
                "component_count": len(comp),
                "src_port_valid": len(src), "dst_port_valid": len(dst),
                "nmi_component_src_port": metrics["src_port"],
                "nmi_component_dst_port": metrics["dst_port"],
                "nmi_component_src_port_category": metrics["src_port_category"],
                "nmi_component_dst_port_category": metrics["dst_port_category"],
                "top5_src_ports": _top_values(src),
                "top5_dst_ports": _top_values(dst),
                "src_port_categories": _top_values(src_cat, 3),
                "dst_port_categories": _top_values(dst_cat, 3),
            })
    return output


def time_audit(rows: Sequence[dict]) -> List[dict]:
    output = []
    for class_name, split, subset in _subsets(rows):
        valid_bins = [row for row in subset if row.get("relative_time_bin") is not None]
        nmi = _safe_nmi(
            [row["component_id"] for row in valid_bins],
            [row["relative_time_bin"] for row in valid_bins],
        )
        for component in sorted({int(row["component_id"]) for row in subset}):
            comp = [row for row in subset if int(row["component_id"]) == component]
            start = np.asarray([float(row["start_time"]) for row in comp if row.get("start_time") is not None])
            relative = np.asarray([float(row["relative_time"]) for row in comp if row.get("relative_time") is not None])
            output.append({
                "class": class_name, "split": split, "component_id": component,
                "component_count": len(comp), "start_time_valid": len(start),
                "start_time_min": start.min() if len(start) else float("nan"),
                "start_time_median": np.median(start) if len(start) else float("nan"),
                "start_time_max": start.max() if len(start) else float("nan"),
                "relative_time_valid": len(relative),
                "relative_time_min": relative.min() if len(relative) else float("nan"),
                "relative_time_median": np.median(relative) if len(relative) else float("nan"),
                "relative_time_max": relative.max() if len(relative) else float("nan"),
                "nmi_component_relative_time_bin": nmi,
                "time_normalization": "source-PCAP range fit on train only; 10 clipped bins",
            })
    return output


def rf_predictability(rows: Sequence[dict]) -> List[dict]:
    output = []
    for class_name in sorted({row["class"] for row in rows}):
        subset = [row for row in rows if row["class"] == class_name and row["split"] == "train"]
        y = np.asarray([int(row["component_id"]) for row in subset])
        component_counts = Counter(y.tolist())
        base = {
            "class": class_name,
            "samples": len(subset),
            "components": len(component_counts),
            "component_counts": json.dumps(component_counts, sort_keys=True),
            "features": ";".join(RF_FEATURES),
            "n_estimators": 200,
            "random_state": SEED,
            "class_weight": "balanced",
            "internal_split": "train 80/20",
        }
        evaluation_size = math.ceil(0.2 * len(y))
        fit_size = len(y) - evaluation_size
        if (
            len(component_counts) < 2
            or min(component_counts.values()) < 2
            or evaluation_size < len(component_counts)
            or fit_size < len(component_counts)
        ):
            output.append({**base, "status": "skipped", "skip_reason": "component too small for stratification", "macro_f1": float("nan"), "balanced_accuracy": float("nan"), "fit_samples": 0, "eval_samples": 0})
            continue
        matrix = []
        for row in subset:
            matrix.append([
                float(row[field]) if row.get(field) is not None else float("nan")
                for field in RF_FEATURES
            ])
        x = np.asarray(matrix, dtype=float)
        train_idx, eval_idx = train_test_split(
            np.arange(len(y)), test_size=0.2, random_state=SEED, stratify=y
        )
        medians = np.nanmedian(x[train_idx], axis=0)
        if np.isnan(medians).any():
            output.append({**base, "status": "skipped", "skip_reason": "one or more metadata features entirely missing", "macro_f1": float("nan"), "balanced_accuracy": float("nan"), "fit_samples": 0, "eval_samples": 0})
            continue
        x = np.where(np.isnan(x), medians, x)
        model = RandomForestClassifier(
            n_estimators=200,
            random_state=SEED,
            class_weight="balanced",
            # A single worker avoids oversubscribing the shared host.  This is
            # computational only and does not alter the fitted estimator set.
            n_jobs=1,
        )
        model.fit(x[train_idx], y[train_idx])
        predicted = model.predict(x[eval_idx])
        output.append({
            **base, "status": "completed", "skip_reason": "",
            "macro_f1": f1_score(y[eval_idx], predicted, average="macro"),
            "balanced_accuracy": balanced_accuracy_score(y[eval_idx], predicted),
            "fit_samples": len(train_idx), "eval_samples": len(eval_idx),
            "feature_importances": json.dumps(
                dict(zip(RF_FEATURES, map(float, model.feature_importances_))),
                sort_keys=True,
            ),
        })
    return output


def train_val_consistency(rows: Sequence[dict]) -> List[dict]:
    output = []
    for class_name in sorted({row["class"] for row in rows}):
        class_rows = [row for row in rows if row["class"] == class_name]
        components = sorted({int(row["component_id"]) for row in class_rows})
        split_counts = {
            split: Counter(int(row["component_id"]) for row in class_rows if row["split"] == split)
            for split in ("train", "val")
        }
        split_totals = {split: sum(counts.values()) for split, counts in split_counts.items()}
        tv = 0.5 * sum(
            abs(
                split_counts["train"][component] / split_totals["train"]
                - split_counts["val"][component] / split_totals["val"]
            )
            for component in components
        )
        for component in components:
            for feature in CONTINUOUS_FEATURES + ("relative_time",):
                values = {}
                for split in ("train", "val"):
                    values[split] = np.asarray([
                        float(row[feature]) for row in class_rows
                        if row["split"] == split
                        and int(row["component_id"]) == component
                        and row.get(feature) is not None
                    ])
                train_median = np.median(values["train"]) if len(values["train"]) else float("nan")
                val_median = np.median(values["val"]) if len(values["val"]) else float("nan")
                if len(values["train"]):
                    q25, q75 = np.quantile(values["train"], [0.25, 0.75])
                    train_iqr = q75 - q25
                else:
                    train_iqr = float("nan")
                absolute = abs(val_median - train_median) if np.isfinite(train_median) and np.isfinite(val_median) else float("nan")
                normalized = absolute / train_iqr if np.isfinite(train_iqr) and train_iqr > 0 else float("nan")
                output.append({
                    "class": class_name, "component_id": component, "feature": feature,
                    "train_count": len(values["train"]), "val_count": len(values["val"]),
                    "train_component_ratio": split_counts["train"][component] / split_totals["train"],
                    "val_component_ratio": split_counts["val"][component] / split_totals["val"],
                    "absolute_component_ratio_difference": abs(
                        split_counts["train"][component] / split_totals["train"]
                        - split_counts["val"][component] / split_totals["val"]
                    ),
                    "component_distribution_total_variation": tv,
                    "train_median": train_median, "val_median": val_median,
                    "absolute_median_difference": absolute,
                    "train_iqr": train_iqr,
                    "median_difference_over_train_iqr": normalized,
                })
    return output


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "class"


def make_figures(rows: Sequence[dict], figures_dir: Path) -> Dict[str, List[str]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    rng = np.random.default_rng(SEED)
    for class_name in sorted({row["class"] for row in rows}):
        subset = [row for row in rows if row["class"] == class_name]
        components = sorted({int(row["component_id"]) for row in subset})
        pcaps = sorted({row["source_pcap"] for row in subset if row.get("source_pcap") is not None})
        matrix = np.asarray([
            [sum(row["component_id"] == component and row.get("source_pcap") == pcap for row in subset) for pcap in pcaps]
            for component in components
        ], dtype=float)
        row_sums = matrix.sum(axis=1, keepdims=True)
        ratios = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums > 0)
        fig, ax = plt.subplots(figsize=(max(5, len(pcaps) * 1.2), 4.5))
        image = ax.imshow(ratios, aspect="auto", vmin=0, vmax=1, cmap="Blues")
        ax.set_xticks(range(len(pcaps)), pcaps, rotation=30, ha="right")
        ax.set_yticks(range(len(components)), [f"C{x}" for x in components])
        ax.set_xlabel("Source PCAP")
        ax.set_ylabel("Fixed GMM component")
        ax.set_title(f"{class_name}: component x source PCAP (train+val)")
        fig.colorbar(image, ax=ax, label="within-component ratio")
        fig.tight_layout()
        paths = [f"figures/{_slug(class_name)}_component_source_pcap.png"]
        fig.savefig(figures_dir.parent / paths[-1], dpi=180)
        plt.close(fig)

        for feature in ("packet_count", "duration", "total_bytes"):
            arrays = []
            for component in components:
                values = np.asarray([
                    float(row[feature]) for row in subset
                    if int(row["component_id"]) == component and row.get(feature) is not None
                ])
                if len(values) > 10000:
                    values = rng.choice(values, 10000, replace=False)
                arrays.append(values)
            fig, ax = plt.subplots(figsize=(7, 4.8))
            ax.boxplot(arrays, tick_labels=[f"C{x}" for x in components], showfliers=False)
            ax.set_xlabel("Fixed GMM component")
            ax.set_ylabel(feature)
            ax.set_title(f"{class_name}: {feature} by component (train+val)")
            ax.grid(axis="y", alpha=0.25)
            fig.tight_layout()
            path = f"figures/{_slug(class_name)}_{feature}_distribution.png"
            fig.savefig(figures_dir.parent / path, dpi=180)
            plt.close(fig)
            paths.append(path)
        outputs[class_name] = paths
    return outputs


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return NA
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return NA if not np.isfinite(number) else f"{number:.{digits}f}"


def build_summary(
    rows: Sequence[dict],
    pcap_rows: Sequence[dict],
    test_rows: Sequence[dict],
    port_rows: Sequence[dict],
    time_rows: Sequence[dict],
    rf_rows: Sequence[dict],
    consistency_rows: Sequence[dict],
    missing_counts: Mapping[str, int],
) -> Tuple[str, Dict[str, str]]:
    labels = {}
    lines = [
        "# Stage 2.6 Component / Shortcut Audit",
        "",
        "This is a diagnostic audit of fixed Stage 2.5 residual components. It does not select K and does not establish semantic modes or real attack subfamilies.",
        "",
        "## Protocol safeguards",
        "",
        "- z_f standardization, PCA-64, and full-covariance GMM were fit on Known train only.",
        "- Fixed K: FTP=5, Cridex=4, Miuref=5, Outlook=5; no K search or reselection was performed.",
        "- Validation was used only for assignment and consistency diagnostics; no test embedding or test metadata was loaded.",
        "- Metadata came from existing Stage 0 PKLs. Raw PCAP files were not reparsed.",
        "- Relative time uses `(start_time - train_pcap_min) / (train_pcap_max - train_pcap_min)` per source PCAP; validation reuses train ranges.",
        "- Kruskal-Wallis effect size is `epsilon_squared = max(0, (H-k+1)/(n-k))`.",
        f"- Missing metadata fields are serialized as `NA`; aggregate missing counts: `{json.dumps(dict(missing_counts), sort_keys=True)}`.",
        "",
        "## Per-class answers",
        "",
    ]
    for class_name in sorted({row["class"] for row in rows}):
        train_pcap = [row for row in pcap_rows if row["class"] == class_name and row["split"] == "train"]
        pcap_count = max((row["source_pcap_count_in_class"] for row in train_pcap), default=0)
        pcap_nmi = train_pcap[0]["nmi_component_source_pcap"] if train_pcap else float("nan")
        class_tests = [row for row in test_rows if row["class"] == class_name]
        train_tests = {row["feature"]: row for row in class_tests if row["split"] == "train"}
        val_tests = {row["feature"]: row for row in class_tests if row["split"] == "val"}
        strongest_train = max(train_tests.values(), key=lambda row: (-1 if not np.isfinite(row["epsilon_squared"]) else row["epsilon_squared"]))
        strongest_val = max(val_tests.values(), key=lambda row: (-1 if not np.isfinite(row["epsilon_squared"]) else row["epsilon_squared"]))
        class_ports = [row for row in port_rows if row["class"] == class_name and row["split"] == "train"]
        port_metrics = class_ports[0] if class_ports else {}
        class_time = [row for row in time_rows if row["class"] == class_name and row["split"] == "train"]
        time_nmi = class_time[0]["nmi_component_relative_time_bin"] if class_time else float("nan")
        rf = next(row for row in rf_rows if row["class"] == class_name)
        consistency = [row for row in consistency_rows if row["class"] == class_name]
        tv = consistency[0]["component_distribution_total_variation"] if consistency else float("nan")

        train_eps = [row["epsilon_squared"] for row in train_tests.values() if np.isfinite(row["epsilon_squared"])]
        val_eps = [row["epsilon_squared"] for row in val_tests.values() if np.isfinite(row["epsilon_squared"])]
        max_train_eps = max(train_eps, default=0.0)
        max_val_eps = max(val_eps, default=0.0)
        port_nmis = [
            port_metrics.get("nmi_component_src_port_category", float("nan")),
            port_metrics.get("nmi_component_dst_port_category", float("nan")),
        ]
        max_port_nmi = max((x for x in port_nmis if np.isfinite(x)), default=0.0)
        # Diagnostic labels use observed factor dominance, not an RF score threshold.
        if pcap_count >= 2 and np.isfinite(pcap_nmi) and pcap_nmi >= max(max_train_eps, max_port_nmi, time_nmi):
            label = "A. Likely acquisition / dataset shortcut dominated"
        elif max_train_eps >= 0.14 and max_val_eps >= 0.14 and max_train_eps >= max(max_port_nmi, time_nmi):
            label = "B. Likely simple traffic-statistics dominated"
        elif max(max_train_eps, max_port_nmi, time_nmi) >= 0.05:
            label = "C. Mixed structure"
        else:
            label = "D. No obvious shortcut explanation"
        labels[class_name] = label

        lines.extend([
            f"### {class_name}",
            "",
            f"1. **Source PCAP:** {pcap_count} source PCAP(s) in train; NMI={_fmt(pcap_nmi)}. " + (
                "Because the class has only one source PCAP, component-PCAP dependence is not identifiable; 100% dominant ratios are tautological and are not shortcut evidence."
                if pcap_count < 2 else "The >=0.90 dominance flag is diagnostic only."
            ),
            f"2. **Packet/byte/duration statistics:** strongest train effect is {strongest_train['feature']} (epsilon_squared={_fmt(strongest_train['epsilon_squared'])}); strongest validation effect is {strongest_val['feature']} (epsilon_squared={_fmt(strongest_val['epsilon_squared'])}).",
            f"3. **Ports:** train NMI src raw/category={_fmt(port_metrics.get('nmi_component_src_port'))}/{_fmt(port_metrics.get('nmi_component_src_port_category'))}, dst raw/category={_fmt(port_metrics.get('nmi_component_dst_port'))}/{_fmt(port_metrics.get('nmi_component_dst_port_category'))}.",
            f"4. **Capture time:** train component-vs-relative-time-bin NMI={_fmt(time_nmi)}; time bases were normalized separately by source PCAP.",
            f"5. **Simple metadata predictability:** status={rf['status']}, Macro-F1={_fmt(rf['macro_f1'])}, Balanced Accuracy={_fmt(rf['balanced_accuracy'])}. These values are reported directly without a self-defined 'high' threshold.",
            f"6. **Train/validation consistency:** component-proportion total-variation distance={_fmt(tv)}; detailed per-component ratios and median shifts are in `train_val_component_consistency.csv`.",
            "",
            f"**Diagnostic label:** {label}",
            "",
        ])
    label_counts = Counter(labels.values())
    if label_counts.get("B. Likely simple traffic-statistics dominated", 0) == len(labels):
        global_text = "Across all four classes, residual components are primarily associated with simple flow statistics under this audit."
    elif label_counts.get("D. No obvious shortcut explanation", 0) == len(labels):
        global_text = "No obvious shortcut explanation was identified. This does not imply a semantic interpretation of the components."
    else:
        global_text = "The four classes show mixed diagnostic patterns; no single acquisition, traffic-statistics, port, or time explanation accounts for every class."
    lines.extend([
        "## Global conclusion",
        "",
        global_text,
        "Source-PCAP effects cannot be assessed within these four classes because each has only one source PCAP. Residual component structure must not be described as semantic modes or real attack subfamilies on this evidence.",
        "",
        "## Interpretation boundary",
        "",
        "This audit identifies statistical associations, not causal mechanisms. A component assignment may correlate with multiple metadata fields. No Stage 3 procedure was run.",
        "",
    ])
    return "\n".join(lines), labels


def _csv_value(value: Any) -> Any:
    if value is None:
        return NA
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return NA
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fields})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2.6 component / shortcut audit")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--zt-dir", default=None)
    parser.add_argument("--zg-dir", default=None)
    parser.add_argument("--label-map", default=None)
    parser.add_argument("--flow-map", default=None)
    parser.add_argument("--flows-dir", default=None)
    parser.add_argument("--out", default="outputs/stage2_6_component_shortcut_audit")
    parser.add_argument("--classes", default=",".join(FIXED_CLASS_K))
    parser.add_argument(
        "--diagnostic-row-cap", type=int, default=None,
        help="Deterministic per-class/split cap for isolated smoke runs only; formal run leaves this unset",
    )
    args = parser.parse_args()

    started = time.time()
    root = Path(args.project_root).resolve()
    zt_dir = Path(args.zt_dir).resolve() if args.zt_dir else root / "outputs/stage1/modelA/embeddings"
    zg_dir = Path(args.zg_dir).resolve() if args.zg_dir else root / "outputs/stage1/modelB/seeds/seed2"
    label_map = Path(args.label_map).resolve() if args.label_map else root / "data/fig_graph/all_flows/label_map.json"
    flow_map = Path(args.flow_map).resolve() if args.flow_map else root / "data/trafficformer_input/compatible_min1/flow_map.csv"
    flows_dir = Path(args.flows_dir).resolve() if args.flows_dir else root / "data/flows"
    output_dir = _safe_output_dir(root, args.out)
    _validate_inputs(zt_dir, zg_dir, label_map, flow_map, flows_dir)

    label_payload = json.loads(label_map.read_text(encoding="utf-8"))
    requested = [item.strip() for item in args.classes.split(",") if item.strip()]
    class_pairs = resolve_classes(label_payload["class_to_id"], requested)
    if {_normalise_class_name(name) for name, _ in class_pairs} != {_normalise_class_name(name) for name in FIXED_CLASS_K}:
        raise ValueError("Stage 2.6 formal protocol requires exactly FTP, Cridex, Miuref, Outlook")
    fixed_k = fixed_k_for_classes(class_pairs)

    print("[1/8] Fit train-only z_f standardization/PCA-64 and fixed full GMMs")
    assignments, model_meta, warning_rows = load_fixed_assignments(
        zt_dir, zg_dir, class_pairs, fixed_k, args.diagnostic_row_cap
    )
    print("[2/8] Join exact flow ids to existing Stage 0 PKL metadata")
    assignments, metadata_meta = join_stage0_metadata(assignments, flow_map, flows_dir)
    time_ranges = add_relative_time(assignments)

    print("[3/8] Audit source-PCAP and continuous flow statistics")
    pcap_rows = pcap_component_audit(assignments)
    flow_stats, feature_tests = continuous_audit(assignments)
    print("[4/8] Audit ports and source-PCAP-normalized capture time")
    port_rows = port_audit(assignments)
    time_rows = time_audit(assignments)
    print("[5/8] Evaluate metadata-only component predictability")
    rf_rows = rf_predictability(assignments)
    consistency_rows = train_val_consistency(assignments)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage_dir = Path(tempfile.mkdtemp(prefix=".stage2_6_component_audit_", dir=output_dir.parent))
    try:
        print("[6/8] Create required figures and auditable tables")
        figure_map = make_figures(assignments, stage_dir / "figures")
        assignment_fields = (
            "flow_id", "split", "class", "class_id", "fixed_K", "component_id",
            "component_posterior", "source_pcap", "packet_count", "total_bytes",
            "duration", "forward_packet_count", "backward_packet_count", "src_port",
            "dst_port", "src_port_category", "dst_port_category", "start_time",
            "relative_time", "relative_time_bin",
        )
        pcap_fields = (
            "class", "split", "component_id", "source_pcap", "count", "component_total",
            "within_component_ratio", "source_pcap_count_in_class", "source_pcap_identifiable",
            "nmi_component_source_pcap", "ami_component_source_pcap", "dominant_pcap",
            "dominant_pcap_ratio", "pcap_entropy_bits", "possible_pcap_shortcut",
            "diagnostic_caveat",
        )
        flow_stat_fields = (
            "class", "split", "component_id", "feature", "count", "mean", "std",
            "median", "q25", "q75", "min", "max",
        )
        test_fields = (
            "class", "split", "feature", "valid_samples", "groups_tested", "kruskal_H",
            "p_value", "epsilon_squared", "effect_size_formula", "note",
        )
        port_fields = (
            "class", "split", "component_id", "component_count", "src_port_valid",
            "dst_port_valid", "nmi_component_src_port", "nmi_component_dst_port",
            "nmi_component_src_port_category", "nmi_component_dst_port_category",
            "top5_src_ports", "top5_dst_ports", "src_port_categories", "dst_port_categories",
        )
        time_fields = (
            "class", "split", "component_id", "component_count", "start_time_valid",
            "start_time_min", "start_time_median", "start_time_max", "relative_time_valid",
            "relative_time_min", "relative_time_median", "relative_time_max",
            "nmi_component_relative_time_bin", "time_normalization",
        )
        rf_fields = (
            "class", "status", "skip_reason", "samples", "components", "component_counts",
            "features", "n_estimators", "random_state", "class_weight", "internal_split",
            "fit_samples", "eval_samples", "macro_f1", "balanced_accuracy", "feature_importances",
        )
        consistency_fields = (
            "class", "component_id", "feature", "train_count", "val_count",
            "train_component_ratio", "val_component_ratio", "absolute_component_ratio_difference",
            "component_distribution_total_variation", "train_median", "val_median",
            "absolute_median_difference", "train_iqr", "median_difference_over_train_iqr",
        )
        _write_csv(stage_dir / "component_assignments.csv", assignments, assignment_fields)
        _write_csv(stage_dir / "pcap_component_matrix.csv", pcap_rows, pcap_fields)
        _write_csv(stage_dir / "component_flow_statistics.csv", flow_stats, flow_stat_fields)
        _write_csv(stage_dir / "continuous_feature_tests.csv", feature_tests, test_fields)
        _write_csv(stage_dir / "port_component_statistics.csv", port_rows, port_fields)
        _write_csv(stage_dir / "time_component_statistics.csv", time_rows, time_fields)
        _write_csv(stage_dir / "component_predictability.csv", rf_rows, rf_fields)
        _write_csv(stage_dir / "train_val_component_consistency.csv", consistency_rows, consistency_fields)

        summary, diagnostic_labels = build_summary(
            assignments, pcap_rows, feature_tests, port_rows, time_rows, rf_rows,
            consistency_rows, metadata_meta["missing_field_counts"],
        )
        (stage_dir / "audit_summary.md").write_text(summary, encoding="utf-8")

        print("[7/8] Write provenance, safeguards and file hashes")
        metadata = {
            "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "project_root": str(root),
            "inputs": {
                "zt_dir": str(zt_dir), "zg_dir": str(zg_dir),
                "label_map": str(label_map), "flow_map": str(flow_map),
                "flows_dir": str(flows_dir),
            },
            "loaded_splits": ["train", "val"],
            "test_embedding_loaded": False,
            "test_metadata_loaded": False,
            "encoder_retrained": False,
            "stage3_run": False,
            "k_reselected": False,
            "representation": "train-standardized z_f -> train-fitted PCA64",
            "model": {
                "covariance_type": "full", "reg_covar": REG_COVAR,
                "n_init": N_INIT, "max_iter": MAX_ITER, "seed": SEED,
                "fixed_class_K": {name: fixed_k[class_id] for name, class_id in class_pairs},
                **model_meta,
            },
            "thread_environment": {
                name: os.environ.get(name, NA)
                for name in (
                    "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                )
            },
            "diagnostic_row_cap": args.diagnostic_row_cap,
            "formal_unsampled": args.diagnostic_row_cap is None,
            "metadata": metadata_meta,
            "relative_time_ranges": time_ranges,
            "gmm_warnings": warning_rows,
            "diagnostic_labels": diagnostic_labels,
            "figures": figure_map,
            "elapsed_seconds": time.time() - started,
        }
        files = sorted(path for path in stage_dir.rglob("*") if path.is_file())
        metadata["output_files"] = [str(path.relative_to(stage_dir)) for path in files]
        metadata["output_sha256"] = {
            str(path.relative_to(stage_dir)): _sha256(path) for path in files
        }
        (stage_dir / "run_metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

        if output_dir.exists():
            output_dir.rmdir()
        os.replace(stage_dir, output_dir)
    except Exception:
        shutil.rmtree(stage_dir, ignore_errors=True)
        raise

    print(f"[8/8] Complete: {output_dir}")
    print(f"Assignments: {len(assignments)}; GMM warnings: {len(warning_rows)}")


if __name__ == "__main__":
    main()
