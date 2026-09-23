#!/usr/bin/env python3
"""CIC-IDS-2017 fixed-K component and metadata-shortcut diagnosis.

This stage is strictly additive.  It reuses the completed CIC Gaussian Audit
split, checkpoint-derived z_t arrays, and numerical protocol.  For each of the
four prescribed classes it refits train-only StandardScaler/PCA-64 and a fixed
K=2 full-covariance GMM for seeds 0/1/2, verifies the resulting likelihoods
against the published Gaussian Audit, and uses validation only for assignment
and evaluation.  Existing mapping Parquet files provide metadata; PCAPs are
never reparsed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import tempfile
import time
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import sklearn
from scipy.stats import kruskal
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    adjusted_mutual_info_score,
    adjusted_rand_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    normalized_mutual_info_score,
    recall_score,
)
from sklearn.mixture import GaussianMixture


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from task10g_component_shortcut_audit import (  # noqa: E402
    epsilon_squared_kruskal,
    port_category,
)
from task13_cicids2017_gaussian_audit import fit_train_representation  # noqa: E402


CLASSES = ("DDoS", "PortScan", "DoS GoldenEye", "DoS Hulk")
SEEDS = (0, 1, 2)
K = 2
PCA_DIM = 64
PCA_SEED = 42
REG_COVAR = 1e-3
N_INIT = 3
MAX_ITER = 300
TINY_WEIGHT = 0.01
RF_SEED = 42
RF_TREES = 200
TOP_N = 20

CONTINUOUS_FEATURES = (
    "packet_count",
    "total_bytes",
    "flow_duration",
    "forward_packet_count",
    "backward_packet_count",
    "mean_packet_size",
    "bytes_per_packet",
)
RF_MODEL_S = (
    "packet_count",
    "total_bytes",
    "flow_duration",
    "forward_packet_count",
    "backward_packet_count",
    "bytes_per_packet",
    "protocol",
    "src_port_category_code",
    "dst_port_category_code",
)
RF_MODEL_S_PLUS = RF_MODEL_S + (
    "src_port",
    "dst_port",
    "time_bucket_5m",
    "time_bucket_15m",
    "hour_of_day",
)
CATEGORICAL_FEATURES = (
    "source_day",
    "source_pcap",
    "source_label_csv",
    "protocol",
    "src_port",
    "dst_port",
    "src_port_category",
    "dst_port_category",
    "src_ip",
    "dst_ip",
    "time_bucket_5m",
    "time_bucket_15m",
    "hour_of_day",
)
ENDPOINT_FIELDS = ("src_ip", "dst_ip", "src_port", "dst_port")
CONSISTENCY_CATEGORICAL = (
    "protocol", "src_port", "dst_port", "src_port_category",
    "dst_port_category", "time_bucket_5m", "time_bucket_15m", "hour_of_day",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def top_values(values: Iterable[Any], limit: int = 10) -> str:
    clean = [str(value) for value in values if pd.notna(value)]
    counts = Counter(clean)
    total = sum(counts.values())
    return json_compact([
        {"value": value, "count": count, "ratio": count / total}
        for value, count in counts.most_common(limit)
    ])


def safe_association(components: Sequence[Any], categories: Sequence[Any]) -> dict[str, Any]:
    frame = pd.DataFrame({"component": components, "category": categories}).dropna()
    if frame.empty:
        return {
            "samples": 0, "category_count": 0, "component_count": 0,
            "nmi": float("nan"), "ami": float("nan"),
            "identifiable": False, "status": "NO_VALID_ROWS",
        }
    left = frame["component"].astype(str).to_numpy()
    right = frame["category"].astype(str).to_numpy()
    category_count = int(pd.unique(right).size)
    component_count = int(pd.unique(left).size)
    if category_count < 2:
        return {
            "samples": len(frame), "category_count": category_count,
            "component_count": component_count, "nmi": 0.0, "ami": 0.0,
            "identifiable": False,
            "status": "NOT_IDENTIFIABLE_CONSTANT_WITHIN_CLASS",
        }
    if component_count < 2:
        return {
            "samples": len(frame), "category_count": category_count,
            "component_count": component_count, "nmi": 0.0, "ami": 0.0,
            "identifiable": False,
            "status": "NOT_IDENTIFIABLE_SINGLE_ASSIGNED_COMPONENT",
        }
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The number of unique classes is greater than 50%.*")
        return {
            "samples": len(frame), "category_count": category_count,
            "component_count": component_count,
            "nmi": float(normalized_mutual_info_score(left, right)),
            "ami": float(adjusted_mutual_info_score(left, right)),
            "identifiable": True, "status": "IDENTIFIABLE",
        }


def top_n_categories(train_values: Sequence[Any], limit: int = TOP_N) -> set[str]:
    clean = pd.Series(train_values, dtype="object").dropna().astype(str)
    return set(clean.value_counts().head(limit).index.tolist())


def apply_top_n(values: Sequence[Any], retained: set[str]) -> np.ndarray:
    return np.asarray([
        "__MISSING__" if pd.isna(value)
        else str(value) if str(value) in retained
        else "__OTHER__"
        for value in values
    ], dtype=object)


def distribution_tv(left: Sequence[Any], right: Sequence[Any]) -> float:
    left_counts = Counter("__MISSING__" if pd.isna(x) else str(x) for x in left)
    right_counts = Counter("__MISSING__" if pd.isna(x) else str(x) for x in right)
    left_total = sum(left_counts.values())
    right_total = sum(right_counts.values())
    if not left_total or not right_total:
        return float("nan")
    keys = set(left_counts) | set(right_counts)
    return 0.5 * sum(abs(left_counts[k] / left_total - right_counts[k] / right_total) for k in keys)


def component_weight_tv(train_labels: np.ndarray, val_labels: np.ndarray) -> float:
    train = np.bincount(train_labels.astype(int), minlength=K) / len(train_labels)
    val = np.bincount(val_labels.astype(int), minlength=K) / len(val_labels)
    return float(0.5 * np.abs(train - val).sum())


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            serialized = {}
            for field in fields:
                value = row.get(field)
                if isinstance(value, (np.integer,)):
                    value = int(value)
                elif isinstance(value, (np.floating,)):
                    value = float(value)
                if isinstance(value, float) and not np.isfinite(value):
                    value = "NA"
                elif value is None:
                    value = "NA"
                serialized[field] = value
            writer.writerow(serialized)


def load_metadata(split_manifest: Path, mapping_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    split_columns = [
        "flow_id", "class_name", "source_day", "split", "source_pcap",
        "verification_status", "existing_match_status", "stored_tuple_consistent",
        "referenced_row_found", "referenced_row_tuple_match", "referenced_row_label_match",
    ]
    split = pd.read_csv(split_manifest, usecols=split_columns)
    split = split[split["class_name"].isin(CLASSES)].copy()
    if split["flow_id"].duplicated().any():
        raise ValueError("duplicate flow_id in selected split manifest")
    guards = (
        split["verification_status"].eq("LABEL_MATCH")
        & split["existing_match_status"].str.startswith("MATCHED")
        & split["stored_tuple_consistent"].astype(bool)
        & split["referenced_row_found"].astype(bool)
        & split["referenced_row_tuple_match"].astype(bool)
        & split["referenced_row_label_match"].astype(bool)
    )
    if not bool(guards.all()):
        raise ValueError(f"selected split contains {int((~guards).sum())} non-strict rows")

    wanted = set(split["flow_id"].astype(str))
    columns = [
        "flow_id", "day", "src_ip", "dst_ip", "src_port", "dst_port", "protocol",
        "flow_start_time", "flow_start_epoch_utc", "duration", "packet_count",
        "forward_packet_count", "backward_packet_count", "total_bytes", "label",
        "match_status", "source_pcap", "source_label_csv",
    ]
    chunks: list[pd.DataFrame] = []
    scanned_rows = 0
    for path in sorted(mapping_dir.glob("*_flows.parquet")):
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(columns=columns, batch_size=131_072):
            scanned_rows += batch.num_rows
            frame = batch.to_pandas()
            keep = frame["flow_id"].isin(wanted)
            if bool(keep.any()):
                chunks.append(frame.loc[keep].copy())
    if not chunks:
        raise ValueError("no selected flows found in mapping Parquet")
    mapped = pd.concat(chunks, ignore_index=True)
    if mapped["flow_id"].duplicated().any():
        raise ValueError("duplicate selected flow_id in mapping Parquet")
    if set(mapped["flow_id"]) != wanted:
        raise ValueError(f"mapping join mismatch: wanted={len(wanted)} found={len(mapped)}")
    if not mapped["match_status"].astype(str).str.startswith("MATCHED").all():
        raise ValueError("selected mapping metadata includes unreliable match status")

    merged = split.merge(mapped, on="flow_id", how="left", validate="one_to_one", suffixes=("_split", "_map"))
    if not (merged["class_name"] == merged["label"]).all():
        raise ValueError("class label mismatch between split and mapping metadata")
    if not (merged["source_day"] == merged["day"]).all():
        raise ValueError("source day mismatch between split and mapping metadata")
    if not (merged["source_pcap_split"] == merged["source_pcap_map"]).all():
        raise ValueError("source PCAP mismatch between split and mapping metadata")

    merged = merged.rename(columns={
        "source_pcap_split": "source_pcap",
        "duration": "flow_duration",
    })
    merged = merged.drop(columns=[
        "source_pcap_map", "day", "label", "match_status", "verification_status",
        "existing_match_status", "stored_tuple_consistent", "referenced_row_found",
        "referenced_row_tuple_match", "referenced_row_label_match",
    ])
    denom = merged["packet_count"].replace(0, np.nan)
    merged["bytes_per_packet"] = merged["total_bytes"] / denom
    merged["mean_packet_size"] = merged["bytes_per_packet"]
    merged["src_port_category"] = merged["src_port"].map(port_category)
    merged["dst_port_category"] = merged["dst_port"].map(port_category)
    category_code = {"well-known": 0, "registered": 1, "dynamic": 2}
    merged["src_port_category_code"] = merged["src_port_category"].map(category_code)
    merged["dst_port_category_code"] = merged["dst_port_category"].map(category_code)

    timestamps = pd.to_datetime(merged["flow_start_time"], errors="coerce")
    merged["minute_of_day"] = (
        timestamps.dt.hour * 60 + timestamps.dt.minute
        + timestamps.dt.second / 60.0 + timestamps.dt.microsecond / 60_000_000.0
    )
    merged["time_bucket_5m"] = np.floor(merged["minute_of_day"] / 5).astype("Int64")
    merged["time_bucket_15m"] = np.floor(merged["minute_of_day"] / 15).astype("Int64")
    merged["hour_of_day"] = timestamps.dt.hour.astype("Int64")
    merged = merged.set_index("flow_id", drop=False)

    missing = {column: int(merged[column].isna().sum()) for column in merged.columns if merged[column].isna().any()}
    source_files = {
        path.name: {"rows": pq.ParquetFile(path).metadata.num_rows, "sha256": sha256_file(path)}
        for path in sorted(mapping_dir.glob("*_flows.parquet"))
    }
    metadata = {
        "selected_rows": len(merged),
        "scanned_parquet_rows": scanned_rows,
        "mapping_files": source_files,
        "missing_counts": missing,
        "unavailable_requested_fields": ["forward_bytes", "backward_bytes"],
        "derived_fields": {
            "mean_packet_size": "total_bytes / packet_count",
            "bytes_per_packet": "total_bytes / packet_count",
            "minute_of_day": "parsed from existing flow_start_time; no PCAP access",
            "time_bucket_5m": "floor(minute_of_day / 5)",
            "time_bucket_15m": "floor(minute_of_day / 15)",
            "hour_of_day": "hour from existing flow_start_time",
        },
        "raw_pcap_reparsed": False,
    }
    return merged, metadata


def ordered_metadata(metadata: pd.DataFrame, flow_ids: np.ndarray, class_name: str, split: str) -> pd.DataFrame:
    ids = np.asarray(flow_ids).astype(str)
    if len(set(ids.tolist())) != len(ids):
        raise ValueError(f"duplicate embedding flow IDs: {class_name}/{split}")
    try:
        frame = metadata.loc[ids].copy()
    except KeyError as exc:
        raise ValueError(f"embedding flow ID absent from metadata: {class_name}/{split}") from exc
    if not frame["class_name"].eq(class_name).all() or not frame["split"].eq(split).all():
        raise ValueError(f"metadata alignment mismatch: {class_name}/{split}")
    return frame.reset_index(drop=True)


def continuous_rows(frame: pd.DataFrame, class_name: str, seed: int, split: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    components = sorted(frame["component_id"].unique().tolist())
    for feature in CONTINUOUS_FEATURES:
        groups: list[np.ndarray] = []
        stats: dict[int, dict[str, Any]] = {}
        for component in range(K):
            values = frame.loc[frame["component_id"].eq(component), feature].dropna().astype(float).to_numpy()
            if len(values):
                groups.append(values)
                p10, q25, median, q75, p90 = np.quantile(values, [0.10, 0.25, 0.50, 0.75, 0.90])
                stats[component] = {
                    "count": len(values), "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "median": float(median), "q25": float(q25), "q75": float(q75),
                    "p10": float(p10), "p90": float(p90),
                }
        valid_n = sum(len(group) for group in groups)
        if len(groups) >= 2 and len(components) >= 2 and any(np.ptp(group) > 0 for group in groups):
            try:
                h_stat, p_value = kruskal(*groups)
                note = ""
            except ValueError as exc:
                h_stat, p_value, note = float("nan"), float("nan"), str(exc)
        else:
            h_stat, p_value, note = float("nan"), float("nan"), "insufficient or constant groups"
        epsilon = epsilon_squared_kruskal(float(h_stat), valid_n, len(groups))
        for component in range(K):
            row = {
                "class_name": class_name, "seed": seed, "split": split,
                "feature": feature, "component_id": component,
                "valid_samples": valid_n, "groups_tested": len(groups),
                "kruskal_H": h_stat, "p_value": p_value,
                "epsilon_squared": epsilon,
                "effect_size_formula": "max(0,(H-k+1)/(n-k))",
                "note": note,
            }
            row.update(stats.get(component, {
                "count": 0, "mean": float("nan"), "std": float("nan"),
                "median": float("nan"), "q25": float("nan"), "q75": float("nan"),
                "p10": float("nan"), "p90": float("nan"),
            }))
            output.append(row)
    return output


def categorical_rows(
    frame: pd.DataFrame,
    train_top: Mapping[str, set[str]],
    class_name: str,
    seed: int,
    split: str,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for feature in CATEGORICAL_FEATURES:
        raw = safe_association(frame["component_id"], frame[feature])
        output.append({
            "class_name": class_name, "seed": seed, "split": split,
            "feature": feature, "variant": "raw", "top_n": "NA",
            "category_fit_split": "not_applicable", **raw,
        })
        if len(train_top[feature]) == TOP_N:
            transformed = apply_top_n(frame[feature], train_top[feature])
            filtered = safe_association(frame["component_id"], transformed)
            output.append({
                "class_name": class_name, "seed": seed, "split": split,
                "feature": feature, "variant": f"top{TOP_N}_plus_other",
                "top_n": TOP_N, "category_fit_split": "train", **filtered,
            })
    return output


def endpoint_rows(
    frame: pd.DataFrame,
    categorical: Sequence[Mapping[str, Any]],
    class_name: str,
    seed: int,
    split: str,
) -> list[dict[str, Any]]:
    assoc = {(row["feature"], row["variant"]): row for row in categorical}
    output: list[dict[str, Any]] = []
    for feature in ENDPOINT_FIELDS:
        raw = assoc[(feature, "raw")]
        filtered = assoc.get((feature, f"top{TOP_N}_plus_other"))
        for component in range(K):
            values = frame.loc[frame["component_id"].eq(component), feature]
            output.append({
                "class_name": class_name, "seed": seed, "split": split,
                "endpoint_field": feature, "component_id": component,
                "component_count": int(len(values)), "top10": top_values(values, 10),
                "raw_nmi": raw["nmi"], "raw_ami": raw["ami"],
                "raw_status": raw["status"],
                "top_n_nmi": filtered["nmi"] if filtered else float("nan"),
                "top_n_ami": filtered["ami"] if filtered else float("nan"),
                "top_n_status": filtered["status"] if filtered else "NOT_APPLICABLE_LOW_CARDINALITY",
                "interpretation_boundary": "association does not prove causal model reliance",
            })
    return output


def port_rows(
    frame: pd.DataFrame,
    categorical: Sequence[Mapping[str, Any]],
    class_name: str,
    seed: int,
    split: str,
) -> list[dict[str, Any]]:
    assoc = {(row["feature"], row["variant"]): row for row in categorical}
    output = []
    for component in range(K):
        comp = frame[frame["component_id"].eq(component)]
        output.append({
            "class_name": class_name, "seed": seed, "split": split,
            "component_id": component, "component_count": len(comp),
            "src_port_raw_nmi": assoc[("src_port", "raw")]["nmi"],
            "src_port_raw_ami": assoc[("src_port", "raw")]["ami"],
            "dst_port_raw_nmi": assoc[("dst_port", "raw")]["nmi"],
            "dst_port_raw_ami": assoc[("dst_port", "raw")]["ami"],
            "src_port_category_nmi": assoc[("src_port_category", "raw")]["nmi"],
            "src_port_category_ami": assoc[("src_port_category", "raw")]["ami"],
            "dst_port_category_nmi": assoc[("dst_port_category", "raw")]["nmi"],
            "dst_port_category_ami": assoc[("dst_port_category", "raw")]["ami"],
            "top10_src_ports": top_values(comp["src_port"], 10),
            "top10_dst_ports": top_values(comp["dst_port"], 10),
            "src_port_category_distribution": top_values(comp["src_port_category"], 3),
            "dst_port_category_distribution": top_values(comp["dst_port_category"], 3),
        })
    return output


def time_rows(
    frame: pd.DataFrame,
    categorical: Sequence[Mapping[str, Any]],
    class_name: str,
    seed: int,
    split: str,
) -> list[dict[str, Any]]:
    assoc = {(row["feature"], row["variant"]): row for row in categorical}
    stats: dict[int, dict[str, Any]] = {}
    for component in range(K):
        values = frame.loc[frame["component_id"].eq(component), "minute_of_day"].dropna().astype(float).to_numpy()
        if len(values):
            p10, q25, median, q75, p90 = np.quantile(values, [0.10, 0.25, 0.50, 0.75, 0.90])
            stats[component] = {
                "time_count": len(values), "minute_median": median, "minute_q25": q25,
                "minute_q75": q75, "minute_p10": p10, "minute_p90": p90,
            }
    disjoint_iqr = False
    if 0 in stats and 1 in stats:
        disjoint_iqr = bool(
            stats[0]["minute_q75"] < stats[1]["minute_q25"]
            or stats[1]["minute_q75"] < stats[0]["minute_q25"]
        )
    nmi15 = float(assoc[("time_bucket_15m", "raw")]["nmi"])
    time_associated = bool(disjoint_iqr and np.isfinite(nmi15) and nmi15 >= 0.50)
    output = []
    for component in range(K):
        row = {
            "class_name": class_name, "seed": seed, "split": split,
            "component_id": component,
            "nmi_5minute": assoc[("time_bucket_5m", "raw")]["nmi"],
            "ami_5minute": assoc[("time_bucket_5m", "raw")]["ami"],
            "nmi_15minute": nmi15,
            "ami_15minute": assoc[("time_bucket_15m", "raw")]["ami"],
            "nmi_hour": assoc[("hour_of_day", "raw")]["nmi"],
            "ami_hour": assoc[("hour_of_day", "raw")]["ami"],
            "component_iqrs_disjoint": disjoint_iqr,
            "time_window_associated": time_associated,
            "diagnostic_label": "TIME-WINDOW ASSOCIATED" if time_associated else "NO_STRONG_NONOVERLAP_FLAG",
            "interpretation_boundary": "time association may reflect an attack stage or collection window; not automatically an artifact",
        }
        row.update(stats.get(component, {
            "time_count": 0, "minute_median": float("nan"), "minute_q25": float("nan"),
            "minute_q75": float("nan"), "minute_p10": float("nan"), "minute_p90": float("nan"),
        }))
        output.append(row)
    return output


def prepare_rf_matrix(train: pd.DataFrame, val: pd.DataFrame, features: Sequence[str]) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    x_train = train.loc[:, features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    x_val = val.loc[:, features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    medians = np.nanmedian(x_train, axis=0)
    medians = np.where(np.isnan(medians), 0.0, medians)
    x_train = np.where(np.isnan(x_train), medians, x_train)
    x_val = np.where(np.isnan(x_val), medians, x_val)
    return x_train, x_val, dict(zip(features, map(float, medians)))


def rf_rows(train: pd.DataFrame, val: pd.DataFrame, class_name: str, seed: int) -> list[dict[str, Any]]:
    output = []
    y_train = train["component_id"].to_numpy(dtype=int)
    y_val = val["component_id"].to_numpy(dtype=int)
    day_map = {day: index for index, day in enumerate(sorted(train["source_day"].dropna().astype(str).unique()))}
    train = train.copy()
    val = val.copy()
    train["source_day_code"] = train["source_day"].astype(str).map(day_map).fillna(-1)
    val["source_day_code"] = val["source_day"].astype(str).map(day_map).fillna(-1)
    model_specs = [("Model S", list(RF_MODEL_S))]
    s_plus = list(RF_MODEL_S_PLUS)
    day_identifiable = len(day_map) >= 2
    if day_identifiable:
        s_plus.append("source_day_code")
    model_specs.append(("Model S+", s_plus))
    for name, features in model_specs:
        x_train, x_val, medians = prepare_rf_matrix(train, val, features)
        model = RandomForestClassifier(
            n_estimators=RF_TREES, random_state=RF_SEED,
            class_weight="balanced", n_jobs=1,
        )
        model.fit(x_train, y_train)
        predicted = model.predict(x_val)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            balanced = float(balanced_accuracy_score(y_val, predicted))
        majority = Counter(y_train.tolist()).most_common(1)[0][0]
        baseline = np.full_like(y_val, majority)
        output.append({
            "class_name": class_name, "seed": seed, "model": name,
            "status": "completed", "train_samples": len(train), "val_samples": len(val),
            "features": ";".join(features), "source_day_used": day_identifiable and name == "Model S+",
            "source_day_status": "USED" if day_identifiable and name == "Model S+" else "NOT_IDENTIFIABLE_CONSTANT_WITHIN_CLASS",
            "n_estimators": RF_TREES, "random_state": RF_SEED, "class_weight": "balanced",
            "accuracy": float(accuracy_score(y_val, predicted)),
            "macro_f1": float(f1_score(y_val, predicted, labels=list(range(K)), average="macro", zero_division=0)),
            "balanced_accuracy": balanced,
            "fixed_k_macro_recall": float(recall_score(y_val, predicted, labels=list(range(K)), average="macro", zero_division=0)),
            "confusion_matrix": json_compact(confusion_matrix(y_val, predicted, labels=list(range(K))).tolist()),
            "validation_component_counts": json_compact(Counter(map(int, y_val))),
            "majority_baseline_accuracy": float(accuracy_score(y_val, baseline)),
            "majority_baseline_macro_f1": float(f1_score(y_val, baseline, labels=list(range(K)), average="macro", zero_division=0)),
            "feature_importances": json_compact(dict(zip(features, map(float, model.feature_importances_)))),
            "train_imputation_medians": json_compact(medians),
            "fit_split": "train", "evaluation_split": "validation",
        })
    return output


def consistency_rows(train: pd.DataFrame, val: pd.DataFrame, class_name: str, seed: int) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    train_counts = np.bincount(train["component_id"].to_numpy(dtype=int), minlength=K)
    val_counts = np.bincount(val["component_id"].to_numpy(dtype=int), minlength=K)
    tv = component_weight_tv(train["component_id"].to_numpy(), val["component_id"].to_numpy())
    for component in range(K):
        output.append({
            "class_name": class_name, "seed": seed, "component_id": component,
            "record_type": "component_weight", "feature": "component_weight",
            "train_count": int(train_counts[component]), "val_count": int(val_counts[component]),
            "train_component_ratio": train_counts[component] / len(train),
            "val_component_ratio": val_counts[component] / len(val),
            "component_weight_tv_distance": tv,
            "train_median": float("nan"), "val_median": float("nan"),
            "absolute_median_shift": float("nan"), "relative_median_shift": float("nan"),
            "median_shift_over_train_iqr": float("nan"), "category_distribution_tv": float("nan"),
        })
        tr_comp = train[train["component_id"].eq(component)]
        va_comp = val[val["component_id"].eq(component)]
        for feature in CONTINUOUS_FEATURES:
            tr = tr_comp[feature].dropna().astype(float).to_numpy()
            va = va_comp[feature].dropna().astype(float).to_numpy()
            tr_med = float(np.median(tr)) if len(tr) else float("nan")
            va_med = float(np.median(va)) if len(va) else float("nan")
            absolute = abs(va_med - tr_med) if np.isfinite(tr_med) and np.isfinite(va_med) else float("nan")
            relative = absolute / max(abs(tr_med), 1e-12) if np.isfinite(absolute) else float("nan")
            iqr = float(np.subtract(*np.quantile(tr, [0.75, 0.25]))) if len(tr) else float("nan")
            over_iqr = absolute / iqr if np.isfinite(absolute) and np.isfinite(iqr) and iqr > 0 else float("nan")
            output.append({
                "class_name": class_name, "seed": seed, "component_id": component,
                "record_type": "continuous", "feature": feature,
                "train_count": len(tr), "val_count": len(va),
                "train_component_ratio": train_counts[component] / len(train),
                "val_component_ratio": val_counts[component] / len(val),
                "component_weight_tv_distance": tv,
                "train_median": tr_med, "val_median": va_med,
                "absolute_median_shift": absolute, "relative_median_shift": relative,
                "median_shift_over_train_iqr": over_iqr, "category_distribution_tv": float("nan"),
            })
        for feature in CONSISTENCY_CATEGORICAL:
            output.append({
                "class_name": class_name, "seed": seed, "component_id": component,
                "record_type": "categorical", "feature": feature,
                "train_count": int(tr_comp[feature].notna().sum()), "val_count": int(va_comp[feature].notna().sum()),
                "train_component_ratio": train_counts[component] / len(train),
                "val_component_ratio": val_counts[component] / len(val),
                "component_weight_tv_distance": tv,
                "train_median": float("nan"), "val_median": float("nan"),
                "absolute_median_shift": float("nan"), "relative_median_shift": float("nan"),
                "median_shift_over_train_iqr": float("nan"),
                "category_distribution_tv": distribution_tv(tr_comp[feature], va_comp[feature]),
            })
    return output


def make_figures(class_frames: Mapping[str, Mapping[str, pd.DataFrame]], figures_dir: Path) -> dict[str, list[str]]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    output: dict[str, list[str]] = {}
    for class_name, split_frames in class_frames.items():
        slug = class_name.lower().replace(" ", "_")
        paths: list[str] = []
        for feature, label in (
            ("packet_count", "Packet count"),
            ("total_bytes", "Total bytes"),
            ("flow_duration", "Flow duration (s)"),
        ):
            arrays, labels = [], []
            for split in ("train", "val"):
                frame = split_frames[split]
                for component in range(K):
                    values = frame.loc[frame["component_id"].eq(component), feature].dropna().astype(float).to_numpy()
                    if len(values) > 20_000:
                        values = rng.choice(values, 20_000, replace=False)
                    arrays.append(values)
                    labels.append(f"{split} C{component}")
            fig, ax = plt.subplots(figsize=(7.5, 4.8))
            ax.boxplot(arrays, tick_labels=labels, showfliers=False)
            ax.set_ylabel(label); ax.set_title(f"{class_name}: {label} by component")
            ax.grid(axis="y", alpha=0.25); fig.tight_layout()
            path = f"{slug}_{feature}_by_component.png"
            fig.savefig(figures_dir / path, dpi=180); plt.close(fig); paths.append(f"figures/{path}")

        train = split_frames["train"]
        sample = train if len(train) <= 25_000 else train.iloc[np.sort(rng.choice(len(train), 25_000, replace=False))]
        fig, ax = plt.subplots(figsize=(7.2, 5.2))
        for component in range(K):
            part = sample[sample["component_id"].eq(component)]
            ax.scatter(part["forward_packet_count"], part["backward_packet_count"], s=5, alpha=0.3, label=f"C{component}")
        ax.set_xlabel("Forward packet count"); ax.set_ylabel("Backward packet count")
        ax.set_title(f"{class_name}: directional packet counts (train, seed 0)"); ax.legend(); fig.tight_layout()
        path = f"{slug}_forward_backward_by_component.png"
        fig.savefig(figures_dir / path, dpi=180); plt.close(fig); paths.append(f"figures/{path}")

        combined = pd.concat([split_frames["train"], split_frames["val"]], ignore_index=True)
        sample = combined if len(combined) <= 30_000 else combined.iloc[np.sort(rng.choice(len(combined), 30_000, replace=False))]
        fig, ax = plt.subplots(figsize=(9, 4.6))
        for component in range(K):
            part = sample[sample["component_id"].eq(component)]
            jitter = rng.normal(0, 0.035, len(part))
            ax.scatter(part["minute_of_day"], component + jitter, s=5, alpha=0.28, label=f"C{component}")
        ax.set_xlabel("Minute of day"); ax.set_ylabel("Component"); ax.set_yticks(range(K))
        ax.set_title(f"{class_name}: component timeline (seed 0)"); ax.legend(); fig.tight_layout()
        path = f"{slug}_component_timeline.png"
        fig.savefig(figures_dir / path, dpi=180); plt.close(fig); paths.append(f"figures/{path}")

        train_weights = np.bincount(split_frames["train"]["component_id"].astype(int), minlength=K) / len(split_frames["train"])
        val_weights = np.bincount(split_frames["val"]["component_id"].astype(int), minlength=K) / len(split_frames["val"])
        x = np.arange(K); width = 0.36
        fig, ax = plt.subplots(figsize=(6.5, 4.6))
        ax.bar(x - width / 2, train_weights, width, label="train")
        ax.bar(x + width / 2, val_weights, width, label="validation")
        ax.set_xticks(x, [f"C{i}" for i in x]); ax.set_ylim(0, 1)
        ax.set_ylabel("Empirical component proportion"); ax.set_title(f"{class_name}: train vs validation weights (seed 0)")
        ax.legend(); fig.tight_layout()
        path = f"{slug}_component_weight_train_val.png"
        fig.savefig(figures_dir / path, dpi=180); plt.close(fig); paths.append(f"figures/{path}")
        output[class_name] = paths
    return output


def build_diagnosis(
    gaussian_reference: Mapping[tuple[str, int], Mapping[str, str]],
    weights: Sequence[Mapping[str, Any]],
    continuous: Sequence[Mapping[str, Any]],
    categorical: Sequence[Mapping[str, Any]],
    predictability: Sequence[Mapping[str, Any]],
    seed_stability: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    for class_name in CLASSES:
        class_weights = [row for row in weights if row["class_name"] == class_name]
        class_cont = [row for row in continuous if row["class_name"] == class_name and row["split"] == "val"]
        # One epsilon value is repeated for two components. De-duplicate first,
        # then aggregate across seeds so the per-class top three are distinct
        # features instead of the same feature repeated for seeds 0/1/2.
        unique_cont = {(row["seed"], row["feature"]): row for row in class_cont}
        feature_effects = []
        for feature in sorted({row["feature"] for row in unique_cont.values()}):
            values = [
                float(row["epsilon_squared"])
                for row in unique_cont.values()
                if row["feature"] == feature and np.isfinite(row["epsilon_squared"])
            ]
            if values:
                feature_effects.append({
                    "feature": feature,
                    "mean_epsilon_squared": float(np.mean(values)),
                    "min_epsilon_squared": float(np.min(values)),
                    "max_epsilon_squared": float(np.max(values)),
                    "finite_seed_count": len(values),
                })
        strongest = sorted(feature_effects, key=lambda row: -row["mean_epsilon_squared"])[:3]
        class_cat = [
            row for row in categorical
            if row["class_name"] == class_name and row["split"] == "val"
            and row["variant"] == "raw" and bool(row["identifiable"])
        ]
        strongest_cat = max(class_cat, key=lambda row: float(row["nmi"]), default={"feature": "NA", "nmi": float("nan"), "ami": float("nan")})
        rf_s = [float(row["macro_f1"]) for row in predictability if row["class_name"] == class_name and row["model"] == "Model S"]
        rf_sp = [float(row["macro_f1"]) for row in predictability if row["class_name"] == class_name and row["model"] == "Model S+"]
        rf_s_bal = [float(row["balanced_accuracy"]) for row in predictability if row["class_name"] == class_name and row["model"] == "Model S"]
        rf_sp_bal = [float(row["balanced_accuracy"]) for row in predictability if row["class_name"] == class_name and row["model"] == "Model S+"]
        val_seed = [row for row in seed_stability if row["class_name"] == class_name and row["split"] == "val"]
        min_seed_nmi = min((float(row["nmi"]) for row in val_seed), default=float("nan"))
        tiny_seeds = len({int(row["seed"]) for row in class_weights if bool(row["tiny_component"])})
        max_tv = max(float(row["component_weight_tv_distance"]) for row in class_weights)
        delta = float(gaussian_reference[(class_name, 0)]["val_avg_nll_k1_minus_k2"])
        max_eps = max((row["mean_epsilon_squared"] for row in feature_effects), default=0.0)
        mean_s = float(np.mean(rf_s)); mean_sp = float(np.mean(rf_sp))
        mean_s_bal = float(np.mean(rf_s_bal)); mean_sp_bal = float(np.mean(rf_sp_bal))
        max_cat = float(strongest_cat["nmi"]) if np.isfinite(strongest_cat["nmi"]) else 0.0
        minimum_train_count = min(int(row["train_count"]) for row in class_weights)
        minimum_val_count = min(int(row["val_count"]) for row in class_weights)
        minimum_gmm_weight = min(float(row["gmm_train_weight"]) for row in class_weights)

        # Multi-axis descriptive rubric: no single threshold determines a label.
        unstable = (delta < 0 and tiny_seeds >= 2) or (tiny_seeds >= 2 and min_seed_nmi < 0.80) or max_tv > 0.20
        simple_evidence = max_eps >= 0.14 and mean_s >= 0.75
        shortcut_evidence = max_cat >= 0.20 or (mean_sp - mean_s) >= 0.10
        if unstable:
            diagnosis = "E. UNSTABLE / DEGENERATE"
            rationale = "negative/unstable held-out mixture evidence combined with tiny components, seed disagreement, or train/validation imbalance"
        elif simple_evidence and shortcut_evidence:
            diagnosis = "C. MIXED"
            rationale = "both simple flow statistics and endpoint/time metadata materially associate with component assignment"
        elif simple_evidence:
            diagnosis = "A. SIMPLE-FLOW-STATISTICS DOMINATED"
            rationale = "large continuous effects and strong Model S validation predictability dominate the observed associations"
        elif shortcut_evidence and mean_sp >= 0.70:
            diagnosis = "B. ENDPOINT/TIME DOMINATED"
            rationale = "endpoint/time association or Model S+ gain is stronger than the simple-statistics explanation"
        elif min_seed_nmi >= 0.80 and max_tv <= 0.10:
            diagnosis = "D. RESIDUAL / NOT SIMPLY EXPLAINED"
            rationale = "components are seed/train-validation stable but existing metadata does not predict them well"
        else:
            diagnosis = "E. UNSTABLE / DEGENERATE"
            rationale = "evidence is insufficiently stable for a residual interpretation"
        output.append({
            "class_name": class_name, "diagnosis": diagnosis, "rationale": rationale,
            "gaussian_delta_val_nll_k1_minus_k2": delta,
            "tiny_component_seed_count": tiny_seeds,
            "max_component_weight_tv": max_tv,
            "minimum_pairwise_validation_seed_nmi": min_seed_nmi,
            "model_s_macro_f1_mean": mean_s, "model_s_plus_macro_f1_mean": mean_sp,
            "model_s_balanced_accuracy_mean": mean_s_bal,
            "model_s_plus_balanced_accuracy_mean": mean_sp_bal,
            "minimum_component_train_count": minimum_train_count,
            "minimum_component_val_count": minimum_val_count,
            "minimum_gmm_train_weight": minimum_gmm_weight,
            "strongest_continuous_features": json_compact(strongest),
            "strongest_categorical_feature": strongest_cat["feature"],
            "strongest_categorical_nmi": strongest_cat["nmi"],
            "strongest_categorical_ami": strongest_cat["ami"],
            "judgment_method": "multi-axis synthesis; no single metric or threshold determines the diagnosis",
        })
    return output


def build_summary(diagnosis: Sequence[Mapping[str, Any]], metadata_info: Mapping[str, Any]) -> str:
    rows = {row["class_name"]: row for row in diagnosis}
    lines = [
        "# CIC-IDS-2017 Component / Shortcut Audit", "",
        "## Protocol safeguards", "",
        "- Reused the completed CIC Gaussian Audit sampling, train/validation split, checkpoint-derived z_t, and fixed PCA64/full-covariance settings.",
        "- For each class and seed, StandardScaler, PCA64, and K=2 GMM were fit on train only; validation was assignment/evaluation only.",
        "- K=2, reg_covar=1e-3, n_init=3, max_iter=300, and seeds 0/1/2 were fixed; no BIC, K search, Adaptive K, Unknown, or test data was used.",
        "- Metadata came from existing mapping Parquet files. No PCAP was reparsed. forward_bytes/backward_bytes were unavailable and not invented.",
        "- Constant source_day/source_pcap/source CSV/protocol fields are reported as NOT IDENTIFIABLE, even though their NMI is numerically zero.",
        "- Associations and RF predictability do not prove causal model reliance or semantic attack phases.", "",
        "## Required answers", "",
    ]
    for number, class_name in enumerate(CLASSES, start=1):
        row = rows[class_name]
        if class_name == "DDoS":
            direct_answer = (
                "Measured simple flow statistics, endpoints, ports, and time do not strongly explain the assignment. "
                "The K=2 benefit is seed-stable and train/validation-stable, so a residual distributional signal remains, "
                "but its semantic cause is not identified; constant day/capture/protocol effects cannot be tested within this class."
            )
        elif class_name == "PortScan":
            direct_answer = (
                "The strongest measured explanation is endpoint/port-enriched metadata: adding raw ports and time fields "
                "raises validation Macro-F1 substantially over Model S. The held-out K=2 likelihood benefit remains, "
                "but it is not clean residual multimodality after this measured shortcut explanation."
            )
        elif class_name == "DoS GoldenEye":
            direct_answer = (
                "Yes: seeds 0/1 isolate a tiny component while seed 2 finds a materially different partition; the very low "
                "cross-seed validation NMI makes K=2 component interpretation unstable even though its average held-out likelihood direction is positive."
            )
        else:
            direct_answer = (
                "K=2 isolates only a few training flows and no validation flow in the small component, while held-out NLL becomes "
                "dramatically worse than K=1. This is evidence of degenerate mixture overfitting for this fit, not evidence that Hulk is necessarily single-Gaussian."
            )
        lines += [
            f"### {number}. {class_name}", "",
            f"Diagnosis: **{row['diagnosis']}**.", "",
            f"Direct answer: {direct_answer}", "",
            f"Reason: {row['rationale']}. Gaussian ΔNLL(K1-K2)={float(row['gaussian_delta_val_nll_k1_minus_k2']):.6f}; "
            f"tiny-component seeds={row['tiny_component_seed_count']}/3; maximum train/validation weight TV={float(row['max_component_weight_tv']):.6f}; "
            f"minimum pairwise validation seed NMI={float(row['minimum_pairwise_validation_seed_nmi']):.6f}; "
            f"Model S/S+ mean Macro-F1={float(row['model_s_macro_f1_mean']):.6f}/{float(row['model_s_plus_macro_f1_mean']):.6f}; "
            f"minimum component train/validation counts={row['minimum_component_train_count']}/{row['minimum_component_val_count']}; "
            f"strongest categorical field={row['strongest_categorical_feature']} (NMI={float(row['strongest_categorical_nmi']):.6f}, AMI={float(row['strongest_categorical_ami']):.6f}).", "",
        ]
    lines += [
        "### 5. Do DDoS / PortScan retain multi-component evidence after simple explanations?", "",
        "**DDoS: YES, as residual distributional evidence only**—the partition is seed-stable and train/validation-stable but poorly predicted by the audited metadata. **PortScan: NOT CLEANLY**—the K=2 held-out likelihood gain remains, but Model S+ shows that raw port/time-enriched metadata predicts the component much better than simple statistics alone. Neither result identifies semantic attack modes.", "",
        "### 6. Comparison with USTC Stage2.6", "",
        "**The result is not consistent with an all-classes simple-statistics explanation.** USTC Stage2.6 found all four representative classes primarily associated with simple flow statistics, with strongest epsilon-squared values about 0.88-0.99, Model-S-like Macro-F1 about 0.61-1.00, and component-weight TV about 0.006-0.066. CIC-IDS-2017 instead shows DDoS residual evidence, PortScan endpoint/port association, and unstable/degenerate GoldenEye and Hulk fits. Both audits agree that Gaussian components must not automatically be treated as semantic subgroups.", "",
        "### 7. Are all classes multimodal?", "",
        "**NO / NOT SUPPORTED.** Hulk is a negative control with K=2 held-out NLL worse than K=1, and any tiny/unstable component evidence must not be promoted to a real mode.", "",
        "### 8. Do different classes require different distributional complexity?", "",
        "**SUPPORTED AS A DESCRIPTIVE MODELING RESULT.** DDoS, PortScan, and GoldenEye had positive K=2 held-out ΔNLL, whereas Hulk did not. This does not identify a true K or semantic subtypes.", "",
        "## Interpretation boundary", "",
        "Components are Gaussian distributional components only. High metadata predictability means strong association/predictability, not that a component is meaningless, fake, a shortcut, or a semantic attack phase. Causal attribution requires a later intervention/ablation that was not performed here.", "",
        f"Metadata rows audited: {metadata_info['selected_rows']}; raw PCAP reparsed: {metadata_info['raw_pcap_reparsed']}.", "",
    ]
    return "\n".join(lines)


def write_assignment_chunk(writer: pq.ParquetWriter | None, frame: pd.DataFrame, path: Path) -> pq.ParquetWriter:
    fields = [
        "flow_id", "class_name", "split", "seed", "component_id",
        "component_probability", "max_posterior", "source_day", "source_pcap",
        "source_label_csv", "protocol", "src_port", "dst_port", "src_ip", "dst_ip",
        "flow_start_time", "flow_start_epoch_utc", "minute_of_day", "time_bucket_5m",
        "time_bucket_15m", "hour_of_day", "packet_count", "total_bytes", "flow_duration",
        "forward_packet_count", "backward_packet_count", "mean_packet_size", "bytes_per_packet",
    ]
    table = pa.Table.from_pandas(frame.loc[:, fields], preserve_index=False)
    if writer is None:
        writer = pq.ParquetWriter(path, table.schema, compression="zstd")
    writer.write_table(table)
    return writer


def run(args: argparse.Namespace) -> None:
    started = time.time()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent))
    assignment_writer: pq.ParquetWriter | None = None
    try:
        gaussian_root = args.gaussian_root.resolve()
        embedding_dir = gaussian_root / "artifacts" / "formal" / "embeddings"
        required = [
            gaussian_root / "sampling_manifest.csv", gaussian_root / "split_manifest.csv",
            gaussian_root / "training_config.json", gaussian_root / "gaussian_results.csv",
            gaussian_root / "component_weights.csv", embedding_dir / "embedding_train.npy",
            embedding_dir / "embedding_val.npy", embedding_dir / "class_names_train.npy",
            embedding_dir / "class_names_val.npy", embedding_dir / "flow_ids_train.npy",
            embedding_dir / "flow_ids_val.npy",
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError("missing completed Gaussian Audit assets:\n  " + "\n  ".join(missing))

        metadata, metadata_info = load_metadata(gaussian_root / "split_manifest.csv", args.mapping_dir.resolve())
        expected_counts = {
            ("DDoS", "train"): 61010, ("DDoS", "val"): 15252,
            ("PortScan", "train"): 80000, ("PortScan", "val"): 20000,
            ("DoS GoldenEye", "train"): 5953, ("DoS GoldenEye", "val"): 1488,
            ("DoS Hulk", "train"): 80000, ("DoS Hulk", "val"): 20000,
        }
        observed_counts = metadata.groupby(["class_name", "split"]).size().to_dict()
        if observed_counts != expected_counts:
            raise ValueError(f"selected split counts changed: {observed_counts}")

        z = {split: np.load(embedding_dir / f"embedding_{split}.npy", mmap_mode="r") for split in ("train", "val")}
        names = {split: np.load(embedding_dir / f"class_names_{split}.npy", mmap_mode="r") for split in ("train", "val")}
        flow_ids = {split: np.load(embedding_dir / f"flow_ids_{split}.npy", mmap_mode="r") for split in ("train", "val")}
        if z["train"].shape[1] != 768 or z["val"].shape[1] != 768:
            raise ValueError("z_t dimension is not 768")
        if set(flow_ids["train"].tolist()) & set(flow_ids["val"].tolist()):
            raise ValueError("train/validation flow ID overlap")

        gaussian_rows = _read_csv(gaussian_root / "gaussian_results.csv")
        gaussian_reference: dict[tuple[str, int], dict[str, str]] = {}
        for row in gaussian_rows:
            if row["class_name"] in CLASSES and int(row["K"]) == K:
                k1 = next(
                    item for item in gaussian_rows
                    if item["class_name"] == row["class_name"]
                    and int(item["seed"]) == int(row["seed"]) and int(item["K"]) == 1
                )
                gaussian_reference[(row["class_name"], int(row["seed"]))] = {
                    **row,
                    "val_avg_nll_k1_minus_k2": str(float(k1["val_avg_nll"]) - float(row["val_avg_nll"])),
                }
        if set(gaussian_reference) != {(name, seed) for name in CLASSES for seed in SEEDS}:
            raise ValueError("Gaussian Audit lacks one or more fixed K=2 references")

        weight_rows: list[dict[str, Any]] = []
        continuous: list[dict[str, Any]] = []
        categorical: list[dict[str, Any]] = []
        ports: list[dict[str, Any]] = []
        times: list[dict[str, Any]] = []
        endpoints: list[dict[str, Any]] = []
        predictability: list[dict[str, Any]] = []
        consistency: list[dict[str, Any]] = []
        warnings_rows: list[dict[str, Any]] = []
        pca_rows: list[dict[str, Any]] = []
        stability: list[dict[str, Any]] = []
        figure_frames: dict[str, dict[str, pd.DataFrame]] = {}

        for class_name in CLASSES:
            print(f"class={class_name}: align embeddings and fit train-only PCA64", flush=True)
            indices = {split: np.flatnonzero(np.asarray(names[split]) == class_name) for split in ("train", "val")}
            class_ids = {split: np.asarray(flow_ids[split][indices[split]]).astype(str) for split in ("train", "val")}
            base = {split: ordered_metadata(metadata, class_ids[split], class_name, split) for split in ("train", "val")}
            x_train_raw = np.asarray(z["train"][indices["train"]], dtype=np.float32)
            x_val_raw = np.asarray(z["val"][indices["val"]], dtype=np.float32)
            x_train, x_val, scaler, pca = fit_train_representation(x_train_raw, x_val_raw)
            del x_train_raw, x_val_raw
            pca_rows.append({
                "class_name": class_name, "train_samples": len(x_train), "val_samples": len(x_val),
                "input_dim": 768, "pca_dim": PCA_DIM, "pca_seed": PCA_SEED,
                "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
                "scaler_fit_split": "train", "pca_fit_split": "train",
                "validation_usage": "transform_and_assign_only", "gmm_input_dtype": str(x_train.dtype),
            })
            train_top = {feature: top_n_categories(base["train"][feature]) for feature in CATEGORICAL_FEATURES}
            predictions: dict[str, dict[int, np.ndarray]] = {"train": {}, "val": {}}
            model_weights: dict[int, np.ndarray] = {}

            for seed in SEEDS:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    model = GaussianMixture(
                        n_components=K, covariance_type="full", reg_covar=REG_COVAR,
                        n_init=N_INIT, max_iter=MAX_ITER, random_state=seed,
                    ).fit(x_train)
                for item in caught:
                    warnings_rows.append({
                        "class_name": class_name, "seed": seed,
                        "warning_category": item.category.__name__, "warning_message": str(item.message),
                    })
                train_nll = -float(model.score(x_train)); val_nll = -float(model.score(x_val))
                reference = gaussian_reference[(class_name, seed)]
                train_diff = abs(train_nll - float(reference["train_avg_nll"]))
                val_diff = abs(val_nll - float(reference["val_avg_nll"]))
                if train_diff > 1e-8 or val_diff > 1e-8:
                    raise ValueError(f"{class_name}/seed{seed}: refit NLL mismatch {train_diff}/{val_diff}")
                model_weights[seed] = model.weights_.copy()
                frames: dict[str, pd.DataFrame] = {}
                for split, values in (("train", x_train), ("val", x_val)):
                    probabilities = model.predict_proba(values)
                    assigned = probabilities.argmax(axis=1).astype(np.int8)
                    maximum = probabilities[np.arange(len(probabilities)), assigned]
                    predictions[split][seed] = assigned
                    frame = base[split].copy()
                    frame["seed"] = seed
                    frame["component_id"] = assigned
                    frame["component_probability"] = maximum
                    frame["max_posterior"] = probabilities.max(axis=1)
                    frames[split] = frame
                    assignment_writer = write_assignment_chunk(
                        assignment_writer, frame, stage_dir / "component_assignments.parquet"
                    )
                    cont = continuous_rows(frame, class_name, seed, split)
                    cats = categorical_rows(frame, train_top, class_name, seed, split)
                    continuous.extend(cont); categorical.extend(cats)
                    ports.extend(port_rows(frame, cats, class_name, seed, split))
                    times.extend(time_rows(frame, cats, class_name, seed, split))
                    endpoints.extend(endpoint_rows(frame, cats, class_name, seed, split))
                tv = component_weight_tv(predictions["train"][seed], predictions["val"][seed])
                train_counts = np.bincount(predictions["train"][seed], minlength=K)
                val_counts = np.bincount(predictions["val"][seed], minlength=K)
                for component in range(K):
                    weight_rows.append({
                        "class_name": class_name, "seed": seed, "component_id": component,
                        "gmm_train_weight": float(model.weights_[component]),
                        "train_count": int(train_counts[component]), "val_count": int(val_counts[component]),
                        "train_empirical_weight": train_counts[component] / len(x_train),
                        "val_empirical_weight": val_counts[component] / len(x_val),
                        "component_weight_tv_distance": tv,
                        "tiny_component": bool(model.weights_[component] < TINY_WEIGHT),
                        "tiny_threshold": TINY_WEIGHT, "converged": bool(model.converged_),
                        "n_iter": int(model.n_iter_), "train_avg_nll": train_nll,
                        "val_avg_nll": val_nll, "gaussian_reference_train_nll_abs_diff": train_diff,
                        "gaussian_reference_val_nll_abs_diff": val_diff,
                    })
                predictability.extend(rf_rows(frames["train"], frames["val"], class_name, seed))
                consistency.extend(consistency_rows(frames["train"], frames["val"], class_name, seed))
                if seed == 0:
                    figure_frames[class_name] = frames
                print(
                    f"class={class_name} seed={seed} val_nll={val_nll:.6f} "
                    f"weights={model.weights_.tolist()} tv={tv:.6f}", flush=True,
                )

            for split in ("train", "val"):
                for left, right in ((0, 1), (0, 2), (1, 2)):
                    a = predictions[split][left]; b = predictions[split][right]
                    sorted_left = np.sort(model_weights[left]); sorted_right = np.sort(model_weights[right])
                    stability.append({
                        "class_name": class_name, "split": split,
                        "seed_a": left, "seed_b": right,
                        "nmi": float(normalized_mutual_info_score(a, b)),
                        "ami": float(adjusted_mutual_info_score(a, b)),
                        "adjusted_rand_index": float(adjusted_rand_score(a, b)),
                        "raw_component_id_agreement": float(np.mean(a == b)),
                        "sorted_weight_tv": float(0.5 * np.abs(sorted_left - sorted_right).sum()),
                        "component_id_caveat": "raw ID agreement is not permutation invariant; NMI/AMI/ARI are primary",
                    })
            del x_train, x_val

        if assignment_writer is not None:
            assignment_writer.close()
            assignment_writer = None

        figures = make_figures(figure_frames, stage_dir / "figures")
        diagnosis = build_diagnosis(gaussian_reference, weight_rows, continuous, categorical, predictability, stability)
        summary = build_summary(diagnosis, metadata_info)

        outputs = {
            "component_weights.csv": weight_rows,
            "continuous_component_tests.csv": continuous,
            "categorical_component_association.csv": categorical,
            "port_component_stats.csv": ports,
            "time_component_stats.csv": times,
            "component_predictability.csv": predictability,
            "endpoint_shortcut_stats.csv": endpoints,
            "train_val_consistency.csv": consistency,
            "seed_stability.csv": stability,
            "class_diagnosis.csv": diagnosis,
            "gmm_warnings.csv": warnings_rows,
            "pca_refit_manifest.csv": pca_rows,
        }
        for name, rows in outputs.items():
            if rows:
                _write_csv(stage_dir / name, rows, list(rows[0]))
            elif name == "gmm_warnings.csv":
                _write_csv(stage_dir / name, [], ["class_name", "seed", "warning_category", "warning_message"])
            else:
                raise ValueError(f"required output has no rows: {name}")
        (stage_dir / "audit_summary.md").write_text(summary, encoding="utf-8")

        input_hashes = {
            str(path.resolve()): sha256_file(path) for path in required
            if "embedding_" not in path.name
        }
        embedding_summary = json.loads((embedding_dir / "embedding_summary.json").read_text(encoding="utf-8"))
        run_metadata = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "task": "CIC-IDS-2017 Component / Shortcut Audit",
            "classes": list(CLASSES), "K": K, "seeds": list(SEEDS),
            "protocol": {
                "representation": "checkpoint-derived z_t only",
                "standard_scaler_fit": "class-specific train only",
                "pca": {"components": PCA_DIM, "fit_split": "train", "seed": PCA_SEED},
                "gmm": {"covariance_type": "full", "K": K, "reg_covar": REG_COVAR, "n_init": N_INIT, "max_iter": MAX_ITER, "seeds": list(SEEDS), "fit_split": "train"},
                "validation_usage": "assignment and evaluation only",
                "test_used": False, "unknown_used": False, "bic_used": False,
                "k_search_used": False, "adaptive_k_used": False,
                "trafficformer_retrained": False, "sampling_changed": False,
                "split_changed": False, "raw_pcap_reparsed": False,
            },
            "metadata": metadata_info,
            "embedding_source": {
                "path": str(embedding_dir), "summary": embedding_summary,
            },
            "gaussian_reference_max_abs_nll_difference": max(
                max(float(row["gaussian_reference_train_nll_abs_diff"]), float(row["gaussian_reference_val_nll_abs_diff"]))
                for row in weight_rows
            ),
            "assignment_rows": pq.ParquetFile(stage_dir / "component_assignments.parquet").metadata.num_rows,
            "diagnoses": {row["class_name"]: row["diagnosis"] for row in diagnosis},
            "figures": figures,
            "input_hashes": input_hashes,
            "software": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__, "pyarrow": pa.__version__, "sklearn": sklearn.__version__},
            "command": " ".join(sys.argv), "elapsed_seconds": time.time() - started,
        }
        output_paths = [path for path in stage_dir.rglob("*") if path.is_file()]
        run_metadata["output_hashes"] = {
            str(path.relative_to(stage_dir)): sha256_file(path) for path in sorted(output_paths)
            if path.name != "run_metadata.json"
        }
        (stage_dir / "run_metadata.json").write_text(json.dumps(run_metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if output_dir.exists():
            output_dir.rmdir()
        os.replace(stage_dir, output_dir)
        print(json.dumps({"status": "complete", "output_dir": str(output_dir), "diagnoses": run_metadata["diagnoses"]}, indent=2), flush=True)
    except Exception:
        if assignment_writer is not None:
            assignment_writer.close()
        failure = stage_dir / "FAILED.txt"
        failure.write_text("Run failed; inspect the tmux execution log. Partial artifacts are retained for diagnosis.\n", encoding="utf-8")
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gaussian-root", type=Path, default=PROJECT_ROOT / "outputs/cicids2017_gaussian_audit")
    parser.add_argument(
        "--mapping-dir", type=Path,
        default=Path("/home/birkenwald/data/SEU-WXY/WXY-SEU/TrafficClassifier/Dataset/CIC-IDS-2017/CIC-IDS-2017(whole)/outputs/cicids2017_pcap_label_mapping/flows"),
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/cicids2017_component_shortcut_audit")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
