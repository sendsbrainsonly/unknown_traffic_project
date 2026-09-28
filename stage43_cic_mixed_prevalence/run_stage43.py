#!/usr/bin/env python3
"""Stage 43: frozen-score mixed-Unknown composition/prevalence stress test."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


OUT = Path(__file__).resolve().parent
PROJECT = OUT.parent
PROTOCOL = OUT / "mixed_protocol.json"
METHODS = ("msp", "energy", "centroid", "des_v1")
METRICS = (
    "auroc", "auprc", "ufar", "known_frr", "known_acceptance",
    "binary_accuracy", "balanced_accuracy", "mcc",
    "unknown_precision", "unknown_recall", "binary_f1",
)
SEEDS = tuple(range(2022, 2042))
TOTAL_SAMPLES = 2000

SOURCES = {
    "slowloris": ("DoS slowloris", PROJECT / "stage42s_cic_favorable_open_set/unknown_slowloris/detection/sample_scores.csv"),
    "goldeneye": ("DoS GoldenEye", PROJECT / "stage42t_cic_goldeneye_open_set/unknown_goldeneye/detection/sample_scores.csv"),
    "hulk": ("DoS Hulk", PROJECT / "stage42u_cic_hulk_open_set/unknown_hulk/detection/sample_scores.csv"),
    "ddos": ("DDoS", PROJECT / "stage42v_cic_ddos_open_set/unknown_ddos/detection/sample_scores.csv"),
    "bot": ("Bot", PROJECT / "stage42w_cic_bot_open_set/unknown_bot/detection/sample_scores.csv"),
    "ftp_patator": ("FTP-Patator", PROJECT / "stage42x_cic_ftp_patator_open_set/unknown_ftp_patator/detection/sample_scores.csv"),
    "ssh_patator": ("SSH-Patator", PROJECT / "stage42y_cic_remaining_four_open_set/ssh_patator/unknown_ssh_patator/detection/sample_scores.csv"),
    "web_bruteforce": ("Web Attack - Brute Force", PROJECT / "stage42y_cic_remaining_four_open_set/web_bruteforce/unknown_web_bruteforce/detection/sample_scores.csv"),
    "web_xss": ("Web Attack - XSS", PROJECT / "stage42y_cic_remaining_four_open_set/web_xss/unknown_web_xss/detection/sample_scores.csv"),
    "slowhttptest": ("DoS Slowhttptest", PROJECT / "stage42y_cic_remaining_four_open_set/slowhttptest_full/unknown_slowhttptest_full/detection/sample_scores.csv"),
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def write_rows(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def progress(status: str, phase: str, **extra: object) -> None:
    payload = {"status": status, "phase": phase, "updated_at_utc": now(), **extra}
    write_json(OUT / "progress.json", payload)
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def result_path(score_path: Path) -> Path:
    return score_path.with_name("open_set_results.csv")


def score_signature(row: dict[str, str]) -> tuple[str, ...]:
    fields = ["flow_id", "role", "class_name", "is_unknown", "predicted_known"]
    for method in METHODS:
        fields.extend((f"score_{method}", f"reject_{method}"))
    return tuple(row[name] for name in fields)


def largest_remainder(total: int, weights: dict[str, float]) -> dict[str, int]:
    if total <= 0 or not weights or any(value < 0 for value in weights.values()):
        raise ValueError("invalid quota request")
    scale = sum(weights.values())
    raw = {key: total * value / scale for key, value in weights.items()}
    quota = {key: int(math.floor(value)) for key, value in raw.items()}
    remaining = total - sum(quota.values())
    order = sorted(weights, key=lambda key: (-(raw[key] - quota[key]), key))
    for key in order[:remaining]:
        quota[key] += 1
    if sum(quota.values()) != total:
        raise RuntimeError("quota allocation failed")
    return quota


def scenario_definitions(pool_counts: dict[str, int]) -> list[dict]:
    labels = [value[0] for value in SOURCES.values()]
    easy = [label for key, (label, _) in SOURCES.items() if key not in {"bot", "ddos"}]
    hard = [SOURCES["bot"][0], SOURCES["ddos"][0]]
    dos = [SOURCES[key][0] for key in ("slowloris", "goldeneye", "hulk", "ddos", "slowhttptest")]
    auth_web = [SOURCES[key][0] for key in ("ftp_patator", "ssh_patator", "web_bruteforce", "web_xss")]
    scenarios: list[dict] = []
    prevalence = {
        "known9_unknown1": (1800, 200, "9:1"),
        "known3_unknown1": (1500, 500, "3:1"),
        "known1_unknown1": (1000, 1000, "1:1"),
        "known1_unknown3": (500, 1500, "1:3"),
        "known1_unknown9": (200, 1800, "1:9"),
    }
    for setting, (known_n, unknown_n, ratio) in prevalence.items():
        scenarios.append({
            "experiment": "prevalence",
            "setting": setting,
            "ratio": ratio,
            "known_quotas": {"BENIGN": known_n // 2, "PortScan": known_n // 2},
            "unknown_quotas": largest_remainder(unknown_n, {label: 1.0 for label in labels}),
            "primary": True,
        })
    compositions = {
        "all_balanced": {label: 1.0 for label in labels},
        "authentication_web": {label: 1.0 for label in auth_web},
        "dos": {label: 1.0 for label in dos},
        "previously_easy": {label: 1.0 for label in easy},
        "previously_hard": {label: 1.0 for label in hard},
        "easy_hard_mixed": {
            **{label: 0.5 / len(easy) for label in easy},
            **{label: 0.5 / len(hard) for label in hard},
        },
        "natural_frequency": {label: float(pool_counts[label]) for label in labels},
    }
    for setting, weights in compositions.items():
        scenarios.append({
            "experiment": "composition",
            "setting": setting,
            "ratio": "1:1",
            "known_quotas": {"BENIGN": 500, "PortScan": 500},
            "unknown_quotas": largest_remainder(1000, weights),
            "primary": setting in {"all_balanced", "easy_hard_mixed", "previously_hard"},
        })
    for setting, known_quotas in {
        "benign1_portscan1": {"BENIGN": 500, "PortScan": 500},
        "benign3_portscan1": {"BENIGN": 750, "PortScan": 250},
        "benign1_portscan3": {"BENIGN": 250, "PortScan": 750},
    }.items():
        scenarios.append({
            "experiment": "known_composition",
            "setting": setting,
            "ratio": "1:1",
            "known_quotas": known_quotas,
            "unknown_quotas": largest_remainder(1000, {label: 1.0 for label in labels}),
            "primary": True,
        })
    for scenario in scenarios:
        for label, count in scenario["unknown_quotas"].items():
            if count > pool_counts[label]:
                raise RuntimeError(f"insufficient {label}: need {count}, have {pool_counts[label]}")
        if sum(scenario["known_quotas"].values()) + sum(scenario["unknown_quotas"].values()) != TOTAL_SAMPLES:
            raise RuntimeError(f"scenario size error: {scenario}")
    return scenarios


def load_and_audit():
    known_by_class: dict[str, list[dict[str, str]]] = defaultdict(list)
    unknown_by_class: dict[str, list[dict[str, str]]] = defaultdict(list)
    source_manifest: list[dict] = []
    audit: list[dict] = []
    reference: dict[str, tuple[str, ...]] | None = None
    thresholds_seen: dict[str, set[float]] = {method: set() for method in METHODS}
    known_ids: set[str] = set()
    unknown_owner: dict[str, str] = {}
    for source_key, (expected_label, score_path) in SOURCES.items():
        if not score_path.is_file() or not result_path(score_path).is_file():
            raise FileNotFoundError(score_path)
        rows = read_rows(score_path)
        known = [row for row in rows if row["role"] == "known_test"]
        unknown = [row for row in rows if row["role"] == "unknown_test"]
        if len(known) != 2272 or not unknown:
            raise RuntimeError(f"role counts failed: {source_key}")
        if {row["class_name"] for row in unknown} != {expected_label}:
            raise RuntimeError(f"label mismatch: {source_key}")
        current = {row["flow_id"]: score_signature(row) for row in known}
        if len(current) != len(known):
            raise RuntimeError(f"duplicate Known IDs: {source_key}")
        if reference is None:
            reference = current
            known_ids = set(current)
            for row in known:
                row["source_key"] = "known_reference"
                known_by_class[row["class_name"]].append(row)
        elif current != reference:
            raise RuntimeError(f"Known score population drift: {source_key}")
        for row in unknown:
            flow_id = row["flow_id"]
            if flow_id in known_ids:
                raise RuntimeError(f"Known/Unknown ID overlap: {flow_id}")
            if flow_id in unknown_owner:
                raise RuntimeError(f"cross-Unknown duplicate: {flow_id} in {unknown_owner[flow_id]} and {source_key}")
            unknown_owner[flow_id] = source_key
            row["source_key"] = source_key
            unknown_by_class[expected_label].append(row)
        results = [row for row in read_rows(result_path(score_path)) if row["view"] == "natural"]
        if {row["method"] for row in results} != set(METHODS):
            raise RuntimeError(f"method result mismatch: {source_key}")
        local_thresholds = {row["method"]: float(row["threshold"]) for row in results}
        for method, threshold in local_thresholds.items():
            thresholds_seen[method].add(threshold)
        source_manifest.append({
            "source_key": source_key,
            "unknown_class": expected_label,
            "score_path": str(score_path),
            "score_sha256": digest(score_path),
            "result_path": str(result_path(score_path)),
            "result_sha256": digest(result_path(score_path)),
            "known_rows": len(known),
            "unknown_rows": len(unknown),
            "total_rows": len(rows),
            "independent_verification_path": str(score_path.with_name("independent_verification.json")),
        })
    if reference is None:
        raise RuntimeError("no score source")
    thresholds = {}
    for method, values in thresholds_seen.items():
        if len(values) != 1:
            raise RuntimeError(f"threshold drift for {method}: {values}")
        thresholds[method] = next(iter(values))
    for label, rows in known_by_class.items():
        audit.append({
            "check": "known_class_count", "key": label, "observed": len(rows),
            "expected": 1136, "status": "PASS" if len(rows) == 1136 else "FAIL",
            "detail": "single canonical copy after 10-source equality check",
        })
    audit.extend([
        {
            "check": "known_population_identical_across_sources", "key": "all_sources",
            "observed": len(reference), "expected": 2272, "status": "PASS",
            "detail": "score/reject/prediction signatures exactly equal; balanced_selected ignored",
        },
        {
            "check": "cross_unknown_duplicate_flow_ids", "key": "all_unknown_classes",
            "observed": 0, "expected": 0, "status": "PASS",
            "detail": f"{len(unknown_owner)} unique Unknown IDs",
        },
        {
            "check": "known_unknown_flow_id_overlap", "key": "all_roles",
            "observed": 0, "expected": 0, "status": "PASS", "detail": "no overlap",
        },
        {
            "check": "threshold_consistency", "key": "all_methods_all_sources",
            "observed": len(thresholds), "expected": len(METHODS), "status": "PASS",
            "detail": json.dumps(thresholds, sort_keys=True),
        },
    ])
    for method, threshold in thresholds.items():
        mismatches = 0
        for rows in list(known_by_class.values()) + list(unknown_by_class.values()):
            for row in rows:
                mismatches += int((float(row[f"score_{method}"]) > threshold) != bool(int(row[f"reject_{method}"])))
        audit.append({
            "check": "frozen_decision_replay", "key": method, "observed": mismatches,
            "expected": 0, "status": "PASS" if mismatches == 0 else "FAIL",
            "detail": "score > frozen Known-Val P95 threshold",
        })
        if mismatches:
            raise RuntimeError(f"saved decision mismatch for {method}")
    return known_by_class, unknown_by_class, thresholds, source_manifest, audit


def freeze() -> None:
    if PROTOCOL.exists():
        raise FileExistsError(f"protocol already frozen: {PROTOCOL}")
    progress("RUNNING", "source_score_audit")
    known, unknown, thresholds, sources, audit = load_and_audit()
    pool_counts = {label: len(rows) for label, rows in unknown.items()}
    scenarios = scenario_definitions(pool_counts)
    write_rows(OUT / "frozen_score_pool_manifest.csv", list(sources[0]), sources)
    write_rows(OUT / "duplicate_and_alignment_audit.csv", list(audit[0]), audit)
    protocol = {
        "status": "PASS",
        "frozen_at_utc": now(),
        "claim_scope": "post-hoc diagnostic",
        "dataset": "CIC-IDS-2017",
        "known_classes": ["BENIGN", "PortScan"],
        "known_test_unique": sum(len(rows) for rows in known.values()),
        "unknown_classes": list(pool_counts),
        "unknown_pool_counts": pool_counts,
        "methods": list(METHODS),
        "threshold_source": "unchanged Stage40 Known Validation P95",
        "thresholds": thresholds,
        "total_samples_per_mixture": TOTAL_SAMPLES,
        "resampling_seeds": list(SEEDS),
        "scenarios": scenarios,
        "source_files": sources,
        "runner_sha256": digest(Path(__file__)),
        "verifier_sha256": digest(OUT / "verify_stage43.py"),
        "unknown_fitting_count": 0,
        "test_fitting_count": 0,
        "encoder_training_count": 0,
        "pcap_reads": 0,
        "limitations": [
            "All candidate classes and their scores were already exposed before Stage43.",
            "CIC attack day, endpoint and capture confounding remain.",
            "DoS Slowhttptest contains 132 flows previously used in Stage40 Unknown Test.",
            "Resampling quantiles measure mixture-sampling variability, not training uncertainty.",
        ],
    }
    write_json(PROTOCOL, protocol)
    progress(
        "FROZEN", "protocol_frozen", source_files=len(sources),
        scenarios=len(scenarios), seeds=len(SEEDS),
        expected_mixture_rows=len(scenarios) * len(SEEDS) * TOTAL_SAMPLES,
    )


def verify_frozen_protocol() -> dict:
    protocol = json.loads(PROTOCOL.read_text())
    if protocol["status"] != "PASS":
        raise RuntimeError("protocol status is not PASS")
    if digest(Path(__file__)) != protocol["runner_sha256"]:
        raise RuntimeError("runner changed after protocol freeze")
    if digest(OUT / "verify_stage43.py") != protocol["verifier_sha256"]:
        raise RuntimeError("verifier changed after protocol freeze")
    for source in protocol["source_files"]:
        if digest(Path(source["score_path"])) != source["score_sha256"]:
            raise RuntimeError(f"score source changed: {source['source_key']}")
        if digest(Path(source["result_path"])) != source["result_sha256"]:
            raise RuntimeError(f"result source changed: {source['source_key']}")
    return protocol


def derived_rng(seed: int, experiment: str, setting: str) -> np.random.Generator:
    raw = hashlib.sha256(f"{seed}|{experiment}|{setting}".encode()).digest()
    return np.random.default_rng(int.from_bytes(raw[:8], "big"))


def sample_rows(pools, quotas: dict[str, int], rng: np.random.Generator):
    selected = []
    for label in sorted(quotas):
        count = quotas[label]
        rows = pools[label]
        if count > len(rows):
            raise RuntimeError(f"sample exceeds pool for {label}")
        indices = rng.choice(len(rows), size=count, replace=False)
        selected.extend(rows[int(index)] for index in indices)
    return selected


def metric(truth: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, float]:
    decision = score > threshold
    known = truth == 0
    unknown = truth == 1
    return {
        "auroc": float(roc_auc_score(truth, score)),
        "auprc": float(average_precision_score(truth, score)),
        "ufar": float(np.mean(~decision[unknown])),
        "known_frr": float(np.mean(decision[known])),
        "known_acceptance": float(np.mean(~decision[known])),
        "binary_accuracy": float(accuracy_score(truth, decision)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, decision)),
        "mcc": float(matthews_corrcoef(truth, decision)),
        "unknown_precision": float(precision_score(truth, decision, zero_division=0)),
        "unknown_recall": float(recall_score(truth, decision, zero_division=0)),
        "binary_f1": float(f1_score(truth, decision, zero_division=0)),
    }


def summary_rows(run_rows: list[dict], experiment: str) -> list[dict]:
    groups = defaultdict(list)
    for row in run_rows:
        if row["experiment"] == experiment:
            groups[(row["setting"], row["method"])].append(row)
    output = []
    for (setting, method), rows in sorted(groups.items()):
        base = {
            "experiment": experiment, "setting": setting, "method": method,
            "runs": len(rows), "known_samples": rows[0]["known_samples"],
            "unknown_samples": rows[0]["unknown_samples"],
        }
        for name in METRICS:
            values = np.asarray([float(row[name]) for row in rows])
            base[f"{name}_mean"] = float(values.mean())
            base[f"{name}_std"] = float(values.std(ddof=1))
            base[f"{name}_q025"] = float(np.quantile(values, 0.025))
            base[f"{name}_q975"] = float(np.quantile(values, 0.975))
        output.append(base)
    return output


def paired_comparisons(run_rows: list[dict]) -> list[dict]:
    indexed = {
        (row["experiment"], row["setting"], int(row["seed"]), row["method"]): row
        for row in run_rows
    }
    output = []
    settings = sorted({(row["experiment"], row["setting"]) for row in run_rows})
    for experiment, setting in settings:
        for baseline in ("msp", "energy", "centroid"):
            deltas = defaultdict(list)
            for seed in SEEDS:
                des = indexed[(experiment, setting, seed, "des_v1")]
                other = indexed[(experiment, setting, seed, baseline)]
                for name in ("auroc", "auprc", "ufar", "known_frr", "binary_f1"):
                    deltas[name].append(float(des[name]) - float(other[name]))
            row = {
                "experiment": experiment, "setting": setting,
                "comparison": f"des_v1_minus_{baseline}", "runs": len(SEEDS),
                "positive_auroc_runs": sum(value > 0 for value in deltas["auroc"]),
                "lower_ufar_runs": sum(value < 0 for value in deltas["ufar"]),
            }
            for name, values_list in deltas.items():
                values = np.asarray(values_list)
                row[f"delta_{name}_mean"] = float(values.mean())
                row[f"delta_{name}_std"] = float(values.std(ddof=1))
                row[f"delta_{name}_q025"] = float(np.quantile(values, 0.025))
                row[f"delta_{name}_q975"] = float(np.quantile(values, 0.975))
            output.append(row)
    return output


def determine_gate(prevalence: list[dict], composition: list[dict]):
    prev_des = [row for row in prevalence if row["method"] == "des_v1"]
    comp_des = [row for row in composition if row["method"] == "des_v1"]
    all_balanced = next(row for row in comp_des if row["setting"] == "all_balanced")
    prev_pass = all(
        row["auroc_mean"] >= 0.95 and row["ufar_mean"] <= 0.10 and row["known_frr_mean"] <= 0.10
        for row in prev_des
    )
    comp_pass = all(
        row["auroc_mean"] >= 0.95 and row["ufar_mean"] <= 0.10 and row["known_frr_mean"] <= 0.10
        for row in comp_des
    )
    baseline_pass = (
        all_balanced["auroc_mean"] >= 0.95
        and all_balanced["ufar_mean"] <= 0.10
        and all_balanced["known_frr_mean"] <= 0.10
    )
    if prev_pass and comp_pass:
        gate = "MIXED_OPEN_SET_ROBUST"
    elif prev_pass and not comp_pass:
        gate = "COMPOSITION_SENSITIVE"
    elif not prev_pass and baseline_pass:
        gate = "PREVALENCE_SENSITIVE"
    else:
        gate = "MIXED_OPEN_SET_NOT_ROBUST"
    return gate, {
        "prevalence_cells_pass": prev_pass,
        "composition_cells_pass": comp_pass,
        "all_balanced_pass": baseline_pass,
        "criteria": "mean AUROC>=0.95, mean UFAR<=0.10, mean Known FRR<=0.10",
    }


def run() -> None:
    protocol = verify_frozen_protocol()
    outputs = [
        "mixture_sample_manifest.csv", "run_level_metrics.csv",
        "per_unknown_class_metrics.csv", "known_class_metrics.csv",
        "prevalence_summary.csv", "composition_summary.csv",
        "known_composition_summary.csv", "method_comparison.csv",
        "stage43_mixed_open_set_report.md", "run_completion.json",
    ]
    if any((OUT / name).exists() for name in outputs):
        raise FileExistsError("run outputs already exist; refusing overwrite")
    progress("RUNNING", "load_frozen_score_pool")
    known, unknown, thresholds, _, _ = load_and_audit()
    scenario_lookup = {(row["experiment"], row["setting"]): row for row in protocol["scenarios"]}
    sample_fields = ["experiment", "setting", "seed", "sample_index", "role", "class_name", "flow_id", "source_key"]
    run_fields = [
        "experiment", "setting", "seed", "method", "known_samples",
        "unknown_samples", "unknown_prevalence", "threshold", *METRICS,
    ]
    per_class_fields = [
        "experiment", "setting", "seed", "method", "unknown_class",
        "known_samples", "unknown_samples", "auroc", "auprc", "ufar", "unknown_recall",
    ]
    known_class_fields = [
        "experiment", "setting", "seed", "method", "known_class",
        "samples", "known_frr", "known_acceptance",
    ]
    run_rows: list[dict] = []
    per_class_count = 0
    known_class_count = 0
    with (OUT / "mixture_sample_manifest.csv").open("w", newline="") as sample_handle, \
         (OUT / "run_level_metrics.csv").open("w", newline="") as run_handle, \
         (OUT / "per_unknown_class_metrics.csv").open("w", newline="") as class_handle, \
         (OUT / "known_class_metrics.csv").open("w", newline="") as known_handle:
        sample_writer = csv.DictWriter(sample_handle, fieldnames=sample_fields)
        run_writer = csv.DictWriter(run_handle, fieldnames=run_fields)
        class_writer = csv.DictWriter(class_handle, fieldnames=per_class_fields)
        known_writer = csv.DictWriter(known_handle, fieldnames=known_class_fields)
        sample_writer.writeheader()
        run_writer.writeheader()
        class_writer.writeheader()
        known_writer.writeheader()
        completed = 0
        for experiment, setting in sorted(scenario_lookup):
            scenario = scenario_lookup[(experiment, setting)]
            for seed in SEEDS:
                rng = derived_rng(seed, experiment, setting)
                selected_known = sample_rows(known, scenario["known_quotas"], rng)
                selected_unknown = sample_rows(unknown, scenario["unknown_quotas"], rng)
                selected = selected_known + selected_unknown
                if len(selected) != TOTAL_SAMPLES or len({row["flow_id"] for row in selected}) != TOTAL_SAMPLES:
                    raise RuntimeError(f"membership integrity failed: {experiment}/{setting}/{seed}")
                for index, row in enumerate(selected):
                    sample_writer.writerow({
                        "experiment": experiment, "setting": setting, "seed": seed,
                        "sample_index": index, "role": row["role"],
                        "class_name": row["class_name"], "flow_id": row["flow_id"],
                        "source_key": row["source_key"],
                    })
                truth = np.asarray([int(row["is_unknown"]) for row in selected])
                known_mask = truth == 0
                unknown_mask = truth == 1
                for method in METHODS:
                    scores = np.asarray([float(row[f"score_{method}"]) for row in selected])
                    values = metric(truth, scores, thresholds[method])
                    result = {
                        "experiment": experiment, "setting": setting, "seed": seed,
                        "method": method, "known_samples": int(known_mask.sum()),
                        "unknown_samples": int(unknown_mask.sum()),
                        "unknown_prevalence": float(unknown_mask.mean()),
                        "threshold": thresholds[method], **values,
                    }
                    run_writer.writerow(result)
                    run_rows.append(result)
                    for label in sorted(scenario["unknown_quotas"]):
                        label_mask = np.asarray([
                            row["role"] == "unknown_test" and row["class_name"] == label
                            for row in selected
                        ])
                        pair_mask = known_mask | label_mask
                        pair_truth = truth[pair_mask]
                        pair_score = scores[pair_mask]
                        decision = pair_score > thresholds[method]
                        label_unknown = pair_truth == 1
                        class_writer.writerow({
                            "experiment": experiment, "setting": setting, "seed": seed,
                            "method": method, "unknown_class": label,
                            "known_samples": int((pair_truth == 0).sum()),
                            "unknown_samples": int(label_unknown.sum()),
                            "auroc": float(roc_auc_score(pair_truth, pair_score)),
                            "auprc": float(average_precision_score(pair_truth, pair_score)),
                            "ufar": float(np.mean(~decision[label_unknown])),
                            "unknown_recall": float(np.mean(decision[label_unknown])),
                        })
                        per_class_count += 1
                    decision_all = scores > thresholds[method]
                    for label in sorted(scenario["known_quotas"]):
                        label_mask = np.asarray([
                            row["role"] == "known_test" and row["class_name"] == label
                            for row in selected
                        ])
                        known_writer.writerow({
                            "experiment": experiment, "setting": setting, "seed": seed,
                            "method": method, "known_class": label,
                            "samples": int(label_mask.sum()),
                            "known_frr": float(np.mean(decision_all[label_mask])),
                            "known_acceptance": float(np.mean(~decision_all[label_mask])),
                        })
                        known_class_count += 1
                completed += 1
                if completed % 20 == 0:
                    progress(
                        "RUNNING", "resampling", completed_mixtures=completed,
                        total_mixtures=len(protocol["scenarios"]) * len(SEEDS),
                    )
    prevalence = summary_rows(run_rows, "prevalence")
    composition = summary_rows(run_rows, "composition")
    known_composition = summary_rows(run_rows, "known_composition")
    for filename, rows in (
        ("prevalence_summary.csv", prevalence),
        ("composition_summary.csv", composition),
        ("known_composition_summary.csv", known_composition),
    ):
        write_rows(OUT / filename, list(rows[0]), rows)
    comparisons = paired_comparisons(run_rows)
    write_rows(OUT / "method_comparison.csv", list(comparisons[0]), comparisons)
    gate, gate_checks = determine_gate(prevalence, composition)
    des_comp = [row for row in composition if row["method"] == "des_v1"]
    des_prev = [row for row in prevalence if row["method"] == "des_v1"]
    report_lines = [
        "# Stage 43 — CIC Mixed-Unknown Composition and Prevalence Stress Test", "",
        "- Status: complete; frozen-score diagnostic",
        f"- Final Gate: **{gate}**",
        "- Encoder training / threshold fitting / PCAP reads: 0 / 0 / 0",
        f"- Mixtures: {len(protocol['scenarios']) * len(SEEDS)}; membership rows: {len(protocol['scenarios']) * len(SEEDS) * TOTAL_SAMPLES:,}",
        "", "## DES-v1 prevalence results", "",
        "| Setting | AUROC mean | AUPRC mean | UFAR mean | Known FRR mean |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in des_prev:
        report_lines.append(
            f"| {row['setting']} | {row['auroc_mean']:.6f} | {row['auprc_mean']:.6f} | {row['ufar_mean']:.6f} | {row['known_frr_mean']:.6f} |"
        )
    report_lines.extend([
        "", "## DES-v1 composition results", "",
        "| Setting | AUROC mean | AUPRC mean | UFAR mean | Known FRR mean |",
        "|---|---:|---:|---:|---:|",
    ])
    for row in des_comp:
        report_lines.append(
            f"| {row['setting']} | {row['auroc_mean']:.6f} | {row['auprc_mean']:.6f} | {row['ufar_mean']:.6f} | {row['known_frr_mean']:.6f} |"
        )
    report_lines.extend([
        "", "## Interpretation boundary", "",
        "- AUROC, UFAR and Known FRR are primary across prevalence; AUPRC, Accuracy and F1 change mechanically with class prevalence.",
        "- previously_easy and previously_hard are retrospective stress sets defined from exposed Stage42 outcomes.",
        "- Per-class results must be read alongside aggregate mixtures so large/easy attack classes cannot hide Bot/DDoS failures.",
        "- CIC day/capture/endpoint confounding is not removed by score-level mixing.",
        "", "## Gate checks", "",
        f"- Prevalence cells pass: {gate_checks['prevalence_cells_pass']}",
        f"- Composition cells pass: {gate_checks['composition_cells_pass']}",
        f"- All-balanced cell pass: {gate_checks['all_balanced_pass']}",
        f"- Rule: {gate_checks['criteria']}", "",
    ])
    (OUT / "stage43_mixed_open_set_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    write_json(OUT / "run_completion.json", {
        "status": "PASS", "completed_at_utc": now(), "final_gate": gate,
        "gate_checks": gate_checks, "scenarios": len(protocol["scenarios"]),
        "seeds": len(SEEDS), "mixtures": len(protocol["scenarios"]) * len(SEEDS),
        "mixture_membership_rows": len(protocol["scenarios"]) * len(SEEDS) * TOTAL_SAMPLES,
        "run_metric_rows": len(run_rows), "per_unknown_class_rows": per_class_count,
        "known_class_rows": known_class_count, "unknown_fitting_count": 0,
        "test_fitting_count": 0, "encoder_training_count": 0, "pcap_reads": 0,
    })
    progress("COMPLETE", "run_outputs_written", final_gate=gate)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("freeze", "run"))
    args = parser.parse_args()
    if args.phase == "freeze":
        freeze()
    else:
        run()


if __name__ == "__main__":
    main()
