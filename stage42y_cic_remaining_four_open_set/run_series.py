#!/usr/bin/env python3
"""Pre-freeze four CIC Unknown labels, then evaluate sequentially with frozen Stage40 evidence."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
DATA = Path("/home/birkenwald/data/SEU-WXY/WXY-SEU/TrafficClassifier/Dataset/CIC-IDS-2017/CIC-IDS-2017(whole)")
STAGE34 = PROJECT / "stage34_ustc_cic_closed_set"
STAGE40 = PROJECT / "stage40_ustc_cic_open_set"
STAGE42S = PROJECT / "stage42s_cic_favorable_open_set"
STAGE42X = PROJECT / "stage42x_cic_ftp_patator_open_set"
KNOWN_ROLES = STAGE40 / "unknown_slowhttptest_roles.csv"
KNOWN_DETECTION = STAGE40 / "unknown_slowhttptest/detection"
KNOWN_RUN = STAGE40 / "unknown_slowhttptest/runs/ustc/A-2"
SERIES_PROTOCOL = ROOT / "series_protocol.json"
QUEUE_PROGRESS = ROOT / "queue_progress.json"
SELECTOR = Path("/home/birkenwald/data/SEU-WXY/WXY-SEU/TrafficClassifier/skills/using-superpowers/scripts/select_gpu.py")
SPECS = (
    ("ssh_patator", "SSH-Patator", "Tuesday", 2987),
    ("web_bruteforce", "Web Attack - Brute Force", "Thursday", 1364),
    ("web_xss", "Web Attack - XSS", "Thursday", 629),
    ("slowhttptest_full", "DoS Slowhttptest", "Wednesday", 5096),
)
SPEC_BY_KEY = {spec[0]: spec for spec in SPECS}
FIELDS = ("flow_id", "day", "source_pcap", "label", "match_status",
          "flow_start_epoch_utc", "canonical_flow_key", "first_packet_index",
          "last_packet_index", "packet_count")
REUSED_CODE = ("stage42s_common.py", "build_unknown_cache.py",
               "evaluate_candidate.py", "verify_candidate.py")
METHODS = ("msp", "energy", "centroid", "des_v1")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_new(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def replace_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_csv(path: Path) -> tuple[list[str], list[dict]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write_csv_new(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"empty CSV: {path}")
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def replace_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def candidate_paths(key: str) -> tuple[Path, Path, Path, str]:
    if key not in SPEC_BY_KEY:
        raise ValueError(f"unknown candidate key: {key}")
    home = ROOT / key
    return home, home / "candidate_unknown_manifest.csv", home / "candidate_protocol.json", "unknown_" + key


def candidate_progress(key: str, status: str, phase: str, **extra) -> None:
    home, _, _, _ = candidate_paths(key)
    value = {"status": status, "phase": phase,
             "updated_at_utc": datetime.now(timezone.utc).isoformat(), **extra}
    replace_json(home / "progress.json", value)
    print(json.dumps({"candidate": key, **value}, ensure_ascii=False), flush=True)


def queue_progress(status: str, phase: str, **extra) -> None:
    value = {"status": status, "phase": phase,
             "updated_at_utc": datetime.now(timezone.utc).isoformat(), **extra}
    replace_json(QUEUE_PROGRESS, value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def freeze() -> None:
    if SERIES_PROTOCOL.exists() or any((ROOT / key).exists() for key, *_ in SPECS):
        raise FileExistsError("four-candidate series already frozen or partially initialized")
    previous = read_json(STAGE42X / "queue_progress.json")
    if previous["status"] != "COMPLETE" or previous["verification"]["status"] != "PASS":
        raise RuntimeError("preceding FTP-Patator candidate is not independently complete")
    stage34 = read_json(STAGE34 / "cicids2017_protocol_audit.json")
    stage40 = read_json(STAGE40 / "protocols.json")
    known = next(row for row in stage40["units"] if row.get("key") == "unknown_slowhttptest")
    if known["known_classes"] != ["BENIGN", "PortScan"] or digest(KNOWN_ROLES) != known["role_manifest_sha256"]:
        raise RuntimeError("Stage40 Known roles changed")
    calibration_path = KNOWN_DETECTION / "calibration.json"
    calibration = read_json(calibration_path)
    if calibration["status"] != "PASS" or calibration["unknown_fitting_count"] or calibration["test_fitting_count"]:
        raise RuntimeError("Known-only calibration is not reusable")
    _, role_rows = read_csv(KNOWN_ROLES)
    known_ids = {row["flow_id"] for row in role_rows if row["role"].startswith("known_")}
    known_test_ids = [row["flow_id"] for row in role_rows if row["role"] == "known_test"]
    if len(known_test_ids) != 2272 or len(set(known_test_ids)) != 2272:
        raise RuntimeError("Known Test membership drift")
    old_unknown_ids = {row["flow_id"] for row in role_rows if row["role"] == "unknown_test"}
    source_hashes = {}
    day_rows = {}
    for day in {spec[2] for spec in SPECS}:
        source = DATA / f"outputs/cicids2017_pcap_label_mapping/flows/{day}_flows.parquet"
        source_hashes[day] = digest(source)
        if source_hashes[day] != stage34["source_parquet_sha256"][day]:
            raise RuntimeError(f"{day} mapped-flow parquet changed since Stage34")
        day_rows[day] = pq.read_table(source, columns=list(FIELDS)).to_pylist()
    reused_hashes = {name: digest(STAGE42S / name) for name in REUSED_CODE}
    frozen = []
    for key, label, day, expected in SPECS:
        chosen = [row for row in day_rows[day] if row["label"] == label
                  and str(row["match_status"]).startswith("MATCHED")]
        ids = {row["flow_id"] for row in chosen}
        if len(chosen) != expected or len(ids) != expected or ids.intersection(known_ids):
            raise RuntimeError(f"{key} population/uniqueness/Known-overlap failed: {len(chosen)}")
        prior_unknown_overlap = ids.intersection(old_unknown_ids)
        if key == "slowhttptest_full":
            if len(prior_unknown_overlap) != 132:
                raise RuntimeError("prior Slowhttptest Unknown Test overlap changed")
        elif prior_unknown_overlap:
            raise RuntimeError(f"{key} unexpectedly intersects prior Unknown Test")
        source_pcaps = {row["source_pcap"] for row in chosen}
        if len(source_pcaps) != 1 or {row["day"] for row in chosen} != {day}:
            raise RuntimeError(f"{key} day/PCAP metadata mismatch")
        pcap = DATA / "PCAPs" / next(iter(source_pcaps))
        sidecar = pcap.with_suffix(".md5")
        if not pcap.is_file() or not sidecar.is_file():
            raise FileNotFoundError(f"{key} source PCAP or MD5 sidecar missing")
        home, manifest, protocol_path, unit = candidate_paths(key)
        home.mkdir(exist_ok=False)
        rows = []
        for row in chosen:
            start = float(row["flow_start_epoch_utc"])
            rows.append({
                "flow_id": row["flow_id"], "class_name": label, "role": "unknown_test",
                "day": row["day"], "source_pcap": row["source_pcap"],
                "group_id": f"{row['source_pcap']}|{int(start // 300)}",
                "flow_start_epoch_utc": row["flow_start_epoch_utc"],
                "canonical_flow_key": row["canonical_flow_key"],
                "first_packet_index": row["first_packet_index"],
                "last_packet_index": row["last_packet_index"],
                "packet_count": row["packet_count"], "match_status": row["match_status"],
            })
        rows.sort(key=lambda row: row["flow_id"])
        write_csv_new(manifest, rows)
        manifest_hash = digest(manifest)
        payload = {
            "status": "PASS", "unit": unit, "claim_scope": "post-hoc development diagnostic",
            "candidate_order_index": len(frozen) + 1,
            "known_classes": ["BENIGN", "PortScan"], "unknown_classes": [label],
            "unknown_test": expected, "unknown_groups": len({row["group_id"] for row in rows}),
            "prior_unknown_test_overlap": len(prior_unknown_overlap),
            "unknown_selection": "all MATCHED flows for the prelisted class; no feature/outcome filtering",
            "known_model_source": str(KNOWN_RUN), "known_roles": str(KNOWN_ROLES),
            "known_role_manifest_sha256": digest(KNOWN_ROLES),
            "known_calibration": str(calibration_path), "known_calibration_sha256": digest(calibration_path),
            "known_test_sample_scores": str(KNOWN_DETECTION / "sample_scores.csv"),
            "known_test_sample_scores_sha256": digest(KNOWN_DETECTION / "sample_scores.csv"),
            "candidate_manifest": str(manifest), "candidate_manifest_sha256": manifest_hash,
            "source_parquet": str(DATA / f"outputs/cicids2017_pcap_label_mapping/flows/{day}_flows.parquet"),
            "source_parquet_sha256": source_hashes[day],
            "source_pcap": str(pcap), "source_pcap_size_bytes": pcap.stat().st_size,
            "source_pcap_md5_sidecar": str(sidecar), "source_pcap_md5_sidecar_sha256": digest(sidecar),
            "reused_stage42s_code_sha256": reused_hashes,
            "frozen_before_candidate_packet_feature_access": True,
            "unknown_fitting_count": 0, "unknown_threshold_calibration_count": 0,
            "threshold_rule": "reuse Stage40 BENIGN+PortScan Known-Validation P95",
            "methods": list(METHODS),
            "balanced_view_selection": "SHA256(seed=2022, flow_id), n=min(Known Test, Unknown Test)",
        }
        write_json_new(protocol_path, payload)
        write_json_new(home / "cicids2017_protocol_v2_audit.json", {
            "status": "PASS", "v2_manifest_sha256": manifest_hash,
            "actual_protocol": f"Stage42Y-{key}",
        })
        if expected < len(known_test_ids):
            selected = sorted(known_test_ids, key=lambda flow_id: (
                hashlib.sha256(f"2022|balanced|{flow_id}".encode()).digest(), flow_id))[:expected]
            with (home / "balanced_known_manifest.csv").open("x", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["flow_id"])
                writer.writerows((flow_id,) for flow_id in sorted(selected))
            write_json_new(home / "balanced_view_implementation_audit.json", {
                "status": "PASS", "candidate_manifest_sha256": manifest_hash,
                "known_role_manifest_sha256": digest(KNOWN_ROLES),
                "balanced_known_manifest_sha256": digest(home / "balanced_known_manifest.csv"),
                "balanced_known_selected": expected, "balanced_unknown_selected": expected,
                "selection_rule": "lowest SHA256(2022|balanced|flow_id), score-blind Known Test IDs",
                "reason": "frozen Stage42-S evaluator retains all Known rows when Unknown < Known; isolated correction enforces predeclared 1:1 view",
                "frozen_before_candidate_packet_feature_access": True,
            })
        candidate_progress(key, "RUNNING", "protocol_frozen", unknown_test=expected,
                           unknown_groups=payload["unknown_groups"])
        frozen.append({"key": key, "label": label, "day": day, "unknown_test": expected,
                       "unknown_groups": payload["unknown_groups"],
                       "prior_unknown_test_overlap": len(prior_unknown_overlap),
                       "candidate_manifest_sha256": manifest_hash,
                       "candidate_protocol_sha256": digest(protocol_path),
                       "balanced_known_manifest_sha256": digest(home / "balanced_known_manifest.csv")
                       if expected < len(known_test_ids) else None})
    write_json_new(SERIES_PROTOCOL, {
        "status": "PASS", "claim_scope": "post-hoc sequential development diagnostic",
        "fixed_order": [spec[0] for spec in SPECS], "candidates": frozen,
        "runner_sha256": digest(ROOT / "run_series.py"),
        "stage40_known_role_manifest_sha256": digest(KNOWN_ROLES),
        "source_parquet_sha256": source_hashes,
        "checkpoint_and_threshold_rule": "reuse frozen Stage40 BENIGN+PortScan Known-only model and Known-Val P95",
        "unknown_fit_count": 0, "test_fit_count": 0,
        "frozen_before_any_new_candidate_packet_feature_access": True,
    })
    queue_progress("FROZEN", "all_four_candidate_protocols_frozen", candidates=frozen)


def frozen(key: str) -> tuple[Path, Path, dict, str]:
    series = read_json(SERIES_PROTOCOL)
    if series["status"] != "PASS" or digest(ROOT / "run_series.py") != series["runner_sha256"]:
        raise RuntimeError("series protocol or runner changed")
    home, manifest, protocol_path, unit = candidate_paths(key)
    entry = next(row for row in series["candidates"] if row["key"] == key)
    if digest(manifest) != entry["candidate_manifest_sha256"] or digest(protocol_path) != entry["candidate_protocol_sha256"]:
        raise RuntimeError(f"{key} frozen candidate evidence changed")
    protocol = read_json(protocol_path)
    if protocol["status"] != "PASS":
        raise RuntimeError(f"{key} candidate protocol not PASS")
    for name, expected in protocol["reused_stage42s_code_sha256"].items():
        if digest(STAGE42S / name) != expected:
            raise RuntimeError(f"reused Stage42-S code changed: {name}")
    return home, manifest, protocol, unit


def cache(key: str) -> None:
    home, manifest, protocol, unit = frozen(key)
    day = SPEC_BY_KEY[key][2]
    sys.path.insert(0, str(STAGE34))
    import cic_build_inputs as builder  # noqa: E402

    def rows_for_test(_phase):
        _, rows = read_csv(manifest)
        for row in rows:
            row["label"] = row.pop("class_name")
            row["split"] = "test"
        return rows

    def frozen_gate():
        calibration_path = KNOWN_DETECTION / "calibration.json"
        calibration = read_json(calibration_path)
        if digest(calibration_path) != protocol["known_calibration_sha256"]:
            raise RuntimeError("Known calibration drift before Unknown packet extraction")
        if calibration["unknown_fitting_count"] or calibration["test_fitting_count"]:
            raise RuntimeError("Known calibration was not Unknown/Test free")
        return calibration

    builder.ROOT = home
    builder.CIC = home / unit
    builder.MANIFEST = manifest
    builder.protocol.DAYS = (day,)
    builder.read_manifest = rows_for_test
    builder.frozen_head_gate = frozen_gate
    old_argv = sys.argv
    sys.argv = [str(builder.__file__), "--phase", "test"]
    candidate_progress(key, "RUNNING", "unknown_packet_cache", unknown_test=protocol["unknown_test"])
    try:
        builder.main()
    finally:
        sys.argv = old_argv
    base = home / unit / "input_caches/ustc/A-2"
    audit = {
        "status": "PASS", "dataset": "CIC-IDS-2017", "unit": unit,
        "role": "unknown_test", "flows": protocol["unknown_test"],
        "candidate_manifest_sha256": protocol["candidate_manifest_sha256"],
        "source_parquet_sha256": protocol["source_parquet_sha256"],
        "feature_formulas": "unchanged Stage34 TrafficFormer/FIG/YaTC packet views",
        "max_trafficformer_packets": 5, "max_graph_packets": 30, "max_yatc_packets": 5,
        "unknown_samples_transformed": protocol["unknown_test"],
        "unknown_samples_used_for_fitting": 0, "unknown_samples_used_for_calibration": 0,
    }
    for path in (base / "tf_fig_test", base / "yatc_mfr_test"):
        replace_json(path / "cache_audit.json", audit)
    candidate_progress(key, "RUNNING", "unknown_cache_complete", unknown_test=protocol["unknown_test"])


def load_stage42s_phase(filename: str, key: str):
    home, manifest, protocol, unit = frozen(key)
    sys.path.insert(0, str(STAGE42S))
    source = STAGE42S / filename
    spec = importlib.util.spec_from_file_location(f"stage42y_{key}_{filename.replace('.', '_')}", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load frozen Stage42-S code: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name, value in {
        "ROOT": home, "UNIT": unit, "MANIFEST": manifest,
        "PROTOCOL": home / "candidate_protocol.json",
        "progress": lambda status, phase, **extra: candidate_progress(key, status, phase, **extra),
    }.items():
        setattr(module, name, value)
    if filename == "evaluate_candidate.py":
        module.OUT = home / unit / "detection"
    return module


def evaluate(key: str) -> None:
    home, _, protocol, unit = frozen(key)
    base = home / unit / "input_caches/ustc/A-2"
    for name in ("tf_fig_test", "yatc_mfr_test"):
        data = read_json(base / name / "cache_audit.json")
        if (data["status"] != "PASS" or data["flows"] != protocol["unknown_test"]
                or data["candidate_manifest_sha256"] != protocol["candidate_manifest_sha256"]
                or data["unknown_samples_used_for_fitting"] or data["unknown_samples_used_for_calibration"]):
            raise RuntimeError(f"{key} cache audit failed before evaluation")
    load_stage42s_phase("evaluate_candidate.py", key).main()


def fix_balanced(key: str) -> None:
    home, _, protocol, unit = frozen(key)
    if protocol["unknown_test"] >= 2272:
        raise RuntimeError("balanced correction is only for short Unknown pools")
    audit = read_json(home / "balanced_view_implementation_audit.json")
    selected_path = home / "balanced_known_manifest.csv"
    if (audit["status"] != "PASS" or digest(selected_path) != audit["balanced_known_manifest_sha256"]
            or audit["candidate_manifest_sha256"] != protocol["candidate_manifest_sha256"]):
        raise RuntimeError("balanced Known membership drift")
    _, selected_rows = read_csv(selected_path)
    known_ids = {row["flow_id"] for row in selected_rows}
    expected = protocol["unknown_test"]
    if len(known_ids) != expected:
        raise RuntimeError("balanced Known selection cardinality drift")
    out = home / unit / "detection"
    score_path = out / "sample_scores.csv"
    score_fields, scores = read_csv(score_path)
    if len(scores) != 2272 + expected:
        raise RuntimeError("saved natural score population drift")
    for row in scores:
        if row["role"] == "known_test":
            row["balanced_selected"] = int(row["flow_id"] in known_ids)
        elif row["role"] == "unknown_test":
            row["balanced_selected"] = 1
        else:
            raise RuntimeError("unexpected Test role")
    subset = [row for row in scores if int(row["balanced_selected"])]
    truth = np.asarray([int(row["is_unknown"]) for row in subset], dtype=np.int64)
    if len(subset) != 2 * expected or int(truth.sum()) != expected:
        raise RuntimeError("true 1:1 subset not achieved")
    results_path = out / "open_set_results.csv"
    result_fields, results = read_csv(results_path)
    metric = load_stage42s_phase("evaluate_candidate.py", key).metric
    for row in results:
        if row["view"] != "balanced_1to1":
            continue
        score = np.asarray([float(item[f"score_{row['method']}"]) for item in subset])
        row.update({"known_test": expected, "unknown_test": expected,
                    "unknown_prevalence": 0.5,
                    **metric(truth, score, float(row["threshold"]))})
    if len(results) != 8:
        raise RuntimeError("expected eight method-by-view rows")
    replace_csv(score_path, score_fields, scores)
    replace_csv(results_path, result_fields, results)
    evaluation_path = out / "evaluation_audit.json"
    evaluation = read_json(evaluation_path)
    evaluation.update({"balanced_known_manifest_sha256": audit["balanced_known_manifest_sha256"],
                       "balanced_known_test": expected, "balanced_unknown_test": expected,
                       "balanced_view_implementation_audit": str(home / "balanced_view_implementation_audit.json")})
    replace_json(evaluation_path, evaluation)
    candidate_progress(key, "RUNNING", "balanced_view_corrected", balanced_known=expected,
                       balanced_unknown=expected)


def verify(key: str) -> None:
    home, _, protocol, unit = frozen(key)
    load_stage42s_phase("verify_candidate.py", key).main()
    audit = read_json(home / unit / "detection/evaluation_audit.json")
    if protocol["unknown_test"] < 2272:
        if (audit["balanced_known_test"] != protocol["unknown_test"]
                or audit["balanced_unknown_test"] != protocol["unknown_test"]):
            raise RuntimeError("balanced 1:1 audit failed after independent replay")


def sequential() -> None:
    series = read_json(SERIES_PROTOCOL)
    if series["fixed_order"] != [spec[0] for spec in SPECS]:
        raise RuntimeError("frozen candidate order changed")
    completed = []
    try:
        for key, _, _, expected in SPECS:
            frozen(key)
            for phase in ("cache", "evaluate", "fix_balanced", "verify"):
                if phase == "fix_balanced" and expected >= 2272:
                    continue
                queue_progress("RUNNING", phase, candidate=key, completed=completed)
                if phase == "evaluate":
                    command = [sys.executable, str(SELECTOR), "--min-free-gb", "16",
                               "--max-utilization", "30", "--", sys.executable,
                               str(ROOT / "run_series.py"), phase, key]
                else:
                    command = [sys.executable, str(ROOT / "run_series.py"), phase, key]
                subprocess.run(command, cwd=ROOT, check=True)
            home, manifest, protocol, unit = frozen(key)
            verification = read_json(home / unit / "detection/independent_verification.json")
            evaluation = read_json(home / unit / "detection/evaluation_audit.json")
            if (verification["status"] != "PASS" or verification["unknown_test"] != expected
                    or verification["known_test"] != 2272 or verification["metrics_replayed"] != 8
                    or evaluation["status"] != "PASS" or not evaluation["checkpoint_hashes_unchanged"]
                    or evaluation["unknown_fit_count"] or evaluation["test_fit_count"]
                    or digest(manifest) != protocol["candidate_manifest_sha256"]):
                raise RuntimeError(f"{key} completion verification failed")
            completed.append(key)
            queue_progress("RUNNING", "candidate_verified", candidate=key, completed=completed)
        queue_progress("COMPLETE", "all_four_independently_verified", completed=completed)
    except BaseException as exc:
        write_json_new(ROOT / "queue_failure.json", {
            "status": "FAILED", "completed": completed, "error": repr(exc),
            "traceback": traceback.format_exc(),
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        })
        queue_progress("FAILED", "stopped", completed=completed, error=repr(exc))
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("freeze", "cache", "evaluate", "fix_balanced", "verify", "sequential"))
    parser.add_argument("candidate", nargs="?", choices=tuple(SPEC_BY_KEY))
    args = parser.parse_args()
    if args.phase == "freeze":
        if args.candidate is not None:
            parser.error("freeze takes no candidate")
        freeze()
    elif args.phase == "sequential":
        if args.candidate is not None:
            parser.error("sequential takes no candidate")
        sequential()
    else:
        if args.candidate is None:
            parser.error(f"{args.phase} requires a candidate")
        {"cache": cache, "evaluate": evaluate,
         "fix_balanced": fix_balanced, "verify": verify}[args.phase](args.candidate)


if __name__ == "__main__":
    main()
