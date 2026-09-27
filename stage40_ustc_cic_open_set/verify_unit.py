#!/usr/bin/env python3
"""Independently replay saved sample-score metrics and frozen role membership."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from freeze_protocols import ROOT, digest

METHODS = ("msp", "energy", "centroid", "des_v1")


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def close(actual: float, expected: str | float) -> bool:
    return math.isclose(float(actual), float(expected), rel_tol=0, abs_tol=1e-9)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", choices=("ustc_a2", "unknown_slowhttptest", "unknown_portscan"), required=True)
    args = parser.parse_args()
    frozen = json.loads((ROOT / "protocols.json").read_text())
    unit = next(x for x in frozen["units"] if x.get("key", "ustc_a2") == args.key)
    if digest(Path(unit["role_manifest"])) != unit["role_manifest_sha256"]:
        raise RuntimeError("frozen role hash changed")
    out = ROOT / args.key / "detection"
    if not (out / "SUCCESS").exists():
        raise RuntimeError("evaluation incomplete")
    expected = sorted((r for r in rows(Path(unit["role_manifest"]))
                       if r["role"] in ("known_test", "unknown_test")),
                      key=lambda r: r["flow_id"])
    saved = rows(out / "sample_scores.csv")
    if len(expected) != len(saved) or any((r["flow_id"], r["role"], r["class_name"]) !=
                                          (s["flow_id"], s["role"], s["class_name"])
                                          for r, s in zip(expected, saved, strict=True)):
        raise RuntimeError("sample score and protocol role mismatch")
    truth = np.asarray([int(r["role"] == "unknown_test") for r in expected])
    cal = json.loads((out / "calibration.json").read_text())
    if cal["unknown_fitting_count"] or cal["test_fitting_count"]:
        raise RuntimeError("Unknown/Test used for calibration")
    reported = {r["method"]: r for r in rows(out / "open_set_results.csv")}
    for method in METHODS:
        score = np.asarray([float(r[f"score_{method}"]) for r in saved])
        threshold = float(cal["thresholds"][method])
        decision = score > threshold
        run = reported[method]
        actual = {"auroc": roc_auc_score(truth, score),
                  "auprc": average_precision_score(truth, score),
                  "ufar": float(np.mean(~decision[truth == 1])),
                  "known_frr": float(np.mean(decision[truth == 0])),
                  "known_acceptance": float(np.mean(~decision[truth == 0])),
                  "binary_f1": f1_score(truth, decision),
                  "unknown_prevalence": truth.mean()}
        if any(not close(value, run[name]) for name, value in actual.items()):
            raise RuntimeError(f"{method} reported metric replay failed")
        if not close(threshold, run["threshold"]):
            raise RuntimeError(f"{method} threshold drift")
        if any(int(item[f"reject_{method}"]) != int(flag)
               for item, flag in zip(saved, decision, strict=True)):
            raise RuntimeError(f"{method} saved binary decisions drift")
    per = rows(out / "per_unknown_class.csv")
    for r in per:
        chosen = (truth == 0) | np.asarray([item["class_name"] == r["unknown_class"] and
                                            item["role"] == "unknown_test" for item in saved])
        score = np.asarray([float(item[f"score_{r['method']}"]) for item in saved])[chosen]
        if not close(roc_auc_score(truth[chosen], score), r["auroc"]) or \
           not close(average_precision_score(truth[chosen], score), r["auprc"]):
            raise RuntimeError("per-Unknown-class metric replay failed")
    if any(digest(Path(path)) != sha for path, sha in unit["source_hashes"].items()):
        raise RuntimeError("source protocol changed")
    audit = json.loads((out / "evaluation_audit.json").read_text())
    if audit["status"] != "PASS" or not audit["checkpoint_hashes_unchanged"]:
        raise RuntimeError("evaluation audit not PASS")
    payload = {"status": "PASS", "unit": args.key, "replayed_methods": list(METHODS),
               "sample_rows": len(saved), "known_test": int((truth == 0).sum()),
               "unknown_test": int((truth == 1).sum()),
               "per_unknown_class_rows": len(per), "role_and_source_hashes_unchanged": True}
    with (out / "independent_verification.json").open("x", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
