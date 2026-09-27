#!/usr/bin/env python3
"""Replay, report, and validate the completed Stage34B CIC result."""
from __future__ import annotations

import csv
import fcntl
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from prepare_balanced import AUDIT, CLASSES, MANIFEST, ROOT, SOURCE, sha, verified_rows

PROJECT = ROOT.parent
WORKSPACE = PROJECT.parents[1]
SKILL = WORKSPACE / ".agents/skills/experiment-data-preservation/scripts"
PYTHON = "/home/birkenwald/data/SEU-WXY/conda_envs/2025-10-8-WXY-dgl_py310/bin/python"
RUN = ROOT / "cicids2017/runs/ustc/A-2"
TEST = RUN / "known_test_evaluation"


def replay() -> tuple[dict, dict, dict, list[dict]]:
    balanced = verified_rows()
    audit = json.loads(AUDIT.read_text())
    target = {r["flow_id"]: r["label"] for r in balanced if r["split"] == "test"}
    if len(target) != 2536:
        raise RuntimeError("balanced Test count drift")
    report = json.loads((TEST / "results.json").read_text())
    if report["status"] != "PASS" or report["sample_manifest_sha256"] != sha(MANIFEST):
        raise RuntimeError("Test report/protocol hash mismatch")
    if report["known_test_samples"] != len(target):
        raise RuntimeError("Test count mismatch")
    if not (TEST / "SUCCESS").is_file():
        raise RuntimeError("Test success marker absent")
    with (TEST / "sample_predictions.csv").open(newline="", encoding="utf-8") as handle:
        predictions = list(csv.DictReader(handle))
    ids = sorted(target)
    if [r["flow_id"] for r in predictions] != ids:
        raise RuntimeError("Test prediction IDs/order mismatch")
    if any(r["true_class"] != target[r["flow_id"]] for r in predictions):
        raise RuntimeError("Test labels mismatch frozen manifest")
    truth = [r["true_class"] for r in predictions]
    pred = [r["predicted_class"] for r in predictions]
    logits = np.load(TEST / "logits.npy", allow_pickle=False)
    if logits.shape != (len(ids), len(CLASSES)) or not np.isfinite(logits).all():
        raise RuntimeError("Test logits shape/finiteness failed")
    if [CLASSES[i] for i in logits.argmax(1)] != pred:
        raise RuntimeError("saved logits do not replay saved predictions")
    score = {"accuracy": float(accuracy_score(truth, pred)),
             "macro_f1": float(f1_score(truth, pred, labels=CLASSES, average="macro", zero_division=0)),
             "weighted_f1": float(f1_score(truth, pred, labels=CLASSES, average="weighted", zero_division=0))}
    for key, value in score.items():
        if abs(value - report[key]) > 1e-12:
            raise RuntimeError(f"Test {key} replay failed")
    p, r, f, n = precision_recall_fscore_support(truth, pred, labels=CLASSES, zero_division=0)
    per = [{"class": name, "precision": float(p[i]), "recall": float(r[i]),
            "f1": float(f[i]), "support": int(n[i])} for i, name in enumerate(CLASSES)]
    for reference, observed in zip(report["per_class"], per, strict=True):
        if reference["class"] != observed["class"] or any(
                abs(reference[key] - observed[key]) > 1e-12 for key in ("precision", "recall", "f1", "support")):
            raise RuntimeError("per-class Test replay failed")
    for branch in ("trafficformer", "graph", "yatc"):
        sub = RUN / branch
        meta = json.loads((sub / "metrics.json").read_text())
        if meta["status"] != "PASS" or not (sub / "SUCCESS").is_file() or sha(sub / "model_best.pt") != meta["checkpoint_sha256"]:
            raise RuntimeError(f"branch checkpoint/hash failure: {branch}")
    head = json.loads((RUN / "T0_equal/known_validation_metrics.json").read_text())
    for name, expected in head["checkpoint_hashes"].items():
        if sha(RUN / "T0_equal" / name) != expected:
            raise RuntimeError(f"fusion checkpoint/hash failure: {name}")
    test_audit = json.loads((ROOT / "cicids2017/input_caches/ustc/A-2/tf_fig_test/cache_audit.json").read_text())
    if test_audit["status"] != "PASS" or test_audit["flows"] != len(ids) or test_audit["sample_manifest_sha256"] != sha(MANIFEST):
        raise RuntimeError("balanced Test packet extraction audit failed")
    if sha(SOURCE / "cicids2017_three_class_manifest_v2.csv") != audit["source_manifest_sha256"]:
        raise RuntimeError("Stage34 original manifest drift")
    return report, score, head, per


def append_locked(path: Path, text: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.write(text)
        handle.flush()
        fcntl.flock(handle, fcntl.LOCK_UN)


def main() -> None:
    report, score, head, per = replay()
    audit = json.loads(AUDIT.read_text())
    counts = audit["balanced_counts"]
    original_cic = SOURCE / "cicids2017/runs/ustc/A-2/known_test_evaluation/results.json"
    if original_cic.exists():
        raise RuntimeError("old CIC Test became available; use paired comparison before finalizing")
    table = "\n".join(f"| {row['class']} | {row['support']} | {row['precision']:.6f} | {row['recall']:.6f} | {row['f1']:.6f} |" for row in per)
    content = f"""# Stage34B — CIC benign/malicious 1:1 retest

- Status: `complete / PASS`
- Experiment type: `benchmark`; claim scope: `development diagnostic`
- Objective: retest the same three-view closed-set model after BENIGN is sampled to the total attack count.

## Data and split

- Original Stage34 five-minute group-disjoint split retained. Original manifest SHA256: `{audit['source_manifest_sha256']}`. Balanced subset manifest SHA256: `{audit['balanced_manifest_sha256']}`.
- All Slowhttptest and PortScan flows retained. BENIGN chosen by seed-2022 SHA256 rank separately in each split, before Test outcomes existed.
- Train BENIGN/Slowhttptest/PortScan: {counts['BENIGN|train']}/{counts['DoS Slowhttptest|train']}/{counts['PortScan|train']} ({2*counts['BENIGN|train']:,} total).
- Validation: {counts['BENIGN|validation']}/{counts['DoS Slowhttptest|validation']}/{counts['PortScan|validation']} ({2*counts['BENIGN|validation']:,} total).
- Test: {counts['BENIGN|test']}/{counts['DoS Slowhttptest|test']}/{counts['PortScan|test']} ({2*counts['BENIGN|test']:,} total).

## Configuration and execution

- Same Stage31/34 TrafficFormer, FIG/TAGCN, YaTC and equal-feature-fusion recipe, seeds, optimizer, epochs and Known-Validation selection; only sample membership changed.
- Physical GPUs 0/1, at most two concurrent branch jobs. Named `stage34b_balanced_queue_0926` orchestrated all project-local tmux worker sessions; see `queue_progress.json` and `.tmux-task/stage34b_*/`.
- Unknown use=0. Test used only for one final evaluation after all branch and fusion checkpoint hashes were fixed.

## Core results

| Known Test samples | Accuracy | Macro-F1 | Weighted-F1 | Fusion Known-Val Macro-F1 |
|---:|---:|---:|---:|---:|
| {report['known_test_samples']:,} | {score['accuracy']:.6f} | {score['macro_f1']:.6f} | {score['weighted_f1']:.6f} | {head['metrics']['macro_f1']:.6f} |

| Class | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
{table}

## Preserved evidence

- `balanced_manifest.csv`, `balanced_manifest_audit.json`, exact-ID Train/Val caches, raw-PCAP extracted balanced Test cache, all three branch histories/checkpoints/features, fusion history/checkpoints, Test predictions/logits, `completion_verification.json` and `manifest.json`. Execution logs and exit statuses reside under project-local `.tmux-task/`.

## Limitations

- The user stopped the original CIC training before its fusion/Test result. Thus there is no completed old model on these exact Test flows and no valid paired performance delta.
- Attack classes are not individually balanced; Slowhttptest Test has only 132 flows. Slowhttptest and PortScan each remain tied to one capture day/PCAP, so high scores may exploit capture/endpoint artifacts.
- Changing the Test class prior changes Accuracy and Weighted-F1 interpretation. This is a one-seed development diagnostic, not independent validation or Unknown detection.

## Conclusion and next step

- Balanced Test Macro-F1 is `{score['macro_f1']:.6f}`. Inspect branch training curves and per-class recall before deciding whether any optimization change is warranted. Stage34 historical assets remain preserved as interrupted evidence.
"""
    (ROOT / "RESULTS.md").write_text(content)
    manifest_path = ROOT / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update({"status": "running", "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                     "execution": {"tmux_session": "stage34b_balanced_queue_0926",
                                   "command": "python stage34b_cic_balanced_retest/run_queue.py",
                                   "exit_code": 0,
                                   "environment": "/home/birkenwald/data/SEU-WXY/conda_envs/2025-10-8-WXY-dgl_py310",
                                   "physical_gpu_ids": [0, 1]},
                     "core_results": [{"dataset": "CIC-IDS-2017", "protocol": "balanced 3-class group-disjoint",
                                       "test_samples": report["known_test_samples"], **score}],
                     "next_step": "Stop; interpret one-seed balanced Test result and preserve old interrupted evidence."})
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    completion = ROOT / "completion_verification.json"
    completion.write_text(json.dumps({"status": "CHECKING", "test_samples": report["known_test_samples"],
                                      "source_manifest_unchanged": True, "balanced_manifest_unchanged": True,
                                      "saved_logit_metric_replay": True, "all_checkpoint_hashes_valid": True,
                                      "unknown_samples_used": 0, "test_parameter_selection": 0}, indent=2) + "\n")
    def bundle_check():
        subprocess.run([PYTHON, str(SKILL / "refresh_artifact_manifest.py"), str(ROOT)], check=True, cwd=PROJECT)
        subprocess.run([PYTHON, str(SKILL / "validate_experiment_bundle.py"), str(ROOT), "--verify-hashes"], check=True, cwd=PROJECT)
    bundle_check()
    completed = json.loads(completion.read_text())
    completed["status"] = "PASS"
    completed["accuracy"] = score["accuracy"]
    completed["macro_f1"] = score["macro_f1"]
    completed["weighted_f1"] = score["weighted_f1"]
    completion.write_text(json.dumps(completed, indent=2) + "\n")
    manifest = json.loads(manifest_path.read_text())
    manifest["status"] = "success"
    manifest["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    progress_path = ROOT / "queue_progress.json"
    progress = json.loads(progress_path.read_text())
    progress.update({"status": "PASS", "phase": "COMPLETE",
                     "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                     "completion_verification": str(completion)})
    progress_path.write_text(json.dumps(progress, indent=2) + "\n")
    bundle_check()
    append_locked(PROJECT / "EXPERIMENT_RESULTS.md",
                  f"\n- `stage34b-cic-benign-malicious-1to1-20260926` terminal: `success / single-seed diagnostic`; "
                  f"CIC balanced Test n={report['known_test_samples']}, Accuracy/Macro-F1/Weighted-F1 "
                  f"`{score['accuracy']:.6f}/{score['macro_f1']:.6f}/{score['weighted_f1']:.6f}`; "
                  "original CIC fusion/Test was stopped before completion, so no paired delta exists. "
                  "[bundle](stage34b_cic_balanced_retest/RESULTS.md).\n")
    append_locked(PROJECT / "EXECUTION_PROGRESS.md",
                  f"\nStage34B terminal update {datetime.now(timezone.utc).isoformat()}: status `complete / PASS`. "
                  f"Balanced CIC Test n={report['known_test_samples']}, Accuracy/Macro-F1/Weighted-F1="
                  f"`{score['accuracy']:.6f}/{score['macro_f1']:.6f}/{score['weighted_f1']:.6f}`; "
                  "saved logits, per-class metrics, sample IDs, branch/head SHA256 and experiment bundle verified. "
                  "Old Stage34 CIC fusion/Test had been stopped by user before completion; no paired delta is claimed. "
                  "See `stage34b_cic_balanced_retest/RESULTS.md`.\n")
    print(json.dumps({"status": "PASS", "known_test_samples": report["known_test_samples"], **score}), flush=True)


if __name__ == "__main__":
    main()
