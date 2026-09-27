#!/usr/bin/env python3
"""Independent replay and immutable-evidence summary for the matched USTC run."""
from __future__ import annotations

import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import joint_ustc as exp

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
WORKSPACE = PROJECT.parents[1]
BASELINE = exp.STAGE34 / "runs/ustc/A-2/known_test_evaluation/results.json"
RESULT = exp.RUN / "known_test_evaluation/comparison.json"
REPLAY = ROOT / "completion_verification.json"
PYTHON = "/home/birkenwald/data/SEU-WXY/conda_envs/2025-10-8-WXY-dgl_py310/bin/python"


def check_close(a: float, b: float, label: str) -> None:
    if abs(a - b) > 1e-9:
        raise RuntimeError(f"independent replay mismatch {label}: {a} vs {b}")


def main() -> None:
    if REPLAY.exists():
        raise FileExistsError(REPLAY)
    exp.code_guard()
    baseline, result = json.loads(BASELINE.read_text()), json.loads(RESULT.read_text())
    train = json.loads((exp.RUN / "train_summary.json").read_text())
    preflight = json.loads((ROOT / "preflight.json").read_text())
    if any(value["status"] != "PASS" for value in (baseline, result, train, preflight)):
        raise RuntimeError("input report not PASS")
    if exp.sha(BASELINE) != result["staged_result_sha256"]:
        raise RuntimeError("staged baseline result hash changed")
    if exp.sha(exp.RUN / "model_best.pt") != train["checkpoint_sha256"]:
        raise RuntimeError("joint checkpoint hash changed")
    if exp.source_guard()["sample_manifest_sha256"] != preflight["sample_manifest_sha256"]:
        raise RuntimeError("sample manifest hash changed")
    classes = result["classes"]
    rows_path = exp.RUN / "known_test_evaluation/sample_predictions.csv"
    with rows_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    logits = np.load(exp.RUN / "known_test_evaluation/joint_logits.npy", allow_pickle=False)
    n = len(rows)
    if n != 4333 or logits.shape != (n, len(classes)):
        raise RuntimeError("Test row or joint logit shape mismatch")
    if len({row["flow_id"] for row in rows}) != n:
        raise RuntimeError("duplicate Test flow ID")
    truth = np.asarray([classes.index(row["true_class"]) for row in rows], dtype=np.int64)
    joint = np.asarray([classes.index(row["joint_prediction"]) for row in rows], dtype=np.int64)
    staged = np.asarray([classes.index(row["staged_prediction"]) for row in rows], dtype=np.int64)
    if not np.array_equal(logits.argmax(1), joint):
        raise RuntimeError("saved joint logits do not replay predictions")
    for name, prediction in (("joint", joint), ("staged", staged)):
        score = exp.metrics(truth, prediction, classes)
        for metric in ("accuracy", "macro_f1", "weighted_f1"):
            check_close(score[metric], result[name][metric], f"{name}.{metric}")
    for metric in ("accuracy", "macro_f1", "weighted_f1"):
        check_close(result["staged"][metric], baseline[metric], f"staged baseline {metric}")
    if train["completed_epochs"] != exp.EPOCHS or train["test_feature_values_loaded"] != 0:
        raise RuntimeError("joint training budget or Test isolation violated")
    if result["unknown_samples_loaded"] != 0 or result["test_parameter_selection"] != 0:
        raise RuntimeError("Unknown/Test selection boundary violated")
    staged_run = exp.STAGE34 / "runs/ustc/A-2"
    staged_seconds = sum(json.loads((staged_run / name / "metrics.json").read_text())["elapsed_seconds"]
                         for name in ("trafficformer", "graph", "yatc"))
    staged_seconds += json.loads((staged_run / "T0_equal/verification.json").read_text())["elapsed_seconds"]
    gpu_log = (PROJECT / ".tmux-task/stage35_joint_train/output.log").read_text()
    selected_lines = [line.removeprefix("gpu_selection=") for line in gpu_log.splitlines()
                      if line.startswith("gpu_selection=")]
    if len(selected_lines) != 1:
        raise RuntimeError("missing unique live GPU selector evidence for joint training")
    gpu_id = json.loads(selected_lines[0])["selected_physical_ids"]
    if len(gpu_id) != 1:
        raise RuntimeError("joint training did not select exactly one physical GPU")
    summary = ROOT / "comparison_summary.csv"
    with summary.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("dataset", "protocol", "method", "known_test_samples", "accuracy", "macro_f1", "weighted_f1"))
        for method in ("staged", "joint"):
            value = result[method]
            writer.writerow(("USTC-TFC2016", result["protocol"], method, n,
                             value["accuracy"], value["macro_f1"], value["weighted_f1"]))
    with (ROOT / "paired_per_class.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("class", "support", "staged_f1", "joint_f1", "delta_joint_minus_staged"))
        for before, after in zip(result["staged"]["per_class"], result["joint"]["per_class"], strict=True):
            if before["class"] != after["class"] or before["support"] != after["support"]:
                raise RuntimeError("per-class report alignment mismatch")
            writer.writerow((before["class"], before["support"], before["f1"], after["f1"],
                             after["f1"] - before["f1"]))
    verification = {"status": "PASS", "matched_test_flows": n,
                    "independent_saved_logit_replay": True, "baseline_metric_replay": True,
                    "sample_manifest_sha256": preflight["sample_manifest_sha256"],
                    "baseline_result_sha256": result["staged_result_sha256"],
                    "joint_checkpoint_sha256": train["checkpoint_sha256"],
                    "unknown_samples_used": 0, "test_parameter_selection": 0,
                    "joint_physical_gpu_ids": gpu_id,
                    "staged_training_gpu_hours": staged_seconds/3600,
                    "joint_training_gpu_hours": train["elapsed_seconds"]/3600}
    REPLAY.write_text(json.dumps(verification, indent=2) + "\n")
    d = result["delta_joint_minus_staged"]
    report = f"""# Stage 35 — Three-view joint vs staged training

- Status: `success / MATCHED_USTC_SINGLE_SEED_DIAGNOSTIC`
- Experiment type: `ablation`; claim scope: `diagnostic`

## Data and split

USTC-TFC2016 A-2, 17 Known classes, seed 2022, same Stage 34 10% selected flows:
Known Train 34,665; Known Validation 4,333; matched Known Test {n:,}.
Sample-manifest SHA256: `{preflight['sample_manifest_sha256']}`.

## Configuration and execution

Both methods fuse TrafficFormer, TAGCN and YaTC **feature vectors**, then use one
final classifier; neither averages branch probabilities. The staged baseline
trains encoders for 20/50/200 epochs, then adapters/fusion for 30/30. The joint
run backpropagates final fused cross-entropy through all three encoders for
{train['completed_epochs']} epochs, effective batch 64, microbatch 32, and
selects its checkpoint solely by fused Known-Validation Macro-F1. Joint best
epoch: {train['best_epoch']}; best Val Macro-F1: {train['best_validation']['macro_f1']:.6f}.
Joint elapsed: {train['elapsed_seconds']/3600:.2f} h; peak allocated GPU memory:
{train['peak_gpu_memory_bytes']/2**30:.2f} GiB. Joint checkpoint SHA256:
`{train['checkpoint_sha256']}`.
Staged branch-plus-fusion measured GPU-time sum: {staged_seconds/3600:.2f} h;
joint training measured GPU time: {train['elapsed_seconds']/3600:.2f} h.

## Core results

| Training recipe | Known Test Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|
| Staged | {result['staged']['accuracy']:.6f} | {result['staged']['macro_f1']:.6f} | {result['staged']['weighted_f1']:.6f} |
| Joint end-to-end | {result['joint']['accuracy']:.6f} | {result['joint']['macro_f1']:.6f} | {result['joint']['weighted_f1']:.6f} |
| Joint − Staged | {d['accuracy']:+.6f} | {d['macro_f1']:+.6f} | {d['weighted_f1']:+.6f} |

Joint-only correct: {result['joint_only_correct']:,}; staged-only correct:
{result['staged_only_correct']:,}. Full per-class and sample-level comparisons
are in `paired_per_class.csv` and `runs/ustc/joint_e2e20/known_test_evaluation/`.

## Preserved evidence

`preflight.json`, both smoke-run result files, all training history and selected
checkpoint, `comparison_summary.csv`, `paired_per_class.csv`, `completion_verification.json`,
and named tmux logs under the project `.tmux-task/`. Independent saved-logit
replay and Stage 34 metric/hash comparisons: PASS. Unknown data used: 0; Test
selection/tuning: 0.

## Limitations

One protocol and one seed. Staged and joint allocation of epochs and losses
differs, so this is a comparison of fixed practical recipes, not a causal
isolation of gradient coupling. The Test is a development Test already exposed
to earlier project work, not untouched external validation. No CIC inference
is made from USTC.

## Conclusion and next step

On this matched USTC setting, the larger Macro-F1 belongs to
**{'joint end-to-end' if d['macro_f1'] > 0 else 'staged' if d['macro_f1'] < 0 else 'neither (tie)'}**.
Do not promote a universal training choice from this single setting.
"""
    (ROOT / "RESULTS.md").write_text(report)
    manifest_path = ROOT / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update({"updated_at_utc": datetime.now(timezone.utc).isoformat(), "status": "success",
                     "inputs": [{"path": str(exp.STAGE34 / "ustc_a2_10pct_manifest.csv"),
                                 "sha256": preflight["sample_manifest_sha256"], "role": "frozen matched flow manifest"}],
                     "code": {"revision": None, "dirty": True,
                              "changes": ["isolated Stage35 joint runner, queue and replay"]},
                     "execution": {"tmux_session": "stage35_joint_queue_two_gpu", "command": "python stage35_joint_vs_staged_training/run_queue.py (Stage34 two-GPU continuation dependency)",
                                   "exit_code": 0, "environment": PYTHON, "physical_gpu_ids": gpu_id},
                     "configuration": {"files": ["EXPERIMENT_PLAN.md", "runs/ustc/joint_e2e20/config.json"],
                                       "parameters": {"epochs": 20, "effective_batch": 64, "microbatch": 32,
                                                      "fusion": "three 64D features, fixed equal scaled concat"},
                                       "seeds": [2022]},
                     "core_results": [{"method": method, "accuracy": result[method]["accuracy"],
                                       "macro_f1": result[method]["macro_f1"],
                                       "weighted_f1": result[method]["weighted_f1"]}
                                      for method in ("staged", "joint")],
                     "limitations": ["one USTC protocol and seed", "unequal staged/joint training budgets and objectives",
                                     "historically exposed development Test"],
                     "next_step": "Report matched outcome; decide separately whether a matched-budget multi-seed study is warranted."})
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    preserve = WORKSPACE / "skills/experiment-data-preservation/scripts"
    subprocess.run([PYTHON, str(preserve / "refresh_artifact_manifest.py"), str(ROOT)], check=True)
    subprocess.run([PYTHON, str(preserve / "validate_experiment_bundle.py"), str(ROOT), "--verify-hashes"], check=True)
    index = PROJECT / "EXPERIMENT_RESULTS.md"
    line = ("\n- `stage35-joint-vs-staged-ustc-20260925-v1` terminal: "
            f"`success / MATCHED_USTC_SINGLE_SEED_DIAGNOSTIC`; matched n={n:,}; "
            f"Joint−Staged Test Macro-F1={d['macro_f1']:+.6f}, Accuracy={d['accuracy']:+.6f}; "
            "three feature branches and one classifier; no Unknown/Test tuning. "
            "[bundle](stage35_joint_vs_staged_training/RESULTS.md).\n")
    if "stage35-joint-vs-staged-ustc-20260925-v1` terminal:" not in index.read_text():
        with index.open("a") as stream:
            stream.write(line)
    print(json.dumps({"status": "PASS", "delta": d, "matched_test_flows": n}), flush=True)


if __name__ == "__main__":
    main()
