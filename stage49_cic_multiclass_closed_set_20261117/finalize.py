#!/usr/bin/env python3
"""Independently replay the saved final classifier, then seal evidence."""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

import protocol

ROOT, PROJECT = protocol.ROOT, protocol.PROJECT
RUN = ROOT / "cicids2017/runs/ustc/A-2"
TEST = RUN / "known_test_evaluation"


def main():
    source_after = protocol.verify_sources()
    rows = protocol.verified_rows()
    selected = sorted((r["flow_id"], r["label"]) for r in rows if r["split"] == "test")
    with (TEST / "sample_predictions.csv").open(newline="", encoding="utf-8") as f:
        saved = list(csv.DictReader(f))
    if [(r["flow_id"], r["true_class"]) for r in saved] != selected:
        raise RuntimeError("Test IDs/labels/order do not match the frozen manifest")
    labels = list(protocol.CLASSES)
    label_index = {name: i for i, name in enumerate(labels)}
    truth = np.array([label_index[r["true_class"]] for r in saved])
    pred = np.array([label_index[r["predicted_class"]] for r in saved])
    logits = np.load(TEST / "logits.npy", allow_pickle=False)
    if logits.shape != (len(saved), len(labels)) or not np.isfinite(logits).all() or not np.array_equal(logits.argmax(1), pred):
        raise RuntimeError("saved logits do not replay saved predictions")
    metric = {"accuracy": float(accuracy_score(truth, pred))}
    for average in ("macro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(truth, pred, labels=range(len(labels)),
                                                   average=average, zero_division=0)
        metric.update({f"{average}_precision": float(p), f"{average}_recall": float(r), f"{average}_f1": float(f)})
    report = json.loads((TEST / "results.json").read_text())
    for key in ("accuracy", "macro_f1", "weighted_f1"):
        if abs(metric[key] - report[key]) > 1e-12:
            raise RuntimeError(f"saved metric discrepancy: {key}")
    p, r, f, support = precision_recall_fscore_support(truth, pred, labels=range(len(labels)), zero_division=0)
    per = [{"class": name, "precision": float(p[i]), "recall": float(r[i]),
            "f1": float(f[i]), "support": int(support[i])} for i, name in enumerate(labels)]
    for item, old in zip(per, report["per_class"], strict=True):
        if item["class"] != old["class"] or any(abs(item[key] - old[key]) > 1e-12 for key in ("precision", "recall", "f1", "support")):
            raise RuntimeError("per-class metric discrepancy")
    before = json.loads((ROOT / "checkpoint_hashes_before_test.json").read_text())
    after = json.loads((ROOT / "checkpoint_hashes_after_test.json").read_text())
    if before != after or any(protocol.sha(ROOT / name) != h for name, h in after.items()):
        raise RuntimeError("checkpoint hash changed")
    branch_epochs = {}
    for name, budget in (("trafficformer", 20), ("graph", 50), ("yatc", 200)):
        metadata = json.loads((RUN / name / "metrics.json").read_text())
        if metadata["status"] != "PASS" or metadata["completed_epochs"] != budget:
            raise RuntimeError(f"incomplete branch: {name}")
        branch_epochs[name] = metadata["best_epoch"]
    head = json.loads((RUN / "T0_equal/known_validation_metrics.json").read_text())
    protocol_config = json.loads((ROOT / "protocol.json").read_text())
    run_row = {"dataset": "CIC-IDS-2017", "method": "Three-view equal-feature fusion", "classes": len(labels),
               "seed": protocol.SEED, **{f"{role}_samples": n for role, n in protocol_config["split_counts"].items()},
               **metric, "protocol_sha256": protocol.sha(ROOT / "protocol.json"), "status": "PASS"}
    protocol.write_csv(ROOT / "run_results.csv", tuple(run_row), [run_row])
    protocol.write_csv(ROOT / "per_class_results.csv", tuple(per[0]), per)
    cm = confusion_matrix(truth, pred, labels=range(len(labels)))
    with (ROOT / "confusion_matrix.csv").open("x", newline="", encoding="utf-8") as out:
        writer = csv.writer(out)
        writer.writerow(("true_class", *labels))
        for name, values in zip(labels, cm, strict=True):
            writer.writerow((name, *map(int, values)))
    short = [name.replace("Web Attack - ", "Web ").replace("DoS ", "") for name in labels]
    normalized = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
    ax.set(xticks=range(len(labels)), yticks=range(len(labels)), xticklabels=short, yticklabels=short,
           xlabel="Predicted class", ylabel="True class",
           title=f"CIC-IDS-2017 eleven-class closed set | Macro-F1 {metric['macro_f1']:.2%}")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8,
                    color="white" if normalized[i, j] > .5 else "black")
    fig.colorbar(im, ax=ax, label="Fraction of each true class")
    fig.tight_layout()
    fig.savefig(ROOT / "confusion_matrix.png", dpi=300)
    fig.savefig(ROOT / "confusion_matrix.pdf")
    plt.close(fig)
    protocol.write_json(ROOT / "source_hashes_after.json", source_after)
    verification = {"status": "PASS", "classes": len(labels), "test_samples": len(saved),
                    "saved_logit_metric_replay": True, "per_class_replay": True,
                    "group_disjoint_protocol": True, "source_hashes_unchanged": len(source_after),
                    "checkpoint_hashes_unchanged": len(after), "unknown_samples_used": 0,
                    "test_parameter_selection": 0, **metric}
    protocol.write_json(ROOT / "completion_verification.json", verification)
    queue = json.loads((ROOT / "queue_progress.json").read_text())
    queue.update(status="SUCCESS", phase="COMPLETE", updated_at_utc=datetime.now(timezone.utc).isoformat(),
                 final_metrics=metric)
    queue["jobs"]["finalize"].update(state="verified")
    protocol.write_json(ROOT / "queue_progress.json", queue)
    table = "\n".join(f"|{item['class']}|{item['support']}|{item['precision']:.4%}|{item['recall']:.4%}|{item['f1']:.4%}|" for item in per)
    (ROOT / "RESULTS.md").write_text(f"""# Stage49 — CIC eleven-class closed-set result

- Status: success / independently replayed PASS
- Seed:2022; current three-view method, no baseline retraining.

## Data and split

Frozen eleven-class flow task. Train/Validation/Test={protocol_config['split_counts']}. Entire five-minute source-PCAP groups are disjoint. Original matched pool2087440; label-disagreement quarantine783. Large attack classes10% per split, smaller attacks all retained; BENIGN equals total attacks in each role. Full counts and exclusions are in class_audit.csv and split_counts.csv.

## Configuration and execution

Unchanged inherited TF20/FIG50/YaTC200/adapter30/head30 recipe. Official TF/YaTC pretraining; Validation-only selection and Train-only normalization. Branch best epochs={branch_epochs}. Internal ustc/A-2 keys are documented path aliases, not USTC data. At most two capacity-selected GPU workers.

## Core results

Accuracy={metric['accuracy']:.6%}; Macro-F1={metric['macro_f1']:.6%}; Weighted-F1={metric['weighted_f1']:.6%}.

|Class|Test support|Precision|Recall|F1|
|---|---:|---:|---:|---:|
{table}

## Preserved evidence

Frozen protocol/members/group assignments, label quarantine, packet-cache audits, branch/adapter/head checkpoints and histories, per-sample Test predictions/logits, confusion matrices, independent replay, source/checkpoint hashes, task logs and manifest. Logs/statuses remain under project-local .tmux-task/stage49_cic_*.

## Limitations

Single-seed development experiment on previously exposed CIC data; not untouched validation. Attack classes still share capture-day/endpoint shortcuts. BENIGN-versus-attacks balancing does not equalize individual attacks. Excluded tiny classes and GoldenEye are explicitly listed. Historical Stage34B99.75% Macro-F1 describes a different three-class task, not a matched performance comparator.

## Conclusion and next step

Completed the preregistered eleven-class retest. Inspect per-class errors; no automatic class removal, split reselection, baseline run or open-set experiment.
""")
    manifest = json.loads((ROOT / "manifest.json").read_text())
    manifest.update(status="success", updated_at_utc=datetime.now(timezone.utc).isoformat(),
                    execution={"tmux_session": "stage49_cic_queue_20261117", "command": "python stage49_cic_multiclass_closed_set_20261117/run_queue.py",
                               "environment": str(Path(sys.executable).resolve().parent.parent),
                               "physical_gpu_ids": sorted({j['physical_gpu_id'] for j in queue['jobs'].values() if j.get('physical_gpu_id') is not None})},
                    inputs=[{"path": str(ROOT / "protocol.json"), "sha256": protocol.sha(ROOT / "protocol.json")}],
                    configuration={"files": ["PLAN.md", "protocol.json"], "parameters": {"num_classes": 11}, "seeds": [2022]},
                    core_results=[run_row], limitations=["single-seed development task", "class-day/endpoint confounding"],
                    next_step="Review recorded result; no automatic follow-up training")
    for job in queue["jobs"].values():
        if job["session"].endswith("finalize_20261117"):
            continue
        for name in ("output.log", "exit.status"):
            path = PROJECT / ".tmux-task" / job["session"] / name
            if path.is_file():
                manifest["artifacts"].append({"path": str(path), "scope": "external", "role": "execution-log",
                                              "size_bytes": path.stat().st_size, "sha256": protocol.sha(path), "hash_status": "verified"})
    protocol.write_json(ROOT / "manifest.json", manifest)
    tools_dir = PROJECT.parents[1] / "skills/experiment-data-preservation/scripts"
    subprocess.run([sys.executable, str(tools_dir / "refresh_artifact_manifest.py"), str(ROOT)], check=True)
    subprocess.run([sys.executable, str(tools_dir / "validate_experiment_bundle.py"), str(ROOT), "--verify-hashes"], check=True)
    print(json.dumps(verification), flush=True)


if __name__ == "__main__":
    main()
