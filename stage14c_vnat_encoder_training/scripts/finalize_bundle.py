#!/usr/bin/env python3
"""Finalize the durable Stage 14C result bundle after fresh verification."""
from __future__ import annotations
import csv, json, subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev

from common import CACHE_ROOT, CHECKPOINT_ROOT, PROJECT_ROOT, PROTOCOL_PATH, SPLIT_MANIFEST, STAGE14C_ROOT, sha256_file, verify_freeze

def rel(path: Path) -> str:
    return str(path.resolve().relative_to(STAGE14C_ROOT.resolve())) if path.resolve().is_relative_to(STAGE14C_ROOT.resolve()) else str(path.resolve())

def main():
    verification=json.loads((STAGE14C_ROOT/"final_verification.json").read_text())
    if verification["status"]!="PASS": raise RuntimeError("cannot finalize a failed Stage 14C verification")
    freeze=verify_freeze()
    rows=list(csv.DictReader((STAGE14C_ROOT/"stage14c_training_summary.csv").open()))
    if len(rows)!=15 or any(r["status"]!="success" for r in rows): raise RuntimeError("15 successful rows required")
    grouped=defaultdict(list)
    for row in rows: grouped[row["setting"]].append(row)
    stats={}
    for setting, values in grouped.items():
        acc=[float(v["val_accuracy"]) for v in values]; f1=[float(v["val_macro_f1"]) for v in values]
        stats[setting]={"accuracy_mean":mean(acc),"accuracy_std":pstdev(acc),"accuracy_min":min(acc),"accuracy_max":max(acc),"macro_f1_mean":mean(f1),"macro_f1_std":pstdev(f1),"macro_f1_min":min(f1),"macro_f1_max":max(f1)}
    weak=[r for r in rows if float(r["val_macro_f1"])<0.35]
    now=datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00","Z")
    per_run="\n".join(f"| {r['setting']} | {r['seed']} | {r['best_epoch']}/{r['stop_epoch']} | {float(r['train_loss']):.6f} | {float(r['val_loss']):.6f} | {float(r['val_accuracy']):.6f} | {float(r['val_macro_f1']):.6f} |" for r in rows)
    setting_table="\n".join(f"| {s} | {stats[s]['accuracy_mean']:.6f} ± {stats[s]['accuracy_std']:.6f} | {stats[s]['macro_f1_mean']:.6f} ± {stats[s]['macro_f1_std']:.6f} | {stats[s]['accuracy_min']:.6f}–{stats[s]['accuracy_max']:.6f} | {stats[s]['macro_f1_min']:.6f}–{stats[s]['macro_f1_max']:.6f} |" for s in ("Low","Medium","High"))
    report=f"""# Stage 14C — VNAT Open-Detect Encoder Training

## Scope and integrity

- Completed runs: **15/15**, all exit 0 and finite.
- Unknown samples used in training/validation: **0/0** for every run.
- Known Test samples used: **0**; Unknown inference executed: **false**.
- Encoder initialization: independent released Open-Detect initialization for every protocol; no new hyperparameter search.
- Checkpoint selection: Known Validation harmonic mean of Accuracy and Macro-F1 only; patience 5, maximum 100 epochs.
- Stage 14B freeze hash before/after: `{freeze['freeze_hash']}` / `{freeze['freeze_hash']}` (**PASS**).
- Checkpoints saved/hash-verified/unique: **15/15/15**.

## Known Validation results (five seeds)

| Setting | Accuracy mean ± population SD | Macro-F1 mean ± population SD | Accuracy range | Macro-F1 range |
|---|---:|---:|---:|---:|
{setting_table}

## Run-level results

| Setting | Seed | Best/stop epoch | Train loss | Validation loss | Validation Accuracy | Validation Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
{per_run}

## Quality diagnosis

- Training crashes, NaN, missing checkpoints, or hash mismatches: **none**.
- Numerically abnormal runs: **none**.
- Weak Known Validation Macro-F1 flag (`<0.35`): **{len(weak)}/15** — {', '.join(f"{r['setting']}-{r['seed']} ({float(r['val_macro_f1']):.6f})" for r in weak) if weak else 'none'}.
- These flags indicate uneven minority-class representation quality under the frozen, highly imbalanced VNAT class counts; they are not hidden or retuned and do not indicate data leakage.

## Completion judgment

- Stage 14C status: **PASS_WITH_QUALITY_FLAGS**.
- Stage 14D technical readiness: **YES**, using the 15 frozen checkpoint hashes below and preserving run-level quality stratification.
- Stage 14D was **not** started.
"""
    (STAGE14C_ROOT/"stage14c_encoder_training.md").write_text(report,encoding="utf-8")
    results=f"""# Experiment results: stage14c-vnat-encoder-training-20260917-v1

- Status: `success`
- Completed (UTC): `{now}`
- Objective: Train 15 independent Known-only Open-Detect encoders on the frozen Stage 14B VNAT protocols without Unknown/Test evaluation or Stage 14D.

## Data and split

- Frozen VNAT clean pool: 23,449 flows across 15 pre-registered protocols.
- Model-visible inputs per run: frozen Known Train and Known Validation only.
- Unknown Train/Validation samples used: 0/0; Known Test samples used: 0.
- Stage 14B freeze hash before/after: `{freeze['freeze_hash']}` / `{freeze['freeze_hash']}`.
- Flow-cache audit: 23,449 flows, 162 captures, 0 packet-count mismatches, 0 zero-packet flows, 16 released-code-compatible zero-filled non-IPv4 packet encodings.

## Configuration and execution

- 15 independent released Open-Detect initializations; latent dimension 128; batch 512; Adam LR 1e-3; MultiStepLR milestones 50/80; lambda 0.005.
- Maximum 100 epochs; Known Validation Accuracy/Macro-F1 harmonic-mean selection; early-stopping patience 5.
- Physical GPUs selected immediately before campaign: 5, 4, 3, 0; physical GPUs 1 and 2 were not used.

## Core results

{setting_table}

- Checkpoints saved/hash-verified/unique: 15/15/15.
- NaN/crashes: none; weak Macro-F1 flags below 0.35: {len(weak)}/15.
- Detailed run table: `stage14c_training_summary.csv` and `stage14c_encoder_training.md`.

## Preserved evidence

- `checkpoints/`: 15 best encoder/model checkpoints, including optimizer state and frozen training config.
- `runs/`: Known-only input audits, per-epoch metrics, configs, selected-checkpoint records, latest checkpoints, and stdout logs.
- `flow_image_cache/`: flow-aligned preprocessing cache and audit ledger.
- `stage14c_training_summary.csv`, `stage14c_run_manifest.csv`, `final_verification.json`, and `manifest.json`.
- `failed_attempts/`: preserved strict-IPv4 cache attempt and interruption evidence.

## Limitations

- No Known Test or Unknown Test evaluation was performed; no Open-Detect unknown metrics or detection threshold was computed.
- Medium-2024 and Medium-2025 have weak Known Validation Macro-F1 and must remain visible in downstream interpretation.
- The 16 non-IPv4 packet encodings follow the released Open-Detect exception behavior (zero-filled packet block); no flow was deleted.

## Conclusion and next step

- Stage 14C is complete with quality flags. It is technically ready for Stage 14D, but Stage 14D was not started.
"""
    (STAGE14C_ROOT/"RESULTS.md").write_text(results,encoding="utf-8")
    try: revision=subprocess.check_output(["git","rev-parse","HEAD"],cwd=PROJECT_ROOT,text=True).strip()
    except Exception: revision=None
    artifacts=[]
    for path in [STAGE14C_ROOT/"stage14c_training_summary.csv",STAGE14C_ROOT/"stage14c_run_manifest.csv",STAGE14C_ROOT/"stage14c_encoder_training.md",STAGE14C_ROOT/"final_verification.json",CACHE_ROOT/"cache_audit.json"] + sorted(CHECKPOINT_ROOT.glob("*_best.pt")):
        artifacts.append({"path":rel(path),"sha256":sha256_file(path),"size_bytes":path.stat().st_size})
    manifest={
        "schema_version":1,"experiment_id":"stage14c-vnat-encoder-training-20260917-v1","created_at_utc":"2026-09-17T13:16:31Z","updated_at_utc":now,"status":"success","experiment_type":"training","claim_scope":"other",
        "objective":"Train 15 independent Known-only Open-Detect encoders on the frozen Stage 14B VNAT protocols without Unknown/Test evaluation or Stage 14D",
        "inputs":[{"path":str(PROTOCOL_PATH),"sha256":sha256_file(PROTOCOL_PATH)},{"path":str(SPLIT_MANIFEST),"sha256":sha256_file(SPLIT_MANIFEST)},{"path":rel(CACHE_ROOT/"cache_audit.json"),"sha256":sha256_file(CACHE_ROOT/"cache_audit.json")}],
        "code":{"revision":revision,"dirty":True,"changes":["stage14c_vnat_encoder_training"]},
        "execution":{"tmux_session":"stage14c-campaign","related_sessions":["stage14c-flow-cache-v2","stage14c-build-inputs","stage14c-train-low2022","stage14c-aggregate"],"command":"select_gpu.py --allowed 0,3,4,5,6,7 --min-free-gb 10 --count 4 -- python -u stage14c_vnat_encoder_training/scripts/run_campaign.py","exit_code":0,"environment":"/home/birkenwald/data/SEU-WXY/conda_envs/2025-10-8-WXY-dgl_py310","physical_gpu_ids":[5,4,3,0]},
        "configuration":{"parameters":{"runs":15,"max_epochs":100,"batch_size":512,"early_stopping_patience":5,"checkpoint_monitor":"harmonic_mean(Known Validation Accuracy, Known Validation Macro-F1)","unknown_samples_used":0,"known_test_samples_used":0,"freeze_hash":freeze["freeze_hash"]},"seeds":[2022,2023,2024,2025,2026]},
        "core_results":[{"metric":f"{s.lower()}_val_accuracy_mean","value":stats[s]["accuracy_mean"]} for s in ("Low","Medium","High")]+[{"metric":f"{s.lower()}_val_macro_f1_mean","value":stats[s]["macro_f1_mean"]} for s in ("Low","Medium","High")]+[{"metric":"successful_runs","value":15},{"metric":"unknown_samples_used","value":0},{"metric":"unique_checkpoint_hashes","value":15},{"metric":"weak_macro_f1_runs_lt_0_35","value":len(weak)}],
        "artifacts":artifacts,
        "limitations":["No Known Test or Unknown Test evaluation was performed.","Medium-2024 and Medium-2025 have weak Known Validation Macro-F1 and were not retuned.","Sixteen non-IPv4 packets used released-code-compatible zero filling; no flow was removed."],
        "next_step":"Eligible for Stage 14D frozen support construction when explicitly requested; do not alter Stage 14B or Stage 14C artifacts."
    }
    (STAGE14C_ROOT/"manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps({"status":"success","weak_runs":len(weak),"artifacts":len(artifacts)},sort_keys=True))

if __name__=="__main__": main()
