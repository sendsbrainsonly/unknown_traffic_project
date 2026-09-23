#!/usr/bin/env python3
"""Aggregate Stage 14C results without reading Known Test or Unknown samples."""
from __future__ import annotations
import csv, json, math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from common import CHECKPOINT_ROOT, STAGE14C_ROOT, protocols_by_id, sha256_file, verify_freeze

FIELDS=["setting","seed","num_known_classes","num_unknown_classes","train_samples","val_samples","best_epoch","stop_epoch","train_loss","val_loss","val_accuracy","val_macro_f1","checkpoint_path","checkpoint_sha256","protocol_hash","status"]

def main():
    before=verify_freeze(); protocols=protocols_by_id(); rows=[]
    for pid,p in sorted(protocols.items()):
        run=STAGE14C_ROOT/"runs"/pid; audit=json.loads((run/"inputs/input_audit.json").read_text()); sel=json.loads((run/"checkpoint_selection.json").read_text()); ck=Path(sel["checkpoint_path"])
        checks=[sel["checkpoint_sha256"]==sha256_file(ck),sel["freeze_hash"]==before["freeze_hash"],sel["unknown_samples_used"]==0,sel["known_test_samples_used"]==0,audit["unknown_samples_used_in_training"]==0,audit["unknown_samples_used_in_validation"]==0,all(math.isfinite(float(sel[k])) for k in ["train_loss","val_loss","val_accuracy","val_macro_f1"])]
        rows.append({"setting":p["setting"],"seed":p["seed"],"num_known_classes":len(p["known_applications"]),"num_unknown_classes":len(p["unknown_applications"]),"train_samples":audit["train_samples"],"val_samples":audit["validation_samples"],"best_epoch":sel["best_epoch"],"stop_epoch":sel["stop_epoch"],"train_loss":sel["train_loss"],"val_loss":sel["val_loss"],"val_accuracy":sel["val_accuracy"],"val_macro_f1":sel["val_macro_f1"],"checkpoint_path":str(ck),"checkpoint_sha256":sel["checkpoint_sha256"],"protocol_hash":sel["freeze_hash"],"status":"success" if all(checks) else "failed"})
    with (STAGE14C_ROOT/"stage14c_training_summary.csv").open("w",encoding="utf-8",newline="") as h:
        w=csv.DictWriter(h,fieldnames=FIELDS);w.writeheader();w.writerows(rows)
    by=defaultdict(list)
    for r in rows: by[r["setting"]].append(r)
    setting_lines=[]
    for setting in ("Low","Medium","High"):
        values=by[setting]; setting_lines.append(f"| {setting} | {len(values)} | {sum(float(x['val_accuracy']) for x in values)/len(values):.6f} | {sum(float(x['val_macro_f1']) for x in values)/len(values):.6f} |")
    after=verify_freeze(); all_ok=len(rows)==15 and all(r["status"]=="success" for r in rows) and before==after and len({r["checkpoint_sha256"] for r in rows})==15
    report=f"""# Stage 14C — VNAT Open-Detect Encoder Training

## Scope and integrity

- Completed runs: **{sum(r['status']=='success' for r in rows)}/15**.
- Unknown samples used in training/validation: **0/0** for every run.
- Known Test samples used: **0**; Unknown inference executed: **false**.
- Encoder initialization: independent released Open-Detect initialization for every protocol.
- Checkpoint selection: Known Validation harmonic mean of Accuracy and Macro-F1 only; patience 5.
- Stage 14B freeze hash before/after: `{before['freeze_hash']}` / `{after['freeze_hash']}` (**{'PASS' if before==after else 'FAIL'}**).

## Known Validation results (five-seed mean)

| Setting | Runs | Accuracy | Macro-F1 |
|---|---:|---:|---:|
{chr(10).join(setting_lines)}

## Completion judgment

- NaN/training crashes: **{'none' if all_ok else 'see failed rows'}**.
- Saved checkpoints with verified SHA-256: **{sum(r['status']=='success' for r in rows)}/15**.
- Stage 14C status: **{'PASS' if all_ok else 'FAIL'}**.
- Stage 14D readiness: **{'YES' if all_ok else 'NO'}**. Stage 14D was not started.
"""
    (STAGE14C_ROOT/"stage14c_encoder_training.md").write_text(report,encoding="utf-8")
    final={"generated_at_utc":datetime.now(timezone.utc).isoformat(),"status":"PASS" if all_ok else "FAIL","runs":len(rows),"successful_runs":sum(r["status"]=="success" for r in rows),"unique_checkpoint_hashes":len({r["checkpoint_sha256"] for r in rows}),"unknown_train_samples_used":0,"unknown_validation_samples_used":0,"known_test_samples_used":0,"unknown_inference_executed":False,"freeze_before":before,"freeze_after":after}
    (STAGE14C_ROOT/"final_verification.json").write_text(json.dumps(final,indent=2,sort_keys=True)+"\n")
    print(json.dumps(final,sort_keys=True));
    if not all_ok: raise SystemExit(1)

if __name__=="__main__": main()
