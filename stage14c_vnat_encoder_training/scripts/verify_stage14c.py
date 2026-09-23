#!/usr/bin/env python3
"""Fresh completion verification for VNAT Stage 14C."""
from __future__ import annotations
import csv, json, math
from pathlib import Path
import torch
from common import CHECKPOINT_ROOT, EXPECTED_FREEZE_HASH, PROTOCOL_SEEDS, SETTINGS, STAGE14C_ROOT, load_protocol, sha256_file, verify_freeze

def main():
    freeze=verify_freeze()
    required=["stage14c_training_summary.csv","stage14c_run_manifest.csv","stage14c_encoder_training.md","manifest.json","RESULTS.md","final_verification.json"]
    missing=[name for name in required if not (STAGE14C_ROOT/name).is_file()]
    summary=list(csv.DictReader((STAGE14C_ROOT/"stage14c_training_summary.csv").open()))
    identities={(r["setting"],int(r["seed"])) for r in summary}
    expected={(s,seed) for s in SETTINGS for seed in PROTOCOL_SEEDS}
    failures=[]
    if missing: failures.append(f"missing outputs: {missing}")
    if len(summary)!=15 or identities!=expected: failures.append("summary protocol grid mismatch")
    if any(r["status"]!="success" for r in summary): failures.append("non-success summary row")
    checkpoint_hashes=[]
    for row in summary:
        pid=f"{row['setting'].lower()}_seed{row['seed']}"; protocol=load_protocol(pid)
        input_dir=STAGE14C_ROOT/"runs"/pid/"inputs"; audit=json.loads((input_dir/"input_audit.json").read_text())
        selection=json.loads((STAGE14C_ROOT/"runs"/pid/"checkpoint_selection.json").read_text())
        checkpoint=Path(row["checkpoint_path"])
        actual=sha256_file(checkpoint); checkpoint_hashes.append(actual)
        if actual!=row["checkpoint_sha256"] or actual!=selection["checkpoint_sha256"]: failures.append(f"{pid}: checkpoint hash")
        payload=torch.load(checkpoint,map_location="cpu",weights_only=False); config=payload["config"]
        checks=[
            config["freeze_hash"]==EXPECTED_FREEZE_HASH,
            set(config["known_classes"])==set(protocol["known_applications"]),
            set(config["unknown_classes_excluded"])==set(protocol["unknown_applications"]),
            config["unknown_samples_loaded"]==0, config["known_test_samples_loaded"]==0,
            audit["unknown_samples_used_in_training"]==0, audit["unknown_samples_used_in_validation"]==0,
            audit["known_test_samples_used"]==0,
            audit["known_train_class_intersection_unknown"]==[], audit["known_validation_class_intersection_unknown"]==[],
            audit["train_validation_flow_overlap"]==0,
            selection["unknown_inference_executed"] is False,
            all(math.isfinite(float(row[k])) for k in ["train_loss","val_loss","val_accuracy","val_macro_f1"]),
            not (input_dir/"test_images.npy").exists(), not (input_dir/"unknown_images.npy").exists(),
        ]
        if not all(checks): failures.append(f"{pid}: integrity/config gate")
    if len(checkpoint_hashes)!=15 or len(set(checkpoint_hashes))!=15: failures.append("checkpoint count/uniqueness")
    manifest=json.loads((STAGE14C_ROOT/"manifest.json").read_text())
    for artifact in manifest["artifacts"]:
        path=STAGE14C_ROOT/artifact["path"]
        if not path.is_file() or sha256_file(path)!=artifact["sha256"]: failures.append(f"manifest artifact: {artifact['path']}")
    final=json.loads((STAGE14C_ROOT/"final_verification.json").read_text())
    if final["status"]!="PASS" or final["successful_runs"]!=15: failures.append("aggregate verification")
    result={"status":"PASS" if not failures else "FAIL","failures":failures,"runs":len(summary),"checkpoint_files":len(list(CHECKPOINT_ROOT.glob("*_best.pt"))),"unique_checkpoint_hashes":len(set(checkpoint_hashes)),"unknown_train_samples_used":0,"unknown_validation_samples_used":0,"known_test_samples_used":0,"unknown_inference_executed":False,"freeze_hash":freeze["freeze_hash"],"stage14d_started":False}
    (STAGE14C_ROOT/"completion_verification.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps(result,sort_keys=True))
    if failures: raise SystemExit(1)

if __name__=="__main__": main()
