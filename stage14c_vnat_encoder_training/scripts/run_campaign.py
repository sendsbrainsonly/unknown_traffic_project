#!/usr/bin/env python3
"""Schedule the 15 independent VNAT encoder runs over selected physical GPUs."""

from __future__ import annotations
import csv, json, os, subprocess, sys, time
from pathlib import Path
from common import STAGE14C_ROOT, SETTINGS, PROTOCOL_SEEDS, protocols_by_id, sha256_file, verify_freeze

def write(rows):
    path=STAGE14C_ROOT/"stage14c_run_manifest.csv"
    fields=["protocol_id","setting","seed","physical_gpu","pid","started_at","finished_at","exit_code","status","log_path"]
    with path.open("w",encoding="utf-8",newline="") as h:
        w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows(rows)

def main():
    verify_freeze(); protocols=protocols_by_id(); visible=[x.strip() for x in os.environ.get("CUDA_VISIBLE_DEVICES","").split(",") if x.strip()]
    if not visible: raise RuntimeError("GPU selector must provide CUDA_VISIBLE_DEVICES")
    order=[pid for s in SETTINGS for seed in PROTOCOL_SEEDS for pid,p in protocols.items() if p["setting"]==s and int(p["seed"])==seed]
    rows=[]; pending=[]; active={}
    for pid in order:
        run_dir=STAGE14C_ROOT/"runs"/pid
        selection_path=run_dir/"checkpoint_selection.json"
        if selection_path.is_file():
            selection=json.loads(selection_path.read_text())
            checkpoint=Path(selection["checkpoint_path"])
            if selection["checkpoint_sha256"] != sha256_file(checkpoint):
                raise RuntimeError(f"{pid}: precompleted checkpoint hash mismatch")
            config=json.loads((run_dir/"training_config.json").read_text())
            precompleted_log=(STAGE14C_ROOT.parent/".tmux-task/stage14c-train-low2022/output.log" if pid=="low_seed2022" else run_dir/"training_stdout.log")
            rows.append({"protocol_id":pid,"setting":protocols[pid]["setting"],"seed":protocols[pid]["seed"],"physical_gpu":config["physical_gpu"],"pid":"","started_at":"","finished_at":"","exit_code":0,"status":"success","log_path":str(precompleted_log)})
            print(json.dumps({"event":"verified_precompleted","protocol_id":pid}),flush=True)
        else:
            pending.append(pid)
    while pending or active:
        for gpu in visible:
            if gpu in active or not pending: continue
            pid=pending.pop(0); run_dir=STAGE14C_ROOT/"runs"/pid; log=run_dir/"training_stdout.log"; env=os.environ.copy(); env["CUDA_VISIBLE_DEVICES"]=gpu; env.update({"OMP_NUM_THREADS":"4","MKL_NUM_THREADS":"4","OPENBLAS_NUM_THREADS":"4"})
            h=log.open("w",encoding="utf-8"); cmd=[sys.executable,"-u",str(STAGE14C_ROOT/"scripts/train_encoder.py"),"--protocol-id",pid]
            proc=subprocess.Popen(cmd,stdout=h,stderr=subprocess.STDOUT,env=env,cwd=STAGE14C_ROOT.parent)
            row={"protocol_id":pid,"setting":protocols[pid]["setting"],"seed":protocols[pid]["seed"],"physical_gpu":gpu,"pid":proc.pid,"started_at":time.time(),"finished_at":"","exit_code":"","status":"running","log_path":str(log)}
            rows.append(row); active[gpu]=(proc,h,row); print(json.dumps({"event":"started","protocol_id":pid,"gpu":gpu,"pid":proc.pid}),flush=True)
        write(rows); time.sleep(5)
        for gpu,(proc,h,row) in list(active.items()):
            rc=proc.poll()
            if rc is None: continue
            h.close(); row.update({"finished_at":time.time(),"exit_code":rc,"status":"success" if rc==0 else "failed"}); del active[gpu]
            print(json.dumps({"event":"finished","protocol_id":row["protocol_id"],"gpu":gpu,"exit_code":rc}),flush=True); write(rows)
    if any(r["status"]!="success" for r in rows): raise SystemExit(1)

if __name__=="__main__": main()
