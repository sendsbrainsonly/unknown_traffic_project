#!/usr/bin/env python3
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

from e4_model import E4Encoder
from stage18_common import OUT, SEEDS, SERVICES, VARIANT_ORDER, apply_robust_scaler, load_frozen_protocol, protocol_id, select_variant_channels, sha256_file, variant_spec, write_json


@torch.no_grad()
def replay_infer(model, x, mask, device, batch_size=512):
    model.eval(); logits_all=[]; embedding_all=[]
    for start in range(0, len(x), batch_size):
        tx=torch.as_tensor(x[start:start+batch_size],dtype=torch.float32,device=device)
        tm=torch.as_tensor(mask[start:start+batch_size],dtype=torch.bool,device=device)
        logits,embedding=model(tx,tm);logits_all.append(logits.cpu().numpy());embedding_all.append(embedding.cpu().numpy())
    logits=np.concatenate(logits_all).astype(np.float32);embedding=np.concatenate(embedding_all).astype(np.float32)
    return logits,embedding,logits.argmax(1).astype(np.int64)


def main() -> None:
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checks=[]
    for service in SERVICES:
        frozen=load_frozen_protocol(service);indices=frozen["indices"]
        for seed in SEEDS:
            run=OUT/"runs"/protocol_id(service)/f"seed{seed}"
            closed=pd.read_csv(run/"closed_set_results.csv");predictions=pd.read_csv(run/"predictions.csv")
            for variant in VARIANT_ORDER:
                checkpoint=run/f"{variant}_best.pt";payload=torch.load(checkpoint,map_location="cpu",weights_only=False);spec=variant_spec(variant)
                if payload["variant_spec"] != spec: raise RuntimeError(f"variant spec mismatch {checkpoint}")
                scaled=apply_robust_scaler(frozen["raw_x"],frozen["mask"],payload["feature_scaler"]);selected=select_variant_channels(scaled,str(spec["features"]))
                model=E4Encoder(payload["input_dim"],payload["labels_num"],payload["embedding_dim"],bool(spec["multi_scale"]),bool(spec["recurrent"]),0.2).to(device);model.load_state_dict(payload["state_dict"],strict=True)
                saved=np.load(run/f"{variant}_embeddings.npz",allow_pickle=False);max_abs=0.0;pred_equal=True;flow_equal=True
                for role in ("known_train","known_validation","known_test","unknown_test"):
                    idx=indices[role];logits,embedding,pred=replay_infer(model,selected[idx],frozen["mask"][idx],device)
                    max_abs=max(max_abs,float(np.max(np.abs(logits-saved[f"{role}_logits"]))),float(np.max(np.abs(embedding-saved[f"{role}_embedding"]))))
                    pred_equal=pred_equal and bool(np.array_equal(pred,saved[f"{role}_pred"]))
                    flow_equal=flow_equal and bool(np.array_equal(np.asarray([row["flow_id"] for row in frozen["rows"][role]]),saved[f"{role}_flow_ids"]))
                    if role != "known_train":
                        expected=predictions[(predictions.variant==variant)&(predictions.role==role)].sort_values("flow_id")
                        actual=pd.DataFrame({"flow_id":[row["flow_id"] for row in frozen["rows"][role]],"pred":pred}).sort_values("flow_id")
                        pred_equal=pred_equal and bool(np.array_equal(expected.predicted_local_label.to_numpy(dtype=int),actual.pred.to_numpy(dtype=int)))
                expected_hash=closed[closed.variant==variant].iloc[0].checkpoint_sha256
                check={"service":service,"seed":seed,"variant":variant,"checkpoint_sha256":sha256_file(checkpoint),"checkpoint_hash_match":sha256_file(checkpoint)==expected_hash,"predictions_equal":pred_equal,"flow_ids_equal":flow_equal,"max_abs_tensor_difference":max_abs,"pass":bool(pred_equal and flow_equal and max_abs<=2e-4 and sha256_file(checkpoint)==expected_hash)}
                checks.append(check);del model;torch.cuda.empty_cache()
    status="PASS" if all(item["pass"] for item in checks) and len(checks)==42 else "FAIL"
    result={"status":status,"device":str(device),"models":len(checks),"passed":sum(item["pass"] for item in checks),"max_abs_tensor_difference":max(item["max_abs_tensor_difference"] for item in checks),"checks":checks}
    write_json(OUT/"replay_verification.json",result);print(json.dumps({key:result[key] for key in ("status","device","models","passed","max_abs_tensor_difference")}),flush=True)
    if status != "PASS": raise SystemExit(1)


if __name__=="__main__":main()
