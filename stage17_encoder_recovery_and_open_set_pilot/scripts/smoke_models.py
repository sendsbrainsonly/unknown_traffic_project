#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from stage17_common import CACHE, OUT, PROJECT, sha256_file, write_json

sys.path.insert(0, str(PROJECT)); sys.path.insert(0, str(PROJECT / "tf_runtime" / "code"))
from src.stage1.tagcn import TAGCN
from train_pilot_run import make_tf_args, normalize_adjacency, tf_forward, load_tf_module


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--device",default="cuda:0");args=ap.parse_args();device=torch.device(args.device)
    if not CACHE.is_file():raise FileNotFoundError(CACHE)
    z=np.load(CACHE,allow_pickle=False);evidence={"cache_sha256":sha256_file(CACHE),"checks":{}}
    module=load_tf_module();tf_args=make_tf_args(5,1);model=module.Classifier(tf_args).to(device)
    for name,p in model.named_parameters():
        if "gamma" not in name and "beta" not in name:p.data.normal_(0,.02)
    src=torch.from_numpy(z["token_ids"][:2]).long().to(device);seg=torch.from_numpy(z["segments"][:2]).long().to(device);y=torch.tensor([0,1],device=device)
    logits,zt=tf_forward(model,src,seg);loss=nn.CrossEntropyLoss()(logits,y);loss.backward();assert torch.isfinite(loss) and torch.isfinite(zt).all() and zt.std()>0
    evidence["checks"]["E1"]={"forward_backward":"PASS","input_shape":list(src.shape),"output_shape":list(zt.shape),"loss":float(loss.detach()),"finite":True,"noncollapsed":True}
    del model,src,seg,logits,zt,loss;torch.cuda.empty_cache()
    mask=z["fig_mask"][:4];adj=normalize_adjacency(z["fig_adj"][:4],mask);gx=z["fig_x"][:4].astype(np.float32);nodes=gx[mask];mu=nodes.mean(0);sd=nodes.std(0);sd[sd<1e-8]=1;gx=(gx-mu)/sd
    graph=TAGCN(in_dim=7,hidden=128,labels_num=5,k_hops=2,dropout=.5).to(device);tx=torch.from_numpy(gx).to(device);ta=torch.from_numpy(adj).to(device);tm=torch.from_numpy(mask).to(device);ty=torch.tensor([0,1,2,3],device=device)
    glog,zg=graph(tx,ta,tm);gloss=nn.CrossEntropyLoss()(glog,ty);gloss.backward();assert torch.isfinite(gloss) and torch.isfinite(zg).all() and zg.std()>0
    with torch.no_grad(): _,zg2=graph.eval()(tx,ta,tm);_,zg3=graph.eval()(tx,ta,tm)
    evidence["checks"]["E2"]={"forward_backward":"PASS","input_shape":list(tx.shape),"output_shape":list(zg.shape),"loss":float(gloss.detach()),"finite":True,"noncollapsed":True,"deterministic_eval":bool(torch.equal(zg2,zg3))}
    fusion=nn.Linear(896,5).to(device);fz=torch.randn(4,896,device=device);fl=fusion(fz);floss=nn.CrossEntropyLoss()(fl,ty);floss.backward();evidence["checks"]["E3"]={"forward_backward":"PASS","input_shape":list(fz.shape),"output_shape":list(fl.shape),"loss":float(floss.detach()),"finite":bool(torch.isfinite(fl).all())}
    old_b=PROJECT/"outputs/stage1/modelB/seeds/seed2/modelB_best.pt";state=torch.load(old_b,map_location="cpu",weights_only=True);old=TAGCN(in_dim=7,hidden=int(state["hidden"]),labels_num=int(state["labels_num"]),k_hops=int(state["k_hops"]),dropout=.5);old.load_state_dict(state["state_dict"],strict=True);evidence["checks"]["historical_E2_checkpoint"]={"load_strict":"PASS","sha256":sha256_file(old_b),"labels_num":int(state["labels_num"]),"epoch":int(state["epoch"])}
    old_a=PROJECT/"outputs/stage1/modelA/finetuned_model.bin";old_state=torch.load(old_a,map_location="cpu",weights_only=True);evidence["checks"]["historical_E1_checkpoint"]={"load_tensor_state":"PASS","sha256":sha256_file(old_a),"tensor_count":sum(torch.is_tensor(v) for v in old_state.values()),"classifier_output_shape":list(old_state["output_layer_2.weight"].shape)}
    evidence["status"]="PASS";evidence["single_batch_single_epoch_smoke"]="PASS";evidence["peak_gpu_memory_bytes"]=int(torch.cuda.max_memory_allocated())
    write_json(OUT/"smoke_verification.json",evidence);print(json.dumps(evidence))


if __name__=="__main__":main()
