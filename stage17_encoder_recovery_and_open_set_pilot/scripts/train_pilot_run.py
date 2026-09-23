#!/usr/bin/env python3
"""Train exact recovered E1/E2/E3 branches on one frozen Service-LOSO run."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_recall_fscore_support, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from stage17_common import CACHE, OUT, PROJECT, percentile95, protocol_id, protocol_rows, seed_everything, sha256_file, write_csv, write_json

sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "tf_runtime" / "code"))
from src.stage1.tagcn import TAGCN
from uer.opts import finetune_opts
from uer.utils.config import load_hyperparam
from uer.utils.tokenizers import BertTokenizer


def load_tf_module():
    path = PROJECT / "tf_runtime" / "code" / "fine-tuning" / "run_classifier.py"
    spec = importlib.util.spec_from_file_location("stage17_tf_run_classifier", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_tf_args(labels_num: int, train_steps: int):
    import argparse as _argparse
    parser = _argparse.ArgumentParser()
    finetune_opts(parser)
    root = PROJECT / "tf_runtime" / "code"
    args = parser.parse_args([
        "--vocab_path", str(root / "models" / "encryptd_vocab.txt"),
        "--config_path", str(root / "models" / "bert" / "base_config.json"),
        "--embedding", "word_pos_seg", "--encoder", "transformer",
        "--mask", "fully_visible", "--seq_length", "320",
        "--learning_rate", "6e-5", "--batch_size", "16",
    ])
    args = load_hyperparam(args)
    args.labels_num = labels_num
    args.pooling = "first"
    args.soft_targets = False
    args.soft_alpha = 0.5
    args.tokenizer = BertTokenizer(args)
    args.is_moe = False
    args.train_steps = train_steps
    return args


def tf_forward(model, src, seg):
    emb = model.embedding(src, seg)
    output = model.encoder(emb, seg)
    z = output[:, 0, :]
    logits = model.output_layer_2(torch.tanh(model.output_layer_1(z)))
    return logits, z


def classification_metrics(y, pred):
    p, r, f, support = precision_recall_fscore_support(y, pred, average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "per_class": [{"label": int(i), "precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(support[i])} for i in range(len(support))],
    }


@torch.no_grad()
def infer_tf(model, tokens, segments, device, batch=64):
    model.eval(); zs=[]; preds=[]
    for start in range(0, len(tokens), batch):
        src=torch.as_tensor(tokens[start:start+batch],dtype=torch.long,device=device)
        seg=torch.as_tensor(segments[start:start+batch],dtype=torch.long,device=device)
        logits,z=tf_forward(model,src,seg); zs.append(z.cpu().numpy()); preds.append(logits.argmax(1).cpu().numpy())
    return np.concatenate(zs).astype(np.float32),np.concatenate(preds)


def train_tf(train, val, labels_num, seed, device, output):
    seed_everything(seed); module=load_tf_module(); epochs=3; bs=16
    steps=math.ceil(len(train[0])/bs)*epochs; tf_args=make_tf_args(labels_num,steps)
    model=module.Classifier(tf_args).to(device)
    for name,param in model.named_parameters():
        if "gamma" not in name and "beta" not in name:
            param.data.normal_(0,0.02)
    optimizer,scheduler=module.build_optimizer(tf_args,model); criterion=nn.CrossEntropyLoss()
    history=[]; best=None; best_f1=-1.0; generator=torch.Generator().manual_seed(seed)
    loader=DataLoader(TensorDataset(torch.from_numpy(train[0]).long(),torch.from_numpy(train[1]).long(),torch.from_numpy(train[2]).long()),batch_size=bs,shuffle=True,generator=generator,num_workers=0)
    for epoch in range(1,epochs+1):
        model.train(); total=0.0; count=0
        for src,seg,y in loader:
            src,seg,y=src.to(device),seg.to(device),y.to(device); optimizer.zero_grad(set_to_none=True)
            logits,_=tf_forward(model,src,seg); loss=criterion(logits,y); loss.backward(); optimizer.step(); scheduler.step(); total+=loss.item()*len(y);count+=len(y)
        _,vp=infer_tf(model,val[0],val[1],device); metrics=classification_metrics(val[2],vp)
        history.append({"epoch":epoch,"train_loss":total/count,"val_accuracy":metrics["accuracy"],"val_macro_f1":metrics["macro_f1"]})
        if metrics["macro_f1"]>best_f1:
            best_f1=metrics["macro_f1"];best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};best_epoch=epoch
        print(json.dumps({"encoder":"E1","epoch":epoch,"train_loss":total/count,"val_macro_f1":metrics["macro_f1"]}),flush=True)
    model.load_state_dict(best); torch.save({"state_dict":best,"seed":seed,"best_epoch":best_epoch,"initialization":"random_normal_0.02_strict_unknown_free","config":{"epochs":3,"batch_size":16,"lr":6e-5,"seq_length":320}},output)
    return model,history,best_epoch


def normalize_adjacency(adj,mask):
    result=np.zeros_like(adj,dtype=np.float32)
    for i in range(len(adj)):
        n=int(mask[i].sum()); a=adj[i,:n,:n].astype(np.float32); a+=np.eye(n,dtype=np.float32); d=np.maximum(a.sum(1),1e-12); result[i,:n,:n]=a/np.sqrt(d[:,None]*d[None,:])
    return result


@torch.no_grad()
def infer_graph(model,x,adj,mask,device,batch=256):
    model.eval();zs=[];preds=[]
    for s in range(0,len(x),batch):
        tx=torch.as_tensor(x[s:s+batch],device=device);ta=torch.as_tensor(adj[s:s+batch],device=device);tm=torch.as_tensor(mask[s:s+batch],device=device)
        logits,z=model(tx,ta,tm);zs.append(z.cpu().numpy());preds.append(logits.argmax(1).cpu().numpy())
    return np.concatenate(zs).astype(np.float32),np.concatenate(preds)


def train_graph(train,val,labels_num,seed,device,output):
    seed_everything(seed);epochs=50;bs=64
    model=TAGCN(in_dim=7,hidden=128,labels_num=labels_num,k_hops=2,dropout=.5).to(device);optimizer=torch.optim.Adam(model.parameters(),lr=1e-3);criterion=nn.CrossEntropyLoss()
    loader=DataLoader(TensorDataset(torch.from_numpy(train[0]),torch.from_numpy(train[1]),torch.from_numpy(train[2]),torch.from_numpy(train[3]).long()),batch_size=bs,shuffle=True,generator=torch.Generator().manual_seed(seed),num_workers=0)
    history=[];best=None;best_f1=-1.0
    for epoch in range(1,epochs+1):
        model.train();total=0.;count=0
        for x,a,m,y in loader:
            x,a,m,y=x.to(device),a.to(device),m.to(device),y.to(device);optimizer.zero_grad(set_to_none=True);logits,_=model(x,a,m);loss=criterion(logits,y);loss.backward();optimizer.step();total+=loss.item()*len(y);count+=len(y)
        _,vp=infer_graph(model,val[0],val[1],val[2],device);metrics=classification_metrics(val[3],vp);history.append({"epoch":epoch,"train_loss":total/count,"val_accuracy":metrics["accuracy"],"val_macro_f1":metrics["macro_f1"]})
        if metrics["macro_f1"]>best_f1:
            best_f1=metrics["macro_f1"];best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};best_epoch=epoch
        if epoch==1 or epoch%10==0: print(json.dumps({"encoder":"E2","epoch":epoch,"val_macro_f1":metrics["macro_f1"]}),flush=True)
    model.load_state_dict(best);torch.save({"state_dict":best,"seed":seed,"best_epoch":best_epoch,"config":{"epochs":50,"batch_size":64,"lr":1e-3,"hidden":128,"k_hops":2}},output)
    return model,history,best_epoch


def train_fusion(z_train,y_train,z_val,y_val,seed,device,output):
    seed_everything(seed);epochs=30;bs=256;model=nn.Linear(z_train.shape[1],len(np.unique(y_train))).to(device);optimizer=torch.optim.Adam(model.parameters(),lr=1e-3);criterion=nn.CrossEntropyLoss();history=[];best=None;best_f1=-1
    loader=DataLoader(TensorDataset(torch.from_numpy(z_train).float(),torch.from_numpy(y_train).long()),batch_size=bs,shuffle=True,generator=torch.Generator().manual_seed(seed),num_workers=0)
    for epoch in range(1,epochs+1):
        model.train();total=0.;count=0
        for x,y in loader:
            x,y=x.to(device),y.to(device);optimizer.zero_grad(set_to_none=True);logits=model(x);loss=criterion(logits,y);loss.backward();optimizer.step();total+=loss.item()*len(y);count+=len(y)
        model.eval();
        with torch.no_grad(): vp=model(torch.from_numpy(z_val).float().to(device)).argmax(1).cpu().numpy()
        met=classification_metrics(y_val,vp);history.append({"epoch":epoch,"train_loss":total/count,"val_accuracy":met["accuracy"],"val_macro_f1":met["macro_f1"]})
        if met["macro_f1"]>best_f1: best_f1=met["macro_f1"];best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};best_epoch=epoch
    model.load_state_dict(best);torch.save({"state_dict":best,"seed":seed,"best_epoch":best_epoch,"input_dim":z_train.shape[1],"config":{"epochs":30,"batch_size":256,"lr":1e-3,"fusion":"train-zscore each branch then concat"}},output)
    return model,history,best_epoch


def predict_linear(model,z,device):
    model.eval();
    with torch.no_grad(): return model(torch.from_numpy(z).float().to(device)).argmax(1).cpu().numpy()


def des_v1(train_z,train_y,val_z,val_pred,test_z,test_pred,unknown_z,unknown_pred):
    classes=sorted(np.unique(train_y));centroids={c:train_z[train_y==c].mean(0) for c in classes}
    def score(z,pred):
        dg=np.asarray([np.sum((v-centroids[int(c)])**2) for v,c in zip(z,pred)],dtype=np.float64)
        dl=[]
        for v,c in zip(z,pred):
            support=train_z[train_y==int(c)];dist=np.linalg.norm(support-v,axis=1);dl.append(float(np.partition(dist,9)[:10].mean()))
        return dg,np.asarray(dl)
    vg,vl=score(val_z,val_pred);medg,medl=np.median(vg),np.median(vl);madg=np.median(np.abs(vg-medg));madl=np.median(np.abs(vl-medl))
    def combine(z,p):
        g,l=score(z,p);return .5*(g-medg)/(madg+1e-12)+.5*(l-medl)/(madl+1e-12)
    vs=.5*(vg-medg)/(madg+1e-12)+.5*(vl-medl)/(madl+1e-12);return vs,combine(test_z,test_pred),combine(unknown_z,unknown_pred),{"global_median":float(medg),"global_mad":float(madg),"local_median":float(medl),"local_mad":float(madl)}


def open_metrics(known_y,known_pred,unknown_pred,ks,us,threshold):
    binary=np.r_[np.zeros(len(ks),dtype=int),np.ones(len(us),dtype=int)];scores=np.r_[ks,us];reject=scores>=threshold
    open_true=np.r_[known_y,np.full(len(us),5)];open_pred=np.r_[known_pred,unknown_pred];open_pred[reject]=5
    return {"auroc":float(roc_auc_score(binary,scores)),"auprc":float(average_precision_score(binary,scores)),"ufar":float(np.mean(us<threshold)),"known_frr":float(np.mean(ks>=threshold)),"open_macro_f1":float(f1_score(open_true,open_pred,average="macro",zero_division=0)),"threshold":float(threshold),"known_samples":len(ks),"unknown_samples":len(us),"unknown_prevalence":float(len(us)/(len(us)+len(ks)))}


def geometry(z,y):
    classes=sorted(np.unique(y));centroids=np.stack([z[y==c].mean(0) for c in classes]);pair=[]
    for i in range(len(classes)):
        for j in range(i+1,len(classes)):pair.append(float(np.linalg.norm(centroids[i]-centroids[j])))
    intra=np.asarray([np.linalg.norm(v-centroids[int(c)]) for v,c in zip(z,y)]);s=np.linalg.svd(z-z.mean(0),compute_uv=False);eff=float((s.sum()**2)/(np.square(s).sum()+1e-12))
    return {"embedding_dim":int(z.shape[1]),"effective_rank":eff,"norm_mean":float(np.linalg.norm(z,axis=1).mean()),"intra_distance_mean":float(intra.mean()),"centroid_distance_mean":float(np.mean(pair)),"zero_variance_dimensions":int(np.sum(z.std(0)<1e-8))}


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--unknown-service",choices=("Email","Streaming"),required=True);ap.add_argument("--seed",type=int,choices=(2022,2023,2024),required=True);args=ap.parse_args()
    if not CACHE.is_file(): raise FileNotFoundError(CACHE)
    run=OUT/"runs"/protocol_id(args.unknown_service)/f"seed{args.seed}";run.mkdir(parents=True,exist_ok=False);start=time.time();device=torch.device("cuda:0")
    data=np.load(CACHE,allow_pickle=False);pos={str(f):i for i,f in enumerate(data["flow_ids"])};rows=protocol_rows(args.unknown_service);by_role={r:[] for r in ("known_train","known_validation","known_test","unknown_test")}
    for row in rows: by_role[row["role"]].append(row)
    for role in by_role: by_role[role].sort(key=lambda x:int(x["role_row_index"]))
    idx={role:np.asarray([pos[r["flow_id"]] for r in rr]) for role,rr in by_role.items()};lab={role:np.asarray([int(r["local_label"]) for r in rr]) for role,rr in by_role.items()}
    assert all(r["service_label"]!=args.unknown_service for role in ("known_train","known_validation","known_test") for r in by_role[role]);assert all(r["service_label"]==args.unknown_service for r in by_role["unknown_test"])
    tx=data["token_ids"];sg=data["segments"];gx=data["fig_x"].astype(np.float32);gm=data["fig_mask"]
    train_nodes=gx[idx["known_train"]][gm[idx["known_train"]]];mu=train_nodes.mean(0);sd=train_nodes.std(0);sd[sd<1e-8]=1;gx=(gx-mu)/sd;ga=normalize_adjacency(data["fig_adj"],gm)
    ttr=(tx[idx["known_train"]],sg[idx["known_train"]],lab["known_train"]);tva=(tx[idx["known_validation"]],sg[idx["known_validation"]],lab["known_validation"])
    gtr=(gx[idx["known_train"]],ga[idx["known_train"]],gm[idx["known_train"]],lab["known_train"]);gva=(gx[idx["known_validation"]],ga[idx["known_validation"]],gm[idx["known_validation"]],lab["known_validation"])
    peak0=torch.cuda.max_memory_allocated();tf,th,te=train_tf(ttr,tva,5,args.seed,device,run/"E1_model_best.pt")
    e1={};
    for role in idx:e1[role]=infer_tf(tf,tx[idx[role]],sg[idx[role]],device)
    del tf;torch.cuda.empty_cache();graph,gh,ge=train_graph(gtr,gva,5,args.seed,device,run/"E2_model_best.pt");e2={}
    for role in idx:e2[role]=infer_graph(graph,gx[idx[role]],ga[idx[role]],gm[idx[role]],device)
    del graph;torch.cuda.empty_cache()
    mt=e1["known_train"][0].mean(0);st=e1["known_train"][0].std(0);st[st<1e-8]=1;mg=e2["known_train"][0].mean(0);sgd=e2["known_train"][0].std(0);sgd[sgd<1e-8]=1
    e3z={role:np.concatenate([(e1[role][0]-mt)/st,(e2[role][0]-mg)/sgd],1).astype(np.float32) for role in idx}
    fusion,fh,fe=train_fusion(e3z["known_train"],lab["known_train"],e3z["known_validation"],lab["known_validation"],args.seed,device,run/"E3_model_best.pt");e3={role:(e3z[role],predict_linear(fusion,e3z[role],device)) for role in idx}
    results=[];predrows=[];geometry_rows=[]
    for enc,rep,best_epoch,history in (("E1",e1,te,th),("E2",e2,ge,gh),("E3",e3,fe,fh)):
        vs,ks,us,norm=des_v1(rep["known_train"][0],lab["known_train"],rep["known_validation"][0],rep["known_validation"][1],rep["known_test"][0],rep["known_test"][1],rep["unknown_test"][0],rep["unknown_test"][1]);threshold=percentile95(vs);closed=classification_metrics(lab["known_test"],rep["known_test"][1]);opened=open_metrics(lab["known_test"],rep["known_test"][1],rep["unknown_test"][1],ks,us,threshold)
        results.append({"protocol_id":protocol_id(args.unknown_service),"unknown_service":args.unknown_service,"seed":args.seed,"encoder":enc,"detector":"DES-v1","status":"SUCCESS","initialization":"random_normal_0.02" if enc in {"E1","E3"} else "random_seeded","pretraining_exposure":"NONE_STRICT" if enc in {"E1","E3"} else "NONE","best_epoch":best_epoch,"known_test_accuracy":closed["accuracy"],"known_test_macro_f1":closed["macro_f1"],"known_test_weighted_f1":closed["weighted_f1"],**opened,"runtime_seconds":time.time()-start,"peak_gpu_memory_bytes":int(torch.cuda.max_memory_allocated())})
        geometry_rows.append({"encoder":enc,**geometry(rep["known_train"][0],lab["known_train"]),**norm})
        for role,scores in (("known_validation",vs),("known_test",ks),("unknown_test",us)):
            for row,pred,score in zip(by_role[role],rep[role][1],scores):predrows.append({"protocol_id":protocol_id(args.unknown_service),"unknown_service":args.unknown_service,"seed":args.seed,"encoder":enc,"role":role,"flow_id":row["flow_id"],"true_service":row["service_label"],"true_local_label":row["local_label"],"predicted_local_label":int(pred),"des_v1_score":float(score),"threshold":threshold,"rejected":int(score>=threshold)})
        np.savez_compressed(run/f"{enc}_embeddings.npz",**{f"{role}_z":rep[role][0] for role in idx},**{f"{role}_pred":rep[role][1] for role in idx},**{f"{role}_flow_ids":np.asarray([r["flow_id"] for r in by_role[role]]) for role in idx})
        write_json(run/f"{enc}_training_history.json",history)
    write_csv(run/"results.csv",results);write_csv(run/"predictions.csv",predrows);write_json(run/"geometry.json",geometry_rows)
    hashes={p.name:sha256_file(p) for p in run.iterdir() if p.is_file()};write_json(run/"run_manifest.json",{"status":"SUCCESS","protocol_id":protocol_id(args.unknown_service),"unknown_service":args.unknown_service,"seed":args.seed,"strict_unknown_free":True,"unknown_training_samples":0,"unknown_validation_samples":0,"unknown_support_samples":0,"unknown_normalization_samples":0,"unknown_threshold_samples":0,"cache_sha256":sha256_file(CACHE),"artifacts":hashes});(run/"SUCCESS").write_text("SUCCESS\n")
    print(json.dumps({"status":"SUCCESS","run":str(run),"results":results}),flush=True)


if __name__=="__main__":main()
