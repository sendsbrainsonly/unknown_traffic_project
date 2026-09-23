#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import multiprocessing.util
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset

from common import CONFIG_PATH, ROOT, pilot_specs, protocol_rows, read_json, seed_everything, sha256_file, write_csv, write_json
from model import MaskSafeSequenceCNN


class SequenceDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.from_numpy(x.astype(np.float32, copy=False))
        self.y = torch.from_numpy(y.astype(np.int64, copy=False))
    def __len__(self): return len(self.y)
    def __getitem__(self, index): return self.x[index], self.y[index]


def short_tmp(output: Path):
    actual = output / "runtime_tmp"
    actual.mkdir(parents=True)
    fd = os.open(actual, os.O_RDONLY | os.O_DIRECTORY)
    os.set_inheritable(fd, True)
    alias = f"/proc/self/fd/{fd}"
    for name in ("TMPDIR", "TMP", "TEMP"):
        os.environ[name] = alias
    tempfile.tempdir = None
    multiprocessing.util._tempdir = None
    return alias, fd


def load_role(dataset: str, rows: list[dict], window: int):
    cache = ROOT / "sequence_cache" / dataset
    uids = np.load(cache / "flow_uids.npy", allow_pickle=False)
    index = {str(uid): i for i, uid in enumerate(uids)}
    pos = np.asarray([index[row["sample_id"]] for row in rows], dtype=np.int64)
    lengths = np.load(cache / "packet_lengths.npy", mmap_mode="r", allow_pickle=False)[pos, :window]
    iats = np.load(cache / "packet_iat_seconds.npy", mmap_mode="r", allow_pickle=False)[pos, :window]
    directions = np.load(cache / "packet_directions.npy", mmap_mode="r", allow_pickle=False)[pos, :window]
    mask = np.load(cache / "packet_mask.npy", mmap_mode="r", allow_pickle=False)[pos, :window].astype(np.float32)
    signed = np.sign(directions).astype(np.float32) * np.log1p(lengths.astype(np.float32))
    log_iat = np.log1p(iats.astype(np.float32) * 1_000_000.0)
    return np.stack((signed, log_iat, mask), axis=1), np.asarray([row["label"] for row in rows], np.int64), np.asarray([row["sample_id"] for row in rows])


def normalize(train, val):
    train, val = train.copy(), val.copy()
    valid = train[:, 2, :] > 0
    audit = {"fit_role": "Known Train only", "channels": {}}
    for channel, name in ((0, "signed_log1p_length"), (1, "log1p_iat_microseconds")):
        observed = train[:, channel, :][valid]
        median = float(np.median(observed))
        q1, q3 = np.quantile(observed, [0.25, 0.75])
        iqr = float(max(q3 - q1, 1e-6))
        for array in (train, val):
            array[:, channel, :] = np.clip((array[:, channel, :] - median) / iqr, -4.0, 4.0) * array[:, 2, :]
        audit["channels"][name] = {"median": median, "iqr": iqr}
    return train, val, audit


def metrics(model, loader, device, criterion, n_classes):
    model.eval(); losses=[]; truths=[]; preds=[]; logits=[]
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            out = model(x)
            losses.append(float(criterion(out, y)) * len(y))
            truths.append(y.cpu().numpy()); preds.append(out.argmax(1).cpu().numpy()); logits.append(out.cpu().numpy())
    truth=np.concatenate(truths); pred=np.concatenate(preds); logit=np.concatenate(logits)
    precision, recall, f1, support = precision_recall_fscore_support(truth, pred, labels=np.arange(n_classes), zero_division=0)
    return {"loss":sum(losses)/len(truth),"accuracy":accuracy_score(truth,pred),"macro_f1":f1.mean(),"weighted_f1":np.average(f1,weights=support),"precision":precision,"recall":recall,"f1":f1,"support":support,"confusion":confusion_matrix(truth,pred,labels=np.arange(n_classes)),"truth":truth,"pred":pred,"logits":logit}


def train_epoch(model, loader, device, criterion, optimizer):
    model.train(); total=correct=0; loss_sum=0.0
    for x,y in loader:
        x,y=x.to(device,non_blocking=True),y.to(device,non_blocking=True)
        optimizer.zero_grad(set_to_none=True); out=model(x); loss=criterion(out,y); loss.backward(); optimizer.step()
        total += len(y); loss_sum += float(loss)*len(y); correct += int((out.argmax(1)==y).sum())
    return loss_sum/total, correct/total


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--dataset",required=True); parser.add_argument("--protocol-id",required=True); parser.add_argument("--window",type=int,choices=(8,16,32),required=True); parser.add_argument("--device",default="cuda:0"); args=parser.parse_args()
    config=read_json(CONFIG_PATH)
    if (args.dataset,args.protocol_id) not in {(x["dataset"],x["protocol_id"]) for x in pilot_specs()}:
        raise RuntimeError("protocol not preregistered")
    output=ROOT/"runs"/f"T{args.window}"/args.dataset/args.protocol_id
    if output.exists(): raise RuntimeError(f"refusing overwrite: {output}")
    output.mkdir(parents=True); tmp_alias,tmp_fd=short_tmp(output); started=time.time()
    classes,train_rows,val_rows=protocol_rows(args.dataset,args.protocol_id)
    train_x,train_y,train_ids=load_role(args.dataset,train_rows,args.window); val_x,val_y,val_ids=load_role(args.dataset,val_rows,args.window)
    train_x,val_x,norm=normalize(train_x,val_x)
    if set(np.unique(train_y)) != set(range(len(classes))) or set(np.unique(val_y)) != set(range(len(classes))): raise RuntimeError("class coverage mismatch")
    seed=int(config["training"]["seed"]); seed_everything(seed)
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    device=torch.device(args.device); model=MaskSafeSequenceCNN(len(classes)).to(device); torch.cuda.reset_peak_memory_stats(device)
    train_set,val_set=SequenceDataset(train_x,train_y),SequenceDataset(val_x,val_y)
    generator=torch.Generator().manual_seed(seed); common={"num_workers":4,"pin_memory":True,"persistent_workers":False}
    train_loader=DataLoader(train_set,batch_size=config["training"]["batch_size"],shuffle=True,generator=generator,**common)
    val_loader=DataLoader(val_set,batch_size=config["training"]["eval_batch_size"],shuffle=False,**common)
    criterion=nn.CrossEntropyLoss(); optimizer=torch.optim.Adam(model.parameters(),lr=config["training"]["learning_rate"],betas=tuple(config["training"]["betas"]),weight_decay=config["training"]["weight_decay"])
    scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=config["training"]["scheduler_milestones"],gamma=config["training"]["scheduler_gamma"])
    history=[]; best_accuracy=-1.0; best_epoch=-1; checkpoint=output/"model_best.pt"
    for epoch in range(1,config["training"]["epochs"]+1):
        lr=float(optimizer.param_groups[0]["lr"]); train_loss,train_accuracy=train_epoch(model,train_loader,device,criterion,optimizer); val=metrics(model,val_loader,device,criterion,len(classes)); scheduler.step()
        improved=float(val["accuracy"])>best_accuracy
        if improved:
            best_accuracy=float(val["accuracy"]); best_epoch=epoch; torch.save({"model_state_dict":model.state_dict(),"window":args.window,"dataset":args.dataset,"protocol_id":args.protocol_id,"class_names":classes,"epoch":epoch},checkpoint)
        row={"epoch":epoch,"learning_rate":lr,"train_loss":train_loss,"train_accuracy":train_accuracy,"validation_loss":float(val["loss"]),"validation_accuracy":float(val["accuracy"]),"validation_macro_f1":float(val["macro_f1"]),"validation_weighted_f1":float(val["weighted_f1"]),"is_best":int(improved),"elapsed_seconds":time.time()-started}
        history.append(row); write_csv(output/"history.csv",history); print(json.dumps({"dataset":args.dataset,"protocol":args.protocol_id,"window":args.window,**row}),flush=True)
    saved=torch.load(checkpoint,map_location=device,weights_only=True); model.load_state_dict(saved["model_state_dict"]); final=metrics(model,val_loader,device,criterion,len(classes))
    np.save(output/"validation_confusion_matrix.npy",final["confusion"],allow_pickle=False)
    np.savez_compressed(output/"validation_predictions.npz",sample_ids=val_ids,true_labels=final["truth"],predicted_labels=final["pred"],logits=final["logits"].astype(np.float32))
    per_class=[{"window":args.window,"dataset":args.dataset,"protocol_id":args.protocol_id,"class_name":name,"precision":float(final["precision"][i]),"recall":float(final["recall"][i]),"f1":float(final["f1"][i]),"support":int(final["support"][i])} for i,name in enumerate(classes)]
    write_csv(output/"per_class_results.csv",per_class)
    result={"status":"PASS","dataset":args.dataset,"protocol_id":args.protocol_id,"window":args.window,"known_train_samples":len(train_y),"known_validation_samples":len(val_y),"known_test_samples_used":0,"unknown_test_samples_used":0,"class_names":classes,"epochs_completed":config["training"]["epochs"],"best_epoch":best_epoch,"validation_loss":float(final["loss"]),"validation_accuracy":float(final["accuracy"]),"validation_macro_f1":float(final["macro_f1"]),"validation_weighted_f1":float(final["weighted_f1"]),"runtime_seconds":time.time()-started,"peak_gpu_memory_bytes":int(torch.cuda.max_memory_allocated(device)),"checkpoint_path":str(checkpoint.resolve()),"checkpoint_sha256":sha256_file(checkpoint),"normalization":norm,"physical_gpu":os.environ.get("CUDA_VISIBLE_DEVICES","UNSET"),"data_loader_workers":4,"mask_safe":True}
    write_json(output/"result.json",result); write_json(output/"run_config.json",{"config":config,"invocation":vars(args),"config_sha256":sha256_file(CONFIG_PATH),"cache_audit_sha256":sha256_file(ROOT/"sequence_cache"/args.dataset/"cache_audit.json"),"strict_unknown_free":True,"visible_roles":["known_train","known_validation"],"tmp_alias":tmp_alias,"tmp_fd":tmp_fd}); print(json.dumps(result,sort_keys=True),flush=True)


if __name__=="__main__": main()
