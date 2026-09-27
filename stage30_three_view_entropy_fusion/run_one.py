#!/usr/bin/env python3
"""True three-view Known-only ER-CMGI-inspired entropy fusion diagnostic."""
from __future__ import annotations

import argparse
import csv
import json
import random
import time

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from preflight import OUT, known_labels, load_three, sources, sha256

VIEWS = ("trafficformer", "graph", "yatc")
DIMS = (768, 128, 192)
LATENT = 64
EPOCHS = 30
BATCH = 256


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def save_csv(path, rows):
    if not rows:
        raise RuntimeError(f"empty CSV: {path}")
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def seeded(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def metrics(truth, prediction, nclasses):
    p, r, f, n = precision_recall_fscore_support(
        truth, prediction, labels=np.arange(nclasses), zero_division=0)
    return {"accuracy": float(np.mean(truth == prediction)), "macro_f1": float(f.mean()),
            "weighted_f1": float(np.average(f, weights=n))}, (p, r, f, n)


class View(nn.Module):
    def __init__(self, dim, nclasses):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(dim, 128), nn.ReLU(),
                                      nn.Linear(128, 128), nn.ReLU())
        self.mean = nn.Linear(128, LATENT)
        self.logvar = nn.Linear(128, LATENT)
        self.classifier = nn.Linear(LATENT, nclasses)

    def forward(self, x):
        hidden = self.backbone(x)
        return self.mean(hidden), self.logvar(hidden).clamp(-8., 4.)


class Adapters(nn.Module):
    def __init__(self, nclasses):
        super().__init__()
        self.views = nn.ModuleList([View(dim, nclasses) for dim in DIMS])

    def training_loss(self, xs, labels):
        ces, kls = [], []
        for module, x in zip(self.views, xs, strict=True):
            mu, logvar = module(x)
            latent = mu + torch.randn_like(mu) * torch.exp(logvar * 0.5)
            ces.append(F.cross_entropy(module.classifier(latent), labels))
            kls.append(-0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(1).mean() / LATENT)
        ce, kl = torch.stack(ces).mean(), torch.stack(kls).mean()
        return ce + .05 * kl, ce.detach(), kl.detach()


@torch.no_grad()
def encode(model, values, device):
    model.eval()
    mu_chunks, h_chunks, pred_chunks = [], [], []
    for start in range(0, len(values["labels"]), 512):
        xs = [torch.from_numpy(values[name][start:start+512]).to(device) for name in VIEWS]
        mus, entropies, predictions = [], [], []
        for module, x in zip(model.views, xs, strict=True):
            mu, logvar = module(x)
            mus.append(mu)
            entropies.append(0.5 * np.log(2*np.pi*np.e) + 0.5 * logvar.mean(1))
            predictions.append(module.classifier(mu).argmax(1))
        mu_chunks.append(torch.stack(mus, 1).cpu().numpy())
        h_chunks.append(torch.stack(entropies, 1).cpu().numpy())
        pred_chunks.append(torch.stack(predictions, 1).cpu().numpy())
    return (np.concatenate(mu_chunks).astype(np.float32),
            np.concatenate(h_chunks).astype(np.float32), np.concatenate(pred_chunks))


def train_adapters(train, val, nclasses, seed, device, run):
    seeded(seed)
    model = Adapters(nclasses).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    arrays = [torch.from_numpy(train[name]) for name in VIEWS] + [torch.from_numpy(train["labels"])]
    loader = DataLoader(TensorDataset(*arrays), batch_size=BATCH, shuffle=True,
                        generator=torch.Generator().manual_seed(seed), num_workers=0)
    history, best, best_epoch, best_state = [], -1.0, -1, None
    for epoch in range(1, EPOCHS+1):
        model.train()
        totals = np.zeros(3, dtype=np.float64)
        for batch in loader:
            xs = [a.to(device) for a in batch[:3]]
            labels = batch[3].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, ce, kl = model.training_loss(xs, labels)
            if not torch.isfinite(loss):
                raise RuntimeError(f"nonfinite adapter loss at epoch {epoch}")
            loss.backward(); optimizer.step()
            totals += np.array([float(loss.detach()), float(ce), float(kl)]) * len(labels)
        _, _, pred = encode(model, val, device)
        f1s = [metrics(val["labels"], pred[:, i], nclasses)[0]["macro_f1"] for i in range(3)]
        mean_f1 = float(np.mean(f1s))
        history.append({"phase": "adapters", "epoch": epoch,
                        "train_loss": totals[0]/len(train["labels"]),
                        "train_ce": totals[1]/len(train["labels"]),
                        "train_kl": totals[2]/len(train["labels"]),
                        **{f"val_{name}_macro_f1": score for name, score in zip(VIEWS, f1s, strict=True)},
                        "val_mean_unimodal_macro_f1": mean_f1})
        if mean_f1 > best:
            best, best_epoch = mean_f1, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % 5 == 0:
            print(json.dumps({"phase": "adapters", "epoch": epoch,
                              "val_mean_unimodal_macro_f1": mean_f1}), flush=True)
    assert best_state is not None
    model.load_state_dict(best_state)
    torch.save({"state_dict": best_state, "seed": seed, "best_epoch": best_epoch,
                "epochs_completed": EPOCHS, "selection": "Known Val mean unimodal Macro-F1"},
               run / "adapters_best.pt")
    return model, history, best_epoch


class Fusion(nn.Module):
    def __init__(self, nclasses, dynamic, center, mad):
        super().__init__()
        self.dynamic = dynamic
        self.register_buffer("center", torch.as_tensor(center, dtype=torch.float32))
        self.register_buffer("mad", torch.as_tensor(mad, dtype=torch.float32))
        self.mix_logits = nn.Parameter(torch.zeros(3)) if dynamic else None
        self.fuser = nn.Sequential(nn.Linear(3*LATENT, 128), nn.ReLU(), nn.Dropout(.1),
                                   nn.Linear(128, 128), nn.ReLU())
        self.classifier = nn.Linear(128, nclasses)

    def forward(self, mu, entropy, force_equal=False):
        if self.dynamic and not force_equal:
            u = F.softplus((entropy - self.center)/(self.mad+1e-6)) + 0.1
            high = u/u.sum(1, keepdim=True)
            inv = u.reciprocal(); low = inv/inv.sum(1, keepdim=True)
            lam = .1+.8*torch.sigmoid(self.mix_logits)
            raw = lam*high+(1-lam)*low
            weight = raw/raw.sum(1, keepdim=True)
        else:
            weight = torch.full((len(mu), 3), 1/3, dtype=mu.dtype, device=mu.device)
        joined = (mu*weight.unsqueeze(-1)).flatten(1)
        return self.classifier(self.fuser(joined)), weight


@torch.no_grad()
def infer(model, encoded, device, force_equal=False, entropy_override=None, batch_size=512):
    model.eval()
    mu, entropy = encoded[:2]
    if entropy_override is not None:
        entropy = entropy_override
    logits, weights = [], []
    for start in range(0, len(mu), batch_size):
        m = torch.from_numpy(mu[start:start+batch_size]).to(device)
        h = torch.from_numpy(entropy[start:start+batch_size]).to(device)
        y, w = model(m, h, force_equal=force_equal)
        logits.append(y.cpu().numpy()); weights.append(w.cpu().numpy())
    return np.concatenate(logits), np.concatenate(weights)


def train_fusion(name, train, val, train_y, val_y, nclasses, seed, device, run):
    seeded(seed)
    center = np.median(train[1], axis=0)
    mad = np.median(np.abs(train[1]-center), axis=0)
    model = Fusion(nclasses, name == "T1_three_entropy", center, mad).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = DataLoader(TensorDataset(torch.from_numpy(train[0]), torch.from_numpy(train[1]),
                                      torch.from_numpy(train_y)), batch_size=BATCH, shuffle=True,
                        generator=torch.Generator().manual_seed(seed), num_workers=0)
    history, best, best_epoch, state = [], -1.0, -1, None
    for epoch in range(1, EPOCHS+1):
        model.train(); total = 0.0
        for mu, h, y in loader:
            mu, h, y = mu.to(device), h.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(mu, h)
            loss = F.cross_entropy(logits, y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"nonfinite fusion loss {name} epoch {epoch}")
            loss.backward(); optimizer.step()
            total += float(loss.detach())*len(y)
        val_logits, w = infer(model, val, device)
        score = metrics(val_y, val_logits.argmax(1), nclasses)[0]
        lam = (.1+.8*torch.sigmoid(model.mix_logits)).detach().cpu().numpy() if model.dynamic else np.full(3,.5)
        history.append({"phase": name, "epoch": epoch, "train_loss": total/len(train_y),
                        **{f"val_{k}": v for k, v in score.items()},
                        **{f"lambda_{k}": float(v) for k, v in zip(VIEWS, lam, strict=True)},
                        **{f"weight_{k}_mean": float(w[:,i].mean()) for i,k in enumerate(VIEWS)},
                        "collapse_ratio": float((w.max(1)>.95).mean())})
        if score["macro_f1"] > best:
            best, best_epoch = score["macro_f1"], epoch
            state = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if epoch == 1 or epoch % 5 == 0:
            print(json.dumps({"phase": name, "epoch": epoch,
                              "val_macro_f1": score["macro_f1"]}), flush=True)
    assert state is not None
    model.load_state_dict(state)
    torch.save({"state_dict": state, "name": name, "seed": seed,
                "best_epoch": best_epoch, "epochs_completed": EPOCHS,
                "known_train_entropy_median": center.tolist(), "known_train_entropy_mad": mad.tolist(),
                "selection": "Known Val fine Service Macro-F1"}, run/f"{name}_best.pt")
    logits, w = infer(model, val, device)
    return model, history, best_epoch, logits, w


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("iscx_vpn","iscx_tor"), required=True)
    parser.add_argument("--encoder-seed", type=int, choices=(2022,2023), required=True)
    parser.add_argument("--training-seed", type=int, choices=range(2022,2027), required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("no selected CUDA GPU")
    torch.set_num_threads(4)
    device = torch.device("cuda:0")
    run = OUT/"runs"/args.dataset/f"encoder{args.encoder_seed}_train{args.training_seed}"
    if not (run/"manifest.json").is_file() or (run/"SUCCESS").exists():
        raise RuntimeError("run bundle missing/already completed")
    start = time.monotonic()
    before = {k: sha256(v) for k,v in sources(args.dataset,args.encoder_seed).items()}
    expected = json.loads((OUT/"frozen_input_hashes_before.json").read_text())
    save_json(run/"source_hashes_before.json", before)
    if before != expected[f"{args.dataset}/{args.encoder_seed}"]:
        raise RuntimeError("source differs from preflight")
    pair = load_three(args.dataset,args.encoder_seed,known_labels()[args.dataset])
    train,val = (pair["roles"][role] for role in ("known_train","known_validation"))
    nclasses = len(pair["services"])
    model,history,adapter_epoch = train_adapters(train,val,nclasses,args.training_seed,device,run)
    encoded_train,encoded_val = encode(model,train,device),encode(model,val,device)
    rows,per_class,predictions,weights = [],[],[],[]
    logits_store = {}
    counterfactual = None
    for name in ("T0_equal","T1_three_entropy"):
        head,curve,epoch,logits,w = train_fusion(name,encoded_train,encoded_val,
            train["labels"],val["labels"],nclasses,args.training_seed,device,run)
        history.extend(curve)
        logits_store[name] = logits
        pred = logits.argmax(1)
        score,pc = metrics(val["labels"],pred,nclasses)
        rows.append({"dataset":args.dataset,"encoder_seed":args.encoder_seed,
                     "training_seed":args.training_seed,"method":name,"samples":len(pred),
                     "best_epoch":epoch,**score})
        for i,service in enumerate(pair["services"]):
            per_class.append({"dataset":args.dataset,"encoder_seed":args.encoder_seed,
                              "training_seed":args.training_seed,"method":name,"class":service,
                              "precision":float(pc[0][i]),"recall":float(pc[1][i]),
                              "f1":float(pc[2][i]),"support":int(pc[3][i])})
        for fid,truth,chosen,weight,h in zip(val["flow_ids"],val["labels"],pred,w,encoded_val[1],strict=True):
            predictions.append({"dataset":args.dataset,"encoder_seed":args.encoder_seed,
                                "training_seed":args.training_seed,"method":name,"flow_id":str(fid),
                                "true_service":pair["services"][int(truth)],
                                "predicted_service":pair["services"][int(chosen)],
                                "correct":int(truth==chosen)})
            weights.append({"dataset":args.dataset,"encoder_seed":args.encoder_seed,
                            "training_seed":args.training_seed,"method":name,"flow_id":str(fid),
                            "class":pair["services"][int(truth)],"correct":int(truth==chosen),
                            **{f"H_{k}":float(h[i]) for i,k in enumerate(VIEWS)},
                            **{f"w_{k}":float(weight[i]) for i,k in enumerate(VIEWS)}})
        if name == "T1_three_entropy":
            equal_logits,_ = infer(head,encoded_val,device,force_equal=True)
            permutation = np.random.default_rng(args.training_seed+991).permutation(len(pred))
            shuffle_logits,_ = infer(head,encoded_val,device,entropy_override=encoded_val[1][permutation])
            batch_logits,_ = infer(head,encoded_val,device,batch_size=17)
            batch_delta = float(np.max(np.abs(batch_logits-logits)))
            if batch_delta > 1e-5:
                raise RuntimeError(f"batch invariance failure: {batch_delta}")
            counterfactual = {"main":score,
                              "forced_equal":metrics(val["labels"],equal_logits.argmax(1),nclasses)[0],
                              "shuffled_entropy":metrics(val["labels"],shuffle_logits.argmax(1),nclasses)[0],
                              "max_batch_abs_logit_difference":batch_delta,
                              "permutation_seed":args.training_seed+991}
            logits_store["T1_forced_equal"] = equal_logits
            logits_store["T1_shuffled_entropy"] = shuffle_logits
        print(json.dumps({"method":name,"best_epoch":epoch,"val_macro_f1":score["macro_f1"]}),flush=True)
    assert counterfactual is not None
    save_csv(run/"training_history.csv",history)
    save_csv(run/"run_metrics.csv",rows)
    save_csv(run/"per_class.csv",per_class)
    save_csv(run/"known_validation_predictions.csv",predictions)
    save_csv(run/"known_validation_entropy_weights.csv",weights)
    np.savez_compressed(run/"known_validation_logits.npz",flow_ids=val["flow_ids"],
                        labels=val["labels"],**logits_store)
    save_json(run/"counterfactual.json",counterfactual)
    after = {k:sha256(v) for k,v in sources(args.dataset,args.encoder_seed).items()}
    save_json(run/"source_hashes_after.json",after)
    if before != after:
        raise RuntimeError("frozen source hash changed")
    verification = {"status":"PASS","dataset":args.dataset,"encoder_seed":args.encoder_seed,
                    "training_seed":args.training_seed,"train_samples":len(train["labels"]),
                    "validation_samples":len(val["labels"]),"views":3,
                    "known_test_usage":0,"unknown_usage":0,"encoder_updates":0,
                    "adapter_best_epoch":adapter_epoch,"source_hashes_unchanged":True,
                    "checkpoint_selection_role":"known_validation",
                    "entropy_calibration_role":"known_train",
                    "checkpoints":{name:sha256(run/name) for name in
                      ("adapters_best.pt","T0_equal_best.pt","T1_three_entropy_best.pt")},
                    "runtime_seconds":time.monotonic()-start}
    save_json(run/"verification.json",verification)
    (run/"SUCCESS").write_text("SUCCESS\n")
    print(json.dumps({"status":"SUCCESS","dataset":args.dataset,"encoder_seed":args.encoder_seed,
                      "training_seed":args.training_seed,"metrics":rows,"counterfactual":counterfactual}),flush=True)


if __name__ == "__main__":
    main()
