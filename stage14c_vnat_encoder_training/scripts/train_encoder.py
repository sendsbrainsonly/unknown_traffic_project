#!/usr/bin/env python3
"""Train one frozen VNAT Known-only Open-Detect encoder."""

from __future__ import annotations

import argparse, csv, json, math, os, random, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import RandomCrop, RandomHorizontalFlip

from common import AUDIT_ROOT, CHECKPOINT_ROOT, EXPECTED_FREEZE_HASH, STAGE14C_ROOT, UPSTREAM_CODE, load_protocol, seed_everything, sha256_file, verify_freeze

sys.path.insert(0, str(AUDIT_ROOT))
sys.path.insert(0, str(STAGE14C_ROOT.parent / "stage3_unknown_utility/scripts"))
from adapters.opendetect_model import released_weight_init  # noqa: E402
from scripts.train_opendetect import released_reset_prototypes, run_epoch, validation_composite  # noqa: E402
from stage3_model import Stage3OpenDetectNet  # noqa: E402


class KnownDataset(Dataset):
    def __init__(self, root: Path, split: str, augment: bool):
        self.images = np.load(root / f"{split}_images.npy", mmap_mode="r", allow_pickle=False)
        self.labels = np.load(root / f"{split}_labels.npy", mmap_mode="r", allow_pickle=False)
        if len(self.images) != len(self.labels): raise RuntimeError("image/label count mismatch")
        self.augment, self.crop, self.flip = augment, RandomCrop(32, padding=4), RandomHorizontalFlip()
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        image = torch.from_numpy(np.asarray(self.images[i]).copy()).unsqueeze(0)
        if self.augment: image = self.flip(self.crop(image))
        return image.float().div_(255.0), int(self.labels[i])


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--protocol-id", required=True); p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=512); p.add_argument("--workers", type=int, default=0)
    p.add_argument("--learning-rate", type=float, default=1e-3); p.add_argument("--patience", type=int, default=5)
    args = p.parse_args()
    if args.batch_size != 512 or args.patience != 5: raise RuntimeError("frozen training config changed")
    freeze = verify_freeze(); protocol = load_protocol(args.protocol_id)
    run_dir = STAGE14C_ROOT / "runs" / args.protocol_id; input_dir = run_dir / "inputs"
    audit = json.loads((input_dir / "input_audit.json").read_text())
    required_zero = ["unknown_samples_used_in_training", "unknown_samples_used_in_validation", "known_test_samples_used", "train_validation_flow_overlap"]
    if audit["status"] != "PASS" or audit["freeze_hash"] != EXPECTED_FREEZE_HASH or any(audit[k] != 0 for k in required_zero): raise RuntimeError("input integrity gate failed")
    if audit["input_manifest_sha256"] != sha256_file(input_dir / "input_manifest.csv"): raise RuntimeError("input manifest changed")
    if audit["known_train_class_intersection_unknown"] or audit["known_validation_class_intersection_unknown"]: raise RuntimeError("Unknown class leakage")
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required")
    metrics_path, config_path = run_dir / "training_metrics.csv", run_dir / "training_config.json"
    selection_path = run_dir / "checkpoint_selection.json"; latest_path = run_dir / "latest_checkpoint.pt"
    CHECKPOINT_ROOT.mkdir(exist_ok=True); best_path = CHECKPOINT_ROOT / f"{args.protocol_id}_best.pt"
    if any(x.exists() for x in [metrics_path, config_path, selection_path, latest_path, best_path]): raise RuntimeError("refusing to overwrite run evidence")

    seed = int(protocol["seed"]); seed_everything(seed); device = torch.device("cuda:0")
    train_ds, val_ds = KnownDataset(input_dir, "train", True), KnownDataset(input_dir, "validation", False)
    gen = torch.Generator().manual_seed(seed); common = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=True, persistent_workers=False)
    train_loader = DataLoader(train_ds, shuffle=True, generator=gen, **common); val_loader = DataLoader(val_ds, shuffle=False, **common)
    num_classes = len(protocol["known_applications"])
    model = Stage3OpenDetectNet(upstream_code=UPSTREAM_CODE, channels=1, latent_dim=128, num_classes=num_classes, temp_inter=1.0, temp_intra=1.0).to(device)
    model.apply(released_weight_init); optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(.9,.999))
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50,80], gamma=.1)
    config = {
        "protocol_id": args.protocol_id, "setting": protocol["setting"], "seed": seed,
        "known_classes": protocol["known_applications"], "unknown_classes_excluded": protocol["unknown_applications"],
        "unknown_samples_loaded": 0, "known_test_samples_loaded": 0, "freeze_hash": freeze["freeze_hash"],
        "protocol_sha256": freeze["protocol_sha256"], "split_manifest_sha256": freeze["split_manifest_sha256"],
        "input_manifest_sha256": audit["input_manifest_sha256"], "epochs": args.epochs, "batch_size": args.batch_size,
        "workers": args.workers, "learning_rate": args.learning_rate, "optimizer": "Adam(beta1=0.9,beta2=0.999)",
        "scheduler": "MultiStepLR(milestones=[50,80],gamma=0.1)", "lambda": .005, "latent_dimension": 128,
        "checkpoint_selection": "highest harmonic mean of Known Validation Accuracy and Macro-F1",
        "early_stopping_monitor": "Known Validation composite only", "early_stopping_patience": args.patience,
        "initialization": "released_weight_init from scratch; no checkpoint", "numerical_stability_guard": "raw logvar upper tail clamped at 20",
        "train_samples": len(train_ds), "validation_samples": len(val_ds), "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "unrecorded"),
        "cuda_device_name": torch.cuda.get_device_name(0), "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
    }
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True)+"\n")
    rows=[]; best_score=-math.inf; best_epoch=-1; no_improve=0; started=time.time()
    for epoch_index in range(args.epochs):
        guard0=model.stability_guard_activations; train=run_epoch(model, train_loader, device, .005, optimizer); guard1=model.stability_guard_activations
        reset=epoch_index in (50,80)
        if reset: model.prototypes = released_reset_prototypes(model, train_loader, device)
        val=run_epoch(model, val_loader, device, .005, None); guard2=model.stability_guard_activations; scheduler.step()
        score=validation_composite(val["accuracy"], val["macro_f1"])
        if not all(math.isfinite(float(v)) for v in list(train.values())+list(val.values())+[score]): raise RuntimeError("non-finite training metric")
        row={"epoch":epoch_index+1,"learning_rate":optimizer.param_groups[0]["lr"],"prototype_reset":reset,"elapsed_seconds":time.time()-started,"val_composite_score":score,"stability_guard_train_activations":guard1-guard0,"stability_guard_validation_activations":guard2-guard1,"stability_guard_cumulative_activations":guard2}
        for prefix, values in (("train",train),("val",val)):
            row.update({f"{prefix}_{k}":v for k,v in values.items()})
        rows.append(row); write_csv(metrics_path, rows)
        improved=score>best_score
        if improved: best_score, best_epoch, no_improve=score, epoch_index+1, 0
        else: no_improve += 1
        payload={"model_state_dict":model.state_dict(),"optimizer_state_dict":optimizer.state_dict(),"scheduler_state_dict":scheduler.state_dict(),"epoch":epoch_index+1,"train_metrics":train,"val_metrics":val,"config":config,"best_epoch":best_epoch,"best_validation_composite":best_score,"epochs_without_improvement":no_improve}
        if improved: torch.save(payload,best_path)
        torch.save(payload,latest_path)
        print(json.dumps({"protocol_id":args.protocol_id,"epoch":epoch_index+1,"train_total":train["total"],"val_total":val["total"],"val_accuracy":val["accuracy"],"val_macro_f1":val["macro_f1"],"best_epoch":best_epoch,"no_improve":no_improve}),flush=True)
        if no_improve>=args.patience: break
    best=torch.load(best_path,map_location="cpu",weights_only=False)
    selection={"protocol_id":args.protocol_id,"selection_data":"Known Validation only","unknown_samples_used":0,"known_test_samples_used":0,"best_epoch":int(best["epoch"]),"stop_epoch":int(rows[-1]["epoch"]),"train_loss":float(best["train_metrics"]["total"]),"val_loss":float(best["val_metrics"]["total"]),"val_accuracy":float(best["val_metrics"]["accuracy"]),"val_macro_f1":float(best["val_metrics"]["macro_f1"]),"combined_score":float(best["best_validation_composite"]),"checkpoint_path":str(best_path),"checkpoint_sha256":sha256_file(best_path),"freeze_hash":freeze["freeze_hash"],"converged_without_nan":True,"unknown_inference_executed":False}
    selection_path.write_text(json.dumps(selection,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"event":"run_complete",**selection},sort_keys=True),flush=True)


if __name__ == "__main__": main()
