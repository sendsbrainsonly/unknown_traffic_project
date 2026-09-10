# -*- coding: utf-8 -*-
"""Stage 1 Model B（FIG→TAGCN）训练脚本。

按 v2 计划 §7 执行：
  - 数据：与 Model A 相同的 splits（默认 compatible_min1，官方 80/10/10）
  - 训练：SGD lr 1e-4 / 50 epoch / batch 64（FEC-OSL 原文配方）；
    仅分类交叉熵（Stage 1 不上聚类分支）
  - 每 epoch 末 dev 评估，best 按 dev Macro-F1 保存
  - 结束后 test 评估（§7.3 全套指标 + 混淆矩阵 + 每类 F1）
  - §7.4 导出 z_g embedding 三件套 + labels + flow_ids 到 --out

示例：
  # 冒烟（2 类 320 流，本地 CPU，几分钟）
  python scripts/task07_train_model_b.py --split smoke_gmail_zeus --epochs 3
  # 全量
  python scripts/task07_train_model_b.py
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.stage1.fig_dataset import (FeatureNormalizer, collate_graphs,
                                    load_fig_splits)
from src.stage1.tagcn import TAGCN


def evaluate(model, loader, device):
    """返回 (metrics dict, y_true, y_pred)。metrics 含 §7.3 全套。"""
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, adj, mask, y, _fids in loader:
            x, adj, mask, y = (t.to(device) for t in (x, adj, mask, y))
            logits, _ = model(x, adj, mask)
            y_pred.extend(logits.argmax(dim=1).cpu().tolist())
            y_true.extend(y.cpu().tolist())
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro")),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro")),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "per_class_f1": {str(c): float(f)
                         for c, f in enumerate(f1_score(y_true, y_pred,
                                                        average=None))},
    }
    return metrics, y_true, y_pred


def export_embeddings(model, loader, device, out_dir, part):
    """§7.4：导出 z_g、labels、flow_ids 三个 npy。返回 (Z, Y, F)。"""
    model.eval()
    zs, ys, fids = [], [], []
    with torch.no_grad():
        for x, adj, mask, y, batch_fids in loader:
            x, adj, mask = (t.to(device) for t in (x, adj, mask))
            _, z_g = model(x, adj, mask)
            zs.append(z_g.cpu().numpy())
            ys.extend(y.tolist())
            fids.extend(batch_fids)
    Z = np.concatenate(zs, axis=0).astype(np.float32)
    Y = np.asarray(ys)
    F = np.asarray(fids)
    np.save(out_dir / f"embedding_{part}.npy", Z)
    np.save(out_dir / f"labels_{part}.npy", Y)
    np.save(out_dir / f"flow_ids_{part}.npy", F)
    return Z, Y, F


def main():
    parser = argparse.ArgumentParser(description="Stage 1 Model B (FIG→TAGCN)")
    parser.add_argument("--project-root", default="D:/unknown_traffic_project")
    parser.add_argument("--split", default="compatible_min1")
    parser.add_argument("--fig-policy", default="all_flows")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--optimizer", default="sgd", choices=["sgd", "adam"])
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--k-hops", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default="outputs/stage1/modelB")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cpu", "cuda"])
    parser.add_argument("--no-cache", action="store_true",
                        help="禁用内存缓存（显存/内存紧张时）")
    parser.add_argument("--no-export", action="store_true",
                        help="跳过 §7.4 embedding 导出与 run_summary（消融用）")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = ("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else args.device

    root = Path(args.project_root)
    out_dir = root / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 装载 FIG splits: "
          f"{args.split} (device={device})")
    datasets, labels_num, normalizer = load_fig_splits(
        root, args.split, args.fig_policy, normalize=True,
        build_cache=not args.no_cache)
    for part, ds in datasets.items():
        print(f"  {part}: {len(ds)} 流")
    print(f"  labels_num={labels_num}, 特征标准化常数: "
          f"mean={normalizer.mean.round(3).tolist()}, "
          f"std={normalizer.std.round(3).tolist()}")

    gen = torch.Generator().manual_seed(args.seed)
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=args.batch_size,
                            shuffle=True, generator=gen,
                            collate_fn=collate_graphs, num_workers=0),
        "val": DataLoader(datasets["val"], batch_size=args.batch_size,
                          collate_fn=collate_graphs, num_workers=0),
        "test": DataLoader(datasets["test"], batch_size=args.batch_size,
                           collate_fn=collate_graphs, num_workers=0),
    }

    model = TAGCN(in_dim=7, hidden=args.hidden, labels_num=labels_num,
                  k_hops=args.k_hops, dropout=args.dropout).to(device)
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                    momentum=args.momentum)  # 默认 = FEC-OSL 配方
    criterion = nn.CrossEntropyLoss()

    history = []
    best = {"epoch": 0, "macro_f1": -1.0, "state": None}
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_batch = 0
        for x, adj, mask, y, _fids in loaders["train"]:
            x, adj, mask, y = (t.to(device) for t in (x, adj, mask, y))
            optimizer.zero_grad()
            logits, _ = model(x, adj, mask)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batch += 1
        dev_m, _, _ = evaluate(model, loaders["val"], device)
        history.append({"epoch": epoch, "train_loss": total_loss / n_batch,
                        **{k: round(v, 4) for k, v in dev_m.items()
                           if isinstance(v, float)}})
        if dev_m["macro_f1"] > best["macro_f1"]:
            best = {"epoch": epoch, "macro_f1": dev_m["macro_f1"],
                    "state": {k: v.clone() for k, v in model.state_dict().items()}}
        print(f"epoch {epoch:3d} | loss {total_loss / n_batch:.4f} | "
              f"dev acc {dev_m['accuracy']:.4f} | dev macro-f1 "
              f"{dev_m['macro_f1']:.4f} | best {best['macro_f1']:.4f}"
              f"(e{best['epoch']}) | {time.time() - t0:.0f}s")

    model.load_state_dict(best["state"])
    torch.save({"state_dict": best["state"], "epoch": best["epoch"],
                "labels_num": labels_num,
                "hidden": args.hidden, "k_hops": args.k_hops},
               out_dir / "modelB_best.pt")

    # test 评估（§7.3）
    test_m, y_true, y_pred = evaluate(model, loaders["test"], device)
    cm = confusion_matrix(y_true, y_pred, labels=range(labels_num))
    print("\n===== Test set evaluation. =====")
    print(f"Acc. (Correct/Total): {test_m['accuracy']:.4f} "
          f"({int(np.sum(y_true == y_pred))}/{len(y_true)})")
    print(f"Macro precision: {test_m['macro_precision']:.4f}, "
          f"Micro precision: {test_m['accuracy']:.4f}")
    print(f"Macro recall: {test_m['macro_recall']:.4f}, "
          f"Macro f1: {test_m['macro_f1']:.4f}")
    print("per-class F1:", {k: round(v, 3) for k, v in
                            test_m["per_class_f1"].items()})
    print("Confusion matrix:")
    print(cm)

    # §7.4 embedding 导出
    if args.no_export:
        print(f"\n(no-export：跳过 embedding 导出与 summary 写入)")
    else:
        for part in ("train", "val", "test"):
            Z, Y, F = export_embeddings(model, loaders[part], device, out_dir, part)
            print(f"  embedding_{part}: {Z.shape} labels {Y.shape} "
                  f"flow_ids {F.shape}")

        summary = {
            "generated_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "model": "TAGCN (FEC-OSL K-hop polynomial, mean readout)",
            "split": args.split, "fig_policy": args.fig_policy,
            "hyper": {"epochs": args.epochs, "lr": args.lr,
                      "optimizer": args.optimizer, "momentum": args.momentum,
                      "batch_size": args.batch_size, "hidden": args.hidden,
                      "k_hops": args.k_hops, "dropout": args.dropout,
                      "seed": args.seed, "device": device},
            "labels_num": labels_num,
            "normalizer": normalizer.to_dict(),
            "best_dev_epoch": best["epoch"],
            "history": history,
            "test": {k: round(v, 4) if isinstance(v, float) else v
                     for k, v in test_m.items()},
            "confusion_matrix": cm.tolist(),
        }
        (out_dir / "run_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n结果与 embedding 已写入 {out_dir}")


if __name__ == "__main__":
    main()
