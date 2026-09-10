# -*- coding: utf-8 -*-
"""Stage 1 §7.1 Model C：z_f=[z_t;z_g] 融合分类器 + §7.5 验收。

输入：
  z_t 三件套：outputs/stage1/modelA/embeddings/embedding_{train,val,test}.npy
              + labels_*.npy + flow_ids_*.npy（task08 导出，768 维）
  z_g 三件套：outputs/stage1/modelB/embedding_{train,val,test}.npy
              + labels_*.npy + flow_ids_*.npy（task07 导出，128 维）

协议：
  1) 严格对齐：z_g 按 z_t 的 flow_id 规范序重排后 flow_ids / labels
     逐元素一致（§7.5 检查项 1/6）
  2) 每分支按 train 集逐特征 z-score 标准化后拼接（§7.5 检查项 2/3）
  3) 三个 probe 同协议对比：z_t-only / z_g-only / 融合（concat）
     线性分类头，Adam 1e-3 / 30 epoch / batch 256 / seed 7，
     每 epoch val 评估，按 val Macro-F1 存 best，最后 test 评估（§7.3 全套）
  4) 验收（§7.5）：
     F1_TF+FIG >= max(F1_TF, F1_FIG)
     其中 F1_TF / F1_FIG 为已训练好的端到端单分支 test Macro-F1
     （默认 0.9920 / 0.7306，可用 --f1-tf-ref / --f1-fig-ref 覆盖）

用法：
  python scripts/task09_model_c_fusion.py
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _load_triplet(dir_path, part):
    d = Path(dir_path)
    Z = np.load(d / f"embedding_{part}.npy")
    Y = np.load(d / f"labels_{part}.npy")
    F = np.load(d / f"flow_ids_{part}.npy")
    return Z, Y, F


def _load_and_align(zt_dir, zg_dir):
    """§7.5 检查项 1：flow 对齐。

    z_t（task08）按 tsv_line_number 升序保存 == 切分 TSV 行序（规范顺序）；
    z_g（task07）按 FIG 图节点序保存，行序不同。以 z_t 的 flow_id 序为
    准重排 z_g，再逐元素校验 flow_ids / labels 一致。
    返回 (zt, labels, zg)。
    """
    zt, zg, labels = {}, {}, {}
    for part in ("train", "val", "test"):
        Zt, Yt, ft = _load_triplet(zt_dir, part)
        Zg, Yg, fg = _load_triplet(zg_dir, part)
        assert ft.shape == fg.shape, f"{part}: flow_ids 形状不同"
        if not np.array_equal(ft, fg):
            assert len(set(ft.tolist())) == len(ft), f"{part}: z_t flow_ids 有重复"
            pos_g = {f: i for i, f in enumerate(fg)}  # z_g 中每个 flow 的行号
            missing = set(ft.tolist()) - set(fg.tolist())
            assert not missing, \
                f"{part}: z_g 缺少 {len(missing)} 个 flow: {list(missing)[:3]}"
            idx = np.asarray([pos_g[f] for f in ft])  # 按 z_t 序取 z_g 行
            Zg, Yg, fg = Zg[idx], Yg[idx], fg[idx]
            print(f"  {part}: z_g 行序与 z_t 不同，已按 z_t flow_id 序重排")
        assert np.array_equal(ft, fg), f"{part}: flow_ids 不一致"
        assert np.array_equal(Yt, Yg), f"{part}: labels 不一致"
        zt[part], labels[part], zg[part] = Zt, Yt, Zg
    return zt, labels, zg


def _train_probe(X_tr, Y_tr, X_va, Y_va, dim, labels_num, epochs, lr, bs, seed,
                 device, mlp_hidden=None, tag=""):
    """线性/MLP probe：best-by-val-macro-f1，返回 (model, history, best)。"""
    torch.manual_seed(seed)
    layers = ([nn.Linear(dim, mlp_hidden), nn.ReLU(), nn.Dropout(0.5),
               nn.Linear(mlp_hidden, labels_num)] if mlp_hidden
              else [nn.Linear(dim, labels_num)])
    model = nn.Sequential(*layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    X_tr = torch.as_tensor(X_tr, dtype=torch.float32)
    X_va = torch.as_tensor(X_va, dtype=torch.float32)
    Y_tr = torch.as_tensor(Y_tr, dtype=torch.long)
    Y_va = torch.as_tensor(Y_va, dtype=torch.long)
    n = len(X_tr)
    gen = torch.Generator().manual_seed(seed)
    history, best = [], {"macro_f1": -1.0, "state": None, "epoch": 0}
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, generator=gen)
        total_loss, nb = 0.0, 0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = crit(model(X_tr[idx]), Y_tr[idx])
            loss.backward()
            opt.step()
            total_loss += loss.item()
            nb += 1
        model.eval()
        with torch.no_grad():
            pred = model(X_va).argmax(1)
        f1 = f1_score(Y_va.numpy(), pred.numpy(), average="macro")
        acc = accuracy_score(Y_va.numpy(), pred.numpy())
        if f1 > best["macro_f1"]:
            best = {"macro_f1": float(f1), "epoch": epoch,
                    "state": {k: v.clone() for k, v in model.state_dict().items()}}
        history.append({"epoch": epoch, "val_acc": float(acc), "val_f1": float(f1)})
        print(f"  [{tag}] epoch {epoch:2d} | val acc {acc:.4f} | "
              f"val f1 {f1:.4f} | best {best['macro_f1']:.4f}")
    model.load_state_dict(best["state"])
    return model, history, best


def _evaluate(model, X, Y, device, bs=512):
    model.eval()
    X = torch.as_tensor(X, dtype=torch.float32)
    Y = torch.as_tensor(Y, dtype=torch.long)
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            preds.append(model(X[i:i + bs].to(device)).argmax(1).cpu())
    pred = torch.cat(preds).numpy()
    y = Y.numpy()
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_precision": float(precision_score(y, pred, average="macro")),
        "macro_recall": float(recall_score(y, pred, average="macro")),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "weighted_f1": float(f1_score(y, pred, average="weighted")),
        "per_class_f1": {str(c): float(f)
                         for c, f in enumerate(f1_score(y, pred, average=None))},
    }, y, pred


def main():
    parser = argparse.ArgumentParser(description="Model C 融合 + §7.5 验收")
    parser.add_argument("--project-root", default="D:/unknown_traffic_project")
    parser.add_argument("--zt-dir", default=None, help="默认 <root>/outputs/stage1/modelA/embeddings")
    parser.add_argument("--zg-dir", default=None, help="默认 <root>/outputs/stage1/modelB")
    parser.add_argument("--out", default="outputs/stage1/modelC")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--mlp-hidden", type=int, default=None,
                        help="默认线性头；给定则 MLP(hidden→ReLU→Dropout→out)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--f1-tf-ref", type=float, default=0.9920,
                        help="Model A 端到端 test Macro-F1")
    parser.add_argument("--f1-fig-ref", type=float, default=0.7306,
                        help="Model B 端到端 test Macro-F1")
    args = parser.parse_args()

    root = Path(args.project_root)
    zt_dir = Path(args.zt_dir) if args.zt_dir else root / "outputs/stage1/modelA/embeddings"
    zg_dir = Path(args.zg_dir) if args.zg_dir else root / "outputs/stage1/modelB"
    out_dir = root / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    device = ("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else args.device

    print("[1/4] 装载 z_t / z_g 三件套 + 严格对齐检查")
    zt, labels, zg = _load_and_align(zt_dir, zg_dir)
    labels_num = len(set(labels["train"].tolist()))
    print(f"  z_t: {[(p, zt[p].shape) for p in zt]}")
    print(f"  z_g: {[(p, zg[p].shape) for p in zg]}")
    print(f"  labels_num={labels_num}, flow_ids/labels 对齐 ✓")

    print("[2/4] 逐特征标准化（train 统计）后拼接")
    mu_t = zt["train"].mean(axis=0, keepdims=True)
    sd_t = zt["train"].std(axis=0, keepdims=True)
    sd_t[sd_t < 1e-8] = 1.0
    mu_g = zg["train"].mean(axis=0, keepdims=True)
    sd_g = zg["train"].std(axis=0, keepdims=True)
    sd_g[sd_g < 1e-8] = 1.0
    zt_s, zg_s, zf = {}, {}, {}
    for p in ("train", "val", "test"):
        zt_s[p] = (zt[p] - mu_t) / sd_t
        zg_s[p] = (zg[p] - mu_g) / sd_g
        zf[p] = np.concatenate([zt_s[p], zg_s[p]], axis=1)
    del zt, zg
    print(f"  z_f: {[(p, zf[p].shape) for p in zf]}")

    print("[3/4] 三个 probe 同协议训练")
    cfg = dict(epochs=args.epochs, lr=args.lr, bs=args.batch_size,
               seed=args.seed, device=device, labels_num=labels_num)
    probes = {}
    probes["zt_only"], h_zt, b_zt = _train_probe(
        zt_s["train"], labels["train"], zt_s["val"], labels["val"],
        zt_s["train"].shape[1], tag="zt-only", mlp_hidden=args.mlp_hidden, **cfg)
    probes["zg_only"], h_zg, b_zg = _train_probe(
        zg_s["train"], labels["train"], zg_s["val"], labels["val"],
        zg_s["train"].shape[1], tag="zg-only", mlp_hidden=args.mlp_hidden, **cfg)
    probes["fusion"], h_f, b_f = _train_probe(
        zf["train"], labels["train"], zf["val"], labels["val"],
        zf["train"].shape[1], tag="fusion ", mlp_hidden=args.mlp_hidden, **cfg)

    print("[4/4] test 评估 + §7.5 验收")
    results = {}
    test_x = {"zt_only": zt_s["test"], "zg_only": zg_s["test"],
              "fusion": zf["test"]}
    for tag, model in probes.items():
        m, y_true, y_pred = _evaluate(model, test_x[tag], labels["test"], device)
        results[tag] = m
        if tag == "fusion":
            cm = confusion_matrix(y_true, y_pred, labels=range(labels_num))
            torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                        "input_dim": zf["train"].shape[1],
                        "labels_num": labels_num, "seed": args.seed},
                       out_dir / "modelC_best.pt")
            print(f"\n===== Model C (fusion) Test set evaluation. =====")
            print(f"Acc. (Correct/Total): {m['accuracy']:.4f} "
                  f"({int(np.sum(y_true == y_pred))}/{len(y_true)})")
            print(f"Macro precision: {m['macro_precision']:.4f}, "
                  f"Micro precision: {m['accuracy']:.4f}")
            print(f"Macro recall: {m['macro_recall']:.4f}, "
                  f"Macro f1: {m['macro_f1']:.4f}")
            print("per-class F1:", {k: round(v, 3) for k, v in
                                    m["per_class_f1"].items()})
            print("Confusion matrix:")
            print(cm)
        else:
            print(f"  {tag} test: acc {m['accuracy']:.4f} / "
                  f"macro-f1 {m['macro_f1']:.4f}")

    f1_fusion = results["fusion"]["macro_f1"]
    f1_max = max(args.f1_tf_ref, args.f1_fig_ref)
    verdict = "PASS" if f1_fusion >= f1_max else "FAIL"
    print(f"\n===== §7.5 验收 =====\n"
          f"F1_TF (Model A 端到端)  = {args.f1_tf_ref:.4f}\n"
          f"F1_FIG (Model B 端到端) = {args.f1_fig_ref:.4f}\n"
          f"F1_TF+FIG (Model C)     = {f1_fusion:.4f}\n"
          f"max(F1_TF, F1_FIG)      = {f1_max:.4f}\n"
          f"判定: {verdict} (F1_TF+FIG {'≥' if verdict == 'PASS' else '<'} max)")

    summary = {
        "generated_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": "Model C: z_f=[z_t;z_g] fusion classifier (linear head)",
        "zt_dir": str(zt_dir), "zg_dir": str(zg_dir),
        "hyper": {"epochs": args.epochs, "lr": args.lr,
                  "batch_size": args.batch_size, "seed": args.seed,
                  "mlp_hidden": args.mlp_hidden, "device": device},
        "standardization": "per-feature z-score fitted on train, per branch",
        "probe_histories": {"zt_only": h_zt, "zg_only": h_zg, "fusion": h_f},
        "probe_best": {k: {"macro_f1": v["macro_f1"], "epoch": v["epoch"]}
                       for k, v in {"zt_only": b_zt, "zg_only": b_zg,
                                    "fusion": b_f}.items()},
        "test": {k: {kk: round(vv, 4) if isinstance(vv, float) else vv
                     for kk, vv in v.items()} for k, v in results.items()},
        "confusion_matrix_fusion": cm.tolist(),
        "acceptance": {"f1_tf_ref": args.f1_tf_ref, "f1_fig_ref": args.f1_fig_ref,
                       "f1_fusion": f1_fusion, "max_ref": f1_max,
                       "verdict": verdict},
    }
    (out_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n结果已写入 {out_dir}")


if __name__ == "__main__":
    main()
