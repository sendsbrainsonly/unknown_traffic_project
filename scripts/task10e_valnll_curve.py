# -*- coding: utf-8 -*-
"""Stage 2 §8 补充诊断：val NLL 随 K 的曲线（K=1..10）。

task10b 发现训练集 BIC 在 K≤10 内单调降（饱和）。本脚本回答决定性
问题：K>5 时 held-out val NLL 是继续改善（真结构）还是变差（过拟合/
方差地板伪影）——即 §8.5 检验精神的延伸。

协议：diag / reg_covar=1e-3 / max_iter=100 / n_init=1（探索性），
seed 0，只读 train+val（Unknown-Free）。

用法：python scripts/task10e_valnll_curve.py --classes 2,9,3
输出：outputs/stage2/valnll_curve.csv + valnll_curve_summary.md
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.mixture import GaussianMixture

sys.path.insert(0, str(Path(__file__).resolve().parent))
from task10_stage2_audit import _load_aligned, _standardize  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="val NLL 随 K 曲线（K=1..10）")
    ap.add_argument("--project-root", default="D:/unknown_traffic_project")
    ap.add_argument("--out", default="outputs/stage2")
    ap.add_argument("--classes", default="2,9,3")
    ap.add_argument("--max-k", type=int, default=10)
    args = ap.parse_args()

    root = Path(args.project_root)
    out_dir = root / args.out
    cids = [int(c) for c in args.classes.split(",")]
    names = {2: "FTP", 3: "Facetime", 9: "Neris"}

    zt, labels, zg = _load_aligned(
        root / "outputs/stage1/modelA/embeddings",
        root / "outputs/stage1/modelB/seeds/seed2")
    zt_s, zf = _standardize(zt, zg)

    rows = []
    for emb, Z in (("zf", zf), ("zt", zt_s)):
        for cid in cids:
            X_tr = Z["train"][labels["train"] == cid]
            X_va = Z["val"][labels["val"] == cid]
            t0 = time.time()
            prev = None
            for k in range(1, args.max_k + 1):
                gmm = GaussianMixture(n_components=k, covariance_type="diag",
                                      n_init=1, max_iter=100, reg_covar=1e-3,
                                      random_state=0).fit(X_tr)
                nll = float(-gmm.score(X_va))
                rows.append({"embedding": emb, "class_id": cid,
                             "class_name": names[cid], "K": k,
                             "val_nll": f"{nll:.6f}",
                             "train_bic": f"{gmm.bic(X_tr):.2f}",
                             "d_val_nll_from_prev":
                                 f"{(nll - prev) if prev is not None else np.nan:.6f}"
                             })
                prev = nll
            print(f"  {emb} {names[cid]} K=1..{args.max_k} "
                  f"({time.time() - t0:.0f}s)")
    path = out_dir / "valnll_curve.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    lines = ["# val NLL 随 K 曲线（K=1..%d，n_init=1 探索性）" % args.max_k,
             "", "NLL 越低越好；d_val_nll_from_prev < 0 表示该步 K 增加"
             "在 held-out 上仍有益。", ""]
    for r in rows:
        d = float(r["d_val_nll_from_prev"])
        sign = "↓改善" if d < 0 else ("↑变差" if d > 0 else "—")
        lines.append(f"| {r['embedding']} | {r['class_name']} | K={r['K']} | "
                     f"{r['val_nll']} | {sign} |")
    (out_dir / "valnll_curve_summary.md").write_text(
        "\n".join(lines), encoding="utf-8")
    print(f"完成 → {path}")


if __name__ == "__main__":
    main()
