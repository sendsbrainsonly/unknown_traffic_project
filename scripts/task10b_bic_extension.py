# -*- coding: utf-8 -*-
"""Stage 2 §8 补充诊断：BIC 延伸到 K=6..10，检验 K=5 处 BIC 是否饱和。

背景：task10 冒烟发现全部 3 个代表类 BIC 在 K=1..5 内单调下降、K*=5
（上限饱和）。本脚本对代表类把 K 延伸到 10，看 BIC 最小值是否存在于
K>5；只做诊断，不改变 §8.3 规定的 K=1..5 主协议。

协议与 task10 一致（diag / reg_covar=1e-3 / max_iter=100），但
n_init=1（探索性，控制耗时）。只读 train，Unknown-Free。

用法：
  python scripts/task10b_bic_extension.py --classes 2,9,3
输出：outputs/stage2/bic_extension.csv + bic_extension_summary.md
"""
import argparse
import csv
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture

sys.path.insert(0, str(Path(__file__).resolve().parent))
from task10_stage2_audit import _load_aligned, _standardize, _fit_k_range  # noqa: E402


def _fit_unit(emb, cid, class_name, X, s, max_k):
    """单个 (embedding, class, seed) 单元：K=1..max_k 全曲线。"""
    warn = [0]
    fits = _fit_k_range(X, s, warn, n_init=1)  # K=1..5
    # K=6..max_k 同样协议
    for k in range(6, max_k + 1):
        gmm = GaussianMixture(n_components=k, covariance_type="diag",
                              n_init=1, max_iter=100, reg_covar=1e-3,
                              random_state=s)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            gmm.fit(X)
        warn[0] += sum(1 for w in caught
                       if issubclass(w.category, ConvergenceWarning))
        fits[k] = (float(gmm.bic(X)), gmm)
    kstar = min(fits, key=lambda k: fits[k][0])
    return {"embedding": emb, "class_id": cid, "class_name": class_name,
            "N_train": len(X), "seed": s,
            **{f"BIC_K{k}": f"{fits[k][0]:.2f}" for k in range(1, max_k + 1)},
            "selected_K_full": kstar, "convergence_warnings": warn[0]}


def main():
    ap = argparse.ArgumentParser(description="BIC K=6..10 饱和诊断")
    ap.add_argument("--project-root", default="D:/unknown_traffic_project")
    ap.add_argument("--classes", default="2,9,3")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--max-k", type=int, default=10)
    ap.add_argument("--out", default="outputs/stage2")
    ap.add_argument("--n-jobs", type=int, default=20)
    args = ap.parse_args()

    root = Path(args.project_root)
    zt_dir = root / "outputs/stage1/modelA/embeddings"
    zg_dir = root / "outputs/stage1/modelB/seeds/seed2"
    out_dir = root / args.out
    cids = [int(c) for c in args.classes.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    lm = json.loads((root / "data/fig_graph/all_flows/label_map.json").read_text(encoding="utf-8"))
    id2name = {v: k for k, v in lm["class_to_id"].items()}

    print("[1/2] 装载 + 标准化（同 task10）")
    zt, labels, zg = _load_aligned(zt_dir, zg_dir)
    zt_s, zf = _standardize(zt, zg)
    Z = {"zf": zf, "zt": zt_s}
    for emb in Z:  # float32：减半 loky 序列化成本（GMM 内部转 float64）
        for split in Z[emb]:
            Z[emb][split] = Z[emb][split].astype(np.float32)

    print("[2/2] K=6..%d BIC 曲线（n_init=1，探索性，loky %d workers）"
          % (args.max_k, args.n_jobs))
    # 断点续跑：已完成的 (embedding, class_id, seed) 跳过
    rows = []
    done = set()
    csv_path = out_dir / "bic_extension.csv"
    if csv_path.exists():
        with open(csv_path, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                rows.append(r)
                done.add((r["embedding"], int(r["class_id"]), int(r["seed"])))
        print(f"  续跑：已有 {len(done)} 个单元，跳过")
    units = [(emb, cid, s)
             for emb in ("zf", "zt")
             for cid in cids
             for s in seeds
             if (emb, cid, s) not in done]
    from joblib import Parallel, delayed
    results = Parallel(n_jobs=args.n_jobs, backend="loky", verbose=10)(
        delayed(_fit_unit)(emb, cid, id2name[cid],
                           Z[emb]["train"][labels["train"] == cid], s,
                           args.max_k)
        for emb, cid, s in units)
    rows.extend(results)
    path = out_dir / "bic_extension.csv"
    if rows:
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    # 汇总：每类各 seed 的 K*（1..10 域）分布
    lines = ["# BIC 延伸诊断（K=1..%d，n_init=1 探索性）" % args.max_k, "",
             "| embedding | class | seed | K* (1..%d) |" % args.max_k,
             "|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['embedding']} | {r['class_name']} | {r['seed']} | "
                     f"{r['selected_K_full']} |")
    (out_dir / "bic_extension_summary.md").write_text(
        "\n".join(lines), encoding="utf-8")
    print(f"完成 → {out_dir}")


if __name__ == "__main__":
    main()
