# -*- coding: utf-8 -*-
"""Stage 2 §8.8 可视化补跑（独立于 task10 主流程）。

task10 全量跑完时 umap-learn 尚未装好，figures/ 为空。本脚本从
selected_k_per_seed.csv 读取 consensus K，对指定类重做 UMAP/t-SNE 图
（GMM 按 consensus K 重新拟合 seed 0，绘图点子采样）。

用法：
  python scripts/task10c_figures.py --classes 2,9,3
输出：outputs/stage2/figures/umap_{emb}_class{cid}_K{k}.png 等
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from task10_stage2_audit import _load_aligned, _standardize  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Stage 2 §8.8 可视化补跑")
    ap.add_argument("--project-root", default="D:/unknown_traffic_project")
    ap.add_argument("--out", default="outputs/stage2")
    ap.add_argument("--classes", default="2,9,3",
                    help="逗号分隔 label_id（默认 FTP/Neris/Facetime）")
    ap.add_argument("--fig-max-points", type=int, default=5000)
    args = ap.parse_args()

    root = Path(args.project_root)
    out_dir = root / args.out
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    cids = [int(c) for c in args.classes.split(",")]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import umap
    from sklearn.manifold import TSNE
    from sklearn.mixture import GaussianMixture

    # 读取 consensus K
    consensus, names = {}, {}
    with open(out_dir / "selected_k_per_seed.csv", newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if int(r["class_id"]) in cids:
                ks = [int(r["K_seed0"]), int(r["K_seed1"]), int(r["K_seed2"])]
                consensus[int(r["class_id"])] = max(set(ks), key=ks.count)
                names[int(r["class_id"])] = r["class_name"]
    print(f"consensus K = {consensus}")

    # 装载 embedding（同 task10 协议）
    zt, labels, zg = _load_aligned(
        root / "outputs/stage1/modelA/embeddings",
        root / "outputs/stage1/modelB/seeds/seed2")
    zt_s, zf = _standardize(zt, zg)
    for p in ("train", "val"):
        zt_s[p] = zt_s[p].astype("float32")
        zf[p] = zf[p].astype("float32")

    cmap = plt.get_cmap("tab10")
    made = []
    for emb, Z in (("zf", zf), ("zt", zt_s)):
        for cid in cids:
            k = consensus[cid]
            X_all = Z["train"][labels["train"] == cid]
            gmm = GaussianMixture(n_components=k, covariance_type="diag",
                                  n_init=3, max_iter=100, reg_covar=1e-3,
                                  random_state=0).fit(X_all)
            plot_idx = np.random.default_rng(0).choice(
                len(X_all), size=min(args.fig_max_points, len(X_all)),
                replace=False)
            X = X_all[plot_idx]
            lab = gmm.predict(X)
            # UMAP
            red = umap.UMAP(n_components=2, n_neighbors=min(15, len(X) - 1),
                            min_dist=0.1, metric="euclidean", random_state=0)
            E = red.fit_transform(X)
            fig, ax = plt.subplots(figsize=(7, 6))
            for j in range(k):
                ax.scatter(E[lab == j, 0], E[lab == j, 1], s=6, alpha=0.5,
                           color=cmap(j % 10), label=f"comp {j}")
            C = red.transform(gmm.means_)
            ax.scatter(C[:, 0], C[:, 1], marker="X", s=220, edgecolors="k",
                       linewidths=1.2, color="w", label="center")
            ax.set_title(f"{emb} | {names[cid]} (K={k}) UMAP")
            ax.legend(markerscale=2, fontsize=8)
            p = figures_dir / f"umap_{emb}_class{cid}_K{k}.png"
            fig.savefig(p, dpi=150, bbox_inches="tight")
            plt.close(fig)
            made.append(str(p))
            # t-SNE
            perp = min(30, max(5, len(X) // 3))
            T = TSNE(n_components=2, perplexity=perp, init="pca",
                     random_state=0, learning_rate="auto").fit_transform(X)
            fig, ax = plt.subplots(figsize=(7, 6))
            for j in range(k):
                ax.scatter(T[lab == j, 0], T[lab == j, 1], s=6, alpha=0.5,
                           color=cmap(j % 10), label=f"comp {j}")
            ax.set_title(f"{emb} | {names[cid]} (K={k}) t-SNE (perp={perp})")
            ax.legend(markerscale=2, fontsize=8)
            p = figures_dir / f"tsne_{emb}_class{cid}_K{k}.png"
            fig.savefig(p, dpi=150, bbox_inches="tight")
            plt.close(fig)
            made.append(str(p))
            print(f"  [fig] {emb} class {cid} ({names[cid]}) K={k} 完成")

    print(f"完成 {len(made)} 张图 → {figures_dir}")


if __name__ == "__main__":
    main()
