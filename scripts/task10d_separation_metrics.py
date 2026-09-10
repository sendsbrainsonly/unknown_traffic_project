# -*- coding: utf-8 -*-
"""Stage 2 §8.8 第 4 条判据的数值化：可视化分离度指标。

对 task10c 的每个 (embedding, class)：重算 UMAP/t-SNE 2D 坐标（同 seed），
GMM（consensus K, seed 0, 同协议）硬分配标签，报告：
  silhouette_umap / silhouette_tsne  —— 2D 图内分量分离度（越接近 1 越分离）
  mixed_nn_ratio_umap/tsne           —— 最近邻跨分量占比（低 = 图上无混叠）

判据参照（本实验约定）：
  图与统计一致的多峰结构 ≈ 至少一个投影的 silhouette ≥ 0.3 且
  mixed_nn_ratio ≤ 0.5（分量在图上有可见边界而非全混叠）。

用法：python scripts/task10d_separation_metrics.py --classes 2,9,3
输出：outputs/stage2/separation_metrics.csv + separation_metrics_summary.md
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from task10_stage2_audit import _load_aligned, _standardize  # noqa: E402


def _metrics(E, lab):
    from sklearn.metrics import silhouette_score
    sil = silhouette_score(E, lab, sample_size=min(3000, len(E)),
                           random_state=0)
    # 最近邻跨分量比（2D 空间，暴力近邻，点数已采样）
    if len(E) > 3000:
        idx = np.random.default_rng(0).choice(len(E), 3000, replace=False)
        E, lab = E[idx], lab[idx]
    d2 = ((E[:, None, :] - E[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    nn = d2.argmin(1)
    mixed = float((lab[nn] != lab).mean())
    return float(sil), mixed


def main():
    ap = argparse.ArgumentParser(description="可视化分离度指标")
    ap.add_argument("--project-root", default="D:/unknown_traffic_project")
    ap.add_argument("--out", default="outputs/stage2")
    ap.add_argument("--classes", default="2,9,3")
    ap.add_argument("--fig-max-points", type=int, default=5000)
    args = ap.parse_args()

    root = Path(args.project_root)
    out_dir = root / args.out
    cids = [int(c) for c in args.classes.split(",")]

    import umap
    from sklearn.manifold import TSNE
    from sklearn.mixture import GaussianMixture

    zt, labels, zg = _load_aligned(
        root / "outputs/stage1/modelA/embeddings",
        root / "outputs/stage1/modelB/seeds/seed2")
    zt_s, zf = _standardize(zt, zg)
    names = {2: "FTP", 3: "Facetime", 9: "Neris"}
    rows = []
    for emb, Z in (("zf", zf), ("zt", zt_s)):
        for cid in cids:
            X_all = Z["train"][labels["train"] == cid]
            gmm = GaussianMixture(n_components=5, covariance_type="diag",
                                  n_init=3, max_iter=100, reg_covar=1e-3,
                                  random_state=0).fit(X_all)
            plot_idx = np.random.default_rng(0).choice(
                len(X_all), size=min(args.fig_max_points, len(X_all)),
                replace=False)
            X = X_all[plot_idx]
            lab = gmm.predict(X)
            E_u = umap.UMAP(n_components=2, n_neighbors=min(15, len(X) - 1),
                            min_dist=0.1, metric="euclidean",
                            random_state=0).fit_transform(X)
            sil_u, mix_u = _metrics(E_u, lab)
            T = TSNE(n_components=2, perplexity=min(30, max(5, len(X) // 3)),
                     init="pca", random_state=0,
                     learning_rate="auto").fit_transform(X)
            sil_t, mix_t = _metrics(T, lab)
            rows.append({"embedding": emb, "class_id": cid,
                         "class_name": names[cid], "N_plot": len(X),
                         "silhouette_umap": f"{sil_u:.4f}",
                         "silhouette_tsne": f"{sil_t:.4f}",
                         "mixed_nn_ratio_umap": f"{mix_u:.4f}",
                         "mixed_nn_ratio_tsne": f"{mix_t:.4f}",
                         "criterion4_pass":
                             int((sil_u >= 0.3 or sil_t >= 0.3)
                                 and (mix_u <= 0.5 or mix_t <= 0.5))})
            print(f"  {emb} {names[cid]}: sil_u={sil_u:.3f} sil_t={sil_t:.3f} "
                  f"mix_u={mix_u:.3f} mix_t={mix_t:.3f}")
    path = out_dir / "separation_metrics.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    n_pass = sum(r["criterion4_pass"] for r in rows)
    lines = ["# 可视化分离度（第 4 条判据数值化）", "",
             "判据：至少一个投影 silhouette ≥ 0.3 且 mixed_nn_ratio ≤ 0.5。",
             f"代表类通过 {n_pass}/{len(rows)}（如果大多数通过，可视化与统计一致）。",
             "", "| embedding | class | silhouette UMAP | silhouette t-SNE |",
             "|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['embedding']} | {r['class_name']} | "
                     f"{r['silhouette_umap']} | {r['silhouette_tsne']} |")
    (out_dir / "separation_metrics_summary.md").write_text(
        "\n".join(lines), encoding="utf-8")
    print(f"完成 → {path}")


if __name__ == "__main__":
    main()
