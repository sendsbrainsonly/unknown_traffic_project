# -*- coding: utf-8 -*-
"""Stage 2 §8 Known-Space Distribution Audit。

对每个 Known 类 c（train 样本）：
  1) GMM K=1..5，covariance_type="diag"（§8.3 第一版）；正式 seed 拟合
     n_init=3，bootstrap 拟合 n_init=1，max_iter=100（sklearn 默认）；
     reg_covar=1e-3（对 §8.3 的唯一实现偏离：sklearn 默认 1e-6 在冒烟中
     被发现会让类内近常数维度把方差压到地板、以 +5.5/维度 的量级支配
     似然尺度，BIC 因此全类饱和选 K=5；1e-3 是单位方差下 0.1% 的方差
     地板，把 BIC/NLL 比较拉回真实结构）；
  2) BIC 自动选 K（§8.4）：K_c* = argmin BIC(GMM_K(Z_c))；
  3) held-out NLL 检验（§8.5）：在 Known val 上比较 NLL_single 与 NLL_gmm；
  4) 稳定性（§8.6）：seed 0/1/2 + bootstrap（80% 样本 × N 次）统计 P(K*=k)；
  5) 小 cluster 统计（§8.7）：N_min = max(20, 0.01*N_c) 及 0.5%/1%/2% 敏感性；
  6) 可视化（§8.8，仅辅助）：K=1 / K=2 / K≥3 代表类的 UMAP + t-SNE。

主分析 embedding = z_f（[z_t;z_g] 按 train 统计标准化，与 Model C 输入一致）；
对照 = 标准化 z_t。只使用 train + val（§8.2；严禁 Unknown 类与 test 泄漏）。

产物（§8.10）：bic_per_class.csv / validation_nll.csv / selected_k_per_seed.csv /
bootstrap_k_stability.csv / cluster_size_statistics.csv /
distribution_audit_summary.md（含 §8.11 验收判定）/ figures/。

用法：
  冒烟：python scripts/task10_stage2_audit.py --classes 2,9,3 --bootstrap-reps 2 --skip-figs
  全量：python scripts/task10_stage2_audit.py
"""
import argparse
import csv
import json
import sys
import time
import warnings
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

K_RANGE = [1, 2, 3, 4, 5]
EMBEDDINGS = ("zf", "zt")  # 主分析 z_f，对照 z_t


def _load_triplet(dir_path, part):
    d = Path(dir_path)
    return (np.load(d / f"embedding_{part}.npy"),
            np.load(d / f"labels_{part}.npy"),
            np.load(d / f"flow_ids_{part}.npy"))


def _load_aligned(zt_dir, zg_dir):
    """同 task09：z_g 按 z_t 的 flow_id 规范序重排，逐元素校验。只取 train/val（§8.2）。"""
    zt, labels, zg = {}, {}, {}
    for part in ("train", "val"):
        Zt, Yt, ft = _load_triplet(zt_dir, part)
        Zg, Yg, fg = _load_triplet(zg_dir, part)
        if not np.array_equal(ft, fg):
            assert len(set(ft.tolist())) == len(ft), f"{part}: z_t flow_ids 有重复"
            pos_g = {f: i for i, f in enumerate(fg)}
            missing = set(ft.tolist()) - set(fg.tolist())
            assert not missing, f"{part}: z_g 缺少 {len(missing)} 个 flow"
            idx = np.asarray([pos_g[f] for f in ft])
            Zg, Yg, fg = Zg[idx], Yg[idx], fg[idx]
        assert np.array_equal(ft, fg), f"{part}: flow_ids 不一致"
        assert np.array_equal(Yt, Yg), f"{part}: labels 不一致"
        zt[part], labels[part], zg[part] = Zt, Yt, Zg
    return zt, labels, zg


def _standardize(zt, zg):
    """每分支按 train 统计逐特征 z-score（sd<1e-8→1），拼接 z_f；同 task09 协议。"""
    mu_t, sd_t = zt["train"].mean(axis=0, keepdims=True), zt["train"].std(axis=0, keepdims=True)
    sd_t[sd_t < 1e-8] = 1.0
    mu_g, sd_g = zg["train"].mean(axis=0, keepdims=True), zg["train"].std(axis=0, keepdims=True)
    sd_g[sd_g < 1e-8] = 1.0
    zt_s, zf = {}, {}
    for p in ("train", "val"):
        zt_s[p] = (zt[p] - mu_t) / sd_t
        zf[p] = np.concatenate([(zt[p] - mu_t) / sd_t, (zg[p] - mu_g) / sd_g], axis=1)
    return zt_s, zf


def _fit_k_range(X, seed, warn_box, n_init):
    """拟合 K=1..5 diag GMM，返回 {K: (BIC, gmm)}；EM 未收敛次数计入 warn_box[0]。

    n_init：正式 seed 拟合用 3（BIC 可靠性优先）；bootstrap 用 1
    （20 次重采样本身已覆盖初值可变性，控制总耗时）。max_iter=100 为
    sklearn 默认；reg_covar=1e-3 理由见文件头。"""
    out = {}
    for k in K_RANGE:
        gmm = GaussianMixture(n_components=k, covariance_type="diag",
                              n_init=n_init, max_iter=100, reg_covar=1e-3,
                              random_state=seed)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            gmm.fit(X)
        warn_box[0] += sum(1 for w in caught
                           if issubclass(w.category, ConvergenceWarning))
        out[k] = (float(gmm.bic(X)), gmm)
    return out


def _audit_class(emb_name, cid, cname, X_tr, X_va, seeds, boot_reps, boot_seed):
    """单类全套审计，返回结构化 dict（joblib 并行单元）。"""
    N_tr, N_va = len(X_tr), len(X_va)
    warn_box = [0]
    res = {"embedding": emb_name, "class_id": cid, "class_name": cname,
           "N_train": N_tr, "N_val": N_va,
           "bic_rows": [], "nll_rows": [], "cluster_rows": [],
           "K_per_seed": {}, "boot_counts": {}, "warn": 0}
    # ---- §8.4/8.5/8.6/8.7：seed 0/1/2 ----
    for s in seeds:
        fits = _fit_k_range(X_tr, s, warn_box, n_init=3)
        kstar = min(fits, key=lambda k: fits[k][0])
        res["K_per_seed"][s] = kstar
        res["bic_rows"].append(
            {"seed": s, **{f"BIC_K{k}": f"{fits[k][0]:.4f}" for k in K_RANGE},
             "selected_K": kstar})
        nll_single = float(-fits[1][1].score(X_va))
        nll_gmm = float(-fits[kstar][1].score(X_va))
        res["nll_rows"].append(
            {"seed": s, "selected_K": kstar,
             "nll_single": f"{nll_single:.6f}",
             "nll_gmm": f"{nll_gmm:.6f}",
             "nll_improved": int(nll_gmm < nll_single),
             "relative_improvement_pct":
                 f"{100.0 * (nll_single - nll_gmm) / max(abs(nll_single), 1e-12):.4f}"
             })
        # ---- §8.7 小 cluster ----
        sizes = np.bincount(fits[kstar][1].predict(X_tr), minlength=kstar)
        thr_05, thr_1, thr_2 = (0.005 * N_tr, 0.01 * N_tr, 0.02 * N_tr)
        res["cluster_rows"].append(
            {"seed": s, "selected_K": kstar,
             "component_sizes": ",".join(str(int(x)) for x in sizes),
             "min_component_size": int(sizes.min()),
             "N_min_1pct": int(max(20, 0.01 * N_tr)),
             "thr_0.5pct": f"{thr_05:.1f}", "thr_1pct": f"{thr_1:.1f}",
             "thr_2pct": f"{thr_2:.1f}",
             "n_below_0.5pct": int((sizes < thr_05).sum()),
             "n_below_1pct": int((sizes < thr_1).sum()),
             "n_below_2pct": int((sizes < thr_2).sum())})
    # ---- §8.6 bootstrap：80% × boot_reps（seed 固定 boot_seed）----
    rng = np.random.default_rng(boot_seed)
    counts = Counter()
    for r in range(boot_reps):
        idx = rng.choice(N_tr, size=max(2, int(0.8 * N_tr)), replace=False)
        fits = _fit_k_range(X_tr[idx], boot_seed + r, warn_box, n_init=1)
        counts[min(fits, key=lambda k: fits[k][0])] += 1
    res["boot_counts"] = {k: counts.get(k, 0) for k in K_RANGE}
    res["warn"] = warn_box[0]
    return res


def _figure_representatives(out_dir, figures_dir, results_by_emb, label_names,
                            fig_max_points=5000):
    """§8.8：K=1 / K=2 / K≥3 各挑一个代表类（最大 N_train），UMAP+t-SNE 着色。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import umap
        from sklearn.manifold import TSNE
    except ImportError as e:
        print(f"[figures] 跳过：{e}（pip install umap-learn scikit-learn matplotlib）")
        return []
    made = []
    for emb_name, res_list in results_by_emb.items():
        consensus = {r["class_id"]: Counter(r["K_per_seed"].values()).most_common(1)[0][0]
                     for r in res_list}
        groups = {"K1": [c for c, k in consensus.items() if k == 1],
                  "K2": [c for c, k in consensus.items() if k == 2],
                  "K3p": [c for c, k in consensus.items() if k >= 3]}
        # 代表类 = 各组 N_train 最大者（组为空则跳过）
        reps = {}
        for g, cids in groups.items():
            if cids:
                reps[g] = max(cids, key=lambda c: next(
                    r["N_train"] for r in res_list if r["class_id"] == c))
        for g, cid in reps.items():
            r = next(x for x in res_list if x["class_id"] == cid)
            k = consensus[cid]
            # 重新拟合用于着色（seed 0，与正式协议一致）；绘图点子采样
            X_all = _CLASS_X[emb_name]["train"][_CLASS_MASK[emb_name]["train"] == cid]
            gmm = GaussianMixture(n_components=k, covariance_type="diag",
                                  n_init=3, max_iter=100, reg_covar=1e-3,
                                  random_state=0).fit(X_all)
            plot_idx = np.random.default_rng(0).choice(
                len(X_all), size=min(fig_max_points, len(X_all)), replace=False)
            X = X_all[plot_idx]
            lab = gmm.predict(X)
            cmap = plt.get_cmap("tab10")
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
            ax.set_title(f"{emb_name} | {r['class_name']} (K={k}) UMAP")
            ax.legend(markerscale=2, fontsize=8)
            p = figures_dir / f"umap_{emb_name}_class{cid}_K{k}.png"
            fig.savefig(p, dpi=150, bbox_inches="tight")
            plt.close(fig)
            # t-SNE（无 transform 接口，只画点）
            perp = min(30, max(5, len(X) // 3))
            T = TSNE(n_components=2, perplexity=perp, init="pca",
                     random_state=0, learning_rate="auto").fit_transform(X)
            fig, ax = plt.subplots(figsize=(7, 6))
            for j in range(k):
                ax.scatter(T[lab == j, 0], T[lab == j, 1], s=6, alpha=0.5,
                           color=cmap(j % 10), label=f"comp {j}")
            ax.set_title(f"{emb_name} | {r['class_name']} (K={k}) t-SNE (perp={perp})")
            ax.legend(markerscale=2, fontsize=8)
            p = figures_dir / f"tsne_{emb_name}_class{cid}_K{k}.png"
            fig.savefig(p, dpi=150, bbox_inches="tight")
            plt.close(fig)
            made.append(str(p))
            print(f"  [fig] {emb_name} class {cid} ({r['class_name']}) K={k} 完成")
    return made


# 供可视化使用的全局容器（仅主进程写，绘图在主进程顺序执行）
_CLASS_X, _CLASS_MASK = {}, {}


def main():
    ap = argparse.ArgumentParser(description="Stage 2 §8 Known-Space Distribution Audit")
    ap.add_argument("--project-root", default="D:/unknown_traffic_project")
    ap.add_argument("--zt-dir", default=None, help="默认 <root>/outputs/stage1/modelA/embeddings")
    ap.add_argument("--zg-dir", default=None, help="默认 <root>/outputs/stage1/modelB/seeds/seed2")
    ap.add_argument("--out", default="outputs/stage2")
    ap.add_argument("--classes", default=None, help="逗号分隔 label_id 子集（冒烟）；默认全部 20 类")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--bootstrap-reps", type=int, default=20)
    ap.add_argument("--bootstrap-seed", type=int, default=0)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--backend", default="loky", choices=["loky", "threading"],
                    help="joblib 后端；numpy 小算子抢 GIL，threading 会退化单线程")
    ap.add_argument("--skip-figs", action="store_true")
    ap.add_argument("--fig-max-points", type=int, default=5000,
                    help="可视化每类最多采样的点数（辅助图，不改变统计）")
    args = ap.parse_args()

    root = Path(args.project_root)
    zt_dir = Path(args.zt_dir) if args.zt_dir else root / "outputs/stage1/modelA/embeddings"
    zg_dir = Path(args.zg_dir) if args.zg_dir else root / "outputs/stage1/modelB/seeds/seed2"
    out_dir = root / args.out
    figures_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_figs:
        figures_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",")]
    cids = [int(c) for c in args.classes.split(",")] if args.classes else list(range(20))

    # 类名映射
    lm = json.loads((root / "data/fig_graph/all_flows/label_map.json").read_text(encoding="utf-8"))
    id2name = {v: k for k, v in lm["class_to_id"].items()}
    names = {c: id2name.get(c, str(c)) for c in cids}

    print("[1/4] 装载 z_t / z_g（train+val） + 严格对齐")
    zt, labels, zg = _load_aligned(zt_dir, zg_dir)
    zt_s, zf = _standardize(zt, zg)
    del zt, zg
    # float32：GMM 内部会转 float64，这里只减 loky 序列化与主进程内存
    zt_s = {p: z.astype(np.float32) for p, z in zt_s.items()}
    zf = {p: z.astype(np.float32) for p, z in zf.items()}
    Z = {"zf": zf, "zt": zt_s}
    n_classes_in_train = len(set(labels["train"].tolist()))
    print(f"  z_f: {[(p, zf[p].shape) for p in zf]}")
    print(f"  z_t(标准化): {[(p, zt_s[p].shape) for p in zt_s]}")
    print(f"  train 类数 = {n_classes_in_train}，审计类子集 = {cids}")

    print("[2/4] 逐类 GMM 审计（joblib 并行）")
    t0 = time.time()
    tasks = []
    for emb in EMBEDDINGS:
        _CLASS_X[emb], _CLASS_MASK[emb] = {}, {}
        for p in ("train", "val"):
            _CLASS_X[emb][p] = Z[emb][p]
            _CLASS_MASK[emb][p] = labels[p]
    for emb in EMBEDDINGS:
        for cid in cids:
            X_tr = Z[emb]["train"][labels["train"] == cid]
            X_va = Z[emb]["val"][labels["val"] == cid]
            assert len(X_tr) > 5 and len(X_va) > 0, \
                f"{emb} class {cid}: train={len(X_tr)} val={len(X_va)} 样本不足"
            tasks.append((emb, cid, names[cid], X_tr, X_va,
                          seeds, args.bootstrap_reps, args.bootstrap_seed))
    results = Parallel(n_jobs=args.n_jobs, backend=args.backend, verbose=10)(
        delayed(_audit_class)(*t) for t in tasks)
    print(f"[2/4] 审计完成，耗时 {(time.time() - t0) / 60:.1f} 分钟，"
          f"共 {len(results)} 个类-embedding 单元")

    # ---- 落盘 CSV ----
    by_emb = {emb: [r for r in results if r["embedding"] == emb] for emb in EMBEDDINGS}
    def _csv(name, fieldnames, row_fn):
        path = out_dir / name
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            w.writeheader()
            for r in results:
                for row in row_fn(r):
                    w.writerow(row)
        print(f"  写出 {path.name}")

    _csv("bic_per_class.csv",
         ["embedding", "class_id", "class_name", "N_train", "N_val", "seed"]
         + [f"BIC_K{k}" for k in K_RANGE] + ["selected_K"],
         lambda r: [{**{"embedding": r["embedding"], "class_id": r["class_id"],
                        "class_name": r["class_name"], "N_train": r["N_train"],
                        "N_val": r["N_val"]}, **row} for row in r["bic_rows"]])
    _csv("validation_nll.csv",
         ["embedding", "class_id", "class_name", "N_train", "N_val", "seed",
          "selected_K", "nll_single", "nll_gmm", "nll_improved",
          "relative_improvement_pct"],
         lambda r: [{**{"embedding": r["embedding"], "class_id": r["class_id"],
                        "class_name": r["class_name"], "N_train": r["N_train"],
                        "N_val": r["N_val"]}, **row} for row in r["nll_rows"]])
    _csv("selected_k_per_seed.csv",
         ["embedding", "class_id", "class_name", "N_train"]
         + [f"K_seed{s}" for s in seeds] + ["consensus_K"],
         lambda r: [{"embedding": r["embedding"], "class_id": r["class_id"],
                     "class_name": r["class_name"], "N_train": r["N_train"],
                     **{f"K_seed{s}": r["K_per_seed"][s] for s in seeds},
                     "consensus_K": Counter(r["K_per_seed"].values()).most_common(1)[0][0]}])
    _csv("bootstrap_k_stability.csv",
         ["embedding", "class_id", "class_name", "N_train", "K", "count", "prob"],
         lambda r: [{"embedding": r["embedding"], "class_id": r["class_id"],
                     "class_name": r["class_name"], "N_train": r["N_train"],
                     "K": k, "count": r["boot_counts"][k],
                     "prob": f"{r['boot_counts'][k] / args.bootstrap_reps:.3f}"}
                    for k in K_RANGE])
    _csv("cluster_size_statistics.csv",
         ["embedding", "class_id", "class_name", "N_train", "seed", "selected_K",
          "component_sizes", "min_component_size", "N_min_1pct", "thr_0.5pct",
          "thr_1pct", "thr_2pct", "n_below_0.5pct", "n_below_1pct", "n_below_2pct"],
         lambda r: [{**{"embedding": r["embedding"], "class_id": r["class_id"],
                        "class_name": r["class_name"], "N_train": r["N_train"]},
                     **row} for row in r["cluster_rows"]])

    # ---- §8.9 统计量 + §8.11 验收 + summary md ----
    print("[3/4] 汇总 §8.9 统计量 / §8.11 验收")
    lines = []
    add = lines.append
    add(f"# Stage 2 Known-Space Distribution Audit 汇总")
    add(f"\n生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    add(f"\n输入：z_t = {zt_dir}；z_g(seed2) = {zg_dir}；只使用 train+val（§8.2）。")
    add(f"协议：GMM K=1..5 diag，BIC 选 K；seed {seeds} 拟合 n_init=3；"
        f"bootstrap 80%×{args.bootstrap_reps}（seed {args.bootstrap_seed}）"
        f"拟合 n_init=1；max_iter=100；reg_covar=1e-3（理由：冒烟发现 1e-6 "
        f"会让类内近常数维度支配似然尺度、BIC 全类饱和 K=5）；"
        f"审计类 = {cids}。")
    add("\n## §8.9 关键统计量")
    verdicts = {}
    for emb in EMBEDDINGS:
        rs = by_emb[emb]
        C = len(rs)
        consensus = [Counter(r["K_per_seed"].values()).most_common(1)[0][0] for r in rs]
        ks_all = [r["K_per_seed"][s] for r in rs for s in seeds]
        r_multi = sum(1 for k in consensus if k > 1) / C
        n_improved_all = sum(1 for r in rs
                             if all(row["nll_improved"] for row in r["nll_rows"]))
        rel_imps = [float(row["relative_improvement_pct"])
                    for r in rs for row in r["nll_rows"]
                    if row["nll_improved"]]
        top_prob = [max(r["boot_counts"].values()) / args.bootstrap_reps for r in rs]
        stable = sum(1 for p in top_prob if p >= 0.9) / C
        add(f"\n### {emb}（{'主分析' if emb == 'zf' else '对照'}）")
        add(f"- R_multi（consensus K>1 的类占比）= {r_multi:.2f}（{sum(1 for k in consensus if k > 1)}/{C}）")
        add(f"- selected K（全部 {len(seeds)} seed × {C} 类）：mean {np.mean(ks_all):.2f}，"
            f"median {np.median(ks_all):.2f}，max {max(ks_all)}")
        labels_txt = ", ".join(
            f"{names.get(r['class_id'], r['class_id'])}={k}"
            for r, k in zip(rs, consensus))
        add(f"- 逐类 consensus：{labels_txt}")
        add(f"- held-out NLL：3 seed 均改进的类 {n_improved_all}/{C}；"
            f"改进类的中位相对提升 {np.median(rel_imps):.3f}%（n={len(rel_imps)}）")
        add(f"- bootstrap 稳定性（top prob ≥ 0.9 的类占比）= {stable:.2f}")
        add(f"- EM 收敛警告次数：{sum(r['warn'] for r in rs)}")
        verdicts[emb] = dict(C=C, r_multi=r_multi, n_improved_all=n_improved_all,
                             rel_imp_median=float(np.median(rel_imps)) if rel_imps else 0.0,
                             stable=stable, consensus=consensus)
    add("\n## §8.11 验收（至少满足两条；操作化阈值为本实验约定）")
    add("1. 「较多类别稳定选择 K>1」→ 判据：consensus K>1 的类占比 ≥ 0.30")
    add("2. 「GMM 在 val 上 NLL 明显优于 single」→ 判据：3 seed 均改进的类占比 ≥ 0.60，"
        "且改进类中位相对提升 ≥ 0.10%")
    add("3. 「bootstrap 下 selected K 稳定」→ 判据：top prob ≥ 0.9 的类占比 ≥ 0.60")
    add("4. 「可视化出现与统计一致的多峰结构」→ 人工判定（见 figures/）")
    overall = {}
    for emb in EMBEDDINGS:
        v = verdicts[emb]
        v["crit"] = {
            "1_multimodal": bool(v["r_multi"] >= 0.30),
            "2_nll": bool(v["n_improved_all"] / v["C"] >= 0.60
                          and v["rel_imp_median"] >= 0.10),
            "3_bootstrap_stable": bool(v["stable"] >= 0.60),
            "4_visual": None}
        add(f"\n{emb}：crit1={v['crit']['1_multimodal']} crit2={v['crit']['2_nll']} "
            f"crit3={v['crit']['3_bootstrap_stable']} crit4=待看图")
    if not args.skip_figs:
        print("  生成代表性 UMAP/t-SNE 图（K=1/K=2/K≥3 各一，zf+zt）")
        made = _figure_representatives(out_dir, figures_dir, by_emb, names,
                                       args.fig_max_points)
        for emb in EMBEDDINGS:
            v = verdicts[emb]
            figs = [m for m in made if f"_{emb}_" in Path(m).stem]
            v["crit"]["4_visual"] = len(figs) > 0  # 占位；人工复核后以报告为准
        for line in [f"  {m}" for m in made]:
            print(line)
    for emb in EMBEDDINGS:
        v = verdicts[emb]
        crits = [c for k, c in v["crit"].items() if k != "4_visual" and c]
        add(f"\n{emb}：满足 {len(crits)}/4 条（第 4 条以人工看图为准）")
    # “几乎所有类 K=1” → 主线停止标志
    for emb in EMBEDDINGS:
        v = verdicts[emb]
        frac_k1 = sum(1 for k in v["consensus"] if k == 1) / v["C"]
        if frac_k1 >= 0.9:
            add(f"\n> ⚠ {emb}：{frac_k1:.0%} 的类 consensus K=1 → 按 §8.11，"
                f"若两条 embedding 皆如此则应停止 Multi-Prototype 主线，先重新检查 embedding。")
    add("\n## 产物")
    for name in ["bic_per_class.csv", "validation_nll.csv", "selected_k_per_seed.csv",
                 "bootstrap_k_stability.csv", "cluster_size_statistics.csv",
                 "distribution_audit_summary.md", "figures/"]:
        add(f"- {name}")
    (out_dir / "distribution_audit_summary.md").write_text(
        "\n".join(lines), encoding="utf-8")
    print(f"[4/4] 完成 → {out_dir}")


if __name__ == "__main__":
    main()
