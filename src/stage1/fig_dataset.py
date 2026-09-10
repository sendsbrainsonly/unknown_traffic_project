# -*- coding: utf-8 -*-
"""Stage 1 Model B 图数据装载：fig_all.jsonl + fig_index.csv + splits。

按 flow_id 从 data/fig_graph/<policy> 随机访问抽取 FIG（Task 0.4 产物），
与 Model A 使用完全相同的 splits（data/splits/<name>/train.txt 等）。

批处理为 padding 图：
  x     B×Nmax×7   节点特征（已按训练集统计标准化）
  adj   B×Nmax×Nmax  归一化邻接（含自环；padding 节点全零、不参与 readout）
  mask  B×Nmax     padding mask
  y     B          label（split 内唯一类排序 → 连续 0..C-1）
  flow_id          原样保留（Model C 对齐与 §7.4 导出用）

特征标准化：用训练集的逐特征 mean/std（7 个特征统一处理），
常数与模型一起保存，验证/测试用同一组常数。选全局统计而非逐图统计：
单节点图 std=0，逐图标准化会退化；FEC-OSL 未公开代码，此为
实现选择，见 docs/MODEL_B_TAGCN.md。
"""
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class FeatureNormalizer:
    """训练集逐特征 mean/std（流式统计 jsonl，只取训练集 flow）。"""

    N_FEATURES = 7

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, root, split_name, fig_policy, train_ids, jsonl_path):
        train_set = set(train_ids)
        sums = np.zeros(self.N_FEATURES, dtype=np.float64)
        sumsq = np.zeros(self.N_FEATURES, dtype=np.float64)
        n = 0
        with jsonl_path.open("rb") as fh:
            for line in fh:
                rec = json.loads(line)
                if rec["flow_id"] in train_set:
                    x = np.asarray(rec["features"], dtype=np.float64)
                    sums += x.sum(axis=0)
                    sumsq += (x * x).sum(axis=0)
                    n += x.shape[0]
        if n == 0:
            raise ValueError("normalizer: 训练集为空")
        self.mean = sums / n
        var = np.maximum(sumsq / n - self.mean ** 2, 0.0)
        self.std = np.sqrt(var)
        self.std[self.std < 1e-8] = 1.0  # 常数特征（如方向±1 若恒同）不缩放

    def apply(self, x):
        """x: N×7 float64/float32 → (x-mean)/std 返回 float32。"""
        if self.mean is None:
            raise RuntimeError("normalizer 未 fit")
        return ((np.asarray(x, dtype=np.float64) - self.mean) / self.std).astype(np.float32)

    def to_dict(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d):
        obj = cls()
        obj.mean = np.asarray(d["mean"], dtype=np.float64)
        obj.std = np.asarray(d["std"], dtype=np.float64)
        return obj


def _normalized_adjacency(edges, n_nodes):
    """Â = D^-1/2 (A+I) D^-1/2，n×n float32；padding 图由 collate 扩零。"""
    adj = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    adj[np.arange(n_nodes), np.arange(n_nodes)] = 1.0
    for u, v in edges:
        adj[u, v] = 1.0
        adj[v, u] = 1.0
    deg = adj.sum(axis=1)
    adj = adj / np.sqrt(deg[:, None] * deg[None, :])
    return adj


def collate_graphs(batch):
    """变长图 → padding 批。batch 元素 = (x, adj, y, flow_id)。"""
    xs, adjs, ys, fids = zip(*batch)
    n_max = max(x.shape[0] for x in xs)
    b = len(batch)
    x_b = torch.zeros(b, n_max, FeatureNormalizer.N_FEATURES)
    adj_b = torch.zeros(b, n_max, n_max)
    mask = torch.zeros(b, n_max, dtype=torch.bool)
    for i, (x, adj) in enumerate(zip(xs, adjs)):
        n = x.shape[0]
        x_b[i, :n] = torch.as_tensor(x, dtype=torch.float32)
        adj_b[i, :n, :n] = torch.as_tensor(adj, dtype=torch.float32)
        mask[i, :n] = True
    return x_b, adj_b, mask, torch.stack(ys), fids


class FIGSplitDataset(Dataset):
    """单个 split 的 FIG 数据集。

    flow_id 顺序 = split txt 文件顺序（与 Model A 的 TSV 同源划分）。
    """

    def __init__(self, root, split_name, part, fig_policy, flow_ids, labels,
                 index, jsonl_path, normalizer):
        # flow_ids/labels/index 由 load_fig_splits 预计算
        self._flow_ids = flow_ids
        self._labels = labels
        self._index = index          # flow_id -> (byte_offset, label_id, n_nodes, n_edges)
        self._jsonl_path = jsonl_path
        self._jsonl = jsonl_path.open("rb")
        self._normalizer = normalizer
        self._cache = None           # 显式 build_cache 后启用

    def build_cache(self):
        """全量读入内存缓存（特征+归一化邻接，约 600B/流 × 49 万 ≈ 300MB）。"""
        if self._cache is not None:
            return
        cache = []
        for i in range(len(self._flow_ids)):
            cache.append(self._load_one(self._flow_ids[i]))
        self._jsonl.close()
        self._cache = cache

    def _load_one(self, fid):
        offset, _, n_nodes, _ = self._index[fid]
        self._jsonl.seek(offset)
        rec = json.loads(self._jsonl.readline())
        x = np.asarray(rec["features"], dtype=np.float32)
        if self._normalizer is not None:
            x = self._normalizer.apply(x)
        adj = _normalized_adjacency(rec["edges"], x.shape[0])
        assert x.shape[0] == n_nodes, f"{fid}: 节点数不符"
        return x, adj

    def __len__(self):
        return len(self._flow_ids)

    def __getitem__(self, i):
        fid = self._flow_ids[i]
        if self._cache is not None:
            x, adj = self._cache[i]
        else:
            x, adj = self._load_one(fid)
        return x, torch.from_numpy(adj), torch.tensor(self._labels[i], dtype=torch.long), fid


def load_fig_splits(root, split_name, fig_policy="all_flows", normalize=True,
                    build_cache=True):
    """装载 train/val/test 三个 FIGSplitDataset 及 labels_num。

    返回 (datasets, labels_num, normalizer)，datasets = {"train":..., "val":..., "test":...}。
    label 映射：split 内唯一 label_id 排序 → 连续 0..C-1（全量 20 类本为 0-19；
    冒烟子集自动重映射）。若 normalize=False 则不做特征标准化。
    """
    root = Path(root)
    fig_dir = root / "data" / "fig_graph" / fig_policy
    split_dir = root / "data" / "splits" / split_name
    jsonl_path = fig_dir / "fig_all.jsonl"

    # 1) fig_index → flow_id 索引（对齐校验一并做）
    index = {}
    with (fig_dir / "fig_index.csv").open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            index[r["flow_id"]] = (int(r["byte_offset"]), int(r["label_id"]),
                                   int(r["node_count"]), int(r["edge_count"]))

    parts = {}
    for part, fname in (("train", "train.txt"), ("val", "val.txt"), ("test", "test.txt")):
        ids = [ln.strip() for ln in (split_dir / fname).read_text(encoding="utf-8").splitlines()
               if ln.strip()]
        missing = [fid for fid in ids if fid not in index]
        if missing:
            raise ValueError(f"split {split_name}/{fname}: {len(missing)} 条 flow_id "
                             f"不在 fig_index 中（首个: {missing[0]}）——FIG 与 splits 未对齐")
        parts[part] = ids

    # 2) split 内唯一类排序 → 连续 id
    all_ids = parts["train"] + parts["val"] + parts["test"]
    labels_raw = [index[fid][1] for fid in all_ids]
    uniq = sorted(set(labels_raw))
    remap = {lab: i for i, lab in enumerate(uniq)}
    labels_num = len(uniq)
    offsets = {"train": (0, len(parts["train"])),
               "val": (len(parts["train"]), len(parts["train"]) + len(parts["val"])),
               "test": (len(parts["train"]) + len(parts["val"]), len(all_ids))}

    # 3) 特征标准化（仅训练集统计）
    normalizer = None
    if normalize:
        normalizer = FeatureNormalizer()
        normalizer.fit(root, split_name, fig_policy, parts["train"], jsonl_path)

    # 4) 组装数据集
    datasets = {}
    for part in parts:
        lo, hi = offsets[part]
        datasets[part] = FIGSplitDataset(
            root, split_name, part, fig_policy, parts[part],
            [remap[lab] for lab in labels_raw[lo:hi]], index, jsonl_path, normalizer)
    if build_cache:
        for ds in datasets.values():
            ds.build_cache()
    return datasets, labels_num, normalizer
