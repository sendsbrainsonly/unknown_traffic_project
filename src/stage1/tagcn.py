# -*- coding: utf-8 -*-
"""Stage 1 Model B：TAGCN（K 跳多项式滤波器图卷积）+ readout + 分类头。

结构（对应 v2 计划 §7.1 Model B）：
    FIG → TAGCN → z_g → Classifier

TAGCN 依据 FEC-OSL 原文（Yang et al., IEEE TIFS 2026）公式 5-6 的
K 跳多项式滤波器实现：

    H = Σ_{k=0}^{K} (Â^k X) W_k + b,  Â = D^-1/2 (A+I) D^-1/2

readout = 真实节点的 masked mean pooling → z_g（FEC-OSL 原文 "readout→hg"
未指定池化形式；取 mean，见 docs/MODEL_B_TAGCN.md 的已知偏差节）。

forward 返回 (logits, z_g)；z_g 用于 §7.4 embedding 导出与 Model C 融合。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TAGCN(nn.Module):
    def __init__(self, in_dim=7, hidden=128, labels_num=20, k_hops=2,
                 dropout=0.5):
        super().__init__()
        self.in_dim = in_dim
        self.hidden = hidden
        self.labels_num = labels_num
        self.k_hops = k_hops
        self.dropout = dropout

        # 每个 hop 一个线性投影；bias 只加一次（在求和后）
        self.hop_lins = nn.ModuleList([
            nn.Linear(in_dim, hidden, bias=False) for _ in range(k_hops + 1)])
        self.hop_bias = nn.Parameter(torch.zeros(hidden))
        self.classifier = nn.Linear(hidden, labels_num)

    def forward(self, x, adj, mask):
        """x: B×N×in_dim；adj: B×N×N 归一化邻接（含自环，padding 全零）；
        mask: B×N bool。返回 (logits B×labels_num, z_g B×hidden)。"""
        agg = x
        h = self.hop_lins[0](agg)
        for k in range(1, self.k_hops + 1):
            agg = torch.bmm(adj, agg)
            h = h + self.hop_lins[k](agg)
        h = h + self.hop_bias
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # masked mean readout → z_g
        mask_f = mask.to(h.dtype).unsqueeze(-1)          # B×N×1
        z_g = (h * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
        logits = self.classifier(z_g)
        return logits, z_g


def build_model(labels_num, **kw):
    """统一入口：根据 labels_num 构造 TAGCN（其余超参透传）。"""
    return TAGCN(labels_num=labels_num, **kw)
