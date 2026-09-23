#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import sys

import numpy as np
import torch
import torch.nn as nn

from stage21_common import OUT, PROJECT, cache_path

sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "tf_runtime" / "code"))
sys.path.insert(0, str(PROJECT / "stage17_encoder_recovery_and_open_set_pilot" / "scripts"))
from src.stage1.tagcn import TAGCN
from train_pilot_run import load_tf_module, make_tf_args, normalize_adjacency, tf_forward


def main() -> None:
    data = np.load(cache_path("iscx_vpn"), allow_pickle=False)
    device = torch.device("cuda:0")
    module = load_tf_module()
    args = make_tf_args(6, 1)
    tf = module.Classifier(args).to(device)
    src = torch.from_numpy(data["token_ids"][:4]).long().to(device)
    seg = torch.from_numpy(data["segments"][:4]).long().to(device)
    labels = torch.tensor([0, 1, 2, 3], device=device)
    logits, z_tf = tf_forward(tf, src, seg)
    nn.CrossEntropyLoss()(logits, labels).backward()
    graph = TAGCN(in_dim=7, hidden=128, labels_num=6, k_hops=2, dropout=.5).to(device)
    adj = normalize_adjacency(data["fig_adj"][:4], data["fig_mask"][:4])
    g_logits, z_graph = graph(
        torch.from_numpy(data["fig_x"][:4]).float().to(device),
        torch.from_numpy(adj).float().to(device),
        torch.from_numpy(data["fig_mask"][:4]).to(device),
    )
    nn.CrossEntropyLoss()(g_logits, labels).backward()
    fusion = nn.Linear(z_tf.shape[1] + z_graph.shape[1], 6).to(device)
    fused = fusion(torch.cat([z_tf.detach(), z_graph.detach()], dim=1))
    nn.CrossEntropyLoss()(fused, labels).backward()
    result = {
        "status": "PASS",
        "trafficformer_shape": list(z_tf.shape),
        "tagcn_shape": list(z_graph.shape),
        "fusion_shape": list(fused.shape),
        "finite": bool(torch.isfinite(fused).all()),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
    }
    (OUT / "smoke_result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
