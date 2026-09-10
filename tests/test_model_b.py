# -*- coding: utf-8 -*-
"""Model B（FIG→TAGCN）单元测试：批处理/模型数学/集成冒烟。"""
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.stage1.fig_dataset import (_normalized_adjacency, collate_graphs,
                                    load_fig_splits)
from src.stage1.tagcn import TAGCN

D_ROOT = Path("D:/unknown_traffic_project")


def make_item(n, fill=1.0):
    """合成 n 节点图：特征全 fill、链式边 0-1-2-...。"""
    x = torch.full((n, 7), fill)
    edges = [[i, i + 1] for i in range(n - 1)]
    adj = torch.from_numpy(_normalized_adjacency(edges, n))
    return x, adj, torch.tensor(0), f"synth_{n}"


class TestCollate(unittest.TestCase):
    def test_padding_and_mask(self):
        batch = [make_item(1), make_item(2), make_item(3)]
        x, adj, mask, y, fids = collate_graphs(batch)
        self.assertEqual(x.shape, (3, 3, 7))
        self.assertEqual(adj.shape, (3, 3, 3))
        self.assertTrue(torch.equal(mask,
                                    torch.tensor([[1, 0, 0], [1, 1, 0],
                                                  [1, 1, 1]])))
        self.assertEqual(fids, ("synth_1", "synth_2", "synth_3"))
        # padding 区域全零（无自环 → 不参与传播/readout）
        self.assertTrue(torch.all(adj[0, :, 1:] == 0))
        self.assertTrue(torch.all(adj[1, :, 2] == 0))
        # 真实节点有自环
        self.assertGreater(adj[0, 0, 0], 0)

    def test_normalized_adjacency_degree_property(self):
        adj = _normalized_adjacency([[0, 1], [1, 2]], 3)
        d = np.array([2.0, 3.0, 2.0])  # A+I 的度（链 0-1-2 加自环）
        # Â·√d = √d（对称归一化的不动点性质）
        np.testing.assert_allclose(adj @ np.sqrt(d), np.sqrt(d), atol=1e-6)


class TestTAGCN(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = TAGCN(in_dim=7, hidden=16, labels_num=5, k_hops=2)
        self.model.eval()

    def test_forward_shapes(self):
        batch = [make_item(1), make_item(2), make_item(3), make_item(4)]
        x, adj, mask, _, _ = collate_graphs(batch)
        logits, z_g = self.model(x, adj, mask)
        self.assertEqual(logits.shape, (4, 5))
        self.assertEqual(z_g.shape, (4, 16))

    def test_padded_nodes_isolated(self):
        # padding 节点零特征零边 → k 跳后仍为零，不泄漏进 readout
        batch = [make_item(2), make_item(1)]
        x, adj, mask, _, _ = collate_graphs(batch)
        with torch.no_grad():
            logits, z_g = self.model(x, adj, mask)
        # 单节点图（第 1 个样本）的 z_g 只取决于自身节点
        single = collate_graphs([make_item(1)])
        with torch.no_grad():
            _, z_single = self.model(*single[:3])
        torch.testing.assert_close(z_g[1], z_single[0], rtol=1e-5, atol=1e-6)

    def test_single_graph_batched_consistency(self):
        # 同一张图单独过模型 vs 混在批里过模型，z_g 一致
        item = make_item(4, fill=2.0)
        x, adj, mask, _, _ = collate_graphs([item])
        with torch.no_grad():
            logits_solo, _ = self.model(x, adj, mask)
        x2, adj2, mask2, _, _ = collate_graphs([item, make_item(3), item])
        with torch.no_grad():
            logits_batch, z_batch = self.model(x2, adj2, mask2)
        torch.testing.assert_close(logits_batch[0], logits_solo[0],
                                   rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(logits_batch[2], logits_solo[0],
                                   rtol=1e-5, atol=1e-6)


@unittest.skipUnless(D_ROOT.exists(), "D: 数据目录不存在")
class TestRealDataSmoke(unittest.TestCase):
    def test_smoke_split_loads_and_aligns(self):
        datasets, labels_num, normalizer = load_fig_splits(
            D_ROOT, "smoke_gmail_zeus", "all_flows", normalize=True)
        self.assertEqual(labels_num, 2)
        self.assertEqual(len(datasets["train"]), 320)
        self.assertEqual(len(datasets["val"]), 40)
        self.assertEqual(len(datasets["test"]), 40)
        self.assertEqual(normalizer.mean.shape, (7,))
        # 标签连续重映射：Gmail=5, Zeus=19 → {0,1}
        self.assertEqual(set(datasets["train"]._labels), {0, 1})

    def test_smoke_one_epoch_converges(self):
        torch.manual_seed(7)
        datasets, labels_num, _ = load_fig_splits(
            D_ROOT, "smoke_gmail_zeus", "all_flows", normalize=True)
        from torch.utils.data import DataLoader
        loader = DataLoader(datasets["train"], batch_size=32, shuffle=True,
                            collate_fn=collate_graphs, num_workers=0)
        model = TAGCN(in_dim=7, hidden=32, labels_num=labels_num, k_hops=2)
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        criterion = torch.nn.CrossEntropyLoss()
        losses = []
        for _ in range(2):  # 320 流两个 epoch，lr 0.1 应明显下降
            for x, adj, mask, y, _ in loader:
                opt.zero_grad()
                logits, _ = model(x, adj, mask)
                loss = criterion(logits, y)
                loss.backward()
                opt.step()
                losses.append(loss.item())
        self.assertTrue(np.isfinite(losses).all())
        self.assertLess(np.mean(losses[-5:]), np.mean(losses[:5]))


if __name__ == "__main__":
    unittest.main()
