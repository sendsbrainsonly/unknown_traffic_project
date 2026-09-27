#!/usr/bin/env python3
"""Preregistered Stage 24 early-eight-packet behavior encoders."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence


@dataclass(frozen=True)
class Features:
    nodes: np.ndarray
    mask: np.ndarray
    stats: np.ndarray
    burst_ids: np.ndarray
    burst_count: np.ndarray
    burst_order: np.ndarray


def _signed_log(value: float) -> float:
    return float(np.sign(value) * np.log1p(abs(value)))


def make_features(graph_x: np.ndarray, mask: np.ndarray, flow_ids: np.ndarray,
                  seed: int, shuffle_bursts: bool = False) -> Features:
    if graph_x.ndim != 3 or graph_x.shape[1:] != (8, 7):
        raise ValueError(f"expected N×8×7 FIG input, got {graph_x.shape}")
    if mask.shape != graph_x.shape[:2] or len(flow_ids) != len(graph_x):
        raise ValueError("feature/mask/flow-ID shape mismatch")
    n_flows = len(graph_x)
    nodes = np.zeros((n_flows, 8, 9), dtype=np.float32)
    stats = np.zeros((n_flows, 19), dtype=np.float32)
    burst_ids = np.zeros((n_flows, 8), dtype=np.int64)
    burst_count = np.zeros(n_flows, dtype=np.int64)
    burst_order = np.tile(np.arange(8, dtype=np.int64), (n_flows, 1))

    for i in range(n_flows):
        n = int(mask[i].sum())
        if n < 1 or not np.all(mask[i, :n]) or np.any(mask[i, n:]):
            raise ValueError(f"invalid contiguous packet mask: {flow_ids[i]}")
        x = np.asarray(graph_x[i, :n], dtype=np.float64)
        if not np.isfinite(x).all():
            raise ValueError(f"non-finite FIG values: {flow_ids[i]}")
        directions, lengths, times = x[:, 0], x[:, 1], x[:, 2]
        if np.any(lengths < 0) or np.any(np.diff(times) < -1e-8):
            raise ValueError(f"invalid packet length/timestamp: {flow_ids[i]}")
        iats = np.r_[0.0, np.maximum(0.0, np.diff(times))]
        length_delta = np.r_[0.0, np.diff(lengths)]
        nodes[i, :n, :7] = x.astype(np.float32)
        nodes[i, :n, 7] = np.log1p(iats).astype(np.float32)
        nodes[i, :n, 8] = np.asarray([_signed_log(v) for v in length_delta], dtype=np.float32)

        bid = 0
        burst_sizes: list[float] = []
        burst_bytes: list[float] = []
        b_size = 0.0
        b_bytes = 0.0
        for j in range(n):
            if j and directions[j] != directions[j - 1]:
                burst_sizes.append(b_size)
                burst_bytes.append(b_bytes)
                bid += 1
                b_size = 0.0
                b_bytes = 0.0
            burst_ids[i, j] = bid
            b_size += 1
            b_bytes += lengths[j]
        burst_sizes.append(b_size)
        burst_bytes.append(b_bytes)
        burst_count[i] = bid + 1
        if shuffle_bursts and burst_count[i] > 1:
            digest = hashlib.sha256(f"{seed}|{flow_ids[i]}".encode()).digest()
            rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
            burst_order[i, :bid + 1] = rng.permutation(bid + 1)

        up = directions > 0
        down = ~up
        total = float(lengths.sum())
        positive_iats = iats[1:]
        stats[i] = np.asarray([
            float(n), np.log1p(total), np.log1p(float(times[-1] - times[0])),
            float(np.log1p(lengths).mean()), float(np.log1p(lengths).std()),
            np.log1p(float(lengths.max())), np.log1p(float(lengths.min())),
            float(np.log1p(positive_iats).mean()) if n > 1 else 0.0,
            float(np.log1p(positive_iats).std()) if n > 1 else 0.0,
            float(np.log1p(positive_iats).max()) if n > 1 else 0.0,
            float(up.mean()), float(lengths[up].sum() / (total + 1e-8)),
            float(down.mean()), float(lengths[down].sum() / (total + 1e-8)),
            float(bid + 1), float(np.mean(burst_sizes)), float(max(burst_sizes)),
            np.log1p(float(np.mean(burst_bytes))), np.log1p(float(max(burst_bytes))),
        ], dtype=np.float32)
    return Features(nodes=nodes, mask=mask.astype(bool, copy=True), stats=stats,
                    burst_ids=burst_ids, burst_count=burst_count, burst_order=burst_order)


class StatsMLP(nn.Module):
    def __init__(self, labels_num: int):
        super().__init__()
        self.proj = nn.Linear(19, 128)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(128, labels_num)

    def forward(self, batch: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.dropout(F.relu(self.proj(batch[3])))
        return self.classifier(z), z


class FigTagcn(nn.Module):
    def __init__(self, labels_num: int):
        super().__init__()
        from src.stage1.tagcn import TAGCN
        self.model = TAGCN(in_dim=9, hidden=128, labels_num=labels_num, k_hops=2, dropout=0.5)

    def forward(self, batch: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(batch[0], batch[1], batch[2])


class BurstHierarchy(nn.Module):
    """Packet projection -> same-direction burst means -> ordered burst GRU."""

    def __init__(self, labels_num: int):
        super().__init__()
        self.packet_proj = nn.Linear(9, 32)
        self.burst_gru = nn.GRU(32, 32, batch_first=True)
        self.readout = nn.Linear(32, 128)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(128, labels_num)

    def forward(self, batch: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, torch.Tensor]:
        nodes, _adj, mask, _stats, burst_ids, burst_count, burst_order = batch[:7]
        packet = F.relu(self.packet_proj(nodes)) * mask.unsqueeze(-1)
        burst = packet.new_zeros((len(packet), 8, 32))
        burst.scatter_add_(1, burst_ids.unsqueeze(-1).expand_as(packet), packet)
        counts = packet.new_zeros((len(packet), 8, 1))
        counts.scatter_add_(1, burst_ids.unsqueeze(-1), mask.unsqueeze(-1).to(packet.dtype))
        burst = burst / counts.clamp(min=1.0)
        burst = torch.gather(burst, 1, burst_order.unsqueeze(-1).expand(-1, -1, 32))
        packed = pack_padded_sequence(burst, burst_count.cpu(), batch_first=True,
                                      enforce_sorted=False)
        _output, state = self.burst_gru(packed)
        z = self.dropout(F.relu(self.readout(state[-1])))
        return self.classifier(z), z


def make_model(method: str, labels_num: int) -> nn.Module:
    if method == "S1":
        return StatsMLP(labels_num)
    if method == "G1":
        return FigTagcn(labels_num)
    if method in ("G2", "G2-shuffle"):
        return BurstHierarchy(labels_num)
    raise ValueError(f"unknown Stage24 candidate: {method}")
