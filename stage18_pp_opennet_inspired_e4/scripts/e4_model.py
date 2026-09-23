#!/usr/bin/env python3
from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.unsqueeze(-1).to(x.dtype)
    return (x * weights).sum(1) / weights.sum(1).clamp_min(1.0)


def masked_max(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    fill = torch.finfo(x.dtype).min
    value = x.masked_fill(~mask.unsqueeze(-1), fill).max(1).values
    return torch.where(torch.isfinite(value), value, torch.zeros_like(value))


def masked_stats(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mean = masked_mean(x, mask)
    weights = mask.unsqueeze(-1).to(x.dtype)
    var = ((x - mean.unsqueeze(1)).square() * weights).sum(1) / weights.sum(1).clamp_min(1.0)
    maximum = masked_max(x, mask)
    return torch.cat([mean, torch.sqrt(var + 1e-6), maximum], dim=1)


class E4Encoder(nn.Module):
    """Independent PP-OpenNet-inspired encoder; no Open-Detect components."""

    def __init__(
        self,
        input_dim: int,
        labels_num: int,
        embedding_dim: int = 128,
        multi_scale: bool = True,
        recurrent: bool = True,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.multi_scale = multi_scale
        self.recurrent = recurrent
        if multi_scale:
            branch_dim = embedding_dim // 4
            self.branches = nn.ModuleList([
                nn.Sequential(nn.Conv1d(input_dim, branch_dim, kernel_size=k, padding=k // 2), nn.GELU())
                for k in (1, 3, 5)
            ])
            self.pool_branch = nn.Sequential(
                nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
                nn.Conv1d(input_dim, branch_dim, kernel_size=1),
                nn.GELU(),
            )
            self.local_projection = nn.Sequential(nn.Conv1d(branch_dim * 4, embedding_dim, 1), nn.GELU())
        else:
            self.single_conv = nn.Sequential(
                nn.Conv1d(input_dim, embedding_dim, kernel_size=3, padding=1),
                nn.GELU(),
            )
        self.stat_encoder = nn.Sequential(
            nn.Linear(input_dim * 3, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        if recurrent:
            self.packet_gru = nn.GRU(embedding_dim, embedding_dim, batch_first=True)
            self.fusion_gru = nn.GRU(embedding_dim, embedding_dim, batch_first=True)
        else:
            self.nonrecurrent_fusion = nn.Sequential(
                nn.Linear(embedding_dim * 3, embedding_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        self.output_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embedding_dim, labels_num)

    def local_tokens(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Zero invalid positions before convolution so arbitrary padding values
        # cannot change valid packet tokens near the sequence boundary.
        channels = (x * mask.unsqueeze(-1).to(x.dtype)).transpose(1, 2)
        if self.multi_scale:
            pieces = [branch(channels) for branch in self.branches]
            pieces.append(self.pool_branch(channels))
            tokens = self.local_projection(torch.cat(pieces, dim=1)).transpose(1, 2)
        else:
            tokens = self.single_conv(channels).transpose(1, 2)
        return tokens * mask.unsqueeze(-1).to(tokens.dtype)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3 or mask.ndim != 2 or x.shape[:2] != mask.shape:
            raise ValueError("expected x [B,L,C] and mask [B,L]")
        lengths = mask.sum(1).clamp_min(1).to(torch.long)
        local = self.local_tokens(x, mask)
        local_pool = masked_mean(local, mask)
        stat = self.stat_encoder(masked_stats(x, mask))
        if self.recurrent:
            packed = pack_padded_sequence(local, lengths.cpu(), batch_first=True, enforce_sorted=False)
            _, hidden = self.packet_gru(packed)
            global_state = hidden[-1]
            branch_sequence = torch.stack([stat, local_pool, global_state], dim=1)
            _, fused = self.fusion_gru(branch_sequence)
            embedding = fused[-1]
        else:
            global_pool = masked_max(local, mask)
            embedding = self.nonrecurrent_fusion(torch.cat([stat, local_pool, global_pool], dim=1))
        embedding = self.output_norm(embedding)
        logits = self.classifier(self.dropout(embedding))
        return logits, embedding
