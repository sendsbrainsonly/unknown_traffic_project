from __future__ import annotations

import torch
import torch.nn as nn


class MaskedBatchNorm1d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-5, momentum: float = 0.1) -> None:
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.register_buffer("running_mean", torch.zeros(channels))
        self.register_buffer("running_var", torch.ones(channels))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.to(dtype=x.dtype)
        if self.training:
            count = mask.sum().clamp_min(1.0)
            mean = (x * mask).sum(dim=(0, 2)) / count
            var = ((x - mean[None, :, None]).square() * mask).sum(dim=(0, 2)) / count
            with torch.no_grad():
                self.running_mean.mul_(1 - self.momentum).add_(self.momentum * mean)
                self.running_var.mul_(1 - self.momentum).add_(self.momentum * var)
        else:
            mean, var = self.running_mean, self.running_var
        out = (x - mean[None, :, None]) / torch.sqrt(var[None, :, None] + self.eps)
        out = out * self.weight[None, :, None] + self.bias[None, :, None]
        return out * mask


class MaskSafeSequenceCNN(nn.Module):
    def __init__(self, n_classes: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(3, 64, 3, padding=1)
        self.bn1 = MaskedBatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, 3, padding=1)
        self.bn2 = MaskedBatchNorm1d(128)
        self.relu = nn.ReLU()
        self.head = nn.Linear(128, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = x[:, 2:3, :]
        clean = torch.cat((x[:, :2, :] * mask, mask), dim=1)
        hidden = self.relu(self.bn1(self.conv1(clean), mask)) * mask
        hidden = self.relu(self.bn2(self.conv2(hidden), mask)) * mask
        pooled = hidden.sum(dim=2) / mask.sum(dim=2).clamp_min(1.0)
        return self.head(pooled)


class LegacySequenceCNN(nn.Module):
    def __init__(self, n_classes: int) -> None:
        super().__init__()
        self.block1 = nn.Sequential(nn.Conv1d(3, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU())
        self.block2 = nn.Sequential(nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU())
        self.head = nn.Linear(128, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = x[:, 2:3, :]
        hidden = self.block1(x) * mask
        hidden = self.block2(hidden) * mask
        pooled = hidden.sum(dim=2) / mask.sum(dim=2).clamp_min(1.0)
        return self.head(pooled)
