"""Device-safe wrapper preserving released Open-Detect model semantics.

The encoder/decoder objects are instantiated directly from the read-only author
checkout.  Only device placement and exposure of ``mu``/``logvar`` are adapted
locally; the released loss equations and decoder graph are left unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_UPSTREAM_CODE = (
    Path(__file__).resolve().parents[2].parent / "Open-Detect" / "code"
)


def author_networks(
    upstream_code: Path, arch: str, channels: int, latent_dim: int
) -> tuple[nn.Module, nn.Module]:
    code_path = str(upstream_code.resolve())
    if code_path not in sys.path:
        sys.path.insert(0, code_path)
    from networks import net  # type: ignore

    return net(arch, channels, latent_dim)


class AuditedOpenDetectNet(nn.Module):
    def __init__(
        self,
        upstream_code: Path = DEFAULT_UPSTREAM_CODE,
        arch: str = "resnet18",
        channels: int = 1,
        latent_dim: int = 128,
        num_classes: int = 20,
        temp_inter: float = 1.0,
        temp_intra: float = 1.0,
    ) -> None:
        super().__init__()
        self.arch = arch
        self.channel = channels
        self.latent_dim = latent_dim
        self.n_classes = num_classes
        self.temp_inter = temp_inter
        self.temp_intra = temp_intra
        self.encoder, self.decoder = author_networks(
            upstream_code, arch, channels, latent_dim
        )
        self.prototypes = nn.Parameter(torch.randn(num_classes, latent_dim))
        nn.init.kaiming_normal_(self.prototypes)

    @staticmethod
    def sampler(mu: torch.Tensor, logvar: torch.Tensor, training: bool) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std) if training else mu

    @staticmethod
    def distance(latent_z: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
        matrix_a_square = torch.sum(latent_z**2, 1, keepdim=True)
        matrix_b_square = torch.sum(prototypes**2, 1).unsqueeze(0)
        product = torch.matmul(latent_z, prototypes.t())
        return matrix_a_square + matrix_b_square - 2 * product

    def kl_div_to_prototypes(
        self, mean: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        kl_div = self.distance(mean, self.prototypes) + torch.sum(
            logvar.exp() - logvar - 1, dim=1, keepdim=True
        )
        return 0.5 * kl_div

    def forward_details(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        mu, logvar, lateral_z = self.encoder(x)
        latent_z = self.sampler(mu, logvar, self.training)
        dist = self.distance(latent_z, self.prototypes)
        kl_div = self.kl_div_to_prototypes(mu, logvar)
        recon_x = self.decoder(latent_z, lateral_z)
        return {
            "mu": mu,
            "logvar": logvar,
            "latent_z": latent_z,
            "dist": dist,
            "kl_div": kl_div,
            "recon": recon_x,
        }

    def forward(self, x: torch.Tensor):
        details = self.forward_details(x)
        return (
            details["latent_z"],
            details["dist"],
            details["kl_div"],
            details["recon"],
        )

    def loss(self, x: torch.Tensor, y: torch.Tensor):
        details = self.forward_details(x)
        dist = details["dist"]
        kl_div = details["kl_div"]
        preds = torch.min(dist, dim=1).indices
        y_one_hot = F.one_hot(y, num_classes=self.n_classes).bool()
        dist_y = dist[y_one_hot].view(len(dist), 1)
        kl_div_y = kl_div[y_one_hot].view(len(kl_div), 1)
        q_w_z_y = F.softmax(-dist_y / self.temp_intra, dim=1)
        rec_loss = F.mse_loss(details["recon"], x)
        q_w_z_y = torch.clamp(q_w_z_y, min=1e-7)
        kld_loss = torch.mean(torch.sum(q_w_z_y * kl_div_y, dim=1))
        ent_loss = torch.mean(
            torch.sum(q_w_z_y * torch.log(q_w_z_y * self.n_classes), dim=1)
        )
        lse_all = torch.logsumexp(-dist / self.temp_inter, dim=1)
        lse_target = torch.logsumexp(-dist_y / self.temp_inter, dim=1)
        dis_loss = torch.mean(lse_all - lse_target)
        losses = {
            "dis": dis_loss,
            "rec": rec_loss,
            "kld": kld_loss,
            "ent": ent_loss,
        }
        return details, preds, losses


def released_weight_init(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)

