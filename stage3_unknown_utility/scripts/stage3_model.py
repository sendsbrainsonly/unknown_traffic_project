#!/usr/bin/env python3
"""Stage 3 Open-Detect adapter with an upper-logvar stability guard.

The released path can have a finite forward loss while gradients already
overflow because ``exp(logvar)`` is close to the float32 limit.  The previously
audited corrected Open-Detect implementation bounds log-variance at 20.  Stage
3 applies only that upper bound, leaving negative values unchanged, and records
every batch in which the guard is active.
"""

from __future__ import annotations

import torch

from adapters.opendetect_model import AuditedOpenDetectNet


class Stage3OpenDetectNet(AuditedOpenDetectNet):
    max_logvar = 20.0

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.stability_guard_activations = 0

    @classmethod
    def bound_logvar(cls, raw_logvar: torch.Tensor) -> torch.Tensor:
        """Bound only the overflow-prone upper tail of log-variance."""
        return torch.clamp(raw_logvar, max=cls.max_logvar)

    def stable_encode(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        mean, raw_logvar, lateral_z = self.encoder(x)
        if bool((raw_logvar > self.max_logvar).any().detach().cpu()):
            self.stability_guard_activations += 1
            logvar = self.bound_logvar(raw_logvar)
        else:
            logvar = raw_logvar
        return mean, logvar, lateral_z

    def forward_details(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        mu, logvar, lateral_z = self.stable_encode(x)
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
