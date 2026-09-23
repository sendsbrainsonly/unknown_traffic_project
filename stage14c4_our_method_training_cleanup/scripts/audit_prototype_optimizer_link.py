#!/usr/bin/env python3
"""Demonstrate optimizer ownership before/after Parameter replacement vs in-place reset."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.optim as optim


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT / "opendetect_ustc_encoder_audit"))
sys.path.insert(0, str(PROJECT / "stage3_unknown_utility" / "scripts"))
from stage3_model import Stage3OpenDetectNet  # noqa: E402


def linked(optimizer: optim.Optimizer, parameter: torch.nn.Parameter) -> bool:
    return any(item is parameter for group in optimizer.param_groups for item in group["params"])


def main() -> None:
    upstream = PROJECT.parent / "Open-Detect" / "code"
    model = Stage3OpenDetectNet(
        upstream_code=upstream,
        channels=1,
        latent_dim=128,
        num_classes=7,
        temp_inter=1.0,
        temp_intra=1.0,
    )
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    original = model.prototypes
    before = linked(optimizer, original)

    replacement = torch.nn.Parameter(torch.zeros_like(original), requires_grad=True)
    model.prototypes = replacement
    replacement_linked = linked(optimizer, model.prototypes)
    original_still_linked = linked(optimizer, original)

    model.prototypes = original
    before_id = id(model.prototypes)
    with torch.no_grad():
        model.prototypes.copy_(torch.ones_like(model.prototypes))
    in_place_linked = linked(optimizer, model.prototypes)
    id_preserved = before_id == id(model.prototypes)

    result = {
        "status": "PASS" if before and not replacement_linked and original_still_linked and in_place_linked and id_preserved else "FAIL",
        "optimizer_links_original_before_reset": before,
        "parameter_replacement_links_new_parameter": replacement_linked,
        "parameter_replacement_leaves_old_parameter_in_optimizer": original_still_linked,
        "in_place_reset_preserves_parameter_id": id_preserved,
        "in_place_reset_preserves_optimizer_link": in_place_linked,
        "conclusion": "Parameter replacement disconnects the active model prototype from Adam; in-place copy preserves optimizer ownership.",
    }
    (ROOT / "prototype_optimizer_link_audit.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
