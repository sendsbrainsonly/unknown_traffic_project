#!/usr/bin/env python3
"""Check Stage31 T0 classes against immutable Stage30 T0 on identical tensors."""
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train_equal_fusion as current

stored_preflight = sys.modules.pop("preflight")
sys.path.insert(0, str(HERE.parent / "stage30_three_view_entropy_fusion"))
try:
    previous = importlib.import_module("run_one")
finally:
    sys.modules["preflight"] = stored_preflight

torch.manual_seed(2022)
old_adapter = previous.Adapters(7)
torch.manual_seed(2022)
new_adapter = current.Adapters(7)
adapter_keys_equal = list(old_adapter.state_dict()) == list(new_adapter.state_dict())
adapter_values_equal = all(torch.equal(a, b) for a, b in
    zip(old_adapter.state_dict().values(), new_adapter.state_dict().values()))
xs = [torch.randn(4, dim) for dim in current.DIMS]
y = torch.tensor([0, 1, 2, 3])
torch.manual_seed(123)
old_loss = old_adapter.training_loss(xs, y)[0]
torch.manual_seed(123)
new_loss = new_adapter.training_loss(xs, y)[0]
center = np.array([1., 2., 3.], dtype=np.float32)
mad = np.array([.1, .2, .3], dtype=np.float32)
torch.manual_seed(2022)
old_head = previous.Fusion(7, False, center, mad)
torch.manual_seed(2022)
new_head = current.EqualFusion(7, center, mad)
head_keys_equal = list(old_head.state_dict()) == list(new_head.state_dict())
head_values_equal = all(torch.equal(a, b) for a, b in
    zip(old_head.state_dict().values(), new_head.state_dict().values()))
old_head.eval(); new_head.eval()
mu = torch.randn(4, 3, 64)
entropy = torch.randn(4, 3)
old_logits, old_weight = old_head(mu, entropy)
new_logits, new_weight = new_head(mu, entropy)
report = {
    "status": "PASS" if all((adapter_keys_equal, adapter_values_equal,
                            head_keys_equal, head_values_equal,
                            torch.equal(old_loss, new_loss),
                            torch.equal(old_logits, new_logits),
                            torch.equal(old_weight, new_weight))) else "FAIL",
    "adapter_state_keys_equal": adapter_keys_equal,
    "adapter_initialized_state_equal": adapter_values_equal,
    "adapter_loss_absolute_difference": float(abs(old_loss-new_loss)),
    "fusion_state_keys_equal": head_keys_equal,
    "fusion_initialized_state_equal": head_values_equal,
    "fusion_max_logit_absolute_difference": float((old_logits-new_logits).abs().max()),
    "fusion_max_weight_absolute_difference": float((old_weight-new_weight).abs().max()),
}
(HERE / "architecture_parity.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report))
if report["status"] != "PASS":
    raise SystemExit(1)
