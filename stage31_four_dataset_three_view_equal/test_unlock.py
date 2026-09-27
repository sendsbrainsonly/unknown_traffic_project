"""Hard gate: no Stage31 Known Test features before all five T0 heads are frozen."""
from __future__ import annotations

import json

from preflight import CELLS, OUT, sha


def ensure_all_heads_frozen() -> None:
    for dataset, protocol in CELLS:
        run = OUT / "runs" / dataset / protocol / "T0_equal"
        if not (run / "SUCCESS").is_file():
            raise RuntimeError(f"Known Test locked: T0 incomplete for {dataset}/{protocol}")
        result = json.loads((run / "known_validation_metrics.json").read_text())
        for name, expected in result["checkpoint_hashes"].items():
            if sha(run / name) != expected:
                raise RuntimeError(f"Known Test locked: checkpoint hash mismatch {dataset}/{protocol}/{name}")
