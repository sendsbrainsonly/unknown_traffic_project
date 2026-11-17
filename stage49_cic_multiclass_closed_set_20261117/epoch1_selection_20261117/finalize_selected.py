#!/usr/bin/env python3
"""Reuse the independent finalizer with the explicit user budget amendment."""
import ast
import json
import sys
import types
from pathlib import Path

ATTEMPT = Path(__file__).resolve().parent
ROOT = ATTEMPT.parent
sys.path.insert(0, str(ROOT))
import protocol


def main():
    selection = json.loads((ATTEMPT / "epoch1_replay_verification.json").read_text())
    if selection["status"] != "PASS" or selection["selected_epoch"] != 1:
        raise RuntimeError("selected epoch1 not verified")
    for name, digest in json.loads((ATTEMPT / "execution_code_hashes.json").read_text()).items():
        if protocol.sha(name) != digest:
            raise RuntimeError(f"selection code changed: {name}")
    model = ROOT / "cicids2017/runs/ustc/A-2/trafficformer"
    meta = json.loads((model / "metrics.json").read_text())
    if meta["best_epoch"] != 1 or meta["completed_epochs"] != 1 or protocol.sha(model / "model_best.pt") != selection["checkpoint_sha256"]:
        raise RuntimeError("actual fusion branch does not use the authorized epoch1")
    original = (ROOT / "finalize.py").read_text()
    substitutions = {
        '(("trafficformer", 20), ("graph", 50), ("yatc", 200))': '(("trafficformer", 1), ("graph", 50), ("yatc", 200))',
        'Unchanged inherited TF20/FIG50/YaTC200/adapter30/head30 recipe.':
            'User-selected replayed TF epoch1 (original20-epoch scheduler horizon); unchanged FIG50/YaTC200/adapter30/head30 recipe. Original first-epoch metrics independently reproduced before selection. Full-attempt history preserved.',
        '"tmux_session": "stage49_cic_queue_20261117"': '"tmux_session": "stage49_epoch1_controller_20261117"',
        '"command": "python stage49_cic_multiclass_closed_set_20261117/run_queue.py"':
            '"command": "python stage49_cic_multiclass_closed_set_20261117/epoch1_selection_20261117/controller.py"',
    }
    adapted = original
    for before, after in substitutions.items():
        if adapted.count(before) != 1:
            raise RuntimeError(f"unexpected finalizer implementation for substitution: {before}")
        adapted = adapted.replace(before, after, 1)
    namespace = {"__file__": str(ROOT / "finalize.py"), "__name__": "stage49_epoch1_finalizer"}
    exec(compile(ast.parse(adapted), "<Stage49 selected-epoch finalizer>", "exec"), namespace)
    namespace["main"]()


if __name__ == "__main__":
    main()
