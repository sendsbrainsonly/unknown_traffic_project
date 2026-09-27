#!/usr/bin/env python3
"""One-shot project-local progress summary; never launches or modifies a model."""
from __future__ import annotations

import json
from datetime import datetime, timezone

from preflight import CELLS, OUT


def status(path):
    if (path / "SUCCESS").is_file():
        return "PASS"
    if (path / "FAILURE.json").is_file():
        return "FAIL"
    if path.exists():
        return "RUNNING_OR_PARTIAL"
    return "PENDING"


rows = []
for dataset, protocol in CELLS:
    cache = OUT / "input_caches" / dataset
    if dataset == "vnat":
        cache = cache / protocol
        mfr = cache / ("yatc_mfr_attempt3" if protocol == "medium_seed2025" else "yatc_mfr")
    elif dataset == "ustc":
        cache = cache / protocol
        mfr = cache / "yatc_mfr"
    else:
        mfr = cache / "yatc_mfr"
    tf_fig = cache / "tf_fig"
    run = OUT / "runs" / dataset / protocol
    row = {"dataset": dataset, "protocol": protocol,
           "mfr_input": "PASS" if (mfr / "cache_audit.json").is_file() else "PENDING",
           "tf_fig_input": "PASS" if (tf_fig / "cache_audit.json").is_file() else "PENDING",
           **{name: status(run / name) for name in ("trafficformer", "graph", "yatc", "T0_equal")},
           "known_test": "PASS" if (run / "known_test_results.json").is_file() else "LOCKED_OR_PENDING"}
    rows.append(row)
summary = {"updated_at_utc": datetime.now(timezone.utc).isoformat(),
           "cells": len(rows), "input_ready": sum(r["mfr_input"] == r["tf_fig_input"] == "PASS" for r in rows),
           "branch_complete": sum(r[name] == "PASS" for r in rows for name in ("trafficformer", "graph", "yatc")),
           "branch_total": 15, "t0_complete": sum(r["T0_equal"] == "PASS" for r in rows),
           "test_complete": sum(r["known_test"] == "PASS" for r in rows), "rows": rows}
# After final bundle validation, keep the hashed progress artifact immutable.
completion = OUT / "completion_verification.json"
if not completion.is_file() or json.loads(completion.read_text()).get("status") != "PASS":
    (OUT / "progress.json").write_text(json.dumps(summary, indent=2) + "\n")
print(f"Stage31 inputs {summary['input_ready']}/5; branches {summary['branch_complete']}/15; "
      f"T0 heads {summary['t0_complete']}/5; Known Test {summary['test_complete']}/5")
for row in rows:
    print("{dataset:9} {protocol:16} input={mfr_input}/{tf_fig_input} "
          "branches={trafficformer}/{graph}/{yatc} T0={T0_equal} Test={known_test}".format(**row))
