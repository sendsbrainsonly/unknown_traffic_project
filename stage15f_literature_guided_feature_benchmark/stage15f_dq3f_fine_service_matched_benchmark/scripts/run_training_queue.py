#!/usr/bin/env python3
"""Run the six preregistered DQ-3F paired jobs serially on one selected GPU."""

from __future__ import annotations

import json

from dq3f_common import SEEDS
from train_dq3f import run


def main() -> int:
    results = []
    for seed in SEEDS:
        for task in ("fine", "service"):
            result = run(task, seed)
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
    print(json.dumps({"status": "SUCCESS", "runs": len(results)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
