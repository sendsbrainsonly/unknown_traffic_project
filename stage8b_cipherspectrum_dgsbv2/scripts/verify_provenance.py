#!/usr/bin/env python3
"""Verify Stage 6/7/8A frozen assets before Stage 8B calibration."""

from __future__ import annotations

import json

from stage8b_common import STAGE8B_ROOT, verify_all_provenance, write_json


def main() -> None:
    output = STAGE8B_ROOT / "outputs/summary/provenance_verification.json"
    if output.exists():
        raise RuntimeError(f"refusing to overwrite provenance record: {output}")
    payload = verify_all_provenance()
    write_json(output, payload)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
