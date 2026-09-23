#!/usr/bin/env python3
"""Pre-Test verification of every frozen Stage 6/7/8A/8B dependency."""

from __future__ import annotations

import json

from stage9_common import STAGE9_ROOT, verify_all_provenance, write_json


def main() -> None:
    output = STAGE9_ROOT / "outputs/provenance_verification.json"
    if output.exists():
        raise RuntimeError(f"refusing to overwrite provenance evidence: {output}")
    payload = verify_all_provenance()
    write_json(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
