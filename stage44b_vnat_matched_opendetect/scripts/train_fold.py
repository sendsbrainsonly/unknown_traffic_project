#!/usr/bin/env python3
"""Run the unchanged Stage 14C native training routine on Stage 44 roles."""
from __future__ import annotations

import sys

from matched_common import PROJECT, ROOT, load_protocols, verify_freeze


sys.path.insert(0, str(PROJECT / "stage14c_native_opendetect_vnat" / "scripts"))
import train_one as native  # noqa: E402


# The released model, augmentation, optimizer, epoch loop, prototype reset and
# Known-Val checkpoint rule remain in train_one.py. Only frozen input/output
# paths and the service-level class names are redirected to this bundle.
native.RUNS_ROOT = ROOT / "runs"
native.run_input_dir = lambda fold: ROOT / "inputs" / fold
native.verify_freeze = verify_freeze
native.load_protocols = load_protocols


if __name__ == "__main__":
    native.main()
