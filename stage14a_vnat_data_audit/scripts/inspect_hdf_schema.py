#!/usr/bin/env python3
"""Read-only, bounded schema inspection for the two VNAT HDF5 artifacts."""

from __future__ import annotations

import json
import pickletools
from pathlib import Path

import h5py


DATASET = Path("/home/birkenwald/data/SEU-WXY/WXY-SEU/TrafficClassifier/Dataset/VNAT/VNAT")


def decode(values):
    return [value.decode(errors="replace") if isinstance(value, bytes) else str(value) for value in values]


def main() -> None:
    for name, key in (
        ("VNAT_Dataframe_release_1.h5", "data"),
        ("VNAT_Feature_Dataframe_release_1.h5", "features"),
    ):
        path = DATASET / name
        print(f"FILE {name}")
        with h5py.File(path, "r") as handle:
            group = handle[key]
            print("ATTRIBUTES", json.dumps({name: repr(value) for name, value in group.attrs.items()}, indent=2))
            for dataset in ("axis0", "axis1", "block0_items", "block1_items"):
                if dataset in group:
                    values = group[dataset]
                    if len(values) <= 200:
                        print(dataset, decode(values[:]))
                    else:
                        print(dataset, {"shape": list(values.shape), "dtype": str(values.dtype)})

    # The feature table is only about 9 MB on disk. Pandas' optional PyTables
    # dependency is not available in the fixed environment, so decode the one
    # pickled object block directly with h5py. This does not touch the 1 GB raw
    # packet dataframe's object block.
    with h5py.File(DATASET / "VNAT_Feature_Dataframe_release_1.h5", "r") as handle:
        blob = handle["features/block1_values"][0]
        raw = blob.tobytes() if hasattr(blob, "tobytes") else bytes(blob)
        # Parse opcodes only. Never execute the pickle or instantiate classes.
        string_arguments = []
        for opcode, argument, _ in pickletools.genops(raw):
            if opcode.name in {"BINUNICODE", "SHORT_BINUNICODE", "UNICODE", "STRING", "BINSTRING", "SHORT_BINSTRING"}:
                if isinstance(argument, bytes):
                    argument = argument.decode(errors="replace")
                string_arguments.append(str(argument))
        counts = {}
        for value in string_arguments:
            counts[value] = counts.get(value, 0) + 1
        print("FEATURE_LABEL_PICKLE_BYTES", len(raw))
        print("FEATURE_LABEL_STRING_OPCODE_COUNTS", json.dumps(counts, sort_keys=True))


if __name__ == "__main__":
    main()
