#!/usr/bin/env python3
"""Bounded, read-only inspection of the VNAT official HDF5 storage layout."""

from __future__ import annotations

import io
import pickle
from collections import Counter
from pathlib import Path

import h5py
import numpy as np


DATASET = Path(
    "/home/birkenwald/data/SEU-WXY/WXY-SEU/TrafficClassifier/"
    "Dataset/VNAT/VNAT/VNAT_Dataframe_release_1.h5"
)


class NumpyOnlyUnpickler(pickle.Unpickler):
    """Allow only the NumPy constructors used by a fixed-format ndarray pickle."""

    ALLOWED = {
        ("numpy", "dtype"),
        ("numpy", "ndarray"),
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy.core.multiarray", "scalar"),
        ("numpy._core.multiarray", "scalar"),
    }

    def find_class(self, module: str, name: str):  # type: ignore[override]
        if (module, name) not in self.ALLOWED:
            raise pickle.UnpicklingError(f"blocked global: {module}.{name}")
        imported = __import__(module, fromlist=[name])
        return getattr(imported, name)


def main() -> None:
    with h5py.File(DATASET, "r") as handle:
        group = handle["data"]
        for name, node in group.items():
            print(name, "shape=", node.shape, "dtype=", node.dtype)
            if name == "block0_values":
                for index in range(min(3, len(node))):
                    value = node[index]
                    print(
                        "sample",
                        index,
                        "type=",
                        type(value).__name__,
                        "shape=",
                        getattr(value, "shape", None),
                        "dtype=",
                        getattr(value, "dtype", None),
                        "repr=",
                        repr(value)[:500],
                    )

                raw = node[0].tobytes()
                decoded = NumpyOnlyUnpickler(io.BytesIO(raw)).load()
                print(
                    "decoded",
                    "type=", type(decoded).__name__,
                    "shape=", getattr(decoded, "shape", None),
                    "dtype=", getattr(decoded, "dtype", None),
                )
                if not isinstance(decoded, np.ndarray):
                    raise TypeError(type(decoded))
                for row in range(min(5, decoded.shape[0])):
                    print("decoded_row", row, repr(decoded[row, :3])[:1000])
                    print("decoded_file_name", row, repr(decoded[row, 4])[:1000])
                counts = Counter(str(value) for value in decoded[:, 4])
                print("official_unique_file_names", len(counts))
                print("official_nonvpn_scp_long_capture1", counts["nonvpn_scp_long_capture1.pcap"])
                print("official_vpn_skype_chat_capture6", counts["vpn_skype-chat_capture6.pcap"])
                print("official_vpn_skype_chat_capture3", counts["vpn_skype-chat_capture3.pcap"])
                print("official_vpn_skype_chat_capture4", counts["vpn_skype-chat_capture4.pcap"])


if __name__ == "__main__":
    main()
