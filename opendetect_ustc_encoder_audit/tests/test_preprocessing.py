from __future__ import annotations

import importlib.util
import pickle
import sys
from pathlib import Path

import numpy as np
from scapy.layers.l2 import Ether


AUDIT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = AUDIT_ROOT.parent
UPSTREAM_UTILS = (
    PROJECT_ROOT.parent / "Open-Detect/code/data/Preprocessing/utils.py"
)
sys.path.insert(0, str(AUDIT_ROOT))

from adapters.opendetect_preprocessing import encode_flow, encode_packet  # noqa: E402


def representative_packet() -> bytes:
    path = PROJECT_ROOT / "data/flows/Malware__Cridex__Cridex.pkl"
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    first_flow = next(iter(payload["packets"].values()))
    return first_flow[0][4]


def load_upstream_utils():
    spec = importlib.util.spec_from_file_location("opendetect_upstream_preprocessing", UPSTREAM_UTILS)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_packet_matches_released_helper() -> None:
    raw = representative_packet()
    upstream = load_upstream_utils()
    header, payload = upstream.raw_packet_to_string(Ether(raw))
    expected = np.frombuffer(bytes.fromhex(header + payload), dtype=np.uint8)
    np.testing.assert_array_equal(encode_packet(raw), expected)


def test_flow_shape_padding_and_determinism() -> None:
    raw = representative_packet()
    packet = (0.0, len(raw), len(raw), 1, raw)
    first = encode_flow([packet])
    second = encode_flow([packet])
    assert first.image.shape == (32, 32)
    assert first.image.dtype == np.uint8
    assert first.packets_found == 1
    assert first.packets_used == 1
    np.testing.assert_array_equal(first.image, second.image)
    assert np.count_nonzero(first.image.reshape(-1)[128:]) == 0

