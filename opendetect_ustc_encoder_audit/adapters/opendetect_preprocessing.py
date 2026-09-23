"""Flow-ID preserving adapter for the released Open-Detect byte image.

This module mirrors ``Open-Detect/code/data/Preprocessing/utils.py`` for a raw
Ethernet frame already retained by the main project's Stage-0 splitter.  It is
kept local because the upstream checkout is strictly read-only and its helper
only accepts whole per-flow PCAP files.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scapy.layers.inet import IP
from scapy.layers.l2 import Ether
from scapy.packet import Raw


PACKETS_PER_FLOW = 8
HEADER_BYTES = 80
PAYLOAD_BYTES = 48
IMAGE_SIDE = 32
BYTES_PER_PACKET = HEADER_BYTES + PAYLOAD_BYTES
FLOW_BYTES = IMAGE_SIDE * IMAGE_SIDE


@dataclass(frozen=True)
class EncodedFlow:
    image: np.ndarray
    packets_found: int
    packets_used: int


def encode_packet(raw_frame: bytes) -> np.ndarray:
    """Return the released 80-header + 48-payload representation.

    Raises instead of silently creating an all-zero packet so the alignment
    gate can expose every reconstruction failure before formal training.
    """

    packet = Ether(raw_frame)
    if IP not in packet:
        raise ValueError("raw frame has no IPv4 layer")
    packet = packet.copy()
    ip = packet[IP]
    ip.src = "0.0.0.0"
    ip.dst = "0.0.0.0"

    header_hex = bytes(ip).hex()
    payload_hex = ""
    if Raw in packet:
        payload_hex = bytes(packet[Raw]).hex()
        # Preserve the exact released-code string behavior.
        header_hex = header_hex.replace(payload_hex, "")

    header_hex = header_hex[: 2 * HEADER_BYTES].ljust(2 * HEADER_BYTES, "0")
    payload_hex = payload_hex[: 2 * PAYLOAD_BYTES].ljust(2 * PAYLOAD_BYTES, "0")
    encoded = np.frombuffer(bytes.fromhex(header_hex + payload_hex), dtype=np.uint8).copy()
    if encoded.shape != (BYTES_PER_PACKET,):
        raise AssertionError(f"unexpected packet shape: {encoded.shape}")
    return encoded


def encode_flow(packets: Sequence[Sequence[object]]) -> EncodedFlow:
    """Encode the first eight stored packets into one 32x32 uint8 image."""

    if not packets:
        raise ValueError("empty flow")
    used = min(PACKETS_PER_FLOW, len(packets))
    flat = np.zeros(FLOW_BYTES, dtype=np.uint8)
    for packet_index, packet in enumerate(packets[:used]):
        if len(packet) != 5:
            raise ValueError(f"packet {packet_index} does not have five Stage-0 fields")
        raw_frame = packet[4]
        if not isinstance(raw_frame, (bytes, bytearray)):
            raise TypeError(f"packet {packet_index} raw frame is not bytes")
        start = packet_index * BYTES_PER_PACKET
        flat[start : start + BYTES_PER_PACKET] = encode_packet(bytes(raw_frame))
    return EncodedFlow(flat.reshape(IMAGE_SIDE, IMAGE_SIDE), len(packets), used)

