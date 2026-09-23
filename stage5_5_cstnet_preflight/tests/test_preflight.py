from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scapy.layers.inet import IP, TCP
from scapy.layers.l2 import Ether
from scapy.packet import Raw


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_preflight_audit import (  # noqa: E402
    parse_tls_sni,
    reliability_label,
    select_sample_classes,
)


def synthetic_client_hello(hostname: str) -> bytes:
    name = hostname.encode("ascii")
    server_name = b"\x00" + len(name).to_bytes(2, "big") + name
    sni_data = len(server_name).to_bytes(2, "big") + server_name
    extension = b"\x00\x00" + len(sni_data).to_bytes(2, "big") + sni_data
    body = (
        b"\x03\x03" + b"\x00" * 32 + b"\x00" + b"\x00\x02" + b"\x13\x01"
        + b"\x01\x00" + len(extension).to_bytes(2, "big") + extension
    )
    handshake = b"\x01" + len(body).to_bytes(3, "big") + body
    return b"\x16\x03\x01" + len(handshake).to_bytes(2, "big") + handshake


def test_tls_parser_extracts_complete_sni_and_rejects_truncation() -> None:
    record = synthetic_client_hello("example.com")
    assert parse_tls_sni([record]) == (True, ["example.com"])
    assert parse_tls_sni([record[:20]]) == (False, [])


def test_reliability_thresholds_are_fixed() -> None:
    assert reliability_label(50) == "GOOD"
    assert reliability_label(49) == "ACCEPTABLE"
    assert reliability_label(30) == "ACCEPTABLE"
    assert reliability_label(29) == "WEAK"
    assert reliability_label(20) == "WEAK"
    assert reliability_label(19) == "UNRELIABLE"


def test_class_sampling_is_deterministic_and_covers_roles() -> None:
    eligible = [f"class-{index}" for index in range(40)]
    roles = {}
    for fold_index, fold in enumerate(("low", "medium", "high")):
        for index, name in enumerate(eligible):
            roles[(fold, name)] = "unknown" if index % (fold_index + 3) == 0 else "known"
    first, _ = select_sample_classes(eligible, roles)
    second, _ = select_sample_classes(eligible, roles)
    assert first == second
    assert len(first) == 30
    assert len(set(first)) == 30
    for fold in ("low", "medium", "high"):
        assert {roles[(fold, name)] for name in first} == {"known", "unknown"}


def test_audited_adapter_masks_ip_but_retains_tcp_ports() -> None:
    audit_root = ROOT.parent / "opendetect_ustc_encoder_audit"
    sys.path.insert(0, str(audit_root))
    from adapters.opendetect_preprocessing import encode_packet

    packet = Ether() / IP(src="192.0.2.1", dst="198.51.100.2") / TCP(sport=12345, dport=443) / Raw(b"hello")
    encoded = encode_packet(bytes(packet))
    assert np.array_equal(encoded[12:20], np.zeros(8, dtype=np.uint8))
    ihl = int(encoded[0] & 0x0F) * 4
    assert bytes(encoded[ihl : ihl + 4]) == (12345).to_bytes(2, "big") + (443).to_bytes(2, "big")

