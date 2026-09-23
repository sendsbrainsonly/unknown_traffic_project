from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from stage12_common import (  # noqa: E402
    classify_iscx_tor,
    classify_iscx_vpn,
    frozen_unknown_protocol,
    openness_counts,
    quantile_higher,
)


def test_vpn_mapping_and_domain_collapse():
    root = Path("/dataset")
    cases = {
        "VPN-PCAPS-01/vpn_facebook_chat1a.pcap": ("Facebook", "Chat", "VPN"),
        "NonVPN-PCAPs-03/skype_video1b.pcapng": ("Skype", "VoIP", "NonVPN"),
        "NonVPN-PCAPs-03/scpDown4.pcap": ("SCP", "File-Transfer", "NonVPN"),
        "VPN-PCAPs-02/vpn_vimeo_A.pcap": ("Vimeo", "Streaming", "VPN"),
    }
    for value, expected in cases.items():
        got = classify_iscx_vpn(root / value, root)
        assert (got.canonical_class, got.official_category, got.domain_state) == expected


def test_tor_mapping_and_domain_collapse():
    root = Path("/dataset")
    cases = {
        "Tor/Tor/CHAT_gate_facebook_chat.pcap": ("Facebook", "Chat", "Tor"),
        "Tor/Tor/torVimeo2.pcap": ("Vimeo", "Video", "Tor"),
        "NonTor/Workstation_Thunderbird_POP.pcap": ("Email", "Email", "NonTor"),
        "NonTor/p2p_vuze.pcap": ("P2P", "P2P", "NonTor"),
    }
    for value, expected in cases.items():
        got = classify_iscx_tor(root / value, root)
        assert (got.canonical_class, got.official_category, got.domain_state) == expected


def test_openness_and_nested_protocol_are_frozen():
    classes = [f"C{i:02d}" for i in range(15)]
    assert openness_counts(15) == {"low": 1, "medium": 3, "high": 4}
    first = frozen_unknown_protocol(classes, 42)
    second = frozen_unknown_protocol(reversed(classes), 42)
    assert first == second
    assert first["nested"] is True
    assert set(first["settings"]["low"]["unknown_classes"]) < set(
        first["settings"]["medium"]["unknown_classes"]
    )


def test_quantile_is_higher_not_linear():
    values = np.arange(20, dtype=np.float64)
    assert quantile_higher(values) == 19.0

