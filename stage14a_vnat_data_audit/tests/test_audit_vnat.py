from __future__ import annotations

import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from audit_vnat import assign_effective_group_ids, eligibility, parse_pcap_name, write_csv


def test_filename_parsing_and_conservative_grouping() -> None:
    vpn = parse_pcap_name("vpn_voip_capture2.pcap")
    nonvpn = parse_pcap_name("nonvpn_voip_capture2.pcap")
    assert vpn["application"] == "zoiper"
    assert vpn["vpn_status"] == "vpn"
    assert nonvpn["vpn_status"] == "nonvpn"
    assert vpn["capture_id"] != nonvpn["capture_id"]
    assert vpn["group_id"] == nonvpn["group_id"] == "zoiper::capture::2"


def test_capture_variants_do_not_collapse() -> None:
    regular = parse_pcap_name("nonvpn_scp_capture1.pcap")
    long_capture = parse_pcap_name("nonvpn_scp_long_capture1.pcap")
    new_capture = parse_pcap_name("nonvpn_scp_newcapture1.pcap")
    assert len({regular["group_id"], long_capture["group_id"], new_capture["group_id"]}) == 3


def test_exact_duplicate_pcaps_share_effective_group() -> None:
    rows = [
        {"strict_base_group_id": "skype::capture::3", "group_id": "skype::capture::3", "sha256": "same"},
        {"strict_base_group_id": "skype::capture::4", "group_id": "skype::capture::4", "sha256": "same"},
        {"strict_base_group_id": "skype::capture::5", "group_id": "skype::capture::5", "sha256": "different"},
    ]
    assign_effective_group_ids(rows)
    assert rows[0]["group_id"] == rows[1]["group_id"]
    assert rows[2]["group_id"] != rows[0]["group_id"]


def test_corrupt_capture_is_a_caveat_not_automatic_class_exclusion() -> None:
    row = {
        "application": "example",
        "failed_pcap_count": 1,
        "flow_count": 100,
        "strict_group_count": 4,
        "valid_strict_group_count": 3,
        "effective_group_count": 3,
        "vpn_flow_count": 50,
        "nonvpn_flow_count": 50,
        "exact_duplicate_pcap_count": 0,
        "exact_duplicate_flow_count": 0,
    }
    eligible, excluded = eligibility([row])
    assert [item["application"] for item in eligible["eligible_classes"]] == ["example"]
    assert excluded["excluded_classes"] == []
    assert "exclude failed PCAPs" in eligible["eligible_classes"][0]["caveats"][0]


def test_unified_csv_keeps_late_record_fields(tmp_path) -> None:
    output = tmp_path / "manifest.csv"
    write_csv(output, [{"common": 1}, {"common": 2, "pcap_only": 3}])
    assert output.read_text(encoding="utf-8").splitlines()[0] == "common,pcap_only"
