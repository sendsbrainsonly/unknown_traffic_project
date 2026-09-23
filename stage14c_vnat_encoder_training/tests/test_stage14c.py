from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from common import canonical_tuple, load_protocol_document, tshark_flow_id, verify_freeze
from build_flow_cache import ethernet_compatible_frame

def test_frozen_grid_and_hash():
    assert verify_freeze()["freeze_hash"] == "c904daef04d2c8e79469191121325c0a6ceeaeb3e595ef9c4c7d7a62c980ae2f"
    assert len(load_protocol_document()["protocols"]) == 15

def test_flow_ids():
    assert tshark_flow_id(["1.1.1.1","","2.2.2.2","","6","","1","2","","","7",""]) == "tcp:7"
    assert tshark_flow_id(["1.1.1.1","","2.2.2.2","","17","","","","1","2","","9"]) == "udp:9"
    assert tshark_flow_id(["","","","","","","","","","","",""]) is None
    assert canonical_tuple("2","9","1","8","x") == "x|1:8|2:9"

def test_raw_ipv4_envelope():
    raw=bytes.fromhex("45000014")+b"\x00"*16
    wrapped=ethernet_compatible_frame(raw)
    assert wrapped[12:14] == b"\x08\x00"
    assert wrapped[14:] == raw
