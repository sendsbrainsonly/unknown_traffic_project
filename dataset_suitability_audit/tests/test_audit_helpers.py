from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_audit.py"
SPEC = spec_from_file_location("run_audit", SCRIPT)
MODULE = module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_cipher_filename_metadata():
    path = Path(
        "root/mix/mix/example.org/"
        "traffic_2024-02-19_example.org_chacha20_firefox_7.pcap."
        "TCP_10-0-2-15_12345_1-2-3-4_443.pcap"
    )
    assert MODULE.cipher_metadata(path) == {
        "capture_group": "mix",
        "domain": "example.org",
        "date": "2024-02-19",
        "cipher": "chacha20",
    }


def test_pcap_magic(tmp_path):
    path = tmp_path / "one.pcap"
    path.write_bytes(b"\xd4\xc3\xb2\xa1" + b"\x00" * 20)
    assert MODULE.pcap_magic(path)[:2] == (True, "classic_pcap_le_us")


def test_npy_header_without_loading_body(tmp_path):
    path = tmp_path / "values.npy"
    np.save(path, np.zeros((7, 11), dtype=np.int32))
    shape, dtype, fortran = MODULE.npy_header(path)
    assert shape == (7, 11)
    assert dtype == "int32"
    assert fortran is False


def test_ranking_is_fixed_and_sums_to_twenty_maximum():
    rows = MODULE.ranking_rows()
    assert [row["dataset"] for row in rows] == [
        "CSTNET-TLS1.3",
        "CipherSpectrum",
        "CICDDoS2019",
    ]
    for row in rows:
        assert row["total_score"] == sum(MODULE.SCORES[row["dataset"]].values())
        assert 0 <= row["total_score"] <= 20


def test_suitability_outputs_use_observed_class_counts():
    primary = [
        {"dataset": "CipherSpectrum"} for _ in range(85)
    ] + [
        {"dataset": "CSTNET-TLS1.3"} for _ in range(120)
    ] + [
        {"dataset": "CICDDoS2019"} for _ in range(19)
    ]
    open_set, _, _, _, openness = MODULE.fixed_suitability_rows(primary, {})
    observed = {row["dataset"]: row["num_candidate_classes"] for row in open_set}
    assert observed == {
        "CipherSpectrum": 85,
        "CSTNET-TLS1.3": 120,
        "CICDDoS2019": 19,
    }
    for row in openness:
        for field in ("low_unknown_ratio", "medium_unknown_ratio", "high_unknown_ratio"):
            known, unknown = (int(value) for value in row[field].split("/"))
            assert known + unknown == row["candidate_classes"]
