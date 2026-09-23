from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


STAGE7_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = STAGE7_ROOT.parent
SCRIPT_ROOT = STAGE7_ROOT / "scripts"
sys.path.insert(0, str(SCRIPT_ROOT))

from common import (  # noqa: E402
    VALID_SETTINGS,
    encode_pcap_first_eight,
    label_maps,
    load_fold,
    select_training_rows,
    verify_stage6_protocol_hashes,
)


def test_stage6_protocol_hashes_are_frozen() -> None:
    assert len(verify_stage6_protocol_hashes()) == 16


def test_all_settings_select_only_known_train_and_validation() -> None:
    expected_known = {"low": 38, "medium": 34, "high": 30}
    for setting in VALID_SETTINGS:
        fold = load_fold(setting)
        class_to_local, local_to_class = label_maps(fold)
        assert len(class_to_local) == expected_known[setting]
        assert set(class_to_local.values()) == set(range(expected_known[setting]))
        assert set(local_to_class.values()) == set(fold["known_classes"])
        selected, audit = select_training_rows(setting, "smoke", 2, 1, 2022)
        selected_classes = {
            row["class_name"] for role_rows in selected.values() for row in role_rows
        }
        assert selected_classes == set(fold["known_classes"])
        assert not selected_classes.intersection(fold["unknown_classes"])
        assert audit["unknown_sample_ids_intersect_selected"] == 0
        assert len(selected["KNOWN_TRAIN"]) == 2 * expected_known[setting]
        assert len(selected["KNOWN_VALIDATION"]) == expected_known[setting]


def test_streaming_encoding_matches_released_preprocessing() -> None:
    selected, _ = select_training_rows("low", "smoke", 1, 1, 2022)
    row = selected["KNOWN_TRAIN"][0]
    path = Path(row["pcap_path"])
    streamed, packets_used = encode_pcap_first_eight(path)
    assert streamed.shape == (32, 32)
    assert streamed.dtype == np.uint8
    assert 1 <= packets_used <= 8

    upstream_path = PROJECT_ROOT.parent / "Open-Detect/code/data/Preprocessing/utils.py"
    spec = importlib.util.spec_from_file_location("released_preprocessing", upstream_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    released = module.read_pcap_list(str(path), if_augment=False)[0]["data"].reshape(32, 32)
    np.testing.assert_array_equal(streamed, released)
