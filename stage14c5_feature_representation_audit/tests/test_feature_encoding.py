import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_inputs import SLOT_INDEX, encode, fit_scalers  # noqa: E402


def fixture():
    images = np.arange(2 * 1024, dtype=np.uint16).astype(np.uint8).reshape(2, 32, 32)
    flat = images.reshape(2, 1024)
    flat[:, SLOT_INDEX.reshape(-1)] = 0
    iats = np.asarray([[0, .001, .002, 0, 0, 0, 0, 0], [0, .02, .03, .04, 0, 0, 0, 0]], dtype=np.float32)
    lengths = np.asarray([[100, 200, 300, 0, 0, 0, 0, 0], [80, 90, 100, 110, 0, 0, 0, 0]], dtype=np.float32)
    directions = np.asarray([[1, -1, 1, 0, 0, 0, 0, 0], [1, 1, -1, -1, 0, 0, 0, 0]], dtype=np.int8)
    mask = (lengths > 0).astype(np.uint8)
    stats = np.asarray([[3, 600, .003, 200, 80, .0015, .0005, 2], [4, 380, .09, 95, 11, .03, .01, 1]], dtype=np.float64)
    return images, iats, lengths, directions, mask, stats


def test_all_features_preserve_non_address_bytes():
    images, iats, lengths, directions, mask, stats = fixture()
    scaler = fit_scalers(np.asarray([0, 1]), iats, lengths, mask, stats)
    indices = np.asarray([0, 1])
    protected = np.ones(1024, dtype=bool)
    protected[SLOT_INDEX.reshape(-1)] = False
    for feature in ("f1", "f2", "f3"):
        encoded = encode(images, indices, feature, scaler, iats, lengths, directions, mask, stats)
        assert encoded.shape == (2, 32, 32)
        assert encoded.dtype == np.uint8
        assert np.array_equal(encoded.reshape(2, 1024)[:, protected], images.reshape(2, 1024)[:, protected])


def test_direction_and_mask_are_explicit_in_f3():
    images, iats, lengths, directions, mask, stats = fixture()
    scaler = fit_scalers(np.asarray([0, 1]), iats, lengths, mask, stats)
    encoded = encode(images, np.asarray([0, 1]), "f3", scaler, iats, lengths, directions, mask, stats).reshape(2, 1024)
    assert encoded[0, SLOT_INDEX[0, 4]] == 255
    assert encoded[0, SLOT_INDEX[1, 4]] == 0
    assert encoded[0, SLOT_INDEX[3, 5]] == 0
