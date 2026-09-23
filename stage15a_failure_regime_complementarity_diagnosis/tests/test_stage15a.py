from pathlib import Path
import importlib.util
import sys
import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_stage15a.py"
spec = importlib.util.spec_from_file_location("stage15a", SCRIPT)
stage15a = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = stage15a
spec.loader.exec_module(stage15a)


def test_native_decomposition():
    mu = np.asarray([[1.0, 2.0], [0.0, 0.0]])
    logvar = np.zeros_like(mu)
    prototypes = np.asarray([[0.0, 0.0], [2.0, 2.0]])
    dproto, vpost, pred = stage15a.native_parts(mu, logvar, prototypes)
    np.testing.assert_allclose(dproto, [0.5, 0.0])
    np.testing.assert_allclose(vpost, [0.0, 0.0])
    np.testing.assert_array_equal(pred, [1, 0])


def test_nearest_two_definitions():
    d1, d2, margin, ratio, pred = stage15a.nearest_two(np.asarray([[1.0, 0.0]]), np.asarray([[0.0, 0.0], [3.0, 0.0]]))
    np.testing.assert_allclose(d1, [1.0])
    np.testing.assert_allclose(d2, [2.0])
    np.testing.assert_allclose(margin, [1.0])
    np.testing.assert_allclose(ratio, [0.5])
    np.testing.assert_array_equal(pred, [0])


def test_ranking_quadrants_conserve_pairs():
    counts = stage15a.ranking_counts(0.5, np.asarray([0.1, 0.7, 0.2]), np.asarray([0.4, 0.8, 0.1]), 0.6)
    assert sum(counts) == 3
    assert counts == (2, 0, 0, 1)


def test_p95_higher():
    assert stage15a.p95(np.arange(20)) == 19.0
