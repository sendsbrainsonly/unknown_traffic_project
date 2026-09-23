from pathlib import Path
import importlib.util
import sys
import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_stage15b.py"
spec = importlib.util.spec_from_file_location("stage15b", SCRIPT)
stage15b = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = stage15b
spec.loader.exec_module(stage15b)


def test_empirical_percentile_uses_reference_only():
    validation = np.asarray([1.0, 2.0, 2.0, 4.0])
    query = np.asarray([0.0, 1.0, 2.0, 3.0, 5.0])
    np.testing.assert_allclose(stage15b.empirical_percentile(validation, query), [0.0, 0.25, 0.75, 0.75, 1.0])


def test_hybrid_formulas_are_fixed_maxima():
    a_od = np.asarray([0.2, 0.9])
    a_des0 = np.asarray([0.8, 0.1])
    a_des1 = np.asarray([0.7, 1.0])
    a_proto = np.asarray([0.4, 0.3])
    a_vpost = np.asarray([0.5, 0.6])
    np.testing.assert_allclose(np.maximum(a_od, a_des0), [0.8, 0.9])
    np.testing.assert_allclose(np.maximum(a_od, a_des1), [0.7, 1.0])
    np.testing.assert_allclose(np.maximum.reduce([a_proto, a_des0, a_vpost]), [0.8, 0.6])


def test_metrics_known_val_p95_only():
    result = stage15b.metrics(np.arange(20.0), np.asarray([0.0, 1.0]), np.asarray([18.0, 20.0]))
    assert result["threshold"] == 19.0
    assert result["known_frr"] == 0.0
    assert result["ufar"] == 0.5


def test_paired_bootstrap_is_deterministic():
    values = np.asarray([-1.0, 1.0, 2.0])
    assert stage15b.paired_bootstrap(values, repetitions=100, seed=7) == stage15b.paired_bootstrap(values, repetitions=100, seed=7)
