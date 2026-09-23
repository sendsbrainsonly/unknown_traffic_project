import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_largest_remainder_is_exact_and_proportional():
    module = load_script("task11_cicids2017_prepare.py")
    allocation = module.largest_remainder({"Monday": 3, "Tuesday": 2}, 3)
    assert allocation == {"Monday": 2, "Tuesday": 1}
    assert sum(allocation.values()) == 3


def test_strict_filter_requires_every_guard():
    module = load_script("task11_cicids2017_prepare.py")
    row = {
        "verification_status": "LABEL_MATCH",
        "stored_tuple_consistent": "True",
        "referenced_row_found": "True",
        "referenced_row_tuple_match": "True",
        "referenced_row_label_match": "True",
        "existing_label": "BENIGN",
        "existing_match_status": "MATCHED",
    }
    assert module.is_strict(row)
    for key in (
        "stored_tuple_consistent", "referenced_row_found",
        "referenced_row_tuple_match", "referenced_row_label_match",
    ):
        changed = dict(row)
        changed[key] = "False"
        assert not module.is_strict(changed)


def test_gate_excludes_benign_and_uses_fixed_majority_rule():
    module = load_script("task13_cicids2017_gaussian_audit.py")
    rows = [{"class_name": "BENIGN", "supports_residual_components": True}]
    rows.extend(
        {"class_name": name, "supports_residual_components": index < 4}
        for index, name in enumerate(module.ALL_CLASSES[1:])
    )
    gate, count = module.gate_for(rows)
    assert count == 4
    assert gate == "A. CROSS-DATASET SUPPORT"


def test_summary_uses_paired_seed_deltas():
    module = load_script("task13_cicids2017_gaussian_audit.py")
    rows = []
    for seed in module.SEEDS:
        for k, value in ((1, 10.0), (2, 9.8), (3, 9.7)):
            rows.append({"class_name": "DDoS", "seed": seed, "K": k, "val_avg_nll": value})
    summary = module.summarize(rows, ("DDoS",))[0]
    assert abs(summary["delta_nll_2_mean"] - 0.2) < 1e-12
    assert abs(summary["delta_nll_3_mean"] - 0.3) < 1e-12
    assert summary["stable_meaningful_k2"]
    assert summary["stable_meaningful_k3"]


def test_train_only_pca_outputs_float64_for_full_covariance_gmm():
    module = load_script("task13_cicids2017_gaussian_audit.py")
    rng = np.random.default_rng(42)
    train = rng.normal(size=(96, 768)).astype(np.float32)
    val = rng.normal(size=(24, 768)).astype(np.float32)
    train_rep, val_rep, scaler, pca = module.fit_train_representation(train, val)
    assert train_rep.dtype == np.float64
    assert val_rep.dtype == np.float64
    assert scaler.mean_.shape == (768,)
    assert pca.n_components_ == module.PCA_DIM
