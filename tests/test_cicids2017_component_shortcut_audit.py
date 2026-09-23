import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_module():
    path = ROOT / "scripts" / "task14_cicids2017_component_shortcut_audit.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_safe_association_marks_constant_metadata_unidentifiable():
    module = load_module()
    result = module.safe_association([0, 1, 0, 1], ["Friday"] * 4)
    assert result["nmi"] == 0.0
    assert result["ami"] == 0.0
    assert not result["identifiable"]
    assert result["status"] == "NOT_IDENTIFIABLE_CONSTANT_WITHIN_CLASS"


def test_top_n_is_fit_from_train_values_only():
    module = load_module()
    retained = module.top_n_categories(["a", "a", "b", "c"], limit=2)
    transformed = module.apply_top_n(["a", "validation-only", None], retained)
    assert retained == {"a", "b"}
    assert transformed.tolist() == ["a", "__OTHER__", "__MISSING__"]


def test_component_weight_tv_matches_definition():
    module = load_module()
    train = np.asarray([0, 0, 0, 1])
    val = np.asarray([0, 1, 1, 1])
    assert abs(module.component_weight_tv(train, val) - 0.5) < 1e-12


def test_effect_size_is_exactly_ustc_stage26_definition():
    module = load_module()
    observed = module.epsilon_squared_kruskal(50.0, 100, 2)
    assert abs(observed - ((50.0 - 2 + 1) / (100 - 2))) < 1e-12


def test_port_categories_cover_required_bands():
    module = load_module()
    assert module.port_category(0) == "well-known"
    assert module.port_category(1023) == "well-known"
    assert module.port_category(1024) == "registered"
    assert module.port_category(49151) == "registered"
    assert module.port_category(49152) == "dynamic"
    assert module.port_category(65535) == "dynamic"


def test_diagnosis_consumes_val_rows_from_generated_tables():
    module = load_module()
    gaussian = {
        (class_name, 0): {"val_avg_nll_k1_minus_k2": "1.0"}
        for class_name in module.CLASSES
    }
    weights = [
        {
            "class_name": class_name,
            "seed": 0,
            "tiny_component": False,
            "component_weight_tv_distance": 0.01,
            "train_count": 10,
            "val_count": 4,
            "gmm_train_weight": 0.5,
        }
        for class_name in module.CLASSES
    ]
    continuous = [
        {
            "class_name": class_name,
            "seed": 0,
            "split": "val",
            "feature": "packet_count",
            "epsilon_squared": 0.2,
        }
        for class_name in module.CLASSES
    ]
    categorical = [
        {
            "class_name": class_name,
            "split": "val",
            "variant": "raw",
            "identifiable": True,
            "feature": "src_port",
            "nmi": 0.1,
            "ami": 0.09,
        }
        for class_name in module.CLASSES
    ]
    predictability = [
        {
            "class_name": class_name,
            "model": model,
            "macro_f1": 0.8,
            "balanced_accuracy": 0.8,
        }
        for class_name in module.CLASSES
        for model in ("Model S", "Model S+")
    ]
    stability = [
        {"class_name": class_name, "split": "val", "nmi": 1.0}
        for class_name in module.CLASSES
    ]

    diagnoses = module.build_diagnosis(
        gaussian, weights, continuous, categorical, predictability, stability
    )

    assert len(diagnoses) == 4
    for row in diagnoses:
        assert row["minimum_pairwise_validation_seed_nmi"] == 1.0
        assert row["strongest_categorical_feature"] == "src_port"
        assert "packet_count" in row["strongest_continuous_features"]
        assert "mean_epsilon_squared" in row["strongest_continuous_features"]
        assert row["minimum_component_train_count"] == 10
        assert row["minimum_component_val_count"] == 4
