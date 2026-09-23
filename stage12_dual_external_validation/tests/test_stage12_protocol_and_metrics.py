from pathlib import Path
import sys
from unittest import mock

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from aggregate_results import gate  # noqa: E402
from open_and_evaluate import method_metrics, paired_bootstrap  # noqa: E402
from preprocess_and_freeze_splits import (  # noqa: E402
    best_flow_disjoint_assignment,
    best_group_assignment,
    global_source_group_assignment,
)
from stage12_common import load_config  # noqa: E402
from train_pretest import (  # noqa: E402
    TrafficImages,
    collect_mu_logvar,
    des_scores,
    empirical_centroids,
    import_open_detect,
    native_scores,
)


def _record(source: str, flow: str, image: str) -> dict[str, str]:
    return {"source_file": source, "flow_id_sha256": flow, "image_sha256": image}


def test_source_group_assignment_never_splits_source_or_exact_duplicate():
    records = [
        _record("a.pcap", "f1", "i1"),
        _record("b.pcap", "f2", "i2"),
        _record("c.pcap", "f3", "i3"),
        _record("d.pcap", "f4", "i4"),
        _record("e.pcap", "f5", "same"),
        _record("f.pcap", "f6", "same"),
    ]
    assignment = best_group_assignment(records, "Example", 42)
    assert set(assignment.values()) == {"train", "validation", "test"}
    assert assignment["e.pcap"] == assignment["f.pcap"]


def test_flow_fallback_keeps_byte_duplicates_together_and_roles_nonempty():
    records = [
        _record("only.pcap", "f1", "same"),
        _record("only.pcap", "f2", "same"),
        _record("only.pcap", "f3", "i3"),
        _record("only.pcap", "f4", "i4"),
        _record("only.pcap", "f5", "i5"),
        _record("only.pcap", "f6", "i6"),
    ]
    assignment = best_flow_disjoint_assignment(records, "Example", 42)
    assert set(assignment.values()) == {"train", "validation", "test"}
    assert len(assignment) == 5


def test_global_source_assignment_links_cross_class_byte_duplicates():
    records = []
    for class_name in ("A", "B"):
        for index in range(5):
            records.append({
                **_record(f"{class_name}{index}.pcap", f"{class_name}-f{index}", f"{class_name}-i{index}"),
                "class_name": class_name,
            })
    records[0]["image_sha256"] = "cross-class-same"
    records[5]["image_sha256"] = "cross-class-same"
    assignment, infeasible = global_source_group_assignment(records, ["A", "B"], 42)
    assert infeasible == []
    assert assignment is not None
    assert assignment["A0.pcap"] == assignment["B0.pcap"]
    for class_name in ("A", "B"):
        assert {assignment[f"{class_name}{index}.pcap"] for index in range(5)} == {
            "train", "validation", "test"
        }


def test_des_uses_known_train_empirical_centroids_and_squared_distance():
    mu = np.asarray([[0.0, 0.0], [2.0, 0.0], [10.0, 0.0], [12.0, 0.0]], dtype=np.float32)
    labels = np.asarray([0, 0, 1, 1])
    centroids = empirical_centroids(mu, labels, 2)
    np.testing.assert_allclose(centroids, [[1.0, 0.0], [11.0, 0.0]])
    scores, predictions = des_scores(np.asarray([[3.0, 0.0], [8.0, 0.0]], dtype=np.float32), centroids)
    np.testing.assert_allclose(scores, [4.0, 9.0])
    np.testing.assert_array_equal(predictions, [0, 1])


def test_corrected_open_detect_model_api_supports_shared_readouts_on_cpu():
    config = load_config()
    CorrectedOpenDetectNet, _, _, _ = import_open_detect(config)
    device = torch.device("cpu")
    # The released constructor hard-codes Tensor.cuda() for prototypes. Patch
    # only that constructor call so this API contract test remains CPU-only.
    with mock.patch.object(torch.Tensor, "cuda", lambda tensor, *args, **kwargs: tensor):
        model = CorrectedOpenDetectNet("resnet18", 1, 128, 2, 1, 1).to(device)
    images = np.zeros((4, 32, 32), dtype=np.uint8)
    labels = np.asarray([0, 1, 0, 1], dtype=np.int64)
    loader = DataLoader(TrafficImages(images, labels, [0, 1], train=False), batch_size=2)
    mu, logvar, reindexed = collect_mu_logvar(model, loader, device)
    scores, predictions = native_scores(model, mu, logvar, device)
    assert mu.shape == (4, 128)
    assert logvar.shape == (4, 128)
    assert scores.shape == (4,)
    assert predictions.shape == (4,)
    np.testing.assert_array_equal(reindexed, labels)


def test_metrics_and_paired_bootstrap_have_unknown_as_positive():
    known = np.asarray([0.1, 0.2, 0.3, 0.4])
    unknown = np.asarray([0.7, 0.8, 0.9, 1.0])
    metrics = method_metrics("M", known, unknown, np.asarray([0, 1, 0, 1]), np.asarray([0, 1, 0, 1]), 0.5)
    assert metrics["auroc"] == 1.0
    assert metrics["ufar"] == 0.0
    assert metrics["known_frr"] == 0.0
    rows, summary = paired_bootstrap(known, unknown, known, unknown, 0.5, 0.5, 20, 0)
    assert len(rows) == 20 * 4
    for value in summary.values():
        assert value["delta_mean"] == 0.0
        assert value["ci95_low"] == 0.0
        assert value["ci95_high"] == 0.0


def _gate_rows(delta: float, better: int = 5, frr: float = 0.0, macro_f1: float = 0.0) -> list[dict]:
    return [
        {
            "dataset": dataset,
            "setting": setting,
            "delta_auroc_mean": delta,
            "des_auroc_better_seed_count": better,
            "seed_count": 5,
            "delta_known_frr_mean": frr,
            "delta_known_macro_f1_mean": macro_f1,
            "split_policy": "SOURCE_PCAP_GROUP_AWARE_AND_EXACT_IMAGE_LINKED",
        }
        for dataset in ("iscx_vpn", "iscx_tor")
        for setting in ("low", "medium", "high")
    ]


def test_registered_gate_three_way_decisions():
    config = load_config()
    assert gate(_gate_rows(0.03), config)["decision"] == "EXTERNAL_CONFIRMED"
    partial = _gate_rows(0.005, better=4)
    assert gate(partial, config)["decision"] == "PARTIAL_CONFIRMATION"
    failed = _gate_rows(-0.03, better=0)
    assert gate(failed, config)["decision"] == "NOT_CONFIRMED"
