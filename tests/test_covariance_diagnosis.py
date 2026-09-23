# -*- coding: utf-8 -*-
"""Focused tests for Stage 2.5 covariance diagnosis helpers."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.task10f_covariance_diagnosis import (
    MAIN_SETTINGS,
    SENSITIVITY_REGS,
    Setting,
    build_summary,
    decorate_group_metrics,
    fit_gmm,
    fit_pca_train_only,
    fixed_sample_indices,
    resolve_classes,
    _safe_output_dir,
)


class Stage25ProtocolTests(unittest.TestCase):
    def test_class_names_are_matched_without_hardcoded_ids(self):
        mapping = {"Outlook": 17, "FTP": 2, "Miuref": 11, "Cridex": 4}
        matched = resolve_classes(mapping, ["ftp", "CRIDEX", "miu-ref", "outlook"])
        self.assertEqual(matched, [("FTP", 2), ("Cridex", 4), ("Miuref", 11), ("Outlook", 17)])

    def test_pca_is_fit_on_train_and_only_transforms_val(self):
        rng = np.random.default_rng(0)
        train = rng.normal(size=(40, 6))
        val = rng.normal(loc=100.0, size=(10, 6))
        pca, train_z, val_z = fit_pca_train_only(train, val, 3)
        np.testing.assert_allclose(pca.mean_, train.mean(axis=0), atol=1e-12)
        self.assertEqual(train_z.shape, (40, 3))
        self.assertEqual(val_z.shape, (10, 3))
        self.assertGreater(abs(val_z.mean()), 1.0)

    def test_fixed_sampling_is_deterministic_and_sorted(self):
        indices = np.arange(100)
        first = fixed_sample_indices(indices, 20, seed=0, class_id=3)
        second = fixed_sample_indices(indices, 20, seed=0, class_id=3)
        np.testing.assert_array_equal(first, second)
        self.assertTrue(np.all(first[:-1] < first[1:]))
        self.assertEqual(len(first), 20)

    def test_stage2_output_tree_is_protected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with self.assertRaises(ValueError):
                _safe_output_dir(root, "outputs/stage2")
            with self.assertRaises(ValueError):
                _safe_output_dir(root, "outputs/stage2/diagnosis")
            allowed = _safe_output_dir(
                root, "outputs/stage2_5_covariance_diagnosis"
            )
            self.assertEqual(
                allowed,
                (root / "outputs/stage2_5_covariance_diagnosis").resolve(),
            )

    def test_fit_metrics_and_component_rows_are_auditable(self):
        rng = np.random.default_rng(1)
        train = np.vstack([
            rng.normal(-2.0, 0.3, size=(40, 2)),
            rng.normal(2.0, 0.3, size=(40, 2)),
        ])
        val = np.vstack([
            rng.normal(-2.0, 0.3, size=(20, 2)),
            rng.normal(2.0, 0.3, size=(20, 2)),
        ])
        setting = Setting("PCA64", 64, "full", 1e-3)
        row, components, warning_rows = fit_gmm(
            "Demo", 7, setting, 2, train, val, len(train), False
        )
        self.assertEqual(row["train_samples"], 80)
        self.assertEqual(row["val_samples"], 40)
        self.assertAlmostEqual(row["val_nll"], -row["val_avg_loglik"])
        self.assertEqual(sum(item["component_size"] for item in components), 80)
        self.assertEqual(row["warning_count"], len(warning_rows))

    def test_k1_delta_decoration_uses_absolute_nll(self):
        rows = []
        for k, nll, bic in [(1, 10.0, 100.0), (2, 8.0, 90.0), (3, 9.0, 95.0),
                            (4, 8.5, 97.0), (5, 8.2, 92.0)]:
            rows.append({
                "class_id": 1, "representation": "PCA64", "pca_dim": 64,
                "covariance_type": "full", "reg_covar": 1e-3, "K": k,
                "fit_id": f"k{k}", "val_nll": nll, "val_avg_loglik": -nll,
                "bic_train": bic,
            })
        decorate_group_metrics(rows)
        self.assertEqual(rows[0]["best_K_by_val_nll"], 2)
        self.assertEqual(rows[0]["best_K_by_bic"], 2)
        self.assertAlmostEqual(rows[0]["delta_val_nll_K1_Kbest"], 2.0)
        self.assertAlmostEqual(rows[1]["delta_avg_loglik_from_K1"], 2.0)

    def test_summary_answers_all_questions_and_selects_one_conclusion(self):
        classes = [(1, "FTP"), (2, "Cridex"), (3, "Miuref"), (4, "Outlook")]

        def make_row(class_id, name, setting, k):
            improvement = (k - 1) * (0.2 if setting.covariance_type == "full" else 0.1)
            return {
                "class_id": class_id,
                "class": name,
                "representation": setting.representation,
                "pca_dim": "" if setting.pca_dim is None else setting.pca_dim,
                "covariance_type": setting.covariance_type,
                "reg_covar": setting.reg_covar,
                "K": k,
                "fit_id": f"{class_id}-{setting.label}-{setting.reg_covar}-{k}",
                "val_nll": 10.0 - improvement,
                "val_avg_loglik": -10.0 + improvement,
                "bic_train": 100.0 - k,
                "tiny_components_0_5pct": 0,
                "converged": 1,
            }

        main_rows = [
            make_row(class_id, name, setting, k)
            for class_id, name in classes
            for setting in MAIN_SETTINGS
            for k in range(1, 6)
        ]
        sensitivity_settings = [
            Setting("PCA64", 64, covariance, reg)
            for covariance in ("diag", "full")
            for reg in SENSITIVITY_REGS
        ]
        sensitivity_rows = [
            make_row(class_id, name, setting, k)
            for class_id, name in classes
            for setting in sensitivity_settings
            for k in range(1, 6)
        ]
        decorate_group_metrics(main_rows)
        decorate_group_metrics(sensitivity_rows)
        pca_metadata = {
            64: {"train_rows": 100, "original_dim": 896,
                 "explained_variance_ratio_sum": 0.8},
            128: {"train_rows": 100, "original_dim": 896,
                  "explained_variance_ratio_sum": 0.9},
        }
        text, conclusion = build_summary(
            main_rows, sensitivity_rows, pca_metadata, None, 0.01, 0
        )
        for number in range(1, 7):
            self.assertIn(f"### {number}.", text)
        choices = (
            "A. Evidence supports genuine multi-component structure.",
            "B. Evidence suggests covariance misspecification explains a substantial part of the apparent multimodality.",
            "C. Result is numerically unstable / inconclusive.",
        )
        self.assertIn(conclusion, choices)
        self.assertEqual(sum(text.count(choice) for choice in choices), 1)


if __name__ == "__main__":
    unittest.main()
