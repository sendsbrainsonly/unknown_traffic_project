#!/usr/bin/env python3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import run as experiment


class Contracts(unittest.TestCase):
    def test_current_model_is_directly_reused(self):
        adapter = experiment.current.Adapters(7)
        self.assertEqual([v.mean.out_features for v in adapter.views], [64, 64, 64])
        head = experiment.current.EqualFusion(7, np.zeros(3), np.ones(3))
        head.eval()
        with torch.no_grad():
            logits, weights = head(torch.randn(9, 3, 64), torch.randn(9, 3))
        self.assertEqual(tuple(logits.shape), (9, 7))
        torch.testing.assert_close(weights, torch.full((9, 3), 1/3))
        self.assertEqual(head.fuser[0].in_features, 192)

    def test_normalization_is_train_only(self):
        train = {name: np.asarray([[0., 2.], [2., 4.]], np.float32) for name in experiment.VIEWS}
        val = {name: np.asarray([[101., 103.]], np.float32) for name in experiment.VIEWS}
        stats = experiment.current.normalize(train, val)
        for name in experiment.VIEWS:
            np.testing.assert_array_equal(stats[name][0], [1., 3.])
            np.testing.assert_array_equal(val[name], [[100., 100.]])

    def test_test_guard_before_file_open(self):
        with tempfile.TemporaryDirectory(dir=experiment.ROOT, prefix="test_gate_") as folder:
            with patch.object(experiment, "ROOT", Path(folder)), patch.object(experiment.np, "load") as read_features:
                with self.assertRaises(FileNotFoundError):
                    experiment.test_values("iscx_vpn", 2022, ["Chat"])
                read_features.assert_not_called()
        with self.assertRaises(ValueError):
            experiment.load_known("iscx_vpn", 2022, "unknown_test")

    def test_population_sd_and_seed_pairing(self):
        self.assertEqual(experiment.SEEDS, (2022, 2023))
        self.assertAlmostEqual(float(np.std([.8, .9], ddof=0)), .05)
        self.assertEqual((experiment.EPOCHS, experiment.BATCH), (30, 256))


if __name__ == "__main__":
    torch.set_num_threads(4)
    unittest.main()
