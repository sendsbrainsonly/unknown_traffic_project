#!/usr/bin/env python3
"""CPU regression tests: prefix selection, identity RNG, Test barrier, optimizers."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import TensorDataset

import experiment as e


class Contracts(unittest.TestCase):
    def test_prefix_never_selects_future_or_later_tie(self):
        events = [{"epoch": 1, "val_macro_f1": .2}, {"epoch": 20, "val_macro_f1": .7},
                  {"epoch": 30, "val_macro_f1": .7}, {"epoch": 51, "val_macro_f1": .9}]
        self.assertEqual(e.select_best(events, 20)["epoch"], 20)
        self.assertEqual(e.select_best(events, 50)["epoch"], 20)
        self.assertEqual(e.select_best(events, 100)["epoch"], 51)
        with self.assertRaises(ValueError):
            e.select_best(events, 0)

    def test_identity_wrapper_preserves_sampler_order(self):
        original = TensorDataset(torch.arange(103))
        a = e.joint.load_batches(original, 32, True)
        b = e.joint.load_batches(e.Indexed(original), 32, True)
        for _ in range(3):
            seen_a = torch.cat([batch[0] for batch in a]).tolist()
            seen_b, ids = [], []
            for batch, indices in b:
                seen_b.extend(batch[0].tolist()); ids.extend(indices.tolist())
            self.assertEqual(seen_a, seen_b)
            self.assertEqual(seen_b, ids)
            self.assertEqual(sorted(ids), list(range(103)))

    def test_test_blocked_without_six_checkpoint_gate(self):
        with tempfile.TemporaryDirectory(dir=e.ROOT, prefix="test_gate_") as folder:
            with patch.object(e, "ROOT", Path(folder)), patch.object(e.source, "Flows") as loader:
                with self.assertRaises(FileNotFoundError):
                    e.dataset_for("iscx_vpn", "known_test", np.zeros(7), np.ones(7))
                loader.assert_not_called()
                with self.assertRaises(ValueError):
                    e.dataset_for("iscx_vpn", "unknown_test", np.zeros(7), np.ones(7))

    def test_registered_schedule_and_last_partial_batch(self):
        for dataset in e.DATASETS:
            n = e.source.EXPECTED[dataset][0]
            steps = (n + 63) // 64 * 100
            args = e.joint.tf_branch.source.make_tf_args(len(e.source.CLASSES[dataset]), steps)
            self.assertEqual((args.optimizer, args.scheduler, args.warmup, args.learning_rate), ("adamw", "linear", .1, 6e-5))
            self.assertEqual(args.train_steps, steps)
            self.assertEqual(sum(min(64, n-i) for i in range(0, n, 64)), n)
        opt = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))], lr=5e-4)
        e.joint.set_yatc_lr(opt, 100, 1000)
        self.assertAlmostEqual(opt.param_groups[0]["lr"], 5e-4)
        e.joint.set_yatc_lr(opt, 1000, 1000)
        self.assertAlmostEqual(opt.param_groups[0]["lr"], 1e-6)


if __name__ == "__main__":
    unittest.main()
