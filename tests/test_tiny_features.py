"""Protocol tests: holdout isolation, pairing, metrics and auditable resume."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
import torch

MODULE = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_tiny_features.py"
spec = importlib.util.spec_from_file_location("tiny_benchmark", MODULE)
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)


class ProtocolTests(unittest.TestCase):
    def test_holdout_disjoint_nested_and_semantics_randomized(self):
        y = np.repeat(np.arange(200), 500)
        splits = [b.make_split(y, 3, rho) for rho in (10, 50, 100)]
        for split in splits:
            self.assertEqual(len(split["holdout_indices"]), 10000)
            self.assertEqual(np.intersect1d(split["train_indices"], split["holdout_indices"]).size, 0)
            self.assertEqual(len(set(split["train_indices"])), len(split["train_indices"]))
            self.assertTrue(np.all(np.bincount(y[split["holdout_indices"]]) == 50))
        for left, right in zip(splits, splits[1:]):
            np.testing.assert_array_equal(left["holdout_indices"], right["holdout_indices"])
            np.testing.assert_array_equal(left["class_rank"], right["class_rank"])
            self.assertTrue(set(right["train_indices"]).issubset(left["train_indices"]))
        self.assertEqual(splits[-1]["train_counts"].min(), 4)
        self.assertEqual(splits[-1]["train_counts"].max(), 450)
        self.assertFalse(np.array_equal(splits[0]["class_rank"], np.arange(200)))
        self.assertFalse(np.array_equal(splits[0]["class_rank"], b.make_split(y, 4, 10)["class_rank"]))
        np.testing.assert_array_equal(splits[0]["train_indices"], b.make_split(y, 3, 10)["train_indices"])

    def test_metrics_use_randomized_rank_and_zero_division(self):
        y = np.array([0, 0, 1, 1, 2, 2])
        pred = np.array([0, 0, 0, 0, 2, 2])
        rank = np.array([2, 0, 1])
        result = b.measure(y, pred, rank)
        self.assertAlmostEqual(result["accuracy"], 2 / 3)
        self.assertAlmostEqual(result["macro_recall"], 2 / 3)
        self.assertAlmostEqual(result["head_f1"], 0.)
        self.assertAlmostEqual(result["mid_f1"], 1.)
        self.assertAlmostEqual(result["tail_f1"], 2 / 3)
        self.assertEqual(result["zero_f1_classes"], 1)
        self.assertEqual([len(v) for v in b.class_regions(np.arange(200)).values()], [66, 67, 67])

    def test_dedup_cap_and_pixel_partition_isolation(self):
        y = np.repeat(np.arange(200), 500)
        hashes = [f"train-{i}" for i in range(len(y))]
        eval_hashes = [f"eval-{i}" for i in range(10000)]
        hashes[0] = eval_hashes[0]
        hashes[1:5] = [hashes[5]] * 4
        manifest = dict(train_pixel_sha256=hashes, eval_pixel_sha256=eval_hashes,
                        train_files=[str(i) for i in range(len(y))],
                        eval_files=[str(i) for i in range(10000)],
                        classes=[f"n{i:08}" for i in range(200)])
        eligible, audit = b.clean_training_indices(manifest, y)
        self.assertEqual(audit["max_train_per_class"], 445)
        self.assertEqual(audit["retained_examples"], 99995)
        self.assertEqual(audit["train_eval_overlap_excluded"], [0])
        self.assertEqual(len(audit["duplicate_train_excluded"]), 4)
        split = b.make_split(y, 2, 100, eligible_indices=eligible)
        self.assertEqual(split["train_counts"].max(), 445)
        self.assertEqual(split["train_counts"].min(), 4)
        self.assertEqual(len(split["hygiene_excluded_indices"]), 5)
        fit = set(np.asarray(hashes)[split["train_indices"]])
        holdout = set(np.asarray(hashes)[split["holdout_indices"]])
        self.assertTrue(fit.isdisjoint(holdout))
        self.assertTrue(fit.isdisjoint(eval_hashes))
        self.assertTrue(holdout.isdisjoint(eval_hashes))

    def test_tau_uses_holdout_and_breaks_ties_low(self):
        logits = np.array([[3., 0.], [2., 1.]], dtype=np.float32)
        y = np.array([0, 1])
        tau, curve, prior = b.choose_tau(logits, y, np.array([100, 1]))
        self.assertEqual(tau, .25)
        self.assertEqual(curve[0]["macro_recall"], .5)
        self.assertEqual(curve[1]["macro_recall"], 1.)
        np.testing.assert_array_equal((logits - tau * prior).argmax(1), y)

    def test_resampling_only_uses_supplied_train_and_same_target(self):
        x = np.arange(40, dtype=np.float32).reshape(10, 4)
        y = np.array([0] * 6 + [1] * 4)
        ros_x, ros_y, ros_meta, indices = b.resample_pool(x, y, "ros", 2)
        smote_x, smote_y, meta, _ = b.resample_pool(x, y, "smote", 2)
        np.testing.assert_array_equal(np.bincount(ros_y), [6, 6])
        np.testing.assert_array_equal(np.bincount(smote_y), [6, 6])
        np.testing.assert_array_equal(ros_x, x[indices])
        np.testing.assert_array_equal(smote_x[:10], x)
        self.assertEqual(meta["k_neighbors"], 3)
        self.assertEqual(ros_meta["added_examples"], 2)
        # Minority interpolation stays inside its input class's bounding box.
        self.assertTrue((smote_x[10:] >= x[y == 1].min(0)).all())
        self.assertTrue((smote_x[10:] <= x[y == 1].max(0)).all())

    def test_ncm_matches_original_cosine_rule(self):
        x = np.array([[2., 0.], [20., 0.], [0., 4.], [0., 1.]], dtype=np.float32)
        c = b.ncm_centroids(x, np.array([0, 0, 1, 1]), 2)
        np.testing.assert_allclose(c, np.eye(2))
        logits = b.ncm_logits(c, np.array([[3., 4.]], dtype=np.float32))
        np.testing.assert_allclose(logits, [[.6, .8]])

    def test_matched_initialization_updates_and_deterministic_repeat(self):
        torch.set_num_threads(1)
        x = np.random.default_rng(0).normal(size=(15, 4)).astype(np.float32)
        y = np.repeat(np.arange(3), 5)
        args = SimpleNamespace(device="cpu", lr=.01, steps=4, batch_size=5)
        one, log_one = b.train_head(x, y, np.array([5, 5, 5]), 3, args)
        two, log_two = b.train_head(x, y, np.array([5, 5, 5]), 3, args)
        balanced_cb, log_cb = b.train_head(x, y, np.array([5, 5, 5]), 3, args, weighted=True)
        np.testing.assert_array_equal(one.weight.detach(), two.weight.detach())
        np.testing.assert_allclose(one.weight.detach(), balanced_cb.weight.detach(), atol=1e-7)
        self.assertEqual(log_one["optimizer_updates"], 4)
        self.assertEqual(log_two["sampled_examples"], 20)
        self.assertEqual(log_cb["losses"][-1]["update"], 4)

    def test_resume_rejects_changed_protocol_and_artifact(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            b.write_npz(root / "split.npz", x=np.arange(3))
            record = dict(fingerprint="expected", arms={a: {} for a in b.ARMS},
                          artifacts={"split.npz": b.sha256_file(root / "split.npz")})
            b.write_json(root / "metrics.json", record)
            self.assertTrue(b.result_valid(root / "metrics.json", "expected"))
            with self.assertRaisesRegex(ValueError, "Protocol mismatch"):
                b.result_valid(root / "metrics.json", "different")
            b.write_npz(root / "split.npz", x=np.arange(4))
            with self.assertRaisesRegex(ValueError, "changed artifact"):
                b.result_valid(root / "metrics.json", "expected")

    def test_paired_intervals_and_holm_family(self):
        out = b.paired_comparison([.2, .4, .6], [.1, .2, .3])
        self.assertAlmostEqual(out["mean"], .2)
        self.assertAlmostEqual(out["std"], .1)
        self.assertLess(out["ci95"][0], out["mean"])
        np.testing.assert_allclose(b.holm_adjust([.01, .04, .03]), [.03, .06, .06])
        self.assertIsNone(b.paired_comparison([.2], [.1])["p_two_sided"])
        self.assertEqual(b.paired_comparison([.2, .2], [.2, .2])["p_two_sided"], 1.)
        self.assertIsNone(b.paired_comparison([.3, .3], [.2, .2])["p_two_sided"])
        self.assertEqual(b.paired_comparison([1.] * 5, [0.] * 5)["p_signflip_two_sided"], .0625)

    def test_small_complete_cell_saves_tuning_evidence_and_resumes(self):
        rng = np.random.default_rng(12)
        data = dict(train_features=rng.normal(size=(180, 4)).astype(np.float32),
                    train_labels=np.repeat(np.arange(3), 60),
                    eval_features=rng.normal(size=(6, 4)).astype(np.float32),
                    eval_labels=np.repeat(np.arange(3), 2), eligible_indices=np.arange(180),
                    train_pixel_hashes=np.array([f"train-{i}" for i in range(180)]),
                    eval_pixel_hashes=np.array([f"eval-{i}" for i in range(6)]))
        with tempfile.TemporaryDirectory() as td:
            args = SimpleNamespace(out=Path(td), device="cpu", lr=.01, steps=2, batch_size=8)
            record = b.run_cell(data, {}, 0, 2., args)
            self.assertEqual(set(record["arms"]), set(b.ARMS))
            self.assertEqual(record["holdout_examples"], 150)
            self.assertEqual(len(record["arms"]["ce_la"]["tau_curve"]), 8)
            self.assertEqual(record["arms"]["smote"]["training"]["optimizer_updates"], 2)
            self.assertEqual(record["arms"]["ros"]["training"]["sampled_examples"], 16)
            same = b.run_cell(data, {}, 0, 2., args)
            self.assertEqual(record, same)
            summary = b.aggregate([record], Path(td), [0], [2.])
            self.assertTrue(summary["complete"])
            self.assertEqual(len(summary["comparisons"]), 2)
            self.assertTrue(all(c["p_holm"] is None for c in summary["comparisons"]))


if __name__ == "__main__":
    unittest.main()
