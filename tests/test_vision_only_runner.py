"""CPU mathematical/integrity fixtures only; no scientific training or data."""
import json
import hashlib
from pathlib import Path
import copy
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from vision_only import run


class VisionRunnerMathTests(unittest.TestCase):
    def test_eq_is_mean_compactness_minus_mean_pair_distance(self):
        h = torch.tensor([[1., 0.], [1., 0.], [-1., 0.], [-1., 0.]], requires_grad=True)
        y = torch.tensor([0, 0, 1, 1])
        loss, diagnostics = run.equalization(h, y)
        self.assertAlmostEqual(float(loss.detach()), -4.)
        self.assertEqual(diagnostics["eligible_classes"], 2)
        loss.backward()
        self.assertTrue(torch.isfinite(h.grad).all())
        self.assertGreater(float(h.grad.abs().sum()), 0.)

    def test_eq_singletons_are_ineligible_and_empty_gate_has_zero_gradient(self):
        h = torch.tensor([[0., 0.], [2., 0.], [100., 100.]], requires_grad=True)
        loss, diagnostics = run.equalization(h, torch.tensor([0, 0, 1]))
        self.assertAlmostEqual(float(loss.detach()), 1.)
        self.assertEqual(diagnostics["eligible_classes"], 1)
        loss.backward()
        self.assertTrue(torch.equal(h.grad[2], torch.zeros(2)))
        singleton = torch.randn(3, 4, requires_grad=True)
        zero, diagnostics = run.equalization(singleton, torch.tensor([0, 1, 2]))
        zero.backward()
        self.assertEqual(float(zero.detach()), 0.)
        self.assertTrue(torch.equal(singleton.grad, torch.zeros_like(singleton)))
        self.assertEqual(diagnostics["eligible_classes"], 0)

    def test_natural_importance_weighting_and_uniform_sampling_share_expected_loss_and_gradient(self):
        y = torch.tensor([0, 0, 0, 1])
        counts = torch.tensor([3., 1.])
        first = torch.tensor([[1., -.3], [.2, -.8], [1.4, -.6], [-.5, .6]], requires_grad=True)
        natural = run.classification_loss(first, y, counts, "natural_balanced_eq")
        natural_grad, = torch.autograd.grad(natural, first)
        second = first.detach().clone().requires_grad_(True)
        per_row = F.cross_entropy(second, y, reduction="none")
        uniform_expectation = (per_row * (1. / (2 * counts[y]))).sum()
        uniform_grad, = torch.autograd.grad(uniform_expectation, second)
        self.assertTrue(torch.allclose(natural, uniform_expectation, atol=1e-7))
        self.assertTrue(torch.allclose(natural_grad, uniform_grad, atol=1e-7))
        balanced_softmax = run.classification_loss(first, y, counts, "balanced_softmax")
        self.assertTrue(torch.allclose(balanced_softmax, F.cross_entropy(first + counts.log(), y)))

    def test_sampler_pairing_and_generator_resume_are_exact(self):
        labels = torch.tensor([0] * 9 + [1])
        counts = torch.tensor([9, 1])
        for off, on in (("natural_balanced_off", "natural_balanced_eq"), ("uniform_off", "uniform_eq")):
            first = torch.Generator().manual_seed(11)
            second = torch.Generator().manual_seed(11)
            self.assertTrue(torch.equal(run.batch_ids(counts, labels, 100, off, first),
                                        run.batch_ids(counts, labels, 100, on, second)))
            saved = first.get_state()
            expected = run.batch_ids(counts, labels, 100, off, first)
            restored = torch.Generator()
            restored.set_state(saved)
            self.assertTrue(torch.equal(expected, run.batch_ids(counts, labels, 100, off, restored)))
        uniform = run.batch_ids(counts, labels, 20000, "uniform_off", torch.Generator().manual_seed(71))
        fraction = float(labels[uniform].float().mean())
        self.assertGreater(fraction, .48)
        self.assertLess(fraction, .52)

    def test_eq_changes_adapter_gradient_when_it_has_a_nonzero_signal(self):
        run.seed_everything(42)
        model = run.AdapterHead(4, 2, hidden=3)
        features = torch.randn(6, 4)
        labels = torch.tensor([0, 0, 0, 1, 1, 1])
        logits, h = model(features)
        ce = F.cross_entropy(logits, labels)
        eq, _ = run.equalization(h, labels)
        params = tuple(model.adapter.parameters())
        reference = torch.autograd.grad(ce, params, retain_graph=True)
        treatment = torch.autograd.grad(ce + .05 * eq, params)
        self.assertTrue(any(not torch.allclose(a, b, atol=1e-8) for a, b in zip(reference, treatment)))

    def test_initialization_is_seeded_and_feature_scaler_uses_only_fit_rows(self):
        run.seed_everything(10)
        first = run.AdapterHead(4, 3)
        run.seed_everything(10)
        second = run.AdapterHead(4, 3)
        self.assertTrue(all(torch.equal(v, second.state_dict()[k]) for k, v in first.state_dict().items()))
        features = np.asarray([[0., 2.], [2., 4.], [1.e9, -1.e9]], dtype=np.float32)
        mean, sd = run.feature_scaler(features, np.asarray([0, 1]))
        np.testing.assert_allclose(mean, [1., 3.])
        np.testing.assert_allclose(sd, [1., 1.])

    def test_all_scientific_cli_phases_require_cuda_before_other_work(self):
        args = SimpleNamespace(device="cpu")
        for phase in (run.extract, run.tune, run.train, run.confirm):
            with self.subTest(phase=phase.__name__):
                with patch.object(run, "require_cuda", side_effect=ValueError("CUDA gate")):
                    with self.assertRaisesRegex(ValueError, "CUDA gate"):
                        phase(args)
        with patch.object(torch.cuda, "is_available", return_value=True):
            for device in ("cpu", "mps"):
                with self.assertRaisesRegex(ValueError, "remote CUDA"):
                    run.require_cuda(device)


class VisionRunnerIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_previously_scored_selection_cannot_be_used_as_unopened_selection(self):
        identity = {"fixture": "identity"}
        selection = {"identity": identity, "lr_by_arm": {"natural_ce": .001},
                     "confirmation_scored": True}
        selection["fingerprint"] = run.fingerprint(selection)
        run.write_json(self.root / "selection.json", selection)
        with self.assertRaises(ValueError):
            run.selection_record(self.root, identity)

    def test_confirmation_gate_inspects_actual_checkpoint_budget(self):
        identity = {"fixture": "identity", "fit_rows": 2}
        arm, seed, lr = "natural_ce", 10, .001
        key = run.fingerprint({"identity": identity, "arm": arm, "seed": seed, "lr": lr, "stage": "final"})
        folder = self.root / "final" / arm / str(seed)
        folder.mkdir(parents=True)
        checkpoint = folder / "checkpoint.pt"
        # A deliberately short unit checkpoint with an inflated sidecar budget.
        torch.save({"fingerprint": key, "completed_epochs": 1, "model": {"weight": torch.ones(1)},
                    "optimizer": {"state": {0: {"step": torch.tensor(1.)}}},
                    "steps_per_epoch": 1, "completed_updates": 1,
                    "history": [{"epoch": 1, "optimizer_updates": 1}]}, checkpoint)
        run.write_json(folder / "trained.json", {"status": "trained", "fingerprint": key,
                       "completed_epochs": 50, "checkpoint_sha256": run.digest(checkpoint)})
        spec = {"arms": [arm], "training": {"final_seeds": [seed], "epochs": 50, "batch_size": 2}}
        selected = {"lr_by_arm": {arm: lr}}
        with self.assertRaises(ValueError):
            run.confirmation_gate(self.root, spec, identity, selected)

    def test_context_rejects_wrong_encoder_package_or_transform(self):
        # Mock the already separately tested full-source loader, isolating the
        # cache compatibility check; no full-source dataset or scores are made.
        samples = tuple(SimpleNamespace(sample_id=str(i), label=i % 2) for i in range(4))
        manifest = SimpleNamespace(dataset_id="imagenet1k", scope="full", samples=samples,
                                   fingerprint="manifest", num_classes=2)
        split = SimpleNamespace(fit=("0",), tune=("1",), calibration=("2",), confirmation=("3",),
                                fingerprint="split")
        spec = {"datasets": {"imagenet1k": {}},
                "encoders": {"deit_small": "expected_model"},
                "transform": {"crop": 224}}
        cache = self.root / "cache"
        cache.mkdir()
        np.save(cache / "features.npy", np.zeros((4, 384), dtype=np.float16))
        args = SimpleNamespace(study=self.root / "study.json", manifest=self.root / "manifest.json",
                               split=self.root / "split.json", cache=cache, encoder="deit_small")
        args.study.write_text("{}")
        receipt = {"manifest_fingerprint": "manifest", "encoder": "deit_small", "rows": 4,
                   "features_sha256": run.digest(cache / "features.npy"),
                   "model_name": "wrong_model", "transform": {"crop": 32}}
        run.write_json(cache / "complete.json", receipt)
        with patch.object(run, "settings", return_value=spec), \
             patch.object(run.data, "load_manifest", return_value=manifest), \
             patch.object(run.data, "load_partitions", return_value=split), \
             patch.object(run.data, "validate_partitions"):
            with self.assertRaises(ValueError):
                run.context(args)

    def test_complete_checkpoint_cannot_have_no_optimizer_state(self):
        state = {"fingerprint": "fixture", "completed_epochs": 2, "completed_updates": 4,
                 "steps_per_epoch": 2, "history": [{"epoch": 1}, {"epoch": 2}],
                 "model": {"weight": torch.ones(1)},
                 "optimizer": {"state": {}, "param_groups": [{"params": [0]}]},
                 "mean": torch.zeros(1), "sd": torch.ones(1)}
        with self.assertRaises(ValueError):
            run.validate_checkpoint(state, "fixture", 2, 2, complete=True)

    def test_checkpoint_audit_accepts_valid_unit_state_and_rejects_corrupt_history_moments_or_values(self):
        model = run.AdapterHead(4, 2, hidden=3)
        tensors = model.state_dict()
        optimizer = {"param_groups": [{"params": list(range(len(tensors)))}], "state": {
            index: {"step": torch.tensor(4.), "exp_avg": torch.zeros_like(tensor),
                    "exp_avg_sq": torch.zeros_like(tensor)}
            for index, tensor in enumerate(tensors.values())}}
        state = {"fingerprint": "fixture", "completed_epochs": 2, "completed_updates": 4,
                 "steps_per_epoch": 2, "history": [{"epoch": 1}, {"epoch": 2}],
                 "model": tensors, "optimizer": optimizer, "mean": torch.zeros(4), "sd": torch.ones(4)}
        run.validate_checkpoint(state, "fixture", 2, 2, complete=True)
        for mutation in ("history", "step", "moment_shape", "nonfinite"):
            bad = copy.deepcopy(state)
            if mutation == "history":
                bad["history"][1]["epoch"] = 3
            elif mutation == "step":
                bad["optimizer"]["state"][0]["step"] = torch.tensor(3.)
            elif mutation == "moment_shape":
                bad["optimizer"]["state"][0]["exp_avg"] = torch.zeros(1)
            else:
                bad["model"][next(iter(tensors))].flatten()[0] = float("nan")
            with self.subTest(mutation=mutation):
                with self.assertRaises(ValueError):
                    run.validate_checkpoint(bad, "fixture", 2, 2, complete=True)

    def test_extraction_resume_checks_committed_blocks_but_not_uncommitted_suffix(self):
        values = np.zeros((4, 384), dtype=np.float16)
        identity = {"fixture": "encoder/source"}
        progress = {"identity": identity, "completed_rows": 2, "blocks": [
            {"start": 0, "end": 2, "sha256": hashlib.sha256(values[:2].tobytes()).hexdigest()}]}
        run.validate_progress(values, progress, identity, 4)
        values[3, 0] = np.nan  # incomplete suffix will be overwritten after restart
        run.validate_progress(values, progress, identity, 4)
        values[0, 0] = 1.
        with self.assertRaisesRegex(ValueError, "committed feature block"):
            run.validate_progress(values, progress, identity, 4)

    def test_extraction_resume_rejects_gaps_ranges_dtypes_and_identity_changes(self):
        values = np.zeros((4, 384), dtype=np.float16)
        identity = {"fixture": "encoder/source"}
        good = {"identity": identity, "completed_rows": 2, "blocks": [
            {"start": 0, "end": 2, "sha256": hashlib.sha256(values[:2].tobytes()).hexdigest()}]}
        bad_cases = [
            {**good, "completed_rows": 5}, {**good, "completed_rows": -1},
            {**good, "identity": {"fixture": "different source"}}, {**good, "blocks": []},
            {**good, "blocks": [{**good["blocks"][0], "start": 1}]},
        ]
        for progress in bad_cases:
            with self.subTest(progress=progress):
                with self.assertRaises(ValueError):
                    run.validate_progress(values, progress, identity, 4)
        with self.assertRaisesRegex(ValueError, "layout"):
            run.validate_progress(values.astype(np.float32), good, identity, 4)


if __name__ == "__main__":
    unittest.main()
