"""Global-grid integrity and statistical regressions using explicit CPU fixtures.

The checkpoint files below are synthetic counter/state fixtures, not trained
models or experimental outcomes. No source dataset, CUDA execution, download,
scientific fit, or external application is used.
"""
import copy
import csv
import itertools
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from vision_only import control, run


class GlobalSealTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "synthetic-work"
        self.root.mkdir()
        self.study = self.root.parent / "study.json"
        self.study.write_text(run.DEFAULT_STUDY.read_text())
        self.spec = run.settings(self.study)
        self.source = {"fixture_only": "no_training_performed"}
        source = patch.object(run, "code_identity", return_value=self.source)
        source.start()
        self.addCleanup(source.stop)
        context = patch.object(run, "context", side_effect=self.fake_context)
        context.start()
        self.addCleanup(context.stop)
        # Construct valid optimizer/model-shaped state, without an optimizer step.
        model = run.AdapterHead(4, 3, self.spec["training"]["adapter_width"])
        model_state = model.state_dict()
        epochs = self.spec["training"]["epochs"]
        self.template = {"completed_epochs": epochs, "completed_updates": epochs,
            "steps_per_epoch": 1, "history": [{"epoch": n, "optimizer_updates": n}
                                                for n in range(1, epochs + 1)],
            "model": model_state, "mean": torch.zeros(4), "sd": torch.ones(4), "width": 4, "classes": 3,
            "optimizer": {"param_groups": [{"params": list(range(len(model_state)))}],
                "state": {index: {"step": torch.tensor(float(epochs)),
                    "exp_avg": torch.zeros_like(tensor), "exp_avg_sq": torch.zeros_like(tensor)}
                    for index, tensor in enumerate(model_state.values())}}}
        self.cells = list(control.contexts(self.root, self.study, "cpu"))
        for _, _, _, args in self.cells:
            identity = self.identity(args)
            selection = {"identity": identity, "lr_by_arm": {arm: .001 for arm in self.spec["arms"]},
                         "confirmation_scored": False, "fixture_only": True}
            selection["fingerprint"] = run.fingerprint(selection)
            run.write_json(args.out / "selection.json", selection)
            for arm, seed in itertools.product(self.spec["arms"], self.spec["training"]["final_seeds"]):
                folder = args.out / "final" / arm / str(seed)
                folder.mkdir(parents=True)
                key = run.fingerprint({"identity": identity, "arm": arm, "seed": seed,
                                       "lr": .001, "stage": "final"})
                state = {**self.template, "fingerprint": key}
                torch.save(state, folder / "checkpoint.pt")
                run.write_json(folder / "trained.json", {"status": "trained", "fingerprint": key,
                    "completed_epochs": epochs, "checkpoint_sha256": run.digest(folder / "checkpoint.pt")})

    def identity(self, args):
        return {"fixture_only": args.out.relative_to(self.root).as_posix(), "fit_rows": 3}

    def fake_context(self, args):
        development = self.spec["development"]
        split = SimpleNamespace(lt_ratio=100. if args.split.stem == "lt100" else None,
            seed=development["split_seed"], tune_fraction=development["tune_fraction"],
            calibration_fraction=development["calibration_fraction"])
        return self.spec, None, split, None, None, None, self.identity(args)

    def last_run(self):
        args = self.cells[-1][3]
        return args.out / "final" / self.spec["arms"][-1] / str(self.spec["training"]["final_seeds"][-1])

    def write_prediction_fixtures(self):
        """Create explicit three-class prediction files for replay-gate tests."""
        labels = np.asarray([0, 1, 2])
        predicted = labels.copy()
        top5 = np.asarray([[0, 1, 2], [1, 0, 2], [2, 0, 1]])
        metrics = run.class_metrics(labels, predicted, 3, [0, 1, 2], np.ones(3, dtype=bool))
        for _, _, _, args in self.cells:
            run.write_json(args.split, {"class_counts": {"fit": [1, 1, 1]}, "class_order": [0, 1, 2]})
            selected = run.read_json(args.out / "selection.json")
            for arm, seed in itertools.product(self.spec["arms"], self.spec["training"]["final_seeds"]):
                cell = args.out / "final" / arm / str(seed)
                np.savez_compressed(cell / "predictions.npz", row_indices=np.arange(3), labels=labels,
                    raw_predictions=predicted, adjusted_predictions=predicted,
                    raw_top5=top5, adjusted_top5=top5)
                np.savez_compressed(cell / "calibration_predictions.npz", row_indices=np.arange(3), labels=labels,
                                    tau_grid=np.asarray([0.]), predictions_by_tau=predicted[None, :])
                run.write_json(cell / "confirmed.json", {"status": "complete",
                    "trained_fingerprint": run.read_json(cell / "trained.json")["fingerprint"],
                    "selection_fingerprint": selected["fingerprint"], "arm": arm, "seed": seed,
                    "tau": 0., "calibration_grid": [0.], "calibration_scores": [1.],
                    "raw": metrics, "adjusted": metrics,
                    "predictions_sha256": run.digest(cell / "predictions.npz"),
                    "calibration_predictions_sha256": run.digest(cell / "calibration_predictions.npz")})

    def test_declared_grid_contains_all_final_ids_and_10_contexts(self):
        expected = control.expected_run_ids(self.study)
        size = len(control.REGIMES) * len(self.spec["encoders"]) * len(self.spec["arms"]) * len(self.spec["training"]["final_seeds"])
        self.assertEqual(size, 720)
        self.assertEqual(len(expected), size)
        self.assertEqual(len(set(expected)), size)
        self.assertEqual(len(self.cells), 10)
        seal = control.global_seal(self.root, self.study, "cpu", write=True)
        self.assertEqual(seal["completed_final_fits"], size)
        self.assertEqual(len(seal["cells"]), 10)
        self.assertEqual({(cell["dataset"], cell["regime"], cell["encoder"]) for cell in seal["cells"]},
                         {(d, r, e) for d, r, e, _ in self.cells})
        self.assertEqual(control.global_seal(self.root, self.study, "cpu"), seal)

    def test_missing_last_cross_encoder_checkpoint_blocks_global_seal(self):
        (self.last_run() / "checkpoint.pt").unlink()
        with self.assertRaises((ValueError, OSError)):
            control.global_seal(self.root, self.study, "cpu", write=True)
        self.assertFalse((self.root / "global_confirmation_seal.json").exists())

    def test_global_seal_rejects_split_that_does_not_match_directory_condition(self):
        def wrong_context(args):
            values = list(self.fake_context(args))
            values[2].lt_ratio = 10.
            return tuple(values)
        with patch.object(run, "context", side_effect=wrong_context):
            with self.assertRaisesRegex(ValueError, "declared condition"):
                control.global_seal(self.root, self.study, "cpu", write=True)
        self.assertFalse((self.root / "global_confirmation_seal.json").exists())

    def test_false_sidecar_completion_cannot_hide_short_actual_checkpoint(self):
        folder = self.last_run()
        state = torch.load(folder / "checkpoint.pt", weights_only=True)
        state["completed_epochs"] -= 1
        state["completed_updates"] -= 1
        state["history"].pop()
        for item in state["optimizer"]["state"].values():
            item["step"] -= 1
        torch.save(state, folder / "checkpoint.pt")
        record = run.read_json(folder / "trained.json")
        record["checkpoint_sha256"] = run.digest(folder / "checkpoint.pt")
        run.write_json(folder / "trained.json", record)
        with self.assertRaisesRegex(ValueError, "training budget incomplete"):
            control.global_seal(self.root, self.study, "cpu", write=True)
        self.assertFalse((self.root / "global_confirmation_seal.json").exists())

    def test_post_seal_checkpoint_change_fails_even_with_refreshed_sidecar_hash(self):
        control.global_seal(self.root, self.study, "cpu", write=True)
        folder = self.last_run()
        state = torch.load(folder / "checkpoint.pt", weights_only=True)
        state["model"][next(iter(state["model"]))].flatten()[0] += 1
        torch.save(state, folder / "checkpoint.pt")
        record = run.read_json(folder / "trained.json")
        record["checkpoint_sha256"] = run.digest(folder / "checkpoint.pt")
        run.write_json(folder / "trained.json", record)
        with self.assertRaisesRegex(ValueError, "preconfirmation seal changed"):
            control.global_seal(self.root, self.study, "cpu")

    def test_post_seal_selection_change_and_missing_seal_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "seal is missing"):
            control.global_seal(self.root, self.study, "cpu")
        control.global_seal(self.root, self.study, "cpu", write=True)
        selection_path = self.cells[0][3].out / "selection.json"
        selected = run.read_json(selection_path)
        selected["fixture_only"] = "changed after global freeze"
        selected.pop("fingerprint")
        selected["fingerprint"] = run.fingerprint(selected)
        run.write_json(selection_path, selected)
        with self.assertRaisesRegex(ValueError, "preconfirmation seal changed"):
            control.global_seal(self.root, self.study, "cpu")

    def test_direct_confirmation_seal_verifier_checks_other_context_files(self):
        control.global_seal(self.root, self.study, "cpu", write=True)
        path = self.root / "global_confirmation_seal.json"
        current_identity = self.identity(self.cells[0][3])
        run.verify_global_seal(path, self.study, current_identity)
        # The last run belongs to a different dataset/encoder context. A direct
        # single-context confirmation must still notice that global modification.
        with (self.last_run() / "checkpoint.pt").open("ab") as stream:
            stream.write(b"changed after global freeze")
        with self.assertRaisesRegex(ValueError, "checkpoint changed after global freeze"):
            run.verify_global_seal(path, self.study, current_identity)

    def test_direct_confirmation_seal_verifier_rejects_duplicate_full_count_grid(self):
        seal = control.global_seal(self.root, self.study, "cpu", write=True)
        duplicate = copy.deepcopy(seal)
        duplicate["cells"][-1]["runs"][-1] = copy.deepcopy(duplicate["cells"][-1]["runs"][0])
        duplicate.pop("fingerprint")
        duplicate["fingerprint"] = run.fingerprint(duplicate)
        path = self.root / "global_confirmation_seal.json"
        run.write_json(path, duplicate)
        with self.assertRaisesRegex(ValueError, "Incomplete or repeated"):
            run.verify_global_seal(path, self.study, self.identity(self.cells[0][3]))

    def test_analysis_replays_all_synthetic_predictions_before_aggregate_receipt(self):
        control.global_seal(self.root, self.study, "cpu", write=True)
        self.write_prediction_fixtures()
        args = type("AnalysisArgs", (), {"work": self.root, "study": self.study, "device": "cpu"})()
        control.analyze(args)
        receipt = run.read_json(self.root / "aggregate" / "complete.json")
        self.assertEqual(receipt["completed_final_fits"], 720)
        self.assertEqual(receipt["primary_comparisons"], 30)
        with (self.root / "aggregate" / "per_class.csv").open(newline="") as stream:
            self.assertEqual(sum(1 for _ in csv.DictReader(stream)), 720 * 2 * 3)
        comparisons = run.read_json(self.root / "aggregate" / "primary_comparisons.json")
        self.assertTrue(all(item["effect_pp"] == 0 and item["p_exact"] == 1 for item in comparisons))

    def test_analysis_rejects_self_reported_calibration_or_metric_changes(self):
        control.global_seal(self.root, self.study, "cpu", write=True)
        self.write_prediction_fixtures()
        args = type("AnalysisArgs", (), {"work": self.root, "study": self.study, "device": "cpu"})()
        first = self.cells[0][3].out / "final" / self.spec["arms"][0] / str(self.spec["training"]["final_seeds"][0])
        original = run.read_json(first / "confirmed.json")
        for field, expected in (("calibration_scores", "Decision-offset selection"), ("raw", "Reported metrics differ")):
            changed = copy.deepcopy(original)
            if field == "calibration_scores":
                changed[field] = [0.]
            else:
                changed[field]["macro_f1"] = 0.
            run.write_json(first / "confirmed.json", changed)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, expected):
                control.analyze(args)
            self.assertFalse((self.root / "aggregate" / "complete.json").exists())
            run.write_json(first / "confirmed.json", original)


class StatisticalGateTests(unittest.TestCase):
    def test_paired_effect_sign_scale_and_zero_effect_are_preserved(self):
        values = np.asarray([.01, .02, .03, .04])
        positive = control.paired_summary(values)
        negative = control.paired_summary(-values)
        self.assertAlmostEqual(positive["effect_pp"], 2.5)
        self.assertAlmostEqual(negative["effect_pp"], -2.5)
        self.assertEqual(positive["p_exact"], negative["p_exact"])
        self.assertAlmostEqual(positive["ci_low_pp"], -negative["ci_high_pp"])
        zero = control.paired_summary(np.zeros(10))
        self.assertEqual(zero["p_exact"], 1.)
        self.assertEqual(zero["effect_pp"], 0.)
        self.assertEqual(zero["ci_low_pp"], 0.)
        self.assertEqual(zero["ci_high_pp"], 0.)

    def test_ten_seed_exact_resolution_prevents_30_test_family_rejection(self):
        # This is a mathematical fixture, never an empirical significance claim.
        p = control.paired_summary(np.ones(10))["p_exact"]
        self.assertEqual(p, 2 / 1024)
        self.assertLess(control.holm([p] * 6)[0], .05)
        combined = control.holm([p] * 30)
        self.assertTrue(all(value > .05 for value in combined))
        self.assertAlmostEqual(combined[0], .05859375)

    def test_declared_twelve_seed_grid_has_sufficient_combined_family_resolution(self):
        count = len(run.settings(run.DEFAULT_STUDY)["training"]["final_seeds"])
        self.assertEqual(count, 12)
        p = control.paired_summary(np.ones(count))["p_exact"]
        self.assertEqual(p, 2 / (2 ** count))
        adjusted = control.holm([p] * 30)
        self.assertAlmostEqual(adjusted[0], 30 * 2 / 4096)
        self.assertLess(adjusted[0], .05)

    def test_holm_retains_input_order_and_stepdown_monotonicity(self):
        values = [.04, .001, .03, .02]
        adjusted = control.holm(values)
        np.testing.assert_allclose(adjusted, [.06, .004, .06, .06])
        order = np.argsort(values)
        self.assertTrue(np.all(np.diff(np.asarray(adjusted)[order]) >= 0))

    def test_nonfinite_paired_data_and_invalid_p_values_fail_closed(self):
        for differences in ([.1], [np.nan, .1], [np.inf, .1]):
            with self.subTest(differences=differences), self.assertRaises(ValueError):
                control.paired_summary(differences)
        for p_values in ([np.nan, .1], [np.inf], [-.1, .2], [1.1], [[.1, .2]]):
            with self.subTest(p_values=p_values), self.assertRaises(ValueError):
                control.holm(p_values)


if __name__ == "__main__":
    unittest.main()
