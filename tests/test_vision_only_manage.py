"""Synthetic controller/storage bridge tests. No scientific fits or real data."""
import json
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

from vision_only import control, lifecycle, manage, run


class StudyManagementTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name).resolve()
        self.work = self.base / "owned"
        self.study = self.base / "study.json"
        self.study.write_text(run.DEFAULT_STUDY.read_text())
        self.archive = self.base / "retained" / "fixture.zip"

    def initialize(self):
        return manage.initialize(self.work, self.study)

    def artifact(self, relative, value):
        path = self.work / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
        return path

    def completed_fixture(self):
        self.initialize()
        self.artifact("data/owned.bin", b"synthetic owned source")
        self.artifact("cache/features.bin", b"synthetic cache")
        self.artifact("runs/fixture/trained.json", b'{"status":"trained"}')
        self.artifact("runs/fixture/confirmed.json", b'{"status":"complete"}')
        self.artifact("runs/fixture/predictions.npz", b"synthetic predictions, not an NPZ input")
        self.artifact("runs/fixture/calibration_predictions.npz", b"synthetic calibration")
        self.artifact("runs/fixture/checkpoint.pt", b"synthetic state, not a PyTorch input")
        artifacts = {}
        for name in ("per_seed.csv", "per_class.csv", "primary_comparisons.json"):
            path = self.artifact("aggregate/" + name, b"[]" if name.endswith("json") else b"fixture_only\n")
            artifacts[name] = run.digest(path)
        seal = {"fingerprint": "synthetic-verified-seal", "completed_final_fits": 720}
        run.write_json(self.work / "aggregate/complete.json", {"status": "complete",
            "completed_final_fits": 720, "primary_comparisons": 30,
            "global_seal_fingerprint": seal["fingerprint"], "artifacts": artifacts})
        evidence = ["runs/fixture/" + name for name in
            ("trained.json", "confirmed.json", "predictions.npz", "calibration_predictions.npz", "checkpoint.pt")]
        # The actual scientific verifier is tested separately; these explicit
        # mocks isolate ownership/export transaction behavior at the full ID grid.
        completion = {name: evidence for name in control.expected_run_ids(self.study)}
        manifest = SimpleNamespace(dataset_id=None, scope="full")
        return seal, completion, manifest

    def finalization_patches(self, seal, completion):
        stack = []
        for mock in (patch.object(control, "global_seal", return_value=seal),
                     patch.object(manage, "_verified_completion_artifacts", return_value=completion),
                     patch.object(manage.data, "load_manifest", side_effect=lambda p:
                        SimpleNamespace(dataset_id=Path(p).stem, scope="full")),
                     patch.object(manage.data, "save_manifest", side_effect=lambda manifest, p, public:
                        run.write_json(p, {"dataset_id": manifest.dataset_id, "scope": "full", "public": public}))):
            stack.append(mock.start())
            self.addCleanup(mock.stop)
        return stack

    def test_initialization_claims_only_new_root_and_pins_full_protocol_source(self):
        result = self.initialize()
        self.assertEqual(result["expected_final_fits"], 720)
        self.assertEqual(manage.require_owned(self.work, self.study), result)
        self.assertEqual(self.initialize(), result)
        self.assertTrue((self.work / "code/vision_only/manage.py").is_file())
        self.assertEqual((self.work / "code/vision_only/study.json").read_bytes(), self.study.read_bytes())
        self.assertNotIn(str(self.base), (self.work / manage.SNAPSHOT).read_text())

    def test_existing_unowned_data_is_not_claimed(self):
        self.artifact("borrowed.bin", b"unowned")
        with self.assertRaisesRegex(lifecycle.LifecycleError, "absent or empty"):
            self.initialize()
        self.assertEqual((self.work / "borrowed.bin").read_bytes(), b"unowned")

    def test_changed_protocol_source_or_snapshot_blocks_controller_stages(self):
        self.initialize()
        with patch.object(run, "code_identity", return_value={"changed.py": "changed"}):
            with self.assertRaisesRegex(ValueError, "source or protocol changed"):
                manage.require_owned(self.work, self.study)
        original = self.study.read_bytes()
        changed = run.read_json(self.study)
        changed["training"]["epochs"] += 1
        self.study.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, "different protocol"):
            manage.require_owned(self.work, self.study)
        self.study.write_bytes(original)
        (self.work / "code/vision_only/manage.py").write_bytes(b"changed snapshot")
        with self.assertRaisesRegex(ValueError, "snapshot bytes changed"):
            manage.require_owned(self.work, self.study)

    def test_finalize_requires_physical_global_seal_before_register_or_export(self):
        self.initialize()
        with patch.object(control, "global_seal", side_effect=ValueError("global grid incomplete")):
            with self.assertRaisesRegex(ValueError, "global grid incomplete"):
                manage.finalize(self.work, self.study, self.archive)
        self.assertFalse(self.archive.exists())
        self.assertEqual(lifecycle._state(self.work)[1]["runs"], {})

    def test_finalize_requires_complete_bound_aggregate_and_matching_hashes(self):
        seal, completion, _ = self.completed_fixture()
        self.finalization_patches(seal, completion)
        receipt = self.work / "aggregate/complete.json"
        record = run.read_json(receipt)
        record["completed_final_fits"] = 719
        run.write_json(receipt, record)
        with self.assertRaisesRegex(ValueError, "Aggregate is incomplete"):
            manage.finalize(self.work, self.study, self.archive)
        record["completed_final_fits"] = 720
        run.write_json(receipt, record)
        (self.work / "aggregate/per_seed.csv").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "Aggregate evidence changed"):
            manage.finalize(self.work, self.study, self.archive)
        self.assertFalse(self.archive.exists())

    def test_unknown_or_private_export_metadata_blocks_finalization(self):
        seal, completion, _ = self.completed_fixture()
        self.finalization_patches(seal, completion)
        unknown = self.artifact("private_paths.json", b'{"dataset_root":"private"}')
        with self.assertRaisesRegex(lifecycle.LifecycleError, "Unknown file"):
            manage.finalize(self.work, self.study, self.archive)
        unknown.unlink()
        self.artifact("runs/fixture/private.json", json.dumps({"source_root": str(self.base)}).encode())
        with self.assertRaisesRegex(ValueError, "Private metadata"):
            manage.finalize(self.work, self.study, self.archive)
        self.assertFalse(self.archive.exists())

    def test_verified_export_is_dry_run_retains_code_and_never_borrowed_data(self):
        seal, completion, _ = self.completed_fixture()
        self.finalization_patches(seal, completion)
        # Private operational manifests remain excluded; only public copies are retained.
        self.artifact("manifests/imagenet1k.json", json.dumps({"source_root": str(self.base)}).encode())
        borrowed = self.base / "borrowed-images.bin"
        borrowed.write_bytes(b"existing borrowed data")
        result = manage.finalize(self.work, self.study, self.archive)
        self.assertFalse(result["deleted"])
        self.assertTrue(result["cleanup_plan"]["dry_run"])
        self.assertEqual(result["completed_final_fits"], 720)
        self.assertTrue((self.work / "data/owned.bin").exists())
        self.assertEqual(borrowed.read_bytes(), b"existing borrowed data")
        with zipfile.ZipFile(self.archive) as package:
            self.assertNotIn("data/owned.bin", package.namelist())
            self.assertNotIn("cache/features.bin", package.namelist())
            self.assertNotIn("manifests/imagenet1k.json", package.namelist())
            self.assertIn("public_manifests/imagenet1k.json", package.namelist())
            self.assertNotIn(str(self.base).encode(), package.read("public_manifests/imagenet1k.json"))
        marker = lifecycle._state(self.work)[1]
        self.assertEqual(len(marker["runs"]), 720)
        self.assertTrue(all(record["status"] == "complete" for record in marker["runs"].values()))

    def test_symlink_and_archive_inside_root_fail_without_deletion(self):
        self.initialize()
        with self.assertRaisesRegex(lifecycle.LifecycleError, "outside"):
            manage.finalize(self.work, self.study, self.work / "results.zip")
        target = self.artifact("code/rogue.py", b"unregistered code")
        with self.assertRaisesRegex(ValueError, "Unknown file in preserved code"):
            manage._classification(target.relative_to(self.work).as_posix(), lifecycle._state(self.work)[1])

    def test_metric_completion_rejects_nonfinite_and_wrong_class_coverage(self):
        metrics = {key: 1. for key in ("macro_f1", "top1", "top5", "macro_precision", "macro_recall",
                                      "head_f1", "middle_f1", "tail_f1")}
        metrics.update(per_class_precision=[1., 1., 1.], per_class_recall=[1., 1., 1.],
                       per_class_f1=[1., 1., 1.], support=[1, 1, 1])
        value = {"raw": dict(metrics), "adjusted": dict(metrics)}
        manage._metric_record(value, 3)
        with self.assertRaisesRegex(ValueError, "coverage"):
            manage._metric_record(value, 4)
        value["raw"]["macro_f1"] = float("nan")
        with self.assertRaisesRegex(ValueError, "classification metric"):
            manage._metric_record(value, 3)

    def test_completion_verifier_rejects_missing_failed_or_mismatched_physical_evidence(self):
        self.initialize()
        out = self.work / "runs/cifar100/native/deit_small"
        cell = out / "final/natural_ce/10"
        cell.mkdir(parents=True)
        run.write_json(out / "selection.json", {"fingerprint": "selected-fixture"})
        for name in ("checkpoint.pt", "predictions.npz", "calibration_predictions.npz"):
            (cell / name).write_bytes(b"explicit synthetic hash fixture")
        trained = {"status": "trained", "fingerprint": "trained-fixture",
                   "checkpoint_sha256": run.digest(cell / "checkpoint.pt")}
        run.write_json(cell / "trained.json", trained)
        metrics = {key: 1. for key in ("macro_f1", "top1", "top5", "macro_precision", "macro_recall",
                                      "head_f1", "middle_f1", "tail_f1")}
        metrics.update(per_class_precision=[1., 1., 1.], per_class_recall=[1., 1., 1.],
                       per_class_f1=[1., 1., 1.], support=[1, 1, 1])
        confirmed = {"status": "complete", "arm": "natural_ce", "seed": 10,
                     "trained_fingerprint": "trained-fixture", "selection_fingerprint": "selected-fixture",
                     "raw": metrics, "adjusted": metrics,
                     "predictions_sha256": run.digest(cell / "predictions.npz"),
                     "calibration_predictions_sha256": run.digest(cell / "calibration_predictions.npz")}
        args = SimpleNamespace(out=out)
        spec = {"arms": ["natural_ce"], "training": {"final_seeds": [10]},
                "datasets": {"cifar100": {"classes": 3}}}
        identifier = "cifar100/native/deit_small/natural_ce/10"
        with patch.object(control, "contexts", return_value=[("cifar100", "native", "deit_small", args)]), \
             patch.object(control, "expected_run_ids", return_value=[identifier]), \
             patch.object(run, "settings", return_value=spec):
            run.write_json(cell / "confirmed.json", confirmed)
            results = manage._verified_completion_artifacts(self.work, self.study, {"completed_final_fits": 1})
            self.assertEqual(set(results), {identifier})
            for field, value in (("status", "failed"), ("seed", 11), ("trained_fingerprint", "wrong")):
                changed = copy.deepcopy(confirmed)
                changed[field] = value
                run.write_json(cell / "confirmed.json", changed)
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, "mismatched"):
                    manage._verified_completion_artifacts(self.work, self.study, {"completed_final_fits": 1})
            run.write_json(cell / "confirmed.json", confirmed)
            (cell / "predictions.npz").write_bytes(b"changed physical predictions")
            with self.assertRaisesRegex(ValueError, "evidence file changed"):
                manage._verified_completion_artifacts(self.work, self.study, {"completed_final_fits": 1})


if __name__ == "__main__": unittest.main()
