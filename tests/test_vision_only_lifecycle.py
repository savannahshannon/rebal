"""Synthetic ownership/export tests; no real experiment data is removed."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

spec = importlib.util.spec_from_file_location("vision_lifecycle", Path(__file__).parents[1] / "vision_only" / "lifecycle.py")
lifecycle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lifecycle)


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name).resolve()
        self.root = self.base / "owned-run"
        self.archive = self.base / "retained" / "results.zip"
        self.receipt = self.archive.with_suffix(".zip.receipt.json")

    def init(self, ids=("run-a", "run-b")):
        return lifecycle.initialize_run_root(self.root, "synthetic-study", ids, "0" * 64)

    def artifact(self, path, content, kind):
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        lifecycle.register_artifact(self.root, path, kind)
        return target

    def completed(self):
        self.init()
        self.artifact("code/train.py", b"print('code')\n", "code")
        self.artifact("datasets/images.bin", b"synthetic dataset", "dataset")
        self.artifact("cache/features.bin", b"synthetic cache", "cache")
        self.artifact("checkpoints/a.bin", b"synthetic checkpoint", "checkpoint")
        self.artifact("results/a.json", b'{"metric": 0.7}', "result")
        self.artifact("results/b.json", b'{"metric": 0.8}', "result")
        self.artifact("results/aggregate.json", b'{"mean": 0.75}', "result")
        lifecycle.record_run_completion(self.root, "run-a", "complete", ["results/a.json", "checkpoints/a.bin"])
        lifecycle.record_run_completion(self.root, "run-b", "complete", ["results/b.json"])
        lifecycle.record_aggregates(self.root, ["results/aggregate.json"])

    def export(self):
        return lifecycle.export_results(self.root, self.archive)

    def test_complete_export_excludes_datasets_and_dry_run_changes_nothing(self):
        self.completed()
        before = lifecycle.inventory(self.root)
        self.export()
        with zipfile.ZipFile(self.archive) as package:
            self.assertIn("results/aggregate.json", package.namelist())
            self.assertIn("checkpoints/a.bin", package.namelist())
            self.assertNotIn("datasets/images.bin", package.namelist())
            self.assertNotIn("cache/features.bin", package.namelist())
        plan = lifecycle.cleanup(self.root, self.receipt)
        self.assertTrue(plan["dry_run"])
        self.assertIn("datasets/images.bin", plan["delete"])
        self.assertEqual(plan["preserve_code"], ["code/train.py"])
        self.assertEqual(lifecycle.inventory(self.root), before)

    def test_explicit_cleanup_preserves_only_code_and_retained_export(self):
        self.completed()
        self.export()
        report = lifecycle.cleanup(self.root, self.receipt, execute=True)
        self.assertTrue(report["executed"])
        self.assertFalse(report["dry_run"])
        self.assertEqual([path.relative_to(self.root).as_posix() for path in self.root.rglob("*") if path.is_file()], ["code/train.py"])
        self.assertTrue(self.archive.is_file())
        self.assertTrue(self.receipt.with_suffix(".cleanup.json").is_file())
        with self.assertRaises(lifecycle.LifecycleError):
            lifecycle.cleanup(self.root, self.receipt, execute=True)

    def test_existing_unowned_root_and_protected_paths_are_never_adopted(self):
        self.root.mkdir()
        (self.root / "borrowed-dataset").write_bytes(b"not ours")
        with self.assertRaisesRegex(lifecycle.LifecycleError, "absent or empty"):
            self.init()
        for root in (Path("/"), Path.home(), Path.home().parent, self.base / "x" / ".." / "y"):
            with self.subTest(root=root), self.assertRaises(lifecycle.LifecycleError):
                lifecycle.initialize_run_root(root, "bad", ["a"])
        self.assertEqual((self.root / "borrowed-dataset").read_bytes(), b"not ours")

    def test_missing_pending_failed_and_unexpected_runs_block_exports(self):
        self.init()
        self.artifact("a.json", b"{}", "result")
        lifecycle.record_run_completion(self.root, "run-a", "complete", ["a.json"])
        with self.assertRaisesRegex(lifecycle.LifecycleError, "Not all expected"):
            self.export()
        for status in ("pending", "running", "failed"):
            lifecycle.record_run_completion(self.root, "run-b", status)
            with self.subTest(status=status), self.assertRaisesRegex(lifecycle.LifecycleError, "Incomplete"):
                self.export()
        with self.assertRaisesRegex(lifecycle.LifecycleError, "Unexpected"):
            lifecycle.record_run_completion(self.root, "unplanned", "complete", ["a.json"])
        with self.assertRaisesRegex(lifecycle.LifecycleError, "hashed result"):
            lifecycle.record_run_completion(self.root, "run-b", "complete")
        lifecycle.record_run_completion(self.root, "run-b", "complete", ["a.json"])
        with self.assertRaisesRegex(lifecycle.LifecycleError, "aggregate"):
            self.export()

    def test_unknown_file_in_owned_root_blocks_export_and_cleanup(self):
        self.completed()
        self.export()
        (self.root / "someone-elses-data").write_bytes(b"keep")
        with self.assertRaisesRegex(lifecycle.LifecycleError, "Unowned"):
            lifecycle.plan_cleanup(self.root, self.receipt)
        with self.assertRaisesRegex(lifecycle.LifecycleError, "Unowned"):
            lifecycle.export_results(self.root, self.base / "second.zip")
        self.assertTrue((self.root / "datasets/images.bin").exists())

    def test_changed_run_status_after_export_blocks_explicit_cleanup(self):
        self.completed()
        self.export()
        lifecycle.record_run_completion(self.root, "run-b", "failed", ["results/b.json"])
        with self.assertRaisesRegex(lifecycle.LifecycleError, "Incomplete"):
            lifecycle.cleanup(self.root, self.receipt, execute=True)
        self.assertTrue((self.root / "datasets/images.bin").is_file())
        self.assertTrue((self.root / lifecycle.MARKER).is_file())

    @unittest.skipUnless(hasattr(os, "mkfifo"), "Named pipes unavailable")
    def test_replaced_fifo_is_rejected_without_blocking(self):
        self.completed()
        target = self.root / "cache/features.bin"
        target.unlink()
        os.mkfifo(target)
        with self.assertRaisesRegex(lifecycle.LifecycleError, "ordinary files"):
            self.export()

    def test_external_borrowed_path_and_parent_traversal_cannot_be_registered(self):
        self.init()
        borrowed = self.base / "borrowed.bin"
        borrowed.write_bytes(b"borrowed")
        for value in (borrowed, "../borrowed.bin", "/etc/passwd", "a/../../borrowed.bin", "x\\y"):
            with self.subTest(value=value), self.assertRaises(lifecycle.LifecycleError):
                lifecycle.register_artifact(self.root, value, "dataset")
        self.assertEqual(borrowed.read_bytes(), b"borrowed")

    def test_symlink_file_directory_root_and_export_are_rejected(self):
        self.completed()
        borrowed = self.base / "borrowed.bin"
        borrowed.write_bytes(b"borrowed")
        (self.root / "link.bin").symlink_to(borrowed)
        with self.assertRaises(lifecycle.LifecycleError):
            lifecycle.register_artifact(self.root, "link.bin", "dataset")
        (self.root / "link.bin").unlink()
        (self.root / "linked-directory").symlink_to(self.base, target_is_directory=True)
        with self.assertRaisesRegex(lifecycle.LifecycleError, "Symbolic link"):
            self.export()
        (self.root / "linked-directory").unlink()
        alias = self.base / "alias"
        alias.symlink_to(self.root, target_is_directory=True)
        with self.assertRaisesRegex(lifecycle.LifecycleError, "Symbolic links"):
            lifecycle.inventory(alias)
        export_alias = self.base / "retained-alias"
        export_alias.symlink_to(self.base, target_is_directory=True)
        with self.assertRaisesRegex(lifecycle.LifecycleError, "Symbolic links"):
            lifecycle.export_results(self.root, export_alias / "results.zip")
        self.assertEqual(borrowed.read_bytes(), b"borrowed")

    def test_hardlinked_files_are_not_adopted(self):
        self.init()
        borrowed = self.base / "borrowed.bin"
        borrowed.write_bytes(b"borrowed")
        os.link(borrowed, self.root / "hardlink.bin")
        with self.assertRaisesRegex(lifecycle.LifecycleError, "one link"):
            lifecycle.register_artifact(self.root, "hardlink.bin", "dataset")
        self.assertEqual(borrowed.read_bytes(), b"borrowed")

    def test_batch_and_tree_registration_write_the_marker_once(self):
        self.init()
        for index in range(32):
            target = self.root / "images" / str(index // 8) / (str(index) + ".bin")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(str(index).encode())
        original = lifecycle._atomic_json
        with patch.object(lifecycle, "_atomic_json", wraps=original) as writer:
            registered = lifecycle.register_tree(self.root, "images", "dataset")
        self.assertEqual(len(registered), 32)
        self.assertEqual(writer.call_count, 1)
        self.assertEqual(len(lifecycle.inventory(self.root)), 32)

    def test_failed_batch_cannot_partially_adopt_files(self):
        self.init()
        (self.root / "good.bin").write_bytes(b"good")
        before = (self.root / lifecycle.MARKER).read_bytes()
        with self.assertRaises(lifecycle.LifecycleError):
            lifecycle.register_artifacts(self.root, [("good.bin", "dataset"), ("missing.bin", "dataset")])
        self.assertEqual((self.root / lifecycle.MARKER).read_bytes(), before)
        with self.assertRaisesRegex(lifecycle.LifecycleError, "Duplicate artifact"):
            lifecycle.register_artifacts(self.root, [("good.bin", "dataset"), ("good.bin", "dataset")])
        self.assertEqual((self.root / lifecycle.MARKER).read_bytes(), before)

    def test_tree_registration_rejects_nested_symlink_without_adopting_any_file(self):
        self.init()
        (self.root / "images").mkdir()
        (self.root / "images" / "one.bin").write_bytes(b"owned")
        (self.root / "images" / "linked").symlink_to(self.base, target_is_directory=True)
        with self.assertRaisesRegex(lifecycle.LifecycleError, "Symbolic link directory"):
            lifecycle.register_tree(self.root, "images", "dataset")
        state = json.loads((self.root / lifecycle.MARKER).read_text())
        self.assertEqual(state["artifacts"], {})

    def test_result_and_dataset_modification_and_missing_files_block_cleanup(self):
        self.completed()
        self.export()
        for name in ("results/a.json", "datasets/images.bin"):
            path = self.root / name
            original = path.read_bytes()
            path.write_bytes(b"changed")
            with self.subTest(name=name), self.assertRaisesRegex(lifecycle.LifecycleError, "changed"):
                lifecycle.plan_cleanup(self.root, self.receipt)
            path.write_bytes(original)
        (self.root / "results/b.json").unlink()
        with self.assertRaisesRegex(lifecycle.LifecycleError, "missing"):
            lifecycle.plan_cleanup(self.root, self.receipt)

    def test_missing_or_tampered_archive_and_self_reported_receipt_are_insufficient(self):
        self.completed()
        self.export()
        original = self.archive.read_bytes()
        self.archive.unlink()
        with self.assertRaises(lifecycle.LifecycleError):
            lifecycle.plan_cleanup(self.root, self.receipt)
        self.archive.write_bytes(b"claimed completion, not a zip")
        proof = json.loads(self.receipt.read_text())
        proof["archive"] = lifecycle._file_record(self.archive)
        self.receipt.write_text(json.dumps(proof))
        with self.assertRaisesRegex(lifecycle.LifecycleError, "archive is unavailable or invalid"):
            lifecycle.plan_cleanup(self.root, self.receipt)
        self.archive.write_bytes(original)
        proof["archive"] = lifecycle._file_record(self.archive)
        proof["result_inventory"].pop("results/a.json")
        self.receipt.write_text(json.dumps(proof))
        with self.assertRaisesRegex(lifecycle.LifecycleError, "incomplete result inventory"):
            lifecycle.plan_cleanup(self.root, self.receipt)

    def test_changed_zip_content_fails_even_if_receipt_archive_hash_is_refreshed(self):
        self.completed()
        self.export()
        with zipfile.ZipFile(self.archive) as package:
            contents = {name: package.read(name) for name in package.namelist()}
        contents["results/a.json"] = b"{}"
        with zipfile.ZipFile(self.archive, "w") as package:
            for name, value in contents.items():
                package.writestr(name, value)
        proof = json.loads(self.receipt.read_text())
        proof["archive"] = lifecycle._file_record(self.archive)
        self.receipt.write_text(json.dumps(proof))
        with self.assertRaisesRegex(lifecycle.LifecycleError, "result hash mismatch"):
            lifecycle.plan_cleanup(self.root, self.receipt)

    def test_export_inside_cleanup_root_is_rejected_and_archive_not_overwritten(self):
        self.completed()
        with self.assertRaisesRegex(lifecycle.LifecycleError, "outside"):
            lifecycle.export_results(self.root, self.root / "export.zip")
        with self.assertRaisesRegex(lifecycle.LifecycleError, "outside"):
            lifecycle.export_results(self.root, self.archive, self.root / "receipt.json")
        self.archive.parent.mkdir()
        self.archive.write_bytes(b"existing retained file")
        with self.assertRaisesRegex(lifecycle.LifecycleError, "new files"):
            self.export()
        self.assertEqual(self.archive.read_bytes(), b"existing retained file")

    def test_re_registration_invalidates_completed_run_and_aggregate(self):
        self.completed()
        lifecycle.register_artifact(self.root, "results/a.json", "result")
        with self.assertRaisesRegex(lifecycle.LifecycleError, "Incomplete"):
            self.export()

    def test_forged_extra_run_record_blocks_cleanup(self):
        self.completed()
        self.export()
        marker = self.root / lifecycle.MARKER
        state = json.loads(marker.read_text())
        state["runs"]["unexpected-run"] = {"status": "complete", "artifacts": {}}
        marker.write_text(json.dumps(state))
        with self.assertRaisesRegex(lifecycle.LifecycleError, "Not all expected"):
            lifecycle.plan_cleanup(self.root, self.receipt)

    def test_preflight_is_read_only_and_excludes_identifiers_and_raw_errors(self):
        with patch.object(lifecycle.shutil, "which", return_value=None):
            report = lifecycle.preflight(self.root)
        self.assertFalse(self.root.exists())
        encoded = json.dumps(report)
        for private in (str(self.base), str(Path.home()), "hostname", "username", "ip_address"):
            self.assertNotIn(private, encoded)
        self.assertGreater(report["disk"]["free_bytes"], 0)
        self.assertEqual(report["gpu_probe_status"], "not_available")


if __name__ == "__main__":
    unittest.main()
