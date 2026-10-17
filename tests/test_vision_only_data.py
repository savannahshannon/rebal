"""Integrity tests use tiny, explicitly labeled fixtures; no scientific fits."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from vision_only import data


class VisionDataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.root = self.base / "images"
        index = 1
        for split, counts in (("train", (8, 3, 2)), ("val", (1, 1, 1))):
            for label, count in enumerate(counts):
                directory = self.root / split / ("class" + str(label))
                directory.mkdir(parents=True)
                for image in range(count):
                    Image.new("RGB", (8, 9), (index, index + 20, index + 40)).save(directory / f"{image}.png")
                    index += 1
        self.spec = data.DatasetSpec("fixture", 13, 3, 3, "https://example.invalid/fixture", "test_fixture")
        self.manifest = data.inventory_imagefolder(self.root, self.spec)

    def tearDown(self):
        self.temporary.cleanup()

    def test_official_counts_cannot_be_redefined_as_mini_dataset(self):
        spec = replace(data.dataset_spec("imagenet1k"), train_count=13, confirmation_count=3, num_classes=3)
        with self.assertRaisesRegex(ValueError, "cannot be overridden"):
            data.inventory_imagefolder(self.root, spec)

    def test_native_fitting_retains_all_non_development_training(self):
        plan = data.make_partitions(self.manifest, 901)
        self.assertEqual(plan.regime, "native")
        self.assertFalse(plan.lt_omitted)
        self.assertEqual(len(plan.fit) + len(plan.tune) + len(plan.calibration), 13)
        self.assertEqual(plan.class_counts["fit"], [6, 1, 2])
        self.assertEqual(plan.class_counts["tune"], [1, 1, 0])
        self.assertEqual(plan.class_counts["calibration"], [1, 1, 0])
        self.assertEqual(plan.limited_development_classes, (2,))
        self.assertEqual(set(plan.confirmation), {sample.sample_id for sample in self.manifest.samples
                                                  if sample.source_split == "confirmation"})
        self.assertFalse(set(plan.fit + plan.tune + plan.calibration) & set(plan.confirmation))

    def test_seed_and_longtail_receipts_are_reproducible_and_complete(self):
        first = data.make_partitions(self.manifest, 11, lt_ratio=100)
        second = data.make_partitions(self.manifest, 11, lt_ratio=100)
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(first.regime, "custom_LT")
        self.assertTrue(first.lt_omitted)
        self.assertEqual(len(first.fit + first.tune + first.calibration + first.lt_omitted), 13)
        self.assertTrue(all(first.class_counts["fit"]))
        self.assertNotEqual(data.make_partitions(self.manifest, 12).fingerprint,
                            data.make_partitions(self.manifest, 13).fingerprint)

    def test_saved_manifest_and_split_detect_tampering_and_allow_relocation(self):
        target = self.base / "manifest.json"
        data.save_manifest(self.manifest, target)
        restored = data.load_manifest(target)
        self.assertEqual(restored.fingerprint, self.manifest.fingerprint)
        data.save_manifest(self.manifest, target, public=True)
        self.assertNotIn("source_root", json.loads(target.read_text()))
        restored = data.load_manifest(target, root_override=self.root)
        self.assertEqual(restored.fingerprint, self.manifest.fingerprint)
        split = data.make_partitions(restored, 11)
        split_path = self.base / "split.json"
        data.save_partitions(split, split_path)
        self.assertEqual(data.load_partitions(split_path, restored), split)
        changed = json.loads(split_path.read_text())
        changed["seed"] = 12
        split_path.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            data.load_partitions(split_path, restored)

    def test_changed_source_file_is_detected(self):
        data.verify_manifest_files(self.manifest)
        path = self.root / self.manifest.samples[0].relative_path
        Image.new("RGB", (8, 9), (255, 0, 0)).save(path)
        with self.assertRaisesRegex(ValueError, "bytes changed"):
            data.verify_manifest_files(self.manifest)

    def test_count_and_class_mapping_fail_before_smaller_source_is_accepted(self):
        (self.root / "train/class0/0.png").unlink()
        with self.assertRaisesRegex(ValueError, "count mismatch"):
            data.inventory_imagefolder(self.root, self.spec)
        (self.root / "val/class0").rename(self.root / "val/wrong_class")
        with self.assertRaisesRegex(ValueError, "mappings differ"):
            data.inventory_imagefolder(self.root, self.spec)

    def test_path_traversal_and_symlink_inputs_are_rejected(self):
        for relative in ("../outside", "/etc/passwd", "train/../../outside", "C:\\test"):
            with self.assertRaises(ValueError):
                data.contained_path(self.root, relative)
        (self.root / "link").symlink_to(self.root / "train", target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink"):
            data.contained_path(self.root, "link/class0/0.png")

    def test_train_confirmation_duplicate_requires_explicit_exclusion(self):
        overlap = self.root / "train/class0/0.png"
        overlap.write_bytes((self.root / "val/class0/0.png").read_bytes())
        manifest = data.inventory_imagefolder(self.root, self.spec)
        with self.assertRaisesRegex(ValueError, "content crosses"):
            data.make_partitions(manifest, 11)
        clean = data.exclude_training_exact_duplicates(manifest)
        self.assertEqual(len(clean.samples), 16)
        self.assertEqual(len(clean.exclusions), 1)
        self.assertEqual(clean.expected_counts["train"], 13)
        plan = data.make_partitions(clean, 11)
        self.assertEqual(len(plan.excluded), 1)
        self.assertEqual(len(plan.fit + plan.tune + plan.calibration + plan.excluded), 13)

    def test_conflicting_duplicate_labels_and_fake_partition_membership_are_rejected(self):
        copied = self.root / "train/class1/0.png"
        copied.write_bytes((self.root / "val/class0/0.png").read_bytes())
        manifest = data.inventory_imagefolder(self.root, self.spec)
        with self.assertRaisesRegex(ValueError, "Conflicting labels"):
            data.exclude_training_exact_duplicates(manifest)
        plan = data.make_partitions(self.manifest, 11)
        bad = replace(plan, fit=plan.fit + plan.confirmation[:1])
        with self.assertRaises(ValueError):
            data.validate_partitions(self.manifest, bad)

    def _inat_annotations(self):
        categories = [{"id": label * 10 + 10, "name": "class" + str(label)} for label in range(3)]
        documents = []
        for split in ("train", "confirmation"):
            samples = [sample for sample in self.manifest.samples if sample.source_split == split]
            images = [{"id": index + 100, "file_name": sample.relative_path,
                       "width": sample.width, "height": sample.height} for index, sample in enumerate(samples)]
            annotations = [{"image_id": image["id"], "category_id": sample.label * 10 + 10}
                           for image, sample in zip(images, samples)]
            document = {"categories": categories, "images": images, "annotations": annotations}
            path = self.root / (split + ".json")
            path.write_text(json.dumps(document))
            documents.append(path)
        return documents

    def test_inat_coco_mapping_and_labeled_coverage(self):
        train, val = self._inat_annotations()
        manifest = data.inventory_inat2018(self.root, train, val, self.spec)
        self.assertEqual(manifest.source_class_ids, ("10", "20", "30"))
        self.assertEqual(len(manifest.source_files), 2)
        self.assertEqual(len(manifest.samples), 16)
        doc = json.loads(train.read_text())
        doc["annotations"].append(doc["annotations"][0])
        train.write_text(json.dumps(doc))
        with self.assertRaisesRegex(ValueError, "exactly one"):
            data.inventory_inat2018(self.root, train, val, self.spec)

    def test_cifar_checksum_precedes_official_loader_instantiation(self):
        class NeverLoad:
            base_folder = "cifar-100-python"
            train_list = [("train", "0" * 32)]
            test_list = [("test", "0" * 32)]
            meta = {"filename": "meta", "md5": "0" * 32}

            def __init__(self, **kwargs):
                raise AssertionError("Unverified CIFAR bytes must never be deserialized")

        directory = self.root / NeverLoad.base_folder
        directory.mkdir()
        (directory / "train").write_bytes(b"untrusted pickle bytes")
        with patch.object(data, "_official_cifar100_class", return_value=NeverLoad):
            with self.assertRaisesRegex(ValueError, "before deserialization"):
                data.inventory_cifar100(self.root, self.spec)

    def test_cleanup_preview_requires_verified_export_and_never_deletes(self):
        root = data.create_owned_data_root(self.base / "owned", "run_1")
        (root / "raw").mkdir()
        (root / "raw/image.bin").write_bytes(b"dataset")
        evidence_a, evidence_b = self.base / "evidence_a.zip", self.base / "evidence_b.zip"
        evidence_a.write_bytes(b"preserved verified evidence")
        evidence_b.write_bytes(evidence_a.read_bytes())
        receipt = {"run_id": "run_1", "plan_fingerprint": "plan", "aggregate_fingerprint": "aggregate",
                   "retained_evidence_copies": [{"path": str(path), "sha256": data.sha256_file(path)}
                                                for path in (evidence_a, evidence_b)]}
        path = self.base / "export_verification.json"
        path.write_text(json.dumps(receipt))
        with self.assertRaisesRegex(ValueError, "cleanup blocked"):
            data.cleanup_preview(root, "run_1", ["raw"], path)
        for gate in ("all_required_cells_verified", "checkpoint_audit_verified", "independent_metric_replay_verified",
                     "report_verified", "public_export_verified", "overleaf_compile_verified", "no_active_jobs"):
            receipt[gate] = True
        path.write_text(json.dumps(receipt))
        preview = data.cleanup_preview(root, "run_1", ["raw"], path)
        self.assertEqual(preview["status"], "verified_preview_only")
        self.assertFalse(preview["deleted"])
        self.assertEqual(preview["total_bytes"], 7)
        self.assertTrue((root / "raw/image.bin").is_file())
        with self.assertRaises(ValueError):
            data.cleanup_preview(root, "run_1", ["../images"], path)
        with self.assertRaises(ValueError):
            data.create_owned_data_root(root, "another_run")


if __name__ == "__main__":
    unittest.main()
