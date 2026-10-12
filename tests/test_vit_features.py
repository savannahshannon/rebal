"""Synthetic cache integrity and leakage checks; no downloads or real scores."""
import argparse
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("vit_extractor", ROOT / "scripts/extract_vit_comparison_features.py")
e = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e)


class ToyCorpus(torch.utils.data.Dataset):
    def __init__(self, refuse_heldout=False):
        self.ntrain = 5
        self.labels = np.arange(7, dtype=np.int64) % 3
        self.names = [f"row{i}" for i in range(7)]
        self.original_manifest_sha256 = "1" * 64
        self.original = dict(classes=["a", "b", "c"], train_files=self.names[:5], eval_files=self.names[5:],
                             train_pixel_sha256=["2"*64]*5, eval_pixel_sha256=["3"*64]*2, archive_sha256="4"*64)
        self.refuse_heldout = refuse_heldout

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        if self.refuse_heldout and index >= self.ntrain:
            raise AssertionError("Held-out pilot image access")
        return torch.full((3, 4, 4), float(index)), index


class ToyEncoder(torch.nn.Module):
    def __init__(self, fail_after=None):
        super().__init__()
        self.fail_after, self.calls = fail_after, 0

    def forward(self, x):
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise InterruptedError("Synthetic interruption")
        value = x.mean((1, 2, 3))
        return torch.stack((value, value + .25), dim=1)


def details():
    return dict(encoder_id="resnet18", weights_sha256="5"*64, encoder_state_sha256="6"*64, feature_dim=2)


def args(root):
    return argparse.Namespace(device="cpu", batch_size=2, workers=0, threads=1, encoders=["resnet18", "deit_tiny"],
                              out=root / "pilot.json", pilot_images=4)


class ViTFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.addClassCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)

    def test_shared_preprocessing_constant_rgb(self):
        image = Image.fromarray(np.full((64, 64, 3), [64, 128, 192], dtype=np.uint8))
        actual = e.preprocessing()(image)
        expected = (torch.tensor([64., 128., 192.])/255 - torch.tensor(e.PREPROCESSING["mean"])) / torch.tensor(e.PREPROCESSING["std"])
        self.assertEqual(tuple(actual.shape), (3, 224, 224))
        self.assertTrue(torch.allclose(actual, expected[:, None, None].expand_as(actual), atol=1e-6))
        self.assertEqual(e.PREPROCESSING["interpolation"], "PIL bicubic")

    def test_original_raw_pixel_hash_checked_before_transform(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = Image.fromarray(np.full((64, 64, 3), 50, dtype=np.uint8))
            stream = io.BytesIO()
            image.save(stream, format="PNG")
            archive = root / "images.zip"
            with zipfile.ZipFile(archive, "w") as z:
                z.writestr("one.png", stream.getvalue())
            corpus = e.Corpus.__new__(e.Corpus)
            corpus.archive, corpus.names, corpus.z = str(archive), ["one.png"], None
            corpus.transform, corpus.pixel_hashes = e.preprocessing(), ["0"*64]
            try:
                with self.assertRaisesRegex(ValueError, "raw-pixel identity"):
                    corpus[0]
                corpus.pixel_hashes = [hashlib.sha256(np.asarray(image).tobytes()).hexdigest()]
                x, index = corpus[0]
                self.assertEqual(index, 0)
                self.assertEqual(tuple(x.shape), (3, 224, 224))
            finally:
                if corpus.z is not None:
                    corpus.z.close()

    def test_complete_cache_order_hashes_and_idempotence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus, model = ToyCorpus(), ToyEncoder()
            result = e.extract_cache(corpus, model, details(), args(root), root / "cache")
            features = np.load(root / "cache/features.npy", allow_pickle=False)
            self.assertTrue(np.array_equal(features[:, 0], np.arange(7)))
            self.assertTrue(np.array_equal(np.load(root / "cache/labels.npy"), corpus.labels))
            self.assertEqual(result["artifacts"]["features.npy"], e.sha(root / "cache/features.npy"))
            self.assertFalse(result["evaluation_scores_computed"])
            calls = model.calls
            self.assertEqual(e.extract_cache(corpus, model, details(), args(root), root / "cache"), result)
            self.assertEqual(model.calls, calls)

    def test_interrupted_resume_matches_uninterrupted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = ToyCorpus()
            with self.assertRaises(InterruptedError):
                e.extract_cache(corpus, ToyEncoder(fail_after=1), details(), args(root), root / "resume")
            self.assertFalse((root / "resume/manifest.json").exists())
            self.assertEqual(e.read_json(root / "resume/progress.json")["next_row"], 2)
            resumed = e.extract_cache(corpus, ToyEncoder(), details(), args(root), root / "resume")
            direct = e.extract_cache(corpus, ToyEncoder(), details(), args(root), root / "direct")
            self.assertEqual(resumed["artifacts"], direct["artifacts"])

    def test_partial_feature_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(InterruptedError):
                e.extract_cache(ToyCorpus(), ToyEncoder(fail_after=1), details(), args(root), root / "cache")
            feature = np.load(root / "cache/features.npy.partial", mmap_mode="r+")
            feature[0, 0] += 1
            feature.flush()
            del feature
            with self.assertRaisesRegex(ValueError, "Partial feature block checksum"):
                e.extract_cache(ToyCorpus(), ToyEncoder(), details(), args(root), root / "cache")

    def test_partial_plan_change_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(InterruptedError):
                e.extract_cache(ToyCorpus(), ToyEncoder(fail_after=1), details(), args(root), root / "cache")
            changed = args(root)
            changed.batch_size = 3
            with self.assertRaisesRegex(ValueError, "different source/weight/configuration"):
                e.extract_cache(ToyCorpus(), ToyEncoder(), details(), changed, root / "cache")

    def test_completed_artifact_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            e.extract_cache(ToyCorpus(), ToyEncoder(), details(), args(root), root / "cache")
            np.save(root / "cache/labels.npy", np.arange(7))
            with self.assertRaisesRegex(ValueError, "Completed cache artifact mismatch"):
                e.extract_cache(ToyCorpus(), ToyEncoder(), details(), args(root), root / "cache")

    def test_weight_source_and_bytes_must_match(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            item = dict(e.ENCODERS["deit_tiny"], sha256="a1311bcf" + "0"*56, bytes=4)
            item["url"] = "https://example.invalid/not-the-official-checkpoint"
            with self.assertRaisesRegex(ValueError, "Weight source identity"):
                e.validate_weight(root, "deit_tiny", {"encoders": {"deit_tiny": item}})
            item["url"] = e.ENCODERS["deit_tiny"]["url"]
            (root / item["filename"]).write_bytes(b"fake")
            with self.assertRaisesRegex(ValueError, "Weight bytes differ"):
                e.validate_weight(root, "deit_tiny", {"encoders": {"deit_tiny": item}})

    def test_pilot_never_accesses_heldout_images_or_saves_features(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def load(encoder, unused):
                record = details()
                record["encoder_id"] = encoder
                return ToyEncoder(), record
            with patch.object(e, "load_encoder", side_effect=load):
                result = e.pilot(ToyCorpus(refuse_heldout=True), root, args(root))
            self.assertFalse(result["heldout_images_used"])
            self.assertFalse(result["scores_computed"])
            self.assertTrue(all(index < 5 for index in result["training_row_indices"]))
            self.assertEqual(len(result["records"]), 2)
            self.assertFalse(list(root.rglob("*.npy")))
            self.assertEqual(json.loads((root / "pilot.json").read_text())["execution"], "training_source_image_inference_pilot")


if __name__ == "__main__":
    unittest.main()
