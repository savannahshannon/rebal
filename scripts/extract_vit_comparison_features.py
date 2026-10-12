#!/usr/bin/env python3
"""Frozen ResNet18/DeiT-tiny features under one explicit 224px transform.

This is source-image inference only: no encoder training, labels in model input,
classification scores, classifier selection, or GAN generation. `pilot` touches
only a fixed label-independent sample from official training images. `extract`
is a separate explicit command. Interrupted caches resume only under an identical
plan and after checking every previously committed feature block.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time
import urllib.request
import zipfile

import numpy as np
from PIL import Image
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms as T
from torchvision.models import resnet18

VERSION = 1
OFFICIAL_ARCHIVE_SHA256 = "6198c8ae015e2b3e007c7841da39ec069199b9aa3bfa943a462022fe5e43c821"
SOURCE_READER_SHA256 = "e48692697a091d2459e9659f1d9440f9830577df0418fab240921701be6a36ef"
SOURCE_URL = "https://cs231n.stanford.edu/tiny-imagenet-200.zip"
ENCODERS = {
    "resnet18": {"url": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
                 "filename": "resnet18-f37072fd.pth", "sha_prefix": "f37072fd",
                 "expected_sha256": "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec",
                 "feature_dim": 512, "weights": "ResNet18_Weights.IMAGENET1K_V1"},
    "deit_tiny": {"url": "https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
                  "filename": "deit_tiny_patch16_224-a1311bcf.pth", "sha_prefix": "a1311bcf",
                  "expected_sha256": "a1311bcf4f24e3c95adaa75535db67bc4412d95535b98f7c1dfd1164dda41c97", "feature_dim": 192,
                  "weights": "official non-distilled deit_tiny_patch16_224 ImageNet-2012"},
}
PREPROCESSING = {
    "input": "PIL RGB converted from original 64x64 image",
    "resize_shorter_side": 256, "interpolation": "PIL bicubic", "antialias": True,
    "center_crop": [224, 224], "scale": "uint8 to float32 divided by 255",
    "mean": [.485, .456, .406], "std": [.229, .224, .225],
    "augmentation": "none; deterministic single view for all source images and both encoders",
    "control_note": "Shared bicubic transform matches DeiT evaluation interpolation; ResNet18 V1 normally uses bilinear. Tiny images are enlarged, not higher-resolution observations.",
}
IDENTITY_FIELDS = ("classes", "train_files", "eval_files", "train_pixel_sha256", "eval_pixel_sha256", "archive_sha256")


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def state_hash(model):
    h = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        h.update(key.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def preprocessing():
    return T.Compose([T.Resize(256, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
                      T.CenterCrop(224), T.ToTensor(), T.Normalize(PREPROCESSING["mean"], PREPROCESSING["std"])])


def acquire_weights(folder, resnet_checkpoint=None):
    """Copy an already verified CNN checkpoint or fetch official static weights.

    Remote code is never imported. Redirect URLs are never stored. A completed
    acquisition is immutable and rechecked instead of silently overwritten.
    """
    marker = folder / "weights.json"
    if marker.exists():
        record = read_json(marker)
        require(record.get("version") == VERSION and set(record.get("encoders", {})) == set(ENCODERS),
                "Existing acquisition has a different format")
        for encoder in ENCODERS:
            validate_weight(folder, encoder, record)
        return record
    require(not folder.exists() or not any(folder.iterdir()), "Incomplete acquisition exists; preserve it and choose a new destination")
    folder.mkdir(parents=True, exist_ok=True)
    records = {}
    for encoder, info in ENCODERS.items():
        target = folder / info["filename"]
        temporary = target.with_name(target.name + ".partial")
        if encoder == "resnet18" and resnet_checkpoint is not None:
            require(sha(resnet_checkpoint) == info["expected_sha256"], "Existing ResNet checkpoint checksum mismatch")
            shutil.copyfile(resnet_checkpoint, temporary)
        else:
            request = urllib.request.Request(info["url"], headers={"User-Agent": "REBAL-reproducibility/1"})
            with urllib.request.urlopen(request, timeout=120) as source, open(temporary, "xb") as dest:
                shutil.copyfileobj(source, dest, length=1024 * 1024)
                dest.flush()
                os.fsync(dest.fileno())
        digest = sha(temporary)
        require(digest.startswith(info["sha_prefix"]), "Official checkpoint hash prefix mismatch: " + encoder)
        require(info["expected_sha256"] is None or digest == info["expected_sha256"], "Official full weight hash mismatch")
        temporary.replace(target)
        records[encoder] = {"filename": info["filename"], "url": info["url"], "sha256": digest,
                            "bytes": target.stat().st_size, "weights": info["weights"],
                            "pretraining": "supervised ImageNet-2012 / ImageNet-1K", "distillation": False,
                            "publisher_hash_prefix": info["sha_prefix"],
                            "full_hash_source": "locally calculated over the downloaded/cached official bytes"}
    record = {"version": VERSION, "created_utc": datetime.now(timezone.utc).isoformat(), "encoders": records,
              "redirect_urls_recorded": False, "remote_code_loaded": False}
    write_json(marker, record)
    return record


def validate_weight(folder, encoder, record):
    info, saved = ENCODERS[encoder], record["encoders"][encoder]
    require(saved["filename"] == info["filename"] and saved["url"] == info["url"], "Weight source identity differs")
    path = folder / saved["filename"]
    require(path.is_file() and not path.is_symlink(), "Weight file absent or symlink")
    digest = sha(path)
    require(digest == saved["sha256"] and digest.startswith(info["sha_prefix"])
            and path.stat().st_size == saved["bytes"], "Weight bytes differ from recorded acquisition")
    require(info["expected_sha256"] is None or digest == info["expected_sha256"], "Expected weight hash differs")
    return path


def load_encoder(encoder, weights_folder):
    require(torch.__version__.split("+")[0] == "2.14.1" and torchvision.__version__.split("+")[0] == "0.29.1",
            "Use the validated torch2.14.1/torchvision0.29.1 environment; validate a new snapshot separately")
    record = read_json(weights_folder / "weights.json")
    path = validate_weight(weights_folder, encoder, record)
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if encoder == "resnet18":
        model = resnet18(weights=None)
        model.load_state_dict(saved, strict=True)
        parameters_with_head = sum(p.numel() for p in model.parameters())
        model.fc = torch.nn.Identity()
        implementation = "torchvision.models.resnet18"
    else:
        import timm
        require(timm.__version__ == "1.0.30", "Use the reviewed timm1.0.30 implementation")
        model = timm.create_model("deit_tiny_patch16_224", pretrained=False)
        model.load_state_dict(saved["model"], strict=True)
        parameters_with_head = sum(p.numel() for p in model.parameters())
        model.reset_classifier(0)
        implementation = "timm.models.deit.deit_tiny_patch16_224"
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    details = {"encoder_id": encoder, "implementation": implementation, "weights": record["encoders"][encoder],
               "weights_sha256": record["encoders"][encoder]["sha256"], "encoder_state_sha256": state_hash(model),
               "feature_dim": ENCODERS[encoder]["feature_dim"], "parameters_with_original_classifier": parameters_with_head,
               "encoder_parameters": sum(p.numel() for p in model.parameters()), "trainable_parameters": 0,
               "encoder_training": False, "encoder_mode": "eval", "pretraining_overlap_risk": True,
               "attention_fused": bool(model.blocks[0].attn.fused_attn) if encoder == "deit_tiny" else None,
               "feature_semantics": "final LayerNorm class-token embedding before classifier; not L2-normalized" if encoder == "deit_tiny" else "global-average-pooled trunk before linear classifier"}
    return model, details


class Corpus(Dataset):
    """Preserve the frozen corpus reader's exact ordering and pixel identities."""
    def __init__(self, archive, original_manifest):
        reader = Path(__file__).with_name("extract_tiny_features.py")
        require(sha(reader) == SOURCE_READER_SHA256, "Frozen source-reader hash differs")
        require(sha(archive) == OFFICIAL_ARCHIVE_SHA256, "Tiny ImageNet archive checksum differs")
        spec = importlib.util.spec_from_file_location("vit_source_reader", reader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        source = module.ArchiveImages(archive)
        original = read_json(original_manifest)
        require(original["archive_sha256"] == OFFICIAL_ARCHIVE_SHA256, "Original manifest archive differs")
        require(source.ntrain == 100000 and len(source) == 110000, "Unexpected corpus dimensions")
        require(source.classes == original["classes"] and source.names[:100000] == original["train_files"]
                and source.names[100000:] == original["eval_files"], "Source ordering differs from original study")
        require(len(original["train_pixel_sha256"]) == 100000 and len(original["eval_pixel_sha256"]) == 10000,
                "Original pixel inventory dimensions differ")
        self.archive = str(archive)
        self.original = original
        self.names, self.labels, self.ntrain = source.names, source.labels.astype(np.int64), source.ntrain
        self.pixel_hashes = original["train_pixel_sha256"] + original["eval_pixel_sha256"]
        self.original_manifest_sha256 = sha(original_manifest)
        self.transform = preprocessing()
        self.z = None

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        if self.z is None:
            self.z = zipfile.ZipFile(self.archive)
        with Image.open(io.BytesIO(self.z.read(self.names[index]))) as opened:
            image = opened.convert("RGB")
            raw = np.asarray(image, dtype=np.uint8)
            require(raw.shape == (64, 64, 3), "Source image shape differs")
            require(hashlib.sha256(raw.tobytes()).hexdigest() == self.pixel_hashes[index],
                    "Original raw-pixel identity mismatch")
            x = self.transform(image)
        return x, int(index)


def synchronize(device):
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def configure(args):
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=args.device != "cpu")
    require(args.device != "mps" or torch.backends.mps.is_available(), "MPS is unavailable")
    require(args.device != "cuda" or torch.cuda.is_available(), "CUDA is unavailable")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False


def plan_for(corpus, details, args):
    plan = {"version": VERSION, "encoder_id": details["encoder_id"], "encoder": details,
            "weights_sha256": details["weights_sha256"], "encoder_state_sha256": details["encoder_state_sha256"],
            "archive_sha256": corpus.original["archive_sha256"], "original_manifest_sha256": corpus.original_manifest_sha256,
            "source_reader_sha256": SOURCE_READER_SHA256, "extractor_source_sha256": sha(__file__),
            "preprocessing": PREPROCESSING, "preprocessing_sha256": fingerprint(PREPROCESSING),
            "ntrain": corpus.ntrain, "n_eval": len(corpus) - corpus.ntrain, "feature_dim": details["feature_dim"],
            "dtype": "float32", "row_order": "original official train followed by original official validation",
            "device": args.device, "batch_size": args.batch_size, "workers": args.workers, "threads": args.threads,
            "software": {name: importlib.metadata.version(name) for name in ("torch", "torchvision", "timm", "numpy", "Pillow")},
            "runtime": {"python": platform.python_version(), "OS": platform.system(), "architecture": platform.machine()},
            "determinism": "strict CPU; other backends warn for unsupported deterministic operations; cross-device equality not claimed"}
    plan["fingerprint"] = fingerprint(plan)
    return plan


def array_digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def validate_progress(features, progress, plan):
    require(progress["plan"] == plan["fingerprint"], "Partial cache plan differs")
    expected = 0
    for block in progress["blocks"]:
        require(block["start"] == expected and expected < block["end"] <= len(features), "Partial block ordering differs")
        require(array_digest(features[block["start"]:block["end"]]) == block["sha256"], "Partial feature block checksum mismatch")
        expected = block["end"]
    require(progress["next_row"] == expected, "Partial cache row marker differs")
    return expected


def extract_cache(corpus, model, details, args, target):
    """Transactional sequential cache, with committed-block integrity on resume."""
    plan = plan_for(corpus, details, args)
    target.mkdir(parents=True, exist_ok=True)
    # Prevent concurrent writers, including other processes that share a cache.
    import fcntl
    lock = open(target / "extract.lock", "a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise ValueError("Another extractor is writing this cache") from None
    try:
        marker, plan_path = target / "manifest.json", target / "extraction_plan.json"
        if marker.exists():
            saved = read_json(marker)
            require(saved["plan_fingerprint"] == plan["fingerprint"], "Completed cache plan differs")
            for name, digest in saved["artifacts"].items():
                require(name in ("features.npy", "labels.npy") and sha(target / name) == digest, "Completed cache artifact mismatch")
            require(set(saved["artifacts"]) == {"features.npy", "labels.npy"}, "Completed artifact inventory differs")
            return saved
        temporary = target / "features.npy.partial"
        final = target / "features.npy"
        progress_path = target / "progress.json"
        shape = (len(corpus), details["feature_dim"])
        if plan_path.exists():
            require(read_json(plan_path) == plan, "Partial cache belongs to a different source/weight/configuration")
            require(progress_path.exists() and (temporary.exists() ^ final.exists()), "Incomplete cache transaction cannot be resumed safely")
            labels = np.load(target / "labels.npy", allow_pickle=False)
            require(np.array_equal(labels, corpus.labels), "Partial labels differ from original corpus")
            features = np.load(temporary if temporary.exists() else final, mmap_mode="r+", allow_pickle=False)
            require(features.shape == shape and features.dtype == np.float32, "Partial feature dimensions differ")
            progress = read_json(progress_path)
            offset = validate_progress(features, progress, plan)
            require(not final.exists() or offset == len(corpus), "Prematurely renamed feature cache")
        else:
            require(set(p.name for p in target.iterdir()) <= {"extract.lock"}, "Nonempty cache without an extraction plan")
            write_json(plan_path, plan)
            with open(target / "labels.npy", "xb") as stream:
                np.save(stream, corpus.labels.astype(np.int64), allow_pickle=False)
                stream.flush()
                os.fsync(stream.fileno())
            features = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32, shape=shape)
            progress = {"plan": plan["fingerprint"], "next_row": 0, "blocks": [], "inference_seconds": 0.}
            write_json(progress_path, progress)
            offset = 0
        model.eval().to(args.device)
        indices = range(offset, len(corpus))
        loader = DataLoader(Subset(corpus, indices), batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        synchronize(args.device)
        started = time.perf_counter()
        with torch.inference_mode():
            for x, row_ids in loader:
                require(np.array_equal(row_ids.numpy(), np.arange(offset, offset + len(x))), "Inference row ordering differs")
                y = model(x.to(args.device)).float().cpu().numpy()
                require(y.shape == (len(x), shape[1]) and np.isfinite(y).all(), "Encoder produced invalid features")
                features[offset:offset+len(x)] = y
                features.flush()
                with open(temporary, "rb") as stream:
                    os.fsync(stream.fileno())
                progress["blocks"].append({"start": offset, "end": offset+len(x), "sha256": array_digest(y)})
                offset += len(x)
                progress["next_row"] = offset
                synchronize(args.device)
                now = time.perf_counter()
                progress["inference_seconds"] += now-started
                write_json(progress_path, progress)
                started = time.perf_counter()
                if len(progress["blocks"]) % 25 == 0 or offset == len(corpus):
                    print(json.dumps({"encoder": details["encoder_id"], "rows": offset, "total": len(corpus)}), flush=True)
        validate_progress(features, progress, plan)
        require(offset == len(corpus), "Extraction ended before completing every row")
        del features
        if temporary.exists():
            temporary.replace(final)
        saved = {**{k: corpus.original[k] for k in IDENTITY_FIELDS},
                 "version": VERSION, "source_url": SOURCE_URL, "created_utc": datetime.now(timezone.utc).isoformat(),
                 "encoder_id": details["encoder_id"], "feature_dim": details["feature_dim"],
                 "encoder_parameter_count": details.get("encoder_parameters"),
                 "encoder_parameter_count_scope": "Frozen backbone used for embeddings; original ImageNet classifier excluded",
                 "ntrain": corpus.ntrain, "n_eval": len(corpus)-corpus.ntrain,
                 "weights_sha256": details["weights_sha256"], "encoder_state_sha256": details["encoder_state_sha256"],
                 "encoder": details, "preprocessing": PREPROCESSING, "preprocessing_sha256": fingerprint(PREPROCESSING),
                 "extractor_source_sha256": sha(__file__), "source_reader_sha256": SOURCE_READER_SHA256,
                 "original_manifest_sha256": corpus.original_manifest_sha256, "plan_fingerprint": plan["fingerprint"],
                 "extraction_plan": plan, "extraction_seconds": progress["inference_seconds"],
                 "encoder_training": False, "pretraining_overlap_risk": True, "evaluation_scores_computed": False,
                 "artifacts": {name: sha(target/name) for name in ("features.npy", "labels.npy")}}
        write_json(marker, saved)
        return saved
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def pilot(corpus, weights_folder, args):
    require(not args.out.exists(), "Pilot output already exists")
    require(0 < args.pilot_images <= corpus.ntrain, "Invalid training-image pilot size")
    # Image selection is independent of labels; validation images are excluded.
    indices = np.random.default_rng(1729).choice(corpus.ntrain, args.pilot_images, replace=False)
    reports = []
    for encoder in args.encoders:
        model, details = load_encoder(encoder, weights_folder)
        model.to(args.device)
        loader = DataLoader(Subset(corpus, indices), batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        with torch.inference_mode():
            warm, warm_ids = next(iter(loader))
            require(int(warm_ids.max()) < corpus.ntrain, "Pilot cannot use held-out images")
            model(warm.to(args.device))
            model(warm.to(args.device))
            synchronize(args.device)
            start = time.perf_counter()
            seen = 0
            for x, ids in loader:
                require(int(ids.min()) >= 0 and int(ids.max()) < corpus.ntrain, "Pilot cannot use held-out images")
                output = model(x.to(args.device))
                require(output.shape == (len(x), details["feature_dim"]) and torch.isfinite(output).all().item(),
                        "Invalid pilot features")
                seen += len(x)
            synchronize(args.device)
            seconds = time.perf_counter()-start
        reports.append({"encoder": details, "images": seen, "seconds": seconds, "images_per_second": seen/seconds,
                        "linear_110k_seconds_excluding_write_hash_overhead": len(corpus)*seconds/seen,
                        "feature_values_saved": False})
        del model
        if args.device == "mps":
            torch.mps.empty_cache()
    result = {"version": VERSION, "execution": "training_source_image_inference_pilot", "scores_computed": False,
              "heldout_images_used": False, "encoder_training": False, "training_row_indices": indices.tolist(),
              "device": args.device, "batch_size": args.batch_size, "workers": args.workers, "threads": args.threads,
              "preprocessing": PREPROCESSING, "preprocessing_sha256": fingerprint(PREPROCESSING),
              "archive_sha256": corpus.original["archive_sha256"], "original_manifest_sha256": corpus.original_manifest_sha256,
              "extractor_source_sha256": sha(__file__), "records": reports,
              "limits": "Short inference-only estimate under concurrent system load; excludes full-cache writes/hashes and adapter/head training."}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    acquire = sub.add_parser("acquire")
    acquire.add_argument("--weights-dir", type=Path, required=True)
    acquire.add_argument("--resnet-checkpoint", type=Path)
    for command in ("pilot", "extract"):
        p = sub.add_parser(command)
        p.add_argument("--archive", type=Path, required=True)
        p.add_argument("--original-manifest", type=Path, required=True)
        p.add_argument("--weights-dir", type=Path, required=True)
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--encoders", nargs="+", choices=tuple(ENCODERS), default=list(ENCODERS))
        p.add_argument("--device", choices=("cpu", "mps", "cuda"), default="mps")
        p.add_argument("--batch-size", type=int, default=64)
        p.add_argument("--workers", type=int, default=0)
        p.add_argument("--threads", type=int, default=1)
        if command == "pilot":
            p.add_argument("--pilot-images", type=int, default=512)
    args = parser.parse_args()
    try:
        if args.command == "acquire":
            result = acquire_weights(args.weights_dir, args.resnet_checkpoint)
            print(json.dumps({"status": "weights_acquired_and_hashed", "encoders": result["encoders"]}))
            return 0
        require(args.batch_size > 0 and args.workers >= 0 and args.threads > 0 and len(set(args.encoders)) == len(args.encoders),
                "Invalid extraction controls")
        configure(args)
        corpus = Corpus(args.archive, args.original_manifest)
        if args.command == "pilot":
            result = pilot(corpus, args.weights_dir, args)
            print(json.dumps({"status": "source_image_pilot_complete", "scores_computed": False,
                              "records": [{k: r[k] for k in ("images", "seconds", "images_per_second", "linear_110k_seconds_excluding_write_hash_overhead")} for r in result["records"]]}))
        else:
            for encoder in args.encoders:
                model, details = load_encoder(encoder, args.weights_dir)
                result = extract_cache(corpus, model, details, args, args.out / encoder)
                print(json.dumps({"status": "feature_cache_complete", "encoder": encoder, "artifacts": result["artifacts"],
                                  "scores_computed": False}), flush=True)
                del model
                if args.device == "mps":
                    torch.mps.empty_cache()
        return 0
    except Exception as error:
        reason = f"OS error {error.errno}: {error.strerror}" if isinstance(error, OSError) else str(error)
        print(json.dumps({"status": "failed", "error_type": type(error).__name__, "reason": reason, "scores_computed": False}))
        return 1


if __name__ == "__main__":
    sys.exit(main())
