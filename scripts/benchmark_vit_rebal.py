#!/usr/bin/env python3
"""Matched frozen DeiT-tiny/ResNet18 feature-adapter REBAL experiment.

This is not end-to-end ViT training or a causal GAN-versus-ViT comparison.
Official encoders are extracted separately; this script has no network access.
Both encoders use identical Tiny ImageNet source rows, preprocessing and splits.

  python scripts/benchmark_vit_rebal.py --features data/resnet18 data/deit_tiny \
    --out experiments/vit_adapter --device cpu --aggregate-only
  python scripts/benchmark_vit_rebal.py --features data/resnet18 data/deit_tiny \
    --out experiments/vit_adapter --device cpu

Default: 2 encoders x 3 ratios x 5 seeds = 30 cells, 90 adapter trunks,
270 retrained heads and 810 named readouts, including 180 balanced-LA aliases.
One 15-comparison family is corrected only after the complete grid finishes.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import sys
import time

import numpy as np
from scipy import stats
import torch
from torch import nn
from torch.nn import functional as F


def _module(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


common = _module("vit_adapter_common", "train_tiny_fullscale.py")
nonimage = _module("vit_adapter_nonimage", "train_nonimage.py")
split_helper = _module("vit_adapter_splits", "benchmark_tiny_features.py")
sha, fingerprint, write_json, save_npz = common.sha, common.fingerprint, common.write_json, common.save_npz
HELPER_HASHES = {
    "train_tiny_fullscale.py": "3d9de9336c7dc636f641286eddc0e9290f532048ea669d5eb934ed7ef123b61d",
    "train_nonimage.py": "be58b2706400b85340736c0ca0e3f07cbcb23895c4d07e92a6789de86d072b21",
    "benchmark_tiny_features.py": "315a900f5d4a4bb48db66e8508b3f19b902df0b906200c2091efada95d839309",
}
ENCODERS = ("resnet18", "deit_tiny")
DIMENSIONS = {"resnet18": 512, "deit_tiny": 192}
WEIGHT_HASHES = {
    "resnet18": "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec",
    "deit_tiny": "a1311bcf4f24e3c95adaa75535db67bc4412d95535b98f7c1dfd1164dda41c97",
}
SOURCE_READER_HASH = "e48692697a091d2459e9659f1d9440f9830577df0418fab240921701be6a36ef"
TRUNKS = ("ce", "cb", "rebal")
HEADS = ("natural", "ros", "smote")
METHODS = nonimage.required_methods()
BASE_METHODS = METHODS - {name for name in METHODS if name.endswith("_la")}
METRICS = ("accuracy", "macro_f1", "macro_recall", "head_f1", "mid_f1", "tail_f1")
ROW_FIELDS = ("classes", "train_files", "eval_files", "train_pixel_sha256", "eval_pixel_sha256", "archive_sha256", "ntrain", "n_eval", "original_manifest_sha256", "source_reader_sha256")
VERSION = 1


def require(condition, message):
    if not condition:
        raise ValueError(message)


def safe_file(root, name):
    p = PurePosixPath(name)
    require(name and not p.is_absolute() and ".." not in p.parts and "\\" not in name, "Unsafe relative artifact identity")
    result = Path(root) / name
    require(result.is_file() and not result.is_symlink() and result.resolve().is_relative_to(Path(root).resolve()), "Missing or unsafe artifact: " + name)
    return result


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def verify_hashes(root, inventory, required=None):
    if required is not None:
        require(set(inventory) == set(required), "Artifact inventory differs")
    for name, digest in inventory.items():
        require(sha(safe_file(root, name)) == digest, "Artifact checksum mismatch: " + name)


def verify_helpers():
    for name, digest in HELPER_HASHES.items():
        require(sha(Path(__file__).with_name(name)) == digest, "Frozen helper source differs: " + name)


def validate_cache_pair(caches, production=True):
    """Validate cache identity/row alignment; production enforces official scale."""
    require(set(caches) == set(ENCODERS), "Both declared encoder caches are required")
    a, b = (caches[name] for name in ENCODERS)
    ma, mb = a["manifest"], b["manifest"]
    require(all(ma[k] == mb[k] for k in ROW_FIELDS), "Encoder cache source row identities differ")
    require(ma["preprocessing"] == mb["preprocessing"] and ma["preprocessing_sha256"] == mb["preprocessing_sha256"], "Encoder preprocessing differs")
    require(np.array_equal(a["labels"], b["labels"]), "Encoder labels/order differ")
    for name, record in caches.items():
        m, x, y = record["manifest"], record["features"], record["labels"]
        require(m["encoder_id"] == name and fingerprint(m["preprocessing"]) == m["preprocessing_sha256"], "Invalid encoder/preprocessing identity")
        c = len(m["classes"]); ntrain, neval = m["ntrain"], m["n_eval"]
        require(c >= 3 and m["classes"] == sorted(set(m["classes"])), "Invalid sorted class inventory")
        require(x.shape == (ntrain + neval, m["feature_dim"]) and x.dtype == np.float32 and x.ndim == 2, "Feature dimensions/type differ")
        require(y.shape == (len(x),) and y.dtype == np.int64 and y.min() == 0 and y.max() == c-1, "Label dimensions/type/range differ")
        for part, count in (("train", ntrain), ("eval", neval)):
            require(len(m[part+"_files"]) == len(set(m[part+"_files"])) == len(m[part+"_pixel_sha256"]) == count, "Filename/pixel-hash inventory differs")
        require(not (set(m["train_files"]) & set(m["eval_files"])), "Source partitions repeat a filename")
        require(all(isinstance(m.get(k), str) and len(m[k]) == 64 and all(ch in "0123456789abcdef" for ch in m[k])
                    for k in ("weights_sha256", "encoder_state_sha256")), "Missing encoder weight/state hashes")
        for begin in range(0, len(x), 65536):
            require(np.isfinite(x[begin:begin+65536]).all(), "Nonfinite frozen feature")
        if production:
            require((ntrain, neval, c, m["feature_dim"]) == (100000, 10000, 200, DIMENSIONS[name]), "Unexpected production dataset or encoder dimensions")
            require(m["archive_sha256"] == common.OFFICIAL_SHA, "Unexpected Tiny ImageNet source archive")
            require(m["weights_sha256"] == WEIGHT_HASHES[name] and m["source_reader_sha256"] == SOURCE_READER_HASH, "Encoder weights or frozen source reader differ")
            require(sha(Path(__file__).with_name("extract_tiny_features.py")) == SOURCE_READER_HASH, "Current frozen source-reader bytes differ")
            require(m["extractor_source_sha256"] == sha(Path(__file__).with_name("extract_vit_comparison_features.py")), "Cache extractor source differs from current frozen source")
            require(m["encoder_training"] is False and m["evaluation_scores_computed"] is False and
                    isinstance(m["encoder_parameter_count"], int) and m["encoder_parameter_count"] > 0, "Cache is not a documented frozen-encoder extraction")
            transform = m["preprocessing"]
            require(transform["resize_shorter_side"] == 256 and transform["center_crop"] == [224,224] and
                    transform["interpolation"] == "PIL bicubic" and transform["antialias"] is True and
                    transform["mean"] == [.485,.456,.406] and transform["std"] == [.229,.224,.225] and
                    transform["augmentation"].startswith("none;"), "Preprocessing differs from shared224 protocol")
            require(np.array_equal(np.bincount(y[:ntrain], minlength=c), np.full(c, 500)) and
                    np.array_equal(np.bincount(y[ntrain:], minlength=c), np.full(c, 50)), "Official class supports differ")
    return fingerprint({key: ma[key] for key in ROW_FIELDS})


def load_caches(paths):
    caches = {}
    for directory in map(Path, paths):
        m = read_json(safe_file(directory, "manifest.json")); name = m["encoder_id"]
        require(name in ENCODERS and name not in caches, "Unexpected or repeated encoder cache")
        verify_hashes(directory, m["artifacts"], {"features.npy", "labels.npy"})
        caches[name] = dict(directory=directory, manifest=m, manifest_sha256=sha(directory/"manifest.json"),
                            features=np.load(directory/"features.npy", mmap_mode="r", allow_pickle=False),
                            labels=np.load(directory/"labels.npy", allow_pickle=False))
    row_identity = validate_cache_pair(caches)
    first = caches[ENCODERS[0]]; m = first["manifest"]
    eligible, hygiene = split_helper.clean_training_indices(m, first["labels"][:m["ntrain"]])
    return caches, row_identity, eligible, hygiene


def cell_name(encoder, rho):
    return encoder + "__rho" + format(float(rho), ".12g")


def make_shared_split(caches, eligible, seed, rho, holdout):
    first = caches[ENCODERS[0]]; y, m = first["labels"], first["manifest"]
    split = split_helper.make_split(y[:m["ntrain"]], seed, rho, holdout, eligible)
    split["evaluation_indices"] = np.arange(m["ntrain"], len(y), dtype=np.int64)
    return split


def ensure_shared_split(caches, eligible, args, protocol, seed, rho):
    folder = args.out / "shared_splits" / ("rho" + format(float(rho), ".12g")) / f"seed_{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    expected = dict(protocol=protocol["fingerprint"], row_identity=protocol["row_identity_sha256"], seed=seed, rho=float(rho))
    with open(folder/"write.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        split = make_shared_split(caches, eligible, seed, rho, args.holdout_per_class)
        marker, data = folder/"identity.json", folder/"split.npz"
        if marker.exists():
            saved = read_json(marker)
            require(saved["identity"] == expected and sha(data) == saved["split_sha256"], "Shared split identity/checksum mismatch")
            with np.load(data, allow_pickle=False) as z:
                require(set(z.files) == set(split) and all(np.array_equal(z[k], value) for k, value in split.items()), "Stored split differs from frozen helper")
        else:
            save_npz(data, **split); write_json(marker, dict(identity=expected, split_sha256=sha(data)))
    return split, sha(data), data


class StandardizedFeatures:
    """Transform only requested rows without duplicating full feature caches."""
    def __init__(self, raw, mean, scale):
        self.raw, self.mean, self.scale = raw, np.asarray(mean), np.asarray(scale)
        self.shape = raw.shape

    def __getitem__(self, ids):
        return ((np.asarray(self.raw[ids], dtype=np.float64)-self.mean)/self.scale).astype(np.float32)


def prepare_scaler(raw, fit, target, identity):
    target.mkdir(parents=True, exist_ok=True)
    marker, artifact = target/"scaler.json", target/"scaler.npz"
    if marker.exists():
        info = read_json(marker)
        require(info["identity"] == identity and info["artifact_sha256"] == sha(artifact), "Scaler identity/checksum mismatch")
        with np.load(artifact, allow_pickle=False) as z:
            require(set(z.files) == {"mean", "scale"}, "Scaler fields differ")
            mean, scale = z["mean"], z["scale"]
    else:
        values = np.asarray(raw[fit], dtype=np.float64)
        mean, scale = values.mean(0), values.std(0); scale[scale == 0] = 1.
        save_npz(artifact, mean=mean, scale=scale)
        info = dict(identity=identity, artifact_sha256=sha(artifact), fit_count=len(fit),
                    scope="per-dimension population mean/std fitted only on this cell's retained training rows", feature_dim=raw.shape[1])
        write_json(marker, info)
    require(mean.shape == scale.shape == (raw.shape[1],) and np.isfinite(mean).all() and np.isfinite(scale).all() and (scale>0).all(), "Invalid fitted scaler")
    return StandardizedFeatures(raw, mean, scale), info


class Adapter(nn.Module):
    def __init__(self, input_dim, classes, seed):
        super().__init__()
        common.set_seed(seed)
        self.layers = nn.Sequential(nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(.1),
                                    nn.Linear(256, 128), nn.ReLU(), nn.Dropout(.1))
        self.layers.apply(common.initialize)
        common.set_seed(seed+90909)
        self.head = nn.Linear(128, classes); common.initialize(self.head)
        # Matching dropout stream independent of input-dimensional parameter count.
        common.set_seed(seed+40404)

    def forward(self, x):
        features = self.layers(x)
        return self.head(features), features


def train_adapter(x, y, fit, classes, method, seed, args, target, identity):
    require(method in TRUNKS, "Unknown adapter objective")
    target.mkdir(parents=True, exist_ok=True); checkpoint = target/"checkpoint.pt"
    model = Adapter(x.shape[1], classes, seed).to(args.device)
    initial, initial_head = common.state_hash(model.state_dict()), common.state_hash(model.head.state_dict())
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=.9, nesterov=True)
    counts = np.bincount(y[fit], minlength=classes); weights, beta = nonimage.cb_weights(counts)
    cw = torch.as_tensor(weights, device=args.device)
    steps = math.ceil(len(fit)/args.batch_size); total = steps*args.epochs; warmup = min(args.warmup_epochs, args.epochs)*steps
    start, history, elapsed = 0, [], 0.
    if checkpoint.exists():
        ck = torch.load(checkpoint, map_location=args.device, weights_only=False)
        require(ck["identity"] == identity and ck["method"] == method and ck["initial_state_sha256"] == initial, "Adapter checkpoint identity mismatch")
        require(0 <= ck["completed_epochs"] <= args.epochs and len(ck["history"]) == ck["completed_epochs"], "Invalid checkpoint epoch history")
        model.load_state_dict(ck["model"]); optimizer.load_state_dict(ck["optimizer"]); common.restore_rng(ck["rng"])
        start, history, elapsed = ck["completed_epochs"], ck["history"], ck["training_seconds"]
    for epoch in range(start, args.epochs):
        model.train(); order = np.random.default_rng(np.random.SeedSequence([seed, epoch, 707])).permutation(fit)
        began = time.perf_counter(); loss_sum = 0.; eligibility = np.zeros(classes, dtype=np.int64); batches = 0
        for offset in range(0, len(order), args.batch_size):
            ids = order[offset:offset+args.batch_size]
            batch = nonimage.input_batch(x, ids, "tabular", args.device); label = torch.as_tensor(np.array(y[ids]), device=args.device)
            step = epoch*steps+offset//args.batch_size
            for group in optimizer.param_groups:
                group["lr"] = common.learning_rate(step, total, warmup, args.lr)
            optimizer.zero_grad(set_to_none=True); logits, features = model(batch)
            loss = F.cross_entropy(logits, label, reduction="none")
            if method != "ce" and epoch >= int(.8*args.epochs): loss = loss*cw[label]
            loss = loss.mean()+.0005*nonimage.linear_l2(model)
            if method == "rebal" and epoch >= int(.8*args.epochs): loss = loss+.05*common.equalization(features, label, classes)
            require(bool(torch.isfinite(loss)), "Nonfinite adapter loss")
            loss.backward(); gradients = [p.grad for p in model.parameters() if p.grad is not None]
            norms = torch.stack([g.norm() for g in gradients]); require(bool(torch.isfinite(norms).all()), "Nonfinite adapter gradient")
            for gradient, norm in zip(gradients, norms): gradient.mul_((1./(norm+1e-6)).clamp(max=1.))
            optimizer.step(); loss_sum += float(loss.detach().cpu())*len(ids)
            eligibility += np.bincount(y[ids], minlength=classes)>=2; batches += 1
        duration = time.perf_counter()-began; elapsed += duration
        row = dict(epoch=epoch+1, optimizer_updates=(epoch+1)*steps, loss=loss_sum/len(fit), seconds=duration,
                   EQ_eligible_batches_per_class=eligibility.tolist(), batches=batches,
                   EQ_active=method=="rebal" and epoch>=int(.8*args.epochs))
        history.append(row)
        print(json.dumps(dict(stage="adapter", condition=identity["condition"], seed=seed, method=method, **row)), flush=True)
        if (epoch+1)%args.checkpoint_every == 0 or epoch+1 == args.epochs:
            common.save_torch(checkpoint, dict(identity=identity, method=method, initial_state_sha256=initial,
                initial_head_sha256=initial_head, model=model.state_dict(), optimizer=optimizer.state_dict(), rng=common.rng_state(),
                completed_epochs=epoch+1, history=history, training_seconds=elapsed))
            write_json(target/"history.json", history)
    # A process can stop after the atomic final checkpoint but before the JSON
    # sidecar. Recover that sidecar from the checkpoint's authoritative history.
    write_json(target/"history.json", history)
    prior = counts/counts.sum() if method=="ce" else counts*weights/np.sum(counts*weights)
    return model, dict(initial_state_sha256=initial, initial_head_sha256=initial_head,
                       final_state_sha256=common.state_hash(model.state_dict()), completed_epochs=args.epochs,
                       adapter_parameters=sum(p.numel() for p in model.layers.parameters()),
                       native_head_parameters=sum(p.numel() for p in model.head.parameters()),
                       optimizer_updates=total, training_seconds=elapsed, beta=beta, class_weights=weights.tolist(),
                       expected_sample_weight=float(np.dot(counts, weights)/counts.sum()), LA_prior=prior.tolist(),
                       LA_prior_note="Natural CE prior; final weighted-objective prior approximates deferred CB/CB+EQ training")


class VectorizedLatentSampler(nonimage.LatentSampler):
    """Same five RNG streams as the frozen sampler, with vectorized lookup.

    Concatenating the original ordered membership arrays preserves every anchor
    selected by floor(v*n_class), including interleaved row labels and ties in
    class size. Only the Python class loop is removed; sampling is unchanged.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.flat_members = np.concatenate(self.members)
        self.class_offsets = np.concatenate(([0], np.cumsum(self.counts[:-1])))

    def sample(self, kind, size):
        u, v, w, z, alpha = self.rng.random((5, size))
        if kind == "natural":
            base = (u*len(self.labels)).astype(np.int64); classes = self.labels[base]
        else:
            classes = (u*len(self.members)).astype(np.int64)
            positions = self.class_offsets[classes]+(v*self.counts[classes]).astype(np.int64)
            base = self.flat_members[positions]
        synthetic = (w>=self.counts[classes]/self.counts.max()) if kind=="smote" else np.zeros(size,dtype=bool)
        neighbor = base.copy()
        if synthetic.any():
            neighbor[synthetic] = self.graph[base[synthetic], (z[synthetic]*self.graph.shape[1]).astype(np.int64)]
        require(not (neighbor<0).any() and np.array_equal(self.labels[neighbor],classes), "Invalid cross-class SMOTE neighbor")
        base_t, neighbor_t = torch.as_tensor(base,device=self.device), torch.as_tensor(neighbor,device=self.device)
        amount = torch.as_tensor(alpha*synthetic,dtype=torch.float32,device=self.device)[:,None]
        sample = self.features[base_t]+amount*(self.features[neighbor_t]-self.features[base_t])
        return sample, torch.as_tensor(classes,device=self.device), dict(base=base,neighbor=neighbor,alpha=alpha*synthetic,synthetic=synthetic)


def train_head(features, labels, graph, kind, seed, args, classes, target, identity):
    """Resume fixed-update head training, including the streaming sampler state."""
    target.parent.mkdir(parents=True, exist_ok=True); common.set_seed(seed+90909)
    head = nn.Linear(128, classes).to(args.device); common.initialize(head)
    initial = common.state_hash(head.state_dict()); optimizer = torch.optim.SGD(head.parameters(), lr=.01, momentum=.9)
    sampler = VectorizedLatentSampler(features, labels, graph, args.device, seed+60606)
    start, elapsed, history = 0, 0., []
    if target.exists():
        ck = torch.load(target, map_location=args.device, weights_only=False)
        require(ck["identity"] == identity and ck["kind"] == kind and ck["initial_state_sha256"] == initial, "Head checkpoint identity mismatch")
        require(0 <= ck["completed_updates"] <= args.head_steps, "Invalid head checkpoint budget")
        head.load_state_dict(ck["model"]); optimizer.load_state_dict(ck["optimizer"])
        sampler.rng.bit_generator.state = ck["sampler_rng"]; common.restore_rng(ck["rng"])
        start, elapsed, history = ck["completed_updates"], ck["seconds"], ck["losses"]
    began = time.perf_counter()
    for step in range(start, args.head_steps):
        batch, labels_t, _ = sampler.sample(kind, args.head_batch_size); optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(head(batch), labels_t)+.0005*head.weight.square().sum()
        require(bool(torch.isfinite(loss)), "Nonfinite head loss"); loss.backward()
        require(all(bool(torch.isfinite(p.grad).all()) for p in head.parameters()), "Nonfinite head gradient")
        optimizer.step()
        if step == 0 or (step+1)%100 == 0 or step+1 == args.head_steps:
            history.append(dict(update=step+1, loss=float(loss.detach().cpu())))
        if (step+1)%args.head_checkpoint_every == 0 or step+1 == args.head_steps:
            elapsed += time.perf_counter()-began
            common.save_torch(target, dict(identity=identity, kind=kind, initial_state_sha256=initial,
                completed_updates=step+1, model=head.state_dict(), optimizer=optimizer.state_dict(),
                rng=common.rng_state(), sampler_rng=sampler.rng.bit_generator.state, seconds=elapsed, losses=history))
            began = time.perf_counter()
    prior = np.bincount(labels, minlength=classes).astype(float) if kind=="natural" else np.ones(classes)
    prior /= prior.sum()
    return head, dict(initial_state_sha256=initial, final_state_sha256=common.state_hash(head.state_dict()),
                      head_parameters=sum(p.numel() for p in head.parameters()),
                      optimizer_updates=args.head_steps, seconds=elapsed, losses=history, training_prior=prior.tolist(),
                      sampling="natural uniform rows" if kind=="natural" else "class-uniform streaming "+kind,
                      smote_variant="fresh within-class interpolation, real probability n_class/n_max" if kind=="smote" else None)


def required_artifacts():
    return {"split.npz", "scaler.npz", "scaler.json", "predictions.npz"} | {
        f"{trunk}/{name}" for trunk in TRUNKS for name in
        ("checkpoint.pt", "history.json", "knn_graph.npy", "knn.json", "head_natural.pt", "head_ros.pt", "head_smote.pt")}


def validate_result(result, cell, identity, args):
    require(result["identity"] == identity, "Completed cell identity mismatch")
    require(set(result["metrics"]) == METHODS and set(result["trunks"]) == set(TRUNKS), "Incomplete method inventory")
    verify_hashes(cell, result["artifacts"], required_artifacts())
    require(len({r["initial_state_sha256"] for r in result["trunks"].values()}) == 1, "Unmatched objective initialization")
    heads = {r["initial_head_sha256"] for r in result["trunks"].values()}
    for record in result["trunks"].values():
        require(record["completed_epochs"] == args.epochs and record["optimizer_updates"] == args.epochs*math.ceil(result["fit_count"]/args.batch_size), "Incomplete adapter budget")
    for name, record in result["metrics"].items():
        require(np.isfinite([v for v in record.values() if isinstance(v, (float, int))]+record["per_class_f1"]+record["per_class_recall"]).all(), "Nonfinite result metric")
        if "_crt_" in name and not name.endswith("_la"):
            require(record["training"]["optimizer_updates"] == args.head_steps, "Incomplete classifier budget")
            heads.add(record["training"]["initial_state_sha256"])
    require(len(heads) == 1, "Native/retrained head initializations differ")
    require(set(result["prediction_storage"]) == METHODS, "Missing reconstructable prediction view")
    for name in METHODS:
        storage = result["prediction_storage"][name]
        require(storage == (dict(base=name[:-3], operation="logit_adjustment", adjustment_in_metrics=True)
                           if name.endswith("_la") else dict(base=name, operation="stored")), "Prediction storage recipe differs")
    with np.load(cell/"predictions.npz", allow_pickle=False) as z:
        expected = {"test_labels", "calibration_labels", "class_rank"} | {name+"__"+part+"_logits" for name in BASE_METHODS for part in ("test", "calibration")}
        require(set(z.files) == expected, "Prediction archive inventory differs")


def run_cell(caches, eligible, encoder, rho, seed, args, protocol):
    record = caches[encoder]; m, y, raw = record["manifest"], record["labels"], record["features"]
    key = cell_name(encoder, rho); cell = args.out/key/f"seed_{seed}"
    cell.mkdir(parents=True, exist_ok=True)
    with open(cell/"run.lock", "a") as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: raise ValueError("Another worker already owns this cell") from None
        split, split_hash, split_path = ensure_shared_split(caches, eligible, args, protocol, seed, rho)
        identity = dict(protocol=protocol["fingerprint"], condition=key, encoder=encoder, rho=float(rho), seed=seed,
                        split_sha256=split_hash, cache_manifest_sha256=record["manifest_sha256"])
        result_path = cell/"metrics.json"
        if result_path.exists():
            result = read_json(result_path); validate_result(result, cell, identity, args); return result
        if (cell/"split.npz").exists():
            require(sha(cell/"split.npz") == split_hash, "Cell split changed")
        else:
            (cell/"split.npz").write_bytes(split_path.read_bytes())
        fit, cal, test = (split[k] for k in ("train_indices", "holdout_indices", "evaluation_indices"))
        x, scaler = prepare_scaler(raw, fit, cell, identity); classes = len(m["classes"]); rank = split["class_rank"]
        metrics, trunks, storage = {}, {}, {}
        predictions = dict(test_labels=y[test], calibration_labels=y[cal], class_rank=rank)
        def record_logits(name, calibration_logits, test_logits, prior=None):
            metrics[name] = nonimage.scores(y[test], test_logits.argmax(1), rank)
            predictions[name+"__test_logits"] = test_logits; predictions[name+"__calibration_logits"] = calibration_logits
            storage[name] = dict(base=name, operation="stored")
            if prior is not None:
                cl, tl, adjustment = nonimage.adjust(calibration_logits, y[cal], test_logits, prior)
                metrics[name+"_la"] = nonimage.scores(y[test], tl.argmax(1), rank)
                metrics[name+"_la"]["adjustment"] = adjustment
                storage[name+"_la"] = dict(base=name, operation="logit_adjustment", adjustment_in_metrics=True)
        for trunk in TRUNKS:
            folder = cell/trunk
            model, info = train_adapter(x, y, fit, classes, trunk, seed, args, folder, identity); trunks[trunk] = info
            _, fit_features = nonimage.encode(model, x, fit, "tabular", args)
            cl, cf = nonimage.encode(model, x, cal, "tabular", args); tl, tf = nonimage.encode(model, x, test, "tabular", args)
            record_logits(trunk, cl, tl, info["LA_prior"])
            record_logits(trunk+"_ncm", common.ncm_logits(fit_features, y[fit], cf, classes), common.ncm_logits(fit_features, y[fit], tf, classes))
            graph_identity = dict(cell=identity, trunk=trunk, features_sha256=hashlib.sha256(fit_features.tobytes()).hexdigest())
            if (folder/"knn.json").exists():
                graph_info = read_json(folder/"knn.json")
                require(graph_info["identity"] == graph_identity and graph_info["graph_sha256"] == sha(folder/"knn_graph.npy"), "Neighbor graph identity/checksum mismatch")
                graph = np.load(folder/"knn_graph.npy", allow_pickle=False)
            else:
                began = time.perf_counter()
                graph, graph_info = nonimage.neighbor_graph(fit_features, y[fit], classes, args.smote_k, args.knn_device, args.knn_query_block, args.knn_reference_block)
                np.save(folder/"knn_graph.npy", graph)
                graph_info.update(identity=graph_identity, graph_sha256=sha(folder/"knn_graph.npy"), seconds=time.perf_counter()-began)
                write_json(folder/"knn.json", graph_info)
            for kind in HEADS:
                name = f"{trunk}_crt_{kind}"
                head_identity = dict(cell=identity, trunk=trunk, features_sha256=graph_identity["features_sha256"], graph_sha256=graph_info["graph_sha256"])
                head, training = train_head(fit_features, y[fit], graph, kind, seed, args, classes, folder/f"head_{kind}.pt", head_identity)
                record_logits(name, common.head_logits(head, cf, args), common.head_logits(head, tf, args), training["training_prior"])
                metrics[name]["training"] = training; del head
            del model, fit_features, cf, tf, graph
        save_npz(cell/"predictions.npz", **predictions)
        result = dict(identity=identity, completed_utc=common.now(), fit_count=len(fit), calibration_count=len(cal), test_count=len(test),
                      fit_class_counts=np.bincount(y[fit], minlength=classes).tolist(), input_dim=raw.shape[1], metrics=metrics, trunks=trunks,
                      prediction_storage=storage, artifacts={name: sha(cell/name) for name in sorted(required_artifacts())})
        # Completion marker is written last, only after all artifacts/budgets validate.
        validate_result(result, cell, identity, args); write_json(result_path, result)
        return result


def paired(encoder_a, method_a, encoder_b, method_b, rho, rows, seeds):
    values = np.array([rows[encoder_a, rho, seed]["metrics"][method_a]["macro_f1"]-rows[encoder_b, rho, seed]["metrics"][method_b]["macro_f1"] for seed in seeds])
    n = len(values); mean = float(values.mean()); sd = float(values.std(ddof=1)) if n>1 else None
    half = float(stats.t.ppf(.975, n-1)*sd/math.sqrt(n)) if n>1 else None
    signs = np.asarray(list(itertools.product((-1., 1.), repeat=n)))
    exact = float(np.mean(np.abs((signs*values).mean(1)) >= abs(mean)-1e-12))
    t_p = float(stats.ttest_1samp(values, 0).pvalue) if n>1 and np.any(values!=values[0]) else 1. if np.all(values==0) else None
    return dict(encoder_a=encoder_a, method_a=method_a, encoder_b=encoder_b, method_b=method_b, rho=float(rho),
                endpoint="macro_f1", seeds=seeds, differences=values.tolist(), mean=mean,
                ci95=None if half is None else [mean-half, mean+half], signflip_p=exact, t_p=t_p)


def aggregate(caches, eligible, args, protocol):
    completed, rows = {}, []
    for encoder, rho, seed in itertools.product(ENCODERS, args.rhos, args.seeds):
        key = cell_name(encoder, rho); cell = args.out/key/f"seed_{seed}"
        if not (cell/"metrics.json").exists(): continue
        _, split_hash, _ = ensure_shared_split(caches, eligible, args, protocol, seed, rho)
        identity = dict(protocol=protocol["fingerprint"], condition=key, encoder=encoder, rho=float(rho), seed=seed,
                        split_sha256=split_hash, cache_manifest_sha256=caches[encoder]["manifest_sha256"])
        r = read_json(cell/"metrics.json"); validate_result(r, cell, identity, args)
        completed[encoder, rho, seed] = r
        for method, m in r["metrics"].items():
            rows.append(dict(condition=key, encoder=encoder, rho=float(rho), seed=seed, method=method,
                             **{k:v for k,v in m.items() if isinstance(v, (float, int))}))
    # Pairing across encoders is checked directly, not inferred from same seed labels.
    for rho, seed in itertools.product(args.rhos, args.seeds):
        pair = [completed.get((encoder, rho, seed)) for encoder in ENCODERS]
        if all(pair):
            require(pair[0]["identity"]["split_sha256"] == pair[1]["identity"]["split_sha256"], "Cross-encoder splits differ")
            require(pair[0]["trunks"]["ce"]["initial_head_sha256"] == pair[1]["trunks"]["ce"]["initial_head_sha256"], "Cross-encoder head initialization differs")
    contrasts = []
    for rho in args.rhos:
        candidates = [("deit_tiny", "rebal_crt_smote", "resnet18", "rebal_crt_smote")]
        for encoder in ENCODERS:
            candidates.extend([(encoder, "rebal_crt_smote", encoder, "rebal_crt_ros"),
                               (encoder, "rebal_crt_natural", encoder, "cb_crt_natural")])
        for ea, ma, eb, mb in candidates:
            if all((encoder, rho, seed) in completed for encoder in {ea, eb} for seed in args.seeds):
                contrasts.append(paired(ea, ma, eb, mb, rho, completed, args.seeds))
    expected = len(ENCODERS)*len(args.rhos)*len(args.seeds); family = 5*len(args.rhos)
    if len(completed) == expected and len(contrasts) == family:
        for source, target in (("signflip_p", "signflip_holm"), ("t_p", "t_holm")):
            order = sorted(range(family), key=lambda i: contrasts[i][source] if contrasts[i][source] is not None else 1.)
            running = 0.
            for rank, i in enumerate(order):
                value = contrasts[i][source]
                running = max(running, min(1., (family-rank)*(value if value is not None else 1.)))
                contrasts[i][target] = running if value is not None else None
    summary = []
    for key, method in sorted({(r["condition"], r["method"]) for r in rows}):
        selected = [r for r in rows if r["condition"]==key and r["method"]==method]; metrics = {}
        for metric in METRICS:
            values = np.array([r[metric] for r in selected]); n = len(values); avg = float(values.mean())
            sd = float(values.std(ddof=1)) if n>1 else None; half = float(stats.t.ppf(.975,n-1)*sd/math.sqrt(n)) if n>1 else None
            metrics[metric] = dict(mean=avg, sd=sd, ci95=None if half is None else [avg-half, avg+half])
        summary.append(dict(condition=key, method=method, n=len(selected), metrics=metrics))
    result = dict(protocol=protocol["fingerprint"], status="complete" if len(completed)==expected else "incomplete",
                  completed_cells=len(completed), expected_cells=expected, evaluations=len(rows), rows=rows, summary=summary,
                  paired_comparisons=contrasts, statistical_family=family,
                  inference_note="One frozen whole-grid family; no Holm correction for partial grids. Five-seed exact two-sided p cannot fall below 0.0625.",
                  benchmark_success_claimed=False)
    write_json(args.out/"aggregate.json", result); return result


def make_protocol(caches, row_identity, hygiene, args):
    verify_helpers()
    settings = {k:v for k,v in vars(args).items() if k not in
                {"features", "out", "aggregate_only", "worker_index", "worker_count", "defer_aggregate"}}
    p = dict(version=VERSION, execution="train_only_pilot" if args.pilot_steps else "full_study", settings=settings,
             source_sha256=sha(__file__), helper_sha256=HELPER_HASHES, row_identity_sha256=row_identity,
             cache_manifest_sha256={name:caches[name]["manifest_sha256"] for name in ENCODERS},
             encoders={name:{k:caches[name]["manifest"][k] for k in
                       ("encoder_id", "weights_sha256", "encoder_state_sha256", "feature_dim", "preprocessing", "preprocessing_sha256",
                        "encoder_parameter_count", "encoder_parameter_count_scope", "extractor_source_sha256", "source_reader_sha256", "original_manifest_sha256")} for name in ENCODERS},
             source_archive_sha256=caches[ENCODERS[0]]["manifest"]["archive_sha256"], hygiene=hygiene,
             scope="Frozen official pretrained encoder plus trained 256->128 feature adapter; not end-to-end ViT, no generator or causal GAN comparison",
             encoder_comparison_limit="Official pretraining, architecture, feature dimension and resulting first-layer parameter counts differ; this tests encoder packages, not architecture alone",
             architecture="inputD->256->128 with ReLU/dropout0.1 then shared-initialized native linear head",
             scaler="Per-dimension mean/std fitted separately on each encoder/cell's identical retained fit rows; heldout rows never fit preprocessing",
             outlier_filtering="Off; identical deduplicated fit rows for every encoder/objective",
             optimizer=dict(name="SGD", momentum=.9, nesterov=True, linear_kernel_l2=.0005, per_tensor_clipnorm=1., defer_fraction=.8, eq_lambda=.05),
             CB="beta=1-1/n_max, stable effective number, expected fit-example weight1; deferred final20%",
             SMOTE="Exact fit-latent kNN; class-uniform fresh interpolation with real probability n_class/n_max; no interpolation of pixels",
             sampler_implementation="Flat class-member lookup; bit-exact five-stream outputs relative to frozen non-image sampler in regression checks",
             LA="Fixed calibration macroF1 tau grid; natural or final weighted-objective native prior; uniform ROS/SMOTE prior gives tau0 aliases",
             NCM="Exploratory normalized fit centroids; not in the primary contrast family",
             head_budget=dict(updates=args.head_steps, batch_size=args.head_batch_size, lr=.01, momentum=.9, linear_kernel_l2=.0005),
             primary_endpoint="macro_f1", statistical_family=5*len(args.rhos),
             primary_comparisons=["DeiT REBAL-SMOTE minus ResNet REBAL-SMOTE per rho", "SMOTE minus ROS per encoder/rho", "REBAL-natural minus CB-natural per encoder/rho"],
             prediction_storage="Store only 15 base readout logits per split; reconstruct 12 LA views from saved prior/tau; 6 balanced-LA aliases per cell",
             software={name:importlib.metadata.version(name) for name in ("torch", "numpy", "scipy", "scikit-learn", "imbalanced-learn")},
             runtime=dict(python=sys.version.split()[0], OS=platform.system(), architecture=platform.machine(), device=args.device,
                          GPU_name=torch.cuda.get_device_name(0) if args.device=="cuda" else None, CUDA=torch.version.cuda,
                          deterministic_algorithms=torch.are_deterministic_algorithms_enabled(), determinism_warn_only=torch.is_deterministic_algorithms_warn_only_enabled()))
    p["fingerprint"] = fingerprint(p)
    return p


def train_only_pilot(caches, eligible, encoder, args, protocol):
    seed, rho = args.seeds[0], args.rhos[0]; record = caches[encoder]
    split, split_hash, _ = ensure_shared_split(caches, eligible, args, protocol, seed, rho)
    target = args.out/encoder; target.mkdir(parents=True, exist_ok=True)
    identity = dict(protocol=protocol["fingerprint"], condition=encoder, seed=seed, split_sha256=split_hash)
    fit = split["train_indices"]; x, _ = prepare_scaler(record["features"], fit, target, identity)
    y = record["labels"]; classes = len(record["manifest"]["classes"])
    weights, _ = nonimage.cb_weights(np.bincount(y[fit], minlength=classes)); cw = torch.as_tensor(weights, device=args.device)
    records = []
    for method in TRUNKS:
        model = Adapter(x.shape[1], classes, seed).to(args.device); initial = common.state_hash(model.state_dict())
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=.9, nesterov=True)
        rng = np.random.default_rng(seed+812); began = time.perf_counter(); losses = []
        for _ in range(args.pilot_steps):
            ids = rng.choice(fit, min(args.batch_size, len(fit)), replace=False)
            batch = nonimage.input_batch(x, ids, "tabular", args.device); label = torch.as_tensor(np.array(y[ids]), device=args.device)
            optimizer.zero_grad(set_to_none=True); logits, features = model(batch)
            loss = F.cross_entropy(logits,label,reduction="none")
            if method != "ce": loss = loss*cw[label]
            loss = loss.mean()+.0005*nonimage.linear_l2(model)
            if method == "rebal": loss = loss+.05*common.equalization(features,label,classes)
            require(bool(torch.isfinite(loss)), "Nonfinite pilot loss"); loss.backward()
            for parameter in model.parameters():
                require(bool(torch.isfinite(parameter.grad).all()), "Nonfinite pilot gradient")
                parameter.grad.mul_((1./(parameter.grad.norm()+1e-6)).clamp(max=1.))
            optimizer.step(); losses.append(float(loss.detach().cpu()))
        records.append(dict(method=method, updates=args.pilot_steps, seconds=time.perf_counter()-began,
                            initial_state_sha256=initial, first_loss=losses[0], last_loss=losses[-1]))
    result = dict(protocol=protocol["fingerprint"], encoder=encoder, records=records, calibration_evaluated=False,
                  test_evaluated=False, graph_built=False, scope="Train-only operator/throughput pilot; immediate CB/EQ, weights discarded")
    write_json(target/"pilot.json", result); return result


def configure_runtime(args):
    if args.device=="cuda": require(torch.cuda.is_available(), "CUDA requested but unavailable")
    if args.device=="mps": require(torch.backends.mps.is_available(), "MPS requested but unavailable")
    if args.knn_device=="cuda": require(torch.cuda.is_available(), "CUDA kNN requested but unavailable")
    if args.knn_device=="mps": require(torch.backends.mps.is_available(), "MPS kNN requested but unavailable")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(args.threads); torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True; torch.backends.cuda.matmul.allow_tf32=False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, nargs=2, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0,1,2,3,4])
    parser.add_argument("--rhos", type=float, nargs="+", default=[10.,50.,100.])
    parser.add_argument("--epochs", type=int, default=50); parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1024); parser.add_argument("--lr", type=float, default=.05)
    parser.add_argument("--head-steps", type=int, default=2000); parser.add_argument("--head-batch-size", type=int, default=2048)
    parser.add_argument("--holdout-per-class", type=int, default=50); parser.add_argument("--smote-k", type=int, default=5)
    parser.add_argument("--checkpoint-every", type=int, default=5); parser.add_argument("--head-checkpoint-every", type=int, default=100)
    parser.add_argument("--device", choices=("cpu","cuda","mps"), default="cpu")
    parser.add_argument("--knn-device", choices=("cpu","cuda","mps"), default="cpu")
    parser.add_argument("--knn-query-block", type=int, default=256); parser.add_argument("--knn-reference-block", type=int, default=8192)
    parser.add_argument("--threads", type=int, default=4); parser.add_argument("--pilot-steps", type=int, default=0)
    parser.add_argument("--aggregate-only", action="store_true"); parser.add_argument("--defer-aggregate", action="store_true")
    parser.add_argument("--worker-index", type=int, default=0); parser.add_argument("--worker-count", type=int, default=1)
    args = parser.parse_args()
    require(len(set(args.seeds))==len(args.seeds) and args.seeds==sorted(args.seeds) and min(args.seeds)>=0, "Seeds must be unique, sorted and nonnegative")
    require(len(set(args.rhos))==len(args.rhos) and args.rhos==sorted(args.rhos) and all(math.isfinite(r) and r>=1 for r in args.rhos), "Ratios must be unique, finite and sorted")
    require(min(args.epochs,args.batch_size,args.head_steps,args.head_batch_size,args.holdout_per_class,args.smote_k,args.checkpoint_every,args.head_checkpoint_every,args.threads,args.knn_query_block,args.knn_reference_block)>0, "Positive budgets/block sizes required")
    require(0<=args.warmup_epochs<=args.epochs and args.pilot_steps>=0 and math.isfinite(args.lr) and args.lr>0, "Invalid schedule")
    require(args.worker_count>=1 and 0<=args.worker_index<args.worker_count, "Invalid worker assignment")
    require(args.worker_count==1 or args.defer_aggregate or args.aggregate_only, "Multiple workers require deferred aggregation")
    require(not args.pilot_steps or args.worker_count==1 and not args.aggregate_only, "Pilot cannot aggregate or shard")
    configure_runtime(args); caches,row_identity,eligible,hygiene = load_caches(args.features)
    protocol = make_protocol(caches,row_identity,hygiene,args); args.out.mkdir(parents=True,exist_ok=True)
    with open(args.out/"protocol.lock", "a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX); path=args.out/"protocol.json"
        if path.exists(): require(read_json(path)==protocol, "Protocol mismatch; use a separate output directory")
        else:
            write_json(path,protocol); write_json(args.out/"study_plan.json",dict(created_utc=common.now(),protocol=protocol["fingerprint"],
                execution=protocol["execution"],expected_cells=0 if args.pilot_steps else len(ENCODERS)*len(args.rhos)*len(args.seeds),
                named_readouts_per_cell=27, primary_family=protocol["statistical_family"]))
    if args.pilot_steps:
        for encoder in ENCODERS: train_only_pilot(caches,eligible,encoder,args,protocol)
        return
    if not args.aggregate_only:
        for index,(rho,seed,encoder) in enumerate(itertools.product(args.rhos,args.seeds,ENCODERS)):
            if index%args.worker_count!=args.worker_index: continue
            run_cell(caches,eligible,encoder,rho,seed,args,protocol)
            if not args.defer_aggregate: aggregate(caches,eligible,args,protocol)
    if args.aggregate_only or not args.defer_aggregate:
        aggregate(caches,eligible,args,protocol)


if __name__ == "__main__":
    try:
        main()
    except (ValueError,OSError,KeyError) as error:
        reason = f"OS error {error.errno}: {error.strerror}" if isinstance(error,OSError) else str(error)
        print(json.dumps(dict(status="failed",error_type=type(error).__name__,reason=reason)),file=sys.stderr)
        sys.exit(1)
