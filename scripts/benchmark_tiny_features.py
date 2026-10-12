#!/usr/bin/env python3
"""Controlled classifier-only transfer study on cached Tiny ImageNet features.

This is NOT an end-to-end REBAL evaluation. A single frozen, ImageNet-pretrained
ResNet18 supplies all features. The internal tuning set comes only from the
official training partition; the official validation partition is final eval.
Trainable arms share initialization, optimizer, batch size and update count.
NCM is a nonparametric diagnostic with its own explicitly documented cosine rule.

Example:
  python scripts/benchmark_tiny_features.py --cache FEATURES.npz --out RUNS
  python scripts/benchmark_tiny_features.py --cache FEATURES.npz --out RUNS --aggregate-only

Results are fractions, not percentages. Resume accepts only the identical
protocol, source and input cache, with intact saved evidence. Do not change the
protocol after inspecting final evaluation scores and call it confirmatory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from imblearn.over_sampling import RandomOverSampler, SMOTE
from scipy import stats


VERSION = 1
ARMS = ("ce", "cb", "ros", "smote", "ce_la", "smote_la", "ncm")
TRAINABLE = ("ce", "cb", "ros", "smote")
TAUS = (0., .25, .5, .75, 1., 1.25, 1.5, 2.)
METRICS = ("accuracy", "macro_recall", "macro_f1", "head_f1", "mid_f1", "tail_f1",
           "head_recall", "mid_recall", "tail_recall", "worst_f1", "zero_f1_classes")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def write_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrays)
    tmp.replace(path)


def clean_training_indices(manifest, labels):
    """Remove every train/eval pixel overlap, then keep first remaining train hash.

    This uses image identity only, never evaluation labels or model predictions.
    The official evaluation partition remains intact, including its duplicates.
    """
    train_hashes, eval_hashes = manifest["train_pixel_sha256"], manifest["eval_pixel_sha256"]
    if len(train_hashes) != len(labels) or len(eval_hashes) != 10000:
        raise ValueError("Pixel-hash manifest does not match cache dimensions")
    if len(manifest["train_files"]) != len(labels) or len(manifest["eval_files"]) != 10000:
        raise ValueError("Filename manifest does not match cache dimensions")
    if len(set(manifest["train_files"])) != len(labels) or len(set(manifest["eval_files"])) != 10000:
        raise ValueError("Repeated source filename in manifest")
    if manifest["classes"] != sorted(manifest["classes"]) or len(set(manifest["classes"])) != 200:
        raise ValueError("Expected 200 sorted WordNet class IDs")
    seen, evaluation = set(), set(eval_hashes)
    keep, overlap, repeated = [], [], []
    for i, pixel_hash in enumerate(train_hashes):
        if pixel_hash in evaluation:
            overlap.append(i)
        elif pixel_hash in seen:
            repeated.append(i)
        else:
            seen.add(pixel_hash)
            keep.append(i)
    keep = np.asarray(keep, dtype=np.int64)
    counts = np.bincount(labels[keep], minlength=200)
    return keep, dict(
        policy="exclude every train hash present in final eval, then retain first remaining train hash",
        train_eval_overlap_excluded=overlap, duplicate_train_excluded=repeated,
        excluded_indices=sorted(overlap + repeated), retained_examples=int(len(keep)),
        clean_class_counts=counts.tolist(), max_train_per_class=int(counts.min() - 50),
        eval_duplicate_extra_rows=len(eval_hashes) - len(evaluation))


def make_split(labels, seed, rho, holdout_per_class=50, eligible_indices=None):
    """Nested long-tail samples, with a seed-specific random semantic class rank.

    RNG streams for holdout, rank and training order are independent. The holdout
    and rank are therefore exactly equal across rho, and each class's rho=100
    training examples are a subset of rho=50 and rho=10 for the same seed.
    """
    labels = np.asarray(labels)
    eligible_indices = (np.arange(len(labels), dtype=np.int64) if eligible_indices is None
                        else np.asarray(eligible_indices, dtype=np.int64))
    if len(np.unique(eligible_indices)) != len(eligible_indices):
        raise ValueError("Repeated eligible training index")
    classes, totals = np.unique(labels[eligible_indices], return_counts=True)
    if not np.array_equal(classes, np.arange(len(classes))) or len(classes) < 3:
        raise ValueError("Labels must contain contiguous class IDs 0..C-1, C >= 3")
    if totals.min() <= holdout_per_class:
        raise ValueError("Every class needs examples left after holdout")
    if rho < 1 or holdout_per_class < 1:
        raise ValueError("rho >= 1 and a nonempty internal holdout are required")
    n_classes = len(classes)
    max_count = int(totals.min() - holdout_per_class)
    holdout_rng = np.random.default_rng(np.random.SeedSequence([seed, 101]))
    rank_rng = np.random.default_rng(np.random.SeedSequence([seed, 202]))
    train_rng = np.random.default_rng(np.random.SeedSequence([seed, 303]))
    class_order = rank_rng.permutation(n_classes)
    class_rank = np.empty(n_classes, dtype=np.int64)
    class_rank[class_order] = np.arange(n_classes)
    counts = np.floor(max_count * float(rho) ** (-class_rank / (n_classes - 1))).astype(np.int64)
    if counts.min() < 2:
        raise ValueError("This rho leaves fewer than 2 train examples in a class")
    holdout, train = [], []
    for c in classes:
        ordered = holdout_rng.permutation(eligible_indices[labels[eligible_indices] == c])
        holdout.extend(ordered[:holdout_per_class])
        remaining = train_rng.permutation(ordered[holdout_per_class:])
        train.extend(remaining[:counts[c]])
    train = np.asarray(train, dtype=np.int64)
    holdout = np.asarray(holdout, dtype=np.int64)
    assert len(np.unique(train)) == len(train)
    assert len(np.unique(holdout)) == len(holdout)
    assert np.intersect1d(train, holdout).size == 0
    assert np.array_equal(np.bincount(labels[train], minlength=n_classes), counts)
    assert np.all(np.bincount(labels[holdout], minlength=n_classes) == holdout_per_class)
    excluded = np.setdiff1d(np.arange(len(labels)), np.concatenate([train, holdout]))
    return dict(train_indices=train, holdout_indices=holdout, excluded_indices=excluded,
                hygiene_excluded_indices=np.setdiff1d(np.arange(len(labels)), eligible_indices),
                class_order=class_order, class_rank=class_rank, train_counts=counts,
                max_train_per_class=np.asarray(max_count))


def class_regions(class_rank):
    """Original framework boundaries, applied to randomized ranks: 66/67/67."""
    rank = np.asarray(class_rank)
    n = len(rank)
    return dict(head=np.flatnonzero(rank < n // 3),
                mid=np.flatnonzero((rank >= n // 3) & (rank < 2 * n // 3)),
                tail=np.flatnonzero(rank >= 2 * n // 3))


def measure(labels, prediction, class_rank):
    labels, prediction = np.asarray(labels), np.asarray(prediction)
    n_classes = len(class_rank)
    if labels.shape != prediction.shape or len(labels) == 0:
        raise ValueError("Prediction and label vectors must be equally sized and nonempty")
    if (labels.min() < 0 or labels.max() >= n_classes or prediction.min() < 0
            or prediction.max() >= n_classes):
        raise ValueError("Out-of-range class ID")
    cm = np.bincount(labels * n_classes + prediction,
                     minlength=n_classes ** 2).reshape(n_classes, n_classes)
    tp = np.diag(cm).astype(float)
    support, predicted = cm.sum(axis=1), cm.sum(axis=0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    f1 = np.divide(2 * tp, support + predicted, out=np.zeros_like(tp),
                   where=(support + predicted) > 0)
    result = dict(accuracy=float(tp.sum() / len(labels)), macro_recall=float(recall.mean()),
                  macro_f1=float(f1.mean()), worst_f1=float(f1.min()),
                  zero_f1_classes=int((f1 == 0).sum()))
    for region, ids in class_regions(class_rank).items():
        result[region + "_f1"] = float(f1[ids].mean())
        result[region + "_recall"] = float(recall[ids].mean())
    result["head_tail_f1_gap"] = result["head_f1"] - result["tail_f1"]
    result["per_class"] = dict(recall=recall.tolist(), precision=precision.tolist(),
                               f1=f1.tolist(), support=support.tolist(), predicted=predicted.tolist())
    return result


def macro_recall(labels, predictions, n_classes):
    n = np.bincount(labels, minlength=n_classes)
    if (n == 0).any():
        raise ValueError("Tuning holdout must contain every class")
    correct = np.bincount(labels[predictions == labels], minlength=n_classes)
    return float((correct / n).mean())


def choose_tau(holdout_logits, holdout_labels, train_counts, taus=TAUS):
    """Select from a fixed grid using only the train-derived holdout; ties go low."""
    prior = np.log(train_counts / np.sum(train_counts)).astype(np.float32)
    curve = []
    for tau in sorted(taus):
        pred = np.argmax(holdout_logits - float(tau) * prior, axis=1)
        curve.append(dict(tau=float(tau), macro_recall=macro_recall(
            holdout_labels, pred, len(train_counts))))
    best = max(curve, key=lambda item: item["macro_recall"])
    return best["tau"], curve, prior


def ncm_centroids(features, labels, n_classes):
    normalized = features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-8)
    centers = np.stack([normalized[labels == c].mean(axis=0) for c in range(n_classes)])
    return centers / np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-8)


def ncm_logits(centroids, features):
    normalized = features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-8)
    return normalized @ centroids.T


def resample_pool(features, labels, method, seed):
    counts = np.bincount(labels)
    target = int(counts.max())
    strategy = {int(c): target for c in range(len(counts)) if counts[c] < target}
    meta = dict(method=method, random_state=seed, real_examples=len(labels),
                target_per_class=target, feature_geometry="raw Euclidean")
    if not strategy:
        return features.copy(), labels.copy(), dict(meta, added_examples=0), None
    if method == "ros":
        sampler = RandomOverSampler(sampling_strategy=strategy, random_state=seed)
    elif method == "smote":
        k = min(5, int(counts.min()) - 1)
        if k < 1:
            raise ValueError("SMOTE needs >=2 real examples in every class")
        sampler = SMOTE(sampling_strategy=strategy, random_state=seed, k_neighbors=k)
        meta["k_neighbors"] = k
    else:
        raise ValueError(f"Unknown resampling method: {method}")
    x, y = sampler.fit_resample(features, labels)
    assert np.all(np.bincount(y) == target)
    assert np.array_equal(x[:len(features)], features)
    assert np.isfinite(x).all()
    meta["added_examples"] = int(len(y) - len(labels))
    return x.astype(np.float32), y.astype(np.int64), meta, getattr(sampler, "sample_indices_", None)


def train_head(features, labels, counts, seed, args, weighted=False):
    """Four arms reset to identical initialization; all sample with replacement."""
    torch.manual_seed(seed)
    model = torch.nn.Linear(features.shape[1], len(counts)).to(args.device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=.9, weight_decay=.0005)
    x = torch.as_tensor(features, dtype=torch.float32, device=args.device)
    y = torch.as_tensor(labels, dtype=torch.long, device=args.device)
    # A dedicated CPU generator gives the same draw stream on CPU and MPS.
    sample_rng = torch.Generator(device="cpu").manual_seed(seed + 100000)
    class_weight = None
    if weighted:
        w = (1. - .999) / (1. - .999 ** counts.astype(np.float64))
        w /= w.mean()
        class_weight = torch.as_tensor(w, dtype=torch.float32, device=args.device)
    model.train()
    losses = []
    start = time.perf_counter()
    for step in range(args.steps):
        idx = torch.randint(len(labels), (args.batch_size,), generator=sample_rng).to(args.device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x[idx])
        loss = torch.nn.functional.cross_entropy(logits, y[idx], weight=class_weight,
                                                 reduction="none").mean()
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite training loss at update {step + 1}")
        loss.backward()
        optimizer.step()
        if step == 0 or (step + 1) % 100 == 0 or step + 1 == args.steps:
            losses.append(dict(update=step + 1, minibatch_loss=float(loss.detach().cpu())))
    if args.device == "mps":
        torch.mps.synchronize()
    elapsed = time.perf_counter() - start
    model.eval()
    return model, dict(optimizer_updates=args.steps, sampled_examples=args.steps * args.batch_size,
                       seconds=elapsed, losses=losses,
                       class_weight=None if class_weight is None else class_weight.cpu().tolist())


def predict_logits(model, features, device, batch_size=2048):
    result = []
    with torch.inference_mode():
        for i in range(0, len(features), batch_size):
            result.append(model(torch.as_tensor(features[i:i + batch_size],
                          dtype=torch.float32, device=device)).cpu().numpy())
    logits = np.concatenate(result)
    if not np.isfinite(logits).all():
        raise RuntimeError("Nonfinite evaluation logits")
    return logits


def verify_cache(cache):
    expected = dict(train_features=(100000, 512), train_labels=(100000,),
                    eval_features=(10000, 512), eval_labels=(10000,))
    for key, shape in expected.items():
        if key not in cache or cache[key].shape != shape:
            raise ValueError(f"Expected {key} with shape {shape}")
        if not np.isfinite(cache[key]).all():
            raise ValueError(f"Nonfinite cache array: {key}")
    for prefix, per_class in (("train", 500), ("eval", 50)):
        y = cache[prefix + "_labels"]
        if not np.issubdtype(y.dtype, np.integer) or not np.array_equal(np.unique(y), np.arange(200)):
            raise ValueError(f"{prefix} labels must be integer class IDs 0..199")
        if not np.all(np.bincount(y, minlength=200) == per_class):
            raise ValueError(f"Expected {per_class} {prefix} examples per class")


def result_valid(path, expected_fingerprint):
    """Never silently overwrite results from another protocol or incomplete proof."""
    if not path.exists():
        return False
    record = json.loads(path.read_text())
    if record.get("fingerprint") != expected_fingerprint:
        raise ValueError(f"Protocol mismatch at {path}; choose a new --out directory")
    if set(record.get("arms", {})) != set(ARMS):
        raise ValueError(f"Incomplete arm record at {path}; investigate before resuming")
    for relative, checksum in record["artifacts"].items():
        artifact = path.parent / relative
        if not artifact.is_file() or sha256_file(artifact) != checksum:
            raise ValueError(f"Missing or changed artifact {artifact}; investigate before resuming")
    return True


def run_cell(data, protocol, seed, rho, args):
    cell_dir = args.out / f"rho_{rho:g}" / f"seed_{seed}"
    cell_protocol = dict(protocol, seed=seed, rho=rho)
    cell_id = fingerprint(cell_protocol)
    result_path = cell_dir / "metrics.json"
    if result_valid(result_path, cell_id):
        print(f"resume: rho={rho:g} seed={seed} verified", flush=True)
        return json.loads(result_path.read_text())
    start = time.perf_counter()
    started = utc_now()
    split = make_split(data["train_labels"], seed, rho, eligible_indices=data["eligible_indices"])
    fit_hashes = set(data["train_pixel_hashes"][split["train_indices"]])
    holdout_hashes = set(data["train_pixel_hashes"][split["holdout_indices"]])
    eval_hashes = set(data["eval_pixel_hashes"])
    assert len(fit_hashes) == len(split["train_indices"])
    assert len(holdout_hashes) == len(split["holdout_indices"])
    assert fit_hashes.isdisjoint(holdout_hashes)
    assert fit_hashes.isdisjoint(eval_hashes) and holdout_hashes.isdisjoint(eval_hashes)
    cell_dir.mkdir(parents=True, exist_ok=True)
    write_npz(cell_dir / "split.npz", **split, final_eval_indices=np.arange(len(data["eval_labels"])))
    x = data["train_features"][split["train_indices"]]
    y = data["train_labels"][split["train_indices"]]
    x_holdout = data["train_features"][split["holdout_indices"]]
    y_holdout = data["train_labels"][split["holdout_indices"]]
    counts, rank = split["train_counts"], split["class_rank"]
    result = dict(fingerprint=cell_id, protocol=cell_protocol, started_utc=started,
                  training_examples=len(y), holdout_examples=len(y_holdout),
                  final_eval_examples=len(data["eval_labels"]),
                  train_counts=counts.tolist(), class_rank=rank.tolist(),
                  nominal_imbalance_ratio=rho, max_train_per_class=int(split["max_train_per_class"]),
                  pixel_identity_checks="fit/holdout unique and mutually disjoint; both disjoint from eval",
                  actual_max_min_ratio=float(counts.max() / counts.min()), arms={})
    for arm in TRAINABLE:
        print(f"rho={rho:g} seed={seed} arm={arm} train_n={len(y)}", flush=True)
        if arm in ("ros", "smote"):
            pool_x, pool_y, sampling, sample_indices = resample_pool(x, y, arm, seed)
            if sample_indices is not None:
                write_npz(cell_dir / "ros_source_indices.npz", pool_to_train_indices=sample_indices,
                          pool_to_cache_indices=split["train_indices"][sample_indices])
        else:
            pool_x, pool_y = x, y
            sampling = dict(method="natural_with_replacement", real_examples=len(y), added_examples=0)
        model, training = train_head(pool_x, pool_y, counts, seed, args, weighted=arm == "cb")
        weights = model.weight.detach().cpu().numpy()
        bias = model.bias.detach().cpu().numpy()
        write_npz(cell_dir / "checkpoints" / f"{arm}.npz", weight=weights, bias=bias)
        holdout_logits = predict_logits(model, x_holdout, args.device)
        # Any tuning finishes before this cell's final-eval logits are accessed.
        tau, curve, log_prior = (choose_tau(holdout_logits, y_holdout, counts)
                                 if arm in ("ce", "smote") else (None, None, None))
        eval_logits = predict_logits(model, data["eval_features"], args.device)
        eval_pred, holdout_pred = eval_logits.argmax(1), holdout_logits.argmax(1)
        write_npz(cell_dir / "predictions" / f"{arm}.npz", eval_logits=eval_logits,
                  eval_prediction=eval_pred, holdout_logits=holdout_logits,
                  holdout_prediction=holdout_pred)
        result["arms"][arm] = dict(final_eval=measure(data["eval_labels"], eval_pred, rank),
                                    holdout=measure(y_holdout, holdout_pred, rank),
                                    training=training, sampling=sampling)
        if tau is not None:
            adjusted_eval = (eval_logits - tau * log_prior).argmax(1)
            adjusted_holdout = (holdout_logits - tau * log_prior).argmax(1)
            write_npz(cell_dir / "predictions" / f"{arm}_la.npz", eval_prediction=adjusted_eval,
                      holdout_prediction=adjusted_holdout, log_prior=log_prior, tau=np.asarray(tau))
            result["arms"][arm + "_la"] = dict(
                final_eval=measure(data["eval_labels"], adjusted_eval, rank),
                holdout=measure(y_holdout, adjusted_holdout, rank), tau=tau, tau_curve=curve,
                checkpoint_arm=arm, adjustment="logits - tau * log(empirical natural train prior)")
        print(f"  {training['seconds']:.1f}s; final macro-F1="
              f"{result['arms'][arm]['final_eval']['macro_f1']:.4f}", flush=True)
        del model, pool_x, pool_y
    centroids = ncm_centroids(x, y, len(counts))
    holdout_logits = ncm_logits(centroids, x_holdout)
    eval_logits = ncm_logits(centroids, data["eval_features"])
    write_npz(cell_dir / "checkpoints" / "ncm.npz", normalized_centroids=centroids)
    write_npz(cell_dir / "predictions" / "ncm.npz", eval_logits=eval_logits,
              eval_prediction=eval_logits.argmax(1), holdout_logits=holdout_logits,
              holdout_prediction=holdout_logits.argmax(1))
    result["arms"]["ncm"] = dict(final_eval=measure(data["eval_labels"], eval_logits.argmax(1), rank),
        holdout=measure(y_holdout, holdout_logits.argmax(1), rank), optimizer_updates=0,
        rule="L2-normalize real training features, average by class, L2-normalize centroids; cosine argmax")
    result["finished_utc"] = utc_now()
    result["total_seconds"] = time.perf_counter() - start
    result["artifacts"] = {str(p.relative_to(cell_dir)): sha256_file(p)
                           for p in sorted(cell_dir.rglob("*.npz"))}
    write_json(result_path, result)
    return result


def mean_interval(values):
    a = np.asarray(values, dtype=float)
    out = dict(n=int(len(a)), mean=float(a.mean()), std=None, ci95=None)
    if len(a) > 1:
        sd = float(a.std(ddof=1))
        half = float(stats.t.ppf(.975, len(a) - 1) * sd / math.sqrt(len(a)))
        out.update(std=sd, ci95=[float(a.mean() - half), float(a.mean() + half)])
    return out


def paired_comparison(a, b):
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    out = dict(mean_interval(d), seed_deltas=d.tolist(), p_two_sided=None,
               p_signflip_two_sided=None, t_test_status="too few seeds")
    if len(d) > 1:
        if np.allclose(d, d[0], rtol=0., atol=1e-15):
            out["p_two_sided"] = 1. if np.all(d == 0) else None
            out["t_test_status"] = "all deltas zero" if np.all(d == 0) else "undefined: degenerate variance"
        else:
            out["p_two_sided"] = float(stats.ttest_1samp(d, 0.).pvalue)
            out["t_test_status"] = "computed"
        if len(d) <= 20:
            observed = abs(float(d.mean()))
            exceed = sum(abs(float(np.mean(d * signs))) >= observed - 1e-12
                         for signs in itertools.product((-1., 1.), repeat=len(d)))
            out["p_signflip_two_sided"] = exceed / (2 ** len(d))
    return out


def holm_adjust(pvalues):
    """Holm step-down adjustment; None entries are untested, excluded from m."""
    adjusted = [None] * len(pvalues)
    valid = [(i, p) for i, p in enumerate(pvalues) if p is not None]
    ordered = sorted(valid, key=lambda item: item[1])
    running = 0.
    for rank, (i, p) in enumerate(ordered):
        running = max(running, min(1., (len(ordered) - rank) * p))
        adjusted[i] = float(running)
    return adjusted


def aggregate(records, out, planned_seeds, planned_rhos):
    grouped = {}
    for record in records:
        grouped.setdefault(record["protocol"]["rho"], {})[record["protocol"]["seed"]] = record
    result = dict(created_utc=utc_now(), units="fractions", planned_seeds=planned_seeds,
                  planned_rhos=planned_rhos, observed_cells=len(records),
                  complete=len(records) == len(planned_seeds) * len(planned_rhos),
                  uncertainty="Sample SD and Student-t 95% intervals across paired seeds. "
                  "Seeds share the final evaluation images; intervals do not cover dataset "
                  "sampling, architecture choice or pretraining leakage uncertainty.",
                  comparison_status="Exploratory preplanned comparisons; tiny seed n and one dataset.",
                  comparisons=[], by_rho={})
    table = []
    for rho, by_seed in sorted(grouped.items()):
        seeds = sorted(by_seed)
        summary = dict(seeds=seeds, arms={})
        for arm in ARMS:
            summary["arms"][arm] = {}
            for metric in METRICS:
                values = [by_seed[s]["arms"][arm]["final_eval"][metric] for s in seeds]
                stat = mean_interval(values)
                summary["arms"][arm][metric] = stat
                table.append(dict(rho=rho, arm=arm, metric=metric, n=stat["n"],
                                  mean=stat["mean"], std=stat["std"],
                                  ci95_low=None if stat["ci95"] is None else stat["ci95"][0],
                                  ci95_high=None if stat["ci95"] is None else stat["ci95"][1]))
        result["by_rho"][f"{rho:g}"] = summary
        for reference in ("ros", "ce_la"):
            a = [by_seed[s]["arms"]["smote"]["final_eval"]["tail_f1"] for s in seeds]
            b = [by_seed[s]["arms"][reference]["final_eval"]["tail_f1"] for s in seeds]
            result["comparisons"].append(dict(rho=rho, intervention="smote", reference=reference,
                metric="tail_f1", seeds=seeds, **paired_comparison(a, b)))
    # The correction family always includes all planned rho/comparator tests;
    # missing cells are assigned p=1 for multiplicity, not counted as evidence.
    pvals = [c["p_two_sided"] for c in result["comparisons"]]
    padded = [1. if p is None else p for p in pvals]
    padded.extend([1.] * (2 * len(planned_rhos) - len(padded)))
    adjusted = holm_adjust(padded)
    signflip_padded = [c["p_signflip_two_sided"] if c["p_signflip_two_sided"] is not None else 1.
                      for c in result["comparisons"]]
    signflip_padded.extend([1.] * (2 * len(planned_rhos) - len(signflip_padded)))
    adjusted_signflip = holm_adjust(signflip_padded)
    for i, comp in enumerate(result["comparisons"]):
        comp["p_holm"] = adjusted[i] if comp["p_two_sided"] is not None else None
        comp["p_signflip_holm"] = (adjusted_signflip[i]
                                   if comp["p_signflip_two_sided"] is not None else None)
        comp["holm_family_size"] = 2 * len(planned_rhos)
    write_json(out / "aggregate.json", result)
    if table:
        with open(out / "aggregate.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, help="Provenance/pixel hashes; default CACHE.manifest.json")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rhos", nargs="+", type=float, default=[10., 50., 100.])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(5)))
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=.01)
    ap.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args(argv)
    if (args.steps < 1 or args.batch_size < 1 or args.lr <= 0 or args.threads < 1
            or min(args.seeds) < 0 or len(set(args.seeds)) != len(args.seeds)
            or len(set(args.rhos)) != len(args.rhos) or min(args.rhos) < 1):
        ap.error("Invalid or duplicate protocol values")
    if args.device == "mps" and not torch.backends.mps.is_available():
        ap.error("MPS requested but unavailable; use --device cpu or an MPS-capable execution context")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    # Strict CPU determinism; MPS kernels may not provide cross-version or bitwise guarantees.
    torch.use_deterministic_algorithms(True, warn_only=args.device == "mps")
    if args.manifest is None:
        args.manifest = args.cache.with_suffix(".manifest.json")
    manifest = json.loads(args.manifest.read_text())
    cache_hash = sha256_file(args.cache)
    if manifest["features_sha256"] != cache_hash:
        raise ValueError("Cache checksum does not match feature provenance manifest")
    # Read the small label array before fixing the deduplication-aware protocol.
    with np.load(args.cache, allow_pickle=False) as cache:
        train_labels = cache["train_labels"]
    eligible_indices, hygiene = clean_training_indices(manifest, train_labels)
    max_count = hygiene["max_train_per_class"]
    versions = {name: importlib.metadata.version(name) for name in
                ("numpy", "torch", "scipy", "scikit-learn", "imbalanced-learn")}
    protocol = dict(version=VERSION, scope="frozen pretrained representation; classifier-only transfer",
        planned_seeds=args.seeds, planned_rhos=args.rhos,
        cache_sha256=cache_hash, manifest_sha256=sha256_file(args.manifest),
        source_sha256=sha256_file(__file__), features="raw 512-dimensional features for trainable heads",
        split=f"50/class balanced holdout from deduplicated official train; {max_count}/class maximum train",
        count_formula=f"floor({max_count} * rho ** (-randomized_class_rank / 199))",
        max_train_per_class=max_count, data_hygiene=hygiene,
        rank_scheme="seed-specific random semantic class permutation; same across rho",
        region_boundaries="head rank < C//3; middle C//3 <= rank < 2*C//3; tail rank >= 2*C//3 (66/67/67)",
        eval_partition="official Tiny ImageNet validation, 50/class, only final evaluation",
        head="linear 512 -> 200, bias=True; fresh same initialization for each trainable arm",
        optimizer="SGD", lr=args.lr, momentum=.9, weight_decay=.0005,
        steps=args.steps, batch_size=args.batch_size, sampling="uniform pool draws with replacement",
        effective_number_beta=.999,
        loss_weight_normalization="mean-one class weights; mean of weighted per-example cross-entropy",
        smote_k="min(5, minimum real training class count - 1)", taus=list(TAUS),
        tau_selection="maximum internal-holdout macro recall; ties choose smallest tau",
        ncm="cosine rule on L2-normalized real features and L2-normalized centroids",
        early_stopping=False, device=args.device, threads=args.threads,
        software=versions, python=platform.python_version(), platform=platform.platform(),
        determinism="strict CPU; MPS warns for unsupported deterministic kernels",
        preplanned_comparisons="SMOTE minus ROS and SMOTE minus CE+LA tail macro-F1 at each rho; "
        "paired two-sided t intervals/tests and exact paired sign-flip tests, Holm family across all "
        "planned comparisons separately for each test type; exploratory")
    args.out.mkdir(parents=True, exist_ok=True)
    protocol_path = args.out / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("Existing output has a different protocol. Use a new --out directory.")
    write_json(protocol_path, protocol)
    write_json(args.out / "requested_plan.json", dict(seeds=args.seeds, rhos=args.rhos,
                                                     expected_cells=len(args.seeds) * len(args.rhos)))
    write_json(args.out / "feature_manifest.json", manifest)
    records = []
    if args.aggregate_only:
        for rho in args.rhos:
            for seed in args.seeds:
                path = args.out / f"rho_{rho:g}" / f"seed_{seed}" / "metrics.json"
                if result_valid(path, fingerprint(dict(protocol, seed=seed, rho=rho))):
                    records.append(json.loads(path.read_text()))
    else:
        print("Loading cache and validating exact Tiny ImageNet dimensions...", flush=True)
        with np.load(args.cache, allow_pickle=False) as cache:
            data = {key: cache[key] for key in ("train_features", "train_labels", "eval_features", "eval_labels")}
        verify_cache(data)
        for key in ("train_features", "eval_features"):
            data[key] = np.asarray(data[key], dtype=np.float32)
        for key in ("train_labels", "eval_labels"):
            data[key] = np.asarray(data[key], dtype=np.int64)
        data["eligible_indices"] = eligible_indices
        data["train_pixel_hashes"] = np.asarray(manifest["train_pixel_sha256"])
        data["eval_pixel_hashes"] = np.asarray(manifest["eval_pixel_sha256"])
        for rho in args.rhos:
            for seed in args.seeds:
                records.append(run_cell(data, protocol, seed, rho, args))
                aggregate(records, args.out, args.seeds, args.rhos)
    summary = aggregate(records, args.out, args.seeds, args.rhos)
    print(f"Saved {len(records)} verified cells. Complete requested grid: {summary['complete']}. "
          f"Aggregate: {args.out / 'aggregate.json'}", flush=True)


if __name__ == "__main__":
    main()
