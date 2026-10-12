#!/usr/bin/env python3
"""Independently verify completed non-image studies from saved predictions.

No runner/common metric functions, sklearn metrics, model inference, or pickle
checkpoint loading are used. Classification metrics come from NumPy confusion
matrices. SciPy distribution functions are used only for Student-t inference.
Pilot protocols are rejected before prediction artifacts can be inspected.

  python scripts/verify_nonimage.py --data CONDITION_DIRS --run RUN_DIR --out verification.json

Exit codes: 0 verified complete; 2 incomplete; 1 invalid evidence. A successful
verification confirms numerical consistency, not superiority of any method.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import numpy as np
from scipy.special import stdtr, stdtrit

VERSION = 1
TRUNKS = ("ce", "cb", "rebal")
HEADS = ("natural", "ros", "smote")
BASE_METHODS = set(TRUNKS) | {f"{m}_crt_{h}" for m in TRUNKS for h in HEADS}
METHODS = BASE_METHODS | {m + "_la" for m in BASE_METHODS} | {m + "_ncm" for m in TRUNKS}
TAUS = (0., .25, .5, .75, 1., 1.25, 1.5, 2.)
CONTRASTS = (("rebal_crt_smote", "rebal_crt_ros"), ("rebal_crt_natural", "cb_crt_natural"))
SUMMARY_METRICS = ("accuracy", "macro_f1", "macro_recall", "head_f1", "mid_f1", "tail_f1")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read_json(path):
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def relative_file(root, name):
    p = PurePosixPath(name)
    require(bool(name) and not p.is_absolute() and ".." not in p.parts and "\\" not in name, "Unsafe artifact path")
    target = root / name
    require(target.is_file(), f"Missing artifact: {name}")
    require(target.resolve().is_relative_to(root.resolve()), f"Artifact escapes its directory: {name}")
    return target


def hashes(root, inventory, required=None):
    require(isinstance(inventory, dict) and bool(inventory), "Missing artifact hash inventory")
    if required is not None:
        require(set(inventory) == set(required), "Artifact inventory differs from required files")
    for name, digest in inventory.items():
        require(isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest), f"Invalid SHA256: {name}")
        require(sha(relative_file(root, name)) == digest, f"SHA256 mismatch: {name}")


def close(actual, expected, where, atol=1e-12, rtol=1e-10):
    if expected is None:
        require(actual is None, f"{where}: expected null")
    elif isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), f"{where}: fields differ")
        for key, value in expected.items():
            close(actual[key], value, f"{where}.{key}", atol, rtol)
    elif isinstance(expected, (list, tuple)):
        require(isinstance(actual, (list, tuple)) and len(actual) == len(expected), f"{where}: length differs")
        for i, value in enumerate(expected):
            close(actual[i], value, f"{where}[{i}]", atol, rtol)
    elif isinstance(expected, (int, float, np.number)) and not isinstance(expected, bool):
        require(isinstance(actual, (int, float, np.number)) and not isinstance(actual, bool), f"{where}: numeric value required")
        require(math.isfinite(float(actual)) and math.isfinite(float(expected)) and
                math.isclose(float(actual), float(expected), abs_tol=atol, rel_tol=rtol),
                f"{where}: numeric mismatch (stored={actual}, recomputed={expected})")
    else:
        require(actual == expected, f"{where}: value differs")


def metrics(labels, predicted, rank):
    """Zero-division=0 metrics computed directly from integer counts."""
    labels, predicted, rank = np.asarray(labels), np.asarray(predicted), np.asarray(rank)
    c = len(rank)
    require(labels.ndim == predicted.ndim == 1 and len(labels) == len(predicted) and len(labels) > 0,
            "Invalid metric vectors")
    require(np.issubdtype(labels.dtype, np.integer) and np.issubdtype(predicted.dtype, np.integer), "Nonintegral labels/predictions")
    require(labels.min() >= 0 and labels.max() < c and predicted.min() >= 0 and predicted.max() < c,
            "Metric label outside class range")
    cm = np.bincount(labels.astype(np.int64) * c + predicted.astype(np.int64), minlength=c*c).reshape(c, c)
    tp = np.diag(cm).astype(np.float64)
    support, predicted_count = cm.sum(1), cm.sum(0)
    precision = np.divide(tp, predicted_count, out=np.zeros(c), where=predicted_count != 0)
    recall = np.divide(tp, support, out=np.zeros(c), where=support != 0)
    denominator = support + predicted_count
    f1 = np.divide(2*tp, denominator, out=np.zeros(c), where=denominator != 0)
    result = {"accuracy": float(tp.sum()/cm.sum()), "macro_f1": float(f1.mean()),
              "macro_recall": float(recall.mean()), "worst_f1": float(f1.min()),
              "per_class_f1": f1.tolist(), "per_class_recall": recall.tolist(),
              "per_class_precision": precision.tolist(), "confusion_matrix": cm.tolist()}
    groups = {"head": rank < c//3, "mid": (rank >= c//3) & (rank < 2*c//3), "tail": rank >= 2*c//3}
    for name, selected in groups.items():
        require(selected.any(), "Empty rank group: verifier supports the current multiclass protocol only")
        result[name + "_f1"] = float(f1[selected].mean())
        result[name + "_recall"] = float(recall[selected].mean())
    result["head_tail_f1_gap"] = result["head_f1"] - result["tail_f1"]
    return result


def load_data(directory, protocol, sources_root=None):
    directory = Path(directory)
    meta = read_json(directory / "metadata.json")
    key = re.sub(r"[^A-Za-z0-9_.-]+", "_", meta["dataset"] + "__" + str(meta["condition"]))
    require(protocol["inputs"].get(key) == sha(directory/"metadata.json"), "Prepared manifest differs from frozen protocol")
    required = {"labels.npy", "split.npz", "features.npy" if meta["task_type"] == "tabular" else "texts.jsonl"}
    require(required <= set(meta["files_sha256"]), "Prepared manifest omits required arrays")
    hashes(directory, meta["files_sha256"])
    if "protocol_fingerprint" in meta:
        expected = fingerprint({k: v for k, v in meta.items() if k not in ("created_utc", "protocol_fingerprint")})
        require(meta["protocol_fingerprint"] == expected, "Preparation protocol fingerprint mismatch")
    if "preparation_source_sha256" in meta:
        require(sha(Path(__file__).with_name("prepare_nonimage.py")) == meta["preparation_source_sha256"], "Preparation source hash mismatch")
    source_status = "not supplied by input fixture/manifest"
    if meta.get("source", {}).get("files"):
        root = Path(sources_root) if sources_root is not None else directory.parents[1]/"sources"
        source_dir = root/meta["dataset"]
        source_record = read_json(source_dir/"source.json")
        require(source_record == meta["source"], "Acquisition record differs from prepared provenance")
        hashes(source_dir, {name: record["sha256"] for name, record in source_record["files"].items()})
        source_status = "all acquisition source hashes verified"
    y = np.load(directory/"labels.npy", mmap_mode="r", allow_pickle=False)
    c = int(meta["n_classes"])
    require(y.ndim == 1 and np.issubdtype(y.dtype, np.integer), "Invalid prepared labels")
    require(y.min() >= 0 and y.max() < c, "Prepared labels outside class range")
    with np.load(directory/"split.npz", allow_pickle=False) as data:
        split = {k: data[k] for k in data.files}
    require(set(split) == {"train", "calibration", "test", "class_rank"}, "Unexpected split inventory")
    all_ids = []
    for part in ("train", "calibration", "test"):
        ids = split[part]
        require(ids.ndim == 1 and np.issubdtype(ids.dtype, np.integer) and len(ids) > 0,
                f"Invalid split: {part}")
        require(ids.min() >= 0 and ids.max() < len(y) and len(np.unique(ids)) == len(ids), "Invalid/repeated split IDs")
        require(np.array_equal(np.unique(y[ids]), np.arange(c)), "Missing class in a split")
        all_ids.append(ids)
    joined = np.concatenate(all_ids)
    require(len(np.unique(joined)) == len(joined), "Prepared splits overlap")
    order = split["class_rank"]
    require(np.array_equal(np.sort(order), np.arange(c)), "Class rank is not a permutation")
    semantics = meta.get("rank_semantics", "ordered_class_ids_head_to_tail")
    if "inverse" in semantics:
        rank = order.copy()
    else:
        require(semantics == "ordered_class_ids_head_to_tail", "Unknown class rank semantics")
        rank = np.empty(c, dtype=np.int64); rank[order] = np.arange(c)
    return {"directory": directory, "meta": meta, "key": key, "y": y, "split": split,
            "rank": rank, "source_status": source_status}


def expected_cb(counts):
    n = np.asarray(counts, dtype=np.float64)
    beta = 1.-1./n.max()
    effective = -np.expm1(n*np.log1p(-1./n.max()))
    weight = (1.-beta)/effective
    weight /= np.dot(n, weight)/n.sum()
    return weight.astype(np.float32), float(beta)


def verify_transforms(root, data, protocol):
    info = read_json(root/"transforms.json")
    settings = protocol["settings"]
    identity = dict(input_manifest=sha(data["directory"]/"metadata.json"), ecod=settings["ecod"],
                    max_words=settings["max_words"], max_tokens=settings["max_tokens"], source=protocol["source_sha256"])
    require(info["fingerprint"] == fingerprint(identity), "Transform fingerprint mismatch")
    required = {"inputs.npy", "fit_indices.npy", "excluded_ecod.npy",
                "scaler.npz" if data["meta"]["task_type"] == "tabular" else "vocabulary.json"}
    hashes(root, info["artifacts"], required)
    fit = np.load(root/"fit_indices.npy", allow_pickle=False)
    excluded = np.load(root/"excluded_ecod.npy", allow_pickle=False)
    train = data["split"]["train"]
    require(fit.ndim == excluded.ndim == 1 and np.issubdtype(fit.dtype, np.integer) and np.issubdtype(excluded.dtype, np.integer), "Invalid fit/exclusion arrays")
    combined = np.concatenate([fit, excluded])
    require(len(np.unique(combined)) == len(combined) and set(combined.tolist()) == set(train.tolist()),
            "Fit/excluded rows do not partition original training rows")
    if not settings["ecod"]:
        require(np.array_equal(fit, train) and len(excluded) == 0, "Unexpected exclusions without ECOD")
    require(info["fit_count"] == len(fit) and info["excluded_ecod"] == len(excluded), "Transform counts mismatch")
    require(info["task_type"] == data["meta"]["task_type"], "Transform task type differs")
    x = np.load(root/"inputs.npy", mmap_mode="r", allow_pickle=False)
    width = 54 if info["task_type"] == "tabular" else settings["max_tokens"]
    require(x.shape == (len(data["y"]), width), "Transform input shape mismatch")
    return info, fit


def verify_adjustment(name, record, arrays, labels, rank, prior):
    choice = record["adjustment"]
    prior = np.asarray(prior, dtype=np.float32)
    require(np.isfinite(prior).all() and (prior > 0).all(), "Invalid adjustment prior")
    close(choice["prior"], prior.tolist(), name+".prior", atol=2e-8, rtol=2e-7)
    base_cal, base_test = arrays[name+"__calibration_logits"], arrays[name+"__test_logits"]
    log_prior = np.log(prior)
    uniform = np.ptp(prior) == 0
    curve = []
    for tau in TAUS:
        candidate = base_cal if uniform else base_cal-tau*log_prior
        score = metrics(labels, candidate.argmax(1), rank)["macro_f1"]
        curve.append({"tau": tau, "macro_f1": score})
    close(choice["curve"], curve, name+".calibration_curve")
    # Curve values have just been independently checked. Use the recorded
    # floating-point tie ordering to avoid a false failure from 1-ULP differences
    # between algebraically equivalent NumPy and sklearn F1 formulas.
    selected = 0. if uniform else max(choice["curve"], key=lambda row: row["macro_f1"])["tau"]
    close(choice["tau"], selected, name+".selected_tau")
    require(any(choice["tau"] == t for t in TAUS), "Selected tau outside frozen grid")
    for part, base in (("calibration", base_cal), ("test", base_test)):
        expected = base if uniform else base-choice["tau"]*log_prior
        actual = arrays[name+"_la__"+part+"_logits"]
        require(np.allclose(actual, expected, atol=2e-7, rtol=2e-7), name+" adjusted logits mismatch")
    return selected


def verify_cell(cell, data, protocol, transforms, fit, seed):
    result = read_json(cell/"metrics.json")
    settings, c = protocol["settings"], data["meta"]["n_classes"]
    identity = dict(protocol=protocol["fingerprint"], condition=data["key"], seed=seed, transforms=transforms["fingerprint"])
    require(result["identity"] == identity, "Completed cell identity mismatch")
    require(set(result["metrics"]) == METHODS and set(result["trunks"]) == set(TRUNKS), "Incomplete 27-method inventory")
    required = {"predictions.npz", "fit_indices.npy"} | {f"{m}/{f}" for m in TRUNKS for f in ("checkpoint.pt", "history.json", "knn_graph.npy", "knn.json")} | {f"{m}/head_{h}.pt" for m in TRUNKS for h in HEADS}
    hashes(cell, result["artifacts"], required)
    cell_fit = np.load(cell/"fit_indices.npy", allow_pickle=False)
    require(np.array_equal(cell_fit, fit), "Cell fit rows differ from verified transforms")
    y, split = data["y"], data["split"]
    counts = np.bincount(y[fit], minlength=c)
    require(counts.min() >= 2, "Insufficient fit class examples")
    close(result["fit_class_counts"], counts.tolist(), "fit_class_counts", atol=0, rtol=0)
    for field, value in (("fit_count", len(fit)), ("calibration_count", len(split["calibration"])), ("test_count", len(split["test"]))):
        close(result[field], value, field, atol=0, rtol=0)
    weights, beta = expected_cb(counts)
    natural_prior = counts/counts.sum()
    priors = {}
    initial_trunks, initial_heads = set(), set()
    expected_k = min(settings["smote_k"], int(counts.min())-1)
    for trunk in TRUNKS:
        info = result["trunks"][trunk]
        require(re.fullmatch(r"[0-9a-f]{64}", info["initial_state_sha256"]), "Invalid trunk initialization hash")
        initial_trunks.add(info["initial_state_sha256"])
        close(info["completed_epochs"], settings["epochs"], "trunk epochs", 0, 0)
        close(info["optimizer_updates"], settings["epochs"]*math.ceil(len(fit)/settings["batch_size"]), "trunk updates", 0, 0)
        close(info["beta"], beta, "CB beta")
        close(info["class_weights"], weights.tolist(), "CB weights", 1e-8, 1e-7)
        close(info["expected_sample_weight"], float(np.dot(counts, weights)/counts.sum()), "CB average weight", 1e-8, 1e-7)
        priors[trunk] = natural_prior if trunk == "ce" else counts*weights/np.sum(counts*weights)
        close(info["LA_prior"], priors[trunk].tolist(), trunk+" LA objective prior", 1e-8, 1e-7)
        history = read_json(cell/trunk/"history.json")
        require(len(history) == settings["epochs"], "Incomplete epoch history")
        for epoch, row in enumerate(history):
            close(row["epoch"], epoch+1, "history epoch", 0, 0)
            close(row["optimizer_updates"], (epoch+1)*math.ceil(len(fit)/settings["batch_size"]), "history updates", 0, 0)
            require(row["EQ_active"] == (trunk == "rebal" and epoch >= int(.8*settings["epochs"])), "Equalization schedule mismatch")
        graph_meta = read_json(cell/trunk/"knn.json")
        require(graph_meta["identity"]["identity"] == identity and graph_meta["identity"]["trunk"] == trunk, "Graph identity mismatch")
        require(graph_meta["graph_sha256"] == sha(cell/trunk/"knn_graph.npy"), "Graph metadata hash mismatch")
        graph = np.load(cell/trunk/"knn_graph.npy", allow_pickle=False)
        require(graph.shape == (len(fit), expected_k) and np.issubdtype(graph.dtype, np.integer), "Invalid graph shape/type")
        fit_y = y[fit]
        maximal = counts[fit_y] == counts.max()
        require(np.all(graph[maximal] == -1), "Graph should omit maximal classes")
        ids = np.flatnonzero(~maximal); neighbors = graph[ids]
        require((neighbors >= 0).all() and (neighbors < len(fit)).all(), "Graph neighbor out of range")
        require(np.all(neighbors != ids[:, None]) and np.all(fit_y[neighbors] == fit_y[ids, None]), "Self/cross-class graph edge")
        require(np.all(np.diff(np.sort(neighbors, axis=1), axis=1) != 0), "Repeated graph neighbor")
        for head in HEADS:
            name = f"{trunk}_crt_{head}"
            training = result["metrics"][name]["training"]
            initial_heads.add(training["initial_state_sha256"])
            close(training["optimizer_updates"], settings["head_steps"], name+" updates", 0, 0)
            priors[name] = natural_prior if head == "natural" else np.ones(c)/c
            close(training["training_prior"], priors[name].tolist(), name+" training prior")
    require(len(initial_trunks) == len(initial_heads) == 1, "Unmatched trunk/head initializations")
    # A completed marker, every artifact hash, and final budgets are verified
    # before opening saved held-out prediction arrays.
    with np.load(cell/"predictions.npz", allow_pickle=False) as archive:
        required_arrays = {"test_labels", "calibration_labels", "class_rank"} | {name+"__"+part+"_logits" for name in METHODS for part in ("test", "calibration")}
        require(set(archive.files) == required_arrays, "Incomplete prediction array inventory")
        arrays = {name: archive[name] for name in archive.files}
    for part in ("test", "calibration"):
        require(np.array_equal(arrays[part+"_labels"], y[split[part]]), f"{part} labels/order differ from prepared split")
    require(np.array_equal(arrays["class_rank"], data["rank"]), "Saved inverse class rank differs from original split")
    recomputed = {}
    for name in sorted(METHODS):
        for part in ("test", "calibration"):
            logits = arrays[name+"__"+part+"_logits"]
            require(logits.shape == (len(split[part]), c) and np.issubdtype(logits.dtype, np.floating) and np.isfinite(logits).all(), "Invalid/nonfinite logits")
        calculated = metrics(arrays["test_labels"], arrays[name+"__test_logits"].argmax(1), data["rank"])
        record = result["metrics"][name]
        expected_keys = set(calculated) | ({"adjustment"} if name.endswith("_la") else {"training"} if "_crt_" in name else set())
        require(set(record) == expected_keys, name+" metric fields differ")
        close({key: record[key] for key in calculated}, calculated, name+" metrics")
        recomputed[name] = calculated
    taus = {name: verify_adjustment(name, result["metrics"][name+"_la"], arrays,
                                   arrays["calibration_labels"], data["rank"], priors[name])
            for name in sorted(BASE_METHODS)}
    rows = [dict(condition=data["key"], seed=seed, method=name,
                 **{k: v for k, v in calculated.items() if isinstance(v, (int, float))})
            for name, calculated in recomputed.items()]
    report = {"condition": data["key"], "seed": seed, "methods_verified": len(METHODS),
              "calibration_curves_verified": len(taus), "selected_taus": taus,
              "metrics_sha256": sha(cell/"metrics.json"), "predictions_sha256": result["artifacts"]["predictions.npz"],
              "fit_count": len(fit), "calibration_count": len(split["calibration"]), "test_count": len(split["test"])}
    return rows, report


def sample_summary(values):
    values = np.asarray(values, dtype=np.float64)
    n, mean = len(values), float(values.mean())
    sd = float(np.sqrt(np.sum((values-mean)**2)/(n-1))) if n > 1 else None
    half = float(stdtrit(n-1, .975)*sd/math.sqrt(n)) if n > 1 else None
    return {"mean": mean, "sd": sd, "ci95": None if half is None else [mean-half, mean+half]}


def comparison(condition, a, b, seeds, rows):
    mapping = {(r["seed"], r["method"]): r for r in rows if r["condition"] == condition}
    d = np.array([mapping[(seed, a)]["macro_f1"]-mapping[(seed, b)]["macro_f1"] for seed in seeds])
    summary = sample_summary(d); n = len(d)
    observed = abs(float(d.mean()))
    extreme = sum(abs(float(np.dot(signs, d)/n)) >= observed-1e-12
                  for signs in itertools.product((-1., 1.), repeat=n))
    if np.all(d == 0):
        t_p = 1.
    elif n <= 1 or np.all(d == d[0]):
        t_p = None
    else:
        t = abs(summary["mean"])/(summary["sd"]/math.sqrt(n))
        t_p = float(2*stdtr(n-1, -t))
    return dict(condition=condition, a=a, b=b, endpoint="macro_f1", seeds=seeds,
                differences=d.tolist(), mean=summary["mean"], ci95=summary["ci95"],
                signflip_p=extreme/(2**n), t_p=t_p)


def expected_aggregate(protocol, data, rows, completed):
    seeds = sorted(protocol["settings"]["seeds"])
    summary, comparisons = [], []
    for condition, method in sorted({(r["condition"], r["method"]) for r in rows}):
        group = [r for r in rows if r["condition"] == condition and r["method"] == method]
        summary.append(dict(condition=condition, method=method, n=len(group),
                            metrics={m: sample_summary([r[m] for r in group]) for m in SUMMARY_METRICS}))
    for condition in data:
        key = condition["key"]
        present = sorted(r["seed"] for r in completed if r["condition"] == key)
        if present == seeds:
            comparisons.extend(comparison(key, a, b, seeds, rows) for a, b in CONTRASTS)
    family = 2*len(data)
    if len(comparisons) == family:
        for source, target in (("signflip_p", "signflip_holm"), ("t_p", "t_holm")):
            order = sorted(range(family), key=lambda i: comparisons[i][source] if comparisons[i][source] is not None else 1.)
            running = 0.
            for position, i in enumerate(order):
                value = comparisons[i][source]
                running = max(running, min(1., (family-position)*(value if value is not None else 1.)))
                comparisons[i][target] = running if value is not None else None
    return dict(protocol=protocol["fingerprint"], completed_cells=len(completed), expected_cells=len(data)*len(seeds),
                evaluations=len(rows), summary=summary, rows=rows, paired_comparisons=comparisons)


def verify(data_dirs, run, sources_root=None):
    run = Path(run)
    protocol = read_json(run/"protocol.json")
    require(protocol.get("execution") == "full_study" and protocol.get("settings", {}).get("pilot_steps") == 0,
            "Pilot/non-full-study outputs are intentionally not scored")
    require(protocol.get("fingerprint") == fingerprint({k: v for k, v in protocol.items() if k != "fingerprint"}), "Frozen protocol fingerprint mismatch")
    require(protocol["source_sha256"] == sha(Path(__file__).with_name("train_nonimage.py")), "Runner source hash mismatch")
    require(protocol["shared_helpers_sha256"] == sha(Path(__file__).with_name("train_tiny_fullscale.py")), "Shared source hash mismatch")
    seeds = protocol["settings"]["seeds"]
    require(seeds and len(set(seeds)) == len(seeds) and len(seeds) <= 20 and all(type(s) is int for s in seeds), "Invalid seed plan")
    data = [load_data(directory, protocol, sources_root) for directory in data_dirs]
    require(len({d["key"] for d in data}) == len(data) and set(d["key"] for d in data) == set(protocol["inputs"]), "All frozen study conditions must be supplied exactly once")
    require(protocol["statistical_family"] == 2*len(data), "Statistical family mismatch")
    rows, completed, missing = [], [], []
    for condition in data:
        target = run/condition["key"]
        available = [seed for seed in seeds if (target/f"seed_{seed}"/"metrics.json").is_file()]
        for seed in seeds:
            if seed not in available:
                missing.append(dict(condition=condition["key"], seed=seed))
        if not available:
            continue
        transforms, fit = verify_transforms(target/"transforms", condition, protocol)
        for seed in available:
            cell_rows, evidence = verify_cell(target/f"seed_{seed}", condition, protocol, transforms, fit, seed)
            rows.extend(cell_rows); completed.append(evidence)
    expected = expected_aggregate(protocol, data, rows, completed)
    aggregate_status = "absent"
    if (run/"aggregate.json").is_file():
        stored = read_json(run/"aggregate.json")
        for key in ("protocol", "completed_cells", "expected_cells", "evaluations"):
            close(stored[key], expected[key], "aggregate."+key, 0, 0)
        for key, sort_key in (("rows", lambda r: (r["condition"], r["seed"], r["method"])),
                              ("summary", lambda r: (r["condition"], r["method"])),
                              ("paired_comparisons", lambda r: (r["condition"], r["a"], r["b"]))):
            close(sorted(stored[key], key=sort_key), sorted(expected[key], key=sort_key), "aggregate."+key)
        aggregate_status = "independently recomputed and verified"
    return {"verifier_version": VERSION, "verified_utc": datetime.now(timezone.utc).isoformat(),
            "verifier_sha256": sha(__file__), "protocol": protocol["fingerprint"],
            "runner_source_sha256": protocol["source_sha256"],
            "shared_helpers_sha256": protocol["shared_helpers_sha256"],
            "input_manifest_sha256": protocol["inputs"],
            "aggregate_sha256": sha(run/"aggregate.json") if (run/"aggregate.json").is_file() else None,
            "status": "incomplete" if missing else "verified_complete", "benchmark_success_claimed": False,
            "completed_cells": len(completed), "expected_cells": expected["expected_cells"],
            "method_evaluations_verified": len(rows), "missing_cells": missing,
            "conditions": [{"condition": d["key"], "source_status": d["source_status"]} for d in data],
            "completed_cell_evidence": completed, "aggregate_status": aggregate_status,
            "recomputed_aggregate": expected,
            "limits": ["Stored model logits are checked against labels, metrics and calibration rules; model forward passes are not rerun.",
                       "Checkpoint bytes and graph artifacts are hashed; no pickle checkpoint is loaded, and feature-distance optimality is not reconstructed.",
                       "Verification is consistency evidence, not cryptographic authentication of an untrusted author's records or evidence of method superiority.",
                       "No results are inferred for missing cells; exact five-seed two-sided sign-flip p cannot be below 0.0625."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", nargs="+", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sources-root", type=Path, help="Optional acquisition root containing dataset/source.json; defaults beside prepared dataset directories")
    args = parser.parse_args()
    try:
        report = verify(args.data, args.run, args.sources_root)
        code = 0 if report["status"] == "verified_complete" else 2
    except Exception as error:
        # OSError.__str__ can append the user's absolute filename. Keep only
        # the generic error category/code/message in portable failure reports.
        reason = (f"OS error {error.errno}: {error.strerror}" if isinstance(error, OSError) else str(error))
        report = {"verifier_version": VERSION, "status": "failed", "benchmark_success_claimed": False,
                  "error_type": type(error).__name__, "reason": reason}
        code = 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(args.out.name+".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+"\n")
    temporary.replace(args.out)
    print(json.dumps({k: report[k] for k in ("status", "completed_cells", "expected_cells", "reason") if k in report}))
    return code


if __name__ == "__main__":
    sys.exit(main())
