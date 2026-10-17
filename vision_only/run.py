#!/usr/bin/env python3
"""Full-source frozen-encoder experiment, with separate selection and confirmation.

Production extraction/training requires CUDA. CPU is permitted only for unit
fixtures and analysis. No dataset is silently downloaded, subsetted, or removed.
Run from the repository root with ``python -m vision_only.run --help``.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
import random
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from . import data

DEFAULT_STUDY = Path(__file__).with_name("study.json")
OFF_ARMS = ("natural_balanced_off", "uniform_off", "natural_ce", "balanced_softmax")
EQ_PARENT = {"natural_balanced_eq": "natural_balanced_off", "uniform_eq": "uniform_off"}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    return data.sha256_file(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    data.atomic_json(Path(path), value)


def fingerprint(value):
    return data.fingerprint(value)


def code_identity():
    return {p.name: digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))}


def require_cuda(device):
    require(str(device).startswith("cuda") and torch.cuda.is_available(),
            "Scientific execution requires the authorized remote CUDA machine; CPU/MPS is not a substitute")


def settings(path=DEFAULT_STUDY):
    value = read_json(path)
    require(value["schema_version"] == 1, "Unknown protocol version")
    require(value["training"]["adapter_dropout"] == 0, "Matched deterministic adapter expected")
    return value


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


class AdapterHead(nn.Module):
    def __init__(self, width, classes, hidden=256):
        super().__init__()
        self.adapter = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, hidden),
                                     nn.GELU(), nn.LayerNorm(hidden))
        self.head = nn.Linear(hidden, classes)

    def forward(self, x):
        h = self.adapter(x)
        return self.head(h), F.normalize(h, dim=-1)


def equalization(features, labels):
    """Class-average compactness minus average squared class-centre distance."""
    unique, counts = torch.unique(labels, return_counts=True)
    eligible = unique[counts >= 2]
    if not len(eligible):
        return features.sum() * 0, {"eligible_classes": 0, "represented_classes": len(unique)}
    means, compact = [], []
    for c in eligible:
        rows = features[labels == c]
        centre = rows.mean(0)
        means.append(centre)
        compact.append((rows - centre).square().sum(1).mean())
    within = torch.stack(compact).mean()
    centres = torch.stack(means)
    between = 2 * (centres - centres.mean(0)).square().sum() / (len(means) - 1) if len(means) > 1 else within * 0
    return within - between, {"eligible_classes": len(eligible), "represented_classes": len(unique),
                              "within": float(within.detach()), "between": float(between.detach())}


def classification_loss(logits, y, counts, arm):
    if arm == "balanced_softmax":
        return F.cross_entropy(logits + counts.log(), y)
    loss = F.cross_entropy(logits, y, reduction="none")
    if arm.startswith("natural_balanced"):
        loss = loss * counts.sum() / (len(counts) * counts[y])
    return loss.mean()


def batch_ids(counts, labels, n, sampler, generator):
    if sampler.startswith("uniform"):
        weights = 1.0 / counts[labels].double()
        return torch.multinomial(weights, n, replacement=True, generator=generator)
    return torch.randint(len(labels), (n,), generator=generator)


def class_metrics(labels, predictions, classes, order, top5=None):
    labels, predictions = np.asarray(labels), np.asarray(predictions)
    require(len(labels) > 0 and len(labels) == len(predictions), "Empty/mismatched predictions")
    support = np.bincount(labels, minlength=classes)
    predicted = np.bincount(predictions, minlength=classes)
    tp = np.bincount(labels[labels == predictions], minlength=classes)
    precision = np.divide(tp, predicted, out=np.zeros(classes), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros(classes), where=support > 0)
    f1 = np.divide(2 * tp, support + predicted, out=np.zeros(classes), where=support + predicted > 0)
    groups = np.array_split(np.asarray(order), 3)
    return {"macro_f1": float(f1.mean()), "top1": float((labels == predictions).mean()),
            "top5": None if top5 is None else float(np.mean(top5)),
            "macro_precision": float(precision.mean()), "macro_recall": float(recall.mean()),
            **{name + "_f1": float(f1[ids].mean()) for name, ids in zip(("head", "middle", "tail"), groups)},
            "per_class_precision": precision.tolist(), "per_class_recall": recall.tolist(),
            "per_class_f1": f1.tolist(), "support": support.tolist()}


def source_rows(manifest, split):
    lookup = {s.sample_id: i for i, s in enumerate(manifest.samples)}
    return {name: np.asarray([lookup[x] for x in getattr(split, name)], dtype=np.int64)
            for name in ("fit", "tune", "calibration", "confirmation")}


def labels_for(manifest):
    return np.asarray([s.label for s in manifest.samples], dtype=np.int64)


def feature_scaler(features, ids):
    total = np.zeros(features.shape[1], dtype=np.float64)
    squares = total.copy()
    for start in range(0, len(ids), 8192):
        x = np.asarray(features[ids[start:start + 8192]], dtype=np.float64)
        total += x.sum(0)
        squares += np.square(x).sum(0)
    mean = total / len(ids)
    sd = np.sqrt(np.maximum(squares / len(ids) - mean * mean, 1e-8))
    return mean.astype(np.float32), sd.astype(np.float32)


def feature_batch(features, ids, mean, sd, device):
    x = (np.asarray(features[ids], dtype=np.float32) - mean) / sd
    return torch.from_numpy(x).to(device)


def shared_scaler(features, fit_ids, identity, directory):
    key = fingerprint({"features": identity["cache"]["features_sha256"],
                       "partition": identity["partition"],
                       "fit_rows_sha256": hashlib.sha256(fit_ids.tobytes()).hexdigest()})
    path, receipt_path = directory / "fit_scaler.npz", directory / "fit_scaler.json"
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        require(receipt["fingerprint"] == key and digest(path) == receipt["sha256"], "Fit scaler changed")
        with np.load(path, allow_pickle=False) as values: mean, sd = values["mean"].copy(), values["sd"].copy()
    else:
        mean, sd = feature_scaler(features, fit_ids)
        directory.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp.npz")
        np.savez(temporary, mean=mean, sd=sd)
        os.replace(temporary, path)
        write_json(receipt_path, {"fingerprint": key, "fit_rows": len(fit_ids), "sha256": digest(path)})
    require(mean.shape == sd.shape == (features.shape[1],) and np.isfinite(mean).all()
            and np.isfinite(sd).all() and np.all(sd > 0), "Invalid fit-only scaler")
    return mean, sd


@torch.inference_mode()
def evaluate(model, features, rows, labels, mean, sd, classes, order, device, tau=0., log_prior=None, retain_top5=False):
    model.eval()
    require(np.isfinite(mean).all() and np.isfinite(sd).all() and np.all(sd > 0), "Invalid evaluation scaler")
    predicted, top5, top5_rows = [], [], []
    offset = torch.as_tensor(log_prior if log_prior is not None else np.zeros(classes), device=device)
    for start in range(0, len(rows), 512):
        ids = rows[start:start + 512]
        logits, _ = model(feature_batch(features, ids, mean, sd, device))
        require(bool(torch.isfinite(logits).all()), "Nonfinite evaluation logits")
        logits = logits - tau * offset
        require(bool(torch.isfinite(logits).all()), "Nonfinite adjusted logits")
        pred5 = logits.topk(min(5, classes), dim=1).indices.cpu().numpy()
        predicted.append(pred5[:, 0])
        if retain_top5: top5_rows.append(pred5)
        top5.extend(np.any(pred5 == labels[ids, None], axis=1).tolist())
    predictions = np.concatenate(predicted)
    result = class_metrics(labels[rows], predictions, classes, order, top5), predictions
    return (*result, np.concatenate(top5_rows)) if retain_top5 else result


def context(args):
    spec = settings(args.study)
    manifest = data.load_manifest(args.manifest)
    split = data.load_partitions(args.split, manifest)
    data.validate_partitions(manifest, split)
    require(manifest.dataset_id in spec["datasets"] and manifest.scope == "full", "Full production source required")
    cache = Path(args.cache)
    receipt = read_json(cache / "complete.json")
    require(receipt["manifest_fingerprint"] == manifest.fingerprint, "Different dataset cache")
    require(receipt["encoder"] == args.encoder, "Different encoder cache")
    require(receipt["model_name"] == spec["encoders"][args.encoder] and
            receipt["transform"] == spec["transform"] and receipt["rows"] == len(manifest.samples),
            "Feature model, preprocessing or source coverage changed")
    require(digest(cache / "pilot.json") == receipt["pilot_sha256"] and
            read_json(cache / "pilot.json")["confirmation_accuracy_scored"] is False,
            "Missing or changed fit-only CUDA pilot")
    require(digest(cache / "features.npy") == receipt["features_sha256"], "Changed/incomplete feature cache")
    features = np.load(cache / "features.npy", mmap_mode="r", allow_pickle=False)
    require(features.shape == (len(manifest.samples), 384) and features.dtype == np.float16,
            "Wrong feature-cache shape/type")
    rows, labels = source_rows(manifest, split), labels_for(manifest)
    identity = {"study_sha256": digest(args.study), "code": code_identity(),
                "manifest": manifest.fingerprint, "partition": split.fingerprint,
                "cache": receipt, "encoder": args.encoder, "fit_rows": len(rows["fit"])}
    return spec, manifest, split, features, rows, labels, identity


def _training_key(args, identity, arm, seed, lr, stage):
    return fingerprint({"identity": identity, "arm": arm, "seed": seed, "lr": lr, "stage": stage})


def validate_checkpoint(state, key, epochs, steps, complete=False):
    require(state["fingerprint"] == key, "Checkpoint identity changed")
    done = state["completed_epochs"]
    require(type(done) is int and 0 <= done <= epochs, "Invalid checkpoint epoch coverage")
    require(state["completed_updates"] == done * steps and state["steps_per_epoch"] == steps,
            "Checkpoint optimizer-update budget does not match epochs")
    require(len(state["history"]) == done and
            [r["epoch"] for r in state["history"]] == list(range(1, done + 1)),
            "Checkpoint history is not contiguous")
    if complete: require(done == epochs, "Actual checkpoint training budget incomplete")
    def finite(value):
        if torch.is_tensor(value):
            require(bool(torch.isfinite(value).all()), "Nonfinite checkpoint tensor")
        elif isinstance(value, dict):
            for item in value.values(): finite(item)
        elif isinstance(value, (list, tuple)):
            for item in value: finite(item)
    finite(state["model"])
    finite(state["optimizer"])
    finite(state["mean"])
    finite(state["sd"])
    require(bool((state["sd"] > 0).all()), "Invalid feature scaler")
    parameter_ids = [p for group in state["optimizer"]["param_groups"] for p in group["params"]]
    require(len(parameter_ids) == len(state["model"]) and len(set(parameter_ids)) == len(parameter_ids),
            "Optimizer/model parameter coverage changed")
    require(set(state["optimizer"]["state"]) == set(parameter_ids), "Incomplete optimizer parameter states")
    shapes = [tuple(v.shape) for v in state["model"].values()]
    for parameter_id, shape in zip(parameter_ids, shapes):
        item = state["optimizer"]["state"][parameter_id]
        require("step" in item and int(item["step"]) == state["completed_updates"],
                "Actual Adam optimizer step counter is incomplete")
        require(tuple(item["exp_avg"].shape) == shape and tuple(item["exp_avg_sq"].shape) == shape,
                "Optimizer moment shape changed")


def fit(args, spec, manifest, split, features, rows, labels, identity, arm, seed, lr, stage, out):
    config = spec["training"]
    key = _training_key(args, identity, arm, seed, lr, stage)
    complete = out / "trained.json"
    if complete.exists():
        record = read_json(complete)
        require(record["fingerprint"] == key and digest(out / "checkpoint.pt") == record["checkpoint_sha256"],
                "Changed completed run; use a new run directory")
        steps = math.ceil(len(rows["fit"]) / spec["training"]["batch_size"])
        validate_checkpoint(torch.load(out / "checkpoint.pt", map_location="cpu", weights_only=True),
                            key, config["epochs"], steps, complete=True)
        return record
    out.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)
    model = AdapterHead(features.shape[1], manifest.num_classes, config["adapter_width"]).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=config["weight_decay"])
    mean, sd = shared_scaler(features, rows["fit"], identity, out.parents[2])
    counts_np = np.bincount(labels[rows["fit"]], minlength=manifest.num_classes)
    require(np.all(counts_np > 0), "An expected fit class disappeared")
    counts = torch.as_tensor(counts_np, dtype=torch.float32, device=args.device)
    generator = torch.Generator().manual_seed(seed + 1001)
    fit_labels = torch.from_numpy(labels[rows["fit"]])
    cpu_counts = torch.from_numpy(counts_np)
    steps = math.ceil(len(rows["fit"]) / config["batch_size"])
    epochs, start_epoch, history = config["epochs"], 0, []
    checkpoint = out / "checkpoint.pt"
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        validate_checkpoint(state, key, epochs, steps)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        for values in optimizer.state.values():
            for k, v in values.items():
                if torch.is_tensor(v): values[k] = v.to(args.device)
        generator.set_state(state["generator"])
        start_epoch = state["completed_epochs"]
        history = state["history"]
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    torch.cuda.reset_peak_memory_stats()
    began = time.monotonic()
    for epoch in range(start_epoch, epochs):
        epoch_began = time.monotonic()
        model.train()
        chosen = batch_ids(cpu_counts, fit_labels, steps * config["batch_size"], arm, generator).numpy()
        active = arm in EQ_PARENT and epoch >= math.floor(epochs * config["eq_onset_fraction"])
        diagnostics = []
        eligible_counts = np.zeros(manifest.num_classes, dtype=np.int64)
        represented_counts = eligible_counts.copy()
        distinct_eligible_counts = eligible_counts.copy()
        phase_batches = 0
        epoch_loss = 0.
        for step in range(steps):
            batch = rows["fit"][chosen[step * config["batch_size"]:(step + 1) * config["batch_size"]]]
            y = torch.from_numpy(labels[batch]).to(args.device)
            if epoch >= math.floor(epochs * config["eq_onset_fraction"]):
                batch_counts = np.bincount(labels[batch], minlength=manifest.num_classes)
                eligible_counts += batch_counts >= 2
                represented_counts += batch_counts > 0
                distinct_counts = np.bincount(labels[np.unique(batch)], minlength=manifest.num_classes)
                distinct_eligible_counts += distinct_counts >= 2
                phase_batches += 1
            t = epoch * steps + step
            warmup = max(1, int(config["warmup_fraction"] * epochs * steps))
            fraction = min(1., t / warmup) if t < warmup else .5 * (1 + math.cos(math.pi * (t - warmup) / max(1, epochs * steps - warmup)))
            for group in optimizer.param_groups: group["lr"] = lr * fraction
            logits, h = model(feature_batch(features, batch, mean, sd, args.device))
            ce = classification_loss(logits, y, counts, arm)
            eq, diag = equalization(h, y)
            diag.update(active_phase=bool(active), step=t, classification_loss=float(ce.detach()),
                        eq_loss=float(eq.detach()))
            if active and step % 100 == 0:
                params = tuple(model.adapter.parameters())
                a = torch.autograd.grad(ce, params, retain_graph=True, allow_unused=True)
                b = torch.autograd.grad(config["eq_coefficient"] * eq, params, retain_graph=True, allow_unused=True)
                norm = lambda grads: torch.sqrt(sum(g.square().sum() for g in grads if g is not None))
                diag["weighted_EQ_to_CE_gradient_norm"] = float((norm(b) / norm(a).clamp_min(1e-12)).detach())
            loss = ce + (config["eq_coefficient"] * eq if active else eq * 0)
            require(bool(torch.isfinite(loss)), "Nonfinite loss; run is not a completed result")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
            epoch_loss += float(loss.detach())
            if step % 100 == 0 or step == steps - 1: diagnostics.append(diag)
        history.append({"epoch": epoch + 1, "mean_loss": epoch_loss / steps, "diagnostics": diagnostics,
                        "completed_epoch_seconds": time.monotonic() - epoch_began,
                        "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
                        "late_schedule_phase_batches": phase_batches,
                        "late_phase_eligible_batches_per_class": eligible_counts.tolist(),
                        "late_phase_distinct_image_eligible_batches_per_class": distinct_eligible_counts.tolist(),
                        "late_phase_represented_batches_per_class": represented_counts.tolist()})
        state = {"fingerprint": key, "model": model.cpu().state_dict(), "optimizer": optimizer.state_dict(),
                 "completed_epochs": epoch + 1, "generator": generator.get_state(),
                 "completed_updates": (epoch + 1) * steps, "steps_per_epoch": steps, "history": history,
                 "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
                 "mean": torch.from_numpy(mean), "sd": torch.from_numpy(sd),
                 "classes": manifest.num_classes, "width": features.shape[1]}
        validate_checkpoint(state, key, epochs, steps)
        tmp = checkpoint.with_suffix(".tmp")
        torch.save(state, tmp)
        os.replace(tmp, checkpoint)
        model.to(args.device)
        write_json(out / "history.json", history)
        print(json.dumps({"stage": stage, "arm": arm, "seed": seed, "epoch": epoch + 1, "epochs": epochs}), flush=True)
    model.eval()
    metrics = None
    if stage == "tuning":
        metrics, tuning_predictions, tuning_top5 = evaluate(model, features, rows["tune"], labels, mean, sd, manifest.num_classes,
                              split.class_order, args.device, retain_top5=True)
        np.savez_compressed(out / "tuning_predictions.npz", row_indices=rows["tune"],
                            labels=labels[rows["tune"]], predictions=tuning_predictions, top5=tuning_top5)
    record = {"status": "trained", "fingerprint": key, "identity": identity, "arm": arm,
              "seed": seed, "lr": lr, "stage": stage, "completed_epochs": epochs,
              "checkpoint_sha256": digest(checkpoint), "tune_metrics": metrics,
              "tuning_predictions_sha256": digest(out / "tuning_predictions.npz") if stage == "tuning" else None,
              "elapsed_seconds_this_invocation": time.monotonic() - began,
              "completed_epoch_training_seconds": sum(r["completed_epoch_seconds"] for r in history),
              "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
              "packages": {k: importlib.metadata.version(k) for k in ("torch", "torchvision", "timm", "numpy")}}
    write_json(complete, record)
    return record


def tune(args):
    require_cuda(args.device)
    spec, manifest, split, features, rows, labels, identity = context(args)
    out = Path(args.out)
    require(not (out / "selection.json").exists(), "Selection already frozen; do not rerun selection")
    config, records = spec["training"], {}
    for arm in OFF_ARMS:
        records[arm] = []
        for i, lr in enumerate(config["learning_rate_candidates"]):
            record = fit(args, spec, manifest, split, features, rows, labels, identity, arm,
                         config["tuning_seed"], lr, "tuning", out / "tuning" / arm / str(i))
            records[arm].append(record)
    scores = [np.mean([records[a][i]["tune_metrics"]["macro_f1"]
                       for a in ("natural_balanced_off", "uniform_off")])
              for i in range(len(config["learning_rate_candidates"]))]
    common_index = int(np.argmax(scores))
    chosen = {a: config["learning_rate_candidates"][common_index]
              for a in ("natural_balanced_off", "natural_balanced_eq", "uniform_off", "uniform_eq")}
    for arm in ("natural_ce", "balanced_softmax"):
        index = int(np.argmax([r["tune_metrics"]["macro_f1"] for r in records[arm]]))
        chosen[arm] = config["learning_rate_candidates"][index]
    selection = {"identity": identity, "lr_by_arm": chosen, "tuning_fingerprints":
                 {a: [r["fingerprint"] for r in values] for a, values in records.items()},
                 "rule": "mean raw tuning macroF1 across two EQ-off factorial cells; common LR for all four; anchors separately",
                 "confirmation_scored": False}
    selection["fingerprint"] = fingerprint(selection)
    write_json(out / "selection.json", selection)


def selection_record(out, identity):
    selected = read_json(out / "selection.json")
    copy_ = {k: v for k, v in selected.items() if k != "fingerprint"}
    require(selected["fingerprint"] == fingerprint(copy_) and selected["identity"] == identity,
            "Selection receipt changed or belongs to different source/cache/protocol")
    require(selected.get("confirmation_scored") is False, "Selection was not frozen before confirmation")
    return selected


def train(args):
    require_cuda(args.device)
    spec, manifest, split, features, rows, labels, identity = context(args)
    out = Path(args.out)
    selected = selection_record(out, identity)
    for seed in spec["training"]["final_seeds"]:
        for arm in spec["arms"]:
            fit(args, spec, manifest, split, features, rows, labels, identity, arm, seed,
                selected["lr_by_arm"][arm], "final", out / "final" / arm / str(seed))


def confirmation_gate(out, spec, identity, selected):
    records = {}
    for arm, seed in itertools.product(spec["arms"], spec["training"]["final_seeds"]):
        cell = out / "final" / arm / str(seed)
        record = read_json(cell / "trained.json")
        expected = fingerprint({"identity": identity, "arm": arm, "seed": seed,
                                "lr": selected["lr_by_arm"][arm], "stage": "final"})
        require(record["status"] == "trained" and record["fingerprint"] == expected,
                "Incomplete or changed final grid; confirmation blocked")
        require(digest(cell / "checkpoint.pt") == record["checkpoint_sha256"], "Changed checkpoint")
        require(record["completed_epochs"] == spec["training"]["epochs"], "Budget incomplete")
        checkpoint = torch.load(cell / "checkpoint.pt", map_location="cpu", weights_only=True)
        expected_steps = math.ceil(identity["fit_rows"] / spec["training"]["batch_size"])
        validate_checkpoint(checkpoint, expected, spec["training"]["epochs"],
                            expected_steps, complete=True)
        records[(arm, seed)] = record
    return records


def confirm(args):
    require_cuda(args.device)
    spec, manifest, split, features, rows, labels, identity = context(args)
    out = Path(args.out)
    require(getattr(args, "global_seal", None) is not None, "Global all-dataset confirmation seal required")
    verify_global_seal(Path(args.global_seal), args.study, identity)
    selected = selection_record(out, identity)
    records = confirmation_gate(out, spec, identity, selected)
    for (arm, seed), record in records.items():
        cell = out / "final" / arm / str(seed)
        if (cell / "confirmed.json").exists():
            existing = read_json(cell / "confirmed.json")
            require(existing["trained_fingerprint"] == record["fingerprint"] and
                    digest(cell / "predictions.npz") == existing["predictions_sha256"], "Changed confirmed results")
            continue
        state = torch.load(cell / "checkpoint.pt", map_location="cpu", weights_only=True)
        model = AdapterHead(state["width"], state["classes"], spec["training"]["adapter_width"]).to(args.device)
        model.load_state_dict(state["model"])
        mean, sd = state["mean"].numpy(), state["sd"].numpy()
        counts = np.bincount(labels[rows["fit"]], minlength=manifest.num_classes)
        # The balanced-target factorial and Balanced Softmax have uniform effective priors.
        log_prior = np.log(counts / counts.sum()) if arm == "natural_ce" else np.zeros(manifest.num_classes)
        grid = np.linspace(0, 2, 41) if arm == "natural_ce" else [0.]
        calibration_results = [evaluate(model, features, rows["calibration"], labels, mean, sd, manifest.num_classes,
                                split.class_order, args.device, float(t), log_prior) for t in grid]
        calibration = [item[0]["macro_f1"] for item in calibration_results]
        tau = float(grid[int(np.argmax(calibration))])
        raw, raw_pred, raw_top5 = evaluate(model, features, rows["confirmation"], labels, mean, sd, manifest.num_classes,
                                 split.class_order, args.device, retain_top5=True)
        adjusted, pred, adjusted_top5 = evaluate(model, features, rows["confirmation"], labels, mean, sd, manifest.num_classes,
                                  split.class_order, args.device, tau, log_prior, retain_top5=True)
        np.savez_compressed(cell / "calibration_predictions.npz", row_indices=rows["calibration"],
                            labels=labels[rows["calibration"]], tau_grid=np.asarray(grid),
                            predictions_by_tau=np.stack([item[1] for item in calibration_results]))
        np.savez_compressed(cell / "predictions.npz", row_indices=rows["confirmation"],
                            labels=labels[rows["confirmation"]], raw_predictions=raw_pred, adjusted_predictions=pred,
                            raw_top5=raw_top5, adjusted_top5=adjusted_top5)
        write_json(cell / "confirmed.json", {"status": "complete", "trained_fingerprint": record["fingerprint"],
                   "selection_fingerprint": selected["fingerprint"], "arm": arm, "seed": seed,
                   "tau": tau, "calibration_grid": [float(t) for t in grid], "calibration_scores": calibration,
                   "calibration_predictions_sha256": digest(cell / "calibration_predictions.npz"),
                   "raw": raw, "adjusted": adjusted, "predictions_sha256": digest(cell / "predictions.npz")})


def verify_global_seal(path, study, identity):
    from .control import REGIMES, expected_run_ids
    spec, seal = settings(study), read_json(path)
    unsigned = {k: v for k, v in seal.items() if k != "fingerprint"}
    require(seal["fingerprint"] == fingerprint(unsigned) and seal["study_sha256"] == digest(study)
            and seal["scientific_source"] == code_identity(), "Changed global confirmation seal")
    expected_ids = expected_run_ids(study)
    require(seal["expected_run_ids"] == expected_ids and seal["completed_final_fits"] == len(expected_ids),
            "Global run inventory incomplete")
    expected_contexts = {(d, r, e) for d, r, _ in REGIMES for e in spec["encoders"]}
    require({(c["dataset"], c["regime"], c["encoder"]) for c in seal["cells"]} == expected_contexts
            and len(seal["cells"]) == len(expected_contexts), "Missing/duplicate global context")
    require(sum(c["identity"] == identity for c in seal["cells"]) == 1, "Current context not bound to global seal")
    actual_ids = []
    for cell in seal["cells"]:
        context_dir = path.parent / "runs" / cell["dataset"] / cell["regime"] / cell["encoder"]
        require(digest(context_dir / "selection.json") == cell["selection_sha256"], "Selection changed after global freeze")
        for item in cell["runs"]:
            run_id = "/".join(map(str, (cell["dataset"], cell["regime"], cell["encoder"], item["arm"], item["seed"])))
            require(run_id in expected_ids, "Unknown final run in seal")
            actual_ids.append(run_id)
            checkpoint_dir = context_dir / "final" / item["arm"] / str(item["seed"])
            record = read_json(checkpoint_dir / "trained.json")
            require(record["fingerprint"] == item["trained_fingerprint"] and
                    record["checkpoint_sha256"] == item["checkpoint_sha256"] and
                    digest(checkpoint_dir / "checkpoint.pt") == item["checkpoint_sha256"],
                    "Final checkpoint changed after global freeze")
    require(sorted(actual_ids) == sorted(expected_ids), "Incomplete or repeated global final-run grid")


class ManifestImages(torch.utils.data.Dataset):
    def __init__(self, manifest, transform):
        self.manifest, self.transform = manifest, transform
        self.cifar = None
        if manifest.dataset_id == "cifar100":
            from torchvision.datasets import CIFAR100
            self.cifar = {s: CIFAR100(manifest.source_root, train=s == "train", download=False)
                          for s in ("train", "confirmation")}

    def __len__(self): return len(self.manifest.samples)

    def __getitem__(self, index):
        sample = self.manifest.samples[index]
        if self.cifar:
            image, label = self.cifar[sample.source_split][sample.record_index]
            require(label == sample.label, "CIFAR row/label identity changed")
        else:
            from PIL import Image
            with Image.open(data.contained_path(self.manifest.source_root, sample.relative_path)) as im:
                image = im.convert("RGB")
                image.load()
        return self.transform(image)


def extract(args):
    require_cuda(args.device)
    import timm
    from torchvision import transforms
    spec = settings(args.study)
    manifest = data.load_manifest(args.manifest)
    data.verify_manifest_files(manifest)
    require(manifest.dataset_id in spec["datasets"] and manifest.scope == "full", "Full source required")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = timm.create_model(spec["encoders"][args.encoder], pretrained=True, num_classes=0, img_size=224).eval().to(args.device)
    for parameter in model.parameters(): parameter.requires_grad_(False)
    state_hash = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        state_hash.update(name.encode())
        state_hash.update(value.detach().cpu().contiguous().numpy().tobytes())
    identity = {"manifest_fingerprint": manifest.fingerprint, "encoder": args.encoder,
                "model_name": spec["encoders"][args.encoder], "model_state_sha256": state_hash.hexdigest(),
                "transform": spec["transform"], "timm": timm.__version__, "torch": torch.__version__,
                "source_code_sha256": digest(__file__)}
    complete = out / "complete.json"
    if complete.exists():
        old = read_json(complete)
        require(all(old[k] == v for k, v in identity.items()) and
                digest(out / "features.npy") == old["features_sha256"], "Changed feature cache")
        return
    transform = transforms.Compose([transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(spec["transform"]["mean"], spec["transform"]["std"])])
    corpus = ManifestImages(manifest, transform)
    # A throughput/precision pilot sees fit-source images only and reports no accuracy.
    pilot_ids = [i for i, s in enumerate(manifest.samples)
                 if s.source_split == "train" and s.sample_id not in manifest.exclusions][:64]
    if getattr(args, "split", None) is not None:
        pilot_split = data.load_partitions(args.split, manifest)
        pilot_ids = source_rows(manifest, pilot_split)["fit"][:64].tolist()
    require(len(pilot_ids) > 0, "No fit-only pilot rows")
    pilot_began = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        pilot_x = torch.stack([corpus[i] for i in pilot_ids]).to(args.device)
        pilot_features = model(pilot_x).float()
        restored = pilot_features.half().float()
        delta = (restored - pilot_features).abs()
        require(bool(torch.isfinite(restored).all()) and float(delta.mean()) <= .001
                and float(delta.max()) <= .03, "Fit-only feature precision pilot failed")
        torch.cuda.synchronize()
        pilot = {"identity": identity, "train_source_only": True, "rows": len(pilot_ids),
                 "confirmation_accuracy_scored": False, "elapsed_seconds": time.monotonic() - pilot_began,
                 "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
                 "mean_float16_feature_error": float(delta.mean()), "max_float16_feature_error": float(delta.max())}
        pilot_inputs = pilot_features.clone()
    # Exercise the actual CUDA loss/backward/optimizer path before committing a large cache.
    seed_everything(1901)
    pilot_head = AdapterHead(384, manifest.num_classes, spec["training"]["adapter_width"]).to(args.device)
    pilot_optimizer = torch.optim.AdamW(pilot_head.parameters(), lr=.001)
    pilot_labels = torch.tensor([manifest.samples[i].label for i in pilot_ids], device=args.device)
    pilot_inputs = pilot_inputs.detach().clone()
    for _ in range(2):
        pilot_logits, pilot_h = pilot_head(pilot_inputs)
        pilot_eq, _ = equalization(pilot_h, pilot_labels)
        pilot_loss = F.cross_entropy(pilot_logits, pilot_labels) + .05 * pilot_eq
        require(bool(torch.isfinite(pilot_loss)), "CUDA backward pilot has nonfinite loss")
        pilot_optimizer.zero_grad(set_to_none=True)
        pilot_loss.backward()
        nn.utils.clip_grad_norm_(pilot_head.parameters(), 1., error_if_nonfinite=True)
        pilot_optimizer.step()
    pilot["cuda_optimizer_pilot_updates"] = 2
    pilot["formal_training_fit"] = False
    del pilot_head, pilot_optimizer, pilot_inputs, pilot_labels, pilot_logits, pilot_h, pilot_loss, pilot_eq
    del pilot_x, pilot_features, restored, delta
    write_json(out / "pilot.json", pilot)
    progress = out / "progress.json"
    if progress.exists():
        previous = read_json(progress)
        done = previous["completed_rows"]
        cache = np.load(out / "features.npy", mmap_mode="r+", allow_pickle=False)
        validate_progress(cache, previous, identity, len(corpus))
    else:
        done = 0
        cache = np.lib.format.open_memmap(out / "features.npy", mode="w+", dtype=np.float16, shape=(len(corpus), 384))
        previous = {"identity": identity, "completed_rows": 0, "blocks": []}
        write_json(progress, previous)
    require(cache.shape == (len(corpus), 384), "Extraction cache shape mismatch")
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(corpus, range(done, len(corpus))),
        batch_size=args.batch_size, num_workers=args.workers, shuffle=False, pin_memory=True)
    began = time.monotonic()
    committed = done
    with torch.inference_mode():
        for x in loader:
            features = model(x.to(args.device)).float().cpu().numpy()
            require(features.shape == (len(x), 384) and np.isfinite(features).all(), "Invalid encoder features")
            converted = features.astype(np.float16)
            require(np.isfinite(converted).all(), "Float16 cache overflow")
            cache[done:done + len(x)] = converted
            done += len(x)
            if done - committed >= 8192 or done == len(corpus):
                cache.flush()
                previous["blocks"].append({"start": committed, "end": done,
                    "sha256": hashlib.sha256(np.asarray(cache[committed:done]).tobytes()).hexdigest()})
                previous["completed_rows"] = done
                write_json(progress, previous)
                committed = done
            print(json.dumps({"extracted": done, "total": len(corpus), "encoder": args.encoder}), flush=True)
    require(done == len(corpus), "Incomplete extraction")
    write_json(complete, {**identity, "rows": done, "features_sha256": digest(out / "features.npy"),
                          "pilot_sha256": digest(out / "pilot.json"),
                          "elapsed_seconds_this_invocation": time.monotonic() - began})


def validate_progress(cache, progress, identity, rows):
    require(progress["identity"] == identity, "Extraction resume identity changed")
    done = progress["completed_rows"]
    require(type(done) is int and 0 <= done <= rows, "Invalid extraction progress range")
    require(cache.shape == (rows, 384) and cache.dtype == np.float16, "Invalid extraction cache layout")
    cursor = 0
    for block in progress["blocks"]:
        require(block["start"] == cursor and cursor < block["end"] <= done, "Noncontiguous extraction blocks")
        values = np.asarray(cache[cursor:block["end"]])
        require(np.isfinite(values).all() and hashlib.sha256(values.tobytes()).hexdigest() == block["sha256"],
                "Changed/nonfinite committed feature block")
        cursor = block["end"]
    require(cursor == done, "Unverified feature prefix")


def inventory(args):
    spec = data.dataset_spec(args.dataset)
    if args.dataset == "imagenet1k": manifest = data.inventory_imagefolder(args.root, spec)
    elif args.dataset == "inat2018":
        require(args.train_json and args.val_json, "Official iNat annotations required")
        manifest = data.inventory_inat2018(args.root, args.train_json, args.val_json, spec)
    else: manifest = data.inventory_cifar100(args.root, spec)
    manifest = data.exclude_training_exact_duplicates(manifest)
    data.save_manifest(manifest, args.out)
    print(json.dumps({"dataset": manifest.dataset_id, "scope": manifest.scope, "source_counts": manifest.expected_counts,
                      "exclusions": len(manifest.exclusions), "manifest_fingerprint": manifest.fingerprint}))


def partition(args):
    manifest = data.load_manifest(args.manifest)
    spec = settings(args.study)
    dev = spec["development"]
    plan = data.make_partitions(manifest, dev["split_seed"], tune_fraction=dev["tune_fraction"],
                               calibration_fraction=dev["calibration_fraction"], lt_ratio=args.lt_ratio)
    data.save_partitions(plan, args.out)
    print(json.dumps({"regime": plan.regime, "partition_fingerprint": plan.fingerprint,
                      "counts": {k: len(getattr(plan, k)) for k in ("fit", "tune", "calibration", "confirmation", "lt_omitted")}}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("inventory")
    p.add_argument("--dataset", choices=("imagenet1k", "cifar100", "inat2018"), required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--train-json", type=Path)
    p.add_argument("--val-json", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(action=inventory)
    p = sub.add_parser("partition")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--lt-ratio", type=float)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(action=partition)
    p = sub.add_parser("extract")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--encoder", choices=("deit_small", "dinov2_small"), required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=8)
    p.set_defaults(action=extract)
    for name, action in (("tune", tune), ("train", train), ("confirm", confirm)):
        p = sub.add_parser(name)
        for option in ("manifest", "split", "cache", "out"): p.add_argument("--" + option, type=Path, required=True)
        p.add_argument("--encoder", choices=("deit_small", "dinov2_small"), required=True)
        p.add_argument("--device", default="cuda")
        if name == "confirm": p.add_argument("--global-seal", type=Path, required=True)
        p.set_defaults(action=action)
    args = parser.parse_args()
    args.action(args)


if __name__ == "__main__":
    main()
