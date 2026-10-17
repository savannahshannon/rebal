"""Run the complete declared vision grid; seal all final fits before confirmation.

The paths file contains local dataset locations, not credentials. Datasets must
already be obtained from their authorized official sources. This controller
does not delete data or accept dataset terms. Cleanup is a separate verified
lifecycle operation after export.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy import stats

from . import data, run

REGIMES = (("imagenet1k", "native", None), ("imagenet1k", "lt100", 100.),
           ("cifar100", "native", None), ("cifar100", "lt100", 100.),
           ("inat2018", "native", None))


def selected_datasets(args):
    """An optional scheduling filter; it never changes the declared run grid."""
    selected = getattr(args, "datasets", None)
    if selected is None: return None
    available = run.settings(args.study)["datasets"]
    run.require(bool(selected) and len(set(selected)) == len(selected)
                and set(selected) <= set(available), "Invalid or repeated dataset scheduling filter")
    return tuple(dataset for dataset in available if dataset in selected)


def contexts(root, study, device="cuda", datasets=None):
    spec = run.settings(study)
    for dataset, regime, _ in REGIMES:
        if datasets is not None and dataset not in datasets: continue
        for encoder in spec["encoders"]:
            yield dataset, regime, encoder, SimpleNamespace(study=Path(study),
                manifest=root / "manifests" / (dataset + ".json"),
                split=root / "partitions" / dataset / (regime + ".json"),
                cache=root / "cache" / dataset / encoder,
                out=root / "runs" / dataset / regime / encoder, encoder=encoder, device=device)


def expected_run_ids(study):
    spec = run.settings(study)
    return ["/".join(map(str, (d, r, e, arm, seed)))
            for d, r, _ in REGIMES for e, arm, seed in
            itertools.product(spec["encoders"], spec["arms"], spec["training"]["final_seeds"])]


def prepare(args):
    paths, spec = run.read_json(args.paths), run.settings(args.study)
    selected = selected_datasets(args)
    required = set(spec["datasets"] if selected is None else selected)
    run.require(set(paths) == required, "Paths must contain exactly the scheduled full datasets")
    for dataset, item in paths.items():
        path = args.work / "manifests" / (dataset + ".json")
        if path.exists():
            manifest = data.load_manifest(path)
            data.verify_manifest_files(manifest)
            run.require(manifest.dataset_id == dataset, "Dataset identity mismatch")
        else:
            run.inventory(SimpleNamespace(dataset=dataset, root=Path(item["root"]),
                train_json=item.get("train_json"), val_json=item.get("val_json"), out=path))
    for dataset, regime, ratio in REGIMES:
        if dataset not in required: continue
        destination = args.work / "partitions" / dataset / (regime + ".json")
        manifest = data.load_manifest(args.work / "manifests" / (dataset + ".json"))
        expected = data.make_partitions(manifest, spec["development"]["split_seed"],
            tune_fraction=spec["development"]["tune_fraction"],
            calibration_fraction=spec["development"]["calibration_fraction"], lt_ratio=ratio)
        if destination.exists():
            run.require(data.load_partitions(destination, manifest).fingerprint == expected.fingerprint,
                        "Existing partition conflicts with declared protocol")
        else: data.save_partitions(expected, destination)
    status = {"status": "prepared", "expected_final_fits": len(expected_run_ids(args.study)),
        "completed_final_fits": 0, "confirmation_scored": False,
        "study_sha256": run.digest(args.study), "scientific_source": run.code_identity()}
    if selected is not None:
        status.update(status="selected_datasets_prepared", scheduled_datasets=list(selected),
                      all_final_fits_verified=False)
        # Preparation does not inspect existing training checkpoints. Do not
        # report a global completion count for an incremental preparation.
        status.pop("completed_final_fits")
    run.write_json(args.work / "execution_status.json", status)


def extract(args):
    run.require_cuda(args.device)
    seen = set()
    for dataset, _, encoder, context in contexts(args.work, args.study, args.device, selected_datasets(args)):
        if (dataset, encoder) in seen: continue
        seen.add((dataset, encoder))
        context.batch_size, context.workers, context.out = args.batch_size, args.workers, context.cache
        run.extract(context)


def global_seal(root, study, device="cuda", write=False):
    """Inspect actual checkpoints for all datasets/encoders before any scoring."""
    spec, cells = run.settings(study), []
    for dataset, regime, encoder, args in contexts(root, study, device):
        # context verifies cache/model/transform/source and binds fit row counts.
        _, _, split, _, _, _, identity = run.context(args)
        ratio = next(ratio for d, r, ratio in REGIMES if (d, r) == (dataset, regime))
        run.require(split.lt_ratio == ratio and split.seed == spec["development"]["split_seed"]
                    and split.tune_fraction == spec["development"]["tune_fraction"]
                    and split.calibration_fraction == spec["development"]["calibration_fraction"],
                    "Partition does not match its declared condition")
        selected = run.selection_record(args.out, identity)
        records = run.confirmation_gate(args.out, spec, identity, selected)
        cells.append({"dataset": dataset, "regime": regime, "encoder": encoder,
            "identity": identity, "selection_sha256": run.digest(args.out / "selection.json"),
            "runs": [{"arm": arm, "seed": seed, "trained_fingerprint": record["fingerprint"],
                      "checkpoint_sha256": record["checkpoint_sha256"]}
                     for (arm, seed), record in sorted(records.items())]})
    expected = {"study_sha256": run.digest(study), "scientific_source": run.code_identity(),
                "expected_run_ids": expected_run_ids(study), "cells": cells,
                "completed_final_fits": sum(len(c["runs"]) for c in cells)}
    run.require(expected["completed_final_fits"] == len(expected["expected_run_ids"]), "Global grid incomplete")
    expected["fingerprint"] = run.fingerprint(expected)
    path = root / "global_confirmation_seal.json"
    if path.exists(): run.require(run.read_json(path) == expected, "Global preconfirmation seal changed")
    elif write: run.write_json(path, expected)
    else: raise ValueError("Global confirmation seal is missing")
    return expected


def fit(args):
    run.require_cuda(args.device)
    selected = selected_datasets(args)
    completed = 0
    for _, _, _, context in contexts(args.work, args.study, args.device, selected):
        if not (context.out / "selection.json").exists(): run.tune(context)
        run.train(context)
        if selected is not None:
            # Count actual validated checkpoints for this scheduling scope,
            # never a global completion claim based on finished function calls.
            spec, _, _, _, _, _, identity = run.context(context)
            selection = run.selection_record(context.out, identity)
            completed += len(run.confirmation_gate(context.out, spec, identity, selection))
    if selected is not None:
        run.write_json(args.work / "execution_status.json", {"status": "selected_final_fits_trained",
            "scheduled_datasets": list(selected), "completed_scheduled_final_fits": completed,
            "expected_final_fits": len(expected_run_ids(args.study)), "all_final_fits_verified": False,
            "confirmation_scored": False, "study_sha256": run.digest(args.study),
            "scientific_source": run.code_identity()})
        return
    seal = global_seal(args.work, args.study, args.device, write=True)
    run.write_json(args.work / "execution_status.json", {"status": "all_final_fits_trained",
        "completed_final_fits": seal["completed_final_fits"], "confirmation_scored": False,
        "global_seal_fingerprint": seal["fingerprint"]})


def confirm(args):
    run.require(getattr(args, "datasets", None) is None, "Confirmation requires the complete declared dataset grid")
    run.require_cuda(args.device)
    seal = global_seal(args.work, args.study, args.device)
    for _, _, _, context in contexts(args.work, args.study, args.device):
        context.global_seal = args.work / "global_confirmation_seal.json"
        run.confirm(context)
    run.write_json(args.work / "execution_status.json", {"status": "confirmation_complete",
        "completed_final_fits": seal["completed_final_fits"], "confirmation_scored": True,
        "global_seal_fingerprint": seal["fingerprint"]})


def paired_summary(differences):
    d = np.asarray(differences, dtype=float)
    run.require(len(d) >= 2 and np.isfinite(d).all(), "Invalid paired differences")
    mean, se = d.mean(), d.std(ddof=1) / math.sqrt(len(d))
    q = stats.t.ppf(.975, len(d) - 1)
    signs = np.asarray(list(itertools.product((-1., 1.), repeat=len(d))))
    p = np.mean(np.abs((signs * d).mean(1)) >= abs(mean) - 1e-12)
    return {"effect_pp": float(mean * 100), "ci_low_pp": float((mean - q * se) * 100),
            "ci_high_pp": float((mean + q * se) * 100), "p_exact": float(p),
            "seed_differences": d.tolist()}


def holm(values):
    p = np.asarray(values, dtype=float)
    run.require(p.ndim == 1 and len(p) > 0 and np.isfinite(p).all() and np.all((p >= 0) & (p <= 1)),
                "Holm probabilities must be a nonempty finite vector in [0,1]")
    order = np.argsort(p, kind="stable")
    result = np.empty(len(p))
    maximum = 0.
    for rank, i in enumerate(order):
        maximum = max(maximum, (len(p) - rank) * p[i])
        result[i] = min(1., maximum)
    return result.tolist()


def analyze(args):
    run.require(getattr(args, "datasets", None) is None, "Analysis requires the complete declared dataset grid")
    spec = run.settings(args.study)
    seal = global_seal(args.work, args.study, args.device)
    all_metrics, comparisons = [], []
    destination = args.work / "aggregate"
    destination.mkdir(exist_ok=True)
    class_stream = (destination / "per_class.csv.tmp").open("w", newline="")
    class_writer = csv.DictWriter(class_stream, fieldnames=["dataset", "regime", "encoder", "arm", "seed",
        "readout", "class_id", "fit_count", "support", "precision", "recall", "f1"])
    class_writer.writeheader()
    lookup = {}
    try:
        for dataset, regime, encoder, context in contexts(args.work, args.study, args.device):
            split = run.read_json(context.split)
            counts = split["class_counts"]["fit"]
            for arm, seed in itertools.product(spec["arms"], spec["training"]["final_seeds"]):
                cell = context.out / "final" / arm / str(seed)
                value = run.read_json(cell / "confirmed.json")
                trained = run.read_json(cell / "trained.json")
                run.require(value["status"] == "complete" and value["trained_fingerprint"] == trained["fingerprint"],
                            "Missing/changed confirmation")
                run.require(run.digest(cell / "predictions.npz") == value["predictions_sha256"] and
                            run.digest(cell / "calibration_predictions.npz") == value["calibration_predictions_sha256"],
                            "Prediction artifacts changed")
                with np.load(cell / "calibration_predictions.npz", allow_pickle=False) as saved:
                    grid, predictions = saved["tau_grid"], saved["predictions_by_tau"]
                    scores = [run.class_metrics(saved["labels"], p, len(counts), split["class_order"])["macro_f1"]
                              for p in predictions]
                    run.require(list(grid) == value["calibration_grid"] and scores == value["calibration_scores"]
                                and float(grid[int(np.argmax(scores))]) == value["tau"],
                                "Decision-offset selection differs from saved calibration predictions")
                with np.load(cell / "predictions.npz", allow_pickle=False) as raw:
                    for readout in ("raw", "adjusted"):
                        recomputed = run.class_metrics(raw["labels"], raw[readout + "_predictions"], len(counts),
                            split["class_order"], np.any(raw[readout + "_top5"] == raw["labels"][:, None], axis=1))
                        run.require(recomputed == value[readout], "Reported metrics differ from saved predictions")
                        for c in range(len(counts)):
                            class_writer.writerow({"dataset": dataset, "regime": regime, "encoder": encoder,
                                "arm": arm, "seed": seed, "readout": readout, "class_id": c, "fit_count": counts[c],
                                "support": value[readout]["support"][c], "precision": value[readout]["per_class_precision"][c],
                                "recall": value[readout]["per_class_recall"][c], "f1": value[readout]["per_class_f1"][c]})
                scalar = {k: v for k, v in value["adjusted"].items() if not isinstance(v, list)}
                all_metrics.append({"dataset": dataset, "regime": regime, "encoder": encoder,
                    "arm": arm, "seed": seed, "tau": value["tau"], **scalar})
                lookup[(dataset, regime, encoder, arm, seed)] = scalar["macro_f1"]
        for dataset, regime, _ in REGIMES:
            effects = {}
            for encoder in spec["encoders"]:
                for sampler, off, on in (("natural", "natural_balanced_off", "natural_balanced_eq"),
                                         ("uniform", "uniform_off", "uniform_eq")):
                    d = [lookup[(dataset, regime, encoder, on, s)] - lookup[(dataset, regime, encoder, off, s)]
                         for s in spec["training"]["final_seeds"]]
                    effects[(encoder, sampler)] = np.asarray(d)
                    comparisons.append({"dataset": dataset, "regime": regime,
                        "contrast": sampler + "_EQ_" + encoder, **paired_summary(d)})
            for sampler in ("natural", "uniform"):
                comparisons.append({"dataset": dataset, "regime": regime,
                    "contrast": sampler + "_encoder_by_EQ_interaction",
                    **paired_summary(effects[("dinov2_small", sampler)] - effects[("deit_small", sampler)])})
        for dataset, regime, _ in REGIMES:
            subset = [r for r in comparisons if (r["dataset"], r["regime"]) == (dataset, regime)]
            for row, adjusted in zip(subset, holm([r["p_exact"] for r in subset])): row["p_holm_condition"] = adjusted
        for row, adjusted in zip(comparisons, holm([r["p_exact"] for r in comparisons])): row["p_holm_all"] = adjusted
    finally:
        class_stream.close()
    (destination / "per_class.csv.tmp").replace(destination / "per_class.csv")
    for name, rows in (("per_seed", all_metrics),):
        with (destination / (name + ".csv")).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    run.write_json(destination / "primary_comparisons.json", comparisons)
    run.write_json(destination / "complete.json", {"status": "complete", "completed_final_fits": len(all_metrics),
        "primary_comparisons": len(comparisons), "global_seal_fingerprint": seal["fingerprint"],
        "artifacts": {p.name: run.digest(p) for p in destination.iterdir() if p.name != "complete.json"},
        "inference_scope": "Optimization variability conditional on fixed partitions, not independent datasets",
        "combined_family_resolution": "At n=12, minimum exact two-sided p=2/4096; multiplying by 30 gives .0146484375. This is test resolution, not a power guarantee."})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=run.DEFAULT_STUDY)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--paths", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--datasets", nargs="+", choices=tuple(run.settings(run.DEFAULT_STUDY)["datasets"]),
                        help="Schedule only these datasets for prepare/extract/fit; the full grid is unchanged")
    parser.add_argument("stage", choices=("prepare", "extract", "fit", "seal", "confirm", "analyze", "all"))
    args = parser.parse_args()
    if args.datasets is not None and args.stage not in ("prepare", "extract", "fit"):
        parser.error("--datasets is allowed only for prepare, extract and fit; global stages require all datasets")
    selected_datasets(args)
    from . import manage
    manage.require_owned(args.work, args.study)
    if args.stage in ("prepare", "all"):
        run.require(args.paths is not None, "Private local dataset paths file is required")
        prepare(args)
    if args.stage in ("extract", "all"): extract(args)
    if args.stage in ("fit", "all"): fit(args)
    if args.stage == "seal": global_seal(args.work, args.study, args.device, write=True)
    if args.stage in ("confirm", "all"): confirm(args)
    if args.stage in ("analyze", "all"): analyze(args)


if __name__ == "__main__": main()
