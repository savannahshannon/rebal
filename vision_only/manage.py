"""Bind the full-study controller to owned storage and verified export.

Initialize before acquisition or preparation. Finalization does not delete any
file: it verifies the full study, exports retained evidence, and returns a dry-run
cleanup plan. Borrowed datasets outside the owned work directory are untouched.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from . import data, lifecycle, run

SNAPSHOT = "code/snapshot.json"


def _sources(study):
    directory = Path(__file__).parent
    sources = {"code/vision_only/" + item.name: item for item in directory.glob("*.py")}
    sources["code/vision_only/study.json"] = Path(study)
    for name in ("requirements.txt", ".gitignore"):
        if (directory / name).is_file():
            sources["code/vision_only/" + name] = directory / name
    run.require(all(not item.is_symlink() and item.is_file() for item in sources.values()),
                "Code snapshot sources must be ordinary files")
    return dict(sorted(sources.items()))


def initialize(work, study=run.DEFAULT_STUDY):
    """Claim only a new/empty work directory and snapshot public source/config."""
    from .control import expected_run_ids
    work = lifecycle._root(work)
    if (work / lifecycle.MARKER).exists():
        return require_owned(work, study)
    spec = run.settings(study)
    lifecycle.initialize_run_root(work, spec["study_id"], expected_run_ids(study), run.digest(study))
    files = {}
    for relative, source in _sources(study).items():
        target = work / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(source.read_bytes())
        files[relative] = lifecycle._file_record(target)
    snapshot = {"schema_version": 1, "study_sha256": run.digest(study),
                "scientific_source": run.code_identity(), "files": files}
    run.write_json(work / SNAPSHOT, snapshot)
    lifecycle.register_artifacts(work, [(name, "code") for name in files] + [(SNAPSHOT, "code")])
    return require_owned(work, study)


def require_owned(work, study=run.DEFAULT_STUDY):
    """Validate protocol and original code snapshot before any controller stage."""
    from .control import expected_run_ids
    work, state = lifecycle._state(work)
    spec = run.settings(study)
    run.require(state["study_id"] == spec["study_id"] and state["protocol_sha256"] == run.digest(study)
                and state["expected_run_ids"] == sorted(expected_run_ids(study)),
                "Owned work directory belongs to a different protocol/run grid")
    marker_entry = state["artifacts"].get(SNAPSHOT, {})
    run.require(marker_entry.get("kind") == "code" and lifecycle._file_record(work / SNAPSHOT) ==
                {key: marker_entry.get(key) for key in ("sha256", "bytes")}, "Code snapshot manifest changed")
    snapshot = run.read_json(work / SNAPSHOT)
    run.require(snapshot["schema_version"] == 1 and snapshot["study_sha256"] == run.digest(study)
                and snapshot["scientific_source"] == run.code_identity(),
                "Current source or protocol changed after initialization")
    sources = _sources(study)
    run.require(set(snapshot["files"]) == set(sources), "Code snapshot file inventory changed")
    for relative, source in sources.items():
        copied = lifecycle._plain_path(work / relative)
        expected = snapshot["files"][relative]
        entry = state["artifacts"].get(relative, {})
        run.require(entry.get("kind") == "code" and lifecycle._file_record(copied) == expected
                    and lifecycle._file_record(source) == expected
                    and {key: entry.get(key) for key in ("sha256", "bytes")} == expected,
                    "Code/config snapshot bytes changed")
    return {"ownership_id": state["ownership_id"], "study_sha256": state["protocol_sha256"],
            "expected_final_fits": len(state["expected_run_ids"]), "status": "owned_protocol_verified"}


def _portable_json(path, work):
    """Refuse private operational paths or credential-shaped metadata in exports."""
    value = run.read_json(path)
    forbidden = {"source_root", "resolved_root", "password", "access_token", "refresh_token",
                 "credential", "credentials", "username", "hostname", "ip_address"}
    def check(item):
        if isinstance(item, dict):
            run.require(not forbidden.intersection(str(key).lower() for key in item),
                        "Private metadata cannot be exported: " + path.name)
            for entry in item.values(): check(entry)
        elif isinstance(item, list):
            for entry in item: check(entry)
        elif isinstance(item, str):
            run.require(str(work) not in item and str(Path.home()) not in item
                        and not item.startswith(("/Users/", "/home/", "C:\\Users\\", "C:/Users/")),
                        "Host filesystem path cannot be exported: " + path.name)
    check(value)


def _metric_record(value, classes):
    for name in ("raw", "adjusted"):
        metrics = value[name]
        for key in ("macro_f1", "top1", "top5", "macro_precision", "macro_recall",
                    "head_f1", "middle_f1", "tail_f1"):
            score = metrics[key]
            run.require(isinstance(score, (int, float)) and math.isfinite(score) and 0 <= score <= 1,
                        "Invalid completed classification metric")
        for key in ("per_class_precision", "per_class_recall", "per_class_f1"):
            scores = metrics[key]
            run.require(len(scores) == classes and all(isinstance(score, (int, float)) and
                        math.isfinite(score) and 0 <= score <= 1 for score in scores),
                        "Invalid completed per-class metric coverage")
        support = metrics["support"]
        run.require(len(support) == classes and all(type(n) is int and n >= 0 for n in support)
                    and sum(support) > 0, "Invalid completed confirmation support")


def _verified_completion_artifacts(work, study, seal):
    """Inspect every declared completion; return only physically bound evidence."""
    from .control import contexts, expected_run_ids
    spec = run.settings(study)
    completed = {}
    for dataset, regime, encoder, args in contexts(work, study, "cpu"):
        selected = run.read_json(args.out / "selection.json")
        for arm in spec["arms"]:
            for seed in spec["training"]["final_seeds"]:
                cell = args.out / "final" / arm / str(seed)
                trained = run.read_json(cell / "trained.json")
                confirmed = run.read_json(cell / "confirmed.json")
                run.require(confirmed["status"] == "complete" and confirmed["arm"] == arm
                            and confirmed["seed"] == seed and trained["status"] == "trained"
                            and confirmed["trained_fingerprint"] == trained["fingerprint"]
                            and confirmed["selection_fingerprint"] == selected["fingerprint"],
                            "An expected trained/confirmed run is missing or mismatched")
                for filename, field in (("checkpoint.pt", "checkpoint_sha256"),
                                        ("predictions.npz", "predictions_sha256"),
                                        ("calibration_predictions.npz", "calibration_predictions_sha256")):
                    record = trained if filename == "checkpoint.pt" else confirmed
                    run.require(run.digest(cell / filename) == record[field], "Completed evidence file changed")
                _metric_record(confirmed, spec["datasets"][dataset]["classes"])
                identifier = "/".join(map(str, (dataset, regime, encoder, arm, seed)))
                completed[identifier] = [(cell / filename).relative_to(work).as_posix() for filename in
                    ("checkpoint.pt", "trained.json", "confirmed.json", "predictions.npz", "calibration_predictions.npz")]
    run.require(set(completed) == set(expected_run_ids(study))
                and len(completed) == seal["completed_final_fits"], "Completed export run grid differs from sealed grid")
    return completed


def _classification(relative, state):
    parts = Path(relative).parts
    first = parts[0]
    if first == "code":
        run.require(state["artifacts"].get(relative, {}).get("kind") == "code",
                    "Unknown file in preserved code snapshot")
        return "code"
    if first == "data": return "dataset"
    if first in {"cache", "manifests", "logs"}: return "cache"
    if first == "runs":
        if Path(relative).suffix in {".pt", ".pth", ".safetensors"}: return "checkpoint"
        run.require(Path(relative).suffix in {".json", ".npz", ".csv"}, "Unknown run artifact requires explicit review")
        return "result"
    if first in {"public_manifests", "partitions", "aggregate", "figures", "report"}:
        run.require(Path(relative).suffix in {".json", ".csv", ".pdf", ".png", ".svg", ".md", ".tex"},
                    "Unknown report artifact requires explicit review")
        return "result"
    if relative in {"global_confirmation_seal.json", "execution_status.json", "preflight.json"}:
        return "result"
    raise lifecycle.LifecycleError("Unknown file requires explicit classification: " + relative)


def _owned_files(work, state):
    import os
    entries = []
    for directory, dirs, files in os.walk(work, followlinks=False):
        for name in dirs:
            run.require(not (Path(directory) / name).is_symlink(), "Symlink directory blocks finalization")
        for name in files:
            path = Path(directory) / name
            relative = path.relative_to(work).as_posix()
            if relative == lifecycle.MARKER: continue
            kind = _classification(relative, state)
            lifecycle._plain_path(path)
            if kind == "result" and path.suffix == ".json":
                _portable_json(path, work)
            entries.append((relative, kind))
    return entries


def finalize(work, study, archive):
    """Verify full results, export outside work, then return a deletion preview."""
    from .control import expected_run_ids, global_seal
    require_owned(work, study)
    work, state = lifecycle._state(work)
    archive = lifecycle._outside(work, archive)
    run.require(not archive.exists(), "Retained archive destination already exists")
    seal = global_seal(work, study, "cpu")
    complete_path = work / "aggregate" / "complete.json"
    aggregate = run.read_json(complete_path)
    required = {"per_seed.csv", "per_class.csv", "primary_comparisons.json"}
    run.require(aggregate["status"] == "complete" and aggregate["completed_final_fits"] == len(expected_run_ids(study))
                and aggregate["primary_comparisons"] == 30 and aggregate["global_seal_fingerprint"] == seal["fingerprint"]
                and required <= set(aggregate["artifacts"]), "Aggregate is incomplete or belongs to another sealed study")
    aggregate_paths = ["aggregate/complete.json"]
    for name, digest in aggregate["artifacts"].items():
        relative = lifecycle._relative("aggregate/" + name)
        run.require(Path(name).name == name and run.digest(work / relative) == digest,
                    "Aggregate evidence changed or is outside aggregate directory")
        aggregate_paths.append(relative)
    completed = _verified_completion_artifacts(work, study, seal)
    # Operational manifests need absolute local paths to resume. Retain public
    # copies with the same scientific fingerprint, without exporting those paths.
    spec = run.settings(study)
    for dataset in spec["datasets"]:
        manifest = data.load_manifest(work / "manifests" / (dataset + ".json"))
        run.require(manifest.dataset_id == dataset and manifest.scope == "full", "Full-source manifest required")
        data.save_manifest(manifest, work / "public_manifests" / (dataset + ".json"), public=True)
    lifecycle.register_artifacts(work, _owned_files(work, state))
    work, state = lifecycle._state(work)
    # One validated transaction, rather than rewriting a million-image ownership
    # inventory for each of the 720 completion records.
    records = {}
    for identifier, names in completed.items():
        evidence = {}
        for name in names:
            entry = state["artifacts"].get(name, {})
            run.require(entry.get("kind") in {"result", "checkpoint"}, "Run evidence lacks owned result classification")
            evidence[name] = {key: entry[key] for key in ("sha256", "bytes")}
        records[identifier] = {"status": "complete", "artifacts": evidence}
    state["runs"], state["aggregates"] = records, aggregate_paths
    lifecycle._complete(state)
    lifecycle._atomic_json(work / lifecycle.MARKER, state)
    exported = lifecycle.export_results(work, archive)
    plan = lifecycle.plan_cleanup(work, exported["receipt"])
    return {"status": "verified_export_cleanup_preview_only", "deleted": False,
            "completed_final_fits": len(records), **exported, "cleanup_plan": plan}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=run.DEFAULT_STUDY)
    parser.add_argument("--work", type=Path, required=True)
    sub = parser.add_subparsers(dest="stage", required=True)
    sub.add_parser("init", aliases=["initialize"])
    sub.add_parser("verify")
    export = sub.add_parser("export", aliases=["finalize"])
    export.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.stage in {"init", "initialize"}: result = initialize(args.work, args.study)
    elif args.stage == "verify": result = require_owned(args.work, args.study)
    else: result = finalize(args.work, args.study, args.archive)
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__": main()
