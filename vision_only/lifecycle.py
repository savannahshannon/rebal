"""Read-only machine preflight and guarded lifecycle for a new vision study.

Nothing is deleted by default. A run root must be newly created or empty when
claimed; files must then be explicitly registered. Cleanup requires an exact
completed run grid, aggregate artifacts, and a physically verified result export
outside that root. Existing datasets elsewhere are never cleanup candidates.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import stat
import subprocess
import sys
import uuid
import zipfile

MARKER = ".vision-only-owned.json"
EXPORT_MANIFEST = "_vision_only/export_manifest.json"
KINDS = {"code", "dataset", "result", "cache", "checkpoint"}
STATUSES = {"pending", "running", "failed", "complete"}


class LifecycleError(ValueError):
    """An ownership, completeness, or export check failed."""


def _canonical(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _plain_path(value):
    if ".." in Path(value).parts:
        raise LifecycleError("Parent traversal is not allowed in lifecycle paths")
    path = Path(value).absolute()
    for parent in (path, *path.parents):
        if parent.is_symlink():
            raise LifecycleError("Symbolic links are not allowed in lifecycle paths")
    return path


def _root(value):
    path = _plain_path(value)
    home = Path.home().absolute()
    if path == Path(path.anchor) or path == home or path in home.parents:
        raise LifecycleError("A filesystem root, home, or home ancestor cannot be a run root")
    if len(path.parts) < 3:
        raise LifecycleError("Run root must be a dedicated study subdirectory")
    return path


def _relative(value):
    text = str(value)
    path = Path(text)
    if (not text or path.is_absolute() or "\\" in text or ":" in text
            or any(part in {"", ".", ".."} for part in text.split("/"))
            or text == MARKER or path.parts[0] == "_vision_only"):
        raise LifecycleError("Artifact paths must be normal, relative paths inside the owned root")
    return path.as_posix()


def _file_record(path):
    if path.is_symlink():
        raise LifecycleError("Symbolic links cannot be owned artifacts")
    try:
        candidate = path.lstat()
        if not stat.S_ISREG(candidate.st_mode) or candidate.st_nlink != 1:
            raise LifecycleError("Artifacts must be ordinary files with one link")
        # A replaced FIFO must fail rather than hang a supposedly read-only scan.
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
    except OSError as exc:
        raise LifecycleError("Artifact cannot be opened safely") from exc
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise LifecycleError("Artifacts must be ordinary files with one link")
        digest = hashlib.sha256()
        while chunk := os.read(fd, 1024 * 1024):
            digest.update(chunk)
        after = os.fstat(fd)
        if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
            raise LifecycleError("Artifact changed while hashing")
        return {"sha256": digest.hexdigest(), "bytes": after.st_size}
    finally:
        os.close(fd)


def _atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    try:
        with temporary.open("xb") as stream:
            stream.write(_canonical(value))
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def initialize_run_root(root, study_id, expected_run_ids, protocol_sha256=None):
    """Claim a new/empty dedicated root; never adopt an existing dataset tree."""
    root = _root(root)
    runs = list(expected_run_ids)
    if not study_id or not runs or any(not isinstance(run, str) or not run for run in runs):
        raise LifecycleError("Study ID and nonempty expected run IDs are required")
    if len(set(runs)) != len(runs):
        raise LifecycleError("Expected run IDs must be unique")
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise LifecycleError("Run root must be absent or empty; existing files cannot be adopted")
    if protocol_sha256 is not None and (len(protocol_sha256) != 64 or any(c not in "0123456789abcdef" for c in protocol_sha256)):
        raise LifecycleError("Protocol fingerprint must be a SHA-256 digest")
    root.mkdir(parents=True, exist_ok=True)
    state = {"schema": 1, "study_id": study_id, "ownership_id": uuid.uuid4().hex,
             "protocol_sha256": protocol_sha256, "expected_run_ids": sorted(runs),
             "runs": {}, "artifacts": {}, "aggregates": []}
    _atomic_json(root / MARKER, state)
    return state


def _state(root):
    root = _root(root)
    marker = root / MARKER
    if not root.is_dir() or not marker.is_file() or marker.is_symlink():
        raise LifecycleError("Run root lacks its ownership marker")
    _file_record(marker)
    try:
        state = json.loads(marker.read_text())
        if state["schema"] != 1 or not state["ownership_id"] or not state["study_id"]:
            raise LifecycleError("Invalid ownership marker identity")
        if not isinstance(state["artifacts"], dict) or not isinstance(state["runs"], dict):
            raise LifecycleError("Invalid ownership marker collections")
        ids = state["expected_run_ids"]
        if (not isinstance(ids, list) or not ids or len(set(ids)) != len(ids)
                or not all(isinstance(item, str) and item for item in ids)):
            raise LifecycleError("Invalid expected run ID grid")
        for name, entry in state["artifacts"].items():
            _relative(name)
            if entry["kind"] not in KINDS:
                raise LifecycleError("Invalid artifact ownership kind")
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise LifecycleError("Invalid ownership marker") from exc
    return root, state


def inventory(root):
    """Hash the complete tree and reject links, devices, and unowned files."""
    root, state = _state(root)
    found = {}
    for directory, dirs, files in os.walk(root, followlinks=False):
        for name in dirs:
            path = Path(directory) / name
            if path.is_symlink():
                raise LifecycleError("Symbolic link directory in run root")
        for name in files:
            path = Path(directory) / name
            relative = path.relative_to(root).as_posix()
            if relative == MARKER:
                continue
            if relative not in state["artifacts"]:
                raise LifecycleError("Unowned file blocks export and cleanup: " + relative)
            entry = state["artifacts"][relative]
            record = _file_record(path)
            if record != {key: entry[key] for key in ("sha256", "bytes")}:
                raise LifecycleError("Registered artifact changed: " + relative)
            found[relative] = {**record, "kind": entry["kind"]}
    if set(found) != set(state["artifacts"]):
        raise LifecycleError("Registered artifacts are missing")
    return dict(sorted(found.items()))


def register_artifact(root, relative_path, kind):
    """Register a file created inside the new root, or refresh its final hash."""
    relative = _relative(relative_path)
    return register_artifacts(root, [(relative, kind)])[relative]


def register_artifacts(root, artifacts):
    """Hash an iterable of ``(relative_path, kind)`` pairs with one JSON commit.

    Use this batch API for full extracted image trees. Per-file ownership-marker
    rewrites would be quadratic in a million-image inventory. A failed batch
    leaves the prior marker untouched, so a partial traversal cannot adopt files.
    """
    root, state = _state(root)
    updates = {}
    for relative_path, kind in artifacts:
        relative = _relative(relative_path)
        if relative in updates:
            raise LifecycleError("Duplicate artifact path in registration batch")
        if kind not in KINDS:
            raise LifecycleError("Unknown artifact kind")
        path = _plain_path(root / relative)
        if not path.is_relative_to(root):
            raise LifecycleError("Artifact is outside the owned root")
        if relative in state["artifacts"] and state["artifacts"][relative]["kind"] != kind:
            raise LifecycleError("An artifact's ownership kind cannot be changed")
        updates[relative] = {**_file_record(path), "kind": kind}
    if not updates:
        raise LifecycleError("An artifact registration batch cannot be empty")
    state["artifacts"].update(updates)
    # Updating any result invalidates completion evidence that referenced it.
    for run in state["runs"].values():
        if any(name in updates for name in run.get("artifacts", {})):
            run["status"] = "pending"
    if any(name in updates for name in state["aggregates"]):
        state["aggregates"] = []
    _atomic_json(root / MARKER, state)
    return updates


def register_tree(root, relative_directory, kind):
    """Register all ordinary files in one owned child tree in a single batch."""
    root, _ = _state(root)
    relative = _relative(relative_directory)
    directory = _plain_path(root / relative)
    if not directory.is_dir():
        raise LifecycleError("Artifact tree must be an existing owned directory")

    def entries():
        for current, directories, files in os.walk(directory, followlinks=False):
            for name in directories:
                if (Path(current) / name).is_symlink():
                    raise LifecycleError("Symbolic link directory in artifact tree")
            for name in files:
                yield (Path(current) / name).relative_to(root).as_posix(), kind

    return register_artifacts(root, entries())


def record_run_completion(root, run_id, status, artifact_paths=()):
    root, state = _state(root)
    if run_id not in state["expected_run_ids"] or status not in STATUSES:
        raise LifecycleError("Unexpected run ID or invalid run status")
    artifacts = {}
    for value in artifact_paths:
        relative = _relative(value)
        entry = state["artifacts"].get(relative)
        if entry is None or entry["kind"] not in {"result", "checkpoint"}:
            raise LifecycleError("Completion evidence must be a registered result or checkpoint")
        record = _file_record(root / relative)
        if record != {key: entry[key] for key in ("sha256", "bytes")}:
            raise LifecycleError("Completion artifact changed")
        artifacts[relative] = record
    if status == "complete" and not artifacts:
        raise LifecycleError("A completed run requires hashed result evidence")
    state["runs"][run_id] = {"status": status, "artifacts": artifacts}
    state["aggregates"] = []
    _atomic_json(root / MARKER, state)


def _complete(state):
    expected = set(state["expected_run_ids"])
    if set(state["runs"]) != expected:
        raise LifecycleError("Not all expected protocol runs have completion records")
    for run_id in sorted(expected):
        run = state["runs"][run_id]
        if run.get("status") != "complete" or not run.get("artifacts"):
            raise LifecycleError("Incomplete, failed, or pending run blocks cleanup: " + run_id)
        for name, record in run["artifacts"].items():
            entry = state["artifacts"].get(name, {})
            if entry.get("kind") not in {"result", "checkpoint"} or record != {key: entry.get(key) for key in ("sha256", "bytes")}:
                raise LifecycleError("Run evidence no longer matches registered inventory")


def record_aggregates(root, artifact_paths):
    root, state = _state(root)
    inventory(root)
    _complete(state)
    paths = sorted(set(_relative(item) for item in artifact_paths))
    if not paths or any(state["artifacts"].get(name, {}).get("kind") != "result" for name in paths):
        raise LifecycleError("At least one registered aggregate result is required")
    state["aggregates"] = paths
    _atomic_json(root / MARKER, state)


def _outside(root, value):
    path = _plain_path(value)
    if path == root or path.is_relative_to(root):
        raise LifecycleError("Export and receipt must be outside the cleanup root")
    return path


def export_results(root, archive_path, receipt_path=None):
    """Export results/checkpoints, then verify archive contents before receipting.

    Datasets and caches are deliberately excluded. Copy the verified archive to
    the desired retained destination before cleanup; a receipt alone is not proof.
    """
    root, state = _state(root)
    files = inventory(root)
    _complete(state)
    if not state["aggregates"] or any(files.get(name, {}).get("kind") != "result" for name in state["aggregates"]):
        raise LifecycleError("Completed aggregate results are required before export")
    archive = _outside(root, archive_path)
    receipt = _outside(root, receipt_path or archive.with_suffix(archive.suffix + ".receipt.json"))
    if archive == receipt or archive.exists() or receipt.exists():
        raise LifecycleError("Export destinations must be distinct, new files")
    archive.parent.mkdir(parents=True, exist_ok=True)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"state": state, "inventory": files}
    manifest_bytes = _canonical(manifest)
    retained = {name: value for name, value in files.items() if value["kind"] in {"result", "checkpoint"}}
    try:
        with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as package:
            package.writestr(EXPORT_MANIFEST, manifest_bytes)
            for name in retained:
                package.write(root / name, name)
        # Verify bytes in the physical archive, rather than trusting creation status.
        _verify_archive(archive, retained, manifest_bytes)
        if inventory(root) != files or _state(root)[1] != state:
            raise LifecycleError("Run root changed during export")
        data = {"schema": 1, "ownership_id": state["ownership_id"],
                "archive_path": str(archive), "archive": _file_record(archive),
                "manifest_sha256": _sha_bytes(manifest_bytes), "result_inventory": retained}
        _atomic_json(receipt, data)
    except Exception:
        # A partial archive is retained for diagnosis, but never obtains a receipt.
        raise
    return {"archive": str(archive), "receipt": str(receipt), "exported_files": len(retained)}


def _verify_archive(archive, retained, manifest_bytes):
    try:
        with zipfile.ZipFile(archive) as package:
            names = package.namelist()
            if len(set(names)) != len(names) or set(names) != set(retained) | {EXPORT_MANIFEST}:
                raise LifecycleError("Export archive has missing, duplicate, or unexpected files")
            if package.read(EXPORT_MANIFEST) != manifest_bytes:
                raise LifecycleError("Export manifest does not match current run inventory")
            for name, entry in retained.items():
                digest = hashlib.sha256()
                total = 0
                with package.open(name) as stream:
                    while chunk := stream.read(1024 * 1024):
                        digest.update(chunk)
                        total += len(chunk)
                if digest.hexdigest() != entry["sha256"] or total != entry["bytes"]:
                    raise LifecycleError("Exported result hash mismatch: " + name)
    except (OSError, zipfile.BadZipFile, KeyError) as exc:
        raise LifecycleError("Export archive is unavailable or invalid") from exc


def plan_cleanup(root, receipt_path):
    """Return a read-only deletion plan, refusing any unverified state."""
    root, state = _state(root)
    files = inventory(root)
    _complete(state)
    if not state["aggregates"]:
        raise LifecycleError("Aggregate completion is missing")
    receipt = _outside(root, receipt_path)
    _file_record(receipt)
    try:
        proof = json.loads(receipt.read_text())
        archive = _outside(root, proof["archive_path"])
        if proof["schema"] != 1 or proof["ownership_id"] != state["ownership_id"]:
            raise LifecycleError("Export receipt belongs to another study root")
        if _file_record(archive) != proof["archive"]:
            raise LifecycleError("Export archive hash changed")
        manifest_bytes = _canonical({"state": state, "inventory": files})
        if _sha_bytes(manifest_bytes) != proof["manifest_sha256"]:
            raise LifecycleError("Export receipt does not match current inventory")
        retained = {name: entry for name, entry in files.items() if entry["kind"] in {"result", "checkpoint"}}
        if retained != proof["result_inventory"]:
            raise LifecycleError("Export receipt has incomplete result inventory")
        _verify_archive(archive, retained, manifest_bytes)
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise LifecycleError("Export receipt or archive is missing or invalid") from exc
    remove = [name for name, entry in files.items() if entry["kind"] != "code"]
    return {"dry_run": True, "ownership_id": state["ownership_id"],
            "delete": remove + [MARKER],
            "preserve_code": [name for name, entry in files.items() if entry["kind"] == "code"],
            "delete_bytes": sum(files[name]["bytes"] for name in remove),
            "verified_archive_sha256": proof["archive"]["sha256"]}


def cleanup(root, receipt_path, *, execute=False):
    """Execute only an explicit, freshly verified plan; preserve registered code."""
    plan = plan_cleanup(root, receipt_path)
    if not execute:
        return plan
    root, state = _state(root)
    report_path = _outside(root, Path(receipt_path).with_suffix(".cleanup.json"))
    if report_path.exists():
        raise LifecycleError("Cleanup report already exists")
    # Open directories without following symlinks. An unlink is always relative to
    # the owned root descriptor, never an arbitrary absolute deletion target.
    if not hasattr(os, "O_NOFOLLOW") or os.unlink not in os.supports_dir_fd:
        raise LifecycleError("Safe descriptor-relative cleanup is unsupported on this platform")
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for name in plan["delete"]:
            parts = Path(name).parts
            parent_fd = os.dup(root_fd)
            try:
                for part in parts[:-1]:
                    child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd)
                    os.close(parent_fd)
                    parent_fd = child
                current = os.stat(parts[-1], dir_fd=parent_fd, follow_symlinks=False)
                if not stat.S_ISREG(current.st_mode) or current.st_nlink != 1:
                    raise LifecycleError("Cleanup encountered a changed or linked file")
                # Recheck the artifact immediately before deletion. The marker is
                # last so a partially interrupted cleanup retains ownership proof.
                expected = state["artifacts"].get(name)
                if expected and _file_record(root / name) != {key: expected[key] for key in ("sha256", "bytes")}:
                    raise LifecycleError("Cleanup candidate changed after planning")
                os.unlink(parts[-1], dir_fd=parent_fd)
            finally:
                os.close(parent_fd)
    finally:
        os.close(root_fd)
    for directory, dirs, files in os.walk(root, topdown=False, followlinks=False):
        for name in dirs:
            path = Path(directory) / name
            if not path.is_symlink() and not any(path.iterdir()):
                path.rmdir()
    report = {**plan, "dry_run": False, "executed": True}
    _atomic_json(report_path, report)
    return report


def preflight(root):
    """Read-only, portable capacity report without usernames, hostnames, or IPs."""
    root = _root(root)
    ancestor = root
    while not ancestor.exists():
        ancestor = ancestor.parent
    usage = shutil.disk_usage(ancestor)
    report = {"schema": 1, "visibility": "private", "platform": platform.system(),
              "machine": platform.machine(), "python": platform.python_version(),
              "cpu_count": os.cpu_count(), "disk": {"total_bytes": usage.total, "free_bytes": usage.free},
              "tools": {name: shutil.which(name) is not None for name in ("git", "nvidia-smi", "python3")},
              "python_modules": {name: importlib.util.find_spec(name) is not None for name in ("torch", "torchvision", "numpy", "timm")},
              "gpu": [], "gpu_probe_status": "not_available"}
    if report["tools"]["nvidia-smi"]:
        try:
            result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
                                    capture_output=True, text=True, timeout=15, check=True)
            for line in result.stdout.splitlines():
                name, memory, driver = (item.strip() for item in line.rsplit(",", 2))
                report["gpu"].append({"name": name, "memory_mib": int(memory), "driver": driver})
            report["gpu_probe_status"] = "ok"
        except (OSError, subprocess.SubprocessError, ValueError):
            report["gpu_probe_status"] = "failed"
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "export", "cleanup"):
        sub = subparsers.add_parser(name)
        sub.add_argument("--root", type=Path, required=True)
        if name == "export":
            sub.add_argument("--archive", type=Path, required=True)
            sub.add_argument("--receipt", type=Path)
        if name == "cleanup":
            sub.add_argument("--receipt", type=Path, required=True)
            sub.add_argument("--execute", action="store_true", help="Actually remove verified owned non-code files; default is a dry run")
    args = parser.parse_args(argv)
    try:
        if args.command == "preflight":
            report = preflight(args.root)
        elif args.command == "export":
            report = export_results(args.root, args.archive, args.receipt)
        else:
            report = cleanup(args.root, args.receipt, execute=args.execute)
        print(json.dumps(report, sort_keys=True, indent=2))
    except (LifecycleError, OSError) as exc:
        print("Lifecycle check failed: " + str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
