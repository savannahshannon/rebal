#!/usr/bin/env python3
"""Acquire and prepare traceable Covertype/DBpedia non-image experiments.

Acquisition is an explicit separate command. The module never downloads on
import or during prepare; invoke acquire only after the requested Git upload.

  python scripts/prepare_nonimage.py catalog
  python scripts/prepare_nonimage.py acquire --dataset covertype --output data/sources/covertype
  python scripts/prepare_nonimage.py prepare --dataset covertype --source data/sources/covertype --output data/covertype/natural
  python scripts/prepare_nonimage.py acquire --dataset dbpedia --output data/sources/dbpedia
  python scripts/prepare_nonimage.py prepare --dataset dbpedia --source data/sources/dbpedia --output data/dbpedia/rho100 --rho 100

Prepared arrays retain original rows; split indices alone select fitted and
evaluated rows. No scaler, vocabulary, neighbor model or outlier model is fit.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import os
import re
import shutil
import tarfile
import unicodedata
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import numpy as np

VERSION = 1
SPLIT_SEED = 1729
RANK_SEED = 2026
RANK_SEMANTICS = "ordered_class_ids_head_to_tail"
DBPEDIA_MD5 = "dca7b1ae12b1091090db52aa7ec5ca64"
DATASETS = {
    "covertype": {
        "citation": "Blackard, J. (1998). Covertype. DOI:10.24432/C50K5N",
        "page": "https://archive.ics.uci.edu/dataset/31/covertype",
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz",
        "license": "CC BY 4.0 (current UCI dataset page); retain original attribution notice",
        "rows": 581012, "features": 54, "n_classes": 7,
        "classes": ["Spruce/Fir", "Lodgepole Pine", "Ponderosa Pine", "Cottonwood/Willow", "Aspen", "Douglas-fir", "Krummholz"],
        "class_counts": [211840, 283301, 35754, 2747, 9493, 17367, 20510],
    },
    "dbpedia": {
        "citation": "Zhang, X., Zhao, J., LeCun, Y. (2015). Character-level Convolutional Networks for Text Classification. NeurIPS 28.",
        "page": "https://github.com/zhangxiangxiao/Crepe",
        "url": "https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbQ2Vic1kxMmZZQ1k",
        "archive_md5_source": "https://docs.pytorch.org/text/stable/_modules/torchtext/datasets/dbpedia.html",
        "hf_repo": "fancyzhx/dbpedia_14",
        "hf_page": "https://huggingface.co/datasets/fancyzhx/dbpedia_14",
        "license": "CC BY-SA 3.0 and GNU Free Documentation License (DBpedia-14 dataset card); preserve attribution/license notices",
        "train_rows": 560000, "test_rows": 70000, "n_classes": 14,
        "classes": ["Company", "EducationalInstitution", "Artist", "Athlete", "OfficeHolder", "MeanOfTransportation", "Building", "NaturalPlace", "Village", "Animal", "Plant", "Album", "Film", "WrittenWork"],
    },
}


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")


def _safe_relative(name):
    p = PurePosixPath(name)
    if not name or p.is_absolute() or ".." in p.parts or "\\" in name:
        raise ValueError(f"Unsafe relative path: {name!r}")
    return p


def _stage(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Output already exists; use a new output directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.with_name(output.name + f".partial-{os.getpid()}")
    stage.mkdir(exist_ok=False)
    return stage


def _download(url, destination, limit):
    """Bounded HTTPS download to a staging directory; never execute contents."""
    if not url.startswith("https://"):
        raise ValueError("Only HTTPS sources are supported")
    req = urllib.request.Request(url, headers={"User-Agent": "REBAL-nonimage-research/1"})
    size = 0
    with urllib.request.urlopen(req, timeout=90) as response, open(destination, "wb") as out:
        if not response.url.startswith("https://"):
            raise ValueError("Refusing a non-HTTPS redirect")
        if "text/html" in response.headers.get("Content-Type", "").lower():
            raise ValueError("Dataset URL returned HTML, possibly a download confirmation/quota page")
        while True:
            block = response.read(1024 * 1024)
            if not block:
                break
            size += len(block)
            if size > limit:
                raise ValueError(f"Source exceeds configured download limit: {url}")
            out.write(block)
        # CDN redirects can contain temporary signed query credentials. The
        # pinned public request URL and content digest provide provenance.
        return {"url": url, "bytes": size, "sha256": sha(destination)}


def acquire(dataset, output, distribution="huggingface"):
    """Explicit network operation. The caller is responsible for task sequencing."""
    if dataset not in DATASETS:
        raise ValueError("Unsupported dataset")
    stage = _stage(output)
    try:
        spec = DATASETS[dataset]
        record = {"version": VERSION, "dataset": dataset, "created_utc": now(),
                  "citation": spec["citation"], "license": spec["license"], "files": {}}
        if dataset == "covertype":
            name = "covtype.data.gz"
            record["distribution"] = "uci-original-gzip"
            record["files"][name] = _download(spec["url"], stage / name, 32 * 1024**2)
        elif distribution == "original":
            name = "dbpedia_csv.tar.gz"
            record["distribution"] = "author-original-csv-tar"
            record["files"][name] = _download(spec["url"], stage / name, 1024**3)
            found = sha(stage / name, "md5")
            if found != DBPEDIA_MD5:
                raise ValueError("Original DBpedia archive fails official Torchtext MD5")
            record["files"][name]["verified_md5"] = found
            record["files"][name]["md5_source"] = spec["archive_md5_source"]
        else:
            repo = spec["hf_repo"]
            metadata_url = f"https://huggingface.co/api/datasets/{repo}"
            with urllib.request.urlopen(metadata_url, timeout=60) as r:
                info = json.loads(r.read(4 * 1024 * 1024))
            revision = info.get("sha", "")
            if not re.fullmatch(r"[0-9a-f]{40}", revision):
                raise ValueError("Cannot pin Hugging Face dataset revision")
            names = sorted(s["rfilename"] for s in info.get("siblings", [])
                           if re.search(r"(?:^|/)(?:train|test)-\d+-of-\d+\.parquet$", s["rfilename"]))
            if not names or not all(any(PurePosixPath(n).name.startswith(split + "-") for n in names)
                                    for split in ("train", "test")):
                raise ValueError("Pinned dataset does not expose both expected parquet splits")
            record.update(distribution="huggingface-parquet", hf_repo=repo, revision=revision,
                          metadata_url=metadata_url)
            for name in names:
                _safe_relative(name)
                destination = stage / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{name}?download=true"
                record["files"][name] = _download(url, destination, 1024**3)
        write_json(stage / "source.json", record)
        stage.rename(output)
        return record
    except BaseException:
        shutil.rmtree(stage)
        raise


def _source_record(source, dataset):
    source = Path(source)
    if source.is_file():
        info = {"version": VERSION, "dataset": dataset, "distribution": "provided-local-file",
                "files": {source.name: {"sha256": sha(source), "bytes": source.stat().st_size}},
                "license": DATASETS[dataset]["license"], "citation": DATASETS[dataset]["citation"],
                "intended_source_url": DATASETS[dataset]["url"]}
        if dataset == "dbpedia":
            if sha(source, "md5") != DBPEDIA_MD5:
                raise ValueError("Provided original DBpedia archive fails official Torchtext MD5")
            info["files"][source.name]["verified_md5"] = DBPEDIA_MD5
        return info
    info = json.loads((source / "source.json").read_text())
    if info.get("dataset") != dataset or not info.get("files"):
        raise ValueError("Source manifest does not describe the requested dataset")
    for name, record in info["files"].items():
        _safe_relative(name)
        path = source / name
        if not path.is_file() or sha(path) != record.get("sha256"):
            raise ValueError(f"Source integrity check failed: {name}")
    if dataset == "dbpedia" and info.get("distribution") == "author-original-csv-tar":
        if sha(source / "dbpedia_csv.tar.gz", "md5") != DBPEDIA_MD5:
            raise ValueError("Original DBpedia archive fails official checksum")
    return info


def normalized_text(text):
    return " ".join(unicodedata.normalize("NFKC", str(text)).casefold().split())


def text_keys(title, content):
    title, content = normalized_text(title), normalized_text(content)
    joined = json.dumps([title, content], ensure_ascii=False, separators=(",", ":"))
    keys = [b"F" + hashlib.sha256(joined.encode("utf-8")).digest()]
    if content:
        keys.append(b"C" + hashlib.sha256(content.encode("utf-8")).digest())
    return tuple(keys)


def canonical_development(labels, keys, native_test_start=None):
    """Canonicalize connected duplicate groups, honoring an intact native test.

    Test LABELS are never used to decide development exclusions. Development
    conflicts are excluded as entire groups; exact test matches take priority.
    """
    labels = np.asarray(labels)
    n = len(labels) if native_test_start is None else native_test_start
    if len(keys) != len(labels) or n < 0 or n > len(labels):
        raise ValueError("Invalid keys/boundary")
    parent = np.arange(n, dtype=np.int64)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = int(parent[i])
        return i

    owner = {}
    for i in range(n):
        for key in keys[i]:
            j = owner.get(key)
            if j is None:
                owner[key] = i
            else:
                a, b = find(i), find(j)
                if a != b:
                    parent[max(a, b)] = min(a, b)
    test_keys = set(k for row in keys[n:] for k in row)
    groups = {}
    for i in range(n):
        root = find(i)
        if root not in groups:
            groups[root] = {"first": i, "label": int(labels[i]), "count": 0, "conflict": False, "test_match": False}
        g = groups[root]
        g["count"] += 1
        g["conflict"] |= int(labels[i]) != g["label"]
        g["test_match"] |= any(k in test_keys for k in keys[i])
    kept = []
    audit = {"native_development_rows": n, "native_test_rows": len(labels) - n,
             "development_duplicate_groups": sum(g["count"] > 1 for g in groups.values()),
             "excluded_test_overlap_rows": 0, "excluded_conflicting_label_rows": 0,
             "excluded_redundant_development_rows": 0,
             "test_labels_used_for_exclusions": False}
    for g in groups.values():
        if g["test_match"]:
            audit["excluded_test_overlap_rows"] += g["count"]
        elif g["conflict"]:
            audit["excluded_conflicting_label_rows"] += g["count"]
        else:
            kept.append(g["first"])
            audit["excluded_redundant_development_rows"] += g["count"] - 1
    audit["canonical_development_rows"] = len(kept)
    # Full-record duplicates in native test stay in native test and are disclosed.
    full_keys = [row[0] for row in keys[n:]]
    audit["native_test_redundant_full_record_rows"] = len(full_keys) - len(set(full_keys))
    return np.asarray(sorted(kept), dtype=np.int64), audit


def make_splits(labels, canonical, dataset, rho=100., split_seed=SPLIT_SEED,
                rank_seed=RANK_SEED, native_test_start=None):
    labels = np.asarray(labels, dtype=np.int64)
    n_classes = DATASETS[dataset]["n_classes"]
    rng = np.random.default_rng(split_seed)
    by_class = [rng.permutation(canonical[labels[canonical] == c]) for c in range(n_classes)]
    if any(len(ids) < 3 for ids in by_class):
        raise ValueError("Every class needs at least three canonical development rows")
    details = {}
    if dataset == "covertype":
        parts = [[], [], []]
        for ids in by_class:
            n_train, n_cal = int(.70 * len(ids)), int(.15 * len(ids))
            if min(n_train, n_cal, len(ids) - n_train - n_cal) < 1:
                raise ValueError("A split would miss a class")
            for part, selected in zip(parts, (ids[:n_train], ids[n_train:n_train+n_cal], ids[n_train+n_cal:])):
                part.extend(selected)
        train, cal, test = (np.asarray(p, dtype=np.int64) for p in parts)
        counts = np.bincount(labels[train], minlength=n_classes)
        class_rank = np.lexsort((np.arange(n_classes), -counts)).astype(np.int64)
        details["split_protocol"] = "stratified 70/15/15 after exact development deduplication; floor train/cal per class"
    else:
        if not math.isfinite(rho) or rho < 1:
            raise ValueError("rho must be finite and at least one")
        if native_test_start is None:
            raise ValueError("Native test boundary is required")
        if any(len(ids) < 4002 for ids in by_class):
            raise ValueError("DBpedia class lacks enough canonical rows for 4000 calibration candidates")
        class_rank = np.random.default_rng(rank_seed).permutation(n_classes).astype(np.int64)
        # Common feasible maximum avoids silently clipping only individual ranks.
        n_max = min(36000, min(len(ids) - 4000 for ids in by_class))
        train, cal = [], []
        for rank, c in enumerate(class_rank):
            retention = rho ** (-rank / (n_classes - 1))
            n_train, n_cal = max(1, int(n_max * retention)), max(1, int(4000 * retention))
            if n_train < 2:
                raise ValueError("Requested rho leaves fewer than two training rows in a class; latent SMOTE is undefined")
            ids = by_class[int(c)]
            cal.extend(ids[:n_cal])
            train.extend(ids[4000:4000+n_train])
        train, cal = np.asarray(train, dtype=np.int64), np.asarray(cal, dtype=np.int64)
        test = np.arange(native_test_start, len(labels), dtype=np.int64)
        details.update(split_protocol="native test intact; 4000 calibration candidates/class before matched long-tail thinning",
                       requested_rho=float(rho), feasible_train_n_max=int(n_max), calibration_candidate_n_max=4000,
                       candidate_development_counts=[len(ids) for ids in by_class], rank_seed=int(rank_seed),
                       calibration_thinning="same rank-dependent retention fractions as training; at least one/class")
    splits = {"train": train, "calibration": cal, "test": test, "class_rank": class_rank}
    validate_splits(labels, splits, n_classes)
    details.update(split_seed=int(split_seed), rank_semantics=RANK_SEMANTICS,
                   split_counts={k: int(len(splits[k])) for k in ("train", "calibration", "test")},
                   split_class_counts={k: np.bincount(labels[splits[k]], minlength=n_classes).tolist()
                                       for k in ("train", "calibration", "test")},
                   unused_source_rows=int(len(labels) - len(train) - len(cal) - len(test)))
    return splits, details


def validate_splits(labels, splits, n_classes):
    seen = set()
    for name in ("train", "calibration", "test"):
        ids = splits[name]
        if ids.dtype != np.int64 or len(np.unique(ids)) != len(ids):
            raise ValueError(f"Nonunique/invalid indices: {name}")
        if np.any(ids < 0) or np.any(ids >= len(labels)):
            raise ValueError(f"Out-of-bounds indices: {name}")
        current = set(ids.tolist())
        if current & seen:
            raise ValueError("Split overlap")
        seen.update(current)
        if set(labels[ids].tolist()) != set(range(n_classes)):
            raise ValueError(f"Missing classes: {name}")
    if sorted(splits["class_rank"].tolist()) != list(range(n_classes)):
        raise ValueError("Invalid class-rank permutation")


def _tabular(source, stage):
    source = Path(source)
    path = source if source.is_file() else source / "covtype.data.gz"
    with gzip.open(path, "rt", encoding="ascii") as f:
        rows = np.loadtxt(f, delimiter=",", dtype=np.float64)
    if rows.shape != (581012, 55) or not np.isfinite(rows).all():
        raise ValueError(f"Unexpected Covertype schema: {rows.shape}")
    if not np.equal(rows, np.floor(rows)).all():
        raise ValueError("Covertype source unexpectedly contains nonintegral values")
    labels = rows[:, -1].astype(np.int64) - 1
    if labels.min() != 0 or labels.max() != 6 or np.bincount(labels).tolist() != DATASETS["covertype"]["class_counts"]:
        raise ValueError("Covertype label inventory/counts differ from original metadata")
    features = rows[:, :-1].astype("<f4")
    if not np.equal(features.astype(np.float64), rows[:, :-1]).all():
        raise ValueError("Float32 conversion loses source feature precision")
    del rows
    categorical = features[:, 10:]
    if not np.isin(categorical, [0., 1.]).all() or not (categorical[:, :4].sum(1) == 1).all() or not (categorical[:, 4:].sum(1) == 1).all():
        raise ValueError("Unexpected Covertype categorical indicator encoding")
    np.save(stage / "features.npy", features, allow_pickle=False)
    keys = [(hashlib.sha256(row.tobytes()).digest(),) for row in features]
    canonical, audit = canonical_development(labels, keys)
    return labels, canonical, audit, None


def _csv_records(archive):
    """Read only bounded ordinary members; never extract archive paths."""
    with tarfile.open(archive, "r:gz") as tar:
        members = {}
        for member in tar.getmembers():
            _safe_relative(member.name)
            if not (member.isfile() or member.isdir()):
                raise ValueError("DBpedia tar contains a link/device/nonordinary member")
            if member.name in members:
                raise ValueError("DBpedia tar contains duplicate member paths")
            members[member.name] = member
        for split in ("train", "test"):
            name = f"dbpedia_csv/{split}.csv"
            member = members.get(name)
            if member is None or not member.isfile() or member.size > 2 * 1024**3:
                raise ValueError(f"Missing/oversized DBpedia CSV: {name}")
            stream = tar.extractfile(member)
            if stream is None:
                raise ValueError("Cannot read archive member")
            with io.TextIOWrapper(stream, encoding="utf-8", newline="") as text:
                for row in csv.reader(text):
                    if len(row) != 3:
                        raise ValueError("Unexpected DBpedia CSV schema")
                    yield split, int(row[0]) - 1, row[1], row[2]


def _parquet_records(source, info):
    try:
        import pyarrow.parquet as pq
    except ImportError as e:
        raise RuntimeError("Parquet preparation needs pyarrow; alternatively supply the original CSV archive") from e
    for split in ("train", "test"):
        names = sorted(name for name in info["files"] if PurePosixPath(name).name.startswith(split + "-") and name.endswith(".parquet"))
        if not names:
            raise ValueError(f"No pinned parquet files for {split}")
        for name in names:
            parquet = pq.ParquetFile(Path(source) / name)
            if not {"label", "title", "content"}.issubset(parquet.schema.names):
                raise ValueError("Unexpected DBpedia parquet schema")
            for batch in parquet.iter_batches(batch_size=8192, columns=["label", "title", "content"]):
                values = batch.to_pydict()
                for label, title, content in zip(values["label"], values["title"], values["content"]):
                    if not isinstance(title, str) or not isinstance(content, str):
                        raise ValueError("DBpedia has null/nontext fields")
                    yield split, int(label), title, content


def _text(source, stage, source_info):
    source = Path(source)
    if source_info.get("distribution") == "huggingface-parquet":
        records = _parquet_records(source, source_info)
    else:
        archive = source if source.is_file() else source / "dbpedia_csv.tar.gz"
        records = _csv_records(archive)
    labels, keys, split_counts = [], [], Counter()
    with open(stage / "texts.jsonl", "w", encoding="utf-8") as out:
        for split, label, title, content in records:
            split_counts[split] += 1
            if split_counts[split] > DATASETS["dbpedia"][split + "_rows"] or not 0 <= label < 14:
                raise ValueError("DBpedia source exceeds expected row counts or label range")
            if split == "train" and split_counts["test"]:
                raise ValueError("Native training rows must precede test rows")
            labels.append(label)
            keys.append(text_keys(title, content))
            out.write(json.dumps({"title": title, "content": content}, ensure_ascii=False) + "\n")
    if dict(split_counts) != {"train": 560000, "test": 70000}:
        raise ValueError(f"Unexpected DBpedia split sizes: {dict(split_counts)}")
    labels = np.asarray(labels, dtype=np.int64)
    if np.bincount(labels[:560000], minlength=14).tolist() != [40000]*14 or np.bincount(labels[560000:], minlength=14).tolist() != [5000]*14:
        raise ValueError("Unexpected DBpedia native class counts")
    canonical, audit = canonical_development(labels, keys, 560000)
    audit["normalization"] = "NFKC + Unicode casefold + collapsed whitespace; full title/content and nonempty content-only SHA256 keys"
    audit["limitation"] = "No claim of paraphrase/entity-level deduplication; native test duplicates retained and counted"
    return labels, canonical, audit, 560000


def prepare(dataset, source, output, rho=100., split_seed=SPLIT_SEED, rank_seed=RANK_SEED):
    source_info = _source_record(source, dataset)
    stage = _stage(output)
    try:
        if dataset == "covertype":
            labels, canonical, audit, boundary = _tabular(source, stage)
        else:
            labels, canonical, audit, boundary = _text(source, stage, source_info)
        splits, details = make_splits(labels, canonical, dataset, rho, split_seed, rank_seed, boundary)
        np.save(stage / "labels.npy", labels, allow_pickle=False)
        np.savez_compressed(stage / "split.npz", **splits)
        files = sorted(p.name for p in stage.iterdir() if p.is_file())
        condition = "natural" if dataset == "covertype" else f"rho{rho:g}"
        spec = DATASETS[dataset]
        metadata = {"version": VERSION, "created_utc": now(), "dataset": dataset,
                    "condition": condition, "task_type": "tabular" if dataset == "covertype" else "text",
                    "classes": spec["classes"], "n_classes": spec["n_classes"],
                    "rows": int(len(labels)), "row_id_semantics": "zero-based original source row; native train precedes native test",
                    "source": source_info, "source_page": spec["page"], "license": spec["license"],
                    "citation": spec["citation"], "dedup_audit": audit,
                    "files_sha256": {name: sha(stage / name) for name in files},
                    "fitted_preprocessing": "none; downstream transforms must fit train indices only",
                    "preparation_source_sha256": sha(__file__), **details}
        metadata["protocol_fingerprint"] = fingerprint({k: v for k, v in metadata.items() if k != "created_utc"})
        write_json(stage / "metadata.json", metadata)
        stage.rename(output)
        return metadata
    except BaseException:
        shutil.rmtree(stage)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("catalog", help="Print source/license metadata; no network operation")
    fetch = commands.add_parser("acquire", help="Explicitly download public data to a new directory")
    fetch.add_argument("--dataset", choices=DATASETS, required=True)
    fetch.add_argument("--output", type=Path, required=True)
    fetch.add_argument("--distribution", choices=["huggingface", "original"], default="huggingface")
    ready = commands.add_parser("prepare", help="Prepare an already acquired source without network access")
    ready.add_argument("--dataset", choices=DATASETS, required=True)
    ready.add_argument("--source", type=Path, required=True)
    ready.add_argument("--output", type=Path, required=True)
    ready.add_argument("--rho", type=float, default=100.)
    ready.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    ready.add_argument("--rank-seed", type=int, default=RANK_SEED)
    args = parser.parse_args()
    if args.command == "catalog":
        value = DATASETS
    elif args.command == "acquire":
        value = acquire(args.dataset, args.output, args.distribution)
    else:
        value = prepare(args.dataset, args.source, args.output, args.rho, args.split_seed, args.rank_seed)
    print(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
