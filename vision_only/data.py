"""Local full-source inventories, protected partitions and cleanup previews.

No acquisition or dataset deletion happens in this module. CIFAR loading uses
the official torchvision loader only after verifying its published checksums.
Full source counts are checked before any declared duplicate exclusions.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import random
import struct
from typing import Any

SCHEMA_VERSION = 1
IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"})


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path | str) -> str:
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def atomic_json(path: Path | str, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False, separators=(",", ":"))
        stream.write("\n")
        stream.flush()
        import os
        os.fsync(stream.fileno())
    temporary.replace(path)


def contained_path(root: Path | str, relative: str, *, must_exist: bool = True) -> Path:
    """Reject absolute, traversal and symlink paths; retain one owned namespace."""
    root = Path(root).resolve(strict=True)
    item = PurePosixPath(relative)
    require(bool(relative) and not item.is_absolute() and ".." not in item.parts,
            "Dataset path must be a non-traversing relative path")
    require("\\" not in relative and ":" not in relative, "Ambiguous dataset path")
    path = root.joinpath(*item.parts)
    current = path
    while current != root:
        require(not current.is_symlink(), "Dataset symlinks are not supported")
        current = current.parent
    resolved = path.resolve(strict=must_exist)
    require(resolved.is_relative_to(root) and resolved != root, "Dataset path escapes its root")
    return resolved


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    dataset_id: str
    train_count: int
    confirmation_count: int
    num_classes: int
    source_url: str
    scope: str = "full"


_SPECS = {
    "imagenet1k": DatasetSpec("imagenet1k", 1281167, 50000, 1000,
                              "https://www.image-net.org/challenges/LSVRC/2012/"),
    "cifar100": DatasetSpec("cifar100", 50000, 10000, 100,
                            "https://www.cs.toronto.edu/~kriz/cifar.html"),
    "inat2018": DatasetSpec("inat2018", 437513, 24426, 8142,
                            "https://github.com/visipedia/inat_comp/blob/master/2018/README.md"),
}


def dataset_spec(name: str) -> DatasetSpec:
    require(name in _SPECS, "Unknown full-source dataset: " + name)
    return _SPECS[name]


def _validate_spec(spec: DatasetSpec) -> None:
    require(spec.num_classes > 1 and spec.train_count > 0 and spec.confirmation_count > 0,
            "Invalid expected source dimensions")
    if spec.dataset_id in _SPECS:
        require(spec == _SPECS[spec.dataset_id], "Official full-source dimensions cannot be overridden")
    else:
        require(spec.scope == "test_fixture", "Custom specs are restricted to explicit test fixtures")


@dataclass(frozen=True, slots=True)
class Sample:
    sample_id: str
    relative_path: str
    label: int
    source_split: str
    sha256: str
    pixel_sha256: str
    size_bytes: int
    width: int
    height: int
    record_index: int | None = None


@dataclass(frozen=True)
class Manifest:
    dataset_id: str
    scope: str
    source_root: str
    source_url: str
    class_names: tuple[str, ...]
    source_class_ids: tuple[str, ...]
    expected_counts: dict[str, int]
    source_files: dict[str, str]
    samples: tuple[Sample, ...]
    exclusions: dict[str, str] = field(default_factory=dict)

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    @property
    def by_id(self) -> dict[str, Sample]:
        return {sample.sample_id: sample for sample in self.samples}

    def to_dict(self, *, public: bool = False) -> dict[str, Any]:
        value = asdict(self)
        value["schema_version"] = SCHEMA_VERSION
        value["fingerprint"] = self.fingerprint
        if public:
            value.pop("source_root")
        return value

    @property
    def fingerprint(self) -> str:
        # Million-row inventories must not be deep-copied to compute identity.
        # Hash every ordered sample and bind that stream to all other metadata.
        sample_digest = hashlib.sha256()
        for sample in self.samples:
            sample_digest.update(json.dumps(asdict(sample), sort_keys=True, separators=(",", ":"),
                                            allow_nan=False).encode())
            sample_digest.update(b"\n")
        return fingerprint({"schema_version": SCHEMA_VERSION, "dataset_id": self.dataset_id,
                            "scope": self.scope, "source_url": self.source_url,
                            "class_names": self.class_names, "source_class_ids": self.source_class_ids,
                            "expected_counts": self.expected_counts, "source_files": self.source_files,
                            "sample_count": len(self.samples), "samples_sha256": sample_digest.hexdigest(),
                            "exclusions": self.exclusions})


def validate_manifest(manifest: Manifest) -> None:
    expected = manifest.expected_counts
    require(set(expected) == {"train", "confirmation", "classes"}, "Invalid expected-count receipt")
    require(manifest.num_classes == expected["classes"] == len(manifest.source_class_ids),
            "Source class inventory mismatch")
    require(len(set(manifest.source_class_ids)) == manifest.num_classes,
            "Duplicated source class IDs")
    if manifest.dataset_id in _SPECS:
        spec = dataset_spec(manifest.dataset_id)
        require(manifest.scope == "full" and expected == {
            "train": spec.train_count, "confirmation": spec.confirmation_count, "classes": spec.num_classes},
                "Official full-source expected counts changed")
    else:
        require(manifest.scope == "test_fixture", "Unregistered production dataset")
    ids: set[str] = set()
    counts: Counter[str] = Counter()
    classes: dict[str, set[int]] = defaultdict(set)
    for sample in manifest.samples:
        require(sample.sample_id not in ids, "Duplicate sample ID")
        ids.add(sample.sample_id)
        require(sample.source_split in {"train", "confirmation"}, "Unknown source partition")
        require(type(sample.label) is int and 0 <= sample.label < manifest.num_classes,
                "Label outside the frozen class mapping")
        require(sample.width > 0 and sample.height > 0 and sample.size_bytes > 0,
                "Invalid image dimensions or size")
        require(all(len(digest) == 64 and all(c in "0123456789abcdef" for c in digest)
                    for digest in (sample.sha256, sample.pixel_sha256)), "Invalid image hashes")
        counts[sample.source_split] += 1
        classes[sample.source_split].add(sample.label)
    require(counts["train"] == expected["train"] and counts["confirmation"] == expected["confirmation"],
            "Full-source image count mismatch before exclusions")
    require(all(classes[name] == set(range(manifest.num_classes)) for name in ("train", "confirmation")),
            "Every expected class must be represented in both official source partitions")
    require(set(manifest.exclusions) <= ids, "Exclusion refers to unknown sample")
    by_id = manifest.by_id if manifest.exclusions else {}
    require(all(by_id[item].source_split == "train" and bool(reason)
                for item, reason in manifest.exclusions.items()), "Only explicit training exclusions are allowed")


def _image_record(root: Path, relative: str, label: int, split: str, sample_id: str) -> Sample:
    path = contained_path(root, relative)
    require(path.is_file(), "Missing source image: " + relative)
    # Decode original RGB pixels: encoded-file equality alone misses re-encoding.
    from PIL import Image
    with Image.open(path) as opened:
        image = opened.convert("RGB")
        image.load()
        width, height = image.size
        pixels = image.tobytes()
    pixel_digest = hashlib.sha256(struct.pack(">II", width, height) + pixels).hexdigest()
    return Sample(sample_id, relative, label, split, sha256_file(path), pixel_digest,
                  path.stat().st_size, width, height)


def _manifest(root: Path, spec: DatasetSpec, names: list[str], source_ids: list[str],
              samples: list[Sample], source_files: dict[str, str]) -> Manifest:
    _validate_spec(spec)
    result = Manifest(spec.dataset_id, spec.scope, str(root.resolve()), spec.source_url,
                      tuple(names), tuple(source_ids), {
                          "train": spec.train_count, "confirmation": spec.confirmation_count,
                          "classes": spec.num_classes}, source_files,
                      tuple(sorted(samples, key=lambda row: row.sample_id)))
    validate_manifest(result)
    return result


def inventory_imagefolder(root: Path | str, spec: DatasetSpec | None = None, *,
                          train_dir: str = "train", confirmation_dir: str = "val") -> Manifest:
    """Inventory reorganized official ImageFolder sources; never extract/download."""
    spec = spec or dataset_spec("imagenet1k")
    _validate_spec(spec)
    root = Path(root).resolve(strict=True)
    train = contained_path(root, train_dir)
    confirmation = contained_path(root, confirmation_dir)
    require(train.is_dir() and confirmation.is_dir(), "ImageFolder split directories are missing")
    names = sorted(path.name for path in train.iterdir() if path.is_dir())
    other = sorted(path.name for path in confirmation.iterdir() if path.is_dir())
    require(len(names) == spec.num_classes and names == other,
            "Train/evaluation ImageFolder class-name mappings differ")
    samples = []
    for split, directory in (("train", train), ("confirmation", confirmation)):
        for label, name in enumerate(names):
            files = sorted(path for path in (directory / name).rglob("*")
                           if path.suffix.lower() in IMAGE_EXTENSIONS)
            for path in files:
                relative = path.relative_to(root).as_posix()
                samples.append(_image_record(root, relative, label, split, relative))
    return _manifest(root, spec, names, names, samples, {})


def inventory_inat2018(root: Path | str, train_json: Path | str, val_json: Path | str,
                       spec: DatasetSpec | None = None) -> Manifest:
    """Read labeled COCO-style iNat train/val annotations; do not touch unlabeled test."""
    spec = spec or dataset_spec("inat2018")
    _validate_spec(spec)
    root = Path(root).resolve(strict=True)
    annotations = []
    source_files = {}
    for source in (train_json, val_json):
        path = Path(source)
        if path.is_absolute():
            require(not path.is_symlink(), "Annotation file cannot be a symlink")
            resolved = path.resolve(strict=True)
            require(resolved.is_relative_to(root), "Annotation path is outside data root")
            relative = resolved.relative_to(root).as_posix()
        else:
            relative = path.as_posix()
        path = contained_path(root, relative)
        source_files[relative] = sha256_file(path)
        annotations.append(json.loads(path.read_text(encoding="utf-8")))
    mappings = []
    for document in annotations:
        categories = document.get("categories", [])
        require(len(categories) == spec.num_classes, "Full iNat category inventory is missing")
        mapping = {str(row["id"]): str(row.get("name", row["id"])) for row in categories}
        require(len(mapping) == len(categories), "Duplicate iNat category IDs")
        mappings.append(mapping)
    require(mappings[0] == mappings[1], "Train/val iNat category mappings differ")
    source_ids = sorted(mappings[0], key=lambda key: (0, int(key)) if key.lstrip("-").isdigit() else (1, key))
    label_of = {source_id: label for label, source_id in enumerate(source_ids)}
    samples = []
    used_paths = set()
    for split, document in zip(("train", "confirmation"), annotations):
        images = document.get("images", [])
        by_id = {str(row["id"]): row for row in images}
        require(len(by_id) == len(images), "Duplicate iNat image IDs")
        labels = {}
        for annotation in document.get("annotations", []):
            image_id, category = str(annotation["image_id"]), str(annotation["category_id"])
            require(image_id in by_id and category in label_of, "Unknown iNat image/category reference")
            require(image_id not in labels, "iNat classification image must have exactly one annotation")
            labels[image_id] = label_of[category]
        require(set(labels) == set(by_id), "Missing labeled iNat image annotation")
        for image_id in sorted(by_id):
            row = by_id[image_id]
            relative = str(row["file_name"])
            require(relative not in used_paths, "iNat source image appears in multiple partitions")
            used_paths.add(relative)
            record = _image_record(root, relative, labels[image_id], split, split + ":" + image_id)
            if "width" in row:
                require(record.width == int(row["width"]), "iNat annotated width differs from decoded image")
            if "height" in row:
                require(record.height == int(row["height"]), "iNat annotated height differs from decoded image")
            samples.append(record)
    return _manifest(root, spec, [mappings[0][key] for key in source_ids], source_ids,
                     samples, source_files)


def _official_cifar100_class():
    from torchvision.datasets import CIFAR100
    return CIFAR100


def inventory_cifar100(root: Path | str, spec: DatasetSpec | None = None) -> Manifest:
    """Official torchvision CIFAR loader, download=False, checksums before pickle."""
    spec = spec or dataset_spec("cifar100")
    _validate_spec(spec)
    root = Path(root).resolve(strict=True)
    dataset_type = _official_cifar100_class()
    source_files = {}
    files = list(dataset_type.train_list) + list(dataset_type.test_list)
    files.append((dataset_type.meta["filename"], dataset_type.meta["md5"]))
    for name, expected_md5 in files:
        relative = dataset_type.base_folder + "/" + name
        path = contained_path(root, relative)
        digest = hashlib.md5(path.read_bytes()).hexdigest()  # publisher format, not an identity/security hash
        require(digest == expected_md5, "Official CIFAR checksum failed before deserialization")
        source_files[relative] = sha256_file(path)
    samples = []
    names = None
    for split, training in (("train", True), ("confirmation", False)):
        dataset = dataset_type(root=str(root), train=training, download=False)
        require(dataset._check_integrity(), "Official CIFAR integrity check failed")
        current = tuple(str(name) for name in dataset.classes)
        require(len(current) == spec.num_classes and len(set(current)) == spec.num_classes,
                "CIFAR class mapping differs")
        require(names is None or names == current, "CIFAR train/test label mappings differ")
        names = current
        source_name = (dataset_type.train_list if training else dataset_type.test_list)[0][0]
        relative = dataset_type.base_folder + "/" + source_name
        require(len(dataset.data) == len(dataset.targets), "CIFAR image/label coverage differs")
        for index, (image, label) in enumerate(zip(dataset.data, dataset.targets)):
            require(image.shape == (32, 32, 3) and str(image.dtype) == "uint8", "CIFAR source shape/dtype differs")
            pixels = image.tobytes()
            digest = hashlib.sha256(pixels).hexdigest()
            pixel_digest = hashlib.sha256(struct.pack(">II", 32, 32) + pixels).hexdigest()
            samples.append(Sample(split + ":" + str(index), relative, int(label), split,
                                  digest, pixel_digest, len(pixels), 32, 32, index))
    return _manifest(root, spec, list(names or ()), list(names or ()), samples, source_files)


def save_manifest(manifest: Manifest, path: Path | str, *, public: bool = False) -> None:
    validate_manifest(manifest)
    atomic_json(path, manifest.to_dict(public=public))


def load_manifest(path: Path | str, *, root_override: Path | str | None = None) -> Manifest:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    require(value.pop("schema_version") == SCHEMA_VERSION, "Unsupported manifest schema")
    expected_fingerprint = value.pop("fingerprint")
    root = root_override if root_override is not None else value.get("source_root")
    require(root is not None, "Public manifests need an explicit local root override")
    value["source_root"] = str(Path(root).resolve())
    value["class_names"] = tuple(value["class_names"])
    value["source_class_ids"] = tuple(value["source_class_ids"])
    value["samples"] = tuple(Sample(**sample) for sample in value["samples"])
    result = Manifest(**value)
    validate_manifest(result)
    require(result.fingerprint == expected_fingerprint, "Manifest fingerprint changed")
    return result


def verify_manifest_files(manifest: Manifest) -> None:
    """Recheck local bytes; CIFAR records rely on their publisher-checked file hashes."""
    validate_manifest(manifest)
    root = Path(manifest.source_root)
    for relative, digest in manifest.source_files.items():
        require(sha256_file(contained_path(root, relative)) == digest, "Source annotation/archive changed")
    for sample in manifest.samples:
        path = contained_path(root, sample.relative_path)
        if sample.record_index is None:
            require(path.stat().st_size == sample.size_bytes and sha256_file(path) == sample.sha256,
                    "Source image bytes changed: " + sample.sample_id)


def exclude_training_exact_duplicates(manifest: Manifest) -> Manifest:
    """Preserve official confirmation labels and declare training-only exclusions.

    Every training overlap with confirmation is excluded regardless of labels.
    Ambiguous training-only pixel groups are excluded in full; same-label groups
    retain their lexicographically first sample ID. Source rows remain intact.
    """
    validate_manifest(manifest)
    confirmation = {sample.pixel_sha256 for sample in manifest.samples
                    if sample.source_split == "confirmation"}
    training = defaultdict(list)
    for sample in manifest.samples:
        if sample.source_split == "train":
            training[sample.pixel_sha256].append(sample)
    exclusions = dict(manifest.exclusions)
    for pixel_hash, samples in training.items():
        if pixel_hash in confirmation:
            for sample in samples:
                exclusions[sample.sample_id] = "exact_pixel_overlap_with_official_confirmation"
        elif len({sample.label for sample in samples}) > 1:
            for sample in samples:
                exclusions[sample.sample_id] = "conflicting_label_training_exact_pixel_group"
        else:
            keeper = min(samples, key=lambda sample: sample.sample_id)
            for sample in samples:
                if sample.sample_id != keeper.sample_id:
                    exclusions[sample.sample_id] = "exact_training_pixel_duplicate_of:" + keeper.sample_id
    return Manifest(manifest.dataset_id, manifest.scope, manifest.source_root, manifest.source_url,
                    manifest.class_names, manifest.source_class_ids, manifest.expected_counts,
                    manifest.source_files, manifest.samples, exclusions)


@dataclass(frozen=True)
class SplitPlan:
    manifest_fingerprint: str
    seed: int
    regime: str
    tune_fraction: float
    calibration_fraction: float
    lt_ratio: float | None
    fit: tuple[str, ...]
    tune: tuple[str, ...]
    calibration: tuple[str, ...]
    confirmation: tuple[str, ...]
    lt_omitted: tuple[str, ...]
    excluded: tuple[str, ...]
    class_order: tuple[int, ...]
    class_counts: dict[str, list[int]]
    limited_development_classes: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["schema_version"] = SCHEMA_VERSION
        value["fingerprint"] = fingerprint(value)
        return value

    @property
    def fingerprint(self) -> str:
        return self.to_dict()["fingerprint"]


def _rng(seed: int, namespace: str) -> random.Random:
    return random.Random(int(fingerprint({"seed": seed, "namespace": namespace}), 16))


def make_partitions(manifest: Manifest, seed: int, *, tune_fraction: float = .05,
                    calibration_fraction: float = .05, lt_ratio: float | None = None) -> SplitPlan:
    """5%+5% train-only development by default; n<=2 classes remain fit-only.

    n>=3 reserves at least one tuning and one decision-adjustment example while
    retaining at least one fit row. Native fitting retains every other eligible
    training row. Long-tail sampling is an explicitly labeled sensitivity.
    """
    validate_manifest(manifest)
    require(type(seed) is int and seed >= 0, "Seed must be a nonnegative integer")
    require(0 < tune_fraction < 1 and 0 < calibration_fraction < 1 and
            tune_fraction + calibration_fraction < 1, "Invalid development fractions")
    require(lt_ratio is None or (math.isfinite(lt_ratio) and lt_ratio >= 1), "Invalid long-tail ratio")
    by_class = defaultdict(list)
    by_id = manifest.by_id
    for sample in manifest.samples:
        if sample.source_split == "train" and sample.sample_id not in manifest.exclusions:
            by_class[sample.label].append(sample.sample_id)
    fit_pool, tune, calibration, limited = {}, [], [], []
    for label in range(manifest.num_classes):
        ids = sorted(by_class[label])
        require(ids, "A class has no fit examples after declared exclusions")
        _rng(seed, "partition:" + str(label)).shuffle(ids)
        if len(ids) <= 2:
            n_tune = n_calibration = 0
            limited.append(label)
        else:
            n_tune = max(1, math.floor(len(ids) * tune_fraction))
            n_calibration = max(1, math.floor(len(ids) * calibration_fraction))
            while n_tune + n_calibration >= len(ids):
                if n_tune > n_calibration and n_tune > 1:
                    n_tune -= 1
                elif n_calibration > 1:
                    n_calibration -= 1
                else:
                    raise ValueError("Cannot retain a fit sample with requested development allocation")
        tune.extend(ids[:n_tune])
        calibration.extend(ids[n_tune:n_tune + n_calibration])
        fit_pool[label] = ids[n_tune + n_calibration:]
    require(tune and calibration, "No disjoint development data available; do not substitute official confirmation")
    fit, omitted = [], []
    rank_order = list(range(manifest.num_classes))
    _rng(seed, "longtail_class_rank").shuffle(rank_order)
    rank = {label: index for index, label in enumerate(rank_order)}
    largest = max(map(len, fit_pool.values()))
    for label in range(manifest.num_classes):
        ids = fit_pool[label]
        if lt_ratio is None:
            keep = len(ids)
        else:
            target = max(1, math.floor(largest * lt_ratio ** (-rank[label] / (manifest.num_classes - 1))))
            keep = min(len(ids), target)
        fit.extend(ids[:keep])
        omitted.extend(ids[keep:])
    groups = {"fit": fit, "tune": tune, "calibration": calibration,
              "confirmation": [sample.sample_id for sample in manifest.samples
                               if sample.source_split == "confirmation"]}
    group_counts = {name: Counter(by_id[item].label for item in ids) for name, ids in groups.items()}
    counts = {name: [group_counts[name][label] for label in range(manifest.num_classes)]
              for name in groups}
    order = tuple(sorted(range(manifest.num_classes), key=lambda label: (-counts["fit"][label], label)))
    result = SplitPlan(manifest.fingerprint, seed, "native" if lt_ratio is None else "custom_LT",
                       tune_fraction, calibration_fraction, lt_ratio,
                       tuple(sorted(fit)), tuple(sorted(tune)), tuple(sorted(calibration)),
                       tuple(sorted(groups["confirmation"])), tuple(sorted(omitted)),
                       tuple(sorted(manifest.exclusions)), order, counts, tuple(limited))
    validate_partitions(manifest, result)
    return result


def validate_partitions(manifest: Manifest, plan: SplitPlan) -> None:
    validate_manifest(manifest)
    require(plan.manifest_fingerprint == manifest.fingerprint, "Partition/manifest identity differs")
    by_id = manifest.by_id
    all_ids = set(by_id)
    names = ("fit", "tune", "calibration", "confirmation", "lt_omitted", "excluded")
    seen = set()
    split_of_hash = {}
    for name in names:
        ids = getattr(plan, name)
        require(len(ids) == len(set(ids)) and not set(ids) & seen and set(ids) <= all_ids,
                "Partition IDs are missing, duplicated or overlapping")
        seen.update(ids)
        for item in ids:
            sample = by_id[item]
            require(sample.source_split == ("confirmation" if name == "confirmation" else "train"),
                    "Development/fitting data must come only from the training source")
            if name not in {"lt_omitted", "excluded"}:
                prior = split_of_hash.setdefault(sample.pixel_sha256, name)
                require(prior == name, "Exact image content crosses protected partitions; declare exclusions first")
    require(seen == all_ids, "Partition receipt does not account for every source image")
    require(set(plan.excluded) == set(manifest.exclusions), "Declared exclusions differ")
    require(set(plan.confirmation) == {row.sample_id for row in manifest.samples
                                      if row.source_split == "confirmation"}, "Official confirmation coverage changed")
    require(plan.regime == ("native" if plan.lt_ratio is None else "custom_LT"), "Incorrect native/LT label")
    require(plan.regime != "native" or not plan.lt_omitted, "Native fitting silently discarded training rows")
    require(sorted(plan.class_order) == list(range(manifest.num_classes)), "Invalid frequency class order")
    for name in ("fit", "tune", "calibration", "confirmation"):
        count = Counter(by_id[item].label for item in getattr(plan, name))
        require(plan.class_counts[name] == [count[label] for label in range(manifest.num_classes)],
                "Partition class counts changed")
    require(all(plan.class_counts["fit"]), "A class has zero fit support")
    require(plan.class_order == tuple(sorted(range(manifest.num_classes),
                                             key=lambda label: (-plan.class_counts["fit"][label], label))),
            "Class frequency ranking differs")
    limited = tuple(label for label in range(manifest.num_classes)
                    if plan.class_counts["tune"][label] == 0 or plan.class_counts["calibration"][label] == 0)
    require(plan.limited_development_classes == limited, "Limited development coverage was not declared")


def save_partitions(plan: SplitPlan, path: Path | str) -> None:
    atomic_json(path, plan.to_dict())


def load_partitions(path: Path | str, manifest: Manifest) -> SplitPlan:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    require(value.pop("schema_version") == SCHEMA_VERSION, "Unsupported split schema")
    digest = value.pop("fingerprint")
    for name in ("fit", "tune", "calibration", "confirmation", "lt_omitted", "excluded",
                 "class_order", "limited_development_classes"):
        value[name] = tuple(value[name])
    plan = SplitPlan(**value)
    require(plan.fingerprint == digest, "Partition fingerprint changed")
    validate_partitions(manifest, plan)
    return plan


def create_owned_data_root(path: Path | str, run_id: str) -> Path:
    """Mark a new empty data directory; never acquire ownership of shared data."""
    path = Path(path)
    require(not path.is_symlink(), "Owned data root cannot be a symlink")
    path.mkdir(parents=True, exist_ok=True)
    require(not any(path.iterdir()), "Cannot claim ownership of pre-existing data")
    require(bool(run_id) and all(ch.isalnum() or ch in "_-" for ch in run_id), "Invalid run ID")
    root = path.resolve()
    require(len(root.parts) >= 3 and root != Path.home().resolve(), "Unsafe data root")
    atomic_json(root / ".rebal_vision_data_owner.json", {"schema_version": SCHEMA_VERSION,
                "run_id": run_id, "resolved_root": str(root), "ownership": "task_created"})
    return root


def cleanup_preview(data_root: Path | str, run_id: str, targets: list[str],
                     export_receipt: Path | str) -> dict[str, Any]:
    """Fail-closed cleanup preview only. This function deliberately deletes nothing.

    The caller establishes remote/Git/Overleaf observations, binds them to run
    receipts, and supplies actual retained artifact hashes outside the data root.
    """
    raw_root = Path(data_root)
    require(not raw_root.is_symlink(), "Cleanup data root cannot be a symlink")
    root = raw_root.resolve(strict=True)
    marker = root / ".rebal_vision_data_owner.json"
    require(marker.is_file() and not marker.is_symlink(), "No task-owned dataset marker")
    owner = json.loads(marker.read_text())
    require(owner == {"schema_version": SCHEMA_VERSION, "run_id": run_id,
                      "resolved_root": str(root), "ownership": "task_created"}, "Dataset ownership differs")
    receipt_path = Path(export_receipt).resolve(strict=True)
    require(not receipt_path.is_relative_to(root), "Export verification cannot live only inside deleted data")
    receipt = json.loads(receipt_path.read_text())
    require(receipt.get("run_id") == run_id, "Export verification belongs to a different run")
    gates = ("all_required_cells_verified", "checkpoint_audit_verified", "independent_metric_replay_verified",
             "report_verified", "public_export_verified", "overleaf_compile_verified", "no_active_jobs")
    require(all(receipt.get(gate) is True for gate in gates), "Training/export/verification is incomplete; cleanup blocked")
    require(receipt.get("plan_fingerprint") and receipt.get("aggregate_fingerprint"), "Missing scientific run identities")
    copies = receipt.get("retained_evidence_copies", [])
    require(len(copies) >= 2, "Two retained evidence copies are required before dataset cleanup")
    retained = set()
    for copy in copies:
        path = Path(copy["path"]).resolve(strict=True)
        require(path.is_file() and not path.is_relative_to(root) and path not in retained,
                "Evidence must have distinct existing copies outside data root")
        require(sha256_file(path) == copy["sha256"], "Retained evidence copy hash differs")
        retained.add(path)
    require(len({copy["sha256"] for copy in copies}) == 1, "Evidence copies do not preserve the same verified package")
    require(bool(targets) and len(targets) == len(set(targets)), "Empty or duplicate cleanup targets")
    paths = [contained_path(root, relative) for relative in targets]
    require(all(path != marker for path in paths), "Ownership marker is retained as a tombstone anchor")
    for index, path in enumerate(paths):
        require(all(not path.is_relative_to(other) and not other.is_relative_to(path)
                    for other in paths[:index]), "Overlapping cleanup targets")
    rows = []
    for relative, path in zip(targets, paths):
        files = [path] if path.is_file() else list(path.rglob("*"))
        require(all(not item.is_symlink() for item in files), "Cleanup target contains a symlink")
        files = [item for item in files if item.is_file()]
        rows.append({"relative_path": relative, "bytes": sum(item.stat().st_size for item in files),
                     "files": len(files), "identity": fingerprint([
                         {"path": item.relative_to(root).as_posix(), "sha256": sha256_file(item)}
                         for item in sorted(files)])})
    return {"schema_version": SCHEMA_VERSION, "run_id": run_id, "status": "verified_preview_only",
            "export_receipt_sha256": sha256_file(receipt_path), "targets": rows,
            "total_bytes": sum(row["bytes"] for row in rows), "deleted": False}
