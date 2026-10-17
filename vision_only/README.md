# Vision-only full-source classification study

This version implements a controlled study on **full ImageNet-1K, CIFAR-100 and iNaturalist 2018**. The full official CIFAR-100 release has been downloaded and checksum-verified on the remote CUDA computer. A live checkpoint audit on **October 17, 2026 at 20:57 UTC** verified **16 completed tuning fits and 138 completed final fits**, each at 50 epochs, plus both complete frozen-feature caches. CIFAR final-seed training remains active. Full official iNaturalist acquisition has started; its training is queued behind source verification and independent validation of all 288 CIFAR final checkpoints. These are dated progress counts, and all new manuscript outcomes remain **Pending**. See [STATUS.json](STATUS.json) for the full record and remaining ImageNet source dependency. Unit-test fixtures and the remote CUDA functionality check are software checks, not benchmark measurements.

The standalone [IEEE manuscript](paper/main.tex) begins with image-cataloguing and biodiversity use cases and states the research goal before the protocol. The [study configuration](study.json) is the source of the planned experiment grid. It contains no credentials, remote connection address, private dataset paths or dataset images.

## Datasets and scientific scope

| Dataset | Official training source | Official confirmation source | Classes | Planned conditions |
|---|---:|---:|---:|---|
| ImageNet-1K / ILSVRC 2012 | 1,281,167 | 50,000 validation | 1,000 | Native; study-specific exponential LT-100 |
| CIFAR-100 | 50,000 | 10,000 test | 100 | Native; study-specific exponential LT-100 |
| iNaturalist 2018 | 437,513 | 24,426 validation | 8,142 | Native natural imbalance |

Full inventories are checked **before** documented duplicate exclusions and training-derived development holdouts. No smaller source is accepted under a full-dataset name. A thinned long-tail fit condition is explicitly a subset; it is not a substitute for full ImageNet and is not the published ImageNet-LT release.

Exact RGB-pixel duplicates are handled before the training-source partitions are drawn. Every official confirmation row keeps its original ID and label, including identical images with conflicting official labels. Every training row that exactly overlaps confirmation is excluded regardless of label. For training-only duplicate groups, all rows are excluded when labels conflict; otherwise the lexicographically first sample ID is retained. All original source rows, labels and full-source counts remain in the manifest, with explicit reasons for training exclusions. Every class must still retain fit support. This policy was frozen before scientific feature extraction or classifier fitting; it does not use confirmation predictions or scores.

Official sources: [ImageNet](https://image-net.org/download.php), [CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html), [iNaturalist 2018](https://github.com/visipedia/inat_comp/tree/master/2018). ImageNet requires approved access and may require the researcher to accept access terms. Obtain datasets through those authorized routes. Do not put passwords, account tokens, archives or images in Git. The iNaturalist release restricts image redistribution.

This is a **frozen pretrained-feature adaptation study**. DeiT-S/16 and DINOv2 ViT-S/14 both output 384-dimensional features; both use the same 256-dimensional adapter/head design and deterministic 224-pixel transform. DeiT was pretrained on ImageNet labels. Subsampling current ImageNet labels does not make the pretrained encoder independent of those examples. Comparisons concern complete pretrained packages, not an isolated architecture, a clean from-scratch experiment, or proof that replacing a GAN caused a gain.

## Planned comparisons

Each of five dataset conditions crosses two encoders, two samplers and EQ off/on. Natural sampling uses importance-weighted CE; class-uniform sampling uses ordinary CE. Both target the same class-balanced expected classification objective. Two additional controls per encoder use natural CE and Balanced Softmax.

There are 60 arm conditions. Twelve paired final seeds, 10–21, give **720 planned final fits**. Eight tuning fits per encoder/condition give **80 planned tuning fits**, for **800 planned fits** overall. Feature extraction and the two-update train-source CUDA functionality pilot are separate setup work. None is a completed accuracy result.

The two balanced-target EQ-off samplers jointly select one learning rate per encoder/condition from 0.001 and 0.003; all four factorial cells inherit it. Anchors select their learning rate independently. All formal fits use 50 sampling epochs, complete batches of 256, AdamW, weight decay 0.01, 10% warmup/cosine decay, and gradient clipping at 1. EQ uses coefficient 0.05 during the final 20% of epochs. Repeated-row and distinct-image eligibility are measured separately. The code records periodic gradient diagnostics and exact per-class eligibility counters in that active schedule phase.

Only fit rows determine scaling and learned parameters. Training-source tuning rows determine learning rates; separate training-source calibration rows determine the natural-CE offset. Nominal 5%/5% development fractions adapt to rare classes while retaining fit support. Classes with fewer than three source training examples remain fit-only and their absent development coverage is recorded. Official confirmation data are not scored until **all 720 final fits across all three datasets** are trained and globally sealed.

Reported metrics include top-1/top-5, macro-F1, class precision/recall/F1 and frequency-ranked subgroup scores. Native CIFAR subgroup thirds are descriptive slices, not naturally rare categories. Six primary contrasts per condition comprise four conditional EQ effects and two encoder × EQ interactions. Exact paired sign-flip tests and paired intervals describe optimization variability conditional on the fixed partitions; twelve seeds are not twelve independent datasets. Holm correction is applied within condition, with a combined 30-contrast sensitivity analysis.

## Remote execution

Use Python **3.10 or newer** on the authorized CUDA computer, with a compatible CUDA PyTorch installation. The requirements describe compatible packages; actual package, encoder-state and source identities are recorded with the run. Production extraction, fitting and confirmation reject CPU/MPS fallback. CPU fixtures in the tests do not use benchmark data.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r vision_only/requirements.txt
python -m vision_only.lifecycle preflight --root /scratch/rebal-vision-study
python -m vision_only.manage --work /scratch/rebal-vision-study init
```

Initialize the empty owned run directory **before** downloading images into its `data/` subdirectory. Store all newly downloaded archives, extracted images and model-download caches under that directory. For example, set `TORCH_HOME` and `HF_HOME` to owned `data/model_cache/` subdirectories. Existing shared datasets outside the owned run directory are borrowed read-only and are never cleanup targets.

Create a private paths JSON using [paths.example.json](paths.example.json) as the format; replace its placeholders on the remote machine. Keep that private file outside Git. ImageNet validation images must be arranged under the correct WordNet class folders using the official development-kit mapping. iNaturalist uses the original 2018 train/validation annotation files and image paths. CIFAR uses the official checksummed torchvision layout.

```bash
python -m vision_only.control --work /scratch/rebal-vision-study --paths /scratch/private-vision-paths.json prepare
python -m vision_only.control --work /scratch/rebal-vision-study extract
python -m vision_only.control --work /scratch/rebal-vision-study fit
python -m vision_only.control --work /scratch/rebal-vision-study confirm
python -m vision_only.control --work /scratch/rebal-vision-study analyze
```

Datasets may be prepared, extracted and fitted incrementally as their full official sources become available. The optional `--datasets` flag only schedules work; it does not change the study, its 720 final fits, or the requirement to finish every dataset before confirmation. For example, a private paths JSON containing only the `cifar100` entry can start both full-CIFAR conditions while ImageNet and iNaturalist are being obtained:

```bash
python -m vision_only.control --work /scratch/rebal-vision-study prepare --datasets cifar100 --paths /scratch/private-cifar-paths.json
python -m vision_only.control --work /scratch/rebal-vision-study extract --datasets cifar100
python -m vision_only.control --work /scratch/rebal-vision-study fit --datasets cifar100
```

Place the stage before `--datasets` because that option accepts one or more dataset names. For filtered preparation, the paths JSON must contain exactly the selected datasets. Partial fitting validates the selected checkpoints and records their count as a scheduled subset; it creates no global seal and makes no all-study completion claim. Repeat these stages for the remaining full datasets, then run the unfiltered `fit` command to validate and seal the complete grid. `--datasets` is rejected for `seal`, `confirm`, `analyze` and `all`; confirmation, aggregate reporting, export and cleanup keep their complete-study requirements.

Run the stages in a persistent remote terminal or job manager. Do not edit code/configuration during execution: resume, cache and global-seal checks deliberately reject identity changes. Fit stages resume only from coherent epoch-boundary checkpoints. Never infer completion from a detached process, existing output directory, partial cache or a passed pilot.

Confirmation stores raw/adjusted top-5 and predicted labels, plus calibration predictions for every candidate offset. Aggregation independently replays reported metrics and offset selection. A full-source manifest, per-seed results, per-class results, comparisons, checkpoints, selection receipts and diagnostics must be exported before cleanup. Publish only sanitized code, protocol and validated aggregates; do not upload full image inventories containing private roots or any images.

## Export and remove study data after completion

The lifecycle refuses export/cleanup when a planned run is missing, failed, pending, changed, or has no supporting artifacts. It requires complete aggregate results, an inventory of owned files, and a byte-verified result archive **outside** the run directory. Images and feature caches are excluded from that archive.

```bash
python -m vision_only.manage --work /scratch/rebal-vision-study export --archive /scratch/exports/rebal-vision-results.zip
python -m vision_only.lifecycle cleanup --root /scratch/rebal-vision-study --receipt /scratch/exports/rebal-vision-results.zip.receipt.json
```

Cleanup above is a dry run. After exporting the evidence off the experiment computer and verifying the received copy, use the lifecycle's explicit execute option on the reviewed plan. It removes only this study's registered non-code artifacts and preserves code; it rejects unknown files, symlinks, hardlinks and escapes. Do not remove existing shared datasets or unrelated experiments. POSIX deletion is supported; other systems fail closed. If deletion is interrupted, retain the verified export and ownership marker and recover from that evidence before continuing; do not bypass the ownership checks.

## Software validation

```bash
python -m unittest tests.test_vision_only_data tests.test_vision_only_runner tests.test_vision_only_control tests.test_vision_only_lifecycle tests.test_vision_only_manage
```

These tests exercise protected partitions, official full-source enforcement, EQ mathematics and gradients, sampler pairing, resume corruption, checkpoint completion, cross-dataset confirmation gates and deletion/export safeguards. They do not establish CUDA compatibility or experimental performance; the real remote pilot and complete grid are still required.
