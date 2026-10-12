# Covertype and DBpedia non-image experiment

## Status and scope

Completed on October 12, 2026: all 15 condition-by-seed cells, 45 trained representations, 135 retrained classifiers and 405 named readouts passed independent numerical verification. Every representation completed 50 epochs and every retrained classifier completed 2,000 updates. Three CPU workers completed the full declared grid in 56 minutes 1 second, including aggregation; stage timing sums differ from elapsed wall time.

The experiment trains tabular/text representations from scratch, then compares classifier retraining on the frozen learned representations. It is a controlled non-image adaptation of REBAL-F. It uses neither the original image architecture nor a pretrained language model or generator. HIGGS is a researched secondary option and is not implemented or evaluated by these scripts. The existing Tiny ImageNet report describes a separate study.

## Results and evidence

The verified bundle includes the [experimental report](../results/nonimage_fullscale/experimental_report.pdf), [all-method summary](../results/nonimage_fullscale/summary.csv), [seed-level results](../results/nonimage_fullscale/per_seed_results.csv), [paired comparisons](../results/nonimage_fullscale/paired_comparisons.csv), [aggregate](../results/nonimage_fullscale/aggregate.json) and [independent verification](../results/nonimage_fullscale/independent_verification.json).

Mean macro-F1 percentages across five seeds (selected readouts for context; all 27 are in the report):

| Condition | CE + LA | CB+EQ / natural | CB+EQ / ROS | CB+EQ / SMOTE |
|---|---:|---:|---:|---:|
| Covertype | 78.574 | 77.149 | 70.930 | 71.079 |
| DBpedia 10:1 | 95.000 | 94.674 | 95.726 | 95.727 |
| DBpedia 100:1 | 76.389 | 67.062 | 85.892 | 85.899 |

SMOTE-minus-ROS paired macro-F1 effects are **+0.149 percentage points** for Covertype, **+0.001** for DBpedia 10:1, and **+0.007** for DBpedia 100:1. Covertype's small interpolation effect survives the parametric six-comparison Holm test (adjusted p=0.000575), but its exact sign-flip adjusted p is 0.375. None of the six primary contrasts reaches 0.05 under the exact adjusted test. With only five paired seeds, the minimum unadjusted two-sided exact p is 0.0625.

The absolute controls matter: on Covertype, balanced ROS/SMOTE heads are substantially below natural classifier retraining on the same CB+EQ representation. On DBpedia, balancing the classifier helps under this protocol, but interpolation adds little over ROS. Equalization-versus-CB with natural heads gives -0.039, +0.002 and -0.020 percentage points, respectively; there is no consistent improvement. These are conditional experimental findings, not evidence of universal superiority or equivalence.

The 405 named readout evaluations come from 45 trained trunks and 135 retrained linear heads. They include 90 uniform-prior LA aliases with the same predictions as the corresponding ROS/SMOTE heads. The primary comparisons isolate streaming latent SMOTE versus ROS on the same CB+EQ representations and EQ versus matched CB training with natural classifier heads. NCM comparisons are exploratory representation evidence. Five seeds, fixed dataset splits and the six-comparison family constrain inference; neither the number of test records nor duplicate readout names creates additional independent fits.

## Verified data and splits

| Condition | Source rows | Fit rows | Calibration rows | Test rows | Smallest/largest fit class |
|---|---:|---:|---:|---:|---:|
| Covertype, natural imbalance | 581,012 | 406,704 | 87,150 | 87,158 | 1,922 / 198,310 |
| DBpedia, nominal 10:1 | 630,000 | 202,718 | 22,572 | 70,000 | 3,591 / 35,915 |
| DBpedia, nominal 100:1 | 630,000 | 119,551 | 13,310 | 70,000 | 359 / 35,915 |

The full source size is not the number of fitted examples. Covertype uses a fixed stratified 70/15/15 split after exact-predictor deduplication. No exact duplicates or conflicting-label duplicate groups were found. This is a newly specified large-training split, not the historical Covertype benchmark partition. Geographic dependence can remain; this is not a spatially independent evaluation.

DBpedia retains all 70,000 native test documents, with 5,000 examples per class. Normalized full title/content and nonempty content-only keys identify connected duplicate components. Components matching native test text are excluded from development without consulting test labels; conflicting development labels cause whole-component exclusion. The verified source excluded 19 development rows matching test text and 121 redundant development rows, across 96 development duplicate components, with no label conflicts. There are 559,860 canonical development rows. An independent check found no full/content-key overlap across fitted, calibration and native test partitions, and no full/content duplicate test rows. Paraphrases and related entities remain outside this exact-duplicate audit.

Preparation reserves 4,000 calibration candidates per DBpedia class and sets a common feasible maximum of 35,915 fit rows per class. Both fit and calibration pools receive the same rank-dependent thinning fraction. Class order is fixed by rank seed 2026; split seed is 1729. The nominal ratio uses `floor(n_max * rho ** (-rank / 13))`, so integer rounding changes the realized ratio slightly. Five optimization seeds share this fixed split and class order. A changed class order is a separate sensitivity.

Each prepared directory contains `metadata.json`, original-row labels, split indices and either numeric features or text records. Metadata includes source hashes, attribution, exclusions, actual class counts and preparation source identity. Source rows remain stored, but excluded rows are not fitted. Scalers, vocabulary, class priors and neighbor graphs use retained fit rows only. Outlier filtering is off in the primary protocol; `--ecod` is an explicit, separately recorded Covertype-only sensitivity.

## Sources and identities

[UCI Covertype](https://archive.ics.uci.edu/dataset/31/covertype), Blackard (1998), DOI 10.24432/C50K5N, lists CC BY 4.0. Preserve its attribution notice. The downloaded original gzip SHA256 is:

```text
614360d0257557dd1792834a85a1cdebfadc3c4f30b011d56afee7ffb5b15771
```

[DBpedia-14](https://huggingface.co/datasets/fancyzhx/dbpedia_14) is the classification benchmark of [Zhang, Zhao and LeCun (2015)](https://arxiv.org/abs/1509.01626). The dataset card records CC BY-SA 3.0 and GNU Free Documentation License terms; the original author's code license does not replace the text's license. This preparation used Hugging Face revision `9abd46cf7fc8b4c64290f26993c540b92aa145ac` and the following parquet hashes:

```text
train 0640e4664a99cc94c47db1d7b2e01c14455d5bbecb8183ad1f93bde59f3f28ee
test  05fed41640e97f93ffd442757f6a84170348cf0c7500ecbda9e95ddcd928c631
```

Acquisition records a pinned canonical source URL, file size and checksum. Signed redirect URLs are deliberately omitted. Future acquisition resolves the then-current repository revision; compare its revision and hashes above before claiming an identical source snapshot.

## Controlled comparison

All three trunks share initialization, natural training row order, architecture and optimization budget within a seed:

- **CE:** plain cross-entropy.
- **CB:** cross-entropy with deferred class-balanced weights.
- **CB+EQ / REBAL:** the same deferred weights plus the normalized-feature equalization objective.

CB and EQ activate after 80% of the epoch budget. The isolated equalization contrast is CB+EQ versus CB; the three-trunk design is not a complete two-factor loss ablation. Per-class counts of batches containing at least two examples are saved, along with whether EQ was active, so rare-class inactivity is visible.

Covertype standardizes its first ten numeric columns using fit statistics and leaves the 44 binary indicators unchanged. Its MLP is 54 -> 256 -> 256 -> 128 -> logits. DBpedia fits a 50,000-word vocabulary on condition-specific fit documents only. Text uses NFKC normalization, Unicode case folding, word tokens, title/SEP/content concatenation, and a 256-token cap. PAD, UNK and SEP are explicit; an empty document maps to UNK. Random 128-dimensional embeddings are mean-pooled and projected through 256 and 128 units. OOV and truncation counts are recorded per split and class. Both encoders use ReLU and dropout 0.1.

The full trunk plan is 50 epochs, batch size 1,024, five warmup epochs, initial learning rate 0.05, SGD with momentum 0.9 and Nesterov acceleration, linear-kernel L2 coefficient 0.0005, and per-tensor gradient norm clipping at 1. The preserved equalization coefficient is 0.05. Class-balanced weighting deliberately uses `beta = 1 - 1/n_max` and stable effective-number calculations; weights have expected mean one over fit examples. This large-data adaptation avoids the original beta=0.999 saturating at large class counts and is recorded in the protocol.

Every trained representation receives natural, random-oversampling and SMOTE linear readouts. All readouts share initialization and receive 2,000 updates of batch size 2,048 with SGD learning rate 0.01, momentum 0.9 and linear-kernel L2 coefficient 0.0005. Random oversampling is class-uniform streaming resampling of real fit vectors. SMOTE is a documented **streaming latent variant**: select a class uniformly, choose a fit anchor and a same-class nearest neighbor, and draw a fresh interpolation with probability `1 - n_class/n_max`. Maximal classes draw only real vectors. This is not a finite static imbalanced-learn synthetic pool. It never interpolates token IDs or raw categorical indicators.

Nearest-neighbor search is exhaustive and blocked, on retained fit representations only. Maximal classes need no graph construction. Graph checksums, source-feature hashes, runtime and allocation estimates are retained; compatible completed graphs can be reused after interruption. Blocking limits allocation but leaves quadratic distance work for oversampled classes. Pilot the largest required class before forecasting complete runtime. Raw learned feature geometry is shared across the three heads.

Nearest-class-mean and logit-adjusted readouts are also recorded. Logit adjustment uses each head's training prior. Balanced ROS/SMOTE heads have a uniform prior and fixed tau zero; their adjustment changes no predictions. For native CB/REBAL heads, the final weighted-objective prior is an explicit approximation to the changing training objective. Other tau values are selected only by calibration macro-F1. No evaluation score selects an epoch or checkpoint.

## Full experiment plan and inference

Run Covertype natural, DBpedia rho10 and DBpedia rho100 with seeds 0, 1, 2, 3 and 4. This is 15 condition/seed cells, 45 independently trained trunks, 135 trained classifier heads and 405 recorded readouts. LA/NCM readouts do not represent additional encoder training.

The primary endpoint is macro-F1. Two paired contrasts are declared for each of the three conditions: REBAL-SMOTE minus REBAL-ROS, and REBAL-natural minus CB-natural. The Holm family contains all six comparisons and is adjusted only when every declared seed and condition has completed. Report seed-level results, mean differences, uncertainty, exact sign-flip and paired t comparisons, together with accuracy, macro recall, per-class scores and confusion matrices. Frequency-rank groups contain 2/2/3 classes for Covertype and 4/5/5 for DBpedia, using floor boundaries. With five seeds, the minimum two-sided exact sign-flip p-value is 0.0625; a t-test depends on additional assumptions and more test rows do not create more independent training seeds.

## Reproduction commands

Run these commands from the repository root in a separate environment. The paths are portable examples; the separate virtual environment is outside the checkout. The existing image environment and evidence do not need modification.

```sh
python3 -m venv ../rebal-nonimage-venv
. ../rebal-nonimage-venv/bin/activate
python -m pip install -r requirements-nonimage.txt
python -m unittest discover -s tests -p 'test_nonimage.py' -v
python -m unittest discover -s tests -p 'test_verify_nonimage.py' -v

python scripts/prepare_nonimage.py acquire --dataset covertype --output data/nonimage/sources/covertype
python scripts/prepare_nonimage.py acquire --dataset dbpedia --output data/nonimage/sources/dbpedia
python scripts/prepare_nonimage.py prepare --dataset covertype --source data/nonimage/sources/covertype --output data/nonimage/covertype/natural
python scripts/prepare_nonimage.py prepare --dataset dbpedia --source data/nonimage/sources/dbpedia --output data/nonimage/dbpedia/rho10 --rho 10
python scripts/prepare_nonimage.py prepare --dataset dbpedia --source data/nonimage/sources/dbpedia --output data/nonimage/dbpedia/rho100 --rho 100
```

Acquisition and preparation reject an existing destination; retain the original data/metadata and choose a new destination for a different snapshot. The installed local validation environment used Python 3.12.14, torch 2.14.1, numpy 2.5.4, scipy 1.18.1, scikit-learn 1.9.1, Pillow 12.3.0 and pyarrow 26.0.0. `requirements-nonimage.txt` preserves the tested numeric/data versions but leaves torch unpinned so an existing device-compatible build can satisfy it. Actual versions and source identities are recorded in the protocol; do not assume every backend/build combination has been tested.

Start with a train-only throughput pilot in a separate output directory:

```sh
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -u scripts/train_nonimage.py run \
  --data data/nonimage/covertype/natural data/nonimage/dbpedia/rho10 data/nonimage/dbpedia/rho100 \
  --out experiments/nonimage_pilot_cpu --device cpu --knn-device cpu \
  --pilot-steps 50 --batch-size 256 --seeds 0
```

The pilot exercises training operators with immediate CB/EQ activation and discards weights. It does not encode or score calibration/test examples, run logit adjustment, construct kNN graphs, train readouts or produce final metrics. Permitted fit-only transforms can be applied to held-out rows during data preparation. Pilot throughput excludes exact kNN and several other stages, so it is not a full-run completion estimate. CPU fixtures passed end-to-end and exact interrupted-resume tests. The current CPU/MPS pilots passed for these prepared conditions; CUDA still requires a pilot on its selected environment. Unsupported operations fail rather than silently changing the requested backend. Deterministic-algorithm warn-only settings and known CUDA EmbeddingBag/neighbor-tie limitations are recorded; cross-device bitwise equality is not claimed.

The full CPU reference command is:

```sh
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -u scripts/train_nonimage.py run \
  --data data/nonimage/covertype/natural data/nonimage/dbpedia/rho10 data/nonimage/dbpedia/rho100 \
  --out experiments/nonimage_full_cpu --device cpu --knn-device cpu \
  --epochs 50 --warmup-epochs 5 --batch-size 1024 --lr 0.05 \
  --head-steps 2000 --head-batch-size 2048 --seeds 0 1 2 3 4
```

Use a separately validated CUDA run with `--device cuda --knn-device cuda` and a new output directory when an appropriate GPU is free. Different devices, budgets, conditions or sources require separate outputs. For parallel execution, initialize the same declared grid with `--aggregate-only`, then use distinct visible GPUs and `--worker-index 0`, `1`, or `2` with `--worker-count 3 --defer-aggregate`. Each worker receives disjoint condition/seed cells. A file lock protects shared transforms. After all workers finish, run one aggregator with the identical scientific arguments and `--aggregate-only`; do not run concurrent aggregators.

Keep source, dependencies and protocol settings unchanged for resume. Checkpoints, training histories, split identities, transforms, graph hashes, head states and logits are retained. Aggregate-only validates both result artifacts and fitted transforms. Only validated complete cells enter the final result table. Raw datasets, private paths, credentials and operational connection records are not publication artifacts.


## Independent verification

After every declared cell completes, aggregate once with the original scientific arguments and `--aggregate-only`. Then recompute the saved metrics, calibration choices, summary statistics and all six contrasts using the independent verifier:

```sh
python scripts/verify_nonimage.py \
  --data data/nonimage/covertype/natural data/nonimage/dbpedia/rho10 data/nonimage/dbpedia/rho100 \
  --run experiments/nonimage_full_cpu \
  --sources-root data/nonimage/sources \
  --out experiments/nonimage_full_cpu/independent_verification.json
```

Require `verified_complete`, 15 completed and expected cells, 405 verified evaluations, no missing cells, and `aggregate_status` equal to `independently recomputed and verified`. Verify that the aggregate hash still matches the current file. A partial or pilot run is not a completed experiment. The full verifier checks saved model/graph bytes but does not rerun model inference or reconstruct exact-neighbor optimality. Its 11 portable synthetic and tamper checks exercise the checking software separately from the real study.

The portable prediction-evidence archive excludes raw datasets, model/optimizer states, token caches/vocabulary and graph arrays. Its included checker can rehash the packaged evidence and recompute test metrics, but cannot repeat full audits of omitted bytes. Original artifact inventories preserve the hashes of excluded files. Do not present the reduced archive as a complete training-state backup.

## Method sources and adaptations

Effective-number weighting follows [Cui et al. (CVPR 2019)](https://arxiv.org/abs/1901.05555); separating representation and classifier learning follows [Kang et al. (ICLR 2020)](https://arxiv.org/abs/1910.09217). This study changes CB beta and normalization for larger class counts and uses its declared deferred schedule. The repository-specific normalized-feature EQ objective is separately under test.

Latent streaming interpolation adapts [Chawla et al. (JAIR 2002)](https://arxiv.org/abs/1106.1813), whose arXiv copy dates from 2011. Post-hoc adjustment follows [Menon et al. (ICLR 2021)](https://arxiv.org/abs/2007.07314) with head-specific priors and calibration-only tau selection. The text encoder is related to embedding-average classifiers such as [Joulin et al. (EACL 2017; preprint 2016)](https://aclanthology.org/E17-2068/), but adds nonlinear layers and is not an exact fastText implementation. It does not execute the character-level CNN in the original DBpedia benchmark paper.
