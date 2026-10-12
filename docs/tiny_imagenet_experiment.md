# Tiny ImageNet transfer experiment - October 12, 2026

This extension tests the classifier-rebalancing part of REBAL on a larger image benchmark. It is **not an end-to-end REBAL-F reproduction**. A shared, frozen ResNet18 pretrained on ImageNet supplies the features. Because Tiny ImageNet is a subset of ImageNet, pretraining may overlap the source images/classes. This study cannot establish clean from-scratch generalization, improved learned representations, equalization effects, or cGAN harm.

The [full experimental report](../results/tiny_imagenet_transfer/experimental_report.pdf), [per-seed results](../results/tiny_imagenet_transfer/per_seed_results.csv), [source audit](source_audit_2026-10-12.md), and [saved study plan](../results/tiny_imagenet_transfer/study_plan.json) accompany the runnable code. The study plan was saved before any classifier evaluation. Existing CIFAR aggregate tables are preserved and were not independently rerun.

## Completed experiments

- Official Tiny ImageNet: 100,000 source training images, 200 classes, 64 x 64 pixels; official labeled validation (10,000 images) used only for final evaluation. The unlabeled official test partition was not evaluated.
- Decoded-RGB hash audit: seven train/evaluation duplicates and 20 duplicate training rows excluded before splitting, leaving 99,973 eligible training rows. This detects exact pixels, not near duplicates or pretraining overlap.
- A balanced internal holdout of 50 images per class, drawn only from training. It supplies additional tail calibration labels and must be considered when interpreting logit-adjustment results.
- Five paired seeds (0-4), nominal imbalance ratios 10, 50, 100, and two fixed budgets: primary 4,000 updates and descriptive sensitivity 1,000 updates.
- Seven methods: natural CE, CE + logit adjustment, effective-number class-balanced CE, random oversampling, feature SMOTE, SMOTE + logit adjustment, and cosine nearest-class-mean.
- Four learned heads per cell: **120 trained linear classifiers and 210 method evaluations across 30 cells**. LA reuses an existing head; NCM has no optimization updates.

| Nominal ratio | Real fitting images | Min/max per class | Actual ratio |
| --- | ---: | ---: | ---: |
| 10 | 34,765 | 44/445 | 10.113636 |
| 50 | 22,309 | 8/445 | 55.625 |
| 100 | 19,168 | 4/445 | 111.25 |

These are long-tailed subsets of the larger dataset; no individual classifier uses all 100,000 source images. All final evaluations use all 10,000 held-out images. Seeds randomize semantic class-frequency ranks, and the subsets are nested across ratios for a given seed. Head/middle/tail are rank thirds of 66/67/67 classes.

## Primary outcome

At 4,000 updates, mean tail macro-F1 (percent) is:

| Nominal ratio | Feature SMOTE | Random oversampling | CE + logit adjustment |
| --- | ---: | ---: | ---: |
| 10 | 33.90 | 33.19 | 34.89 |
| 50 | 20.63 | 20.80 | 26.81 |
| 100 | 13.92 | 13.92 | 22.19 |

The observed SMOTE-minus-ROS differences are +0.711, -0.169, and +0.001 percentage points. None passes Holm-adjusted paired t testing within the six prespecified tail-F1 comparisons. This does **not** establish equivalence. The SMOTE-minus-CE+LA differences are -0.990, -6.176, and -8.271 points. Parametric paired t evidence favors CE+LA, but with five seeds the exact two-sided sign-flip test cannot attain p < 0.05: its minimum is 2/32 = 0.0625 before correction. Treat inference as exploratory and conditional on this fixed encoder and evaluation set.

At the shorter budget, every trained method has higher mean top-1 accuracy at every ratio. At ratio 100, SMOTE tail F1 is 15.93% at 1,000 updates versus 13.92% at 4,000. Both budgets are reported; neither is claimed optimal or converged. The complete aggregate JSON/CSV files include all methods, sample SD, t intervals, paired seed differences, raw/adjusted t p-values, exact sign-flip p-values, and their separate Holm correction families.

## Reproduction

Use a **separate Python 3.12 environment** from the original TensorFlow pipeline:

```bash
python3 -m venv .venv-tiny
source .venv-tiny/bin/activate
pip install -r requirements-tiny.txt
mkdir -p data
curl -L --fail https://cs231n.stanford.edu/tiny-imagenet-200.zip -o data/tiny-imagenet-200.zip
python scripts/extract_tiny_features.py --archive data/tiny-imagenet-200.zip --out data/tiny_resnet18_64_features.npz --device mps
python scripts/benchmark_tiny_features.py --cache data/tiny_resnet18_64_features.npz --out experiments/tiny_primary_4000 --steps 4000 --device cpu
python scripts/benchmark_tiny_features.py --cache data/tiny_resnet18_64_features.npz --out experiments/tiny_sensitivity_1000 --steps 1000 --device cpu
python scripts/verify_tiny_experiments.py --root experiments/tiny_primary_4000 --cache data/tiny_resnet18_64_features.npz --out experiments/verify_primary.json --expected-steps 4000 --require-complete
python scripts/verify_tiny_experiments.py --root experiments/tiny_sensitivity_1000 --cache data/tiny_resnet18_64_features.npz --out experiments/verify_sensitivity.json --expected-steps 1000 --require-complete
python -m unittest discover -s tests -v
```

The extraction script also supports `--device cpu` or `--device cuda`; classifier training supports CPU or MPS. The recorded run used MPS for extraction and deterministic CPU computation for linear heads. Native 64-pixel preprocessing is an explicit adaptation of the pretrained model, not its canonical 224-pixel crop. The archive, weights, features, source and software versions are identified in `results/tiny_imagenet_transfer/feature_provenance.json` and the protocol files. A regenerated cache on another platform may have a different numerical hash; resume correctly rejects mixing it with old runs.

All learned heads use SGD 0.01, momentum 0.9, weight decay 0.0005, batch size 512, identical seed-specific initialization and matched update counts. The fixed learning rate and final weights are used without early stopping. ROS/SMOTE produce 89,000-feature pools. Training samples are drawn uniformly with replacement. SMOTE uses raw features and `k=min(5,min_count-1)`. CB uses beta 0.999 and the original mean of per-example weighted losses; this also changes gradient scale relative to regularization. LA selects tau from `[0,.25,.5,.75,1,1.25,1.5,2]` on internal-holdout macro recall only. Ties favor smaller tau. NCM uses normalized real features and normalized class centroids.

Completed runs preserve exact split indices, filenames/pixel-hash manifest, source/protocol hashes, classifier parameters, prediction/logit arrays, loss checkpoints, per-class metrics, and tau curves. Resume verifies saved evidence and rejects source/protocol changes. Aggregate output contains metadata, not independent run replications for each test image.

## Validation and next research step

Twenty-six tests passed. Independent scikit-learn/scipy verification reproduced all 5,040 scalar metrics and all per-class arrays from saved predictions, along with the aggregates, intervals, paired tests, sign-flip tests, and Holm corrections. Both budgets have identical splits and NCM predictions. Optimizer-update counts were reconciled with stored records; training itself was not replayed by the independent verifier.

The next decisive study should repair the open source-audit findings, train matched backbones from scratch, isolate effective-number weighting and equalization, and compare identical NCM/cRT decisions on those backbones. More paired seeds and reduced/naturally imbalanced calibration sets would strengthen interpretation. These experiments have not been performed here.

Primary sources: [Tiny ImageNet](https://cs231n.stanford.edu/reports/2015/pdfs/yle_project.pdf), [SMOTE](https://arxiv.org/abs/1106.1813), [class-balanced loss](https://arxiv.org/abs/1901.05555), [decoupled classifier training](https://arxiv.org/abs/1910.09217), [logit adjustment](https://arxiv.org/abs/2007.07314), and [torchvision ResNet18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html).
