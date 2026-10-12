# GAN-free ViT features with REBAL: matched image comparison

## Status and question

Protocol specified on October 12, 2026, before inspecting results from this comparison. Implementation and execution are in progress; no ViT performance result is claimed yet. The user selected a ViT feature encoder with REBAL, rather than a ViT image generator.

The question is whether a frozen ViT encoder improves the performance of a trained REBAL feature adapter and its classifiers relative to a frozen convolutional encoder under the same downstream conditions. The primary ViT is the official **non-distilled DeiT-tiny/16**, pretrained on ImageNet-1K, with 192-dimensional class-token features. The convolutional comparator is torchvision ResNet18 with `IMAGENET1K_V1` weights and 512-dimensional pooled features.

Both paths are GAN-free. This is a new encoder comparison, not a causal estimate of replacing a GAN: a GAN synthesizes images, whereas these encoders extract features. Existing repository GAN results concern a different CIFAR experiment and remain historical context. They cannot be subtracted from these Tiny ImageNet results to estimate a GAN effect. The separate from-scratch remote ResNet32 study and earlier frozen ResNet18/64px transfer study also remain distinct experiments.

## Data and preprocessing

Use the same original Tiny ImageNet archive as the earlier larger-data study: 100,000 official training images, 10,000 official validation images, and 200 classes. The official validation partition is the final evaluation partition. Reuse the audited raw-pixel duplicate policy: remove training images overlapping final evaluation pixels, then retain the first remaining copy of each training image. The original study retained 99,973 training images. Recheck archive bytes, raw-pixel hashes, labels, class order and filenames when creating both feature caches.

Both encoders receive RGB images resized to 256 pixels with bicubic interpolation, center-cropped to 224 pixels, converted to [0,1], and normalized with ImageNet mean (0.485, 0.456, 0.406) and standard deviation (0.229, 0.224, 0.225). There is no random augmentation in frozen feature extraction. This shared transform follows DeiT's evaluation convention; bicubic interpolation intentionally differs from torchvision ResNet18's documented bilinear preset. The comparison controls preprocessing rather than claiming that each model uses its individually optimal transform. Both encoders must be recached at this resolution; the prior native64 ResNet cache is not the primary comparator.

Patch size16 at224 gives196 image patches for DeiT. Native64 would give16 patches and require positional-grid adaptation, so it is not silently substituted to save compute. No positional interpolation is required for the primary224 run.

For each seed0–4, reserve50 images per class from deduplicated training for calibration, randomize semantic class ranks using the previously audited independent RNG streams, and create nested rho10/50/100 fit sets. The largest class has445 examples; counts are `floor(445 * rho ** (-rank / 199))`. Use exactly the same row IDs and rank assignments for both encoders and all objectives. Frequency regions contain66/67/67 classes.

ImageNet pretraining can overlap Tiny ImageNet. Removing duplicates between the local fit/calibration/evaluation partitions does not eliminate pretraining leakage. This remains a transfer study, not a clean unseen-image generalization benchmark.

## Trainable components and controls

Keep both pretrained encoders frozen and in evaluation mode. Within each encoder/seed/ratio cell, standardize each feature dimension using fit rows only, then train an adapter `input_dimension -> 256 -> 128` with ReLU and dropout0.1 and a linear200-class native classifier. The encoder feature dimensions, encoder parameter counts, original pretraining recipes and input projection parameter counts differ and must be reported. This is a representation-adapter adaptation of REBAL; the EQ objective trains the adapter, not the frozen ViT.

Use three matched adapter objectives: cross-entropy (CE), deferred class-balanced weighting (CB), and deferred CB plus the repository's normalized-feature equalization loss (CB+EQ, called REBAL in the result identifiers). Within one encoder cell, all objectives share initialization, fit-row order and optimization budget. Across encoders the input projections have different shapes, so identical complete adapter initialization cannot be claimed. The128D classifier heads use shared initialization and the same random seed convention across encoders.

The fixed adapter budget is50 epochs, five warmup epochs, batch1024, initial learning rate0.05, cosine schedule, Nesterov SGD with momentum0.9, linear-weight L2 coefficient0.0005 and per-tensor gradient clipping at1. CB and EQ start after80% of the epochs; EQ coefficient is0.05. CB uses stable effective-number weights with `beta = 1 - 1/n_max`, normalized to expected fit-example weight1. Primary outlier filtering is off. Retain per-class EQ batch eligibility counts.

For each trained adapter, freeze its128D representation and train natural-sampling, class-uniform random-oversampling (ROS), and streaming latent-SMOTE linear classifiers. Each receives2,000 updates of batch2048, SGD learning rate0.01, momentum0.9 and linear-weight L2 coefficient0.0005. SMOTE uses exact same-class neighbors, excluding self, and fresh interpolations with probability `1 - n_class/n_max`. Maximal classes receive real samples only. These latent vectors are not generated images.

Record nearest-class-mean (NCM) and calibration-selected logit adjustment (LA) as well. Balanced ROS/SMOTE classifiers have a uniform training prior and tau0, so their LA views repeat the unadjusted predictions. Native CB/REBAL classifiers use the final weighted-objective prior as an explicitly approximate correction for deferred weighting. Epochs, budgets and architecture must not be selected using final evaluation scores.

## Declared grid and inference

Two encoders, three imbalance ratios and five seeds yield30 cells,90 trained adapters,270 retrained classifiers and810 named readout evaluations. Repeated LA views are not independent fits. Five seeds vary both the optimization and the shared seed-specific split/rank construction, unlike the fixed-split non-image study.

Macro-F1 is the primary endpoint. Declare one15-comparison Holm family before scoring:

- DeiT REBAL-SMOTE minus ResNet REBAL-SMOTE at each ratio: three comparisons.
- REBAL-SMOTE minus REBAL-ROS within each encoder and ratio: six comparisons.
- REBAL-natural minus CB-natural within each encoder and ratio: six comparisons.

Report paired seed differences, mean effects,95% paired t intervals, exact two-sided sign-flip tests and paired t tests with their assumptions. Correct each test type over the complete15-member family only after all30 cells finish. With five pairs the minimum two-sided exact sign-flip p-value is0.0625, so an exact p<0.05 is unattainable. Accuracy, region/per-class metrics, NCM and comparisons outside this family are descriptive or exploratory. Correlated ratios do not constitute independent datasets.

Record extraction and downstream runtime, frozen versus trainable parameter counts, weight/source/software hashes, split/scaler identities, optimization histories, checkpoints, exact neighbor graphs, raw predictions and calibration selections. Independent recomputation and privacy review must precede report publication. Credentials, local account paths, remote connection records, raw source images and model download caches are excluded from Git publication.

## Reproduction setup and feature extraction

Use a separate Python3.12 environment. Install `requirements-vit.txt`, which adds the reviewed `timm==1.0.30` implementation to the pinned image-study environment. The actual model files are downloaded from the official PyTorch and DeiT hosts; no remote Python implementation is executed. The extractor requires the validated torch2.14.1/torchvision0.29.1 snapshot and loads only tensor checkpoint data. Both official pretrained classifiers are removed before feature extraction.

The extractor expects the original full source-row manifest, which accompanies the earlier evidence archive. A fresh manifest can instead be rebuilt with the original `scripts/extract_tiny_features.py`; that also recomputes the older native64 ResNet cache for provenance and does not make it the comparator. New manifests include execution details, so their byte hash can differ across reproductions even when the raw-image identities agree. Preserve each execution's own identities rather than editing them to imitate the published hashes.

From the repository root, after placing the verified source archive at `data/tiny-imagenet-200.zip`:

```sh
python3 -m venv ../rebal-vit-venv
. ../rebal-vit-venv/bin/activate
python -m pip install -r requirements-vit.txt
python -m unittest discover -s tests -p 'test_vit*.py' -v
python -m unittest discover -s tests -p 'test_verify_vit_rebal.py' -v

# Skip this command when the original full source-row manifest is available.
TORCH_HOME=data/torch_cache python scripts/extract_tiny_features.py \
  --archive data/tiny-imagenet-200.zip --out data/tiny_resnet18_64_features.npz \
  --device mps --batch-size 256 --workers 0

python scripts/extract_vit_comparison_features.py acquire --weights-dir data/vit_weights
python scripts/extract_vit_comparison_features.py pilot \
  --archive data/tiny-imagenet-200.zip --original-manifest data/tiny_resnet18_64_features.manifest.json \
  --weights-dir data/vit_weights --out experiments/vit_extraction_pilot.json \
  --device mps --batch-size 64 --workers 0 --threads 1
python scripts/extract_vit_comparison_features.py extract \
  --archive data/tiny-imagenet-200.zip --original-manifest data/tiny_resnet18_64_features.manifest.json \
  --weights-dir data/vit_weights --out data/vit_comparison_224 \
  --device mps --batch-size 64 --workers 0 --threads 1
```

Select `--device cpu` or a separately validated CUDA environment when MPS is unavailable; do not silently change the backend of a partial cache. Extraction resumes only with the same source, code, weights, preprocessing and runtime plan, and rechecks every committed feature block. Cached vectors contain all110,000 source images in the original order. Final manifests are written only after both arrays are complete and hashed. Acquisition refuses to overwrite an incomplete weight directory; preserve it for investigation and choose a fresh destination.

The adapter runner is `scripts/benchmark_vit_rebal.py`. Its `--pilot-steps` mode is restricted to fitted training rows and does not evaluate calibration/test images, construct neighbors or save benchmark scores. `scripts/verify_vit_rebal.py` independently checks the final30-cell production study, including raw predictions, calibration choices, fit-only scalers, reconstructed splits, optimization budgets and the15-comparison family. Device and measured-runtime details will be added after the pilots and complete run are verified.

## Sources

- Touvron et al. (2021), [Training data-efficient image transformers & distillation through attention](https://proceedings.mlr.press/v139/touvron21a.html), and the [official DeiT repository](https://github.com/facebookresearch/deit). The non-distilled model is used; the paper's distillation contribution is not executed here.
- [Torchvision ResNet18 weights and documented preprocessing](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html).
- The repository's [non-image methodology](nonimage_experiment.md) documents the CB, decoupled classifier, SMOTE and LA adaptations shared by the downstream adapter comparison.
- The repository's [earlier Tiny ImageNet transfer study](tiny_imagenet_experiment.md) documents source acquisition and the reused split and duplicate controls.
