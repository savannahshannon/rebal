# Prospective vision follow-up

This study compares a frozen DeiT-tiny classifier, partial DeiT-tiny fine-tuning,
and a stronger frozen DINOv2 ViT-S/14 encoder. Each encoder is evaluated with
ordinary cross-entropy and Balanced Softmax. Its six arms are GAN-free. The
study follows the earlier frozen-feature REBAL comparison; it does not replace
the earlier results or claim a completed remote end-to-end training run.

The runner is `scripts/benchmark_vision_followup.py`. The protocol and source
fingerprints, split identities, extraction receipts, development predictions,
training histories, calibration curves and confirmation predictions must be
retained with each execution. A report is an experimental result only after
every planned final fit and independent verification have completed.

## Data and selection

Tiny ImageNet supplies 100,000 development images and 10,000 labeled official
evaluation images. The development source is deduplicated before splitting.
The same immutable partition used by the EQ factorial study contains:

| Partition | Images | Role |
| --- | ---: | --- |
| Imbalanced fit | 15,268 | Gradient updates and feature standardization |
| Tuning | 4,000 | Learning-rate and checkpoint selection |
| Calibration | 4,000 | Logit-offset selection |
| Confirmation | 20,000 | Primary evaluation after all configurations freeze |
| Official evaluation | 10,000 | Secondary exploratory evaluation |

The fit set is thinned from a separate 71,973-image fit pool. The realized
largest-to-smallest class count is 355/3 = 118.33, despite the nominal thinning
ratio of 100. Each confirmation class has 100 images. Head, middle and tail
groups follow the fit class-count order, with 66, 67 and 67 classes respectively.
`class_order` maps rank to class; `class_rank` is its inverse.

Tuning seed 901 evaluates two learning rates per arm. Final seeds 10 through 19
are paired across arms. At each epoch, the calibration set selects the CE
offset on the fixed grid 0 to 2 in steps of 0.05; the tuning set scores the
adjusted predictions. The best tuning epoch and learning rate are selected
without confirmation labels or official-evaluation outcomes. Calibration is
therefore development data, rather than an untouched final test set.

The new confirmation rows were held out from the current round. Historical
models may have trained on those source rows. This is a follow-up on the same
dataset, not an external replication. Ten seeds measure optimization
variability on this fixed partition rather than uncertainty across datasets.

The exact DeiT-tiny weight file is the official model-zoo checkpoint pretrained
on ImageNet 2012. Tiny ImageNet derives from ImageNet; no item-level exclusion
against the DeiT pretraining corpus is established here. Current-round split
disjointness therefore does not establish independence from pretraining images.
These results describe transfer pipelines on this source, and do not estimate
unseen-pretraining generalization. See the [official DeiT model zoo](https://github.com/facebookresearch/deit/blob/main/README_deit.md)
and the [Stanford Tiny ImageNet source discussion](https://cs231n.stanford.edu/reports/2016/pdfs/411_Report.pdf).

## Models and fixed budgets

All images use RGB conversion, bicubic resize to 256, center crop to 224, and
ImageNet normalization. No new stochastic image augmentation is introduced.
The original pretrained weights remain fixed in the encoder controls.

| Encoder | Trainable layers | Classifier input |
| --- | --- | ---: |
| DeiT-tiny frozen | Linear classifier only | 192 |
| DeiT-tiny partial | Last two transformer blocks, final norm and classifier | 192 |
| DINOv2 ViT-S/14 frozen | Linear classifier only | 384 |

The first ten DeiT blocks are frozen in evaluation mode and cached as FP16
token prefixes. A train-only, 256-image gate covers all 200 classes and checks
the reconstructed features against full precision: mean absolute error at
most 0.001 and maximum absolute error at most 0.02. Frozen and partial DeiT use
the same fit-derived feature scaler and paired classifier initialization.
DINO uses its own fit-derived scaler.

The final DeiT block evaluates the CLS query against all keys and values, then
computes only the CLS residual and feed-forward output. DeiT's relevant dropout
is zero. The required outputs and gradients are checked against the complete
block, and the GPU pilot separately checks numerical equivalence. This changes
computation without changing the class-token function.

Every tuning and final fit runs 15 epochs, batch size 128, and 120 updates per
epoch: 1,800 optimizer updates per fit. AdamW uses weight decay 0.01, a 10%
warmup, cosine decay, and global gradient clipping at 1. Frozen classifier
learning rates are 0.001 and 0.003. Partial encoder learning rates are 0.00001
and 0.00003, with a classifier learning rate 100 times the encoder rate. There
are 12 tuning fits and 60 final fits. Validation-selected checkpoints do not
reduce the completed training budget.

Balanced Softmax trains with `CE(logits + log(class_counts), labels)`. CE uses
the empirical fit prior for post-hoc logit adjustment. Balanced Softmax uses
the uniform target prior, so its adjusted readout repeats its raw predictions.
The 180 named final readouts include 30 such aliases; they are not 180
independent training fits.

## Prespecified inference

The primary endpoint is calibrated confirmation macro-F1. The five primary
paired contrasts are:

1. Partial DeiT CE minus frozen DeiT CE.
2. Frozen DINO CE minus frozen DeiT CE.
3. Balanced Softmax minus CE within frozen DeiT.
4. Balanced Softmax minus CE within partial DeiT.
5. Balanced Softmax minus CE within frozen DINO.

The analysis retains all ten paired differences, mean effects, sample standard
deviations, paired-t 95% intervals, exact two-sided sign-flip p-values, and Holm
adjustments across these five comparisons. A global sensitivity analysis
adjusts the 22 primary comparisons across all four new studies. Tail F1,
accuracy, class recall, runtime and official-evaluation scores are secondary.
A one percentage-point macro-F1 effect is a practical reference, not a threshold
for declaring a method universally preferable.

DINOv2 and DeiT differ in pretraining, width, capacity and patch size. Their
contrast is an encoder-package comparison, not evidence isolating architecture
or pretraining alone. Partial-versus-frozen DeiT additionally changes trainable
parameters and optimization; it does not evaluate full end-to-end fine-tuning.

## Method references

- [Balanced Softmax, Ren et al.](https://arxiv.org/abs/2007.10740)
- [Logit adjustment, Menon et al.](https://arxiv.org/abs/2007.07314)
- [DeiT, Touvron et al.](https://arxiv.org/abs/2012.12877)
- [DINOv2, Oquab et al.](https://arxiv.org/abs/2304.07193)

Dense feature caches, pretrained weights, model/optimizer states, raw source
images and private operational logs are excluded from Git publication. Public
evidence retains their cryptographic identities, along with sufficient
predictions and labels to replay the reported metrics and statistical tests.
