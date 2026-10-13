# Equalization and strong-control follow-up: verified experimental report

The full study completed 2026-10-12T21:05:17Z: 320 tuning adapters, 110 selected/reference native slots, 220 retrained heads, and 660 named readouts. Actual checkpoint states, budgets, saved logits, calibration curves and every reported metric were independently checked. Native reference slots and logit-adjustment views can be exact aliases; readouts are not independent experiments.

## Data and protocol

The official cached DeiT-tiny/16 encoder supplies fixed 192-dimensional features of Tiny ImageNet images at 224 pixels. This experiment trains adapters and classifiers on those frozen vectors. It does not train the image backbone. After the existing pixel-level deduplication policy, 99,973 development images were divided once, before long-tail thinning, into a 71,973-image fit pool, 4,000 tuning images, 4,000 calibration images, and 20,000 confirmation images. Every held-out class has 20 tuning, 20 calibration and 100 confirmation examples. The long-tail fit contains 15,268 images across 200 classes; class support ranges from 355 to 3 (nominal rho 100, realized maximum/minimum 118.33).

A fixed seed 20261013 defines data membership and frequency order. Ten optimization seeds 10–19 use the identical fit and holdout rows. Calibration is development selection data: CE candidates choose an offset on calibration rows and are scored on tuning rows after adjustment. All 320 training/tuning records were complete before global candidate selection. The six-contrast analysis amendment was frozen earlier, while tuning was incomplete and before any confirmation predictions or scores existed.

These are new confirmation rows for this follow-up, but historical models may have trained on these source images. The pretrained encoder also has external ImageNet pretraining. This is a protected same-source follow-up, not a fully independent external-data replication. Seed intervals quantify optimization variability on one fixed partition, not variation across datasets, source partitions or pretraining.

## Treatments and selection

The sampling factorial holds the expected class-balanced cross-entropy target fixed from epoch 1. Natural uniform-row replacement sampling uses N/(C n_c) importance weights; class-uniform replacement sampling uses weight1. Each arm receives 50 epochs, 15 full 1,024-example batches per epoch and 750 optimizer updates. The common expected cross-entropy target does not imply the same expected SGD trajectory: minibatch composition, gradient variance, clipping, evolving momentum and EQ eligibility can differ.

Fixed EQ uses onset 0.8 and coefficient 0.05. Adaptive EQ chooses from onset {0.2,0.5,0.8} × coefficient {0.01,0.05}, separately for each sampler. CE and BalancedSoftmax each receive six LR candidates {0.01,0.025,0.05,0.075,0.1,0.2}. The BCL adaptation receives six coefficient/temperature candidates {0.05,0.2,0.5} × {0.07,0.1}, at fixed LR 0.05. Every candidate is evaluated on the mean tuning macro-F1 of the same 10 seeds. This is more development tuning than single-seed selection; the off factorial controls retain fixed LR 0.05.

BCL implements class-averaged denominators, class complement with a classifier-weight prototype for every class, and two independent adapter-dropout views of 64 examples per update. It adds this loss to BalancedSoftmax. Cached frozen vectors provide no image augmentation, and this is not the original full-backbone, image-augmentation, two-projection-branch BCL reproduction. [Original BCL paper](https://openaccess.thecvf.com/content/CVPR2022/html/Zhu_Balanced_Contrastive_Learning_for_Long-Tailed_Visual_Recognition_CVPR_2022_paper.html) and [official loss implementation](https://github.com/FlamieZhu/Balanced-Contrastive-Learning/blob/main/loss/contrastive.py).

Natural and random-oversampling classifiers each receive 2,000 updates on frozen adapter vectors. No SMOTE graph was added in this round. Final offsets use the fixed tau grid 0,0.25,0.5,0.75,1,1.25,1.5,2; class-balanced, BalancedSoftmax, BCL and ROS target priors are uniform, so their offset readouts are exact prediction aliases.

Both selected CE and Balanced Softmax choose LR 0.2, the upper edge of the declared grid. These are the best observed candidates within the frozen grid, and no global-optimum claim is made. BCL retains fixed LR 0.05 while selecting its loss coefficient and temperature.

Selected configurations:

- balanced: `{"coefficient": 0.05, "objective": "class_balanced", "onset": 0.5, "sampler": "balanced"}`
- balanced_softmax: `{"coefficient": 0.0, "objective": "balanced_softmax", "onset": 0.0, "peak_lr": 0.2, "sampler": "natural"}`
- bcl_dropout: `{"coefficient": 0.05, "objective": "bcl_dropout", "onset": 0.0, "sampler": "natural", "temperature": 0.07}`
- ce: `{"coefficient": 0.0, "objective": "ce", "onset": 0.0, "peak_lr": 0.2, "sampler": "natural"}`
- natural: `{"coefficient": 0.05, "objective": "class_balanced", "onset": 0.8, "sampler": "natural"}`

## Confirmation results

Values below are percentage points, mean ± sample SD across 10 seeds. The native+LA column is the endpoint used for the selected strong controls; the fixed factorial uses native logits (equivalent to LA for its uniform prior). The full CSV includes every readout and 95% intervals.

| Method / adapter slot | Native+LA macro-F1 | Native+LA tail-F1 | Natural head+LA macro-F1 | ROS head macro-F1 |
|---|---:|---:|---:|---:|
| Selected CE+LA | 56.68 ± 0.10 | 46.40 ± 0.36 | 55.66 ± 0.27 | 52.34 ± 0.24 |
| Selected BalancedSoftmax+LA | 55.77 ± 0.23 | 47.56 ± 0.47 | 55.80 ± 0.29 | 52.44 ± 0.26 |
| Selected dropout-view BCL+LA | 50.87 ± 0.31 | 35.62 ± 0.57 | 54.29 ± 0.24 | 52.81 ± 0.21 |
| Natural sampling, EQ off | 53.75 ± 0.19 | 46.37 ± 0.40 | 56.75 ± 0.18 | 52.57 ± 0.27 |
| Natural sampling, fixed EQ | 53.75 ± 0.19 | 46.37 ± 0.40 | 56.71 ± 0.20 | 52.57 ± 0.27 |
| Natural sampling, selected EQ | 53.75 ± 0.19 | 46.37 ± 0.40 | 56.71 ± 0.20 | 52.57 ± 0.27 |
| Balanced sampling, EQ off | 53.75 ± 0.21 | 46.31 ± 0.40 | 56.75 ± 0.14 | 52.59 ± 0.25 |
| Balanced sampling, fixed EQ | 53.75 ± 0.21 | 46.31 ± 0.40 | 56.75 ± 0.14 | 52.59 ± 0.25 |
| Balanced sampling, selected EQ | 53.74 ± 0.22 | 46.30 ± 0.40 | 56.73 ± 0.15 | 52.58 ± 0.25 |
| CE reference LR 0.05 + LA | 53.19 ± 0.30 | 37.14 ± 0.59 | 54.24 ± 0.16 | 52.91 ± 0.22 |
| BalancedSoftmax reference LR 0.05 + LA | 52.03 ± 0.26 | 37.87 ± 0.48 | 54.64 ± 0.20 | 52.99 ± 0.19 |

## Prespecified paired primary inference

The six-member family contains four fixed-treatment factorial effects plus two selected strong-control pipeline comparisons. Positive values favor the first method. Intervals are paired 95% t intervals; inference uses two-sided exact sign flips over 10 paired seeds with Holm correction. BCL versus BalancedSoftmax can use different selected learning rates, so it compares the final policies rather than isolating the BCL loss at a common LR.

| Contrast | Mean difference (pp) | 95% paired interval (pp) | Exact p | Holm p (6) |
|---|---:|---:|---:|---:|
| Natural sampling: fixed EQ effect | +0.000 | [-0.014, +0.014] | 0.962891 | 1.000000 |
| Balanced sampling: fixed EQ effect | +0.002 | [-0.008, +0.011] | 0.625000 | 1.000000 |
| Balanced − natural, EQ off | -0.000 | [-0.102, +0.102] | 1.000000 | 1.000000 |
| Fixed EQ × sampling interaction | +0.002 | [-0.015, +0.018] | 0.830078 | 1.000000 |
| Selected BS+LA − selected CE+LA | -0.907 | [-1.088, -0.725] | 0.001953 | 0.011719 |
| Selected BCL+LA − selected BS+LA | -4.904 | [-5.121, -4.687] | 0.001953 | 0.011719 |

The adaptive-selected EQ effects below are secondary. They use different validation-selected treatments across samplers and must not be called a fixed-treatment interaction. The original runner aggregate is retained for provenance; its former four-comparison family is superseded for primary inference.

- selected_natural_EQ_pipeline_effect: +0.000 pp; paired 95% interval [-0.014, +0.014] pp; uncorrected exact p=0.962891.
- selected_balanced_EQ_pipeline_effect: -0.010 pp; paired 95% interval [-0.039, +0.018] pp; uncorrected exact p=0.337891.

## Mechanism and practical interpretation

With natural sampling, EQ sees at least two members of a tail class in 10.59% of batches on average, compared with 96.37% under class-uniform sampling. Exact sampler replay verifies this coverage difference. It identifies a training mechanism; confirmation contrasts determine whether it improves recognition.

- Natural sampling: fixed EQ effect: mean +0.000 pp and does not pass the prespecified Holm-adjusted exact test. A nonsignificant result does not establish equivalence.
- Balanced sampling: fixed EQ effect: mean +0.002 pp and does not pass the prespecified Holm-adjusted exact test. A nonsignificant result does not establish equivalence.
- Balanced − natural, EQ off: mean -0.000 pp and does not pass the prespecified Holm-adjusted exact test. A nonsignificant result does not establish equivalence.
- Fixed EQ × sampling interaction: mean +0.002 pp and does not pass the prespecified Holm-adjusted exact test. A nonsignificant result does not establish equivalence.
- Selected BS+LA − selected CE+LA: mean -0.907 pp and passes the prespecified Holm-adjusted exact test.
- Selected BCL+LA − selected BS+LA: mean -4.904 pp and passes the prespecified Holm-adjusted exact test.

The stronger native pipeline comparison favors CE+LA on the primary macro-F1 endpoint (56.68% versus 55.77%). Balanced Softmax has higher descriptive native tail-F1 (47.56% versus 46.40%); no tail-endpoint multiplicity claim is made. Retrained natural heads can change the ranking, so native outcomes should not be generalized to every downstream classifier. Despite increasing average tail-class EQ eligibility from 10.59% to 96.37%, class-uniform sampling does not yield a detectable fixed-EQ benefit in this study. These results apply to this frozen encoder, fit imbalance, adapter family and fixed data partition.

## Verification and artifacts

Frozen source SHA256: `874e9b744e27f99813d729a16b4dbb4cda5571bf0219f3ee1ea0351311ac2705`. Master protocol: `0dfd887a0676a0a2d7803a6fbd7b6db073421136968260def67c3335752c00c1`. Global selection: `7364508d76a3162df6e91179212826aefd5ec75609165d922bd2f87b13afb9a8`. Pre-confirmation analysis amendment: `5d780339bffa0a11158d01a17cf68e021bc6f9f010adad33f1784a75d859b0bb`.

Nine unit tests cover objective weighting, sampler determinism, split/rank integrity, BalancedSoftmax math, BCL value/gradient against an explicit loop, optimized EQ value/gradient against the prior implementation, actual adapter resume equivalence, missing-tuning confirmation gates, and exact-p/Holm resolution. The training-only throughput pilot selected two CPU workers with one thread each. No held-out scores were used for backend selection.

The independent audit replays 320 tuning scores, 660 confirmation readouts and 330 calibration curves, verifies 320 adapter states and 220 classifier states, and checks all initial/final state hashes, histories, checkpoint identities, momentum mappings and update budgets. It independently reconstructs class eligibility for all 10 seeds. A second amendment audit verifies the exact raw expressions of all six primary and two secondary contrasts against the independently checked per-seed metrics, then independently replays paired statistics and the six-member Holm family; a false-expression fixture with a valid contrast name is rejected. It does not re-extract encoder features or re-execute checkpoint inference; numerical verification uses the saved logits.

Portable numeric artifacts: `amended_aggregate.json`, `independent_verification.json`, `analysis_amendment.json`, `summary.csv`, `paired_comparisons.csv`, `secondary_comparisons.csv`, and `per_seed_results.csv`, `calibration_curves.csv`, `per_class_results.csv` and `amended_expression_verification.json`. Full operational artifacts and raw arrays remain in the experiment directory. Figures are `figures/fixed_factorial_macro_f1.png`, `figures/primary_paired_effects.png`, and `figures/sampling_eligibility.png`.
