# Full-scale vision study: measured results

The fresh A100 study completed 80 tuning fits and 720 final fits, each for 50 epochs, across ImageNet-1K, CIFAR-100 and iNaturalist 2018. A full-grid scientific audit and a separate four-GPU provenance check passed before confirmation. All 720 final fits were confirmed, and analysis recomputed metrics from saved predictions. Historical A800 and hosted T4 fits are not included in these totals.

The primary outcome is confirmation macro-F1 after the decision offset selected on development data. Each EQ effect below is the paired seed mean of EQ-on minus EQ-off, in percentage points. An interaction is the DINOv2 EQ effect minus the DeiT EQ effect. Intervals are pointwise paired 95% t intervals over 12 seeds; exact two-sided sign-flip tests assume paired sign exchangeability or symmetry. Holm-adjusted p values are shown within each six-comparison condition and across all 30 comparisons. These tests describe optimization variability conditional on fixed partitions; the seeds are not independent datasets. Values are rounded to three decimals.

IN = ImageNet-1K; CF = CIFAR-100; iNat = iNaturalist 2018; N/U = natural/uniform sampler; D/V = DeiT/DINOv2; I = encoder-by-EQ interaction.

| Condition | Contrast | Effect (pp) | 95% interval (pp) | Holm condition | Holm across 30 |
|---|---|---:|---:|---:|---:|
| IN native | N-D | +0.030 | +0.014,+0.046 | 0.008 | 0.016 |
| IN native | U-D | +0.024 | +0.011,+0.037 | 0.009 | 0.018 |
| IN native | N-V | +0.054 | +0.045,+0.063 | 0.003 | 0.015 |
| IN native | U-V | +0.038 | +0.025,+0.051 | 0.003 | 0.015 |
| IN native | N-I | +0.024 | +0.006,+0.042 | 0.031 | 0.049 |
| IN native | U-I | +0.014 | +0.000,+0.028 | 0.049 | 0.098 |
| IN LT100 | N-D | +0.055 | +0.019,+0.092 | 0.004 | 0.020 |
| IN LT100 | U-D | +0.438 | +0.410,+0.465 | 0.003 | 0.015 |
| IN LT100 | N-V | +0.353 | +0.317,+0.389 | 0.003 | 0.015 |
| IN LT100 | U-V | +0.333 | +0.298,+0.368 | 0.003 | 0.015 |
| IN LT100 | N-I | +0.298 | +0.260,+0.335 | 0.003 | 0.015 |
| IN LT100 | U-I | -0.104 | -0.131,-0.078 | 0.003 | 0.015 |
| CF native | N-D | +0.698 | +0.600,+0.795 | 0.003 | 0.015 |
| CF native | U-D | +0.688 | +0.605,+0.771 | 0.003 | 0.015 |
| CF native | N-V | +0.433 | +0.355,+0.512 | 0.003 | 0.015 |
| CF native | U-V | +0.471 | +0.408,+0.534 | 0.003 | 0.015 |
| CF native | N-I | -0.264 | -0.396,-0.132 | 0.003 | 0.017 |
| CF native | U-I | -0.217 | -0.310,-0.124 | 0.003 | 0.015 |
| CF LT100 | N-D | -0.738 | -0.797,-0.679 | 0.003 | 0.015 |
| CF LT100 | U-D | +0.318 | +0.252,+0.385 | 0.003 | 0.015 |
| CF LT100 | N-V | -0.882 | -0.948,-0.815 | 0.003 | 0.015 |
| CF LT100 | U-V | +0.241 | +0.205,+0.277 | 0.003 | 0.015 |
| CF LT100 | N-I | -0.144 | -0.241,-0.047 | 0.024 | 0.049 |
| CF LT100 | U-I | -0.077 | -0.165,+0.011 | 0.085 | 0.098 |
| iNat native | N-D | +0.154 | +0.107,+0.202 | 0.003 | 0.015 |
| iNat native | U-D | +0.173 | +0.133,+0.213 | 0.003 | 0.015 |
| iNat native | N-V | +0.371 | +0.324,+0.418 | 0.003 | 0.015 |
| iNat native | U-V | +0.334 | +0.294,+0.373 | 0.003 | 0.015 |
| iNat native | N-I | +0.216 | +0.140,+0.292 | 0.003 | 0.015 |
| iNat native | U-I | +0.161 | +0.097,+0.225 | 0.003 | 0.015 |

On CIFAR-100 LT100, EQ under natural sampling reduced macro-F1 by 0.738 pp for DeiT and 0.882 pp for DINOv2; under uniform sampling it increased macro-F1 by 0.318 and 0.241 pp, respectively. On ImageNet LT100, uniform-sampling EQ effects were +0.438 and +0.333 pp. The four native ImageNet EQ effects ranged from +0.024 to +0.054 pp. Practical importance depends on context; these results do not establish universal benefit or harm.

LT100 is this study's artificial thinning, not the standard ImageNet-LT release. The models use frozen pretrained representations; pretraining differences and possible overlap limit causal and from-scratch claims. The scientific audit checked encoded source bytes and frozen annotations/manifests, but did not independently reconstruct every ImageNet or iNaturalist label or decode every image. No per-class, EQ-activity, efficiency, or external replication claim is made here without separate verified analysis.

The existing IEEE manuscript remains a prospective protocol; the results manuscript is in preparation. No image data, feature caches, checkpoints, credentials, or private machine paths are included in this report.
