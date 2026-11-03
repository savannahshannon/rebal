# Completed REBAL results and separate follow-ups

These three result sets must remain distinct:

| Study | Fits | Budget and inference |
|---|---:|---|
| Primary full-scale study | 80 tuning; 720 final | 50 epochs; 30 planned comparisons, 12 paired optimization seeds |
| Local EQ follow-up | 48 final | 50 epochs; three seeds per split/variant; exploratory, no new confirmatory p-value family |
| Generated-image follow-up | 192 final | 1,950 updates each; six paired synthetic-minus-duplicate comparisons, 12 seeds; conditional on one bank |

The [integrated manuscript](../../paper/results_2026/README.md) reports every primary
contrast and all follow-up conditions. Numerical CSV scores are fractions; manuscript
scores are percentages and paired differences are percentage points.

## Results and interpretation

In the original grid, EQ improved mean macro-F1 in 18 of 20 encoder/sampler/condition
cells. Both adverse effects occurred under natural sampling on CIFAR-100 LT100.
The follow-up repeats the negative natural-sampling direction and the small positive
uniform-sampling direction across three split/rank configurations. Changing the
coefficient or onset modifies those effects. These three-seed results and retained
eligibility/gradient diagnostics do not identify a causal mechanism.

SD-Turbo produced one fixed bank of 35,223 images. Synthetic additions outperform
duplicate-real controls at 10%, 50%, and 100% class-deficit filling for both encoders.
The synthetic-minus-duplicate advantage shrinks from 1.802 to 0.281 points for DeiT-S
and from 2.474 to 1.040 for DINOv2-S. Absolute synthetic scores change little at the
largest dose. This is diminishing return under a fixed fitting budget, not an
equivalence test, a universal synthetic-data ceiling, or a best-pipeline result.
The primary calibrated cross-entropy means (66.616% and 77.972% on this partition)
remain above all generated-condition means; that cross-study comparison is descriptive
because seeds and decision rules differ.

## Reproduce the numerical tables

With NumPy and SciPy available, run:

```bash
python reproduce_tables.py --output reproduced_tables
```

The script reads only released numeric data. It verifies all 30 original contrasts,
all 10 separately labelled exploratory sampler-by-EQ contrasts, and all six
generated-image contrasts, including paired t intervals, exact sign-flip tests,
and each separate Holm family. It also regenerates the 60 primary condition means,
16 generated-condition means, and all five EQ follow-up settings for both samplers.
It does not train models or reproduce predictions from images/checkpoints.

The original 30-family has six comparisons per data condition; adjustment over all
30 is additionally reported as sensitivity analysis. The exploratory ten-family
and generated six-family must not be substituted for that original family.
Intervals describe seed variation conditional on fixed data/model choices and are
pointwise, not simultaneous intervals. Exact sign-flip inference requires the stated
paired symmetry/exchangeability assumption.

`eligibility_by_class.csv.gz` and `retained_eq_diagnostics.csv` contain read-only
diagnostic summaries from 480 original histories, whose source records were bound to
the retained archive. Gradients were logged periodically; they are not all-step
gradient traces or counterfactual experiments.

The two follow-up protocol files omit private ownership/machine fields but retain
the scientific settings, all cells, pretrained model pins, SD-Turbo revision and
class prompts. They are explicitly sanitized public projections, not byte-identical
copies of the private protocol receipts. SHA-256 pins of the original protocols are
recorded in `release_evidence.json`.

## Verification and limits

The primary full scientific and GPU-provenance audits passed before sealed
confirmation. All 48 local follow-up predictions were reproduced in their CPU
environment. All 192 generated-study predictions, top-five indices, and metrics
were reproduced with the original A100 evaluator, alongside checks of the generated
images, pools, scalers, budgets, and barriers. A prior CPU check found one fifth-rank
cross-device difference; no universal bitwise portability is claimed.

All three studies have independently received-byte-verified retained archives.
Source and generated images, checkpoints, credentials, private connection details,
and operational receipts are not part of this public release. Their private retention
does not make them public or create new redistribution rights.

The new experiments reuse the original official test set, frozen encoders, and one
CIFAR-100 LT100 setting. The generated study uses one bank and intended prompt
labels without independent semantic-label validation; pretrained-data independence
is not established. Fixed adapter updates do not equal total computation including
generation. These limits are retained in the manuscript and protocols.
