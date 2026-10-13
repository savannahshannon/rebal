# Fixed-treatment equalization and selected strong-control follow-up

This experiment tests whether rare-class minibatch coverage changes equalization's incremental value on frozen official DeiT-tiny features. It also compares calibrated cross-entropy, Balanced Softmax, and a documented frozen-feature adaptation of Balanced Contrastive Learning (BCL).

The source development pool is Tiny ImageNet's 100,000 original training rows. The existing pixel-hash policy removes every original training/evaluation overlap and retains the first remaining duplicate, leaving 99,973 canonical development rows. A new fixed seed `20261013` assigns 20 tuning, 20 calibration, and 100 confirmation examples per class **before** long-tail thinning. Fit-pool, tuning, calibration and confirmation are disjoint. The official evaluation set is never scored in this follow-up. The new confirmation rows may have trained historical adapters; all adapters in this study restart from the same random initialization per optimization seed, and confirmation predictions are unavailable until the new selection is frozen.

The fixed nominal 100:1 fit sample has 15,268 rows. The largest retained class has 355 examples and the smallest has 3, so the actual largest/smallest ratio is 118.33. The fixed semantic class order and retained fit examples do not change across optimization seeds `10..19`. `class_order` maps frequency rank to class ID; `class_rank` is its inverse. Feature scaling uses retained fit rows only.

## Factorial control

Natural row sampling and class-uniform sampling target the same exactly class-balanced expected cross-entropy objective from epoch one. For class count `n_c`, `N` retained rows and `C` classes, natural sampling uses importance `N/(C*n_c)`, while class-uniform sampling uses importance one. This is an explicit importance correction, not a second class-frequency weighting applied to balanced samples. Both samplers draw with replacement. Their expected cross-entropy loss is the same; gradient variance, minibatch clipping and momentum can still produce different SGD trajectories. Every optimizer update consumes a full batch of 1,024 examples; every epoch has `ceil(N/1024)=15` updates.

Each sampler has a matched equalization-off arm at learning rate 0.05. Equalization-on candidates use onset fractions `{0.2,0.5,0.8}` crossed with coefficients `{0.01,0.05}`. All six candidates complete the same 50-epoch budget. A global candidate per sampler is selected by mean tuning macro-F1 across the ten paired optimization seeds. The shared onset 0.8/coefficient 0.05 reference supplies the **primary fixed-treatment 2×2 factorial**. Its on/off contrasts use the same learning rate and EQ treatment under both samplers. The separately selected onset/coefficient pipelines are secondary: their comparison against the fixed off arm includes six-candidate selection opportunity, and their interaction cannot isolate a common EQ treatment.

Equalization is the same normalized-feature within-class spread minus mean pairwise between-class squared distance used previously. A moment identity computes the pairwise term without allocating a class-by-class-by-feature tensor. A test compares its loss and gradients with the explicit pairwise formulation.

## Strong controls and BCL adaptation

Cross-entropy and Balanced Softmax each receive six peak learning-rate candidates `{0.01,0.025,0.05,0.075,0.1,0.2}`. Cross-entropy's calibration pool selects logit-adjustment tau before its tuning-score evaluation; Balanced Softmax adds the logarithm of retained class counts to logits during training and predicts using unadjusted logits. Each global learning rate maximizes mean tuning macro-F1 across the same ten seeds. Fixed learning rate 0.05 references are retained. This matches candidate counts across selected methods, while the hyperparameter dimensions differ. Calibration is development selection data used for candidate offset choice and final offsets; confirmation is the protected outcome partition.

BCL uses the class-averaged contrastive denominator and one normalized classifier-weight prototype for every class. Each optimizer update produces two independently sampled adapter-dropout views of the first 64 randomly drawn batch examples, and combines the BCL loss with Balanced Softmax classification. Six candidates cross coefficient `{0.05,0.2,0.5}` with temperature `{0.07,0.1}` at learning rate 0.05. Selection uses tuning macro-F1. Selected BCL and selected Balanced Softmax can use different learning rates, so their primary comparison tests selected pipelines and does not isolate the added contrastive loss at a common learning rate. This is a frozen-feature/dropout-view adaptation, not a reproduction of the original image augmentations, backbone training or two projection branches. The vectorized loss and gradients are tested against an explicit loop, including classes absent from the sampled batch but present through their prototypes.

Primary sources: [Balanced Contrastive Learning](https://openaccess.thecvf.com/content/CVPR2022/html/Zhu_Balanced_Contrastive_Learning_for_Long-Tailed_Visual_Recognition_CVPR_2022_paper.html), [official BCL implementation](https://github.com/FlamieZhu/Balanced-Contrastive-Learning/blob/main/loss/contrastive.py), [Balanced Softmax](https://proceedings.nips.cc/paper_files/paper/2020/file/2ba61cc3a8f44143e1f2f13b2b729ab3-Paper.pdf), and [logit adjustment](https://openreview.net/pdf?id=37nvvqkCo5).

## Budgets and confirmation

There are 32 tuning configurations per seed and 320 fully trained adapters. Every adapter completes 50 epochs/750 updates, including five warm-up epochs, SGD with momentum 0.9/Nesterov, linear-weight L2 0.0005, per-tensor gradient clipping at one, and a cosine schedule. A training-only throughput pilot selects CPU execution and one Torch thread per worker. Two workers share the fixed source/split/scaler identity.

After all tuning rows, predictions, budgets and calibration choices are checked, an immutable selection record is written. Only then are the 20,000 balanced confirmation rows inferred. Eleven named selected/reference arms per seed have native readouts plus independently retrained natural and random-oversampled linear classifiers. Each retrained classifier receives 2,000 updates at batch size 2,048 and learning rate 0.01. Calibration-only logit adjustment supplies another named readout for each, including aliases when the training prior is uniform. Thus there are 220 retrained classifiers and 660 named confirmation readouts. Selected/reference arms can coincide; identical policies are disclosed rather than counted as independent evidence. No new SMOTE neighbor graph is constructed.

The current primary family contains **six** paired effects on confirmation macro-F1: four fixed-treatment factorial effects plus two selected strong-control pipeline comparisons. An immutable analysis amendment was frozen at `2026-10-12T20:05:52Z`, with 148 of 320 tuning cells complete, no selection receipt, zero completed confirmation cells and zero confirmation prediction archives. It changes the interpretation and multiplicity family without changing training, source, selection or run budgets. Its fingerprint is `5d780339bffa0a11158d01a17cf68e021bc6f9f010adad33f1784a75d859b0bb`.

| Prespecified contrast | Exact endpoint expression |
|---|---|
| Fixed natural EQ effect | `natural_reference__native − natural_off__native` |
| Fixed balanced EQ effect | `balanced_reference__native − balanced_off__native` |
| Sampling effect with EQ off | `balanced_off__native − natural_off__native` |
| Fixed factorial interaction | `(balanced_reference__native − balanced_off__native) − (natural_reference__native − natural_off__native)` |
| Selected Balanced Softmax versus selected CE+LA | `balanced_softmax__native_la − ce__native_la` |
| Selected BCL versus selected Balanced Softmax | `bcl_dropout__native_la − balanced_softmax__native_la` |

The factorial's native prior is uniform, making its logit-adjustment readout an exact alias. The selected strong-control endpoints use the declared calibrated native policies; Balanced Softmax and BCL also have uniform effective priors. Exact two-sided sign-flip and paired-t p-values receive Holm correction in this six-member family. Ten paired seeds allow an exact two-sided minimum p-value of `2/1024`. The two validation-selected EQ on/off effects are secondary descriptive analyses. Tail F1, calibration and classifier retraining are also descriptive. The original runner aggregate retains its historical adaptive four-comparison family, which is superseded for primary inference by the amended aggregate. A separate pooled 22-comparison Holm analysis is a cross-study sensitivity; per-study families remain primary.

Reproduction phases (confirmation cannot precede frozen selection):

```bash
python scripts/followup_eq.py prepare --features data/deit_tiny --out experiments/followup_eq --threads 1
python scripts/followup_eq.py tune --features data/deit_tiny --out experiments/followup_eq --threads 1
python scripts/followup_eq.py select --features data/deit_tiny --out experiments/followup_eq --threads 1
python scripts/followup_eq.py evaluate --features data/deit_tiny --out experiments/followup_eq --threads 1
python scripts/followup_eq.py aggregate --features data/deit_tiny --out experiments/followup_eq --threads 1
python scripts/followup_eq.py verify --features data/deit_tiny --out experiments/followup_eq --threads 1
```

For two workers, run `tune` and then `evaluate` with `--worker-count 2 --worker-index 0` and `--worker-index 1`, waiting for both tuning workers before `select`. All epoch/update budgets and artifact identities are checked on resume. Verification independently replays stored raw logits, calibration curves, class ranks, scalar and per-class metrics, and summary statistics. It checks model checkpoint hashes but does not rerun inference from checkpoint weights. The additional independent audit checks actual adapter/classifier tensor states, initialization/final-state hashes, optimizer momentum mappings and complete update histories; a separate amendment audit replays all six exact contrast expressions and their statistics from the independently verified per-seed metrics. Neither audit re-extracts encoder features or re-executes checkpoint inference.

The reproduction commands above preserve the frozen training source and its historical aggregate behavior. Current primary inference additionally applies the prospective six-contrast amendment. Published numeric artifacts include the amended aggregate, amendment receipt, per-seed results, full summary, paired/secondary comparisons, calibration curves, per-class metrics and independent verification receipts. The original protocol document is retained privately with its frozen source hash; this public documentation reflects the amendment.

To replay the published six-primary/two-secondary analysis from its archived numeric inputs, run the portable helper below before the pooled 22-comparison sensitivity helper. The ten `confirmation/seed_10..seed_19/metrics.json` files retain every scalar endpoint and exclude training/checkpoint operational metadata. The exact counter snapshot contains only phase/counts/timestamps/protocol, ephemeral process IDs and relative log names; its preserved bytes verify the amendment's preconfirmation hash. The helper checks selection and amendment fingerprints, complete tuning/readout grids, the independent verification receipt, timezone-aware freeze order, per-seed summary consistency and all eight exact contrast expressions. It does not retrain or select models.

```bash
python scripts/apply_followup_amendment.py \
  --aggregate results/stronger_baselines/equalization/public_amendment_inputs/historical_aggregate.json \
  --confirmation-dir results/stronger_baselines/equalization/public_amendment_inputs/confirmation \
  --selection results/stronger_baselines/equalization/public_amendment_inputs/selection.json \
  --amendment results/stronger_baselines/equalization/public_amendment_inputs/analysis_amendment.json \
  --independent-verification results/stronger_baselines/equalization/public_amendment_inputs/independent_verification.json \
  --counter-snapshot results/stronger_baselines/equalization/public_amendment_inputs/preconfirmation_controller_state.json \
  --source scripts/followup_eq.py \
  --out results/stronger_baselines/equalization/reproduced_amended_aggregate.json
```

These paths refer to the published repository evidence. For a copied local archive, replace `results/stronger_baselines/equalization/public_amendment_inputs` with its equivalent local directory, such as `outputs/REBAL_Next_Round_Evidence/equalization/public_amendment_inputs`; the helper checks the input bytes and hashes rather than the directory name. The training source is required and its SHA256 must match the frozen amendment. Ordering is checked against the supplied preserved counter and selection/confirmation receipt timestamps; file hashes bind their bytes, and those timestamps are not an external time attestation. The independently verified original-aggregate hash must match the supplied aggregate; an archived verification receipt cannot validate a newly retrained run with different numeric artifacts.

The complete raw numeric archive is local evidence under `outputs/REBAL_Next_Round_Evidence/equalization/portable`. It preserves the exact original bytes of 320 tuning logit captures and ten confirmation prediction archives, histories, scalar/per-class metrics, source labels, pixel-identity manifest, partition/rank arrays and fit-scaler arrays. It omits all 320 adapter and 220 head checkpoint states, cached dense features, source images and operational logs. Git contains the smaller summaries, comparisons, receipts, catalog, report and source; the multi-gigabyte raw prediction blobs are not uploaded to Git.

From the local `portable` directory, independently replay the historical raw numeric analysis, then apply the prospective amendment:

```bash
python scripts/verify_eq_followup.py \
  --run run --labels data/labels.npy --cache-manifest data/cache_manifest.json \
  --source scripts/followup_eq.py --helper-dir scripts \
  --original-state-audit audit/original_actual_state_verification.json \
  --portable-without-weights --receipt ../portable_numeric_recheck.json
python scripts/apply_followup_amendment.py \
  --aggregate run/aggregate.json --confirmation-dir run/confirmation \
  --selection run/selection.json --amendment analysis/analysis_amendment.json \
  --independent-verification audit/original_actual_state_verification.json \
  --counter-snapshot analysis/preconfirmation_controller_state.json \
  --source scripts/followup_eq.py --out ../amended_aggregate_recheck.json
```

The independent portable verifier imports no scientific scoring helper. It computes NumPy confusion matrices and crosschecks per-class F1/recall with scikit-learn; regenerates pixel-identity exclusions, the seeded split, thinning and rank; checks all candidate groups and global winners; and replays all 660 confirmation views, 330 decision-offset curves, sampling-eligibility histories and four historical contrasts. Its receipt explicitly records zero checkpoint-state loads and 540 omitted states. It binds the original actual-state audit separately and does not reconstruct model inference, optimizer updates, encoder extraction or scaler estimation. Image prediction archives store label/rank arrays rather than explicit per-prediction source IDs; row order is bound to the hash-checked partition indices and labels. The amendment then reproduces all six primary and two secondary expressions from those audited metrics.

Exact sign-flip enumeration assumes paired sign exchangeability or symmetry under the null. Paired-t intervals use the ten optimizer seeds and a normality approximation conditional on the fixed split and selected configurations; they do not cover hyperparameter-selection or external dataset uncertainty. Decision offsets target class-balanced prediction, not probability-calibration accuracy. Experimental split disjointness also does not establish independence from the encoder's pretraining data.
