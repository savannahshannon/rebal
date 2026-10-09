# REBAL: Feature-Space Rebalancing for Long-Tailed Recognition

Code and aggregated results for the paper *"When Image Synthesis Hurts: Feature-Space Rebalancing for Long-Tailed Recognition"* (Shannon, submitted 2026).

## What the paper finds

Long-tailed recognition pipelines often add a conditional GAN to synthesize examples for rare classes. We built one — REBAL, combining a cGAN, feature-space SMOTE, effective-number reweighting, a feature-space equalization regularizer, and decoupled classifier retraining — and then disassembled it under a corrected validation protocol with paired multi-seed tests.

Three results:

1. **The cGAN is harmful, not helpful.** At CIFAR-100-LT ρ = 100, removing it improves top-1 by 3.6 pts (p = 0.003), head macro-F1 by 3.6 (p = 0.012), and mid macro-F1 by 5.6 (p = 0.001), with no change in tail macro-F1. Per-class FID of the pre-trained generator degrades with class rarity (Pearson r = −0.27), so the generator is worst on the classes it exists to help.
2. **The surviving pipeline (REBAL-F) improves the learned representation, not just the classifier.** Under a classifier-free nearest-class-mean probe at ten seeds, tail features are significantly better and the head-to-tail gap is significantly smaller than for a matched baseline.
3. **REBAL has an operating regime.** At CIFAR-100-LT ρ = 100 (baseline tail macro-F1 ≈ 0.18), REBAL-F plus post-hoc logit adjustment leads all comparisons. At CIFAR-10-LT ρ = 100 (baseline tail macro-F1 ≈ 0.67), the baseline is already strong on the tail; logit adjustment post-hoc leads, and REBAL's rebalancing provides no additional benefit. We report this as a scope statement rather than hide it.

The evaluation protocol used throughout — stratified validation split carved out of training, no test-set checkpoint selection, matched backbone/schedule/augmentation across arms, paired t-tests at five or ten seeds — corrects three practices that had inflated an earlier version of the work.

## Repository layout

```
rebal/
├── framework.py                    # REBAL pipeline (cGAN, SMOTE, reweighting, equalization, cRT)
├── baselines.py                    # CE, class-balanced, LDAM, LDAM-DRW, logit adjustment
├── posthoc_la.py                   # Post-hoc logit-adjustment composability test
├── feature_probe.py                # NCM / linear probe / probe+LA representation evaluation
├── scripts/
│   ├── run_corrected.py            # Resumable sweep driver (REBAL + baselines, all tiers)
│   ├── aggregate_corrected.py      # Aggregator for REBAL/ablation runs
│   ├── aggregate_baselines.py      # Aggregator for standalone baseline runs
│   └── prepare_cifar100_for_stylegan2.py
├── results/
│   ├── cifar100/                   # Table I, II, III, IV, VI data (five- and ten-seed means)
│   └── cifar10/                    # Table VII data (operating-regime check)
├── requirements.txt
├── LICENSE
└── README.md
```

## Reproducing the results

### Setup

```bash
git clone https://github.com/<you>/rebal.git
cd rebal
python3 -m venv env && source env/bin/activate
pip install -r requirements.txt
```

The paper's runs used Python 3.10, TensorFlow 2.15, on an NVIDIA T4 (Google Colab Pro). Any CUDA-capable GPU with 8 GB+ VRAM should work. CPU-only is possible but one five-seed arm takes 10+ hours.

### The main CIFAR-100-LT grid (Tables I, II)

```bash
# Full sweep: REBAL main + regularizer ablation + 5 published baselines + leave-one-out ablations
python3 scripts/run_corrected.py --dataset cifar100 --seeds 0 1 2 3 4

# Aggregate
python3 scripts/aggregate_corrected.py --dataset cifar100
python3 scripts/aggregate_baselines.py --dataset cifar100
```

Estimated time: ~50 hours on T4 for the full 50-cell sweep. Each cell writes `results.json` + `per_class_f1.csv` + `run.log` to its own directory under `experiments/`; the sweep is resumable — rerunning the same command skips any cell with a valid `results.json`.

### Operating-regime check at CIFAR-10-LT (Table VII)

```bash
python3 scripts/run_corrected.py --dataset cifar10 --tier 1 3 --seeds 0 1 2 3 4
python3 scripts/aggregate_corrected.py --dataset cifar10
python3 scripts/aggregate_baselines.py --dataset cifar10
```

### Representation probes (Table III) and composability (Table IV)

```bash
python3 feature_probe.py --seeds 0 1 2 3 4 5 6 7 8 9     # NCM + linear probe + probe+LA
python3 posthoc_la.py    --seeds 0 1 2 3 4 5 6 7 8 9     # REBAL-F + LA vs baseline + LA
```

Both read checkpoints produced by `run_corrected.py --tier 1`, so run the main grid first.

## Validation protocol — what the sweep enforces

Three practices from the earlier version of this work that produced inflated numbers, and the fix each sweep cell is required to use:

| Practice | Fix |
|---|---|
| Test set used as validation for model selection | A stratified 10% holdout is carved out of **training** per seed; model selection and early stopping read only this holdout. |
| Early stopping on the imbalanced val set rewarded head-biased weights | Full cosine schedule, no early stopping; the final-epoch weights are reported. |
| Baseline arms used weaker schedules than REBAL | Every arm — plain CE, LDAM, LDAM-DRW, LA, class-balanced, and all REBAL variants — uses the same ResNet-32, 200 epochs with 10-epoch warmup, cosine LR, batch 128, Nesterov SGD mom 0.9, weight decay 5e-4, grad-norm clipping at 1.0, and the Cui et al. 2019 augmentation stack. |

`run_corrected.py` refuses to count any cell as "done" unless its stored `config` fields prove it used the corrected protocol. See `protocol_ok()` in that script.

## Citation

```
@article{shannon2026rebal,
  title={When Image Synthesis Hurts: Feature-Space Rebalancing for Long-Tailed Recognition},
  author={Shannon, Savannah},
  year={2026}
}
```

(Update once the paper is accepted.)

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgments

Baselines reproduce Cui et al. 2019 (class-balanced loss), Cao et al. 2019 (LDAM / LDAM-DRW), Menon et al. 2021 (logit adjustment), and Kang et al. 2020 (decoupled representation-classifier training). The cGAN uses StyleGAN2-ADA (Karras et al. 2020) pretrained on CIFAR-100-LT. Representation evaluation uses the nearest-class-mean protocol as in Snell et al. 2017.
