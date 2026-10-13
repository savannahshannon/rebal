# Covertype follow-up: logit-adjusted MLP, FT-Transformer and boosted trees

This follow-up compares four tabular model families on all 581,012 canonical Covertype records using the REBAL study’s class-aware evaluation and calibration approach. The confirmation partition is withheld from all model and checkpoint selection. It is a new partition of a previously studied dataset; it does not provide an independent geographic or external-dataset confirmation.

## Data and separation

The source is the [original UCI Covertype gzip](https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz), attributed to Blackard (1998), DOI 10.24432/C50K5N. The gzip and canonical feature/label arrays are checked against the previously acquired source hashes. Every source row receives exactly one original zero-based row ID.

A class-stratified 70/10/10/10 split uses seed 314159. Fit, tuning and calibration counts use per-class floors; confirmation receives each class's remainder. The fit split alone estimates continuous-column means and standard deviations. This study does not add synthetic records or alter class frequencies. All partitions contain all seven classes. The source manifest records overlap between the new confirmation partition and the previous study's test partition so reuse is explicit.

The ten continuous features are standardized using fit statistics. MLP and XGBoost retain all 54 original predictors. FT-Transformer receives ten numerical feature tokens and two categorical feature tokens. The two categorical tokens losslessly decode wilderness columns 10:14 (four categories) and soil columns 14:54 (40 categories); every input is checked for exactly one active indicator per group, and one-hot reconstruction is checked exactly. This reduces redundant attention while preserving the predictors.

## Models and fixed search

The four model families are natural cross-entropy MLP, effective-number-weighted MLP, FT-Transformer, and XGBoost histogram boosting. The MLP has 54→256→256→128→7 layers with ReLU and dropout. Effective-number weights use beta=1−1/max(fit class count), stable effective counts, and normalization to expected fit sample weight one; the weighted objective applies throughout training.

The FT-Transformer is a local, explicit implementation of the architecture family: feature-specific numerical affine tokens, category embeddings and feature biases, a learned CLS token, two pre-layer-normalized self-attention/ReGLU feed-forward blocks (the final block computes only the CLS query, preserving the CLS output while avoiding unused token outputs), and a LayerNorm/ReLU classification head reading CLS. It is not a renamed MLP or the unmodified official RTDL implementation. Its small CPU-oriented search compares width 64 with width 96, each with four attention heads, attention/residual dropout 0.1, learning rate 0.0003 and weight decay 0.0001. The architecture follows the feature-tokenization and transformer approach of [Gorishniy et al. (2021)](https://proceedings.neurips.cc/paper/2021/hash/9d86d83f925f2149e9edb0ac3b49229c-Abstract.html); configuration and implementation differences are disclosed.

Each neural family runs 50 full epochs of AdamW with five-epoch linear warmup and cosine decay, gradient norm clipping at 1, and tune-set macro-F1 checkpoint selection. FT batches contain 1,024 records; MLP batches contain 2,048. The MLP search compares learning rates 0.001 and 0.003. Both keep width 256, dropout 0.1 and weight decay 0.0001. Every trial completes the full budget even if an earlier checkpoint is selected.

XGBoost 3.2.0 uses CPU histogram boosting, max_bin 256, subsample 0.8, column subsample 0.9, L2 leaf penalty 1 and minimum child weight 1. Both configurations train 600 boosting rounds (4,200 multiclass trees), comparing depth/learning-rate pairs (6, 0.08) and (10, 0.04). Tune-set macro-F1 chooses the iteration count from 50-round intervals; both trials still complete 600 rounds. The two trials per family equalize search count; neural epochs and tree rounds are model-specific budgets and do not claim equal FLOPs or wall time.

All eight tuning trials use optimization seed 9. Configuration selection considers raw-logit tuning macro-F1 only and breaks ties by configuration order. The selected configurations are frozen with protocol/source hashes before any calibration or confirmation forward pass. FT training uses MPS selected by fit-only throughput measurements; MLP and XGBoost use CPU. Two CPU workers process disjoint final seed ranges (10–14 and 15–19), each using four threads. A training-only concurrent pilot justified the scheduling change, and checkpoint/optimizer identities were audited before the first worker was retargeted. Training and inference timings were recorded under shared machine load, including controlled overlap between FT and other GPU jobs, so they do not establish isolated runtime efficiency. Actual phase wall times are recorded separately from retained model training times. Retargeting the CPU worker discarded 21.529 logged training seconds for uncheckpointed epochs 41–44 of seed 12 CE, plus unknown partial work; the verified checkpoint resumes at epoch 40, and the retained model/optimizer still completes exactly 50 epochs. The neural `total_seconds` record (exported as `fit_and_tune_seconds`) measures the current invocation’s wall time; a resumed invocation excludes earlier checkpoint segments. Use retained `training_seconds` and the separately recorded phase wall time when describing the full run. All architectures, batches and budgets are frozen before held-out scoring. Final training repeats selected configurations with ten optimization seeds, 10–19. CPU final fits can begin after immutable per-family selection, while other families are still tuning. They use only fit and tuning data. All calibration/confirmation scoring remains closed until the global eight-trial freeze and tuning-prediction captures. The controller persists each calibration decision before confirmation inference and loads existing tree artifacts for final evaluation rather than training them again. Each final fit selects its checkpoint/iteration count from tuning data using the frozen rule.

## Evaluation and evidence

Every final fit saves calibration and confirmation logits with exact row IDs and labels. Post-hoc logit adjustment selects tau from {0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2} using calibration macro-F1; ties favor the smaller tau. Natural CE, FT and XGBoost use the fit-frequency prior. Weighted MLP uses the effective weighted-objective prior. Confirmation records cannot select configurations, checkpoints or tau. Here, calibration means macro-F1 selection of a classification decision adjustment; probability reliability is outside the measured endpoints.

The study produces 40 fitted models and 80 readouts (raw and adjusted). Metrics include accuracy, macro-F1, macro recall, per-class F1/precision/recall/support, and head/middle/tail F1 by fit-frequency rank. Class-balanced evaluation diagnostics reweight the same confirmation records by inverse class frequency; they estimate performance under a uniform class prior without creating a second training or tuning dataset. Native-prior macro-F1 remains the primary selection metric. In this single-label multiclass setting, balanced accuracy equals macro recall; both named fields are retained for compatibility rather than treated as separate performance signals.

The three declared primary contrasts compare adjusted FT, adjusted XGBoost and adjusted weighted MLP against adjusted natural CE MLP. All comparisons use ten paired optimization seeds. Summaries retain individual differences, sample SD, unadjusted paired t intervals, t-test p-values, exhaustive 1,024-pattern sign-flip p-values and Holm corrections across the three contrasts. Seed intervals quantify optimization variability on one fixed partition; they do not capture spatial dependence or dataset variation.

The original runner replays its saved predictions and compares its own NumPy
metrics with a second scikit-learn implementation. The integrity overlay checks
data, selection, configuration and artifact bindings and delegates that numeric
replay to the frozen runner. A separate final-prediction auditor imports no
scientific runner and loads no model states: it independently derives priors
from fit counts, replays all eight calibration candidates for each fit, and
recomputes the 640 scalar metrics, 1,680 per-class F1/recall/precision values,
3,920 confusion entries and 560 class supports across 80 readouts. A separate
statistical auditor independently checks the 64 method/endpoint summaries and
three paired contrasts. These checks do not rerun training or model inference.
Tree models use UBJSON; neural states use torch checkpoints. Publication/evidence
packaging may omit model weights while retaining the numeric prediction records
required for replay; original actual-state audits remain separately identified.

## Reproduction

The following commands use the study workspace root, with the repository at
`work/rebal`. Use a fresh output directory for a new training run. Acquisition
and preparation retain the full canonical dataset; the follow-up script makes
its own fixed partition and estimates its scaler from fit rows only.

```bash
python work/rebal/scripts/prepare_nonimage.py acquire --dataset covertype \
  --output work/data/nonimage/sources/covertype
python work/rebal/scripts/prepare_nonimage.py prepare --dataset covertype \
  --source work/data/nonimage/sources/covertype \
  --output work/data/nonimage/covertype/natural
python work/rebal/scripts/benchmark_tabular_followup.py prepare \
  --out work/experiments/covertype_followup_reproduction \
  --data work/data/nonimage/covertype/natural \
  --source work/data/nonimage/sources/covertype/covtype.data.gz
python work/rebal/scripts/benchmark_tabular_followup.py pilot \
  --out work/experiments/covertype_followup_reproduction --threads 2
python work/rebal/scripts/benchmark_tabular_followup.py tune \
  --out work/experiments/covertype_followup_reproduction --threads 4 --ft-device mps
python work/rebal/scripts/verify_tabular_followup.py capture \
  --run work/experiments/covertype_followup_reproduction \
  --script work/rebal/scripts/benchmark_tabular_followup.py
python work/rebal/scripts/run_tabular_followup_controlled.py prefit \
  --out work/experiments/covertype_followup_reproduction \
  --script work/rebal/scripts/benchmark_tabular_followup.py \
  --families mlp_ce mlp_cb ft_transformer xgboost --auto-evaluate
python work/rebal/scripts/verify_tabular_followup.py verify \
  --run work/experiments/covertype_followup_reproduction \
  --script work/rebal/scripts/benchmark_tabular_followup.py
python work/rebal/scripts/verify_tabular_followup_checkpoints.py \
  --run work/experiments/covertype_followup_reproduction \
  --out work/experiments/covertype_followup_reproduction/checkpoint_audit.json \
  --require-complete
python work/rebal/scripts/verify_tabular_followup_statistics.py \
  --run work/experiments/covertype_followup_reproduction \
  --out work/experiments/covertype_followup_reproduction/statistical_replay.json
python work/rebal/scripts/verify_tabular_followup_predictions.py \
  --run work/experiments/covertype_followup_reproduction \
  --source work/rebal/scripts/benchmark_tabular_followup.py \
  --out work/experiments/covertype_followup_reproduction/independent_prediction_verification.json
```

The controlled runner mirrors the production gates: it records immutable
family selections, saves fit artifact receipts, requires all eight tuning
captures, persists the calibration choice, and then evaluates confirmation.
The sequence above runs one final worker. The recorded production scheduling
used separate CPU seed ranges and a separate FT worker with the same scientific
settings. Numeric verification checks saved logits independently; when local
weights are present, the overlay also checks their hashes and trusted checkpoint
states. A weight-free portable replay reports zero present model artifacts and
does not require model inference. The explicit checkpoint auditor requires the
original weights.

Use Python 3.12 and the recorded package versions: NumPy 2.5.4, SciPy 1.18.1,
PyTorch 2.14.1, scikit-learn 1.9.1 and XGBoost 3.2.0. This Mac's XGBoost links
the official Torch package's OpenMP runtime through an operational environment
variable; no private installation path is required in published protocol files.
Other platforms should use their supported XGBoost/OpenMP setup. Protocol
fingerprints bind package versions, source, data and device settings, so a fresh
run on a different supported environment receives a different identity rather
than overwriting the published run's receipts. The two historical source-row
and transform auditor scripts retain the original relative `work/data` and
`work/experiments/rebal_next_tabular` acquisition layout; their receipts and
source hashes document the completed production checks.

Status: implementation, canonical source-row and fit-only transformation audits, training-only pilots, 11 unit tests and complete fixture prediction/statistical replay passed. All eight full-budget tuning trials are complete, their tuning predictions independently reproduce the recorded scores, and the global selection is frozen. Selected settings are width 96 for FT, learning rate 0.003 for both MLP objectives, and depth 10/learning rate 0.04 for XGBoost. The forty final fits are running or trained; calibration/confirmation scoring opens only after the eight capture receipts. Final results, actual-state audits, statistical verification and the weight-free evidence bundle remain in progress.
