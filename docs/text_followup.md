# Pretrained BERT follow-up on DBpedia-14

This follow-up tests stronger contextual text representations, Balanced Softmax,
and selective encoder tuning. It retains the full public DBpedia source: 560,000
native training records and 70,000 native test records. The previous mean-pooled
text model is trained again on the new partitions rather than reusing its old
weights or reporting its old benchmark scores as a matched control.

## Confirmation protocol

Normalized exact full-record and nonempty content-only duplicate groups are
canonicalized before partitioning. Native test matches exclude 19 development
records; 121 redundant development records are also removed. No development
label conflicts are present. The resulting 559,860 canonical training records
are partitioned with a new fixed seed, 20261012, and a fixed class-rank seed,
20261013. The rank order of zero-based classes is
`[12,6,0,10,3,9,13,11,5,7,2,4,8,1]`.

The primary nominal 100:1 training condition has 106,232 fit records. Each class
contributes 2,000 records to learning-rate tuning, 2,000 to calibration, and 4,000
to the fresh train-derived confirmation partition. Consequently there are
28,000 tuning, 28,000 calibration, and 56,000 confirmation records. The long-tail
fit maximum is 31,915 and minimum is 319. The native 70,000-record benchmark test
is retained as an exploratory endpoint because its previous scores informed
the research direction.

Each arm selects one global learning rate from two complete training candidates
at dedicated tuning seed 9. Final optimization seeds are 10–19. All 13 global
choices and all 130 final fits must be saved and verified before confirmation
inference is enabled. Cross-entropy arms freeze their calibration-only logit
adjustment choice before inferring confirmation logits. Model training,
learning-rate selection and calibration cannot use confirmation labels.

## Model and loss controls

The official `google-bert/bert-base-uncased` model is pinned to revision
`86b5e0934494bd15c9632b12f734a8a67f723594`. Its 440,449,768-byte
`model.safetensors` has official SHA256
`68d45e234eb4a928074dfd868cead0219ab85354cc53d20e772753c6bb9169d3`.
The model card identifies Apache 2.0 licensing. Ordinary configuration and
vocabulary assets are verified against their official Git blob digests; model
weights are verified against the official LFS digest. Loading uses local
safetensors only, with no remote code execution or downloaded pickle weights.

Title and content are separate WordPiece input segments. Inputs are right-padded
and truncated by longest-first truncation to 128 positions, including special
tokens. Attention-masked mean pooling uses the final contextual hidden states,
includes CLS/SEP and excludes PAD. It accumulates in float32.

| Arm group | Training regime | Objectives |
|---|---|---|
| Frozen BERT predictive baseline | 50 epochs, 1,024 examples/batch, SGD | CE; Balanced Softmax; natural CE+EQ; class-uniform CE; class-uniform CE+EQ |
| Random mean-pooled baseline | 50 epochs, 1,024 examples/batch, SGD | CE; Balanced Softmax |
| Matched frozen BERT control | 3 epochs, 1,024 examples/batch, AdamW | CE; Balanced Softmax |
| Last transformer block tuned | 3 epochs, 1,024 examples/batch, AdamW | CE; Balanced Softmax |
| Last two transformer blocks tuned | 3 epochs, 1,024 examples/batch, AdamW | CE; Balanced Softmax |

There are 13 arms, 26 dedicated tuning fits, and 130 final fits. Frozen and
mean-pooled learning rates are selected from 0.01/0.05. Matched and partially
tuned learning rates use reference values 0.00002/0.00005; the classifier rate
is ten times the reference, and the trainable transformer layers use the
reference itself. Every final fit reaches its declared epoch budget. The
frozen/mean-pooled methods have 5,200 updates. The three-epoch methods have 318
updates, with matched deterministic length-bucket batch lists.

All BERT arms use a 768→256→128 adapter with dropout 0.1 and a 14-class linear classifier.
Their classifier initial states match within each final seed. The 50-epoch
frozen arms use attention-masked mean representations and a fit-only mean-feature
scaler. The matched three-epoch frozen/last-one/last-two group instead uses the
standard final CLS representation and a separate shared fit-only CLS scaler. Mean pooling uses
a fit-only 50,000-word vocabulary and the earlier randomly initialized
128-dimensional embedding-bag architecture with a new 128-word cap. Unicode
words and BERT WordPiece positions are different token units, so the numerical
cap does not imply identical retained text coverage or adapter input width.

Balanced Softmax trains cross-entropy on `logits + log(fit class counts)` under
natural row sampling. It uses raw logits at evaluation, reflecting its uniform
effective classification prior. Class-uniform controls sample classes uniformly
and real members uniformly within class, with replacement, while retaining the
same number of examples and optimizer updates as natural sampling. They use CE
without an additional class weight. EQ has weight 0.05 throughout the 50-epoch
adapter fit. Its per-class eligibility counts are retained in every epoch.
Natural CE and natural CE+EQ use their fitted training prior for calibration;
Balanced Softmax and class-uniform objectives do not receive a second prior
correction. Calibration selects tau from
`[0,.25,.5,.75,1,1.25,1.5,2]` by macro-F1, breaking ties toward the smallest tau.

## Compute and cache controls

All 630,000 original source rows receive frozen features. A single extraction
pass also saves first-ten-layer hidden states for the 218,232 fit/tune/calibration/
confirmation rows, so later tuning does not repeatedly run the frozen prefix.
The prefix cache is float16; downstream frozen and partial models share its
precision. Pooled feature storage is float32. Extraction uses official SDPA
attention and MPS float16 autocast. Model master parameters stay float32 during
partial tuning, with MPS gradient scaling initialized to 1,024. A prospective amendment made before text training or confirmation uses CLS
pooling for the matched three-epoch group. Its final transformer layer computes
only the CLS query, retaining key/value projections for every token, then the
CLS output projection, residual, layer norms and FFN. The other final-layer
query outputs cannot affect CLS within that layer. An independent double-precision
fixture at dropout zero matched full-layer CLS outputs and every input/parameter
gradient, with maximum errors below 2.3e-15. With training dropout this preserves
the CLS marginal distribution and changes the RNG realization. The separate
frozen CLS cache is derived from the same first-ten-layer prefix. The 50-epoch
mean-feature extraction is unchanged by this amendment.

The first ten
blocks remain frozen in evaluation mode. For last-one-block tuning, the eleventh block
also remains frozen in evaluation mode; the twelfth block trains. For last-two
block tuning, both final blocks train.

Length buckets at 32/64/96/128 positions remove masked PAD computation while
retaining original valid token IDs and absolute positions. Partial and matched
frozen controls use the same complete per-epoch row sets and shuffled batch
lists. The fit bucket counts are 11,020/36,002/37,481/21,729, giving 106 updates
per epoch at batch size 1,024. A fit-only throughput pilot selected batch size 1,024 after comparing 512 and
1,024. The three-epoch row coverage is unchanged; the prescribed update budget
is 318 rather than the earlier 512-batch estimate of 630. No tuning, calibration
or confirmation accuracy informed this batch-size decision.

Transformer dropout ordering changes with the
prescribed batch layout; this is fixed before production training.

A train-only validation against original float32 full BERT found maximum pooled
relative L2 error 0.000517 for the optimized extraction, below the prescribed
0.005 tolerance. Separate invariant checks test masked pooling, token truncation,
source alignment, disjoint deterministic partitions, Balanced Softmax's formula,
partial freezing/dropout modes, SDPA output/gradient agreement, RNG recovery,
and corruption rejection in the committed cache journal. Synthetic full-pipeline
preflight verifies tuning, the global confirmation gate, checkpoint budgets,
calibration, numeric replay and resume without using real held-out examples.

Cache progress uses an immutable identity reservation and a contiguous journal.
Every committed feature block and its corresponding saved prefix rows have
SHA256 digests. All previously committed blocks are rehashed before resuming.
Completed caches have full-file and row-ID digests. Checkpoint identity, completed
epochs, histories, planned updates, candidate selection and numeric logits are
verified before resuming or aggregating. A wrong identity or inconsistent budget
is preserved for inspection rather than overwritten.

## Declared inference

Macro-F1 is the primary endpoint; accuracy, macro recall, head/mid/tail F1 and
class-level metrics remain visible. Eight paired comparisons share one
Holm-corrected family: frozen BERT CE+LA versus the new mean-pooled CE+LA;
frozen Balanced Softmax versus frozen CE+LA; EQ on versus off within each of
natural/class-uniform sampling; last-one/last-two CE+LA versus matched frozen
CE+LA; and last-one/last-two Balanced Softmax versus matched frozen Balanced
Softmax. Both paired t intervals and exact two-sided sign-flip p values are
reported. Ten seeds allow minimum exact p = 2/1024; this is separate from
independence across datasets.

The 50-epoch frozen models establish predictive baselines. Selective tuning is
compared primarily with the matched three-epoch frozen CLS controls. The
50-epoch mean-pooled baseline remains a predictive reference; the pooling and
training-budget differences preclude attributing its contrast to tuning alone. Activating
transformer dropout in trainable blocks is part of the tuning regime, so these
comparisons do not isolate gradients from every other training-mode effect.
The matched head-only control uses CPU/float32, while selective tuning uses
MPS/float16 autocast, so the contrast includes numerical/backend differences as
well as transformer dropout. It is a training/precision/backend regime comparison.
Class-uniform and natural objectives differ; the sampler×EQ table does not
support a universal factorial interaction or unchanged-objective sampler claim.

BERT was pretrained on Wikipedia, and DBpedia abstracts are Wikipedia-derived.
Fresh supervised partitions and normalized duplicate removal do not establish
independence from pretraining content. This is a transfer/model-package study;
results cannot be attributed solely to the transformer architecture or described
as unseen-pretraining generalization. Entity/paraphrase duplicates remain a
limitation beyond the documented exact normalization rules.

## Reproduction

Install the pinned text dependencies recorded in the experiment environment and
run acquisition/preparation separately from modeling:

```sh
python scripts/extract_text_followup_features.py acquire --out data/bert-base-uncased
python scripts/extract_text_followup_features.py prepare --source data/sources/dbpedia --out data/text_followup --rho 100
python scripts/extract_text_followup_features.py tokenize --data data/text_followup --modeldir data/bert-base-uncased
python scripts/extract_text_followup_features.py extract --data data/text_followup --modeldir data/bert-base-uncased --out data/bert_cache --device mps --batch-size 64 --threads 2 --prefix-cache --precision amp_float16 --attention sdpa --length-buckets
python scripts/benchmark_text_followup.py cls-cache --data data/text_followup --cache data/bert_cache --modeldir data/bert-base-uncased --out runs/text_followup --device mps
```

Train CPU and MPS workers against one shared full protocol, with no confirmation
inference during training:

```sh
python scripts/benchmark_text_followup.py run --data data/text_followup --cache data/bert_cache --modeldir data/bert-base-uncased --out runs/text_followup --device cpu --stage train --only-methods bert_ce bert_bsoftmax bert_natural_eq bert_balanced_ce bert_balanced_eq mean_ce mean_bsoftmax last0_ce last0_bsoftmax
python scripts/benchmark_text_followup.py run --data data/text_followup --cache data/bert_cache --modeldir data/bert-base-uncased --out runs/text_followup --device mps --stage train --only-methods last1_ce last1_bsoftmax last2_ce last2_bsoftmax
```

After all choices and all 130 final fits are complete, repeat each worker command
with `--stage evaluate`. Then run independent numeric replay:

```sh
python scripts/benchmark_text_followup.py verify --out runs/text_followup
python -m unittest discover -s tests -p test_text_followup.py -v
```

Official sources: [BERT model assets](https://huggingface.co/google-bert/bert-base-uncased),
[BERT paper](https://arxiv.org/abs/1810.04805),
[Hugging Face BERT documentation](https://huggingface.co/docs/transformers/model_doc/bert),
[DBpedia-14 dataset](https://huggingface.co/datasets/fancyzhx/dbpedia_14), and
[Balanced Softmax paper](https://proceedings.nips.cc/paper_files/paper/2020/file/2ba61cc3a8f44143e1f2f13b2b729ab3-Paper.pdf).
