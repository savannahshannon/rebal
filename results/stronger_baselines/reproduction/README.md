# REBAL stronger-baseline reproduction guide

This guide documents four completed studies and their independently verified
numeric evidence. All four weight-free portable bundles passed exact privacy
checks, and the smaller 293-file derived Git evidence set passed its own scan.
These completion facts do not assert that a combined ZIP has been created,
every report page has passed visual review, or files have been published.
The final artifact identities, archive checks, report QA and publication
readbacks are recorded separately.

The experiments use three public sources: Tiny ImageNet (110,000 source
images), DBpedia-14 (630,000 source records), and Covertype (581,012 source
rows). The two image studies share an immutable current-round partition.
Training uses the declared fit partitions, rather than every source record.
The report identifies the actual fit counts and source canonicalization.

| Study | Tuning fits | Other/final fits | Completed budget per fit |
| --- | ---: | ---: | --- |
| Image EQ and sampling | 320 adapters | 220 separately trained linear heads | 50 adapter epochs/750 updates; 2,000 head updates |
| Image encoder and partial tuning | 12 | 60 | 15 epochs/1,800 AdamW updates |
| Text controls and partial tuning | 26 | 130 | 50 epochs/5,200 updates for the seven CPU adapter/mean arms; 3 epochs/318 updates for six matched CLS arms |
| Tabular controls | 8 | 40 | 50 neural epochs; 600 completed boosting rounds, with the readout iteration selected at 50-round intervals through 600 |

All final comparisons use paired optimization seeds 10–19. Separate development
partitions select configurations, checkpoints and decision offsets. Confirmation
macro-F1 is the primary endpoint. The six EQ, five image, eight text and three
tabular contrast definitions are fixed; the 22-comparison global Holm table is
a sensitivity analysis in addition to the primary within-study families.

The archived image reporting identifier compatibility note maps one frozen
runner identifier to the existing report-analysis identifier. It changes no
paired observations, formulas, family sizes, model choices or scientific run.

## Archive layout and numerical replay

The combined local archive layout retains the complete four portable prediction bundles,
their original catalogs and audit receipts, source/protocol identities, scalar
and class results, scientific figures, and the report PDF. It excludes raw
images, raw text, tabular feature rows, token arrays, dense feature/prefix
caches, pretrained weights, model and optimizer states, credentials and private
operational records. SHA256 identities for omitted scientific assets are
retained where the protocols require them.

Packaging uses ZIP64. The external manifest inventories every member by
name, size, SHA256 and approved numeric schema. Compare that manifest and the
final scan receipt before replay. A digest establishes artifact identity;
independent numerical replay establishes the reported calculations.

The standard-library companion
`reproduction/work/verify_followup_archive_identity.py` checks the received ZIP
and every member against the supplied external manifest. From the extracted
archive directory, replace the example input names with the received paths and
choose a fresh receipt outside the catalogued bundle:

```bash
python reproduction/work/verify_followup_archive_identity.py \
  --archive ../ARCHIVE.zip \
  --manifest ../EXTERNAL_MANIFEST.json \
  --out ../FRESH_IDENTITY_RECEIPT.json
```

This checker streams the archive and members, verifies exact unique names,
sizes, SHA256 values and ZIP CRCs, and rejects unsafe paths, encryption and
symlink members. It does not extract files or load models. Its receipt establishes
identity against the supplied manifest only; it does not authenticate that
manifest, run scientific numeric replay, verify training or perform a privacy
scan. The separate scientific and privacy receipts retain those distinct
scopes. No reporting environment or external Python packages are needed for
this identity-only command.

Use the numeric verification command documented for each portable bundle.
Portable replay must report zero loaded checkpoint states. The preserved
original working-run audits separately cover 540 EQ adapter/head states,
72 image states and 156 text states. The tabular actual-state audit covers
48 trained model fits/identities across 84 weight files: 72 neural checkpoints
(36 best and 36 last states) and 12 boosted-tree artifacts. Its separate
integrity/selection overlay checks 78 selected artifacts, a different scope.
Those original
audit receipts do not imply that the omitted states can be inspected from the
portable archive, or that training and all encoder inference were independently
rerun.

Check the original catalogs and their hashes before running any verifier.
Work on a disposable copy of the extracted archive, or explicitly send each
new receipt to an `--out` path outside the catalogued bundle. Some verification
commands otherwise replace an existing timestamped receipt even when every
numeric value agrees. That replacement changes its SHA256 and invalidates the
original catalog. Keep the supplied receipts and catalogs unchanged; retain
new replay receipts separately. The following reporting commands also create
new files and should be run in a disposable working copy.

The current public study guides provide the exact acquisition, preprocessing,
training and replay commands. Package guides are preserved at their export
revision; current companion guides and the compatibility note accompany the
reporting code. The tabular numeric helper's public documentation correction
preserves its executable code; the original helper remains in its frozen
working-run bundle with its recorded source identity.

## Recompute the paired analysis

From the reporting reproduction directory, run:

```bash
python -m venv reporting-env
source reporting-env/bin/activate
pip install -r work/next_round/requirements-reporting.txt
python work/rebal/scripts/analyze_followup_comparisons.py \
  --eq ../equalization/aggregate.json \
  --vision ../vision/run/aggregate.json \
  --text ../text/experiment/aggregate.json \
  --tabular ../tabular/aggregate.json \
  --out ../global_analysis_reproduced.json
```

This requires all four complete aggregates. It replays the paired differences,
sample standard deviations, paired-t intervals, all 1,024 exact sign assignments,
and both Holm corrections. It rejects incomplete seed grids and undeclared or
duplicated comparison identities. The immutable aggregate digests and analysis
helper digest are retained in the resulting JSON.
The sign-flip interpretation assumes paired sign exchangeability or symmetry
under the null; the paired-t interval approximates uncertainty in the mean of
ten optimization-seed differences on the fixed partition.

The reporting assembler and builder bind their displayed model rows and
contrast fields to these audited source records. Rebuilding with a new
authoring timestamp can change the PDF bytes; the numerical results remain
bound to the retained source identities. Render and visually inspect every page
after any report-formatting change.

From the same `reproduction` directory and activated reporting environment,
assemble the complete evidence, create the five scientific figures, and check
the presentation payload before rendering. The companion source
`reproduction/work/finalize_followup_presentation.py` is part of the mirrored
reporting layout:

```bash
python work/assemble_followup_evidence.py \
  --evidence-root .. \
  --global-analysis ../global_analysis_reproduced.json \
  --references work/next_round/report_references.json \
  --out ../evidence_reproduced.json
MPLCONFIGDIR=./reporting-mpl-cache python work/next_round/plot_results.py \
  --evidence ../evidence_reproduced.json \
  --out ../figures_reproduced
python work/finalize_followup_presentation.py \
  --evidence ../evidence_reproduced.json \
  --figures ../figures_reproduced \
  --out ../evidence_presentation_reproduced.json
python work/build_followup_report.py \
  --evidence ../evidence_presentation_reproduced.json \
  --out ../REBAL_Followup_Report_reproduced.pdf \
  --check-only
python work/build_followup_report.py \
  --evidence ../evidence_presentation_reproduced.json \
  --out ../REBAL_Followup_Report_reproduced.pdf
python work/verify_followup_pdf_text.py \
  --evidence ../evidence_presentation_reproduced.json \
  --pdf ../REBAL_Followup_Report_reproduced.pdf \
  --out ../report_numeric_recheck.json
```

The assembler initially produces a table-based report payload without embedded
figures. The plotter writes exactly five declared PNG files into the new
`figures_reproduced` directory. The finalizer requires that directory to be
contained within the evidence directory and to contain those five files only.
It binds their hashes, embeds the four study figures, retains the global figure
as a standalone artifact, and derives four interpretation paragraphs from the
already verified summaries and primary comparisons. It preserves the numeric,
statistical and source fields and checks the new payload through the builder.
The input evidence remains unchanged; the presentation payload is a fresh JSON
file. The finalizer does not create a PDF.

The presentation payload's immutable `figure_status` records its state at
finalization: four study figures are hash-bound, the global figure is standalone,
and actual PDF production and all-page visual QA remain pending. The original
assembler status is retained in presentation provenance. Subsequent numeric
and visual QA receipts establish what was actually checked after rendering;
the presentation-time status itself is not a claim that those later checks
have happened. Keep those receipts separate from the catalogued originals.

The PDF text check associates model rows with each study's
confirmation table, primary contrasts with its paired-comparison table, and
global rows with their declared sensitivity-table half. It supplements the
source-bound builder checks; it does not establish spatial layout. Render and
visually inspect every page separately before sharing a rebuilt PDF.

For the complete bundle verification and training commands, use the separate
follow-up environment in the current study guides. Its included
`requirements-followup.txt` refers to the accompanying
`requirements-vit.txt` and `requirements-tiny.txt`. Reporting dependencies above
are enough for the paired analysis and PDF tools; they are not a replacement
for training dependencies or omitted training assets.

## Git publication scope

The Git evidence scope under `results/stronger_baselines` consists of smaller derived results, audited
summaries, protocols, frozen choices, original and portable verification
receipts, scientific figures and the final report. Files named
`raw_evidence_catalog.json` describe the complete local raw bundles; their
presence does not mean that the multi-gigabyte prediction arrays were uploaded
to Git. Raw prediction replay requires the supplied local archive.

For the small Git presentation only, the 13 Text configuration receipts use
`text/frozen_choices/<method>__hyperparameters.json`, and the 130 calibration
receipts use `text/calibration/<method>__seed_<seed>__calibration_choice.json`.
The public `text/git_record_mapping.json` lists each original bundle-relative
path, complete Git-relative target, byte count and SHA256. This compact layout
retains every receipt and its exact bytes. The full raw archive keeps the
original `experiment/<method>/...` paths; its catalogs and numeric replay
commands use those unchanged paths. This mapping is created from the completed,
privacy-scanned bundle and is included in the final Git scan. The ten EQ
`public_amendment_inputs/confirmation/seed_<seed>/metrics.json` paths are
preserved.

A separately catalogued Text `posttraining_execution.json` companion records
completion of the two confirmation phases and aggregate verification, plus
unchanged training-artifact hashes. It accompanies the small Git evidence and
combined archive after its own completion and exact privacy checks; the frozen
Text exporter and original raw bundle catalog are unchanged. In the combined
ZIP, this external receipt is stored at
`companions/text/posttraining_execution.json`; evaluation compatibility or
recovery receipts, if included, also stay under `companions/text/`. The archive
`text/` directory contains exactly the immutable exported raw bundle, with no
uncatalogued operational extras. The small Git derived tree instead catalogs
its separate `text/posttraining_execution.json` in its derived manifest. This
operational receipt supplements the independent numeric and actual-state
audits without claiming an independent training or inference reproduction.

The separately scanned RNG-only checkpoint compatibility adapter records the
local evaluation recovery. Its usage guide is `companions/text/usage.md` in the
combined archive layout. It is tied to the original `work/next_round/text/`
location, fixed MPS evaluation arguments, and original model, cache, data and
output assets. A copy under `companions/text/` cannot run directly as a generic
portable evaluator; reuse requires restoring that original layout and assets,
many of which are omitted from this archive. Its eight isolated generator
fixtures check the compatibility adapter only; they are separate from the
130-model/190-confirmation/100-native scientific evidence. Portable numeric
replay uses the independent numeric verifier and does not depend on this
operational adapter.

The publication allowlist is explicit. No model states, feature caches, raw
training inputs or authentication material are added to Git. A publication
receipt must bind independently re-read published bytes to the reviewed SHA256
values; preparing or scanning an upload candidate does not establish publication.

## Interpretation boundaries

Ten seeds measure optimization variation conditional on the source partitions
and selected configurations. They do not quantify uncertainty across external
datasets, geographic locations, entities or foundation-model pretraining
corpora. Current-round split disjointness does not establish independence from
historical training or pretraining. The report retains non-detections and
negative effects alongside improvements.

The DINO/DeiT comparison changes the whole encoder package, including capacity
and pretraining. Partial text tuning also changes the prescribed precision,
backend and dropout regime. Cached-feature BCL is an adaptation rather than a
full image-augmentation reproduction. The studies do not provide a fully
matched GAN-versus-ViT intervention, full-backbone fine-tuning, or a probability
reliability evaluation. Shared-machine timings document this execution rather
than isolated efficiency superiority.

Text EQ uses coefficient 0.05 throughout all 50 adapter epochs, whereas the
fixed image EQ treatment begins at fraction 0.8 of training. Each Text arm
selects its own learning rate, so its objective/EQ contrasts compare selected
pipelines and may include learning-rate differences.
