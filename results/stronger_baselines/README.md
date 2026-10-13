# REBAL stronger-baseline follow-up

The completed follow-up covers Tiny ImageNet, DBpedia-14 and Covertype with ten paired optimization seeds, protected current-round confirmation partitions, stronger controls and independent numeric verification. The report retains positive effects, negative effects and non-detections, with six image-EQ, five image-encoder, eight text and three tabular primary comparisons. The combined 22-comparison Holm analysis is a separate sensitivity analysis.

- [Experimental report](report/experimental_report.pdf): the corrected 23-page PDF.
- [Reproduction guide](reproduction/README.md): archive identity checks, numeric replay, paired analysis and reporting commands.
- [Final presentation evidence](evidence.json): bound model rows, comparisons, references and figure provenance.
- [Assembled evidence](evidence_assembled.json): the unchanged input identified by the finalizer's recorded input SHA-256.
- [Global paired analysis](global_analysis.json): all 22 declared comparisons and both Holm families.
- [Numeric report QA](report/numeric_qa.json), [all-page visual QA](report/visual_qa.json), and [figure QA](report/figure_visual_qa.json).

| Study directory | Completed fitted models | Confirmation readouts | Primary paired comparisons |
| --- | --- | ---: | ---: |
| [Equalization](equalization/) | 320 tuning adapters and 220 separately fitted linear heads | 660 | 6 |
| [Vision](vision/) | 12 tuning and 60 final fits | 120 | 5 |
| [Text](text/) | 26 tuning and 130 final fits | 190 | 8 |
| [Tabular](tabular/) | 8 tuning and 40 final fits | 80 | 3 |

Named readouts and uniform-prior aliases are not independent fitted models. Vision additionally retains 60 official-test exploratory readouts; Text retains 100. Current-round confirmation protects this round's selection, while historical model/source reuse and foundation-model pretraining remain limitations. The report explains the decision-offset selection rules, selected-pipeline learning-rate differences, cached-feature BCL adaptation and encoder-package confounding. Decision offsets were selected by macro-F1; the studies do not establish probability calibration or a fully matched GAN-versus-ViT intervention.

This Git scope contains derived results, protocols, frozen choices, figures, the report and audit receipts. Files named `raw_evidence_catalog.json` describe complete **separate local weight-free raw bundles**; their multi-gigabyte predictions and source label/split arrays are not uploaded here. Full numeric replay needs that local archive and the unchanged bundle paths. No images, input text, tabular feature rows, tokens, feature caches, pretrained/trained weights or checkpoints are included in this Git evidence scope.

Original working-run actual-state audits and portable replay have different coverage: 540 EQ states, 72 Vision states and 156 Text states were checked originally; Tabular covers 48 trained identities across 84 weight files, with a separate 78-selected-artifact integrity overlay. Portable replay loads zero model states. These receipts establish saved numeric and retained-state checks, not independent retraining, encoder extraction, external replication or manifest authenticity.

Text's [143-record Git mapping](text/git_record_mapping.json) retains exact bytes while presenting 13 global configurations and 130 calibration records under compact unique names. The full archive keeps its original Text paths. Equalization's ten amendment seed directories are unchanged. Text operational compatibility companions remain outside its immutable raw archive tree; their narrow original-layout recovery scope is explained in the reproduction guide.

The final presentation was assembled before a layout-only change to body spacing. Its historical builder SHA-256 resolves to [the preserved formatter](companions/report/build_followup_report_before_layout.py); [the public revision receipt](report/layout_revision.json) binds that formatter to the current reporting source without changing scientific evidence. `evidence.json` is the final presentation; the historical provenance filename `evidence.json` for its input is resolved by digest to the separately supplied `evidence_assembled.json`.

Each study's derived manifest inventories its smaller Git set. [The root evidence catalog](root_evidence_manifest.json) inventories these report extras. Publication and combined ZIP creation are verified separately; preparing these files does not assert that either action has completed.
