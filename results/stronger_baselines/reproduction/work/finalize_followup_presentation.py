#!/usr/bin/env python3
"""Add presentation fields only after the complete four-study report gate.

This helper does not train, infer, calculate new statistics, select a model,
render figures or author a PDF. It preserves the numeric/statistical/source
payload, appends four study-specific recommendation paragraphs and four
hash-bound study figures, and updates only the presentation figure-status field
with its prior value retained in provenance. Run it only after the root-owned plotter and
visual review. The fifth, global figure remains a standalone archive artifact.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import importlib.util
import json
import math
import tempfile

STUDIES = ('equalization', 'vision', 'text', 'tabular')
FIGURES = {
    'global_paired_effects': 'rebal_followup_paired_effects.png',
    **{name: 'rebal_followup_' + name + '_performance.png' for name in STUDIES},
}
FIGURE_ROLE = 'scientific_presentation_figure'
PRESENTATION_FIGURE_STATUS = ('Four study figures hash-bound for embedding; global figure retained standalone; '
                             'actual PDF and all-page visual QA pending.')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def number(value):
    require(type(value) in (int, float) and math.isfinite(value),
            'Recommendation requires finite numeric source values')
    return value


def summary_metric(aggregate, method, metric='macro_f1'):
    rows = [row for row in aggregate['summary'] if row.get('method') == method]
    require(len(rows) == 1 and metric in rows[0]['metrics'],
            'Recommendation summary selector is missing or ambiguous: ' + method + '/' + metric)
    return number(rows[0]['metrics'][metric]['mean'])


def vision_metric(aggregate, arm, metric='macro_f1'):
    rows = [row for row in aggregate['summary']
            if row['arm'] == arm and row['endpoint'] == 'confirmation_la' and row['metric'] == metric]
    require(len(rows) == 1, 'Recommendation vision selector is missing or ambiguous')
    return number(rows[0]['mean'])


def contrast_text(comparisons, study, name, description):
    rows = [row for row in comparisons if row['study'] == study and row['name'] == name]
    require(len(rows) == 1, 'Recommendation comparison is missing or ambiguous')
    row = rows[0]
    mean = number(row['mean']); lo, hi = map(number, row['ci95'])
    p = number(row['signflip_p_study_holm'])
    require(0 <= p <= 1 and lo <= hi, 'Invalid recommendation comparison interval or p-value')
    if p <= .05 and mean > 0:
        interpretation = 'a positive effect is detected'
    elif p <= .05 and mean < 0:
        interpretation = 'a negative effect is detected'
    else:
        interpretation = 'no nonzero effect is detected; this does not establish equivalence'
    return (f"{description}: {100*mean:+.4f} pp, 95% paired interval "
            f"[{100*lo:+.4f}, {100*hi:+.4f}], study Holm p={p:.6f}; {interpretation}.")


def derive_recommendations(evidence, aggregates):
    """Describe existing means/contrasts; never replace an endpoint or choice."""
    comparisons = evidence['global_analysis']['comparisons']
    eq, vision, text, tab = (aggregates[name] for name in STUDIES)
    c = lambda study, name, label: contrast_text(comparisons, study, name, label)
    eq_effects = [
        ('fixed_natural_EQ_effect', 'Fixed EQ on minus off with natural sampling'),
        ('fixed_balanced_EQ_effect', 'Fixed EQ on minus off with class-uniform sampling'),
        ('sampling_effect_EQoff', 'Class-uniform minus natural sampling with EQ off'),
        ('fixed_factorial_interaction', 'Fixed EQ-by-sampling interaction'),
        ('selected_balanced_softmax_vs_selected_ce_la', 'Selected Balanced Softmax minus selected CE + LA'),
        ('selected_bcl_vs_selected_balanced_softmax', 'Selected BCL adaptation minus selected Balanced Softmax'),
    ]
    eq_note = ('Image EQ and sampling: retain the fixed-treatment comparisons separately from the selected stronger-loss pipelines. '
        + ' '.join(c('equalization', name, label) for name, label in eq_effects)
        + f" Selected native adjusted macro-F1 means are CE {100*summary_metric(eq,'ce__native_la'):.3f}%, "
          f"Balanced Softmax {100*summary_metric(eq,'balanced_softmax__native_la'):.3f}% and "
          f"BCL adaptation {100*summary_metric(eq,'bcl_dropout__native_la'):.3f}%. "
        + 'Keep CE and Balanced Softmax as controls. The fixed natural/class-uniform policies target the same class-balanced objective; '
          'selected loss pipelines can also differ in their selected settings. BCL uses cached features and adapter-dropout views, '
          'rather than full-backbone image augmentation. New natural/ROS heads are diagnostic readouts, not independent encoder fits.')
    vision_effects = [
        ('partial_vs_frozen_ce', 'Last-two-block DeiT CE + LA minus frozen DeiT CE + LA'),
        ('dino_vs_deit_ce', 'Frozen DINOv2 CE + LA minus frozen DeiT CE + LA'),
        ('balanced_softmax_vs_ce_deit_frozen', 'Balanced Softmax minus CE + LA for frozen DeiT'),
        ('balanced_softmax_vs_ce_deit_last2', 'Balanced Softmax minus CE + LA for last-two-block DeiT'),
        ('balanced_softmax_vs_ce_dino', 'Balanced Softmax minus CE + LA for frozen DINOv2'),
    ]
    vision_note = ('Image encoders: compare each loss with its matched encoder control. '
        + ' '.join(c('vision', name, label) for name, label in vision_effects)
        + f" Adjusted CE macro-F1 means are frozen DeiT {100*vision_metric(vision,'deit_frozen__ce'):.3f}%, "
          f"last-two-block DeiT {100*vision_metric(vision,'deit_last2__ce'):.3f}% and "
          f"frozen DINOv2 {100*vision_metric(vision,'dino_frozen__ce'):.3f}%. "
        + 'DINOv2 versus DeiT changes the whole encoder package, including capacity and pretraining, so its contrast does not isolate '
          'an architecture or loss intervention. Partial versus frozen DeiT compares the declared selected pipelines. '
          'Experimental partition disjointness does not establish independence from ImageNet pretraining.')
    text_effects = [
        ('last1_ce_la__minus__last0_ce_la', 'Last-one-block CE + LA minus matched frozen CLS CE + LA'),
        ('last2_ce_la__minus__last0_ce_la', 'Last-two-block CE + LA minus matched frozen CLS CE + LA'),
        ('last1_bsoftmax__minus__last0_bsoftmax', 'Last-one-block Balanced Softmax minus matched frozen CLS Balanced Softmax'),
        ('last2_bsoftmax__minus__last0_bsoftmax', 'Last-two-block Balanced Softmax minus matched frozen CLS Balanced Softmax'),
    ]
    text_note = ('Text selective tuning: use the four declared partial-versus-matched-frozen comparisons to assess tuning. '
        + ' '.join(c('text', name, label) for name, label in text_effects)
        + f" Matched frozen CLS macro-F1 means are CE + LA {100*summary_metric(text,'last0_ce_la'):.3f}% and "
          f"Balanced Softmax {100*summary_metric(text,'last0_bsoftmax'):.3f}%. "
        + 'Each arm selects its own learning rate; the prescribed tuning groups also change backend, precision and dropout. '
          'These effects therefore describe selected pipelines, not an isolated trainability toggle. The fifty-epoch mean-pooled '
          'BERT references use different pooling and budgets from the matched three-epoch CLS comparison. '
          'Do not substitute those references for the matched frozen control or interpret a non-detection as equivalence.')
    tab_effects = [
        ('xgboost_la__minus__mlp_ce_la', 'Adjusted XGBoost minus adjusted natural CE MLP'),
        ('ft_transformer_la__minus__mlp_ce_la', 'Adjusted local FT-Transformer minus adjusted natural CE MLP'),
        ('mlp_cb_la__minus__mlp_ce_la', 'Adjusted effective-number MLP minus adjusted natural CE MLP'),
    ]
    tab_note = ('Tabular models: retain boosted trees and conventional CE MLP as predictive controls under the natural data distribution. '
        + ' '.join(c('tabular', name, label) for name, label in tab_effects)
        + f" Adjusted macro-F1 means are XGBoost {100*summary_metric(tab,'xgboost_la'):.3f}% and "
          f"CE MLP {100*summary_metric(tab,'mlp_ce_la'):.3f}%. Effective-number versus CE MLP means are "
          f"macro recall {100*summary_metric(tab,'mlp_cb_la','macro_recall'):.3f}% versus "
          f"{100*summary_metric(tab,'mlp_ce_la','macro_recall'):.3f}%, uniform-prior diagnostic macro-F1 "
          f"{100*summary_metric(tab,'mlp_cb_la','balanced_macro_f1'):.3f}% versus "
          f"{100*summary_metric(tab,'mlp_ce_la','balanced_macro_f1'):.3f}%, native-prior macro-F1 "
          f"{100*summary_metric(tab,'mlp_cb_la'):.3f}% versus {100*summary_metric(tab,'mlp_ce_la'):.3f}% and "
          f"tail F1 {100*summary_metric(tab,'mlp_cb_la','tail_f1'):.3f}% versus "
          f"{100*summary_metric(tab,'mlp_ce_la','tail_f1'):.3f}%. XGBoost macro-F1 is "
          f"{100*summary_metric(tab,'xgboost_raw'):.3f}% raw and {100*summary_metric(tab,'xgboost_la'):.3f}% adjusted; "
          f"its macro recall is {100*summary_metric(tab,'xgboost_raw','macro_recall'):.3f}% raw and "
          f"{100*summary_metric(tab,'xgboost_la','macro_recall'):.3f}% adjusted, and its uniform-prior diagnostic macro-F1 is "
          f"{100*summary_metric(tab,'xgboost_raw','balanced_macro_f1'):.3f}% raw and "
          f"{100*summary_metric(tab,'xgboost_la','balanced_macro_f1'):.3f}% adjusted. "
        + 'Preserve the declared adjusted primary endpoint and both raw/adjusted views; calibration-set decision selection need not '
          'improve confirmation macro-F1. Uniform-prior diagnostics reweight the same records, not a separate dataset. '
          'The local FT result does not rank all transformer configurations. Evaluate future objectives or decision policies on new '
          'prespecified data rather than selecting them from confirmation results.')
    result = [eq_note, vision_note, text_note, tab_note]
    require(all(note.isascii() for note in result), 'Presentation recommendations must use ASCII punctuation')
    return result


def figure_identities(root, figures):
    root = Path(root).absolute(); figures = Path(figures).absolute()
    require(not root.is_symlink(), 'Symlinked evidence root is forbidden')
    require(figures.is_dir() and not figures.is_symlink(), 'Figure directory missing or symlinked')
    canonical_root = root.resolve(); canonical_figures = figures.resolve()
    try:
        relative = canonical_figures.relative_to(canonical_root)
    except ValueError as error:
        raise ValueError('Figures must be contained within the evidence directory') from error
    require(relative.parts and '..' not in relative.parts, 'Unsafe figure directory')
    # Resolve OS aliases above the evidence root consistently (macOS /var versus
    # /private/var), while inspecting the original spelling of every descendant
    # to reject internal or external symlink shortcuts into that root.
    cursor = figures
    while True:
        require(not cursor.is_symlink(), 'Symlinked figure path is forbidden')
        if cursor.resolve() == canonical_root:
            break
        require(cursor != cursor.parent, 'Figure ancestry does not reach evidence root')
        cursor = cursor.parent
    root = canonical_root; figures = canonical_figures
    entries = list(figures.iterdir())
    require({p.name for p in entries} == set(FIGURES.values()) and len(entries) == 5,
            'Exactly the five declared plotter PNG files are required; missing or unknown figure')
    result = []
    for role, filename in FIGURES.items():
        path = figures/filename
        require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root),
                'Figure is not a contained regular file')
        require(path.read_bytes()[:8] == b'\x89PNG\r\n\x1a\n', 'Declared figure is not a PNG')
        from PIL import Image
        with Image.open(path) as image:
            require(image.format == 'PNG' and min(image.size) > 0, 'Invalid PNG figure')
            width, height = image.size; image.verify()
        result.append(dict(role=role, path=path.relative_to(root).as_posix(), bytes=path.stat().st_size,
                           sha256=sha(path), width=width, height=height, embedded=role in STUDIES))
    return result


def preserved_projection(presented, original):
    """Strip only this helper's declared additions and compare exact structures."""
    restored = copy.deepcopy(presented)
    provenance = restored.pop('presentation_provenance')
    require(provenance['prior_figure_status'] == original['figure_status'],
            'Original figure status provenance changed')
    require(restored['figure_status'] == PRESENTATION_FIGURE_STATUS,
            'Declared presentation figure status differs')
    restored['figure_status'] = provenance['prior_figure_status']
    require(provenance['recommendation_paragraphs_added'] == 4, 'Presentation paragraph count differs')
    require(restored['recommendation_notes'][:-4] == original['recommendation_notes'],
            'Original recommendation notes changed')
    restored['recommendation_notes'] = restored['recommendation_notes'][:-4]
    for name in STUDIES:
        study = restored['studies'][name]
        require('figure' not in original['studies'][name], 'Input already includes a study figure')
        study.pop('figure')
        added = [r for r in study['evidence_files'] if r['role'] == FIGURE_ROLE]
        require(len(added) == 1, 'Exactly one figure role per study is required')
        study['evidence_files'] = [r for r in study['evidence_files'] if r['role'] != FIGURE_ROLE]
    require(restored == original, 'Numeric, statistical, source or other original payload fields changed')
    return True


def finalize(input_path, figures, output_path):
    input_path = Path(input_path).absolute(); output_path = Path(output_path).absolute()
    require(input_path.is_file() and not input_path.is_symlink(), 'Input evidence must be a regular file')
    require(output_path.suffix == '.json', 'Presentation output must be a JSON file; no PDF is authored')
    require(not output_path.exists() and not output_path.is_symlink(), 'Fresh presentation output only; refusing overwrite')
    require(output_path.parent.resolve() == input_path.parent.resolve(),
            'Presentation output must share the evidence directory to preserve relative identities')
    initial_bytes = input_path.read_bytes()
    builder_path = Path(__file__).absolute().parent/'build_followup_report.py'
    builder = module('presentation_complete_report_gate', builder_path)
    evidence = builder.load_evidence(input_path)  # Full gate precedes recommendation/aggregate/figure reads.
    require(input_path.read_bytes() == initial_bytes, 'Input evidence changed while its complete gate ran')
    require('presentation_provenance' not in evidence, 'Input is already presentation-finalized')
    require(isinstance(evidence.get('recommendation_notes'), list) and
            all(isinstance(x, str) for x in evidence['recommendation_notes']), 'Original recommendation notes required')
    require(isinstance(evidence.get('figure_status'), str), 'Original figure status required')
    for study in evidence['studies'].values():
        require('figure' not in study and all(r['role'] != FIGURE_ROLE for r in study['evidence_files']),
                'Input already contains presentation figures')
    aggregates = {}
    for name in STUDIES:
        record = next(r for r in evidence['studies'][name]['evidence_files'] if r['role'] == 'aggregate')
        path = input_path.parent/record['path']
        require(sha(path) == record['sha256'], 'Aggregate changed after the complete gate')
        aggregates[name] = json.loads(path.read_text())
    identities = figure_identities(input_path.parent, figures)
    notes = derive_recommendations(evidence, aggregates)
    presented = copy.deepcopy(evidence)
    presented['recommendation_notes'].extend(notes)
    presented['figure_status'] = PRESENTATION_FIGURE_STATUS
    for identity in identities:
        if identity['embedded']:
            study = presented['studies'][identity['role']]
            study['figure'] = identity['path']
            study['evidence_files'].append(dict(role=FIGURE_ROLE, path=identity['path'], sha256=identity['sha256']))
    plotter = Path(__file__).absolute().parent/'next_round/plot_results.py'
    require(plotter.is_file(), 'Root-owned plotter source required for presentation provenance')
    presented['presentation_provenance'] = dict(status='complete_four_study_presentation_finalized',
        input_evidence_filename=input_path.name, input_evidence_sha256=hashlib.sha256(initial_bytes).hexdigest(),
        finalizer_source_sha256=sha(Path(__file__)), builder_source_sha256=sha(builder_path),
        plotter_source_sha256=sha(plotter), figures=identities, recommendation_paragraphs_added=4,
        recommendations_use_existing_summary_and_primary_contrasts=True,
        interpretation_rule='Direction of paired mean with per-study exact sign-flip Holm p <= 0.05; otherwise a non-detection, never equivalence.',
        numeric_statistical_and_source_fields_preserved=True, global_figure_embedded=False,
        prior_figure_status=evidence['figure_status'], original_figure_status_preserved_in_provenance=True,
        figure_status_update_limited_to_presentation=True, figures_rendered_by_finalizer=False, pdf_authored=False)
    preserved_projection(presented, evidence)
    body = json.dumps(presented, indent=2, allow_nan=False) + '\n'
    # Recheck the candidate using the current builder from a same-directory temporary
    # file before creating the sole requested fresh final output. No PDF is authored.
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', prefix='.presentation-check-',
                                     dir=output_path.parent, delete=True) as temporary:
        temporary.write(body); temporary.flush()
        builder.load_evidence(Path(temporary.name))
    require(input_path.read_bytes() == initial_bytes, 'Input evidence changed before presentation publication')
    for identity in identities:
        require(sha(input_path.parent/identity['path']) == identity['sha256'], 'Figure changed during presentation validation')
    with output_path.open('x') as output:
        output.write(body)
    return dict(status='complete_presentation_fields_added_numeric_payload_preserved',
                output_sha256=sha(output_path), input_evidence_sha256=hashlib.sha256(initial_bytes).hexdigest(),
                study_figures_embedded=4, standalone_global_figures=1, recommendation_paragraphs_added=4,
                original_input_preserved=True, numeric_statistical_and_source_fields_preserved=True,
                pdf_authored=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--figures', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(finalize(args.evidence, args.figures, args.out), indent=2))


if __name__ == '__main__':
    main()
