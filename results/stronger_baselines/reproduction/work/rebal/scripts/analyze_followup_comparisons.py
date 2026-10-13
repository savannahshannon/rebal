#!/usr/bin/env python3
"""Replay ten-seed paired statistics and the 22-comparison Holm sensitivity.

This consumes complete numeric aggregates and never trains or selects models.
The per-study Holm families remain the primary analyses. The pooled adjustment
is an additional prespecified sensitivity analysis across the four studies.
"""
from pathlib import Path
import argparse
import hashlib
import itertools
import json
import math
import numpy as np
from scipy import stats

FAMILY_SIZES = {'equalization': 6, 'vision': 5, 'text': 8, 'tabular': 3}
SEEDS = list(range(10, 20))
DECLARED_NAMES = {
    'equalization': {'fixed_natural_EQ_effect','fixed_balanced_EQ_effect','sampling_effect_EQoff',
                     'fixed_factorial_interaction','selected_balanced_softmax_vs_selected_ce_la',
                     'selected_bcl_vs_selected_balanced_softmax'},
    'vision': {'partial_vs_frozen_ce','dino_vs_deit_ce','balanced_softmax_vs_ce_deit_frozen',
               'balanced_softmax_vs_ce_deit_last2','balanced_softmax_vs_ce_dino'},
    'text': {a+'__minus__'+b for a,b in [('bert_ce_la','mean_ce_la'),('bert_bsoftmax','bert_ce_la'),
             ('bert_balanced_eq','bert_balanced_ce'),('bert_natural_eq_la','bert_ce_la'),
             ('last1_ce_la','last0_ce_la'),('last2_ce_la','last0_ce_la'),
             ('last1_bsoftmax','last0_bsoftmax'),('last2_bsoftmax','last0_bsoftmax')]},
    'tabular': {a+'__minus__mlp_ce_la' for a in ['ft_transformer_la','xgboost_la','mlp_cb_la']},
}
# The frozen vision runner names its DINO loss contrast with `_dino_frozen`.
# Keep the previously declared report identifier; only this exact source alias
# is accepted. Its paired observations and statistical family remain unchanged.
IDENTIFIER_ALIASES = {
    'vision': {'balanced_softmax_vs_ce_dino_frozen': 'balanced_softmax_vs_ce_dino'},
}

def require(condition, message):
    if not condition:
        raise ValueError(message)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def holm(probabilities):
    probabilities = np.asarray(probabilities, dtype=float)
    require(probabilities.ndim == 1 and np.isfinite(probabilities).all() and
            np.all((probabilities >= 0) & (probabilities <= 1)), 'Invalid p-values')
    indices = np.argsort(probabilities, kind='stable')
    adjusted = np.empty(len(indices), dtype=float)
    previous = 0.
    for rank, index in enumerate(indices):
        previous = max(previous, min(1., (len(indices) - rank) * probabilities[index]))
        adjusted[index] = previous
    return adjusted.tolist()

def paired(differences, tolerance=1e-12):
    d = np.asarray(differences, dtype=float)
    require(d.shape == (10,) and np.isfinite(d).all(), 'Exactly ten finite paired differences required')
    require(np.all(np.abs(d) <= 2), 'Macro-F1 difference or factorial interaction outside its range')
    mean = float(d.mean())
    sd = 0. if np.all(d == d[0]) else float(d.std(ddof=1))
    half = float(stats.t.ppf(.975, 9) * sd / math.sqrt(10))
    signed = np.array(list(itertools.product((-1., 1.), repeat=10))) * d
    exact = float(np.mean(np.abs(signed.mean(1)) >= abs(mean) - tolerance))
    t_p = float(stats.ttest_1samp(d, 0).pvalue) if sd else (0. if mean else 1.)
    return dict(n=10, differences=d.tolist(), mean=mean, sd=sd,
                ci95=[mean-half, mean+half], signflip_p=exact, t_p=t_p)

def normalize(study, aggregate):
    if study == 'equalization':
        require(aggregate['status'] == 'complete_independently_verified_amended_analysis' and
                aggregate['primary_family_size'] == 6 and aggregate['confirmation_cells'] == 10 and
                aggregate['readouts'] == 660 and aggregate['seeds'] == SEEDS and
                aggregate['completed_tuning_cells'] == 320 and all(r['n']==10 for r in aggregate['summary']),
                'Complete amended EQ aggregate required')
        contrasts = aggregate['paired_comparisons']; tolerance = 1e-12
    elif study == 'vision':
        contrasts = aggregate['primary_contrasts']; tolerance = 1e-14
        require(aggregate['status']=='complete_and_prediction_verified' and aggregate['models']==60 and
                aggregate['readouts']==180 and len(aggregate['summary']) == 108 and
                all(r['n']==10 for r in aggregate['summary']), 'Complete vision summaries required')
    elif study == 'text':
        require(aggregate['status'] == 'verified_complete' and aggregate['completed_cells'] == 130 and
                aggregate['expected_cells'] == 130, 'Complete text aggregate required')
        verified=[(r['method'],r['seed']) for r in aggregate['verified_cells']]
        methods={'bert_ce','bert_bsoftmax','bert_natural_eq','bert_balanced_ce','bert_balanced_eq',
                 'mean_ce','mean_bsoftmax','last0_ce','last0_bsoftmax','last1_ce','last1_bsoftmax','last2_ce','last2_bsoftmax'}
        require(len(verified)==130 and set(verified)=={(m,s) for m in methods for s in SEEDS},'Text ten-seed model grid differs')
        contrasts = aggregate['paired_comparisons']; tolerance = 1e-12
    elif study == 'tabular':
        require(aggregate['completed_fits'] == 40 and aggregate['readouts'] == 80,
                'Complete tabular aggregate required')
        require(len(aggregate['summary'])==8 and all(r['seeds']==SEEDS for r in aggregate['summary']), 'Tabular ten-seed summary grid differs')
        contrasts = aggregate['contrasts']; tolerance = 1e-15
    else:
        raise ValueError('Unknown study')
    require(len(contrasts) == FAMILY_SIZES[study], 'Primary comparison family incomplete')
    rows = []
    for c in contrasts:
        differences = c.get('paired_differences', c.get('differences'))
        p_key = next(k for k in ('signflip_p', 'exact_p', 'exact_signflip_p') if k in c)
        computed = paired(differences, tolerance)
        claimed_ci = c.get('ci95', [c.get('ci95_low'), c.get('ci95_high')])
        require(np.asarray(claimed_ci).shape==(2,) and np.isfinite(claimed_ci).all(),'Invalid paired interval')
        for claim, replay in [(c['mean'], computed['mean']), (c['t_p'], computed['t_p']),
                              (c[p_key], computed['signflip_p'])]:
            require(np.isfinite(claim) and abs(claim-replay) < 1e-12, 'Paired statistic mismatch')
        require(np.max(np.abs(np.asarray(claimed_ci)-computed['ci95'])) < 1e-12, 'Paired interval mismatch')
        name = c.get('name', c.get('a', '')+'__minus__'+c.get('b', ''))
        name = IDENTIFIER_ALIASES.get(study, {}).get(name, name)
        rows.append(dict(study=study, name=name, endpoint='confirmation_macro_f1',
                         seeds=SEEDS, **computed))
    require(len({r['name'] for r in rows}) == len(rows), 'Duplicate comparison identity')
    require({r['name'] for r in rows}==DECLARED_NAMES[study],'Comparisons differ from prespecified hypotheses')
    for field in ('signflip_p', 't_p'):
        for row, adjusted in zip(rows, holm([r[field] for r in rows])):
            row[field+'_study_holm'] = adjusted
    return rows

def analyze(aggregates):
    require(set(aggregates) == set(FAMILY_SIZES), 'All four complete studies are required')
    rows = [row for name in FAMILY_SIZES for row in normalize(name, aggregates[name])]
    require(len(rows) == 22, 'Global family must contain exactly 22 primary comparisons')
    for field in ('signflip_p', 't_p'):
        for row, adjusted in zip(rows, holm([r[field] for r in rows])):
            row[field+'_global_holm'] = adjusted
    return dict(status='complete_replayed_global_sensitivity', family_size=22,
                family_sizes=FAMILY_SIZES, seeds=SEEDS, comparisons=rows,
                practical_reference_macro_f1=0.01,
                interpretation='Per-study families remain primary. Global Holm is a cross-study sensitivity. '
                'Intervals describe paired optimization variation conditional on fixed source partitions; '
                'they do not measure external-dataset or foundation-model pretraining uncertainty.')

def main(args):
    paths = {'equalization': args.eq, 'vision': args.vision, 'text': args.text, 'tabular': args.tabular}
    result = analyze({k: json.loads(Path(v).read_text()) for k, v in paths.items()})
    result['input_sha256'] = {k: sha(p) for k, p in paths.items()}
    result['analysis_source_sha256'] = sha(__file__)
    Path(args.out).write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'status': result['status'], 'family_size': result['family_size']}))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('eq', 'vision', 'text', 'tabular', 'out'):
        parser.add_argument('--'+name, required=True)
    main(parser.parse_args())
