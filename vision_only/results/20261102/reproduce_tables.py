"""Reproduce released tables without images, checkpoints, or experiment execution.

Requires NumPy and SciPy. Run from any directory with --data pointing to the
directory containing this script. Outputs go only to --output.
"""
import argparse
import csv
import itertools
import json
from pathlib import Path
import numpy as np
from scipy.stats import t


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def paired(delta):
    delta = np.asarray(delta, dtype=float)
    n = len(delta)
    mean = float(delta.mean())
    radius = float(t.ppf(.975, n - 1) * delta.std(ddof=1) / np.sqrt(n))
    signs = np.asarray(list(itertools.product((-1., 1.), repeat=n)))
    p = float((np.abs((signs * delta).mean(axis=1)) >= abs(mean) - 1e-12).mean())
    return dict(effect_pp=mean * 100, ci_low_pp=(mean-radius)*100,
                ci_high_pp=(mean+radius)*100, p_exact=p)


def holm(rows, field):
    running = 0.
    for rank, index in enumerate(sorted(range(len(rows)), key=lambda i: rows[i]['p_exact'])):
        running = min(1., max(running, (len(rows)-rank) * rows[index]['p_exact']))
        rows[index][field] = running


def validate(actual, expected, keys):
    assert len(actual) == len(expected)
    for a, e in zip(actual, expected):
        for key in keys:
            assert np.isclose(a[key], e[key], atol=1e-9, rtol=0), (key, a, e)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    primary = read_csv(args.data/'primary_720_per_seed.csv')
    eq = read_csv(args.data/'eq_followup_48_per_seed.csv')
    generated = read_csv(args.data/'generated_192_per_seed.csv')
    assert (len(primary), len(eq), len(generated)) == (720, 48, 192)
    encoders = ['deit_small', 'dinov2_small']
    conditions = [('imagenet1k','native'), ('imagenet1k','lt100'),
                  ('cifar100','native'), ('cifar100','lt100'), ('inat2018','native')]
    def primary_values(dataset, regime, encoder, arm):
        rows = sorted([r for r in primary if (r['dataset'],r['regime'],r['encoder'],r['arm'])
                       == (dataset,regime,encoder,arm)], key=lambda r:int(r['seed']))
        assert [int(r['seed']) for r in rows] == list(range(10,22))
        return np.asarray([float(r['macro_f1']) for r in rows])
    contrasts, interactions, means = [], [], []
    for dataset, regime in conditions:
        effects = {}
        family = []
        for encoder in encoders:
            for sampler, arm in [('natural','natural_balanced'), ('uniform','uniform')]:
                delta = primary_values(dataset,regime,encoder,arm+'_eq')-primary_values(dataset,regime,encoder,arm+'_off')
                effects[encoder,sampler] = delta
                family.append(dict(dataset=dataset,regime=regime,contrast=sampler+'_EQ_'+encoder,**paired(delta)))
            interactions.append(dict(dataset=dataset,regime=regime,encoder=encoder,
                **paired(effects[encoder,'uniform']-effects[encoder,'natural'])))
            for arm in ['natural_balanced_off','natural_balanced_eq','uniform_off','uniform_eq','natural_ce','balanced_softmax']:
                values = primary_values(dataset,regime,encoder,arm)
                means.append(dict(dataset=dataset,regime=regime,encoder=encoder,arm=arm,macro_f1_percent=float(values.mean()*100)))
        for sampler in ['natural','uniform']:
            family.append(dict(dataset=dataset,regime=regime,contrast=sampler+'_encoder_by_EQ_interaction',
                **paired(effects[encoders[1],sampler]-effects[encoders[0],sampler])))
        holm(family,'p_holm_condition'); contrasts.extend(family)
    holm(contrasts,'p_holm_all'); holm(interactions,'p_holm_exploratory_10')
    validate(contrasts,json.loads((args.data/'primary_30_comparisons.json').read_text()),
             ['effect_pp','ci_low_pp','ci_high_pp','p_exact','p_holm_condition','p_holm_all'])
    validate(interactions,json.loads((args.data/'exploratory_10_sampler_interactions.json').read_text()),
             ['effect_pp','ci_low_pp','ci_high_pp','p_exact','p_holm_exploratory_10'])
    eq_effects=[]
    for split,variant in [(20261016,'standard'),(202611021,'standard'),(202611022,'standard'),
                          (20261016,'lower_coefficient'),(20261016,'earlier_onset')]:
        for sampler,arm in [('natural','natural_balanced'),('uniform','uniform')]:
            def eq_values(suffix, variant):
                rows=sorted([r for r in eq if int(r['split_seed'])==split and r['variant']==variant and r['arm']==arm+suffix],key=lambda r:int(r['seed']))
                assert [int(r['seed']) for r in rows]==[101,102,103]
                return np.asarray([float(r['macro_f1']) for r in rows])
            delta=(eq_values('_eq',variant)-eq_values('_off','standard'))*100
            eq_effects.append(dict(split_seed=split,variant=variant,sampler=sampler,
                mean_effect_pp=float(delta.mean()),individual_seed_effects_pp=delta.tolist()))
    gen_contrasts, gen_means = [], []
    def gen_values(encoder,condition):
        rows=sorted([r for r in generated if r['encoder']==encoder and r['condition']==condition],key=lambda r:int(r['seed']))
        assert [int(r['seed']) for r in rows]==list(range(101,113))
        return np.asarray([float(r['macro_f1']) for r in rows])
    for encoder in encoders:
        for dose in [.1,.5,1.]:
            gen_contrasts.append(dict(encoder=encoder,dose=dose,
                **paired(gen_values(encoder,f'synthetic_{dose:g}')-gen_values(encoder,f'duplicate_{dose:g}'))))
        for condition in ['real_natural','real_uniform','duplicate_0.1','synthetic_0.1','duplicate_0.5','synthetic_0.5','duplicate_1','synthetic_1']:
            gen_means.append(dict(encoder=encoder,condition=condition,macro_f1_percent=float(gen_values(encoder,condition).mean()*100)))
    holm(gen_contrasts,'holm6_p')
    expected=json.loads((args.data/'generated_6_comparisons.json').read_text())['comparisons']
    expected=[dict(c,ci_low_pp=c['paired_ci95_pp'][0],ci_high_pp=c['paired_ci95_pp'][1],p_exact=c['exact_signflip_p']) for c in expected]
    validate(gen_contrasts,expected,['effect_pp','ci_low_pp','ci_high_pp','p_exact','holm6_p'])
    args.output.mkdir(parents=True,exist_ok=True)
    results={'primary_30':contrasts,'exploratory_10':interactions,'eq_48_effects':eq_effects,
             'generated_6':gen_contrasts,'primary_60_means':means,'generated_16_means':gen_means}
    for name,result in results.items():
        (args.output/(name+'.json')).write_text(json.dumps(result,indent=2)+'\n')
    print('PASS: 720 + 48 + 192 rows; all 30 primary, 10 exploratory, and 6 generated-image comparisons reproduced.')


if __name__=='__main__':
    main()
