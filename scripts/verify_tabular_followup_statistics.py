#!/usr/bin/env python3
"""Independent statistical replay from already verified per-fit metrics.

Uses Python's statistics module and SciPy's paired test/interval APIs, rather
than importing the benchmark's summary or Holm implementation.
"""
from pathlib import Path
import argparse
import hashlib
import itertools
import json
import math
import statistics
from scipy import stats


METRICS = ['accuracy', 'balanced_accuracy', 'balanced_macro_f1', 'macro_f1',
           'macro_recall', 'head_f1', 'mid_f1', 'tail_f1']


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(actual, expected):
    if not math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12):
        raise ValueError(f'Statistical replay mismatch: {actual} versus {expected}')


def interval(values):
    mean = statistics.mean(values)
    sd = statistics.stdev(values)
    if sd == 0:
        return mean, sd, (mean, mean)
    bounds = stats.t.interval(.95, df=len(values) - 1, loc=mean, scale=sd / math.sqrt(len(values)))
    return mean, sd, tuple(float(value) for value in bounds)


def holm_reference(pvalues):
    ordered = sorted(enumerate(pvalues), key=lambda item: item[1])
    adjusted = [0.] * len(pvalues)
    previous = 0.
    for rank, (index, pvalue) in enumerate(ordered):
        previous = max(previous, min(1., (len(pvalues) - rank) * pvalue))
        adjusted[index] = previous
    return adjusted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    root = args.run
    aggregate = read(root / 'aggregate.json')
    verification = read(root / 'portable_verification.json')
    if verification['numeric_replay']['completed_fits'] != 40 or verification['numeric_replay']['readouts'] != 80 or verification['numeric_replay']['aggregate_sha256'] != sha(root / 'aggregate.json'):
        raise ValueError('Complete bound prediction verification required')
    records = {}
    for path in root.glob('final/seed_*/*/metrics.json'):
        record = read(path)
        for readout in ['raw', 'la']:
            records[(record['family'] + '_' + readout, record['seed'])] = record['metrics'][readout]
    summaries = []
    for row in aggregate['summary']:
        seeds = row['seeds']
        if seeds != list(range(10, 20)):
            raise ValueError('Paired optimization seed order mismatch')
        for metric in METRICS:
            values = [records[(row['method'], seed)][metric] for seed in seeds]
            mean, sd, bounds = interval(values)
            expected = row['metrics'][metric]
            for actual, reference in [(mean, expected['mean']), (sd, expected['sd']),
                                      (bounds[0], expected['ci95'][0]), (bounds[1], expected['ci95'][1])]:
                close(actual, reference)
            summaries.append(dict(method=row['method'], metric=metric, mean=mean, sd=sd, ci95=bounds))
    contrasts = []
    for row in aggregate['contrasts']:
        left = [records[(row['a'], seed)]['macro_f1'] for seed in range(10, 20)]
        right = [records[(row['b'], seed)]['macro_f1'] for seed in range(10, 20)]
        differences = [a - b for a, b in zip(left, right)]
        if differences != row['differences']:
            raise ValueError('Paired seed differences mismatch')
        mean, sd, bounds = interval(differences)
        pvalue = float(stats.ttest_rel(left, right).pvalue) if sd else (0. if mean else 1.)
        extreme = 0
        for mask in range(1 << len(differences)):
            signed = [value if mask & (1 << i) else -value for i, value in enumerate(differences)]
            extreme += abs(math.fsum(signed) / len(differences)) >= abs(mean) - 1e-15
        signflip = extreme / (1 << len(differences))
        for actual, reference in [(mean, row['mean']), (sd, row['sd']), (bounds[0], row['ci95'][0]),
                                  (bounds[1], row['ci95'][1]), (pvalue, row['t_p']),
                                  (signflip, row['exact_signflip_p'])]:
            close(actual, reference)
        contrasts.append(dict(a=row['a'], b=row['b'], differences=differences, mean=mean,
                              sd=sd, ci95=bounds, t_p=pvalue, exact_signflip_p=signflip))
    for field in ['t_p', 'exact_signflip_p']:
        adjusted = holm_reference([row[field] for row in contrasts])
        for reference, row in zip(adjusted, aggregate['contrasts']):
            close(reference, row[field + '_holm'])
    result = dict(status='independent_statistical_summary_replay_passed',
                  aggregate_sha256=sha(root / 'aggregate.json'), summaries_verified=len(summaries),
                  paired_contrasts_verified=len(contrasts), paired_seed_differences_verified=30,
                  signflip_patterns_per_contrast=1024, holm_family_size=3,
                  summaries=summaries, contrasts=contrasts,
                  uncertainty_scope='Optimization variability across ten paired seeds on one fixed partition; no spatial or dataset sampling uncertainty.',
                  source_sha256=sha(Path(__file__)), benchmark_summary_code_imported=False,
                  prediction_model_inference=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key not in ['summaries', 'contrasts']}))


if __name__ == '__main__':
    main()
