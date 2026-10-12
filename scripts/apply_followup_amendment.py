#!/usr/bin/env python3
"""Replay the prospectively amended six-comparison EQ analysis.

All inputs are explicit paths. This helper does not train, select, calibrate or
infer models. Per-seed metrics may be numeric-only projections retaining seed,
protocol, selection, completed_utc and all scalar metric endpoints.
"""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math

import numpy as np
from scipy import stats

SEEDS = list(range(10,20))
ARMS = ('natural_off','natural_on','natural_reference','balanced_off','balanced_on','balanced_reference',
        'ce','ce_reference','balanced_softmax','balanced_softmax_reference','bcl_dropout')
READOUTS = {arm+'__'+kind+suffix for arm in ARMS for kind in ('native','natural','ros') for suffix in ('','_la')}
PRIMARY = {
    'fixed_natural_EQ_effect': {'natural_reference__native':1, 'natural_off__native':-1},
    'fixed_balanced_EQ_effect': {'balanced_reference__native':1, 'balanced_off__native':-1},
    'sampling_effect_EQoff': {'balanced_off__native':1, 'natural_off__native':-1},
    'fixed_factorial_interaction': {'balanced_reference__native':1, 'natural_off__native':1,
                                  'balanced_off__native':-1, 'natural_reference__native':-1},
    'selected_balanced_softmax_vs_selected_ce_la': {'balanced_softmax__native_la':1, 'ce__native_la':-1},
    'selected_bcl_vs_selected_balanced_softmax': {'bcl_dropout__native_la':1, 'balanced_softmax__native_la':-1},
}
SECONDARY = {
    'selected_natural_EQ_pipeline_effect': {'natural_on__native':1, 'natural_off__native':-1},
    'selected_balanced_EQ_pipeline_effect': {'balanced_on__native':1, 'balanced_off__native':-1},
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def timestamp(value):
    try:
        parsed = datetime.fromisoformat(value.replace('Z','+00:00'))
    except (AttributeError, ValueError) as error:
        raise ValueError('Invalid provenance timestamp') from error
    require(parsed.tzinfo is not None and parsed.utcoffset() is not None, 'Provenance timestamp requires timezone')
    return parsed.astimezone(timezone.utc)


def close(actual, expected, message):
    actual, expected = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    require(actual.shape == expected.shape and np.isfinite(actual).all() and np.isfinite(expected).all() and
            np.allclose(actual,expected,atol=1e-12,rtol=0), message)


def expression(row):
    require(row.get('endpoint','macro_f1') == 'macro_f1', 'Only declared macro-F1 endpoint is allowed')
    if 'a' in row or 'b' in row:
        require('a' in row and 'b' in row and row['a'] != row['b'] and
                'positive' not in row and 'negative' not in row, 'Invalid pair expression')
        return {row['a']:1, row['b']:-1}
    require('positive' in row and 'negative' in row and len(row['positive']) == len(set(row['positive'])) == 2 and
            len(row['negative']) == len(set(row['negative'])) == 2 and
            not set(row['positive']).intersection(row['negative']), 'Invalid factorial expression')
    return {**{name:1 for name in row['positive']}, **{name:-1 for name in row['negative']}}


def paired(values):
    d = np.asarray(values,dtype=np.float64)
    require(d.shape == (10,) and np.isfinite(d).all(), 'Exactly ten finite paired differences required')
    mean = float(d.mean())
    sd = 0. if np.all(d == d[0]) else float(d.std(ddof=1))
    half = float(stats.t.ppf(.975,9)*sd/math.sqrt(10))
    signs = np.asarray(list(itertools.product((-1.,1.),repeat=10)))
    exact = float(np.mean(np.abs((signs*d).mean(axis=1)) >= abs(mean)-1e-12))
    t_p = float(stats.ttest_1samp(d,0).pvalue) if sd else (0. if mean else 1.)
    return dict(n=10,seeds=SEEDS,differences=d.tolist(),mean=mean,sd=sd,
                ci95=[mean-half,mean+half],signflip_p=exact,t_p=t_p)


def holm(values):
    values = np.asarray(values,dtype=float)
    require(values.ndim == 1 and np.isfinite(values).all() and np.all((0 <= values) & (values <= 1)), 'Invalid p-values')
    result = np.empty(len(values))
    running = 0.
    for rank,index in enumerate(np.argsort(values,kind='stable')):
        running = max(running,min(1.,(len(values)-rank)*values[index]))
        result[index] = running
    return result.tolist()


def verify_selection(selection):
    require(selection['completed_tuning_cells'] == 320 and selection['confirmation_access_before_freeze'] is False,
            'Complete tuning-only selection required')
    body = {k:v for k,v in selection.items() if k not in ('fingerprint','selection_created_utc')}
    require(fingerprint(body) == selection['fingerprint'], 'Selection fingerprint differs')
    inventory = selection['tuning_inventory']
    unique = {(row['seed'],fingerprint(row['config'])) for row in inventory}
    require(len(inventory) == len(unique) == 320 and {row['seed'] for row in inventory} == set(SEEDS) and
            all(sum(row['seed']==seed for row in inventory)==32 for seed in SEEDS), 'Exact 32×10 tuning inventory required')
    groups = {'natural','balanced','ce','balanced_softmax','bcl_dropout'}
    require(set(selection['candidates']) == set(selection['selected']) == groups, 'Global selection method grid differs')
    for name, candidates in selection['candidates'].items():
        require(len(candidates) == len({fingerprint(row['config']) for row in candidates}) == 6, 'Six unique candidates per selected method required')
        for candidate in candidates:
            values = [row['macro_f1'] for row in inventory if row['config'] == candidate['config']]
            require(len(values) == 10 and all(math.isfinite(v) and 0 <= v <= 1 for v in values), 'Candidate paired tuning scores missing')
            close(candidate['mean_macro_f1'],np.mean(values),'Candidate mean tuning score differs')
        require(selection['selected'][name] == max(candidates,key=lambda row:row['mean_macro_f1'])['config'],
                'Global selected configuration differs from frozen tuning-only winner')


def apply_amendment(original, cells, selection, amendment, audit, counter, digests):
    require(original['status'] == 'complete' and original['completed_tuning_cells'] == 320 and
            original['confirmation_cells'] == 10 and original['readouts'] == 660 and original['seeds'] == SEEDS and
            original['selection_frozen_before_confirmation'] is True, 'Complete 320/10/660 original aggregate required')
    verify_selection(selection)
    require(fingerprint({k:v for k,v in amendment.items() if k != 'fingerprint'}) == amendment['fingerprint'],
            'Analysis amendment fingerprint differs')
    require(digests.get('scientific_source') == amendment['source_sha256'], 'Frozen scientific training source is not verified')
    require(original['protocol'] == selection['protocol'] == amendment['source_protocol'] == audit['protocol'] == counter['protocol'],
            'Scientific protocol identities differ')
    require(original['selection'] == audit['selection'] == selection['fingerprint'], 'Frozen selection identities differ')
    require(audit['status'] == 'independently_verified_complete' and audit['aggregate_sha256'] == digests['original_aggregate'] and
            audit['tuning_cells_recomputed'] == 320 and audit['confirmation_cells'] == 10 and audit['readouts_recomputed'] == 660,
            'Independent original aggregate verification required')
    require(amendment['status'] == 'frozen_before_confirmation' and amendment['primary_family_size'] == 6 and
            amendment['confirmation_cells_at_freeze'] == amendment['confirmation_prediction_archives_at_freeze'] == 0 and
            amendment['controller_phase_at_freeze'] == 'tune' and amendment['selection_receipt_exists_at_freeze'] is False and
            amendment['training_changed'] is False and amendment['run_budget_changed'] is False,
            'Prospective unchanged-training amendment required')
    require(digests['counter_snapshot'] == amendment['controller_counter_snapshot_sha256'], 'Exact preconfirmation counter snapshot hash differs')
    require(counter['phase'] == 'tune' and counter['phases'][-1]['phase'] == 'tune' and
            counter['phases'][-1]['completed_tuning_cells'] == amendment['completed_tuning_cells_at_freeze'] and
            all(phase['completed_confirmation_cells'] == 0 for phase in counter['phases']), 'Counter snapshot does not confirm protected holdout')
    require(timestamp(counter['last_observed_utc']) <= timestamp(amendment['created_utc']) <= timestamp(selection['selection_created_utc']),
            'Amendment does not precede selection')
    require(len(cells) == 10 and [cell['seed'] for cell in cells] == SEEDS, 'Exact paired seed order required')
    for cell in cells:
        require(cell['protocol'] == original['protocol'] and cell['selection'] == selection['fingerprint'] and
                timestamp(cell['completed_utc']) >= timestamp(selection['selection_created_utc']), 'Confirmation identity or freeze order differs')
        require(set(cell['metrics']) == READOUTS, 'Exact 66-readout grid per seed required')
        for values in cell['metrics'].values():
            require(math.isfinite(values['macro_f1']) and 0 <= values['macro_f1'] <= 1, 'Invalid confirmation macro-F1')
    summary = original['summary']
    require(len(summary) == 66 and {row['method'] for row in summary} == READOUTS and all(row['n'] == 10 for row in summary),
            'Complete ten-seed original summary grid required')
    for row in summary:
        require('macro_f1' in row['metrics'], 'Original summary missing primary endpoint')
        for endpoint, claim in row['metrics'].items():
            values = np.array([cell['metrics'][row['method']][endpoint] for cell in cells],dtype=float)
            require(values.shape == (10,) and np.isfinite(values).all(), 'Invalid scalar confirmation metric')
            mean,sd = float(values.mean()),float(values.std(ddof=1))
            half = float(stats.t.ppf(.975,9)*sd/math.sqrt(10))
            for key,value in [('mean',mean),('sd',sd),('ci95',[mean-half,mean+half])]:
                close(claim[key],value,'Original summary metric differs from per-seed records')
    output = {}
    for field,prescribed in [('primary_contrasts',PRIMARY),('secondary_contrasts',SECONDARY)]:
        declared = amendment[field]
        require([row['name'] for row in declared] == list(prescribed) and
                all(expression(row) == prescribed[row['name']] for row in declared), 'Exact prospective contrast expressions differ')
        rows = []
        for contrast in declared:
            if 'a' in contrast:
                values = [cell['metrics'][contrast['a']]['macro_f1']-cell['metrics'][contrast['b']]['macro_f1'] for cell in cells]
            else:
                # Preserve the declared sum order to reproduce floating-point expressions.
                values = [sum(cell['metrics'][name]['macro_f1'] for name in contrast['positive'])-
                          sum(cell['metrics'][name]['macro_f1'] for name in contrast['negative']) for cell in cells]
            rows.append(dict(**contrast,**paired(values)))
        output[field] = rows
    primary = output['primary_contrasts']
    for field in ('signflip_p','t_p'):
        for row,adjusted in zip(primary,holm([row[field] for row in primary])):
            row[field+'_holm'] = adjusted
    return dict(status='complete_independently_verified_amended_analysis',protocol=original['protocol'],selection=original['selection'],
                analysis_amendment=amendment['fingerprint'],amendment_sha256=digests['analysis_amendment'],
                original_aggregate_sha256=digests['original_aggregate'],independent_verification_sha256=digests['independent_verification'],
                seeds=SEEDS,completed_tuning_cells=320,confirmation_cells=10,readouts=660,summary=summary,
                paired_comparisons=primary,primary_family_size=6,secondary_comparisons=output['secondary_contrasts'],
                source_and_training_unchanged=True,
                scientific_source_file_verified=True,scientific_source_sha256=digests['scientific_source'],
                provenance_order_scope='Ordering is verified from supplied preserved counter and selection/confirmation receipt timestamps. '
                                       'File SHA256 values bind those supplied bytes; timestamps are not an external time attestation.',
                primary_role='Four fixed-EQ treatment factorial effects plus two equally candidate-budgeted selected strong-control pipeline comparisons; amendment frozen before any confirmation',
                historical_runner_primary_role='Original adaptive-selected four-comparison family retained in original aggregate, superseded for primary inference',
                inference_limit='Ten optimization seeds on one fixed same-source holdout; intervals exclude partition/pretraining/geographic variation. SelectedBCL versus selectedBalancedSoftmax compares policies, not isolated BCL loss at a fixed learning rate.',
                input_sha256=digests,analysis_source_sha256=sha(__file__),
                verified_utc=datetime.now(timezone.utc).isoformat())


def main(args):
    paths = dict(original_aggregate=args.aggregate,selection=args.selection,analysis_amendment=args.amendment,
                 independent_verification=args.independent_verification,counter_snapshot=args.counter_snapshot)
    loaded = {name:json.loads(Path(path).read_text()) for name,path in paths.items()}
    digests = {name:sha(path) for name,path in paths.items()}
    files = sorted(Path(args.confirmation_dir).glob('seed_*/metrics.json'))
    require(len(files) == 10 and {file.parent.name for file in files} == {f'seed_{seed}' for seed in SEEDS},
            'Confirmation directory must contain exactly ten seed_10..seed_19 metrics JSONs')
    files = [Path(args.confirmation_dir)/f'seed_{seed}'/'metrics.json' for seed in SEEDS]
    cells = [json.loads(file.read_text()) for file in files]
    digests['per_seed_metrics'] = {str(seed):sha(file) for seed,file in zip(SEEDS,files)}
    require(sha(args.source) == loaded['analysis_amendment']['source_sha256'], 'Frozen scientific training source differs')
    digests['scientific_source'] = sha(args.source)
    result = apply_amendment(loaded['original_aggregate'],cells,loaded['selection'],loaded['analysis_amendment'],
                             loaded['independent_verification'],loaded['counter_snapshot'],digests)
    Path(args.out).write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'],primary_family_size=6,secondary_comparisons=2)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--aggregate',type=Path,required=True)
    parser.add_argument('--confirmation-dir',type=Path,required=True)
    parser.add_argument('--selection',type=Path,required=True)
    parser.add_argument('--amendment',type=Path,required=True)
    parser.add_argument('--independent-verification',type=Path,required=True)
    parser.add_argument('--counter-snapshot',type=Path,required=True)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    main(parser.parse_args())
