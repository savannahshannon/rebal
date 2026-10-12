#!/usr/bin/env python3
"""Independent portable numerical audit of completed text follow-up artifacts.

No study helper or scoring implementation is imported. Confusion matrices use
NumPy bincount and are crosschecked with sklearn. The CLI is read-only except
for its requested verification receipt. Fixture mode is explicitly separate
from the required production 13-method, ten-seed grid.
"""
from pathlib import Path
import argparse
import hashlib
import itertools
import json
import math
import time

import numpy as np
from scipy import stats
from sklearn.metrics import f1_score, recall_score

METHODS = ('bert_ce', 'bert_bsoftmax', 'bert_natural_eq', 'bert_balanced_ce', 'bert_balanced_eq',
           'mean_ce', 'mean_bsoftmax', 'last0_ce', 'last0_bsoftmax', 'last1_ce', 'last1_bsoftmax',
           'last2_ce', 'last2_bsoftmax')
PRIMARY = (('bert_ce_la', 'mean_ce_la'), ('bert_bsoftmax', 'bert_ce_la'),
           ('bert_balanced_eq', 'bert_balanced_ce'), ('bert_natural_eq_la', 'bert_ce_la'),
           ('last1_ce_la', 'last0_ce_la'), ('last2_ce_la', 'last0_ce_la'),
           ('last1_bsoftmax', 'last0_bsoftmax'), ('last2_bsoftmax', 'last0_bsoftmax'))
PARTS = ('fit', 'tune', 'calibration', 'confirmation', 'native_test')
TAUS = (0., .25, .5, .75, 1., 1.25, 1.5, 2.)
MAX_NUMERIC_DISCREPANCY = 0.


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def close(actual, expected, message):
    global MAX_NUMERIC_DISCREPANCY
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    require(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all() and
            np.allclose(a, b, atol=1e-12, rtol=0), message)
    MAX_NUMERIC_DISCREPANCY = max(MAX_NUMERIC_DISCREPANCY, float(np.max(np.abs(a-b))))


def metrics(labels, predictions, rank):
    labels, predictions, rank = np.asarray(labels), np.asarray(predictions), np.asarray(rank)
    classes = len(rank)
    require(labels.ndim == 1 and len(labels) > 0 and predictions.shape == labels.shape and
            np.issubdtype(labels.dtype, np.integer) and np.issubdtype(predictions.dtype, np.integer) and
            np.all((labels >= 0) & (labels < classes)) and np.all((predictions >= 0) & (predictions < classes)),
            'Invalid metric label/prediction vectors')
    require(sorted(rank.tolist()) == list(range(classes)), 'Invalid inverse frequency rank')
    confusion = np.bincount(labels.astype(np.int64)*classes + predictions.astype(np.int64),
                            minlength=classes*classes).reshape(classes, classes)
    correct = confusion.diagonal().astype(np.float64)
    actual = confusion.sum(axis=1).astype(np.float64)
    predicted = confusion.sum(axis=0).astype(np.float64)
    f1 = np.divide(2*correct, actual+predicted, out=np.zeros(classes), where=actual+predicted > 0)
    recall = np.divide(correct, actual, out=np.zeros(classes), where=actual > 0)
    close(f1, f1_score(labels, predictions, labels=np.arange(classes), average=None, zero_division=0), 'NumPy/sklearn F1 mismatch')
    close(recall, recall_score(labels, predictions, labels=np.arange(classes), average=None, zero_division=0), 'NumPy/sklearn recall mismatch')
    result = dict(accuracy=float(correct.sum()/len(labels)), macro_f1=float(f1.mean()),
                  macro_recall=float(recall.mean()), worst_f1=float(f1.min()),
                  per_class_f1=f1.tolist(), per_class_recall=recall.tolist())
    for name, mask in [('head', rank < classes//3),
                       ('mid', (rank >= classes//3) & (rank < 2*classes//3)),
                       ('tail', rank >= 2*classes//3)]:
        result[name+'_f1'] = float(f1[mask].mean())
        result[name+'_recall'] = float(recall[mask].mean())
    result['head_tail_f1_gap'] = result['head_f1']-result['tail_f1']
    return result


def compare_metrics(claimed, expected):
    require(set(claimed) == set(expected), 'Metric schema differs')
    for name, value in expected.items():
        close(claimed[name], value, 'Raw metric replay differs: '+name)


def logits(array, rows, classes=14):
    require(array.shape == (rows, classes) and np.issubdtype(array.dtype, np.floating) and
            np.isfinite(array).all(), 'Invalid finite logit matrix')


def requires_offset(method):
    return method.endswith('_ce') and 'balanced' not in method or method == 'bert_natural_eq'


def calibration(cal, labels, counts, rank):
    prior = counts/counts.sum()
    log_prior = np.log(prior).astype(np.float32)
    curve = [{'tau': tau, 'macro_f1': metrics(labels, (cal-tau*log_prior).argmax(1), rank)['macro_f1']}
             for tau in TAUS]
    selected = max(curve, key=lambda row: row['macro_f1'])['tau']
    return dict(tau=selected, curve=curve, prior=prior.tolist(), tie_break='first smallest tau; calibration only')


def compare_calibration(claimed, expected):
    if expected is None:
        require(claimed is None, 'Uniform-prior method unexpectedly calibrated')
        return
    require(claimed is not None and set(claimed) == set(expected) and
            claimed['tau'] == expected['tau'] and claimed['tie_break'] == expected['tie_break'],
            'Calibration-only choice/tie break differs')
    close(claimed['prior'], expected['prior'], 'Calibration prior differs from fit-only counts')
    require(len(claimed['curve']) == len(expected['curve']), 'Calibration curve budget differs')
    for actual, wanted in zip(claimed['curve'], expected['curve']):
        require(actual['tau'] == wanted['tau'], 'Calibration tau grid differs')
        close(actual['macro_f1'], wanted['macro_f1'], 'Calibration metric curve differs')


def paired(differences):
    d = np.asarray(differences, dtype=np.float64)
    require(d.ndim == 1 and len(d) >= 2 and np.isfinite(d).all(), 'Invalid paired differences')
    n = len(d)
    mean = float(d.mean())
    sd = 0. if np.all(d == d[0]) else float(d.std(ddof=1))
    half = stats.t.ppf(.975, n-1)*sd/math.sqrt(n)
    exceed = sum(abs(float(np.dot(signs, d)/n)) >= abs(mean)-1e-12
                 for signs in itertools.product((-1., 1.), repeat=n))
    return dict(n=n, differences=d.tolist(), mean=mean, sd=sd,
                ci95=[float(mean-half), float(mean+half)], signflip_p=exceed/2**n,
                t_p=float(stats.ttest_1samp(d, 0).pvalue) if sd else (0. if mean else 1.))


def holm(probabilities):
    order = sorted(range(len(probabilities)), key=lambda i: probabilities[i])
    adjusted = [0.] * len(probabilities)
    previous = 0.
    for rank, index in enumerate(order):
        previous = max(previous, min(1., (len(order)-rank)*probabilities[index]))
        adjusted[index] = previous
    return adjusted


def state_hash(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def validate_fit(folder, selection, identity, protocol, rank, tune_labels, portable=False):
    folder = Path(folder)
    stored = read(folder/'selection.json')
    require(stored == selection and selection['identity'] == identity, 'Fit selection/identity mismatch')
    budget = read(folder/'budget.json')
    history = read(folder/'history.json')
    epochs = protocol['partial_epochs'] if identity['method'].startswith('last') else protocol['adapter_epochs']
    steps = protocol['partial_steps_per_epoch'] if identity['method'].startswith('last') else protocol['adapter_steps_per_epoch']
    require(budget['identity'] == identity and budget['planned_epochs'] == epochs == budget['completed_epochs'] == selection['epochs'] and
            budget['steps_per_epoch'] == steps and selection['updates'] == epochs*steps and
            budget['fit_examples_per_epoch'] == selection['fit_examples'] == protocol['fit_examples'], 'Fit budget differs from full protocol')
    require(len(history) == epochs and budget['initial_state_sha256'] == selection['initial_state_sha256'], 'Fit history/initial identity differs')
    for i, row in enumerate(history, 1):
        require(row['epoch'] == i and row['optimizer_updates'] == i*steps and row['batches'] == steps and
                math.isfinite(row['loss']) and math.isfinite(row['seconds']) and row['seconds'] >= 0 and
                len(row['EQ_eligible_batches_per_class']) == 14 and all(0 <= v <= steps for v in row['EQ_eligible_batches_per_class']),
                'Fit history incomplete or invalid')
    require(set(budget['files']) == {'checkpoint.pt', 'history.json'} and
            sha(folder/'history.json') == budget['files']['history.json'] and
            sha(folder/'budget.json') == selection['budget_sha256'] and
            budget['files']['checkpoint.pt'] == selection['checkpoint_sha256'], 'Fit sidecar integrity mismatch')
    tune = np.load(folder/'tune_logits.npy', allow_pickle=False)
    logits(tune, len(tune_labels))
    require(sha(folder/'tune_logits.npy') == selection['tune_logits_sha256'], 'Fit tuning logits changed')
    tune_score = metrics(tune_labels, tune.argmax(1), rank)['macro_f1']
    close(selection['tune_macro_f1'], tune_score, 'Fit tuning score mismatch')
    checkpoint = folder/'checkpoint.pt'
    checked = False
    if checkpoint.exists():
        import torch
        require(sha(checkpoint) == selection['checkpoint_sha256'], 'Actual checkpoint hash differs')
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        require(saved['identity'] == identity and saved['initial_state_sha256'] == selection['initial_state_sha256'] and
                saved['completed_epochs'] == epochs and saved['history'] == history, 'Actual checkpoint identity/history differs')
        state = saved['model']
        require(all(isinstance(t, torch.Tensor) and (not t.is_floating_point() or bool(torch.isfinite(t).all())) for t in state.values()),
                'Actual checkpoint state invalid/nonfinite')
        require(state_hash(state) == selection['final_state_sha256'], 'Actual final-state hash differs')
        optimizer = saved['optimizer']
        indices = [i for group in optimizer['param_groups'] for i in group['params']]
        require(len(indices) == len(set(indices)) and set(optimizer['state']).issubset(indices), 'Actual optimizer mapping invalid')
        method = identity['method']
        parameters = [(name,t) for name,t in state.items() if name not in ('feature_mean', 'feature_std')]
        trainable = [(name,t) for name,t in parameters if not (method.startswith('last1') and name.startswith('layers.0.'))]
        require(len(indices) == len(trainable) and
                sum(t.numel() for _,t in parameters) == selection['total_parameters'] and
                sum(t.numel() for _,t in trainable) == selection['trainable_parameters'], 'Actual parameter/optimizer counts differ')
        require(all(t.dtype == torch.float32 for _,t in parameters), 'Master parameter precision differs')
        if method.startswith(('last1', 'last2')):
            expected_group_counts = [sum(name.startswith('layers.') for name,_ in trainable),
                                     sum(name.startswith('adapter.') for name,_ in trainable)]
        else:
            expected_group_counts = [len(trainable)]
        require([len(group['params']) for group in optimizer['param_groups']] == expected_group_counts,
                'Actual optimizer parameter groups differ')
        values = optimizer['state'].values()
        if identity['method'].startswith('last'):
            require(len(optimizer['state']) == len(indices) and all(float(v['step']) == epochs*steps for v in values),
                    'Actual AdamW update counts differ')
        else:
            require(len(optimizer['state']) == len(indices) and all('momentum_buffer' in v for v in values),
                    'Actual SGD momentum coverage differs')
        require(all(not isinstance(t, torch.Tensor) or not t.is_floating_point() or bool(torch.isfinite(t).all())
                    for value in values for t in value.values()), 'Actual optimizer state nonfinite')
        for index, (_,parameter) in zip(indices, trainable):
            value = optimizer['state'][index]
            names = ('exp_avg', 'exp_avg_sq') if method.startswith('last') else ('momentum_buffer',)
            require(all(value[name].shape == parameter.shape and value[name].dtype == parameter.dtype for name in names),
                    'Actual optimizer moment shape/precision differs')
        checked = True
    else:
        require(portable, 'Actual checkpoint is missing; use explicit portable omission mode only for published numeric archives')
    return tune_score, checked


def audit(args):
    global MAX_NUMERIC_DISCREPANCY
    MAX_NUMERIC_DISCREPANCY = 0.
    out, data = Path(args.out), Path(args.data)
    protocol = read(out/'protocol.json')
    require(fingerprint({k:v for k,v in protocol.items() if k != 'fingerprint'}) == protocol['fingerprint'], 'Protocol fingerprint differs')
    methods, seeds = protocol['methods'], protocol['seeds']
    require(len(methods) == len(set(methods)) and set(methods) <= set(METHODS) and
            len(seeds) == len(set(seeds)) and seeds == sorted(seeds) and 9 not in seeds and len(seeds) >= 2,
            'Invalid study method/seed grid')
    require(tuple(protocol['TAUS']) == TAUS and tuple(map(tuple, protocol['primary_comparisons'])) == PRIMARY,
            'Declared calibration grid or hypotheses differ')
    if not args.fixture:
        require(methods == list(METHODS) and seeds == list(range(10, 20)), 'Production requires exact 13×10 grid')
        require(protocol['adapter_epochs'] == 50 and protocol['partial_epochs'] == 3 and
                protocol['adapter_steps_per_epoch'] == 104 and protocol['partial_steps_per_epoch'] == 106,
                'Production budgets differ from frozen 5200/318 update policy')
    if args.source:
        require(sha(args.source) == protocol['source_sha256'], 'Frozen scientific source differs')
    metadata = read(data/'dataset.json')
    require(metadata['fingerprint'] == protocol['dataset_fingerprint'], 'Source dataset/protocol identity differs')
    for name in ('labels.npy', 'split.npz'):
        require(name in metadata['files'] and sha(data/name) == metadata['files'][name], 'Source label/split integrity differs')
    labels = np.load(data/'labels.npy', mmap_mode='r', allow_pickle=False)
    with np.load(data/'split.npz', allow_pickle=False) as z:
        source = {name:z[name] for name in z.files}
    with np.load(out/'partition.npz', allow_pickle=False) as z:
        partition = {name:z[name] for name in z.files}
    order = source['class_rank']
    require(sorted(order.tolist()) == list(range(14)), 'Source class order invalid')
    rank = np.empty(14, dtype=np.int64)
    rank[order] = np.arange(14)
    require(np.array_equal(partition['rank'], rank), 'Saved frequency rank not actual source inverse')
    for part in PARTS:
        ids = source[part]
        require(ids.ndim == 1 and np.issubdtype(ids.dtype, np.integer) and len(ids) > 0 and
                len(ids) == len(np.unique(ids)) and np.all((ids >= 0) & (ids < len(labels))), 'Invalid source partition IDs')
        require(np.array_equal(ids, partition[part+'_ids']) and
                np.array_equal(labels[ids], partition[part+'_labels']), 'Saved source partition IDs/labels differ')
    all_ids = np.concatenate([source[part] for part in PARTS])
    require(len(np.unique(all_ids)) == len(all_ids) and len(source['fit']) == protocol['fit_examples'], 'Source partitions overlap or fit budget differs')
    counts = np.bincount(labels[source['fit']], minlength=14)
    require(np.all(counts > 0), 'Fit class missing')
    gate = read(out/'whole_study_confirmation_gate.json')
    require(gate['protocol_fingerprint'] == protocol['fingerprint'] and gate['confirmation_inferred_at_freeze'] is False,
            'Whole-study confirmation gate missing or invalid')
    require(set(gate['hyperparameter_choices_sha256']) == set(methods), 'Whole-study global choice grid differs')
    gate_fits = {(r['method'], r['seed']):r for r in gate['trained_fits']}
    expected_grid = set(itertools.product(methods, seeds))
    require(len(gate['trained_fits']) == len(gate_fits) == len(expected_grid) and set(gate_fits) == expected_grid,
            'Whole-study final-fit grid differs')
    actual_cells = {(p.parent.parent.name, int(p.parent.name.removeprefix('seed_'))) for p in out.glob('*/seed_*/metrics.json')}
    require(actual_cells == expected_grid, 'Completed metric cells not exact full final grid')
    choices, checked_weights = {}, 0
    tune_fits = []
    for method in methods:
        path = out/method/'hyperparameters.json'
        require(sha(path) == gate['hyperparameter_choices_sha256'][method], 'Global choice changed after gate')
        choice = read(path)
        require(choice['protocol_fingerprint'] == protocol['fingerprint'] and choice['tuning_seed'] == 9 and
                choice['confirmation_scored_at_freeze'] is False, 'Global LR freeze invalid')
        lrs = protocol['partial_lrs'] if method.startswith('last') else protocol['adapter_lrs']
        require(len(lrs) == 2 and len(set(lrs)) == 2 and sorted(c['identity']['lr'] for c in choice['candidates']) == sorted(lrs),
                'Global two-candidate LR grid differs')
        scores = []
        for candidate in choice['candidates']:
            identity = dict(protocol=protocol['fingerprint'], method=method, seed=9, lr=candidate['identity']['lr'])
            score, checked = validate_fit(out/method/'seed_9'/f"lr_{identity['lr']:g}", candidate, identity,
                                         protocol, rank, partition['tune_labels'], args.portable_without_weights)
            checked_weights += checked
            scores.append((score, identity['lr']))
            tune_fits.append(dict(method=method, seed=9, lr=identity['lr'], macro_f1=score))
        winner = sorted(scores, key=lambda x:(-x[0], x[1]))[0][1]
        require(choice['selected_lr'] == winner, 'Global LR differs from raw tuning-only winner')
        choices[method] = choice
    result_rows, native_rows, verified_cells, head_hashes, receipt_order = [], [], [], {}, []
    for method, seed in itertools.product(methods, seeds):
        cell = out/method/f'seed_{seed}'
        record = read(cell/'metrics.json')
        require(record['protocol_fingerprint'] == protocol['fingerprint'] and record['method'] == method and record['seed'] == seed,
                'Confirmation metric identity differs')
        require(read(cell/'frozen_choice.json') == choices[method], 'Cell frozen choice differs from whole-study global choice')
        lr = choices[method]['selected_lr']
        folder = cell/f'lr_{lr:g}'
        require(sha(folder/'selection.json') == gate_fits[(method, seed)]['selection_sha256'], 'Final fit changed after gate')
        _, checked = validate_fit(folder, record['selection'], dict(protocol=protocol['fingerprint'], method=method, seed=seed, lr=lr),
                                  protocol, rank, partition['tune_labels'], args.portable_without_weights)
        checked_weights += checked
        offset = read(cell/'calibration_choice.json')
        require(offset == dict(protocol_fingerprint=protocol['fingerprint'], method=method, seed=seed,
                               adjustment=record['adjustment'], confirmation_inferred_at_freeze=False),
                'Calibration freeze receipt differs')
        pred_path = cell/'predictions.npz'
        require(sha(pred_path) == record['predictions_sha256'], 'Confirmation predictions integrity differs')
        receipt_order.append((cell/'calibration_choice.json').stat().st_mtime <= pred_path.stat().st_mtime)
        with np.load(pred_path, allow_pickle=False) as z:
            for part in ('calibration', 'confirmation'):
                logits(z[part+'_logits'], len(source[part]))
                require(np.array_equal(z[part+'_ids'], source[part]) and np.array_equal(z[part+'_labels'], labels[source[part]]),
                        'Raw logit source-row binding differs')
            require(np.array_equal(z['rank'], rank), 'Raw logit rank binding differs')
            expected = {method:metrics(z['confirmation_labels'], z['confirmation_logits'].argmax(1), rank)}
            choice = calibration(z['calibration_logits'], z['calibration_labels'], counts, rank) if requires_offset(method) else None
            compare_calibration(record['adjustment'], choice)
            if choice:
                log_prior = np.log(np.asarray(choice['prior'])).astype(np.float32)
                adjusted = z['confirmation_logits']-choice['tau']*log_prior
                expected[method+'_la'] = metrics(z['confirmation_labels'], adjusted.argmax(1), rank)
            require(set(record['metrics']) == set(expected), 'Declared confirmation readout grid differs')
            for name, values in expected.items():
                compare_metrics(record['metrics'][name], values)
                result_rows.append(dict(method=name, seed=seed, **values))
            expected_native = {}
            if not method.startswith('last'):
                logits(z['native_test_logits'], len(source['native_test']))
                require(np.array_equal(z['native_test_ids'], source['native_test']) and
                        np.array_equal(z['native_test_labels'], labels[source['native_test']]), 'Native source binding differs')
                expected_native[method] = metrics(z['native_test_labels'], z['native_test_logits'].argmax(1), rank)
                if choice:
                    adjusted = z['native_test_logits']-choice['tau']*np.log(np.asarray(choice['prior'])).astype(np.float32)
                    expected_native[method+'_la'] = metrics(z['native_test_labels'], adjusted.argmax(1), rank)
            require(set(record['native_exploratory'] or {}) == set(expected_native), 'Native exploratory readout grid differs')
            for name, values in expected_native.items():
                compare_metrics(record['native_exploratory'][name], values)
                native_rows.append(dict(method=name, seed=seed, **values))
        if not method.startswith('mean_'):
            head_hashes.setdefault(seed, set()).add(record['selection']['initial_head_state_sha256'])
        verified_cells.append(dict(method=method, seed=seed, predictions_sha256=record['predictions_sha256'],
                                   initial_head_state_sha256=record['selection']['initial_head_state_sha256']))
    require(all(len(hashes) == 1 for hashes in head_hashes.values()), 'BERT initial adapter/head hashes not paired')
    aggregate = read(out/'aggregate.json')
    require(aggregate['status'] == 'verified_complete' and aggregate['protocol_fingerprint'] == protocol['fingerprint'] and
            aggregate['completed_cells'] == aggregate['expected_cells'] == len(expected_grid) and aggregate['readouts'] == len(result_rows),
            'Complete aggregate metadata differs')
    require(aggregate['verified_cells'] == verified_cells, 'Aggregate verified cell grid differs')
    indexed = {(r['method'],r['seed']):r for r in result_rows}
    require(len(indexed) == len(result_rows) and len(aggregate['rows']) == len(result_rows), 'Readout grid duplicated/incomplete')
    for row in aggregate['rows']:
        key = row['method'], row['seed']
        require(key in indexed, 'Aggregate row outside expected readouts')
        compare_metrics({k:v for k,v in row.items() if k not in ('method', 'seed')},
                        {k:v for k,v in indexed[key].items() if k not in ('method', 'seed')})
    require({(r['method'],r['seed']) for r in aggregate['rows']} == set(indexed), 'Aggregate readout rows duplicated/incomplete')
    summary_names = sorted({r['method'] for r in result_rows})
    require(len(aggregate['summary']) == len(summary_names) and {r['method'] for r in aggregate['summary']} == set(summary_names),
            'Aggregate summary grid differs')
    for row in aggregate['summary']:
        require(row['n'] == len(seeds), 'Summary missing paired seeds')
        for endpoint in ('accuracy', 'macro_f1', 'macro_recall', 'head_f1', 'mid_f1', 'tail_f1'):
            values = np.array([indexed[(row['method'],s)][endpoint] for s in seeds])
            close(row['metrics'][endpoint]['mean'], values.mean(), 'Summary mean differs')
            close(row['metrics'][endpoint]['sd'], values.std(ddof=1), 'Summary SD differs')
    available = [(a,b) for a,b in PRIMARY if all((a,s) in indexed and (b,s) in indexed for s in seeds)]
    claimed = aggregate['paired_comparisons']
    require(len(claimed) == len(available) and {(c['a'],c['b']) for c in claimed} == set(available),
            'Primary raw contrast expressions differ')
    replayed = []
    for a,b in available:
        computed = paired([indexed[(a,s)]['macro_f1']-indexed[(b,s)]['macro_f1'] for s in seeds])
        claim = next(c for c in claimed if (c['a'],c['b']) == (a,b))
        require(claim['endpoint'] == 'macro_f1', 'Primary endpoint differs')
        for name,value in computed.items():
            close(claim[name], value, 'Primary paired statistic differs: '+name)
        replayed.append(dict(a=a, b=b, **computed))
    if len(available) == 8:
        for p_key in ('signflip_p', 't_p'):
            adjusted = holm([r[p_key] for r in replayed])
            for replay, value in zip(replayed, adjusted):
                claim = next(c for c in claimed if (c['a'],c['b']) == (replay['a'],replay['b']))
                close(claim[p_key+'_holm'], value, 'Eight-member Holm family differs')
                replay[p_key+'_holm'] = value
    if not args.fixture:
        require(len(tune_fits) == 26 and len(result_rows) == 190 and len(native_rows) == 100 and len(replayed) == 8,
                'Production tuning/readout/contrast counts differ')
    receipt = dict(status='independently_verified_fixture' if args.fixture else 'independently_verified_complete',
                   protocol_fingerprint=protocol['fingerprint'], protocol=protocol['fingerprint'], source_sha256=protocol['source_sha256'],
                   auditor_sha256=sha(__file__), aggregate_sha256=sha(out/'aggregate.json'),
                   partition_sha256=sha(out/'partition.npz'), source_labels_sha256=sha(data/'labels.npy'),
                   source_split_sha256=sha(data/'split.npz'), whole_study_gate_sha256=sha(out/'whole_study_confirmation_gate.json'),
                   model_fits=len(expected_grid), models=len(expected_grid), tuning_fits=len(tune_fits), tuning_scores=tune_fits,
                   confirmation_readouts=len(result_rows), native_exploratory_readouts=len(native_rows),
                   readouts=len(result_rows), native_readouts=len(native_rows), primary_family_size=len(replayed),
                   max_numeric_discrepancy=MAX_NUMERIC_DISCREPANCY,
                   actual_checkpoint_states_verified=checked_weights,
                   checkpoint_states_omitted=len(expected_grid)+len(tune_fits)-checked_weights,
                   omission_mode='explicit portable archive mode' if args.portable_without_weights else 'none',
                   calibration_receipts_before_prediction_archive_mtime=sum(receipt_order),
                   calibration_order_limit='Receipt binds a calibration-only choice with confirmation_inferred_at_freeze=false. '
                                           'Local mtimes are corroboration, not a portable proof of inference event order; source order was separately reviewed.',
                   paired_initial_bert_adapter_hashes=True, primary_contrasts=replayed, seeds=seeds,
                   metrics_implementation='Independent NumPy confusion matrix with sklearn per-class crosscheck',
                   scope='Saved logits, source rows/labels/rank, all tuning choices, all per-class/scalar metrics and paired statistics replayed; '
                         'actual checkpoint tensors/hashes and optimizer completion checked where present. '
                         'No encoder re-extraction or checkpoint inference rerun.',
                   verified_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    Path(args.receipt).write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--fixture', action='store_true')
    parser.add_argument('--portable-without-weights', action='store_true')
    result = audit(parser.parse_args())
    print(json.dumps({name:result[name] for name in ('status', 'model_fits', 'tuning_fits', 'confirmation_readouts',
                                                   'native_exploratory_readouts', 'actual_checkpoint_states_verified')}))
