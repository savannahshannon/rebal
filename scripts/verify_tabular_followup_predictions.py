#!/usr/bin/env python3
"""Independent final-logit replay without importing the scientific runner.

Recompute classification metrics, calibration decisions and fit-count priors
from numeric evidence. No model, optimizer or encoder state is loaded. Actual
training-state provenance and statistical intervals are audited separately.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import time
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

FAMILIES = ('mlp_ce', 'mlp_cb', 'ft_transformer', 'xgboost')
PARTS = ('fit', 'tune', 'calibration', 'confirmation')
TAUS = (0., .25, .5, .75, 1., 1.25, 1.5, 2.)
SCALARS = ('accuracy', 'balanced_accuracy', 'balanced_macro_f1', 'macro_f1',
           'macro_recall', 'head_f1', 'mid_f1', 'tail_f1')
ARRAYS = ('confusion_matrix', 'per_class_f1', 'per_class_recall',
          'per_class_precision', 'per_class_support')
FIELDS = {part + '_' + suffix for part in ('calibration', 'confirmation')
          for suffix in ('ids', 'labels', 'logits')}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


class Comparator:
    def __init__(self):
        self.maximum = 0.

    def check(self, actual, expected, message):
        a, b = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
        require(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all(), message + ' shape/finite guard')
        difference = float(np.max(np.abs(a - b))) if a.size else 0.
        self.maximum = max(self.maximum, difference)
        require(difference <= 1e-12, message)


def objective_prior(counts, family):
    """Derive natural or effective-number-weighted priors from fit counts."""
    counts = np.asarray(counts, np.int64)
    require(counts.shape == (7,) and np.all(counts > 0), 'All seven fit classes are required')
    if family != 'mlp_cb':
        return counts / counts.sum(), np.ones(7, np.float32)
    maximum = int(counts.max())
    require(maximum > 1, 'Effective-number rule requires maximum class count above one')
    logarithm = math.log1p(-1. / maximum)
    raw = [(1. / maximum) / (-math.expm1(int(n) * logarithm)) for n in counts]
    normalizer = math.fsum(int(n) * w for n, w in zip(counts, raw)) / int(counts.sum())
    weights = np.asarray([w / normalizer for w in raw], np.float32)
    masses = counts * weights
    return masses / math.fsum(float(v) for v in masses), weights


def metrics(labels, prediction, rank):
    labels, prediction, rank = np.asarray(labels), np.asarray(prediction), np.asarray(rank)
    require(labels.ndim == prediction.ndim == 1 and labels.shape == prediction.shape and labels.dtype.kind in 'iu' and prediction.dtype.kind in 'iu', 'Invalid label/prediction vector schema')
    require(np.all((labels >= 0) & (labels < 7)) and np.all((prediction >= 0) & (prediction < 7)) and np.array_equal(np.sort(rank), np.arange(7)), 'Class/rank range differs')
    support = np.bincount(labels, minlength=7)
    require(np.all(support > 0), 'Every evaluated partition must contain all classes')
    classes = np.arange(7)
    matrix = np.bincount(labels.astype(np.int64) * 7 + prediction.astype(np.int64),
                         minlength=49).reshape(7, 7)
    correct = np.diag(matrix).astype(np.float64)
    predicted = matrix.sum(0)
    f1 = 2 * correct / (support + predicted)
    recall = correct / support
    precision = np.divide(correct, predicted, out=np.zeros(7), where=predicted > 0)
    # Row-normalizing the confusion counts represents a uniform class prior.
    uniform = matrix.astype(np.float64) / support[:, None]
    balanced_f1 = float((2 * np.diag(uniform) / (uniform.sum(0) + uniform.sum(1))).mean())
    accuracy = float(correct.sum() / len(labels))
    balanced_accuracy = float(np.diag(uniform).sum() / uniform.sum())
    checks = ((matrix, confusion_matrix(labels, prediction, labels=classes)),
              (f1, f1_score(labels, prediction, labels=classes, average=None, zero_division=0)),
              (recall, recall_score(labels, prediction, labels=classes, average=None, zero_division=0)),
              (precision, precision_score(labels, prediction, labels=classes, average=None, zero_division=0)),
              (accuracy, accuracy_score(labels, prediction)),
              (balanced_accuracy, recall.mean()),
              (balanced_f1, f1_score(labels, prediction, labels=classes, average='macro',
                                     sample_weight=1. / support[labels], zero_division=0)))
    for independent, reference in checks:
        require(np.max(np.abs(np.asarray(independent) - np.asarray(reference))) <= 1e-12,
                'Independent NumPy metric formula differs from sklearn crosscheck')
    return dict(accuracy=accuracy,
                balanced_accuracy=balanced_accuracy, balanced_macro_f1=balanced_f1,
                macro_f1=float(f1.mean()), macro_recall=float(recall.mean()),
                head_f1=float(f1[rank[:3]].mean()), mid_f1=float(f1[rank[3:5]].mean()),
                tail_f1=float(f1[rank[5:]].mean()),
                confusion_matrix=matrix.tolist(),
                per_class_f1=f1.tolist(), per_class_recall=recall.tolist(),
                per_class_precision=precision.tolist(), per_class_support=support.tolist())


def calibration_macro(labels, prediction):
    matrix = np.bincount(labels.astype(np.int64) * 7 + prediction.astype(np.int64),
                         minlength=49).reshape(7, 7)
    denominator = matrix.sum(0) + matrix.sum(1)
    value = float(np.divide(2 * np.diag(matrix), denominator, out=np.zeros(7), where=denominator > 0).mean())
    reference = float(f1_score(labels, prediction, labels=np.arange(7), average='macro', zero_division=0))
    require(abs(value - reference) <= 1e-12, 'Independent calibration macro formula differs from sklearn crosscheck')
    return value


def calibration_choice(logits, labels, prior, taus=TAUS):
    require(tuple(taus) == TAUS, 'Frozen eight-value tau grid differs')
    prior = np.asarray(prior, np.float64)
    require(prior.shape == (7,) and np.isfinite(prior).all() and np.all(prior > 0) and abs(prior.sum() - 1) <= 1e-12, 'Invalid fit-derived decision prior')
    logarithm = np.log(prior)
    logarithm -= logarithm.mean()
    curve = [calibration_macro(labels, (logits - tau * logarithm).argmax(1)) for tau in taus]
    # The ordered grid and first maximum implement the declared smaller-tau tie.
    chosen = float(taus[int(np.argmax(curve))])
    return chosen, curve, logarithm


def load_predictions(path, labels, split):
    with np.load(path, allow_pickle=False) as archive:
        require(set(archive.files) == FIELDS, 'Unexpected prediction archive fields')
        result = {name: archive[name] for name in archive.files}
    for part in ('calibration', 'confirmation'):
        ids, y, logits = (result[part + '_' + suffix] for suffix in ('ids', 'labels', 'logits'))
        require(ids.dtype.kind in 'iu' and y.dtype.kind in 'iu' and ids.ndim == y.ndim == 1,
                'Prediction source-ID/label schema differs')
        require(np.array_equal(ids, split[part]) and np.array_equal(y, labels[split[part]]),
                'Prediction source row or label alignment differs')
        require(logits.dtype == np.float32 and logits.shape == (len(ids), 7) and np.isfinite(logits).all(),
                'Prediction logit shape/dtype/finite guard differs')
    return result


def verify(args):
    run = Path(args.run)
    # Complete-grid gate comes before reading any calibration/confirmation array.
    expected = {run / 'final' / f'seed_{seed}' / family / 'metrics.json'
                for seed in range(10, 20) for family in FAMILIES}
    require(set(run.glob('final/seed_*/*/metrics.json')) == expected,
            'Exactly forty final fits are required before independent prediction replay')
    folders = {path.parent for path in expected}
    for name in ('predictions.npz', 'trained_pending_confirmation.json'):
        require(set(run.glob('final/seed_*/*/' + name)) == {folder / name for folder in folders},
                'Exactly forty final ' + name + ' artifacts are required before array access')
    aggregate = read(run / 'aggregate.json')
    require(aggregate['completed_fits'] == 40 and aggregate['readouts'] == 80,
            'Completed aggregate forty-fit/eighty-readout metadata required before array access')
    p, freeze, meta = read(run / 'protocol.json'), read(run / 'frozen_selection.json'), read(run / 'data/metadata.json')
    for record in (p, freeze, meta):
        require(fingerprint({k: v for k, v in record.items() if k != 'fingerprint'}) == record['fingerprint'],
                'Canonical protocol/selection/data fingerprint differs')
    require(sha(args.source) == p['source_sha256'] == freeze['source_sha256'] and p['data_fingerprint'] == meta['fingerprint'] and freeze['protocol'] == p['fingerprint'], 'Scientific source/protocol/data binding differs')
    require(p['seeds'] == list(range(10, 20)) and set(p['grids']) == set(FAMILIES) and tuple(p['posthoc_taus']) == TAUS, 'Frozen seed/family/tau grid differs')
    require(set(freeze['selected']) == set(FAMILIES) and all(freeze['selected'][family]['config'] in p['grids'][family] for family in FAMILIES), 'Selected family/configuration must belong to exact protocol grid')
    for name in ('labels.npy', 'split.npz', 'scaler.npz'):
        require(sha(run / 'data' / name) == meta['files'][name], 'Source numeric artifact digest differs')
    labels = np.load(run / 'data/labels.npy', allow_pickle=False)
    require(labels.ndim == 1 and labels.dtype.kind in 'iu' and np.all((labels >= 0) & (labels < 7)), 'Source label schema differs')
    with np.load(run / 'data/split.npz', allow_pickle=False) as archive:
        require(set(archive.files) == set(PARTS), 'Exact four-part split schema required')
        split = {name: archive[name] for name in PARTS}
    for name, value in split.items():
        require(value.ndim == 1 and value.dtype.kind in 'iu' and len(np.unique(value)) == len(value) and len(value) > 0 and np.all((value >= 0) & (value < len(labels))), 'Invalid or repeated split source IDs')
        require(np.array_equal(np.unique(labels[value]), np.arange(7)), 'All seven classes must occur in every partition')
    joined = np.concatenate(list(split.values()))
    require(len(joined) == len(labels) and len(np.unique(joined)) == len(labels), 'Source split coverage or separation differs')
    counts = np.bincount(labels[split['fit']], minlength=7)
    rank = np.argsort(-counts, kind='stable')
    require(counts.tolist() == meta['split_class_counts']['fit'] and rank.tolist() == meta['class_rank'], 'Fit class counts or frequency rank differs')
    if not args.fixture:
        require(len(labels) == 581012 and len(split['fit']) == 406704 and len(split['tune']) == len(split['calibration']) == 58099 and len(split['confirmation']) == 58110 and p['neural_epochs'] == 50 and p['mlp_batch_size'] == 2048 and p['ft_batch_size'] == 1024, 'Production data/optimizer scope differs')
    comparator, records = Comparator(), []
    for seed in range(10, 20):
        for family in FAMILIES:
            target = run / 'final' / f'seed_{seed}' / family
            record = read(target / 'metrics.json')
            require((record['family'], record['seed'], record['protocol'], record['freeze']) == (family, seed, p['fingerprint'], freeze['fingerprint']), 'Final prediction record identity differs')
            selected = freeze['selected'][family]['config']
            training = record['training']
            pending = read(target / 'trained_pending_confirmation.json')
            require(sha(target / 'trained_pending_confirmation.json') == record['early_fitting_receipt_sha256'] and
                    (pending['family'], pending['seed'], pending['protocol'], pending['config'], pending['training']) ==
                    (family, seed, p['fingerprint'], selected, training), 'Selected fit receipt source/configuration/training binding differs')
            if 'config' in record:
                require(record['config'] == selected, 'Final top-level configuration differs from selection')
            if family == 'xgboost':
                require(training['seed'] == seed and training['completed_rounds'] == selected['n_estimators'] and training['tree_count'] == 7 * selected['n_estimators'], 'Tree metadata budget/seed differs')
                params = {k: value for k, value in selected.items() if k != 'n_estimators'}
                params.update(objective='multi:softprob', num_class=7, tree_method='hist', max_bin=256,
                              nthread=p['threads'], seed=seed, eval_metric='mlogloss', device='cpu')
                require(training['params'] == params, 'Recorded XGBoost configuration/fixed params differ')
            else:
                batch = p['ft_batch_size'] if family == 'ft_transformer' else p['mlp_batch_size']
                identity = training['identity']
                device = p['ft_device'] if family == 'ft_transformer' else p['mlp_device']
                require(identity['config'] == selected and identity['family'] == family and identity['seed'] == seed and
                        identity['device'] == device and identity['epochs'] == p['neural_epochs'] and identity['batch'] == batch and
                        identity['source_sha256'] == p['source_sha256'] and identity['fit_hash'] == hashlib.sha256(split['fit'].tobytes()).hexdigest() and
                        training['completed_epochs'] == p['neural_epochs'] and training['optimizer_updates'] == p['neural_epochs'] * math.ceil(len(split['fit']) / batch), 'Neural metadata source/configuration/budget differs')
            require(sha(target / 'predictions.npz') == record['predictions_sha256'], 'Final raw prediction digest differs')
            arrays = load_predictions(target / 'predictions.npz', labels, split)
            prior, weights = objective_prior(counts, family)
            comparator.check(prior, training['la_prior'], 'Fit-derived objective prior differs')
            if family != 'xgboost':
                comparator.check(weights, training['class_weights'], 'Fit-derived objective weights differ')
            choice, curve, logarithm = calibration_choice(arrays['calibration_logits'], arrays['calibration_labels'], prior, p['posthoc_taus'])
            adjustment = record['calibration']
            require(adjustment['tau'] == choice and adjustment['selection_part'] == 'calibration' and adjustment['tie_rule'] == 'smallesttau' and [v['tau'] for v in adjustment['curve']] == list(TAUS), 'Calibration-only tau/grid/tie rule differs')
            comparator.check(prior, adjustment['prior'], 'Calibration adjustment prior differs')
            comparator.check(curve, [v['macro_f1'] for v in adjustment['curve']], 'Independent calibration curve differs')
            require(set(record['metrics']) == {'raw', 'la'}, 'Exactly raw and adjusted readouts required')
            for view, logits in (('raw', arrays['confirmation_logits']), ('la', arrays['confirmation_logits'] - choice * logarithm)):
                recomputed = metrics(arrays['confirmation_labels'], logits.argmax(1), rank)
                stored = record['metrics'][view]
                require(set(stored) == set(SCALARS) | set(ARRAYS), 'Final metric schema differs')
                for endpoint, value in recomputed.items():
                    comparator.check(value, stored[endpoint], 'Independent confirmation metric differs: ' + family + '/' + view + '/' + endpoint)
            records.append(dict(seed=seed, family=family, metrics_sha256=sha(target / 'metrics.json'), predictions_sha256=record['predictions_sha256'], selected_tau=choice))
    return dict(status='independent_final_prediction_fixture_passed' if args.fixture else 'independent_final_prediction_replay_passed',
                protocol=p['fingerprint'], freeze=freeze['fingerprint'], scientific_source_sha256=sha(args.source),
                helper_source_sha256=sha(Path(__file__)), aggregate_sha256=sha(run / 'aggregate.json'),
                labels_sha256=sha(run / 'data/labels.npy'), split_sha256=sha(run / 'data/split.npz'), scaler_sha256=sha(run / 'data/scaler.npz'),
                completed_fits=40, readouts_recomputed=80, calibration_curves_recomputed=40,
                tau_candidates_per_curve=8, calibration_candidate_scores_recomputed=320,
                scalar_metrics_recomputed=640, per_class_metric_values_recomputed=1680,
                confusion_entries_recomputed=3920, class_supports_verified=560,
                prediction_files_verified=40, model_weights_loaded=0, scientific_runner_imported=False,
                numpy_confusion_and_metric_formulas_crosschecked_against_sklearn=True,
                prediction_model_inference=False, max_numeric_discrepancy=comparator.maximum, records=records,
                limitations='Independent replay of saved numeric predictions, class-count priors, calibration choices and metrics. No training-state reconstruction, model inference, scaler-estimation recomputation, spatial uncertainty or independent attestation of execution chronology. Actual-state receipts and paired statistical replay are separate.',
                verified_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'source', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--fixture', action='store_true')
    args = parser.parse_args()
    result = verify(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'records'}))


if __name__ == '__main__':
    main()
