"""Small numeric archive tests; no training, model states or production outcomes."""
from pathlib import Path
import argparse
import copy
import importlib.util
import json
import math
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

HERE = Path(__file__).resolve()
CANDIDATES = (HERE.with_name('verify_tabular_followup_predictions.py'),
              HERE.parents[1] / 'scripts/verify_tabular_followup_predictions.py')
HELPER = next(path for path in CANDIDATES if path.is_file())
SPEC = importlib.util.spec_from_file_location('independent_tabular_predictions', HELPER)
v = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(v)


def write(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + '\n')


def fixture(folder):
    """Declare a metadata-only toy study, explicitly using --fixture scope."""
    run = folder / 'run'
    data = run / 'data'
    data.mkdir(parents=True)
    source = folder / 'fixture_scientific_source.py'
    source.write_text('"""Numeric fixture only; no trained models."""\n')
    supports = [35, 28, 21, 14, 14, 14, 14]
    y = np.repeat(np.arange(7), supports).astype(np.int64)
    split = {name: [] for name in v.PARTS}
    start = 0
    for count in supports:
        ids = np.arange(start, start + count)
        split['fit'].extend(ids[:-6])
        split['tune'].extend(ids[-6:-4])
        split['calibration'].extend(ids[-4:-2])
        split['confirmation'].extend(ids[-2:])
        start += count
    split = {k: np.asarray(a, np.int64) for k, a in split.items()}
    np.save(data / 'labels.npy', y)
    np.savez_compressed(data / 'split.npz', **split)
    np.savez_compressed(data / 'scaler.npz', mean=np.zeros(10), scale=np.ones(10))
    counts = np.bincount(y[split['fit']], minlength=7)
    rank = np.argsort(-counts, kind='stable')
    meta = dict(fixture=True, class_rank=rank.tolist(),
                split_class_counts={'fit': counts.tolist()},
                files={path.name: v.sha(path) for path in data.iterdir()})
    meta['fingerprint'] = v.fingerprint(meta)
    write(data / 'metadata.json', meta)
    configs = {name: [dict(lr=.001), dict(lr=.003)] for name in v.FAMILIES}
    configs['xgboost'] = [dict(n_estimators=600, max_depth=6), dict(n_estimators=600, max_depth=10)]
    protocol = dict(data_fingerprint=meta['fingerprint'], source_sha256=v.sha(source),
                    seeds=list(range(10, 20)), grids=configs, posthoc_taus=list(v.TAUS),
                    neural_epochs=2, mlp_batch_size=32, ft_batch_size=32,
                    ft_device='cpu', mlp_device='cpu', threads=1)
    protocol['fingerprint'] = v.fingerprint(protocol)
    write(run / 'protocol.json', protocol)
    freeze = dict(protocol=protocol['fingerprint'], source_sha256=v.sha(source),
                  selected={name: {'config': values[1]} for name, values in configs.items()})
    freeze['fingerprint'] = v.fingerprint(freeze)
    write(run / 'frozen_selection.json', freeze)
    write(run / 'aggregate.json', {'fixture': True, 'completed_fits': 40, 'readouts': 80})
    for seed in range(10, 20):
        for i, family in enumerate(v.FAMILIES):
            folder = run / 'final' / f'seed_{seed}' / family
            folder.mkdir(parents=True)
            rng = np.random.default_rng(np.random.SeedSequence([seed, i]))
            arrays = {}
            for part in ('calibration', 'confirmation'):
                arrays[part + '_ids'] = split[part]
                arrays[part + '_labels'] = y[split[part]]
                arrays[part + '_logits'] = rng.normal(size=(len(split[part]), 7)).astype(np.float32)
            np.savez_compressed(folder / 'predictions.npz', **arrays)
            prior, weights = v.objective_prior(counts, family)
            tau, curve, logprior = v.calibration_choice(arrays['calibration_logits'], arrays['calibration_labels'], prior)
            training = dict(la_prior=prior.tolist())
            if family == 'xgboost':
                params = {k: value for k, value in freeze['selected'][family]['config'].items() if k != 'n_estimators'}
                params.update(objective='multi:softprob', num_class=7, tree_method='hist', max_bin=256,
                              nthread=1, seed=seed, eval_metric='mlogloss', device='cpu')
                training.update(seed=seed, completed_rounds=600, tree_count=4200, params=params)
            else:
                training.update(identity=dict(config=freeze['selected'][family]['config'], seed=seed,
                                source_sha256=v.sha(source), family=family, epochs=2, batch=32, device='cpu',
                                fit_hash=__import__('hashlib').sha256(split['fit'].tobytes()).hexdigest()),
                                completed_epochs=2, optimizer_updates=2 * math.ceil(len(split['fit']) / 32),
                                class_weights=weights.tolist())
            record = dict(family=family, seed=seed, protocol=protocol['fingerprint'], freeze=freeze['fingerprint'],
                          training=training, predictions_sha256=v.sha(folder / 'predictions.npz'),
                          calibration=dict(tau=tau, prior=prior.tolist(), selection_part='calibration', tie_rule='smallesttau',
                                           curve=[dict(tau=t, macro_f1=f) for t, f in zip(v.TAUS, curve)]),
                          metrics={name: v.metrics(arrays['confirmation_labels'], logits.argmax(1), rank)
                                   for name, logits in [('raw', arrays['confirmation_logits']),
                                                       ('la', arrays['confirmation_logits'] - tau * logprior)]})
            pending = dict(family=family, seed=seed, protocol=protocol['fingerprint'],
                           config=freeze['selected'][family]['config'], training=training)
            write(folder / 'trained_pending_confirmation.json', pending)
            record['early_fitting_receipt_sha256'] = v.sha(folder / 'trained_pending_confirmation.json')
            write(folder / 'metrics.json', record)
    return argparse.Namespace(run=run, source=source, fixture=True)


class PredictionReplayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.args = fixture(Path(self.temp.name))

    def tearDown(self):
        self.temp.cleanup()

    @property
    def target(self):
        return self.args.run / 'final/seed_10/mlp_ce'

    def mutate_arrays(self, edit):
        path = self.target / 'predictions.npz'
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        edit(arrays)
        np.savez_compressed(path, **arrays)
        record = v.read(self.target / 'metrics.json')
        record['predictions_sha256'] = v.sha(path)
        write(self.target / 'metrics.json', record)

    def test_complete_fixture_40_fits_80_views_320_choices_no_models(self):
        result = v.verify(self.args)
        self.assertEqual(result['status'], 'independent_final_prediction_fixture_passed')
        self.assertEqual((result['completed_fits'], result['readouts_recomputed'], result['calibration_candidate_scores_recomputed']), (40, 80, 320))
        self.assertEqual((result['scalar_metrics_recomputed'], result['per_class_metric_values_recomputed'], result['confusion_entries_recomputed']), (640, 1680, 3920))
        self.assertEqual(result['model_weights_loaded'], 0)
        self.assertFalse(result['scientific_runner_imported'])
        self.assertFalse(result['prediction_model_inference'])

    def test_incomplete_gate_precedes_any_array_access(self):
        (self.target / 'metrics.json').unlink()
        with patch.object(v.np, 'load', side_effect=AssertionError('array access before complete gate')):
            with self.assertRaisesRegex(ValueError, 'forty final fits'):
                v.verify(self.args)

    def test_missing_prediction_and_absent_short_aggregate_precede_array_access(self):
        path = self.target / 'predictions.npz'
        backup = path.read_bytes()
        path.unlink()
        with patch.object(v.np, 'load', side_effect=AssertionError('array access before complete prediction grid')):
            with self.assertRaisesRegex(ValueError, 'forty final predictions.npz'):
                v.verify(self.args)
        path.write_bytes(backup)
        aggregate = self.args.run / 'aggregate.json'
        aggregate.unlink()
        with patch.object(v.np, 'load', side_effect=AssertionError('array access before complete aggregate')):
            with self.assertRaises(FileNotFoundError):
                v.verify(self.args)
            write(aggregate, {'completed_fits': 39, 'readouts': 78})
            with self.assertRaisesRegex(ValueError, 'Completed aggregate'):
                v.verify(self.args)

    def test_unsupported_selected_configuration_rejected_before_array_access(self):
        path = self.args.run / 'frozen_selection.json'
        freeze = v.read(path)
        freeze['selected']['mlp_ce']['config'] = {'unsupported': True}
        freeze['fingerprint'] = v.fingerprint({k: value for k, value in freeze.items() if k != 'fingerprint'})
        write(path, freeze)
        with patch.object(v.np, 'load', side_effect=AssertionError('array access before config-grid binding')):
            with self.assertRaisesRegex(ValueError, 'Selected family/configuration'):
                v.verify(self.args)

    def test_neural_identity_family_epochs_batch_and_xgb_params_rejected(self):
        path = self.target / 'metrics.json'
        original = v.read(path)
        pending_path = self.target / 'trained_pending_confirmation.json'
        pending_original = v.read(pending_path)
        for field, value in [('family', 'mlp_cb'), ('epochs', 3), ('batch', 31), ('device', 'mps')]:
            record = copy.deepcopy(original)
            record['training']['identity'][field] = value
            pending = copy.deepcopy(pending_original)
            pending['training'] = record['training']
            write(pending_path, pending)
            record['early_fitting_receipt_sha256'] = v.sha(pending_path)
            write(path, record)
            with self.assertRaisesRegex(ValueError, 'Neural metadata'):
                v.verify(self.args)
        write(pending_path, pending_original)
        write(path, original)
        path = self.args.run / 'final/seed_10/xgboost/metrics.json'
        pending_path = path.with_name('trained_pending_confirmation.json')
        record = v.read(path)
        record['training']['params']['objective'] = 'reg:squarederror'
        pending = v.read(pending_path)
        pending['training'] = record['training']
        write(pending_path, pending)
        record['early_fitting_receipt_sha256'] = v.sha(pending_path)
        write(path, record)
        with self.assertRaisesRegex(ValueError, 'XGBoost configuration/fixed params'):
            v.verify(self.args)

    def test_raw_bytes_hash_corruption_rejected(self):
        path = self.target / 'predictions.npz'
        path.write_bytes(path.read_bytes() + b'corrupt')
        with self.assertRaisesRegex(ValueError, 'raw prediction digest'):
            v.verify(self.args)

    def test_extra_text_field_rejected_after_digest_resigned(self):
        self.mutate_arrays(lambda arrays: arrays.update(raw_text=np.array(['private text'])))
        with self.assertRaisesRegex(ValueError, 'Unexpected prediction archive fields'):
            v.verify(self.args)

    def test_source_id_alignment_rejected_after_digest_resigned(self):
        self.mutate_arrays(lambda arrays: arrays.update(confirmation_ids=arrays['confirmation_ids'][::-1]))
        with self.assertRaisesRegex(ValueError, 'source row or label alignment'):
            v.verify(self.args)

    def test_nonfinite_and_wrong_dtype_logits_rejected(self):
        def edit(arrays):
            arrays['confirmation_logits'][0, 0] = np.nan
        self.mutate_arrays(edit)
        with self.assertRaisesRegex(ValueError, 'logit shape/dtype/finite'):
            v.verify(self.args)
        self.mutate_arrays(lambda arrays: arrays.update(confirmation_logits=np.zeros((14, 7), np.float64)))
        with self.assertRaisesRegex(ValueError, 'logit shape/dtype/finite'):
            v.verify(self.args)

    def test_tau_and_per_class_precision_corruption_rejected(self):
        path = self.target / 'metrics.json'
        original = v.read(path)
        changed = copy.deepcopy(original)
        changed['calibration']['tau'] = 99.
        write(path, changed)
        with self.assertRaisesRegex(ValueError, 'Calibration-only tau'):
            v.verify(self.args)
        changed = copy.deepcopy(original)
        changed['metrics']['raw']['per_class_precision'][0] += .1
        write(path, changed)
        with self.assertRaisesRegex(ValueError, 'Independent confirmation metric'):
            v.verify(self.args)

    def test_source_and_fit_prior_corruption_rejected(self):
        self.args.source.write_text('changed source\n')
        with self.assertRaisesRegex(ValueError, 'Scientific source/protocol/data'):
            v.verify(self.args)
        self.args.source.write_text('"""Numeric fixture only; no trained models."""\n')
        path = self.target / 'metrics.json'
        record = v.read(path)
        record['training']['la_prior'] = [1 / 7] * 7
        pending_path = self.target / 'trained_pending_confirmation.json'
        pending = v.read(pending_path)
        pending['training'] = record['training']
        write(pending_path, pending)
        record['early_fitting_receipt_sha256'] = v.sha(pending_path)
        write(path, record)
        with self.assertRaisesRegex(ValueError, 'Fit-derived objective prior'):
            v.verify(self.args)

    def test_known_confusion_and_group_metrics(self):
        y = np.tile(np.arange(7), 2)
        pred = y.copy()
        pred[0] = 1
        record = v.metrics(y, pred, np.arange(6, -1, -1))
        self.assertAlmostEqual(record['accuracy'], 13 / 14)
        self.assertAlmostEqual(record['macro_f1'], (2 / 3 + 4 / 5 + 5) / 7)
        self.assertAlmostEqual(record['macro_recall'], (1 / 2 + 6) / 7)
        self.assertEqual(record['confusion_matrix'][0][1], 1)
        self.assertEqual(record['head_f1'], 1.)
        self.assertAlmostEqual(record['tail_f1'], (2 / 3 + 4 / 5) / 2)
        self.assertEqual(record['balanced_accuracy'], record['macro_recall'])

    def test_weighted_uniform_prior_metric_and_effective_weights(self):
        y = np.repeat(np.arange(7), [20, 15, 10, 8, 5, 3, 2])
        pred = np.arange(len(y)) % 7
        result = v.metrics(y, pred, np.arange(7))
        matrix = np.bincount(y * 7 + pred, minlength=49).reshape(7, 7)
        uniform = matrix / matrix.sum(1)[:, None]
        expected = (2 * np.diag(uniform) / (uniform.sum(0) + uniform.sum(1))).mean()
        self.assertAlmostEqual(result['balanced_macro_f1'], expected, places=12)
        counts = np.array([1000, 100, 50, 20, 10, 5, 2])
        prior, weights = v.objective_prior(counts, 'mlp_cb')
        self.assertAlmostEqual(float(counts @ weights / counts.sum()), 1., places=6)
        self.assertGreater(weights[-1], weights[0] * 100)
        self.assertAlmostEqual(float(prior.sum()), 1., places=12)

    def test_smallest_tau_tie_and_no_new_grid(self):
        labels = np.tile(np.arange(7), 2)
        logits = np.zeros((14, 7), np.float32)
        tau, curve, logarithm = v.calibration_choice(logits, labels, np.ones(7) / 7)
        self.assertEqual(tau, 0.)
        self.assertEqual(len(set(curve)), 1)
        self.assertTrue(np.array_equal(logarithm, np.zeros(7)))
        with self.assertRaisesRegex(ValueError, 'eight-value tau grid'):
            v.calibration_choice(logits, labels, np.ones(7) / 7, np.linspace(0, 2, 41))


if __name__ == '__main__':
    unittest.main()
