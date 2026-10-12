"""Portable synthetic completed-cell and tamper checks for the verifier.

Run from the repository root with::

    python -m unittest discover -s tests -p 'test_verify_nonimage.py' -v

Only tiny generated tabular/text fixtures are used; no downloaded data or real
benchmark predictions are read. Completed artifacts are produced by the runner,
then checked and deliberately corrupted to exercise independent verification.
"""
import copy
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


v = load('independent_verifier', ROOT/'scripts/verify_nonimage.py')
f = load('synthetic_fixture', ROOT/'tests/test_nonimage.py')
n = f.n


@contextmanager
def restore(*paths):
    original = {path: path.read_bytes() for path in paths}
    try:
        yield
    finally:
        for path, content in original.items():
            path.write_bytes(content)


class VerifierChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.addClassCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory(prefix='rebal_verifier_fixture_')
        cls.addClassCleanup(cls.temp.cleanup)
        root = Path(cls.temp.name)
        cls.data = [f.fixture(root/'tabular', 'tabular'), f.fixture(root/'text', 'text')]
        cls.conditions = [n.load_condition(path) for path in cls.data]
        cls.args = f.args_for(root); cls.args.seeds = [3, 4]; cls.args.out.mkdir()
        cls.protocol = n.make_protocol(cls.conditions, cls.args)
        n.write_json(cls.args.out/'protocol.json', cls.protocol)
        for condition in cls.conditions:
            for seed in cls.args.seeds:
                n.run_seed(condition, cls.args, cls.protocol, seed)
        n.aggregate(cls.conditions, cls.args, cls.protocol)
        cls.cell = cls.args.out/'covertype__fixture/seed_3'
        cls.result = cls.cell/'metrics.json'
        cls.predictions = cls.cell/'predictions.npz'

    def verify(self):
        return v.verify(self.data, self.args.out)

    def edit_result(self, action):
        result = json.loads(self.result.read_text()); action(result)
        self.result.write_text(json.dumps(result))

    def replace_predictions(self, action):
        with np.load(self.predictions, allow_pickle=False) as z:
            arrays = {key: z[key] for key in z.files}
        action(arrays)
        with open(self.predictions, 'wb') as stream:
            np.savez_compressed(stream, **arrays)
        self.edit_result(lambda result: result['artifacts'].__setitem__('predictions.npz', v.sha(self.predictions)))

    def test_complete_independent_metrics_and_aggregate(self):
        report = self.verify()
        self.assertEqual(report['status'], 'verified_complete')
        self.assertEqual(report['completed_cells'], 4)
        self.assertEqual(report['method_evaluations_verified'], 108)
        self.assertEqual(len(report['recomputed_aggregate']['paired_comparisons']), 4)
        self.assertFalse(report['benchmark_success_claimed'])

    def test_hand_computed_confusion_metrics(self):
        metrics = v.metrics(np.array([0, 0, 1, 2]), np.array([0, 1, 1, 0]), np.arange(3))
        self.assertEqual(metrics['confusion_matrix'], [[1, 1, 0], [0, 1, 0], [1, 0, 0]])
        self.assertEqual(metrics['accuracy'], .5)
        self.assertAlmostEqual(metrics['macro_f1'], (.5+2/3)/3)
        self.assertEqual(metrics['worst_f1'], 0.)
        self.assertEqual(metrics['head_tail_f1_gap'], .5)

    def test_metric_tamper_rejected(self):
        with restore(self.result):
            self.edit_result(lambda result: result['metrics']['ce'].__setitem__('accuracy', .987654321))
            with self.assertRaisesRegex(ValueError, 'numeric mismatch'):
                self.verify()

    def test_prediction_tamper_rejected_even_if_artifact_hash_updated(self):
        with restore(self.result, self.predictions):
            def change(arrays):
                logits = arrays['ce__test_logits']; previous = int(logits[0].argmax())
                logits[0] = -100.; logits[0, (previous+1) % 3] = 100.
            self.replace_predictions(change)
            with self.assertRaisesRegex(ValueError, 'numeric mismatch'):
                self.verify()

    def test_calibration_label_alignment_rejected(self):
        with restore(self.result, self.predictions):
            self.replace_predictions(lambda arrays: arrays['calibration_labels'].__setitem__(0, (arrays['calibration_labels'][0]+1) % 3))
            with self.assertRaisesRegex(ValueError, 'calibration labels/order'):
                self.verify()

    def test_adjusted_logit_tamper_preserving_argmax_rejected(self):
        with restore(self.result, self.predictions):
            self.replace_predictions(lambda arrays: arrays.__setitem__('ce_la__test_logits', arrays['ce_la__test_logits']+.125))
            with self.assertRaisesRegex(ValueError, 'adjusted logits mismatch'):
                self.verify()

    def test_calibration_curve_tamper_rejected(self):
        with restore(self.result):
            self.edit_result(lambda result: result['metrics']['ce_la']['adjustment']['curve'][0].__setitem__('macro_f1', .987654321))
            with self.assertRaisesRegex(ValueError, 'calibration_curve'):
                self.verify()

    def test_artifact_hash_tamper_rejected(self):
        path = self.cell/'ce/knn_graph.npy'
        with restore(path):
            path.write_bytes(path.read_bytes()+b'corrupt')
            with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
                self.verify()

    def test_aggregate_comparison_tamper_rejected(self):
        path = self.args.out/'aggregate.json'
        with restore(path):
            result = json.loads(path.read_text()); result['paired_comparisons'][0]['mean'] += .123
            path.write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError, 'aggregate.paired_comparisons'):
                self.verify()

    def test_partial_aggregate_explicitly_incomplete(self):
        marker = self.args.out/'dbpedia__fixture/seed_4/metrics.json'
        aggregate = self.args.out/'aggregate.json'
        with restore(marker, aggregate):
            marker.unlink()
            n.aggregate(self.conditions, self.args, self.protocol)
            report = self.verify()
            self.assertEqual(report['status'], 'incomplete')
            self.assertEqual(report['completed_cells'], 3)
            self.assertEqual(report['expected_cells'], 4)
            self.assertFalse(report['benchmark_success_claimed'])
            self.assertTrue(all('t_holm' not in item for item in report['recomputed_aggregate']['paired_comparisons']))

    def test_pilot_rejected_before_any_numpy_array_read(self):
        path = self.args.out/'protocol.json'
        with restore(path):
            protocol = copy.deepcopy(self.protocol); protocol['execution'] = 'train_only_pilot'; protocol['settings']['pilot_steps'] = 1
            protocol['fingerprint'] = v.fingerprint({k: value for k, value in protocol.items() if k != 'fingerprint'})
            path.write_text(json.dumps(protocol))
            with patch.object(v.np, 'load', side_effect=AssertionError('Array read forbidden')):
                with self.assertRaisesRegex(ValueError, 'intentionally not scored'):
                    self.verify()


if __name__ == '__main__':
    unittest.main(verbosity=2)
