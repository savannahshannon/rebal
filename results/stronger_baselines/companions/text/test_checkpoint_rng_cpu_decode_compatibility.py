"""Isolated operational decoder fixtures, never production model inference."""
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import torch

P = Path(__file__).with_name('checkpoint_rng_cpu_decode_compatibility.py')
S = importlib.util.spec_from_file_location('compatibility', P)
M = importlib.util.module_from_spec(S)
S.loader.exec_module(M)

class CompatibilityTests(unittest.TestCase):
    def test_cpu_bytes_and_other_objects_unchanged(self):
        state = torch.arange(64, dtype=torch.uint8)
        model, optimizer = object(), object()
        original = dict(mps=state, torch=torch.get_rng_state(), model=model, optimizer=optimizer)
        fixed = M.decode_rng_cpu_bytes(original)
        self.assertIsNot(fixed, original)
        self.assertIs(original['mps'], state)
        self.assertEqual(fixed['mps'].device.type, 'cpu')
        self.assertTrue(torch.equal(fixed['mps'], state))
        self.assertIs(fixed['model'], model)
        self.assertIs(fixed['optimizer'], optimizer)
        self.assertIs(fixed['torch'], original['torch'])

    def test_absent_and_none_state_keep_dictionary_content(self):
        self.assertEqual(M.decode_rng_cpu_bytes({'python': 1}), {'python': 1})
        self.assertEqual(M.decode_rng_cpu_bytes({'mps': None}), {'mps': None})

    def test_invalid_type_dtype_rank_empty_and_device_rejected(self):
        for state in [b'bytes', [1, 2], torch.ones(2), torch.ones(2, 2, dtype=torch.uint8),
                      torch.empty(0, dtype=torch.uint8), torch.empty(64, dtype=torch.uint8, device='meta')]:
            with self.subTest(state_type=type(state).__name__), self.assertRaises(ValueError):
                M.decode_rng_cpu_bytes({'mps': state})

    def test_exact_frozen_evaluation_argv_accepts(self):
        self.assertEqual(M.validate_evaluation_argv(M.FROZEN_MPS_EVALUATION_ARGV), list(M.FROZEN_MPS_EVALUATION_ARGV))

    def test_training_duplicate_device_seed_batch_or_lr_change_rejected(self):
        base = list(M.FROZEN_MPS_EVALUATION_ARGV)
        variants = [base + ['--stage', 'train'], base + ['--device', 'cpu'], base[:-1], []]
        for flag, value in [('--stage', 'train'), ('--device', 'cpu'), ('--partial-epochs', '4'),
                            ('--partial-batch-size', '512'), ('--partial-lrs', '0.00003')]:
            changed = list(base)
            changed[changed.index(flag) + 1] = value
            variants.append(changed)
        for argv in variants:
            with self.assertRaises(ValueError):
                M.validate_evaluation_argv(argv)

    def test_rejected_duplicates_equal_sign_and_missing_values_never_load_source(self):
        base = list(M.FROZEN_MPS_EVALUATION_ARGV)
        rejected = [base + ['--stage', 'train'], base + ['--device', 'cpu'],
                    base + ['--stage=train'], base + ['--device=cpu'],
                    ['run', '--stage'], ['run', '--device'], ['run', '--stage', 'evaluate', '--device', 'mps']]
        for argv in rejected:
            with mock.patch.object(M.sys, 'argv', ['shim', *argv]), mock.patch.object(M, 'load_frozen_source') as load:
                with self.assertRaises(ValueError):
                    M.main()
                load.assert_not_called()

    def test_wrong_scientific_source_rejected_before_import(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / 'source.py'
            source.write_text('raise AssertionError("must not import")\n')
            with self.assertRaises(ValueError):
                M.load_frozen_source(source)

    def test_only_restore_rng_function_changes_in_loaded_frozen_module(self):
        before = hashlib.sha256(M.SOURCE.read_bytes()).hexdigest()
        module = M.load_frozen_source()
        self.assertEqual(module.train_model.__globals__['restore_rng'], module.restore_rng)
        self.assertEqual(module.train_model.__code__.co_filename, str(M.SOURCE))
        self.assertEqual(hashlib.sha256(M.SOURCE.read_bytes()).hexdigest(), before)
        self.assertEqual(before, M.SOURCE_SHA)

if __name__ == '__main__':
    unittest.main()
