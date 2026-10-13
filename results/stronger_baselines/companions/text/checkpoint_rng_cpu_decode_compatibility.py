#!/usr/bin/env python3
"""Evaluation-only compatibility adapter for the unchanged frozen Text source.

torch.load(map_location='mps') also maps saved RNG byte storage to MPS, while
torch.mps.set_rng_state expects CPU byte storage. This changes only that storage
location before the frozen restore_rng call; bytes and model computation remain
unchanged. Never use this wrapper to train or alter the fixed experiment plan.
"""
from pathlib import Path
import hashlib
import importlib.util
import sys
import torch

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'work/rebal/scripts/benchmark_text_followup.py'
SOURCE_SHA = '8f05dc4a87813a9ffa4c1ac1b1de322cb0315cdb0c3237914bcefaa0cb12b83c'
FROZEN_MPS_EVALUATION_ARGV = ('run',
 '--data',
 'work/next_round/text/data',
 '--cache',
 'work/next_round/text/full_bert_cache',
 '--modeldir',
 'work/next_round/text/bert-base-uncased',
 '--out',
 'work/experiments/rebal_next_text',
 '--threads',
 '2',
 '--stage',
 'evaluate',
 '--partial-epochs',
 '3',
 '--adapter-epochs',
 '50',
 '--partial-batch-size',
 '1024',
 '--batch-size',
 '1024',
 '--partial-precision',
 'amp_float16',
 '--attention',
 'sdpa',
 '--adapter-lrs',
 '0.01',
 '0.05',
 '--partial-lrs',
 '0.00002',
 '0.00005',
 '--eq-weight',
 '0.05',
 '--max-words',
 '50000',
 '--methods',
 'bert_ce',
 'bert_bsoftmax',
 'bert_natural_eq',
 'bert_balanced_ce',
 'bert_balanced_eq',
 'mean_ce',
 'mean_bsoftmax',
 'last0_ce',
 'last0_bsoftmax',
 'last1_ce',
 'last1_bsoftmax',
 'last2_ce',
 'last2_bsoftmax',
 '--seeds',
 '10',
 '11',
 '12',
 '13',
 '14',
 '15',
 '16',
 '17',
 '18',
 '19',
 '--device',
 'mps',
 '--only-methods',
 'last1_ce',
 'last1_bsoftmax',
 'last2_ce',
 'last2_bsoftmax')

def decode_rng_cpu_bytes(record):
    """Return a shallow copy, moving only non-None saved MPS RNG bytes to CPU."""
    fixed = dict(record)
    state = fixed.get('mps')
    if state is not None:
        if not isinstance(state, torch.Tensor) or state.dtype != torch.uint8 or state.ndim != 1 or state.numel() == 0 or state.device.type not in ('cpu', 'mps'):
            raise ValueError('Saved MPS RNG storage is not a one-dimensional byte tensor')
        fixed['mps'] = state.detach().cpu()
        if not torch.equal(fixed['mps'], state.detach().cpu()):
            raise ValueError('Saved MPS RNG bytes changed during CPU decode')
    return fixed

def load_frozen_source(source=SOURCE):
    source = Path(source)
    if hashlib.sha256(source.read_bytes()).hexdigest() != SOURCE_SHA:
        raise ValueError('Frozen Text scientific source changed')
    sys.path.insert(0, str(source.parent))
    spec = importlib.util.spec_from_file_location('frozen_text_checkpoint_decode', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    frozen_restore = module.restore_rng
    def restore_rng(record):
        return frozen_restore(decode_rng_cpu_bytes(record))
    module.restore_rng = restore_rng
    return module

def validate_evaluation_argv(argv):
    if tuple(argv) != FROZEN_MPS_EVALUATION_ARGV:
        raise ValueError('Compatibility adapter permits only the exact frozen MPS evaluation argv')
    return list(argv)


def main():
    argv = validate_evaluation_argv(sys.argv[1:])
    module = load_frozen_source()
    sys.argv = [str(SOURCE), *argv]
    module.main()

if __name__ == '__main__':
    main()
