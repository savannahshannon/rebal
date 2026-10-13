#!/usr/bin/env python3
"""Audit trusted, locally created fit artifacts without data inference.

This supplements numeric replay with optimizer and model provenance. Incomplete
trials/fits are skipped; --require-complete requires eight trials and 40 fits.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import numpy as np
import torch


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def state_sha(state):
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        h.update(key.encode())
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def audit(target, record, protocol, fit_ids):
    family = record['family']
    training = record['training']
    config = record['config']
    source = protocol['source_sha256']
    if record['protocol'] != protocol['fingerprint']:
        raise ValueError('Protocol identity mismatch')
    artifacts = {}
    if family == 'xgboost':
        import xgboost as xgb
        model = xgb.Booster()
        model.load_model(target / 'model.ubj')
        actual = json.loads(model.save_config())
        trees = int(actual['learner']['gradient_booster']['gbtree_model_param']['num_trees'])
        if model.num_features() != 54 or model.num_boosted_rounds() != config['n_estimators'] or trees != 7 * config['n_estimators']:
            raise ValueError('Actual tree feature/round budget mismatch')
        if training['model_sha256'] != sha(target / 'model.ubj'):
            raise ValueError('Tree model digest mismatch')
        if training != read(target / 'training.json'):
            raise ValueError('Tree training record mismatch')
        artifacts = {name: sha(target / name) for name in ['model.ubj', 'training.json']}
        updates = config['n_estimators']
        selected = training['selected_rounds']
    else:
        batch = protocol['ft_batch_size'] if family == 'ft_transformer' else protocol['mlp_batch_size']
        steps = math.ceil(len(fit_ids) / batch)
        updates = protocol['neural_epochs'] * steps
        history = read(target / 'history.json')
        last = torch.load(target / 'last.pt', map_location='cpu', weights_only=True)
        best = torch.load(target / 'best.pt', map_location='cpu', weights_only=True)
        identity = training['identity']
        if identity['source_sha256'] != source or identity['config'] != config or identity['family'] != family or identity['batch'] != batch or identity['epochs'] != protocol['neural_epochs'] or identity['fit_hash'] != hashlib.sha256(fit_ids.tobytes()).hexdigest():
            raise ValueError('Neural training identity mismatch')
        if last['identity'] != identity or best['identity'] != identity or last['initial_sha256'] != training['initial_sha256']:
            raise ValueError('Checkpoint identity/initialization mismatch')
        if last['completed_epochs'] != protocol['neural_epochs'] or last['history'] != history or len(history) != protocol['neural_epochs']:
            raise ValueError('Final checkpoint epoch/history mismatch')
        if training['optimizer_updates'] != updates or training['completed_epochs'] != protocol['neural_epochs']:
            raise ValueError('Training budget mismatch')
        for epoch, item in enumerate(history, 1):
            if item['epoch'] != epoch or item['optimizer_updates'] != epoch * steps or not math.isfinite(item['loss']) or not 0 <= item['tune_macro_f1'] <= 1:
                raise ValueError('History budget/range mismatch')
        selected = max(history, key=lambda item: (item['tune_macro_f1'], -item['epoch']))
        if selected['epoch'] != best['epoch'] or selected['epoch'] != training['best_epoch'] or selected['tune_macro_f1'] != best['tune_macro_f1'] or selected['tune_macro_f1'] != training['tune_macro_f1']:
            raise ValueError('Tuning-only best checkpoint mismatch')
        if state_sha(best['model']) != training['selected_state_sha256']:
            raise ValueError('Selected checkpoint tensor digest mismatch')
        if any(not torch.isfinite(value).all() for value in last['model'].values()) or any(not torch.isfinite(value).all() for value in best['model'].values()):
            raise ValueError('Nonfinite checkpoint state')
        for optimizer_state in last['optimizer']['state'].values():
            if int(optimizer_state['step'].item()) != updates:
                raise ValueError('AdamW actual update count mismatch')
        if last['best_epoch'] != training['best_epoch'] or last['best_score'] != training['tune_macro_f1']:
            raise ValueError('Last checkpoint best-choice mismatch')
        selected = selected['epoch']
        artifacts = {name: sha(target / name) for name in ['last.pt', 'best.pt', 'history.json']}
    if 'artifacts' in record and record['artifacts'] != artifacts:
        raise ValueError('Early-fit receipt model/history digest mismatch')
    if (target / 'metrics.json').exists() and 'artifacts' in record:
        final = read(target / 'metrics.json')
        if final['training'] != training or final['early_fitting_receipt_sha256'] != sha(target / 'trained_pending_confirmation.json'):
            raise ValueError('Final prediction record fit provenance mismatch')
    return dict(family=family, seed=record.get('seed', 9), index=record.get('index'),
                optimizer_steps_or_rounds=updates, selected_epoch_or_rounds=selected,
                artifacts=artifacts)


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    root = args.run
    protocol = read(root / 'protocol.json')
    fit_ids = np.load(root / 'data/split.npz', allow_pickle=False)['fit']
    trials = []
    finals = []
    for path in sorted(root.glob('tuning/*/config_*/trial.json')):
        trials.append(audit(path.parent, read(path), protocol, fit_ids))
    for path in sorted(root.glob('final/seed_*/*/trained_pending_confirmation.json')):
        finals.append(audit(path.parent, read(path), protocol, fit_ids))
    if args.require_complete and (len(trials) != 8 or len(finals) != 40):
        raise ValueError('Incomplete inventory for final checkpoint audit')
    result = dict(status='trusted_local_checkpoint_and_fit_provenance_audit_passed',
                  protocol=protocol['fingerprint'], source_sha256=protocol['source_sha256'],
                  training_source_rerun=False, heldout_inference=False,
                  tuning_trials_verified=len(trials), final_fits_verified=len(finals),
                  tuning_trials=trials, final_fits=finals, audit_source_sha256=sha(Path(__file__)))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temp = args.out.with_suffix('.partial.json')
    temp.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    temp.replace(args.out)
    print(json.dumps({k: v for k, v in result.items() if k not in ['tuning_trials', 'final_fits']}))


if __name__ == '__main__':
    main()
