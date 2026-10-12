#!/usr/bin/env python3
"""Independent artifact audit; deliberately imports no benchmark functions."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def close(actual, expected, label, atol=1e-12):
    if not np.allclose(actual, expected, atol=atol, rtol=0, equal_nan=False):
        raise AssertionError(label)


def independent_metrics(labels, predictions, ranks):
    classes = len(ranks)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=np.arange(classes), zero_division=0)
    out = dict(accuracy=float(accuracy_score(labels, predictions)),
               macro_recall=float(balanced_accuracy_score(labels, predictions)),
               macro_f1=float(f1.mean()), worst_f1=float(f1.min()),
               zero_f1_classes=int(np.sum(f1 == 0)))
    for name, mask in [('head', ranks < classes // 3),
                       ('mid', (ranks >= classes // 3) & (ranks < 2 * classes // 3)),
                       ('tail', ranks >= 2 * classes // 3)]:
        out[name + '_f1'] = float(f1[mask].mean())
        out[name + '_recall'] = float(recall[mask].mean())
    out['head_tail_f1_gap'] = out['head_f1'] - out['tail_f1']
    per_class = dict(precision=precision, recall=recall, f1=f1, support=support,
                     predicted=np.bincount(predictions, minlength=classes))
    return out, per_class


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--cache', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--expected-steps', type=int, default=4000)
    ap.add_argument('--require-complete', action='store_true')
    args = ap.parse_args()
    protocol = json.loads((args.root / 'protocol.json').read_text())
    manifest = json.loads((args.root / 'feature_manifest.json').read_text())
    paths = sorted(args.root.glob('rho_*/seed_*/metrics.json'))
    planned = {(float(r), int(s)) for r in protocol['planned_rhos'] for s in protocol['planned_seeds']}
    assert protocol['steps'] == args.expected_steps
    assert digest(args.cache) == protocol['cache_sha256'] == manifest['features_sha256']
    original_manifest = args.cache.with_suffix('.manifest.json')
    assert digest(original_manifest) == protocol['manifest_sha256']
    assert json.loads(original_manifest.read_text()) == manifest
    with np.load(args.cache, allow_pickle=False) as cache:
        train_labels, eval_labels = cache['train_labels'], cache['eval_labels']
    assert train_labels.shape == (100000,) and eval_labels.shape == (10000,)
    assert np.array_equal(np.bincount(eval_labels, minlength=200), np.full(200, 50))
    train_hashes = np.asarray(manifest['train_pixel_sha256'])
    eval_hashes = set(manifest['eval_pixel_sha256'])
    assert len(eval_hashes) == 10000
    summaries, records, splits, errors, scalar_count = [], {}, {}, [], 0
    max_error = 0.0
    expected_arms = {'ce', 'cb', 'ros', 'smote', 'ce_la', 'smote_la', 'ncm'}
    for path in paths:
        cell = path.parent
        try:
            result = json.loads(path.read_text())
            rho = float(cell.parent.name.split('_')[1])
            seed = int(cell.name.split('_')[1])
            assert (rho, seed) in planned
            assert result['protocol'] == dict(protocol, rho=rho, seed=seed)
            assert set(result['arms']) == expected_arms
            for artifact, expected_hash in result['artifacts'].items():
                assert digest(cell / artifact) == expected_hash, artifact + ' checksum'
            with np.load(cell / 'split.npz', allow_pickle=False) as arrays:
                split = {key: arrays[key] for key in arrays.files}
            fit, hold = split['train_indices'], split['holdout_indices']
            excluded, ranks = split['excluded_indices'], split['class_rank']
            assert np.array_equal(np.sort(np.r_[fit, hold, excluded]), np.arange(100000))
            assert np.array_equal(split['final_eval_indices'], np.arange(10000))
            assert np.array_equal(np.sort(ranks), np.arange(200))
            assert np.array_equal(ranks[split['class_order']], np.arange(200))
            assert np.array_equal(np.bincount(train_labels[hold], minlength=200), np.full(200, 50))
            counts = np.bincount(train_labels[fit], minlength=200)
            expected_counts = np.floor(445 * rho ** (-ranks / 199)).astype(int)
            assert np.array_equal(counts, expected_counts)
            assert np.array_equal(counts, split['train_counts'])
            assert np.array_equal(counts, result['train_counts'])
            assert np.array_equal(ranks, result['class_rank'])
            assert int(split['max_train_per_class']) == result['max_train_per_class'] == 445
            assert len(fit) == result['training_examples']
            assert len(hold) == result['holdout_examples'] == result['final_eval_examples'] == 10000
            close(counts.max() / counts.min(), result['actual_max_min_ratio'], 'actual ratio')
            fit_hash_set, hold_hash_set = set(train_hashes[fit]), set(train_hashes[hold])
            assert len(fit_hash_set) == len(fit) and len(hold_hash_set) == len(hold)
            assert not fit_hash_set.intersection(hold_hash_set | eval_hashes)
            assert not hold_hash_set.intersection(eval_hashes)
            assert set(split['hygiene_excluded_indices']) == set(protocol['data_hygiene']['excluded_indices'])
            assert set(split['hygiene_excluded_indices']).issubset(set(excluded))
            regions = [int(np.sum(ranks < 66)), int(np.sum((ranks >= 66) & (ranks < 133))),
                       int(np.sum(ranks >= 133))]
            assert regions == [66, 67, 67]
            predictions, recomputed = {}, {}
            for arm in sorted(expected_arms):
                with np.load(cell / 'predictions' / (arm + '.npz'), allow_pickle=False) as arrays:
                    predictions[arm] = {key: arrays[key] for key in arrays.files}
                for partition, prefix, labels in [('final_eval', 'eval', eval_labels),
                                                   ('holdout', 'holdout', train_labels[hold])]:
                    pred = predictions[arm][prefix + '_prediction']
                    assert pred.shape == (10000,) and np.issubdtype(pred.dtype, np.integer)
                    assert pred.min() >= 0 and pred.max() < 200
                    if prefix + '_logits' in predictions[arm]:
                        logits = predictions[arm][prefix + '_logits']
                        assert logits.shape == (10000, 200) and np.isfinite(logits).all()
                        assert np.array_equal(pred, logits.argmax(1))
                    measured, per_class = independent_metrics(labels, pred, ranks)
                    close(measured['accuracy'], measured['macro_recall'], 'balanced labels imply accuracy=recall')
                    stored = result['arms'][arm][partition]
                    for metric, value in measured.items():
                        close(value, stored[metric], f'{arm}/{partition}/{metric}')
                        max_error = max(max_error, abs(value - stored[metric]))
                        scalar_count += 1
                    for metric, values in per_class.items():
                        close(values, stored['per_class'][metric], f'{arm}/{partition}/per_class/{metric}')
                    if partition == 'final_eval':
                        recomputed[arm] = measured
            for arm in ['ce', 'cb', 'ros', 'smote']:
                evidence = result['arms'][arm]
                assert evidence['training']['optimizer_updates'] == args.expected_steps
                assert evidence['training']['sampled_examples'] == args.expected_steps * protocol['batch_size']
                assert evidence['sampling']['real_examples'] == len(fit)
                if arm in ['ros', 'smote']:
                    assert evidence['sampling']['target_per_class'] == 445
                    assert evidence['sampling']['added_examples'] == 89000 - len(fit)
                    assert evidence['sampling']['random_state'] == seed
                else:
                    assert evidence['sampling']['added_examples'] == 0
            assert result['arms']['smote']['sampling']['k_neighbors'] == min(5, int(counts.min()) - 1)
            with np.load(cell / 'ros_source_indices.npz', allow_pickle=False) as ros:
                assert np.array_equal(fit[ros['pool_to_train_indices']], ros['pool_to_cache_indices'])
                ros_counts = np.bincount(train_labels[ros['pool_to_cache_indices']], minlength=200)
                assert np.array_equal(ros_counts, np.full(200, 445))
                assert np.array_equal(ros['pool_to_train_indices'][:len(fit)], np.arange(len(fit)))
            for base_arm in ['ce', 'smote']:
                arm = base_arm + '_la'
                prior = np.log(counts / counts.sum()).astype(np.float32)
                close(prior, predictions[arm]['log_prior'], arm + ' log prior')
                curve = []
                for tau in sorted(protocol['taus']):
                    pred = (predictions[base_arm]['holdout_logits'] - tau * prior).argmax(1)
                    curve.append((tau, balanced_accuracy_score(train_labels[hold], pred)))
                tau = max(curve, key=lambda pair: pair[1])[0]
                assert tau == result['arms'][arm]['tau'] == float(predictions[arm]['tau'])
                for computed, stored in zip(curve, result['arms'][arm]['tau_curve']):
                    assert computed[0] == stored['tau']
                    close(computed[1], stored['macro_recall'], arm + ' tau selection')
                for prefix in ['eval', 'holdout']:
                    expected = (predictions[base_arm][prefix + '_logits'] - tau * prior).argmax(1)
                    assert np.array_equal(expected, predictions[arm][prefix + '_prediction'])
            records[(rho, seed)], splits[(rho, seed)] = recomputed, split
            summaries.append(dict(rho=rho, seed=seed, training_examples=len(fit),
                                  min_train=int(counts.min()), max_train=int(counts.max()),
                                  regions=regions, arms=7, checks='passed'))
        except Exception as exc:
            errors.append(dict(cell=str(cell.relative_to(args.root)), error=f'{type(exc).__name__}: {exc}'))
    for seed in protocol['planned_seeds']:
        keys = sorted([key for key in splits if key[1] == seed])
        for lower, higher in zip(keys, keys[1:]):
            a, b = splits[lower], splits[higher]
            if (not np.array_equal(a['holdout_indices'], b['holdout_indices'])
                    or not np.array_equal(a['class_rank'], b['class_rank'])
                    or not set(b['train_indices']).issubset(set(a['train_indices']))):
                errors.append(dict(cell=f'seed_{seed}', error='cross-rho holdout/rank/nested-fit mismatch'))
    aggregate_checked = False
    if args.require_complete:
        try:
            assert set(records) == planned, 'missing planned completed cells'
            aggregate = json.loads((args.root / 'aggregate.json').read_text())
            assert aggregate['complete'] is True and aggregate['observed_cells'] == len(planned)
            for rho_key, group in aggregate['by_rho'].items():
                rho = float(rho_key)
                assert group['seeds'] == sorted(protocol['planned_seeds'])
                assert set(group['arms']) == expected_arms
                for arm, metrics in group['arms'].items():
                    for metric, stored in metrics.items():
                        vals = np.asarray([records[(rho, s)][arm][metric] for s in group['seeds']])
                        close(vals.mean(), stored['mean'], 'aggregate mean')
                        close(vals.std(ddof=1), stored['std'], 'aggregate standard deviation')
                        assert stored['n'] == len(vals)
                        radius = stats.t.ppf(0.975, len(vals)-1) * vals.std(ddof=1) / np.sqrt(len(vals))
                        close([vals.mean()-radius, vals.mean()+radius], stored['ci95'], 'aggregate interval')
            ps, exact_ps = [], []
            expected_comparisons = {(float(rho), 'smote', ref) for rho in protocol['planned_rhos']
                                    for ref in ['ros', 'ce_la']}
            assert {(c['rho'], c['intervention'], c['reference']) for c in aggregate['comparisons']} == expected_comparisons
            for comparison in aggregate['comparisons']:
                rho, seeds = comparison['rho'], comparison['seeds']
                assert seeds == sorted(protocol['planned_seeds'])
                ds = np.asarray([records[(rho, s)][comparison['intervention']][comparison['metric']]
                                 - records[(rho, s)][comparison['reference']][comparison['metric']]
                                 for s in seeds])
                close(ds, comparison['seed_deltas'], 'paired deltas')
                close(ds.mean(), comparison['mean'], 'paired mean')
                close(ds.std(ddof=1), comparison['std'], 'paired standard deviation')
                assert comparison['n'] == len(ds)
                radius = stats.t.ppf(0.975, len(ds)-1) * ds.std(ddof=1) / np.sqrt(len(ds))
                close([ds.mean()-radius, ds.mean()+radius], comparison['ci95'], 'paired interval')
                p = float(stats.ttest_1samp(ds, 0).pvalue)
                nulls = [abs(np.mean(ds * signs)) for signs in itertools.product([-1, 1], repeat=len(ds))]
                exact_p = sum(value >= abs(ds.mean()) - 1e-15 for value in nulls) / len(nulls)
                close(p, comparison['p_two_sided'], 'paired t p', atol=1e-10)
                close(exact_p, comparison['p_signflip_two_sided'], 'paired signflip p')
                ps.append(p)
                exact_ps.append(exact_p)
            assert len(ps) == 6
            for values, field in [(ps, 'p_holm'), (exact_ps, 'p_signflip_holm')]:
                prior_adjusted = 0.0
                for rank, index in enumerate(np.argsort(values)):
                    adjusted = min(1.0, max(prior_adjusted, (len(values)-rank)*values[index]))
                    close(adjusted, aggregate['comparisons'][index][field], 'Holm adjustment', atol=1e-10)
                    prior_adjusted = adjusted
            aggregate_checked = True
        except Exception as exc:
            errors.append(dict(cell='aggregate', error=f'{type(exc).__name__}: {exc}'))
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(), root=str(args.root),
                  independent_library='sklearn.metrics; scipy.stats; no benchmark imports',
                  completed_cells_seen=len(paths), passed_cells=len(summaries), expected_cells=len(planned),
                  scalar_metrics_recomputed=scalar_count, max_scalar_metric_error=max_error,
                  aggregate_checked=aggregate_checked, expected_optimizer_updates=args.expected_steps,
                  balanced_eval_count_per_class=50, summaries=summaries, errors=errors,
                  limitations=['SMOTE pool geometry is not saved: its target counts are metadata-checked.',
                               'Optimizer update counts are checked against saved logs, not replayed training.'])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'summaries'}, indent=2))
    raise SystemExit(bool(errors))


if __name__ == '__main__':
    main()
