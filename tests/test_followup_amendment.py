import copy
import importlib.util
import itertools
from pathlib import Path
import unittest

import numpy as np
from scipy import stats

spec = importlib.util.spec_from_file_location('followup_amendment',Path(__file__).resolve().parents[1]/'scripts/apply_followup_amendment.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture():
    groups = {}
    for sampler in ('natural','balanced'):
        groups[sampler] = [dict(objective='class_balanced',sampler=sampler,onset=onset,coefficient=coefficient)
                           for onset,coefficient in itertools.product((.2,.5,.8),(.01,.05))]
    for objective in ('ce','balanced_softmax'):
        groups[objective] = [dict(objective=objective,sampler='natural',onset=0.,coefficient=0.,peak_lr=lr)
                             for lr in (.01,.025,.05,.075,.1,.2)]
    groups['bcl_dropout'] = [dict(objective='bcl_dropout',sampler='natural',onset=0.,coefficient=coefficient,temperature=temperature)
                             for coefficient,temperature in itertools.product((.05,.2,.5),(.07,.1))]
    configurations = sum(groups.values(),[]) + [dict(objective='class_balanced',sampler=sampler,onset=0.,coefficient=0.)
                                                for sampler in ('natural','balanced')]
    inventory = [dict(seed=seed,config=config,macro_f1=.2+.003*i+.0001*(seed-10),tune_json_sha256='a'*64)
                 for seed in module.SEEDS for i,config in enumerate(configurations)]
    candidates = {name:[dict(config=config,mean_macro_f1=float(np.mean([row['macro_f1'] for row in inventory if row['config']==config])))
                        for config in configs] for name,configs in groups.items()}
    selection = dict(protocol='a'*64,completed_tuning_cells=320,confirmation_access_before_freeze=False,
                     candidates=candidates,selected={name:rows[-1]['config'] for name,rows in candidates.items()},tuning_inventory=inventory)
    selection['fingerprint'] = module.fingerprint(selection)
    selection['selection_created_utc'] = '2026-10-12T20:23:40Z'
    cells = [dict(seed=seed,protocol='a'*64,selection=selection['fingerprint'],completed_utc='2026-10-12T20:36:00Z',
                  metrics={name:dict(macro_f1=.4+.001*(seed-10)+.0002*i+.00002*i*np.sin(seed))
                           for i,name in enumerate(sorted(module.READOUTS))}) for seed in module.SEEDS]
    summary = []
    for name in sorted(module.READOUTS):
        values = np.array([cell['metrics'][name]['macro_f1'] for cell in cells])
        mean,sd = float(values.mean()),float(values.std(ddof=1))
        half = float(stats.t.ppf(.975,9)*sd/np.sqrt(10))
        summary.append(dict(method=name,n=10,metrics={'macro_f1':dict(mean=mean,sd=sd,ci95=[mean-half,mean+half])}))
    original = dict(status='complete',completed_tuning_cells=320,confirmation_cells=10,readouts=660,seeds=module.SEEDS,
                    selection_frozen_before_confirmation=True,protocol='a'*64,selection=selection['fingerprint'],summary=summary)
    primary = []
    for name,terms in module.PRIMARY.items():
        positive = [key for key,value in terms.items() if value == 1]
        negative = [key for key,value in terms.items() if value == -1]
        expression = dict(a=positive[0],b=negative[0]) if len(positive)==1 else dict(positive=positive,negative=negative)
        primary.append(dict(name=name,endpoint='macro_f1',**expression))
    secondary = [dict(name=name,a=next(key for key,value in terms.items() if value==1),
                      b=next(key for key,value in terms.items() if value==-1)) for name,terms in module.SECONDARY.items()]
    amendment = dict(status='frozen_before_confirmation',source_protocol='a'*64,source_sha256='b'*64,
                     primary_family_size=6,primary_contrasts=primary,secondary_contrasts=secondary,
                     confirmation_cells_at_freeze=0,confirmation_prediction_archives_at_freeze=0,
                     controller_phase_at_freeze='tune',selection_receipt_exists_at_freeze=False,
                     training_changed=False,run_budget_changed=False,controller_counter_snapshot_sha256='c'*64,
                     completed_tuning_cells_at_freeze=148,created_utc='2026-10-12T20:05:52Z')
    amendment['fingerprint'] = module.fingerprint(amendment)
    audit = dict(status='independently_verified_complete',protocol='a'*64,selection=selection['fingerprint'],
                 aggregate_sha256='d'*64,tuning_cells_recomputed=320,confirmation_cells=10,readouts_recomputed=660)
    counter = dict(protocol='a'*64,phase='tune',last_observed_utc='2026-10-12T20:05:48Z',
                   phases=[dict(phase='tune',completed_tuning_cells=148,completed_confirmation_cells=0)])
    digests = dict(original_aggregate='d'*64,analysis_amendment='e'*64,independent_verification='f'*64,counter_snapshot='c'*64,
                   scientific_source='b'*64)
    return [original,cells,selection,amendment,audit,counter,digests]


def refingerprint(amendment):
    amendment['fingerprint'] = module.fingerprint({k:v for k,v in amendment.items() if k!='fingerprint'})


class AmendmentTests(unittest.TestCase):
    def test_complete_numeric_replay_and_fixed_interaction(self):
        args = fixture()
        result = module.apply_amendment(*args)
        self.assertEqual(result['status'],'complete_independently_verified_amended_analysis')
        self.assertEqual(len(result['paired_comparisons']),6)
        self.assertEqual(len(result['secondary_comparisons']),2)
        contrast = result['paired_comparisons'][3]
        first = args[1][0]['metrics']
        manual = ((first['balanced_reference__native']['macro_f1']-first['balanced_off__native']['macro_f1'])-
                  (first['natural_reference__native']['macro_f1']-first['natural_off__native']['macro_f1']))
        self.assertAlmostEqual(contrast['differences'][0],manual,places=14)
        self.assertEqual(result['summary'],args[0]['summary'])
        self.assertTrue(all(np.isfinite(c['signflip_p_holm']) for c in result['paired_comparisons']))

    def test_false_expression_rejected_even_after_valid_fingerprint(self):
        args = fixture()
        args[3]['primary_contrasts'][0]['a'] = 'balanced_reference__native'
        refingerprint(args[3])
        with self.assertRaisesRegex(ValueError,'Exact prospective contrast expressions'):
            module.apply_amendment(*args)

    def test_timezone_aware_postselection_amendment_rejected(self):
        args = fixture()
        # Lexicographic order would incorrectly put 17:24 before 20:23.
        args[3]['created_utc'] = '2026-10-12T17:24:00-03:00'
        refingerprint(args[3])
        with self.assertRaisesRegex(ValueError,'does not precede selection'):
            module.apply_amendment(*args)

    def test_exact_complete_readout_grid_required(self):
        args = fixture()
        args[1][0]['metrics'].pop('ce__native_la')
        with self.assertRaisesRegex(ValueError,'Exact 66-readout grid'):
            module.apply_amendment(*args)

    def test_duplicate_tuning_cell_rejected_even_after_fingerprint(self):
        args = fixture()
        args[2]['tuning_inventory'][-1] = copy.deepcopy(args[2]['tuning_inventory'][-2])
        body = {k:v for k,v in args[2].items() if k not in ('fingerprint','selection_created_utc')}
        args[2]['fingerprint'] = module.fingerprint(body)
        with self.assertRaisesRegex(ValueError,'Exact 32×10 tuning inventory'):
            module.apply_amendment(*args)

    def test_changed_metrics_rejected_by_frozen_summary(self):
        args = fixture()
        args[1][0]['metrics']['ce__native_la']['macro_f1'] += .05
        with self.assertRaisesRegex(ValueError,'Original summary metric differs'):
            module.apply_amendment(*args)

    def test_exact_counter_snapshot_binding(self):
        args = fixture()
        args[6]['counter_snapshot'] = '0'*64
        with self.assertRaisesRegex(ValueError,'counter snapshot hash differs'):
            module.apply_amendment(*args)

    def test_scientific_source_verified_for_unchanged_source_claim(self):
        args = fixture()
        args[6]['scientific_source'] = '0'*64
        with self.assertRaisesRegex(ValueError,'scientific training source is not verified'):
            module.apply_amendment(*args)

    def test_exact_signflip_resolution_constant_zero_and_holm(self):
        zero = module.paired(np.zeros(10))
        self.assertEqual((zero['t_p'],zero['signflip_p'],zero['sd']),(1.,1.,0.))
        constant = module.paired(np.ones(10)*.01)
        self.assertEqual((constant['t_p'],constant['signflip_p'],constant['sd']),(0.,2/1024,0.))
        self.assertEqual(module.holm([2/1024]*6),[.01171875]*6)


if __name__ == '__main__':
    unittest.main()
