"""Analytical checks for paired inference and incomplete-family rejection."""
import importlib.util
import copy
from pathlib import Path
import unittest
import numpy as np

path=Path(__file__).resolve().parents[1]/'scripts/analyze_followup_comparisons.py'
spec=importlib.util.spec_from_file_location('followup_analysis',path)
analysis=importlib.util.module_from_spec(spec);spec.loader.exec_module(analysis)

class PairedInferenceTests(unittest.TestCase):
    def vision_fixture(self):
        statistic=analysis.paired(np.linspace(-.01,.03,10),1e-14)
        records=[dict(name=name,mean=statistic['mean'],t_p=statistic['t_p'],
                      exact_p=statistic['signflip_p'],ci95_low=statistic['ci95'][0],
                      ci95_high=statistic['ci95'][1],paired_differences=statistic['differences'])
                 for name in sorted(analysis.DECLARED_NAMES['vision'])]
        return dict(status='complete_and_prediction_verified',models=60,readouts=180,
                    summary=[{'n':10}]*108,primary_contrasts=records)

    def test_frozen_vision_identifier_alias_preserves_all_statistics(self):
        canonical=self.vision_fixture();frozen=copy.deepcopy(canonical)
        contrast=next(r for r in frozen['primary_contrasts'] if r['name']=='balanced_softmax_vs_ce_dino')
        contrast['name']='balanced_softmax_vs_ce_dino_frozen'
        before=copy.deepcopy(frozen)
        self.assertEqual(analysis.normalize('vision',frozen),analysis.normalize('vision',canonical))
        self.assertEqual(frozen,before)
        self.assertEqual(analysis.DECLARED_NAMES['vision'],{r['name'] for r in analysis.normalize('vision',frozen)})

    def test_identifier_alias_rejects_undeclared_or_duplicate_contrasts(self):
        for replacement in ('balanced_softmax_vs_ce_dino_other','balanced_softmax_vs_ce_dino_frozen'):
            aggregate=self.vision_fixture()
            contrast=next(r for r in aggregate['primary_contrasts'] if r['name']=='partial_vs_frozen_ce')
            contrast['name']=replacement
            with self.assertRaises(ValueError):analysis.normalize('vision',aggregate)

    def test_exact_ten_seed_resolution_and_zero_effect(self):
        equal=analysis.paired(np.full(10,.01))
        self.assertEqual(equal['signflip_p'],2/1024)
        zero=analysis.paired(np.zeros(10))
        self.assertEqual(zero['signflip_p'],1)
        self.assertEqual(zero['t_p'],1)
        self.assertEqual(zero['ci95'],[0.,0.])

    def test_holm_preserves_identity_and_running_maximum(self):
        self.assertEqual(analysis.holm([.04,.01,.03,.8]),[.09,.04,.09,.8])
        with self.assertRaises(ValueError):analysis.holm([float('nan')])

    def test_missing_study_and_wrong_seed_count_rejected(self):
        with self.assertRaises(ValueError):analysis.analyze({'text':{}})
        with self.assertRaises(ValueError):analysis.paired([.01]*9)
        with self.assertRaises(ValueError):analysis.paired([float('nan')]*10)

    def test_vision_schema_replays_and_detects_false_claim(self):
        statistic=analysis.paired(np.linspace(-.01,.03,10),1e-14)
        record=dict(name='example',mean=statistic['mean'],t_p=statistic['t_p'],
                    exact_p=statistic['signflip_p'],ci95_low=statistic['ci95'][0],
                    ci95_high=statistic['ci95'][1],paired_differences=statistic['differences'])
        aggregate={'status':'complete_and_prediction_verified','models':60,'readouts':180,
                   'summary':[{'n':10}]*108,'primary_contrasts':[dict(record,name=n) for n in analysis.DECLARED_NAMES['vision']]}
        self.assertEqual(len(analysis.normalize('vision',aggregate)),5)
        aggregate['primary_contrasts'][0]['exact_p']=1.
        with self.assertRaises(ValueError):analysis.normalize('vision',aggregate)
        aggregate['primary_contrasts'][0]['exact_p']=statistic['signflip_p']
        aggregate['primary_contrasts'][0]['name']='undeclared'
        with self.assertRaises(ValueError):analysis.normalize('vision',aggregate)
