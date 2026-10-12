"""Regression checks for independent vision verifier completeness and source gates."""
from pathlib import Path
import copy,importlib.util,tempfile,unittest
import numpy as np

spec=importlib.util.spec_from_file_location('vision_portable',Path(__file__).resolve().parents[1]/'scripts/verify_vision_followup.py')
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)

def fixture():
 protocol=dict(fingerprint='p',settings=dict(epochs=15,batch_size=128,seeds=list(v.SEEDS)))
 rows=[dict(arm=arm,seed=seed) for arm in v.ARMS for seed in v.SEEDS]
 summaries=[dict(arm=arm,endpoint=endpoint,metric=metric,n=10,mean=.5,sd=.01)
            for arm in v.ARMS for endpoint in v.ENDPOINTS for metric in v.SCALARS]
 contrasts=[dict(**v.paired_statistic(name,np.zeros(10)),exact_p_holm=1.,t_p_holm=1.) for name,_,_ in v.PAIRS]
 aggregate=dict(status='complete_and_prediction_verified',models=60,readouts=180,protocol='p',selection_sha256='s',summary=summaries,primary_contrasts=contrasts)
 results=dict(protocol='p',selection_sha256='s',rows=rows)
 complete=dict(protocol='p',selection_sha256='s',models=60,epochs=15,training_receipt_sha256={r['arm']+'/'+str(r['seed']):'h' for r in rows})
 return protocol,aggregate,results,complete,'s'

class VisionPortableTests(unittest.TestCase):
 def test_exact_fixture_scope_passes(self):v.verify_scope(*fixture())
 def test_missing_extra_or_reordered_contrasts_rejected(self):
  for change in ('missing','extra','reordered'):
   with self.subTest(change=change):
    a=fixture();rows=a[1]['primary_contrasts']
    if change=='missing':rows.pop()
    elif change=='extra':rows.append(copy.deepcopy(rows[0]))
    else:rows.reverse()
    with self.assertRaisesRegex(ValueError,'five primary'):v.verify_scope(*a)
 def test_empty_missing_duplicate_summary_rejected(self):
  for change in ('empty','missing','duplicate'):
   with self.subTest(change=change):
    a=fixture();rows=a[1]['summary']
    if change=='empty':rows.clear()
    elif change=='missing':rows.pop()
    else:rows[-1]=copy.deepcopy(rows[0])
    with self.assertRaisesRegex(ValueError,'108 finite'):v.verify_scope(*a)
 def test_altered_completion_receipt_keys_rejected(self):
  for change in ('missing','extra','wrong_key','wrong_protocol'):
   with self.subTest(change=change):
    a=fixture();gate=a[3];keys=gate['training_receipt_sha256']
    if change=='missing':keys.pop(next(iter(keys)))
    elif change=='extra':keys['deit_frozen__ce/99']='h'
    elif change=='wrong_key':keys['deit_frozen__ce/99']=keys.pop(next(iter(keys)))
    else:gate['protocol']='other'
    with self.assertRaises(ValueError):v.verify_scope(*a)
 def test_wrong_counts_settings_or_nonfinite_summary_rejected(self):
  for change in ('count','settings','nonfinite'):
   with self.subTest(change=change):
    a=fixture()
    if change=='count':a[1]['readouts']=179
    elif change=='settings':a[0]['settings']['epochs']=14
    else:a[1]['summary'][0]['mean']=float('nan')
    with self.assertRaises(ValueError):v.verify_scope(*a)
 def test_source_and_helper_hashes_required(self):
  with tempfile.TemporaryDirectory() as temp:
   source=Path(temp)/'science.py';helper=Path(temp)/'helper.py';source.write_text('frozen science');helper.write_text('frozen helper')
   protocol=dict(source_sha256=v.sha(source),source_helper_sha256=v.sha(helper));v.verify_sources(protocol,source,helper)
   source.write_text('changed science')
   with self.assertRaisesRegex(ValueError,'source hash'):v.verify_sources(protocol,source,helper)
   source.write_text('frozen science');helper.write_text('changed helper')
   with self.assertRaisesRegex(ValueError,'helper hash'):v.verify_sources(protocol,source,helper)
 def test_zero_and_nonzero_constant_statistics_are_finite(self):
  zero=v.paired_statistic('zero',np.zeros(10));nonzero=v.paired_statistic('constant',np.full(10,.125))
  self.assertEqual((zero['exact_p'],zero['t_p'],zero['ci95_low'],zero['ci95_high']),(1.,1.,0.,0.))
  self.assertEqual((nonzero['exact_p'],nonzero['t_p'],nonzero['ci95_low'],nonzero['ci95_high']),(2/1024,0.,.125,.125))
 def test_independent_metric_regions_follow_rank_order(self):
  labels=np.arange(200);logits=np.eye(200);order=np.arange(200)[::-1]
  logits[0]=0;logits[0,1]=2
  result=v.metrics(logits,labels,order)
  self.assertEqual(result['per_class_support'],[1]*200)
  self.assertAlmostEqual(result['tail_f1'],(65+2/3)/67)
  self.assertEqual(result['head_f1'],1.)

if __name__=='__main__':unittest.main()
