"""Independent formula and checkpoint-sidecar fixtures need no dataset or BERT."""
from pathlib import Path
import copy,importlib.util,json,tempfile,unittest
import numpy as np
import torch
P=Path(__file__).resolve().parents[1]/'scripts/verify_text_numeric.py'
s=importlib.util.spec_from_file_location('text_numeric_auditor_public',P);a=importlib.util.module_from_spec(s);s.loader.exec_module(a)
def write(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n')
def fit_fixture(folder):
 identity=dict(method='last0_ce',seed=10,protocol='a'*64);protocol=dict(partial_epochs=2,partial_steps_per_epoch=3,fit_examples=14)
 history=[dict(epoch=i,optimizer_updates=3*i,batches=3,loss=.3,seconds=.1,EQ_eligible_batches_per_class=[0]*14) for i in (1,2)];write(folder/'history.json',history)
 labels=np.arange(14);logits=np.eye(14,dtype=np.float32);np.save(folder/'tune_logits.npy',logits)
 state={'adapter.head.weight':torch.zeros(14,2),'adapter.head.bias':torch.zeros(14)}
 optimizer={'param_groups':[{'params':[0,1]}],'state':{i:{'step':torch.tensor(6.),'exp_avg':torch.zeros_like(t),'exp_avg_sq':torch.zeros_like(t)} for i,t in enumerate(state.values())}}
 torch.save(dict(identity=identity,initial_state_sha256='b'*64,completed_epochs=2,history=history,model=state,optimizer=optimizer),folder/'checkpoint.pt')
 selection=dict(identity=identity,epochs=2,updates=6,fit_examples=14,initial_state_sha256='b'*64,final_state_sha256=a.state_hash(state),total_parameters=42,trainable_parameters=42,tune_macro_f1=1.,tune_logits_sha256=a.sha(folder/'tune_logits.npy'),checkpoint_sha256=a.sha(folder/'checkpoint.pt'))
 budget=dict(identity=identity,planned_epochs=2,completed_epochs=2,steps_per_epoch=3,fit_examples_per_epoch=14,initial_state_sha256='b'*64,files={'history.json':a.sha(folder/'history.json'),'checkpoint.pt':selection['checkpoint_sha256']});write(folder/'budget.json',budget);selection['budget_sha256']=a.sha(folder/'budget.json');write(folder/'selection.json',selection)
 return selection,identity,protocol,np.arange(14),labels
class FormulaTests(unittest.TestCase):
 def test_confusion_with_absent_classes_and_inverse_frequency_rank(self):
  r=a.metrics(np.array([0,0,1,1],np.int8),np.array([0,1,1,2],np.int8),np.arange(14)[::-1]);self.assertAlmostEqual(r['macro_f1'],(2/3+.5)/14);self.assertAlmostEqual(r['tail_f1'],(2/3+.5)/5);self.assertEqual(r['head_f1'],0)
 def test_constant_signflip_edges_and_holm_eight(self):
  zero=a.paired(np.zeros(10));self.assertEqual((zero['sd'],zero['t_p'],zero['signflip_p']),(0,1,1));positive=a.paired(np.ones(10)*.02);self.assertEqual((positive['sd'],positive['t_p'],positive['signflip_p']),(0,0,2/1024));self.assertEqual(a.holm([2/1024]*8),[.015625]*8)
 def test_calibration_smallest_tau_tie_and_prior(self):
  y=np.arange(14);logits=100*np.eye(14,dtype=np.float32);counts=np.arange(1,15);r=a.calibration(logits,y,counts,np.arange(14));self.assertEqual(r['tau'],0);np.testing.assert_allclose(r['prior'],counts/counts.sum());self.assertTrue(all(p['macro_f1']==1 for p in r['curve']))
  changed=copy.deepcopy(r);changed['curve'][0]['macro_f1']=.9
  with self.assertRaises(ValueError):a.compare_calibration(changed,r)
 def test_invalid_logits_and_invalid_rank_rejected(self):
  with self.assertRaises(ValueError):a.logits(np.full((2,14),np.nan),2)
  with self.assertRaises(ValueError):a.metrics(np.array([0]),np.array([0]),np.zeros(14,dtype=int))
class StateTests(unittest.TestCase):
 def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.folder=Path(self.tmp.name);self.parameters=fit_fixture(self.folder)
 def tearDown(self):self.tmp.cleanup()
 def check(self,portable=False):return a.validate_fit(self.folder,*self.parameters,portable=portable)
 def rebind(self,checkpoint):
  torch.save(checkpoint,self.folder/'checkpoint.pt');selection,identity,protocol,rank,labels=self.parameters;budget=a.read(self.folder/'budget.json');budget['files']['checkpoint.pt']=a.sha(self.folder/'checkpoint.pt');write(self.folder/'budget.json',budget);selection['checkpoint_sha256']=budget['files']['checkpoint.pt'];selection['budget_sha256']=a.sha(self.folder/'budget.json');write(self.folder/'selection.json',selection)
 def test_complete_actual_state_and_explicit_omission(self):
  self.assertEqual(self.check(),(1.,True));(self.folder/'checkpoint.pt').unlink()
  with self.assertRaisesRegex(ValueError,'Actual checkpoint is missing'):self.check()
  self.assertEqual(self.check(portable=True),(1.,False))
 def test_rehashed_wrong_adamw_step_rejected(self):
  ck=torch.load(self.folder/'checkpoint.pt',weights_only=False);ck['optimizer']['state'][0]['step']=torch.tensor(5.);self.rebind(ck)
  with self.assertRaisesRegex(ValueError,'AdamW update counts'):self.check()
 def test_rehashed_wrong_moment_shape_rejected(self):
  ck=torch.load(self.folder/'checkpoint.pt',weights_only=False);ck['optimizer']['state'][0]['exp_avg']=torch.zeros(1);self.rebind(ck)
  with self.assertRaisesRegex(ValueError,'moment shape/precision'):self.check()
 def test_rehashed_incomplete_history_budget_rejected(self):
  history=a.read(self.folder/'history.json');history[-1]['optimizer_updates']=5;write(self.folder/'history.json',history)
  with self.assertRaisesRegex(ValueError,'history incomplete'):self.check()
if __name__=='__main__':torch.set_num_threads(1);unittest.main()
