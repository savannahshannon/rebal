"""Actual small AdamW checkpoint fixtures and adversarial state regressions."""
from pathlib import Path
import copy,importlib.util,json,math,tempfile,unittest
import torch
from torch import nn
spec=importlib.util.spec_from_file_location('vision_state_auditor',Path(__file__).resolve().parents[1]/'scripts/verify_vision_checkpoint_states.py')
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)

class TinyModel(nn.Module):
 def __init__(self,partial):
  super().__init__();torch.manual_seed(11);self.encoder=nn.Linear(2,2);self.head=nn.Linear(2,3);self.register_buffer('feature_mean',torch.zeros(2))
  for p in self.encoder.parameters():p.requires_grad_(partial)

def fixture(folder,partial=False,epochs=2,steps=3,lr=.001):
 model=TinyModel(partial);initial=copy.deepcopy(model);name='deit_last2' if partial else 'deit_frozen';identity=dict(protocol='synthetic',name=name,objective='ce',seed=10,lr=lr)
 groups=[dict(params=[p for _,p in named],lr=base,base_lr=base) for named,base in v.parameter_groups(model,name,lr)]
 optimizer=torch.optim.AdamW(groups,weight_decay=.01);history=[];best_state=None
 for epoch in range(epochs):
  for batch in range(steps):
   step=epoch*steps+batch;total=epochs*steps;factor=min(1.,(step+1)/max(1,.1*total))*.5*(1+math.cos(math.pi*max(0,step-.1*total)/max(1,.9*total)))
   for group in optimizer.param_groups:group['lr']=group['base_lr']*factor
   optimizer.zero_grad()
   for p in model.parameters():
    if p.requires_grad:p.grad=torch.full_like(p,.02)
   optimizer.step()
  history.append(dict(epoch=epoch+1,optimizer_updates=(epoch+1)*steps,loss=.7,tuning_macro_f1=.5+.1*epoch/(epochs-1),tuning_raw_macro_f1=.4+.1*epoch/(epochs-1),calibration_selected_tau=0.,seconds=.25,best_epoch=epoch+1))
  best_state=copy.deepcopy(model.state_dict())
 saved=dict(identity=identity,initial=v.state_hash(initial.state_dict()),model=model.state_dict(),best_state=best_state,optimizer=optimizer.state_dict(),completed_epochs=epochs,history=history,best=.6,seconds=.25*epochs)
 receipt=dict(identity=identity,initial=saved['initial'],epochs=epochs,optimizer_updates=epochs*steps,best_epoch=epochs,best_tune=.6,seconds=.25*epochs,trainable_parameters=sum(p.numel() for p in initial.parameters() if p.requires_grad),total_model_parameters=sum(p.numel() for p in initial.parameters()),scaler_fitted_only_on_fit=True)
 write(folder,saved,receipt,history);return identity,initial,saved,receipt,history

def write(folder,saved,receipt,history):
 torch.save(saved,folder/'checkpoint.pt');receipt['checkpoint_sha256']=v.sha(folder/'checkpoint.pt')
 (folder/'training_complete.json').write_text(json.dumps(receipt));(folder/'history.json').write_text(json.dumps(history))

class ActualStateTests(unittest.TestCase):
 def setUp(self):self.temp=tempfile.TemporaryDirectory();self.folder=Path(self.temp.name)
 def tearDown(self):self.temp.cleanup()
 def test_completed_real_adamw_frozen_and_partial_checkpoints_pass(self):
  for partial in (False,True):
   identity,model,*_=fixture(self.folder,partial);result=v.verify_record(self.folder,identity,model,2,3)
   self.assertEqual(result['optimizer_updates'],6);self.assertEqual(result['optimizer_parameter_tensors'],4 if partial else 2)
 def test_wrong_parameter_step_count_rejected_even_after_rehash(self):
  identity,model,saved,r,h=fixture(self.folder,True);next(iter(saved['optimizer']['state'].values()))['step'].fill_(5);write(self.folder,saved,r,h)
  with self.assertRaisesRegex(ValueError,'parameter step'):v.verify_record(self.folder,identity,model,2,3)
 def test_wrong_or_nonfinite_moment_rejected_even_after_rehash(self):
  for change in ('shape','nonfinite','negative'):
   identity,model,saved,r,h=fixture(self.folder,True);state=next(iter(saved['optimizer']['state'].values()))
   if change=='shape':state['exp_avg']=torch.zeros(1)
   elif change=='nonfinite':state['exp_avg'].fill_(float('nan'))
   else:state['exp_avg_sq'].fill_(-1)
   write(self.folder,saved,r,h)
   with self.assertRaises(ValueError):v.verify_record(self.folder,identity,model,2,3)
 def test_frozen_tensor_and_initial_identity_rejected(self):
  for change in ('frozen','initial'):
   identity,model,saved,r,h=fixture(self.folder)
   if change=='frozen':saved['model']['encoder.weight'][0,0]+=1
   else:saved['initial']='0'*64;r['initial']='0'*64
   write(self.folder,saved,r,h)
   with self.assertRaises(ValueError):v.verify_record(self.folder,identity,model,2,3)
 def test_invalid_best_state_or_epoch_rejected(self):
  for change in ('best_state','epoch'):
   identity,model,saved,r,h=fixture(self.folder,True)
   if change=='best_state':saved['best_state']['head.weight'][0,0]+=1
   else:r['best_epoch']=1
   write(self.folder,saved,r,h)
   with self.assertRaises(ValueError):v.verify_record(self.folder,identity,model,2,3)
 def test_altered_optimizer_hyperparameters_or_registration_rejected(self):
  for change in ('weight_decay','betas','lr','ordering','coverage'):
   identity,model,saved,r,h=fixture(self.folder,True);group=saved['optimizer']['param_groups'][0]
   if change=='weight_decay':group['weight_decay']=.1
   elif change=='betas':group['betas']=(.8,.9)
   elif change=='lr':group['lr']*=2
   elif change=='ordering':group['params'].reverse()
   else:saved['optimizer']['state'].pop(next(iter(saved['optimizer']['state'])))
   write(self.folder,saved,r,h)
   with self.assertRaises(ValueError):v.verify_record(self.folder,identity,model,2,3)
 def test_changed_bytes_without_receipt_rehash_rejected(self):
  identity,model,saved,r,h=fixture(self.folder);torch.save({**saved,'best':.1},self.folder/'checkpoint.pt')
  with self.assertRaisesRegex(ValueError,'receipt hash'):v.verify_record(self.folder,identity,model,2,3)
 def test_zero_terminal_encoder_and_head_learning_rates_rejected(self):
  for group_index in (0,1):
   with self.subTest(group=group_index):
    identity,model,saved,r,h=fixture(self.folder,True,epochs=15,steps=120,lr=1e-5)
    v.verify_record(self.folder,identity,model,15,120)
    scheduled=saved['optimizer']['param_groups'][group_index]['lr']
    self.assertGreater(scheduled,0)
    if group_index==1:self.assertLess(scheduled,1e-10)
    saved['optimizer']['param_groups'][group_index]['lr']=0.;write(self.folder,saved,r,h)
    with self.assertRaisesRegex(ValueError,'final scheduled learning rate'):v.verify_record(self.folder,identity,model,15,120)
 def test_false_budget_and_trainability_rejected(self):
  for change in ('budget','trainable','history'):
   identity,model,saved,r,h=fixture(self.folder,True)
   if change=='budget':saved['completed_epochs']=1
   elif change=='trainable':r['trainable_parameters']-=2
   else:h[0]['optimizer_updates']=2;saved['history']=h
   write(self.folder,saved,r,h)
   with self.assertRaises(ValueError):v.verify_record(self.folder,identity,model,2,3)

if __name__=='__main__':unittest.main()
