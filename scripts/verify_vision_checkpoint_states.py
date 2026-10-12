"""Independent actual-state audit of all 72 completed vision checkpoints.

No scientific validator, model forward pass, or confirmation prediction file is
used. Shapes and initial values are independently constructed from official
DeiT tensors and the fit-only feature scaler. This is a local weight audit;
portable numeric replay remains separate and may legitimately contain no weights.
"""
from pathlib import Path
import argparse,copy,hashlib,importlib.metadata,itertools,json,math
from datetime import datetime,timezone
import numpy as np
import torch
from torch import nn

NAMES=('deit_frozen','deit_last2','dino_frozen');OBJECTIVES=('ce','balanced_softmax');SEEDS=tuple(range(10,20))
def require(ok,message):
 if not ok:raise ValueError(message)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
 return h.hexdigest()
def fp(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def state_hash(state):
 h=hashlib.sha256()
 for name,value in sorted(state.items()):h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def load(path):return json.loads(Path(path).read_text())
def close(x,y,message):require(math.isfinite(float(x)) and math.isfinite(float(y)) and abs(float(x)-float(y))<=1e-10,message)
def close_lr(x,y,message):require(math.isfinite(float(x)) and math.isfinite(float(y)) and math.isclose(float(x),float(y),rel_tol=1e-10,abs_tol=1e-16),message)

class ExpectedModel(nn.Module):
 def __init__(self,name,seed,backbone=None,scaler=None,width=192,classes=200):
  super().__init__()
  if name!='dino_frozen':
   self.blocks=copy.deepcopy(backbone.blocks[10:]);self.norm=copy.deepcopy(backbone.norm)
   self.register_buffer('feature_mean',torch.from_numpy(scaler[0].copy()))
   self.register_buffer('feature_sd',torch.from_numpy(scaler[1].copy()))
   for p in itertools.chain(self.blocks.parameters(),self.norm.parameters()):p.requires_grad_(name=='deit_last2')
  else:width=384 if width==192 else width
  torch.manual_seed(seed+901);self.head=nn.Linear(width,classes);nn.init.trunc_normal_(self.head.weight,std=.02);nn.init.zeros_(self.head.bias)

def parameter_groups(model,name,lr):
 head=[(n,p) for n,p in model.named_parameters() if n.startswith('head.')]
 encoder=[(n,p) for n,p in model.named_parameters() if not n.startswith('head.') and p.requires_grad]
 return [(head,100*lr if name=='deit_last2' else lr)]+([(encoder,lr)] if encoder else [])

def verify_record(folder,identity,expected,epochs,steps):
 folder=Path(folder);receipt=load(folder/'training_complete.json');history=load(folder/'history.json')
 require(receipt['identity']==identity,'Training receipt identity differs')
 require(sha(folder/'checkpoint.pt')==receipt['checkpoint_sha256'],'Checkpoint receipt hash differs')
 saved=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=True)
 require(saved['identity']==identity and saved['initial']==receipt['initial']==state_hash(expected.state_dict()),'Independent initial-state identity differs')
 require(saved['completed_epochs']==receipt['epochs']==epochs and receipt['optimizer_updates']==epochs*steps and saved['history']==history and len(history)==epochs,'Actual completed training budget differs')
 scores=[];elapsed=0.
 for i,row in enumerate(history):
  require(row['epoch']==i+1 and row['optimizer_updates']==(i+1)*steps,'History epoch/update budget differs')
  require(all(math.isfinite(row[k]) for k in ('loss','tuning_macro_f1','tuning_raw_macro_f1','calibration_selected_tau','seconds')) and
          row['loss']>=0 and row['seconds']>=0 and 0<=row['tuning_macro_f1']<=1 and 0<=row['tuning_raw_macro_f1']<=1,'Invalid training history scalar')
  scores.append(row['tuning_macro_f1']);elapsed+=row['seconds']
  require(row['best_epoch']==int(np.argmax(scores))+1,'History best-epoch progression differs')
 best_epoch=int(np.argmax(scores))+1
 require(receipt['best_epoch']==best_epoch,'Receipt best epoch differs')
 close(saved['best'],max(scores),'Checkpoint best score differs');close(receipt['best_tune'],max(scores),'Receipt best score differs')
 close(saved['seconds'],elapsed,'Checkpoint elapsed time differs');close(receipt['seconds'],elapsed,'Receipt elapsed time differs')
 template=expected.state_dict();eligible={n for n,p in expected.named_parameters() if p.requires_grad}
 parameter_names={n for n,_ in expected.named_parameters()};fixed=set(template)-eligible
 for kind in ('model','best_state'):
  state=saved[kind];require(set(state)==set(template),'Model state tensor inventory differs')
  for key,reference in template.items():
   tensor=state[key];require(tensor.shape==reference.shape and tensor.dtype==reference.dtype and bool(torch.isfinite(tensor).all()),'Model state tensor shape/dtype/finite guard failed')
   if key in fixed:require(torch.equal(tensor,reference),'Frozen parameter or fit-only scaler buffer changed')
 if best_epoch==epochs:require(all(torch.equal(saved['model'][key],saved['best_state'][key]) for key in template),'Last-epoch best state differs from final state')
 trainable=sum(p.numel() for p in expected.parameters() if p.requires_grad);total=sum(p.numel() for p in expected.parameters())
 require(receipt['trainable_parameters']==trainable and receipt['total_model_parameters']==total and receipt['scaler_fitted_only_on_fit'] is True,'Trainable parameter/scaler metadata differs')
 groups=parameter_groups(expected,identity['name'],identity['lr']);optimizer=saved['optimizer'];actual_groups=optimizer['param_groups']
 require(len(actual_groups)==len(groups),'AdamW parameter group count differs')
 all_ids=[];moments=0;total_steps=epochs*steps;last_step=total_steps-1
 factor=min(1.,(last_step+1)/max(1,.1*total_steps))*.5*(1+math.cos(math.pi*max(0,last_step-.1*total_steps)/max(1,.9*total_steps)))
 for actual,(parameters,base_lr) in zip(actual_groups,groups):
  ids=actual['params'];require(len(ids)==len(parameters),'AdamW trainable tensor count differs');all_ids+=ids
  close_lr(actual['base_lr'],base_lr,'AdamW base learning rate differs');close_lr(actual['lr'],base_lr*factor,'AdamW final scheduled learning rate differs')
  require(actual['weight_decay']==.01 and tuple(actual['betas'])==(.9,.999) and actual['eps']==1e-8 and actual['amsgrad'] is False and actual['maximize'] is False,'AdamW hyperparameters differ')
  for pid,(name,parameter) in zip(ids,parameters):
   require(pid in optimizer['state'] and name in eligible and name in parameter_names,'AdamW parameter eligibility differs')
   value=optimizer['state'][pid];require(set(value)=={'step','exp_avg','exp_avg_sq'},'AdamW moment field inventory differs')
   step=value['step'];require(torch.is_tensor(step) and step.numel()==1 and bool(torch.isfinite(step).all()) and float(step)==total_steps,'Actual AdamW parameter step count differs')
   for field in ('exp_avg','exp_avg_sq'):
    moment=value[field];require(moment.shape==parameter.shape and moment.dtype==parameter.dtype and bool(torch.isfinite(moment).all()),'AdamW moment shape/dtype/finite guard failed')
   require(bool((value['exp_avg_sq']>=0).all()),'AdamW second moment is negative');moments+=2
 require(all_ids==list(range(len(all_ids))) and set(all_ids)==set(optimizer['state']),'AdamW state coverage or parameter registration order differs')
 return dict(identity=identity,checkpoint_sha256=sha(folder/'checkpoint.pt'),training_receipt_sha256=sha(folder/'training_complete.json'),initial_state_sha256=saved['initial'],final_state_sha256=state_hash(saved['model']),best_state_sha256=state_hash(saved['best_state']),best_epoch=best_epoch,epochs=epochs,optimizer_updates=total_steps,optimizer_parameter_tensors=len(all_ids),moment_tensors_verified=moments,trainable_parameters=trainable,total_model_parameters=total,initial_head_sha256=state_hash({n:p for n,p in template.items() if n.startswith('head.')}))

def inventory(run,protocol,selection):
 require(selection['protocol']==protocol['fingerprint'] and selection['confirmation_scored'] is False and set(selection['selection'])=={n+'__'+o for n in NAMES for o in OBJECTIVES},'Frozen exact six-arm selection required')
 gate=load(run/'final_training_complete.json');expected={n+'__'+o+'/'+str(s) for n in NAMES for o in OBJECTIVES for s in SEEDS}
 require(gate['protocol']==protocol['fingerprint'] and gate['selection_sha256']==sha(run/'selection.json') and gate['models']==60 and gate['epochs']==15 and set(gate['training_receipt_sha256'])==expected,'Exact all-model training gate required')
 cells=[]
 for name,obj in itertools.product(NAMES,OBJECTIVES):
  arm=name+'__'+obj;lrs=[1e-5,3e-5] if name=='deit_last2' else [.001,.003]
  require(selection['selection'][arm]['lr'] in lrs,'Selected LR outside prespecified grid')
  for seed,lr,folder in [(901,lr,run/'tuning'/arm/str(lr)) for lr in lrs]+[(s,selection['selection'][arm]['lr'],run/'final'/arm/str(s)) for s in SEEDS]:
   require((folder/'training_complete.json').exists() and (folder/'checkpoint.pt').exists() and (folder/'history.json').exists(),'All 72 completed actual checkpoints are required')
   if seed!=901:require(sha(folder/'training_complete.json')==gate['training_receipt_sha256'][arm+'/'+str(seed)],'Training gate receipt bytes differ')
   cells.append((dict(protocol=protocol['fingerprint'],name=name,objective=obj,seed=seed,lr=lr),folder))
 require(len(cells)==72,'Actual-state audit requires 72 models');return cells,gate

def main(args):
 torch.set_num_threads(args.threads);torch.set_num_interop_threads(1)
 run=Path(args.run);cache=Path(args.cache);protocol=load(run/'protocol.json');selection=load(run/'selection.json')
 require(fp({k:v for k,v in protocol.items() if k!='fingerprint'})==protocol['fingerprint'],'Protocol fingerprint differs')
 require(sha(args.source)==protocol['source_sha256'] and sha(args.source_helper)==protocol['source_helper_sha256'],'Frozen scientific source/helper file differs')
 require(sha(args.partition)==protocol['partition_sha256'],'Partition bytes differ')
 require(protocol['settings']['epochs']==15 and protocol['settings']['batch_size']==128 and protocol['settings']['seeds']==list(SEEDS),'Actual-state audit is restricted to the frozen production budget')
 require(all(importlib.metadata.version(k)==v for k,v in protocol['packages'].items()),'Frozen package versions differ')
 cells,gate=inventory(run,protocol,selection) # Complete metadata gate BEFORE model states are read.
 manifest=load(cache/'manifest.json');require(sha(cache/'manifest.json')==protocol['deit_cache_manifest_sha256'],'Fit feature manifest differs')
 for file in ('features.npy','labels.npy'):require(sha(cache/file)==manifest['artifacts'][file],'Fit feature/label cache hash differs')
 with np.load(args.partition,allow_pickle=False) as z:fit=z['fit'].copy()
 require(len(fit)==15268 and len(np.unique(fit))==15268,'Fit IDs differ')
 features=np.load(cache/'features.npy',mmap_mode='r',allow_pickle=False);f=np.asarray(features[fit],dtype=np.float64)
 mean=f.mean(0).astype(np.float32);sd=f.std(0).astype(np.float32);sd[sd<1e-8]=1;del f
 weight=Path(args.deit_weights);require(sha(weight)==protocol['deit_weight_sha256'],'Official DeiT weight bytes differ')
 import timm
 backbone=timm.create_model('deit_tiny_patch16_224',pretrained=False);backbone.load_state_dict(torch.load(weight,map_location='cpu',weights_only=True)['model'],strict=True)
 records=[]
 for identity,folder in cells:
  model=ExpectedModel(identity['name'],identity['seed'],backbone,(mean,sd))
  records.append(verify_record(folder,identity,model,15,120));del model
  print(json.dumps(dict(stage='actual_state_verified',arm=identity['name']+'__'+identity['objective'],seed=identity['seed'],lr=identity['lr'])),flush=True)
 # Compare saved LR selection to the actual complete tuning receipts.
 for name,obj in itertools.product(NAMES,OBJECTIVES):
  arm=name+'__'+obj;candidate=[]
  for lr in ([1e-5,3e-5] if name=='deit_last2' else [.001,.003]):
   folder=run/'tuning'/arm/str(lr);r=load(folder/'training_complete.json');candidate.append(dict(lr=lr,score=r['best_tune'],training_sha256=sha(folder/'training_complete.json')))
  require(sorted(candidate,key=lambda r:(-r['score'],r['lr']))[0]==selection['selection'][arm],'Actual complete tuning receipt winner differs')
 for seed in (901,*SEEDS):
  heads={r['initial_head_sha256'] for r in records if r['identity']['seed']==seed and r['identity']['name'] in ('deit_frozen','deit_last2')}
  require(len(heads)==1,'DeiT frozen/partial paired initial classifier heads differ')
 report=dict(status='independently_verified_complete_checkpoint_states',protocol=protocol['fingerprint'],source_sha256=sha(args.source),source_helper_sha256=sha(args.source_helper),selection_sha256=sha(run/'selection.json'),final_training_gate_sha256=sha(run/'final_training_complete.json'),partition_sha256=sha(args.partition),deit_weights_sha256=sha(weight),models_verified=72,checkpoint_states_verified=72,tuning_models_verified=12,final_models_verified=60,actual_optimizer_steps_per_parameter_verified=1800,epochs_per_model_verified=15,moment_tensors_verified=sum(r['moment_tensors_verified'] for r in records),records=records,confirmation_prediction_files_read=0,scientific_validator_functions_used=0,auditor_sha256=sha(__file__),verified_utc=datetime.now(timezone.utc).isoformat(),limitations='Does not independently rerun optimizer updates or model inference. Selected best-state values before the final epoch cannot be reconstructed from absent historical checkpoints; eligibility, finite tensors, shapes, hashes and selection metadata are verified. This receipt audits local actual states; portable numeric archives may omit every checkpoint.')
 Path(args.receipt).write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='records'}))

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('run','cache','partition','source','source-helper','deit-weights','receipt'):p.add_argument('--'+name,required=True)
 p.add_argument('--threads',type=int,default=1);main(p.parse_args())
