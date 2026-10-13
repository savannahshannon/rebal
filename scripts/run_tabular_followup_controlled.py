#!/usr/bin/env python3
"""Fit selected tabular models early; hold calibration/test behind global freeze."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,math,time
import numpy as np
import torch

def read(p):return json.loads(Path(p).read_text())
def now():return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
def module(script):
 s=importlib.util.spec_from_file_location('frozen_tabular_controller',script);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def family_freeze(m,args,family,p):
 root=args.out;trials=[]
 for i,config in enumerate(p['grids'][family]):
  trial=read(root/'tuning'/family/f'config_{i}'/'trial.json')
  if trial['protocol']!=p['fingerprint'] or trial['config']!=config or trial['index']!=i or trial['family']!=family:raise ValueError('Familytrialidentitymismatch')
  training=trial['training']
  if family=='xgboost':
   if training['completed_rounds']!=config['n_estimators'] or training['tree_count']!=7*config['n_estimators']:raise ValueError('Familytreebudgetincomplete')
  else:
   batch=p['ft_batch_size'] if family=='ft_transformer' else p['mlp_batch_size'];fit_count=sum(read(root/'data/metadata.json')['split_class_counts']['fit']);expected=p['neural_epochs']*math.ceil(fit_count/batch)
   if training['completed_epochs']!=p['neural_epochs'] or training['optimizer_updates']!=expected or not 1<=training['best_epoch']<=p['neural_epochs']:raise ValueError('Familyneuralbudgetincomplete')
  trials.append(trial)
 best=max(trials,key=lambda r:(r['training']['tune_macro_f1'],-r['index']));selected=dict(index=best['index'],config=best['config'],tuning_macro_f1=best['training']['tune_macro_f1'])
 record=dict(protocol=p['fingerprint'],family=family,trials_sha256=[m.sha(root/'tuning'/family/f'config_{i}'/'trial.json') for i in range(len(trials))],selected=selected,selection_uses_only='tuning_raw_macro_f1',confirmation_access_before_family_freeze=False)
 record['fingerprint']=m.fingerprint(record);target=root/'family_selection'/f'{family}.json'
 if target.exists():
  if read(target)!=record:raise ValueError('Immutablefamilyselectionchanged')
 else:m.write_json(target,record)
 return record

def load_pending(m,target,p,family,seed,selection):
 record=read(target/'trained_pending_confirmation.json')
 if record['protocol']!=p['fingerprint'] or record['family']!=family or record['seed']!=seed or record['family_selection']!=selection['fingerprint'] or record['config']!=selection['selected']['config']:raise ValueError('Earlyfitidentitymismatch')
 for name,digest in record['artifacts'].items():
  if m.sha(target/name)!=digest:raise ValueError('Earlyfitartifactmismatch:'+name)
 return record

def global_gate(m,args,meta):
 root=args.out
 if not (root/'frozen_selection.json').exists() or not (root/'tuning_capture_manifest.json').exists():return None
 p,freeze=m.verify_freeze(args,meta);capture=read(root/'tuning_capture_manifest.json')
 if capture['protocol']!=p['fingerprint'] or capture['freeze']!=freeze['fingerprint'] or not capture['selection_verified'] or len(capture['trials'])!=8:raise ValueError('Globalcapturegatemismatch')
 return p,freeze

def evaluate_one(m,args,arrays,y,parts,meta,p,freeze,family,seed,selection):
 target=args.out/'final'/f'seed_{seed}'/family;path=target/'metrics.json'
 if path.exists():
  r=read(path)
  if r['protocol']!=p['fingerprint'] or r['freeze']!=freeze['fingerprint'] or r['family']!=family or r['seed']!=seed or m.sha(target/'predictions.npz')!=r['predictions_sha256']:raise ValueError('Existingfinalreadoutmismatch')
  return False
 if freeze['selected'][family]!=selection['selected']:raise ValueError('Familyselectiondiffersfromglobalfreeze')
 pending=load_pending(m,target,p,family,seed,selection);detail=pending['training'];config=selection['selected']['config']
 if family=='xgboost':
  import xgboost as xgb
  model=xgb.Booster();model.load_model(target/'model.ubj')
  if model.num_boosted_rounds()!=config['n_estimators'] or detail['completed_rounds']!=config['n_estimators']:raise ValueError('Existingtreebudgetmismatch')
  cal,calruntime=m.tree_logits(model,arrays,parts['calibration'],detail['selected_rounds'],args)
 else:
  model=(m.FTTransformer if family=='ft_transformer' else m.MLP)(classes=7,**config)
  checkpoint=torch.load(target/'best.pt',map_location='cpu',weights_only=True);model.load_state_dict(checkpoint['model']);model.to(args.ft_device if family=='ft_transformer' else 'cpu')
  if checkpoint['epoch']!=detail['best_epoch'] or checkpoint['identity']!=detail['identity'] or m.state_hash(model)!=detail['selected_state_sha256']:raise ValueError('Selectedneuralcheckpointmismatch')
  cal,calruntime=m.logits_for(model,arrays,parts['calibration'],family)
 # Compute and persist the complete calibration decision before confirmation inference.
 _,la=m.adjust(cal,y[parts['calibration']],np.empty((0,7),dtype=np.float32),detail['la_prior'])
 receipt=dict(protocol=p['fingerprint'],freeze=freeze['fingerprint'],family=family,seed=seed,calibration=la,
  calibration_logits_array_sha256=hashlib.sha256(cal.tobytes()).hexdigest(),calibration_ids_array_sha256=hashlib.sha256(parts['calibration'].tobytes()).hexdigest(),
  persisted_before_confirmation_inference=True,created_utc=now());receipt['fingerprint']=m.fingerprint({k:v for k,v in receipt.items() if k!='created_utc'})
 m.write_json(target/'calibration_choice.json',receipt)
 if family=='xgboost':conf,confruntime=m.tree_logits(model,arrays,parts['confirmation'],detail['selected_rounds'],args)
 else:conf,confruntime=m.logits_for(model,arrays,parts['confirmation'],family)
 adj,replayed=m.adjust(cal,y[parts['calibration']],conf,detail['la_prior'])
 if replayed!=la:raise ValueError('Frozenroundtauchanged')
 resultmetrics={'raw':m.metrics(y[parts['confirmation']],conf.argmax(1),meta['class_rank']),'la':m.metrics(y[parts['confirmation']],adj.argmax(1),meta['class_rank'])}
 m.savez(target/'predictions.npz',calibration_ids=parts['calibration'],confirmation_ids=parts['confirmation'],calibration_labels=y[parts['calibration']],confirmation_labels=y[parts['confirmation']],calibration_logits=cal,confirmation_logits=conf)
 result=dict(seed=seed,family=family,protocol=p['fingerprint'],freeze=freeze['fingerprint'],training=detail,calibration=la,metrics=resultmetrics,
  inference=dict(calibration=calruntime,confirmation=confruntime),predictions_sha256=m.sha(target/'predictions.npz'),
  family_selection=selection['fingerprint'],early_fitting_receipt_sha256=m.sha(target/'trained_pending_confirmation.json'),calibration_choice_receipt_sha256=m.sha(target/'calibration_choice.json'))
 m.write_json(path,result);print(json.dumps(dict(stage='confirmation_complete',family=family,seed=seed,readouts=2)),flush=True);return True

def evaluate_pending(m,args,arrays,y,parts,meta,selections):
 gate=global_gate(m,args,meta)
 if gate is None:return 0
 p,freeze=gate;count=0
 for seed in args.seeds:
  for family in args.families:
   if (args.out/'final'/f'seed_{seed}'/family/'trained_pending_confirmation.json').exists():count+=evaluate_one(m,args,arrays,y,parts,meta,p,freeze,family,seed,selections[family])
 return count

def main():
 a=argparse.ArgumentParser();a.add_argument('phase',choices=['prefit','evaluate']);a.add_argument('--script',type=Path,required=True);a.add_argument('--out',type=Path,required=True);a.add_argument('--families',nargs='+',required=True);a.add_argument('--seeds',type=int,nargs='+',default=list(range(10,20)));a.add_argument('--auto-evaluate',action='store_true');args=a.parse_args()
 m=module(args.script);p=read(args.out/'protocol.json')
 if p['source_sha256']!=m.sha(args.script) or m.fingerprint({k:v for k,v in p.items() if k!='fingerprint'})!=p['fingerprint']:raise ValueError('Frozentrainingsourcechanged')
 if not set(args.families)<=set(m.FAMILIES) or not set(args.seeds)<=set(p['seeds']):raise ValueError('Unregisteredfamilyorseed')
 args.command='run';args.epochs=p['neural_epochs'];args.mlp_batch_size=p['mlp_batch_size'];args.ft_batch_size=p['ft_batch_size'];args.threads=p['threads'];args.ft_device=p['ft_device'];torch.set_num_threads(args.threads);torch.set_num_interop_threads(1)
 arrays,y,parts,meta=m.load_data(args.out);selections={f:family_freeze(m,args,f,p) for f in args.families}
 if args.phase=='prefit':
  for seed in args.seeds:
   for family in args.families:
    target=args.out/'final'/f'seed_{seed}'/family;target.mkdir(parents=True,exist_ok=True);selection=selections[family];config=selection['selected']['config'];marker=target/'trained_pending_confirmation.json'
    if marker.exists():load_pending(m,target,p,family,seed,selection)
    else:
     if family=='xgboost':_,detail=m.train_tree(arrays,y,parts,config,seed,args,target);names=['model.ubj','training.json']
     else:_,detail=m.train_neural(arrays,y,parts,family,config,seed,args,target);names=['last.pt','best.pt','history.json']
     record=dict(protocol=p['fingerprint'],family=family,seed=seed,config=config,family_selection=selection['fingerprint'],training=detail,artifacts={n:m.sha(target/n) for n in names},
      fit_finished_utc=now(),calibration_or_confirmation_evaluated_during_training=False,global_configuration_frozen_at_fit_completion=(args.out/'frozen_selection.json').exists(),controller_source_sha256=m.sha(__file__))
     m.write_json(marker,record);print(json.dumps(dict(stage='fit_complete_pending_confirmation',family=family,seed=seed,global_configuration_frozen_at_fit_completion=record['global_configuration_frozen_at_fit_completion'])),flush=True)
    if args.auto_evaluate:evaluate_pending(m,args,arrays,y,parts,meta,selections)
 if args.phase=='evaluate' or args.auto_evaluate:evaluate_pending(m,args,arrays,y,parts,meta,selections)
 status=dict(status='prefit_completed' if args.phase=='prefit' else 'evaluation_completed',families=args.families,seeds=args.seeds,fit_models_complete=len(list(args.out.glob('final/seed_*/*/trained_pending_confirmation.json'))),
  final_models_scored=len(list(args.out.glob('final/seed_*/*/metrics.json'))),expected_final_fits=40,global_configuration_frozen=(args.out/'frozen_selection.json').exists(),controller_source_sha256=m.sha(__file__))
 m.write_json(args.out/f'controller_{args.phase}_{"_".join(args.families)}.json',status);print(json.dumps(status),flush=True)
if __name__=='__main__':main()
