#!/usr/bin/env python3
"""Integrity and selection overlay for the frozen tabular runner.

Capture performs tuning-checkpoint inference before final evaluation. Verify
checks source, labels, splits, scaler, selection, records and training budgets,
then delegates numeric self-replay to the frozen scientific runner. When local
model weights are present, it also checks their digests and trusted checkpoint
states. Weight-free portable replay needs no model inference and reports zero
present weight artifacts. The separate final-prediction and statistical helpers
provide independent formula replay without scientific-runner imports.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,math
import numpy as np
from sklearn.metrics import f1_score

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def fp(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def check_digest(record):
 if fp({k:v for k,v in record.items() if k!='fingerprint'})!=record['fingerprint']:raise ValueError('Canonicalfingerprintmismatch')
def inspect(root,script):
 meta=read(root/'data/metadata.json');protocol=read(root/'protocol.json');freeze=read(root/'frozen_selection.json')
 for record in [meta,protocol,freeze]:check_digest(record)
 if meta['fingerprint']!=protocol['data_fingerprint'] or freeze['protocol']!=protocol['fingerprint'] or sha(script)!=protocol['source_sha256'] or freeze['source_sha256']!=sha(script):raise ValueError('Source/protocolbindingmismatch')
 for name in ['labels.npy','split.npz','scaler.npz']:
  if sha(root/'data'/name)!=meta['files'][name]:raise ValueError('Inputhashmismatch:'+name)
 labels=np.load(root/'data/labels.npy',allow_pickle=False);parts=dict(np.load(root/'data/split.npz',allow_pickle=False));counts=np.bincount(labels[parts['fit']],minlength=7)
 if counts.tolist()!=meta['split_class_counts']['fit']:raise ValueError('Fitclasscountmismatch')
 if meta['class_rank']!=np.argsort(-counts,kind='stable').tolist():raise ValueError('Fitfrequencyclassrankmismatch')
 selected={}
 for family in protocol['grids']:
  trials=[r for r in freeze['trials'] if r['family']==family]
  if len(trials)!=2 or [r['index'] for r in trials]!=[0,1]:raise ValueError('Tuninginventorymismatch')
  for trial in trials:
   if trial['config']!=protocol['grids'][family][trial['index']] or trial['protocol']!=protocol['fingerprint']:raise ValueError('Trialbindingmismatch')
   training=trial['training']
   if family=='xgboost':
    if training['completed_rounds']!=trial['config']['n_estimators'] or training['tree_count']!=7*trial['config']['n_estimators']:raise ValueError('Tuningtreebudgetmismatch')
   else:
    batch=protocol['ft_batch_size'] if family=='ft_transformer' else protocol['mlp_batch_size'];expected_updates=protocol['neural_epochs']*math.ceil(len(parts['fit'])/batch)
    if training['completed_epochs']!=protocol['neural_epochs'] or training['optimizer_updates']!=expected_updates or not 1<=training['best_epoch']<=protocol['neural_epochs'] or training['identity']['config']!=trial['config'] or training['identity']['source_sha256']!=sha(script):raise ValueError('Tuningneuralbudgetmismatch')
  best=max(trials,key=lambda r:(r['training']['tune_macro_f1'],-r['index']))
  expected=dict(index=best['index'],config=best['config'],tuning_macro_f1=best['training']['tune_macro_f1'])
  if freeze['selected'][family]!=expected:raise ValueError('Selectedconfigurationmismatch')
 return meta,protocol,freeze,labels,parts,counts

def import_runner(script):
 s=importlib.util.spec_from_file_location('frozen_tabular',script);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def capture(root,script):
 meta,p,freeze,y,parts,counts=inspect(root,script);m=import_runner(script);import torch
 torch.set_num_threads(p['threads']);records=[]
 args=argparse.Namespace(threads=p['threads'])
 arrays,_,_,_=m.load_data(root)
 for trial in freeze['trials']:
  family=trial['family'];index=trial['index'];target=root/'tuning'/family/f'config_{index}';config=trial['config']
  if family=='xgboost':
   import xgboost as xgb
   model=xgb.Booster();model.load_model(target/'model.ubj')
   actual_config=json.loads(model.save_config())
   if model.num_boosted_rounds()!=trial['config']['n_estimators'] or model.num_features()!=54 or int(actual_config['learner']['gradient_booster']['gbtree_model_param']['num_trees'])!=7*trial['config']['n_estimators']:raise ValueError('Actualtuningtreebudgetmismatch')
   logits,runtime=m.tree_logits(model,arrays,parts['tune'],trial['training']['selected_rounds'],args);weight_hash=sha(target/'model.ubj')
  else:
   model=(m.FTTransformer if family=='ft_transformer' else m.MLP)(classes=7,**config)
   checkpoint=torch.load(target/'best.pt',map_location='cpu',weights_only=True);model.load_state_dict(checkpoint['model']);model.to(p['ft_device'] if family=='ft_transformer' else 'cpu')
   logits,runtime=m.logits_for(model,arrays,parts['tune'],family);weight_hash=sha(target/'best.pt')
  score=float(f1_score(y[parts['tune']],logits.argmax(1),average='macro',labels=np.arange(7),zero_division=0))
  if abs(score-trial['training']['tune_macro_f1'])>1e-12:raise ValueError('Tuningcheckpointscoremismatch:'+family)
  m.savez(target/'tuning_logits.npz',row_ids=parts['tune'],labels=y[parts['tune']],logits=logits)
  record=dict(family=family,index=index,protocol=p['fingerprint'],freeze=freeze['fingerprint'],tune_macro_f1=score,weights_sha256=weight_hash,tuning_logits_sha256=sha(target/'tuning_logits.npz'),runtime=runtime)
  m.write_json(target/'tuning_capture.json',record);records.append(record);print(json.dumps(record),flush=True)
 m.write_json(root/'tuning_capture_manifest.json',dict(protocol=p['fingerprint'],freeze=freeze['fingerprint'],trials=records,selection_verified=True));return dict(status='tuning_capture_complete',trials=len(records))

def verify(root,script):
 meta,p,freeze,y,parts,counts=inspect(root,script);m=import_runner(script);trials_verified=0;budgets_verified=0;weights_verified=0;checkpoints_verified=0;calibration_receipts_verified=0
 for trial in freeze['trials']:
  family=trial['family'];index=trial['index'];target=root/'tuning'/family/f'config_{index}';record=read(target/'tuning_capture.json');path=target/'tuning_logits.npz'
  if sha(path)!=record['tuning_logits_sha256'] or record['freeze']!=freeze['fingerprint'] or record['protocol']!=p['fingerprint']:raise ValueError('Tuningpredictionbindingmismatch')
  weight_path=target/('model.ubj' if family=='xgboost' else 'best.pt')
  if weight_path.exists():
   if sha(weight_path)!=record['weights_sha256']:raise ValueError('Capturedweightdigestmismatch')
   weights_verified+=1
  with np.load(path,allow_pickle=False) as z:
   if not np.array_equal(z['row_ids'],parts['tune']) or not np.array_equal(z['labels'],y[parts['tune']]) or z['logits'].shape!=(len(parts['tune']),7) or not np.isfinite(z['logits']).all():raise ValueError('Tuningpredictionalignmentmismatch')
   value=float(f1_score(z['labels'],z['logits'].argmax(1),average='macro',labels=np.arange(7),zero_division=0))
   if abs(value-trial['training']['tune_macro_f1'])>1e-12:raise ValueError('Independentconfigurationselectionmismatch')
  trials_verified+=1
 for seed in p['seeds']:
  for family in m.FAMILIES:
   target=root/'final'/f'seed_{seed}'/family;r=read(target/'metrics.json');train=r['training'];selected=freeze['selected'][family]['config']
   if r['protocol']!=p['fingerprint'] or r['freeze']!=freeze['fingerprint'] or r['seed']!=seed or r['family']!=family:raise ValueError('Finalrecordidentitymismatch')
   if (target/'calibration_choice.json').exists():
    receipt=read(target/'calibration_choice.json')
    if sha(target/'calibration_choice.json')!=r['calibration_choice_receipt_sha256'] or fp({k:v for k,v in receipt.items() if k not in ['fingerprint','created_utc']})!=receipt['fingerprint'] or receipt['calibration']!=r['calibration'] or receipt['protocol']!=p['fingerprint'] or receipt['freeze']!=freeze['fingerprint'] or receipt['family']!=family or receipt['seed']!=seed or not receipt['persisted_before_confirmation_inference']:raise ValueError('Calibrationchoicereceiptmismatch')
    with np.load(target/'predictions.npz',allow_pickle=False) as z:
     if hashlib.sha256(z['calibration_logits'].tobytes()).hexdigest()!=receipt['calibration_logits_array_sha256'] or hashlib.sha256(z['calibration_ids'].tobytes()).hexdigest()!=receipt['calibration_ids_array_sha256']:raise ValueError('Calibrationreceiptdatasourcemismatch')
    calibration_receipts_verified+=1
   if family=='xgboost':
    expected={k:v for k,v in selected.items() if k!='n_estimators'}
    if any(train['params'][k]!=v for k,v in expected.items()) or train['seed']!=seed or train['completed_rounds']!=selected['n_estimators'] or train['tree_count']!=7*selected['n_estimators']:raise ValueError('Treeconfigurationbudgetmismatch')
    prior=counts/counts.sum();curve=train['selection_curve'];best=max(curve,key=lambda a:(a['tune_macro_f1'],-a['rounds']))
    if train['selected_rounds']!=best['rounds']:raise ValueError('Treeiterationselectionmismatch')
    if (target/'model.ubj').exists():
     if sha(target/'model.ubj')!=train['model_sha256']:raise ValueError('Finaltreemodeldigestmismatch')
     import xgboost as xgb
     actual=xgb.Booster();actual.load_model(target/'model.ubj');actual_config=json.loads(actual.save_config())
     if actual.num_boosted_rounds()!=selected['n_estimators'] or actual.num_features()!=54 or int(actual_config['learner']['gradient_booster']['gbtree_model_param']['num_trees'])!=7*selected['n_estimators']:raise ValueError('Actualfinaltreebudgetmismatch')
     weights_verified+=1
   else:
    identity=train['identity'];batch=p['ft_batch_size'] if family=='ft_transformer' else p['mlp_batch_size'];expected_steps=p['neural_epochs']*math.ceil(len(parts['fit'])/batch)
    if identity['config']!=selected or identity['seed']!=seed or identity['source_sha256']!=sha(script) or identity['fit_hash']!=hashlib.sha256(parts['fit'].tobytes()).hexdigest() or identity['batch']!=batch or train['optimizer_updates']!=expected_steps or train['completed_epochs']!=p['neural_epochs']:raise ValueError('Neuralconfigurationbudgetmismatch')
    weights=np.ones(7,dtype=np.float32)
    if family=='mlp_cb':
     beta=1-1/counts.max();effective=-np.expm1(counts*np.log1p(-1/counts.max()));w=(1-beta)/effective;weights=(w/(counts@w/counts.sum())).astype(np.float32)
    if not np.array_equal(weights,np.asarray(train['class_weights'],dtype=np.float32)):raise ValueError('Classweightmismatch')
    prior=counts*weights/(counts@weights)
    history=read(target/'history.json');best=max(history,key=lambda a:(a['tune_macro_f1'],-a['epoch']))
    if len(history)!=p['neural_epochs'] or best['epoch']!=train['best_epoch'] or best['tune_macro_f1']!=train['tune_macro_f1']:raise ValueError('Neuralcheckpointselectionmismatch')
    for epoch,h in enumerate(history,1):
     if h['epoch']!=epoch or h['optimizer_updates']!=epoch*math.ceil(len(parts['fit'])/batch) or not math.isfinite(h['loss']) or not math.isfinite(h['tune_macro_f1']):raise ValueError('Historybudgetorrangemismatch')
    if (target/'last.pt').exists() and (target/'best.pt').exists():
     import torch
     last=torch.load(target/'last.pt',map_location='cpu',weights_only=True);chosen=torch.load(target/'best.pt',map_location='cpu',weights_only=True)
     if last['identity']!=identity or chosen['identity']!=identity or last['completed_epochs']!=p['neural_epochs'] or last['history']!=history or chosen['epoch']!=train['best_epoch'] or chosen['tune_macro_f1']!=train['tune_macro_f1'] or last['initial_sha256']!=train['initial_sha256']:raise ValueError('Trustedcheckpointidentitybudgetmismatch')
     for state in last['optimizer']['state'].values():
      if float(state['step'])!=expected_steps:raise ValueError('AdamWstateupdatecountmismatch')
     model=(m.FTTransformer if family=='ft_transformer' else m.MLP)(classes=7,**selected);model.load_state_dict(chosen['model'])
     if m.state_hash(model)!=train['selected_state_sha256']:raise ValueError('Selectedneuralstatehashmismatch')
     if any(not torch.isfinite(t).all() for t in last['model'].values()):raise ValueError('Nonfinitecheckpointstate')
     weights_verified+=2;checkpoints_verified+=1
   if not np.allclose(prior,train['la_prior'],rtol=0,atol=1e-14) or not np.allclose(prior,r['calibration']['prior'],rtol=0,atol=1e-14):raise ValueError('FitonlylogitpriorMismatch')
   budgets_verified+=1
 args=argparse.Namespace(command='verify',out=root,epochs=p['neural_epochs'],mlp_batch_size=p['mlp_batch_size'],ft_batch_size=p['ft_batch_size'],threads=p['threads'],ft_device=p['ft_device'],families=m.FAMILIES,seeds=p['seeds'])
 replay=m.verify(args)
 result=dict(status='independent_integrity_selection_and_numeric_replay_passed',canonical_source_sha256=meta['source_gzip_sha256'],protocol=p['fingerprint'],freeze=freeze['fingerprint'],tuning_trials_verified=trials_verified,final_fit_configs_and_budgets_verified=budgets_verified,model_weight_artifacts_present_verified=weights_verified,neural_checkpoint_states_present_verified=checkpoints_verified,calibration_choices_persisted_before_confirmation_verified=calibration_receipts_verified,model_weight_scope='When present, trustedownweights_only states andactualweightfiledigests checked; portablebundlesmayomitweights,withzeroorlowerpresencecounts. Numericreplayneverrequiresweights.',numeric_replay=replay,overlay_source_sha256=sha(__file__))
 m.write_json(root/'portable_verification.json',result);return result

def main():
 a=argparse.ArgumentParser();a.add_argument('command',choices=['capture','verify']);a.add_argument('--run',type=Path,required=True);a.add_argument('--script',type=Path,required=True);args=a.parse_args()
 print(json.dumps(globals()[args.command](args.run,args.script),indent=2,sort_keys=True))
if __name__=='__main__':main()
