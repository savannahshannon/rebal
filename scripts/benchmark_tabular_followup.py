#!/usr/bin/env python3
"""Fresh-split Covertype: MLP, weighted MLP, FT-Transformer and XGBoost.

Stages enforce fit/tune/calibration/confirmation separation. `run` requires the
configuration frozen by `tune`; final predictions never select a checkpoint.
"""
from __future__ import annotations
import argparse, copy, gzip, hashlib, importlib.metadata, itertools, json, math
import os, platform, sys, time
from pathlib import Path
import numpy as np
from scipy import stats
import torch
from torch import nn
from torch.nn import functional as F

VERSION = 1
SPLIT_SEED = 314159
SEEDS = list(range(10, 20))
TAUS = [0., .25, .5, .75, 1., 1.25, 1.5, 2.]
FAMILIES = ['mlp_ce', 'mlp_cb', 'ft_transformer', 'xgboost']
GRIDS = {
 'mlp_ce': [dict(lr=.001, width=256, dropout=.1, weight_decay=.0001), dict(lr=.003, width=256, dropout=.1, weight_decay=.0001)],
 'mlp_cb': [dict(lr=.001, width=256, dropout=.1, weight_decay=.0001), dict(lr=.003, width=256, dropout=.1, weight_decay=.0001)],
 'ft_transformer': [dict(lr=.0003, width=64, blocks=2, heads=4, dropout=.1, weight_decay=.0001), dict(lr=.0003, width=96, blocks=2, heads=4, dropout=.1, weight_decay=.0001)],
 'xgboost': [dict(max_depth=6, learning_rate=.08, subsample=.8, colsample_bytree=.9, reg_lambda=1., min_child_weight=1., n_estimators=600), dict(max_depth=10, learning_rate=.04, subsample=.8, colsample_bytree=.9, reg_lambda=1., min_child_weight=1., n_estimators=600)],
}

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''): h.update(b)
 return h.hexdigest()
def fingerprint(x): return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def write_json(path,x):
 path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_suffix(path.suffix+'.partial')
 tmp.write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n'); tmp.replace(path)
def savez(path,**values):
 path=Path(path); temp=path.with_suffix('.partial.npz'); np.savez_compressed(temp,**values); temp.replace(path)
def versions():
 out={n:importlib.metadata.version(n) for n in ['numpy','scipy','torch','scikit-learn']}
 try: out['xgboost']=importlib.metadata.version('xgboost')
 except importlib.metadata.PackageNotFoundError: pass
 return dict(packages=out,python=platform.python_version(),platform=platform.platform())
def seed_all(seed): np.random.seed(seed); torch.manual_seed(seed)
def state_hash(model):
 h=hashlib.sha256()
 for k,v in sorted(model.state_dict().items()): h.update(k.encode()); h.update(v.detach().cpu().numpy().tobytes())
 return h.hexdigest()

def stratified_split(labels,seed=SPLIT_SEED):
 labels=np.asarray(labels); rng=np.random.default_rng(seed); parts={k:[] for k in ['fit','tune','calibration','confirmation']}
 for c in np.unique(labels):
  ids=rng.permutation(np.flatnonzero(labels==c)); n=len(ids)
  counts=[int(.7*n),int(.1*n),int(.1*n)]; edges=np.cumsum([0]+counts+[n-sum(counts)])
  for j,k in enumerate(parts): parts[k].extend(ids[edges[j]:edges[j+1]])
 out={k:np.sort(np.asarray(v,dtype=np.int64)) for k,v in parts.items()}; validate_split(out,labels); return out

def validate_split(parts,labels):
 names=['fit','tune','calibration','confirmation']; n=len(labels); classes=np.unique(labels)
 if set(parts)!=set(names): raise ValueError('Split must have exactly four named parts')
 for k in names:
  a=parts[k]
  if a.ndim!=1 or a.dtype.kind not in 'iu' or not len(a) or a.min()<0 or a.max()>=n: raise ValueError('Invalid split index')
  if len(np.unique(a))!=len(a): raise ValueError('Repeated split index')
  if not np.array_equal(np.unique(labels[a]),classes): raise ValueError('Every split must cover every class')
 joined=np.concatenate([parts[k] for k in names])
 if len(joined)!=n or len(np.unique(joined))!=n: raise ValueError('Split overlaps or fails to cover source')

def decode_categories(x):
 x=np.asarray(x)
 if x.ndim!=2 or x.shape[1]!=54 or not np.isfinite(x).all(): raise ValueError('Expected finite54-column Covertype')
 cats=[]
 for lo,hi in [(10,14),(14,54)]:
  onehot=x[:,lo:hi]
  if not np.isin(onehot,[0,1]).all() or not (onehot.sum(1)==1).all(): raise ValueError('Invalid categorical one-hot group')
  decoded=onehot.argmax(1).astype(np.int64)
  if not np.array_equal(np.eye(hi-lo,dtype=onehot.dtype)[decoded],onehot): raise ValueError('Lossless categorical reconstruction failed')
  cats.append(decoded)
 return np.stack(cats,axis=1)

def scaler_fit(x,fit):
 v=np.asarray(x[fit,:10],dtype=np.float64); mean=v.mean(0); scale=v.std(0); scale[scale==0]=1
 return mean,scale

def prepare(args):
 out=Path(args.out); dest=out/'data'; dest.mkdir(parents=True,exist_ok=True)
 if (dest/'metadata.json').exists(): return load_data(out)[3]
 source=Path(args.source); base=Path(args.data); old=json.loads((base/'metadata.json').read_text())
 if sha(source)!=old['source']['files'][source.name]['sha256']: raise ValueError('Canonical gzip source hash mismatch')
 for name in ['features.npy','labels.npy']:
  if sha(base/name)!=old['files_sha256'][name]: raise ValueError('Prepared source hash mismatch')
 x=np.load(base/'features.npy',mmap_mode='r'); y=np.load(base/'labels.npy')
 if x.shape!=(581012,54) or y.shape!=(581012,) or not np.array_equal(np.bincount(y),[211840,283301,35754,2747,9493,17367,20510]): raise ValueError('Canonical Covertype dimensions/counts mismatch')
 cat=decode_categories(x); parts=stratified_split(y); mean,scale=scaler_fit(x,parts['fit'])
 numeric=((np.asarray(x[:,:10],dtype=np.float64)-mean)/scale).astype(np.float32)
 np.save(dest/'numeric.npy',numeric); np.save(dest/'categorical.npy',cat); np.save(dest/'labels.npy',y)
 # MLP/tree retain all54 predictors, with standardized continuous columns.
 flat=np.asarray(x).copy().astype(np.float32); flat[:,:10]=numeric; np.save(dest/'flat.npy',flat)
 savez(dest/'split.npz',**parts); savez(dest/'scaler.npz',mean=mean,scale=scale)
 with np.load(base/'split.npz') as oldsplit: oldtest=oldsplit['test']
 meta=dict(version=VERSION,dataset='Covertype',source_rows=len(y),source_gzip_sha256=sha(source),source_url=old['source']['files'][source.name]['url'],
  source_features_sha256=sha(base/'features.npy'),source_labels_sha256=sha(base/'labels.npy'),split_seed=SPLIT_SEED,
  split_rule='Class-stratified70/10/10/10, per-class floor fit/tune/cal, confirmation receives remainder',
  split_counts={k:len(v) for k,v in parts.items()},split_class_counts={k:np.bincount(y[v],minlength=7).tolist() for k,v in parts.items()},
  scaler_fit_part='fit',categorical_rule='Lossless decode columns10:14 wilderness4 and14:54 soil40; exact one-hot reconstruction checked',
  input_features=dict(continuous=10,one_hot_binary=44,categorical_cardinalities=[4,40]),class_rank=np.argsort(-np.bincount(y),kind='stable').tolist(),
  prior_study_test_overlap_with_new_confirmation=int(len(np.intersect1d(oldtest,parts['confirmation']))),
  confirmation_limit='Fresh partition of previously studied source dataset; not an independent geographic/external dataset. Prior report exposed aggregate outcomes on other splits.',
  files={p.name:sha(p) for p in sorted(dest.iterdir()) if p.is_file()})
 meta['fingerprint']=fingerprint(meta); write_json(dest/'metadata.json',meta); return meta

def load_data(out):
 dest=Path(out)/'data'; meta=json.loads((dest/'metadata.json').read_text())
 for name,digest in meta['files'].items():
  if sha(dest/name)!=digest: raise ValueError('Prepared input hash mismatch:'+name)
 y=np.load(dest/'labels.npy'); parts=dict(np.load(dest/'split.npz')); validate_split(parts,y)
 arrays={k:np.load(dest/(k+'.npy'),mmap_mode='r') for k in ['numeric','categorical','flat']}
 return arrays,y,parts,meta

class FeatureTokenizer(nn.Module):
 def __init__(self,width):
  super().__init__(); self.numeric_weight=nn.Parameter(torch.empty(10,width)); self.numeric_bias=nn.Parameter(torch.empty(10,width))
  self.categorical=nn.ModuleList([nn.Embedding(4,width),nn.Embedding(40,width)]); self.categorical_bias=nn.Parameter(torch.empty(2,width)); self.cls=nn.Parameter(torch.empty(1,1,width))
  nn.init.normal_(self.numeric_weight,std=.02); nn.init.normal_(self.numeric_bias,std=.02); nn.init.normal_(self.categorical_bias,std=.02); nn.init.normal_(self.cls,std=.02)
  for e in self.categorical: nn.init.normal_(e.weight,std=.02)
 def forward(self,numeric,categorical):
  num=numeric.unsqueeze(-1)*self.numeric_weight+self.numeric_bias
  cat=torch.stack([e(categorical[:,i]) for i,e in enumerate(self.categorical)],dim=1)+self.categorical_bias
  return torch.cat([num,cat,self.cls.expand(len(num),-1,-1)],dim=1)
class ReGLU(nn.Module):
 def forward(self,x): a,b=x.chunk(2,dim=-1); return a*F.relu(b)
class TransformerBlock(nn.Module):
 def __init__(self,width,heads,dropout,cls_only=False):
  super().__init__(); self.cls_only=cls_only; self.norm1=nn.LayerNorm(width); self.attention=nn.MultiheadAttention(width,heads,dropout=dropout,batch_first=True)
  self.norm2=nn.LayerNorm(width); self.ffn=nn.Sequential(nn.Linear(width,4*width),ReGLU(),nn.Dropout(dropout),nn.Linear(2*width,width)); self.dropout=nn.Dropout(dropout)
 def forward(self,x):
  a=self.norm1(x); query=a[:,-1:] if self.cls_only else a; residual=x[:,-1:] if self.cls_only else x
  x=residual+self.dropout(self.attention(query,a,a,need_weights=False)[0]); return x+self.dropout(self.ffn(self.norm2(x)))
class FTTransformer(nn.Module):
 def __init__(self,classes=7,width=64,blocks=2,heads=4,dropout=.1,**unused):
  super().__init__(); self.tokenizer=FeatureTokenizer(width); self.blocks=nn.Sequential(*[TransformerBlock(width,heads,dropout,cls_only=(i==blocks-1)) for i in range(blocks)])
  self.head=nn.Sequential(nn.LayerNorm(width),nn.ReLU(),nn.Linear(width,classes))
 def forward(self,numeric,categorical): return self.head(self.blocks(self.tokenizer(numeric,categorical))[:,-1])
class MLP(nn.Module):
 def __init__(self,classes=7,width=256,dropout=.1,**unused):
  super().__init__(); self.layers=nn.Sequential(nn.Linear(54,width),nn.ReLU(),nn.Dropout(dropout),nn.Linear(width,width),nn.ReLU(),nn.Linear(width,128),nn.ReLU(),nn.Dropout(dropout),nn.Linear(128,classes))
 def forward(self,x): return self.layers(x)

def cb_weights(counts):
 counts=np.asarray(counts,dtype=np.float64); beta=1-1/counts.max(); effective=-np.expm1(counts*np.log1p(-1/counts.max())); w=(1-beta)/effective
 return (w/(np.dot(counts,w)/counts.sum())).astype(np.float32)

def logits_for(model,arrays,ids,family,batch=2048):
 model.eval(); out=[]; start=time.perf_counter(); device=next(model.parameters()).device
 with torch.inference_mode():
  for off in range(0,len(ids),batch):
   chosen=ids[off:off+batch]
   if family=='ft_transformer': l=model(torch.from_numpy(np.array(arrays['numeric'][chosen])).to(device),torch.from_numpy(np.array(arrays['categorical'][chosen])).to(device))
   else: l=model(torch.from_numpy(np.array(arrays['flat'][chosen])).to(device))
   out.append(l.cpu().numpy())
 logits=np.concatenate(out); duration=time.perf_counter()-start
 if not np.isfinite(logits).all(): raise ValueError('Nonfinite logits')
 return logits,dict(seconds=duration,rows=len(ids),batch_size=batch,rows_per_second=len(ids)/duration)

def metrics(labels,pred,rank=None):
 n=max(int(np.max(labels))+1,int(np.max(pred))+1); cm=np.bincount(np.asarray(labels)*n+pred,minlength=n*n).reshape(n,n)
 tp=np.diag(cm).astype(float); support=cm.sum(1); predicted=cm.sum(0); f1=np.divide(2*tp,support+predicted,out=np.zeros(n),where=(support+predicted)>0)
 recall=np.divide(tp,support,out=np.zeros(n),where=support>0); precision=np.divide(tp,predicted,out=np.zeros(n),where=predicted>0)
 rank=np.arange(n) if rank is None else np.asarray(rank); groups=np.array_split(rank,3)
 balanced_cm=np.divide(cm,support[:,None],out=np.zeros_like(cm,dtype=float),where=support[:,None]>0); balanced_tp=np.diag(balanced_cm); balanced_f1=np.divide(2*balanced_tp,balanced_cm.sum(1)+balanced_cm.sum(0),out=np.zeros(n),where=(balanced_cm.sum(1)+balanced_cm.sum(0))>0)
 return dict(accuracy=float(tp.sum()/cm.sum()),balanced_accuracy=float(balanced_tp.sum()/balanced_cm.sum()),balanced_macro_f1=float(balanced_f1.mean()),macro_f1=float(f1.mean()),macro_recall=float(recall.mean()),head_f1=float(f1[groups[0]].mean()),mid_f1=float(f1[groups[1]].mean()),tail_f1=float(f1[groups[2]].mean()),
  confusion_matrix=cm.tolist(),per_class_f1=f1.tolist(),per_class_recall=recall.tolist(),per_class_precision=precision.tolist(),per_class_support=support.tolist())

def adjust(cal_logits,cal_y,confirm_logits,prior):
 logprior=np.log(np.asarray(prior,dtype=np.float64)); logprior-=logprior.mean(); candidates=[]
 for tau in TAUS:
  adjusted=cal_logits-tau*logprior; candidates.append(dict(tau=tau,macro_f1=metrics(cal_y,adjusted.argmax(1))['macro_f1']))
 best=max(candidates,key=lambda r:(r['macro_f1'],-r['tau'])); tau=best['tau']
 return confirm_logits-tau*logprior,dict(tau=tau,prior=np.asarray(prior).tolist(),selection_part='calibration',curve=candidates,tie_rule='smallesttau')

def protocol(args,meta):
 p=dict(version=VERSION,data_fingerprint=meta['fingerprint'],source_sha256=sha(__file__),versions=versions(),seeds=SEEDS,tuning_seed=9,
  grids=GRIDS,neural_epochs=args.epochs,ft_device=getattr(args,'ft_device','cpu'),mlp_device='cpu',mlp_batch_size=args.mlp_batch_size,ft_batch_size=args.ft_batch_size,
  neural_optimizer='AdamW,5-epochlinearwarmup thencosine,all50epochsrun; besttune macro-F1checkpoint',
  cb_rule='Count-aware effective-number loss weights all epochs normalized expectedfitweight1',
  tree_rule='XGBoostCPUhist,max_bin256,600rounds,checkpointcomparisonat50-roundstepsbytunemacroF1',
  selection='Two predeclared configurations per family on optimizationseed9. No calibration/confirmation forward pass before configurationfreeze. Equal trial count, model-specific compute budgets disclosed.',
  posthoc_taus=TAUS,confirm_readouts=['raw','calibrated_logit_adjustment'],threads=args.threads,
  primary_contrasts=[['ft_transformer_la','mlp_ce_la'],['xgboost_la','mlp_ce_la'],['mlp_cb_la','mlp_ce_la']],
  seed_inference='10optimizationseeds on one fixeddatasetpartition; intervalcapturesoptimizationvariation,notdataset/geographicuncertainty')
 p['fingerprint']=fingerprint(p); return p

def save_model_checkpoint(path,value):
 tmp=path.with_suffix('.partial.pt'); torch.save(value,tmp); tmp.replace(path)

def train_neural(arrays,y,parts,family,config,seed,args,target,pilot_steps=0):
 target=Path(target); target.mkdir(parents=True,exist_ok=True); seed_all(seed)
 device=getattr(args,'ft_device','cpu') if family=='ft_transformer' else 'cpu'
 model=(FTTransformer if family=='ft_transformer' else MLP)(classes=7,**config).to(device)
 initial=state_hash(model); optimizer=torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])
 fit=parts['fit']; batch=args.ft_batch_size if family=='ft_transformer' else args.mlp_batch_size; steps=math.ceil(len(fit)/batch)
 counts=np.bincount(y[fit],minlength=7); weights=cb_weights(counts) if family=='mlp_cb' else np.ones(7,dtype=np.float32); cw=torch.tensor(weights,device=device)
 total=args.epochs*steps; warmup=min(5,args.epochs)*steps; start_epoch=0; history=[]; best_score=-1.; best_epoch=0; train_seconds=0.; initial_matches=True
 ckpath=target/'last.pt'
 identity=dict(family=family,config=config,seed=seed,device=device,epochs=args.epochs,batch=batch,source_sha256=sha(__file__),fit_hash=hashlib.sha256(fit.tobytes()).hexdigest())
 if ckpath.exists() and not pilot_steps:
  ck=torch.load(ckpath,map_location='cpu',weights_only=True)
  if ck['identity']!=identity or ck['initial_sha256']!=initial: raise ValueError('Resumeidentitymismatch')
  model.load_state_dict(ck['model']); optimizer.load_state_dict(ck['optimizer']); start_epoch=ck['completed_epochs']; history=ck['history']; best_score=ck['best_score']; best_epoch=ck['best_epoch']; train_seconds=ck['train_seconds']
 actual_steps=0; start=time.perf_counter()
 for epoch in range(start_epoch,args.epochs):
  model.train(); torch.manual_seed(int(np.random.SeedSequence([seed,epoch,901]).generate_state(1)[0])); rng=np.random.default_rng(np.random.SeedSequence([seed,epoch,707])); order=rng.permutation(fit)
  epoch_start=time.perf_counter(); loss_sum=0.; rows=0
  for off in range(0,len(order),batch):
   ids=order[off:off+batch]; step=epoch*steps+off//batch; lr=config['lr']*((step+1)/warmup if step<warmup else .5*(1+math.cos(math.pi*(step-warmup)/max(1,total-warmup))))
   for g in optimizer.param_groups:g['lr']=lr
   label=torch.from_numpy(np.asarray(y[ids])).to(device); optimizer.zero_grad(set_to_none=True)
   if family=='ft_transformer': logits=model(torch.from_numpy(np.array(arrays['numeric'][ids])).to(device),torch.from_numpy(np.array(arrays['categorical'][ids])).to(device))
   else: logits=model(torch.from_numpy(np.array(arrays['flat'][ids])).to(device))
   loss=(F.cross_entropy(logits,label,reduction='none')*cw[label]).mean()
   if not torch.isfinite(loss):raise ValueError('Nonfiniteneuralloss')
   loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True); optimizer.step(); actual_steps+=1
   loss_sum+=float(loss.detach())*len(ids); rows+=len(ids)
   if pilot_steps and actual_steps>=pilot_steps:
    return model,dict(family=family,steps=actual_steps,batch_size=batch,seconds=time.perf_counter()-start,rows=rows,parameters=sum(p.numel() for p in model.parameters()),initial_sha256=initial,heldout_evaluated=False)
  duration=time.perf_counter()-epoch_start; train_seconds+=duration
  tl,_=logits_for(model,arrays,parts['tune'],family); tune_score=metrics(y[parts['tune']],tl.argmax(1))['macro_f1']
  if tune_score>best_score:
   best_score=tune_score; best_epoch=epoch+1; save_model_checkpoint(target/'best.pt',dict(model=model.state_dict(),epoch=best_epoch,tune_macro_f1=best_score,identity=identity))
  history.append(dict(epoch=epoch+1,optimizer_updates=(epoch+1)*steps,loss=loss_sum/rows,tune_macro_f1=tune_score,training_seconds=duration))
  write_json(target/'history.json',history); print(json.dumps(dict(family=family,seed=seed,**history[-1])),flush=True)
  if (epoch+1)%5==0 or epoch+1==args.epochs:
   save_model_checkpoint(ckpath,dict(model=model.state_dict(),optimizer=optimizer.state_dict(),completed_epochs=epoch+1,history=history,best_score=best_score,best_epoch=best_epoch,train_seconds=train_seconds,identity=identity,initial_sha256=initial))
 ck=torch.load(target/'best.pt',weights_only=True,map_location='cpu'); model.load_state_dict(ck['model'])
 detail=dict(identity=identity,completed_epochs=args.epochs,optimizer_updates=total,best_epoch=best_epoch,tune_macro_f1=best_score,training_seconds=train_seconds,total_seconds=time.perf_counter()-start,
  parameters=sum(p.numel() for p in model.parameters()),initial_sha256=initial,selected_state_sha256=state_hash(model),class_weights=weights.tolist(),la_prior=(counts*weights/np.dot(counts,weights)).tolist())
 return model,detail

def train_tree(arrays,y,parts,config,seed,args,target,pilot_rounds=0):
 import xgboost as xgb
 target=Path(target); target.mkdir(parents=True,exist_ok=True); nround=pilot_rounds or config['n_estimators']; params={k:v for k,v in config.items() if k!='n_estimators'}
 params.update(objective='multi:softprob',num_class=7,tree_method='hist',max_bin=256,nthread=args.threads,seed=seed,eval_metric='mlogloss',device='cpu')
 start=time.perf_counter(); dtrain=xgb.DMatrix(arrays['flat'][parts['fit']],label=y[parts['fit']],nthread=args.threads); prepared=time.perf_counter()-start
 booster=xgb.train(params,dtrain,num_boost_round=nround); trainsec=time.perf_counter()-start-prepared; booster.save_model(target/'model.ubj')
 if pilot_rounds:return booster,dict(rounds=nround,training_seconds=trainsec,prepare_seconds=prepared,heldout_evaluated=False)
 dtune=xgb.DMatrix(arrays['flat'][parts['tune']],nthread=args.threads); curve=[]
 for end in range(50,nround+1,50):
  logits=booster.predict(dtune,output_margin=True,iteration_range=(0,end)); curve.append(dict(rounds=end,tune_macro_f1=metrics(y[parts['tune']],logits.argmax(1))['macro_f1']))
 best=max(curve,key=lambda r:(r['tune_macro_f1'],-r['rounds'])); counts=np.bincount(y[parts['fit']],minlength=7)
 detail=dict(training_seconds=trainsec,prepare_seconds=prepared,completed_rounds=nround,selected_rounds=best['rounds'],tune_macro_f1=best['tune_macro_f1'],selection_curve=curve,
  tree_count=7*nround,seed=seed,params=params,la_prior=(counts/counts.sum()).tolist(),model_sha256=sha(target/'model.ubj'))
 write_json(target/'training.json',detail); print(json.dumps(dict(family='xgboost',**detail)),flush=True);return booster,detail

def tree_logits(booster,arrays,ids,rounds,args):
 import xgboost as xgb
 start=time.perf_counter(); d=xgb.DMatrix(arrays['flat'][ids],nthread=args.threads); prep=time.perf_counter()-start
 logits=booster.predict(d,output_margin=True,iteration_range=(0,rounds)); elapsed=time.perf_counter()-start
 return logits,dict(seconds=elapsed,prepare_seconds=prep,rows=len(ids),rows_per_second=len(ids)/elapsed)

def pilot(args):
 arrays,y,parts,meta=load_data(args.out); p=protocol(args,meta); target=Path(args.out)/'pilot'; records=[]
 for family in getattr(args,'families',FAMILIES):
  config=GRIDS[family][0]
  if family=='xgboost':_,detail=train_tree(arrays,y,parts,config,9,args,target/family,pilot_rounds=args.pilot_rounds)
  else:_,detail=train_neural(arrays,y,parts,family,config,9,args,target/family,pilot_steps=args.pilot_steps)
  records.append({**detail,'family':family})
 report=dict(protocol=p,records=records,heldout_evaluated=False); write_json(target/'runtime.json',report);return report

def tune(args):
 arrays,y,parts,meta=load_data(args.out); p=protocol(args,meta); root=Path(args.out); trials=[]; selected={}
 if (root/'frozen_selection.json').exists():raise ValueError('Tuningalreadyfrozen;useexistingselection')
 write_json(root/'protocol.json',p)
 for family in getattr(args,'families',FAMILIES):
  subset=[]
  for i,config in enumerate(GRIDS[family]):
   target=root/'tuning'/family/f'config_{i}'; recordpath=target/'trial.json'
   if recordpath.exists():
    trial=json.loads(recordpath.read_text())
    if trial['protocol']!=p['fingerprint']:raise ValueError('Tuningprotocolmismatch')
   else:
    if family=='xgboost':_,detail=train_tree(arrays,y,parts,config,9,args,target)
    else:_,detail=train_neural(arrays,y,parts,family,config,9,args,target)
    trial=dict(family=family,index=i,config=config,training=detail,protocol=p['fingerprint'],selection_part='tune');write_json(recordpath,trial)
   subset.append(trial);trials.append(trial)
  best=max(subset,key=lambda r:(r['training']['tune_macro_f1'],-r['index']));selected[family]=dict(index=best['index'],config=best['config'],tuning_macro_f1=best['training']['tune_macro_f1'])
 # Partial family tuning can run on different available devices; global freeze
 # is created only when every predeclared trial is complete.
 trials=[];selected={}
 for family in FAMILIES:
  paths=[root/'tuning'/family/f'config_{i}'/'trial.json' for i in range(len(GRIDS[family]))]
  if not all(path.exists() for path in paths):continue
  subset=[json.loads(path.read_text()) for path in paths]
  if any(r['protocol']!=p['fingerprint'] for r in subset):raise ValueError('Tuningprotocolmismatch')
  best=max(subset,key=lambda r:(r['training']['tune_macro_f1'],-r['index']));selected[family]=dict(index=best['index'],config=best['config'],tuning_macro_f1=best['training']['tune_macro_f1']);trials.extend(subset)
 if set(selected)!=set(FAMILIES):
  partial=dict(protocol=p['fingerprint'],selected=selected,completed_tuning_trials=len(trials),configuration_frozen=False,calibration_or_confirmation_evaluated=False)
  write_json(root/'partial_tuning.json',partial);return partial
 freeze=dict(version=VERSION,protocol=p['fingerprint'],selected=selected,trials=trials,calibration_or_confirmation_evaluated=False,selection_part='tune',source_sha256=sha(__file__))
 freeze['fingerprint']=fingerprint(freeze);write_json(root/'frozen_selection.json',freeze);return freeze

def verify_freeze(args,meta):
 root=Path(args.out); p=json.loads((root/'protocol.json').read_text()); freeze=json.loads((root/'frozen_selection.json').read_text())
 canonical={k:v for k,v in p.items() if k!='fingerprint'}
 if fingerprint(canonical)!=p['fingerprint'] or p['data_fingerprint']!=meta['fingerprint'] or freeze['protocol']!=p['fingerprint'] or freeze['source_sha256']!=sha(__file__):raise ValueError('Frozenprotocolmismatch')
 if fingerprint({k:v for k,v in freeze.items() if k!='fingerprint'})!=freeze['fingerprint']:raise ValueError('Freezedigestmismatch')
 if args.command!='verify' and p!=protocol(args,meta):raise ValueError('Frozenruntimeprotocolmismatch')
 if freeze['calibration_or_confirmation_evaluated'] or set(freeze['selected'])!=set(FAMILIES):raise ValueError('Invalidfreeze')
 return p,freeze

def run(args):
 arrays,y,parts,meta=load_data(args.out);p,freeze=verify_freeze(args,meta);root=Path(args.out);results=[]
 for seed in args.seeds:
  if seed not in SEEDS:raise ValueError('Finalseedoutsidefrozengrid')
  for family in getattr(args,'families',FAMILIES):
   target=root/'final'/f'seed_{seed}'/family;target.mkdir(parents=True,exist_ok=True);path=target/'metrics.json'
   if path.exists():results.append(json.loads(path.read_text()));continue
   config=freeze['selected'][family]['config']
   if family=='xgboost':
    model,detail=train_tree(arrays,y,parts,config,seed,args,target)
    cal,calruntime=tree_logits(model,arrays,parts['calibration'],detail['selected_rounds'],args);conf,confruntime=tree_logits(model,arrays,parts['confirmation'],detail['selected_rounds'],args)
   else:
    model,detail=train_neural(arrays,y,parts,family,config,seed,args,target)
    cal,calruntime=logits_for(model,arrays,parts['calibration'],family);conf,confruntime=logits_for(model,arrays,parts['confirmation'],family)
   adj,la=adjust(cal,y[parts['calibration']],conf,detail['la_prior']);rank=meta['class_rank']
   metricsout={'raw':metrics(y[parts['confirmation']],conf.argmax(1),rank),'la':metrics(y[parts['confirmation']],adj.argmax(1),rank)}
   savez(target/'predictions.npz',calibration_ids=parts['calibration'],confirmation_ids=parts['confirmation'],calibration_labels=y[parts['calibration']],confirmation_labels=y[parts['confirmation']],calibration_logits=cal,confirmation_logits=conf)
   result=dict(seed=seed,family=family,protocol=p['fingerprint'],freeze=freeze['fingerprint'],training=detail,calibration=la,metrics=metricsout,
    inference=dict(calibration=calruntime,confirmation=confruntime),predictions_sha256=sha(target/'predictions.npz'))
   write_json(path,result);results.append(result)
 return dict(completed_records=len(results))

def holm(ps):
 ps=np.asarray(ps);order=np.argsort(ps);out=np.empty(len(ps)); prev=0.
 for i,j in enumerate(order):prev=max(prev,min(1.,(len(ps)-i)*ps[j]));out[j]=prev
 return out.tolist()
def summarize(records):
 summary=[]
 for family in FAMILIES:
  group=sorted([r for r in records if r['family']==family],key=lambda r:r['seed'])
  for readout in ['raw','la']:
   fields={}
   for key in ['accuracy','balanced_accuracy','balanced_macro_f1','macro_f1','macro_recall','head_f1','mid_f1','tail_f1']:
    values=np.array([r['metrics'][readout][key] for r in group]);avg=float(values.mean());sd=float(values.std(ddof=1));half=float(stats.t.ppf(.975,len(values)-1)*sd/math.sqrt(len(values)))
    fields[key]=dict(mean=avg,sd=sd,ci95=[avg-half,avg+half])
   summary.append(dict(method=family+'_'+readout,seeds=[r['seed'] for r in group],metrics=fields))
 lookup={(r['family'],r['seed']):r for r in records}; contrasts=[]
 for family in ['ft_transformer','xgboost','mlp_cb']:
  ds=np.array([lookup[(family,s)]['metrics']['la']['macro_f1']-lookup[('mlp_ce',s)]['metrics']['la']['macro_f1'] for s in SEEDS]);mean=float(ds.mean());sd=float(ds.std(ddof=1));half=float(stats.t.ppf(.975,9)*sd/math.sqrt(10))
  signs=np.asarray(list(itertools.product([-1,1],repeat=10))); exact=float(np.mean(np.abs((signs*ds).mean(1))>=abs(mean)-1e-15));tp=float(stats.ttest_1samp(ds,0).pvalue) if sd else (0. if mean else 1.)
  contrasts.append(dict(a=family+'_la',b='mlp_ce_la',differences=ds.tolist(),mean=mean,sd=sd,ci95=[mean-half,mean+half],t_p=tp,exact_signflip_p=exact))
 for k in ['t_p','exact_signflip_p']:
  for row,adjusted in zip(contrasts,holm([r[k] for r in contrasts])):row[k+'_holm']=adjusted
 return dict(summary=summary,contrasts=contrasts,completed_fits=len(records),readouts=2*len(records))

def independent_metrics(labels,pred,rank):
 from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
 labels=np.asarray(labels); pred=np.asarray(pred); c=len(rank); classes=np.arange(c)
 cm=confusion_matrix(labels,pred,labels=classes); per_f1=f1_score(labels,pred,labels=classes,average=None,zero_division=0)
 per_recall=recall_score(labels,pred,labels=classes,average=None,zero_division=0); per_precision=precision_score(labels,pred,labels=classes,average=None,zero_division=0)
 groups=np.array_split(np.asarray(rank),3); count=np.bincount(labels,minlength=c); sample_weight=1./count[labels]
 return dict(accuracy=float(np.mean(labels==pred)),balanced_accuracy=float(np.average(labels==pred,weights=sample_weight)),
  balanced_macro_f1=float(f1_score(labels,pred,labels=classes,average='macro',sample_weight=sample_weight,zero_division=0)),macro_f1=float(per_f1.mean()),macro_recall=float(per_recall.mean()),
  head_f1=float(per_f1[groups[0]].mean()),mid_f1=float(per_f1[groups[1]].mean()),tail_f1=float(per_f1[groups[2]].mean()),confusion_matrix=cm.tolist(),per_class_f1=per_f1.tolist(),per_class_recall=per_recall.tolist(),per_class_precision=per_precision.tolist(),per_class_support=count.tolist())

def compare_numbers(expected,actual,tolerance=1e-11):
 if isinstance(expected,dict):
  if set(expected)!=set(actual):raise ValueError('Numericfieldinventorymismatch')
  for k in expected:compare_numbers(expected[k],actual[k],tolerance)
 elif isinstance(expected,list):
  if not np.allclose(np.asarray(expected),np.asarray(actual),rtol=0,atol=tolerance):raise ValueError('Numericarraymismatch')
 elif isinstance(expected,(int,float)):
  if not math.isclose(expected,actual,rel_tol=0,abs_tol=tolerance):raise ValueError('Numericscalarmismatch')
 elif expected!=actual:raise ValueError('Numericmetadatamismatch')

def verify(args):
 # Pure numeric replay: no model deserialization or training, all curves recomputed.
 root=Path(args.out);meta=json.loads((root/'data/metadata.json').read_text());p,freeze=verify_freeze(args,meta)
 parts=dict(np.load(root/'data/split.npz'));labels=np.load(root/'data/labels.npy');validate_split(parts,labels);records=[];differences=[]
 for seed in SEEDS:
  for family in getattr(args,'families',FAMILIES):
   target=root/'final'/f'seed_{seed}'/family;r=json.loads((target/'metrics.json').read_text())
   if r['seed']!=seed or r['family']!=family or r['protocol']!=p['fingerprint'] or r['freeze']!=freeze['fingerprint']:raise ValueError('Resultidentitymismatch')
   path=target/'predictions.npz'
   if sha(path)!=r['predictions_sha256']:raise ValueError('Predictiondigestmismatch')
   with np.load(path,allow_pickle=False) as z:
    for name in ['calibration','confirmation']:
     if not np.array_equal(z[name+'_ids'],parts[name]) or not np.array_equal(z[name+'_labels'],labels[parts[name]]):raise ValueError('Savedroworlabelalignmentmismatch')
     if z[name+'_logits'].shape!=(len(parts[name]),7) or not np.isfinite(z[name+'_logits']).all():raise ValueError('Invalidsavedlogits')
    adj,la=adjust(z['calibration_logits'],z['calibration_labels'],z['confirmation_logits'],r['training']['la_prior'])
    replay={'raw':metrics(z['confirmation_labels'],z['confirmation_logits'].argmax(1),meta['class_rank']),'la':metrics(z['confirmation_labels'],adj.argmax(1),meta['class_rank'])}
   if replay!=r['metrics'] or la!=r['calibration']:raise ValueError('Numericreplaymismatch')
   with np.load(path,allow_pickle=False) as z:
    independent={'raw':independent_metrics(z['confirmation_labels'],z['confirmation_logits'].argmax(1),meta['class_rank']),'la':independent_metrics(z['confirmation_labels'],adj.argmax(1),meta['class_rank'])}
    compare_numbers(r['metrics'],independent)
    for candidate in r['calibration']['curve']:
     logits=z['calibration_logits']-candidate['tau']*(np.log(np.asarray(r['training']['la_prior']))-np.log(np.asarray(r['training']['la_prior'])).mean())
     value=independent_metrics(z['calibration_labels'],logits.argmax(1),meta['class_rank'])['macro_f1']
     if abs(value-candidate['macro_f1'])>1e-12:raise ValueError('Independentcalibrationcurvemismatch')
   if family=='xgboost':
    if r['training']['completed_rounds']!=600:raise ValueError('Treebudgetincomplete')
   elif r['training']['completed_epochs']!=args.epochs or r['training']['optimizer_updates']!=args.epochs*math.ceil(len(parts['fit'])/(args.ft_batch_size if family=='ft_transformer' else args.mlp_batch_size)):raise ValueError('Neuralbudgetincomplete')
   records.append(r)
 aggregate=summarize(records);write_json(root/'aggregate.json',aggregate)
 result=dict(status='verified_complete',protocol=p['fingerprint'],freeze=freeze['fingerprint'],completed_fits=len(records),readouts=len(records)*2,calibration_curves=len(records),scalar_metrics=len(records)*2*8,contrasts=len(aggregate['contrasts']),aggregate_sha256=sha(root/'aggregate.json'),
  verification_scope='Numericpredictionreplay,identities,labels/splits,trainingbudgets,calibrationchoices,pairedsummaries; modelinferenceandtrainingnotindependentlyrerun',source_sha256=sha(__file__))
 write_json(root/'independent_verification.json',result);return result

def main():
 a=argparse.ArgumentParser();a.add_argument('command',choices=['prepare','pilot','tune','run','verify']);a.add_argument('--out',type=Path,required=True);a.add_argument('--data',type=Path);a.add_argument('--source',type=Path)
 a.add_argument('--epochs',type=int,default=50);a.add_argument('--mlp-batch-size',type=int,default=2048);a.add_argument('--ft-batch-size',type=int,default=1024);a.add_argument('--threads',type=int,default=4)
 a.add_argument('--families',nargs='+',choices=FAMILIES,default=FAMILIES);a.add_argument('--ft-device',choices=['cpu','mps'],default='mps');a.add_argument('--seeds',type=int,nargs='+',default=SEEDS);a.add_argument('--pilot-steps',type=int,default=20);a.add_argument('--pilot-rounds',type=int,default=20);args=a.parse_args();torch.set_num_threads(args.threads);torch.set_num_interop_threads(1)
 result=globals()[args.command](args);print(json.dumps(result,indent=2,sort_keys=True,allow_nan=False))
if __name__=='__main__':main()
