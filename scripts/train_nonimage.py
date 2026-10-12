#!/usr/bin/env python3
"""Controlled from-scratch REBAL adaptation for prepared Covertype/DBpedia data.

Input directories are produced by prepare_nonimage.py. No downloading occurs
here. Text SMOTE interpolates learned representations, never token IDs. Primary
runs omit ECOD; --ecod adds a shared fit-only tabular sensitivity. This is a
non-image adaptation, not a reproduction of the original image architecture.

Example: python scripts/train_nonimage.py run --data data/covertype/native \
  data/dbpedia/rho100 --out runs/nonimage --device cuda
"""
from __future__ import annotations
import argparse
from collections import Counter
import fcntl
import hashlib
import importlib.metadata
import importlib.util
import itertools
import json
import math
import os
import platform
from pathlib import Path
import re
import sys
import time
import unicodedata

import numpy as np
from scipy import stats
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.neighbors import NearestNeighbors
import torch
from torch import nn
from torch.nn import functional as F

_spec=importlib.util.spec_from_file_location('nonimage_common',Path(__file__).with_name('train_tiny_fullscale.py'))
common=importlib.util.module_from_spec(_spec); _spec.loader.exec_module(common)
sha,write_json,save_npz=common.sha,common.write_json,common.save_npz
TRUNKS=('ce','cb','rebal')
HEADS=('natural','ros','smote')
TAUS=(0.,.25,.5,.75,1.,1.25,1.5,2.)
VERSION=1


def identifier(meta):
    return re.sub(r'[^A-Za-z0-9_.-]+','_',meta['dataset']+'__'+str(meta['condition']))


def load_condition(directory):
    directory=Path(directory); meta=json.loads((directory/'metadata.json').read_text())
    required={'labels.npy','split.npz','features.npy' if meta['task_type']=='tabular' else 'texts.jsonl'}
    if not required<=set(meta['files_sha256']): raise ValueError('Manifest omits required input checksums')
    for filename,digest in meta['files_sha256'].items():
        if sha(directory/filename)!=digest: raise ValueError(f'Input checksum mismatch: {filename}')
    y=np.load(directory/'labels.npy',mmap_mode='r')
    with np.load(directory/'split.npz') as z: split={name:z[name] for name in z.files}
    c=int(meta['n_classes']); parts=('train','calibration','test')
    for name in parts:
        ids=split[name]
        if ids.ndim!=1 or not np.issubdtype(ids.dtype,np.integer) or len(ids)==0 or ids.min()<0 or ids.max()>=len(y): raise ValueError('Invalid split row IDs')
        if len(np.unique(ids))!=len(ids): raise ValueError('Repeated row ID within split')
        if not np.array_equal(np.unique(y[ids]),np.arange(c)): raise ValueError('Every split must contain every class')
    if len(set(np.concatenate([split[n] for n in parts]).tolist()))!=sum(len(split[n]) for n in parts): raise ValueError('Split row IDs overlap')
    order=np.asarray(split['class_rank'])
    if not np.array_equal(np.sort(order),np.arange(c)): raise ValueError('Invalid class-rank permutation')
    # Preparation contract: class_rank lists class IDs from most to least frequent.
    semantics=meta.get('rank_semantics','ordered_class_ids_head_to_tail')
    if 'inverse' in semantics: rank=order.copy()
    else: rank=np.empty(c,dtype=np.int64); rank[order]=np.arange(c)
    split['rank']=rank
    return directory,meta,y,split


def tokens(title,content):
    def words(x): return re.findall(r'\w+',unicodedata.normalize('NFKC',x).casefold(),flags=re.UNICODE)
    left,right=words(title),words(content)
    return left+(['<sep>'] if left or right else [])+right


def build_vocabulary(path,train_indices,max_words):
    selected=set(map(int,train_indices)); counts=Counter()
    with open(path,encoding='utf-8') as stream:
        for i,line in enumerate(stream):
            if i in selected:
                row=json.loads(line); counts.update(t for t in tokens(row['title'],row['content']) if t!='<sep>')
    ordered=sorted(counts,key=lambda word:(-counts[word],word))[:max_words]
    return {'<pad>':0,'<unk>':1,'<sep>':2,**{word:i+3 for i,word in enumerate(ordered)}}


def verify_transforms(target,info):
    required={'inputs.npy','fit_indices.npy','excluded_ecod.npy','scaler.npz' if info['task_type']=='tabular' else 'vocabulary.json'}
    if set(info['artifacts'])!=required: raise ValueError('Incomplete transform artifact inventory')
    for name,digest in info['artifacts'].items():
        if sha(target/name)!=digest: raise ValueError('Transform artifact checksum mismatch')


def prepare_transforms(directory,meta,y,split,args,target):
    target.mkdir(parents=True,exist_ok=True)
    identity=dict(input_manifest=sha(directory/'metadata.json'),ecod=args.ecod,max_words=args.max_words,max_tokens=args.max_tokens,source=sha(__file__))
    fp=common.fingerprint(identity); lock=open(target/'prepare.lock','a'); fcntl.flock(lock,fcntl.LOCK_EX)
    marker=target/'transforms.json'
    if marker.exists():
        info=json.loads(marker.read_text())
        if info['fingerprint']!=fp: raise ValueError('Existing transforms belong to another protocol')
        verify_transforms(target,info)
        fit=np.load(target/'fit_indices.npy'); x=np.load(target/info['array_file'],mmap_mode='r'); lock.close()
        return x,fit,info
    fit=split['train'].copy(); excluded=np.empty(0,dtype=np.int64); extras=[]
    if meta['task_type']=='tabular':
        raw=np.load(directory/'features.npy',mmap_mode='r')
        if raw.shape!=(len(y),54) or not np.isfinite(raw).all(): raise ValueError('Covertype expects finite54-column predictors')
        if args.ecod:
            from pyod.models.ecod import ECOD
            keep=[]; removed=[]
            for c in range(meta['n_classes']):
                ids=fit[y[fit]==c]
                if len(ids)<10: keep.extend(ids); continue
                detector=ECOD(contamination=.05,n_jobs=1); detector.fit(np.asarray(raw[ids],dtype=np.float64))
                keep.extend(ids[detector.labels_==0]); removed.extend(ids[detector.labels_!=0])
            fit=np.asarray(keep,dtype=np.int64); excluded=np.asarray(removed,dtype=np.int64)
        mean=np.asarray(raw[fit,:10],dtype=np.float64).mean(0); scale=np.asarray(raw[fit,:10],dtype=np.float64).std(0); scale[scale==0]=1.
        x=np.lib.format.open_memmap(target/'inputs.npy',mode='w+',dtype=np.float32,shape=raw.shape)
        for begin in range(0,len(y),65536):
            stop=min(begin+65536,len(y)); x[begin:stop]=raw[begin:stop]; x[begin:stop,:10]=(raw[begin:stop,:10]-mean)/scale
        x.flush(); save_npz(target/'scaler.npz',mean=mean,scale=scale); extras.append('scaler.npz')
        detail=dict(numeric_columns=list(range(10)),categorical_columns=list(range(10,54)),fit_only=True)
    else:
        if args.ecod: raise ValueError('Text ECOD is unsupported; outlier models must not operate on token IDs')
        vocabulary=build_vocabulary(directory/'texts.jsonl',fit,args.max_words)
        write_json(target/'vocabulary.json',vocabulary); extras.append('vocabulary.json')
        x=np.lib.format.open_memmap(target/'inputs.npy',mode='w+',dtype=np.int32,shape=(len(y),args.max_tokens)); x[:]=0
        sets={name:set(map(int,fit if name=='train' else split[name])) for name in ('train','calibration','test')}
        membership={i:name for name,ids in sets.items() for i in ids}; counts={name:np.zeros((meta['n_classes'],4),dtype=np.int64) for name in sets}
        rows=0
        with open(directory/'texts.jsonl',encoding='utf-8') as stream:
            for i,line in enumerate(stream):
                rows+=1
                if i not in membership: continue
                row=json.loads(line); words=tokens(row['title'],row['content']); chosen=words[:args.max_tokens]
                encoded=[vocabulary.get(word,1) for word in chosen] or [1]; x[i,:len(encoded)]=encoded
                tally=counts[membership[i]][int(y[i])]; tally += [1,int(len(words)>args.max_tokens),len(chosen),sum(word not in vocabulary for word in chosen)]
        if rows!=len(y): raise ValueError('Text rows and labels disagree')
        x.flush(); detail=dict(vocabulary_size=len(vocabulary),vocabulary_fit_only=True,tokenization='Unicode NFKC+casefold; Unicode word tokens; title, SEP, content',
                special_tokens={'PAD':0,'UNK':1,'SEP':2},max_tokens=args.max_tokens,
                empty_document_encoding='single UNK token',
                split_class_diagnostics={k:v.tolist() for k,v in counts.items()},diagnostic_columns=['documents','truncated_documents','retained_tokens','OOV_retained_tokens'])
    if np.bincount(y[fit],minlength=meta['n_classes']).min()<2: raise ValueError('Each retained class needs at least2fit rows')
    np.save(target/'fit_indices.npy',fit); np.save(target/'excluded_ecod.npy',excluded)
    names=['inputs.npy','fit_indices.npy','excluded_ecod.npy']+extras
    info=dict(fingerprint=fp,array_file='inputs.npy',task_type=meta['task_type'],fit_count=len(fit),excluded_ecod=len(excluded),detail=detail,
              artifacts={name:sha(target/name) for name in names})
    write_json(marker,info); lock.close(); return x,fit,info


class Encoder(nn.Module):
    def __init__(self,kind,classes,vocabulary_size=None):
        super().__init__(); self.kind=kind
        if kind=='text':
            self.embedding=nn.EmbeddingBag(vocabulary_size,128,mode='mean',padding_idx=0)
            self.layers=nn.Sequential(nn.Linear(128,256),nn.ReLU(),nn.Dropout(.1),nn.Linear(256,128),nn.ReLU(),nn.Dropout(.1))
        else:
            self.layers=nn.Sequential(nn.Linear(54,256),nn.ReLU(),nn.Dropout(.1),nn.Linear(256,256),nn.ReLU(),nn.Linear(256,128),nn.ReLU(),nn.Dropout(.1))
        self.head=nn.Linear(128,classes); self.apply(common.initialize)
        if kind=='text':
            nn.init.normal_(self.embedding.weight,mean=0.,std=.05)
            with torch.no_grad(): self.embedding.weight[0].zero_()
    def forward(self,x):
        if self.kind=='text': x=self.embedding(x)
        features=self.layers(x); return self.head(features),features


def cb_weights(counts):
    counts=np.asarray(counts,dtype=np.float64); beta=1.-1./counts.max()
    effective=-np.expm1(counts*np.log1p(-1./counts.max()))
    w=(1.-beta)/effective; w/=np.sum(counts*w)/counts.sum()
    return w.astype(np.float32),float(beta)


def input_batch(x,ids,kind,device):
    return torch.as_tensor(np.array(x[ids],copy=True),dtype=torch.long if kind=='text' else torch.float32,device=device)


def linear_l2(model):
    return sum((m.weight.square().sum() for m in model.modules() if isinstance(m,nn.Linear)),start=torch.zeros((),device=next(model.parameters()).device))


def train_trunk(x,y,fit,kind,classes,vocab,method,seed,args,target,identity):
    target.mkdir(parents=True,exist_ok=True); checkpoint=target/'checkpoint.pt'; common.set_seed(seed)
    model=Encoder(kind,classes,vocab).to(args.device); initial=common.state_hash(model.state_dict())
    optimizer=torch.optim.SGD(model.parameters(),lr=args.lr,momentum=.9,nesterov=True)
    counts=np.bincount(y[fit],minlength=classes); weights,beta=cb_weights(counts); cw=torch.tensor(weights,device=args.device)
    steps=math.ceil(len(fit)/args.batch_size); total=steps*args.epochs; warmup=min(args.warmup_epochs,args.epochs)*steps
    start=0; history=[]; elapsed=0.
    if checkpoint.exists():
        ck=torch.load(checkpoint,map_location=args.device,weights_only=False)
        if ck['identity']!=identity or ck['method']!=method or ck['initial_state_sha256']!=initial: raise ValueError('Checkpoint identity mismatch')
        model.load_state_dict(ck['model']); optimizer.load_state_dict(ck['optimizer']); common.restore_rng(ck['rng'])
        start=ck['completed_epochs']; history=ck['history']; elapsed=ck['training_seconds']
    for epoch in range(start,args.epochs):
        model.train(); order=np.random.default_rng(np.random.SeedSequence([seed,epoch,707])).permutation(fit)
        t=time.perf_counter(); loss_sum=0.; eligible_counts=np.zeros(classes,dtype=np.int64); batches=0
        for offset in range(0,len(order),args.batch_size):
            ids=order[offset:offset+args.batch_size]; batch=input_batch(x,ids,kind,args.device); label=torch.as_tensor(np.array(y[ids]),device=args.device)
            step=epoch*steps+offset//args.batch_size; lr=common.learning_rate(step,total,warmup,args.lr)
            for group in optimizer.param_groups: group['lr']=lr
            optimizer.zero_grad(set_to_none=True); logits,features=model(batch); loss=F.cross_entropy(logits,label,reduction='none')
            if method!='ce' and epoch>=int(.8*args.epochs): loss=loss*cw[label]
            loss=loss.mean()+.0005*linear_l2(model)
            if method=='rebal' and epoch>=int(.8*args.epochs): loss=loss+.05*common.equalization(features,label,classes)
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite trunk loss')
            loss.backward(); gradients=[p.grad for p in model.parameters() if p.grad is not None]
            norms=torch.stack([g.norm() for g in gradients])
            if not torch.isfinite(norms).all(): raise RuntimeError('Nonfinite trunk gradient')
            for g,n in zip(gradients,norms): g.mul_((1./(n+1e-6)).clamp(max=1.))
            optimizer.step(); loss_sum+=float(loss.detach().cpu())*len(ids); eligible_counts+=(np.bincount(y[ids],minlength=classes)>=2); batches+=1
        duration=time.perf_counter()-t; elapsed+=duration
        history.append(dict(epoch=epoch+1,optimizer_updates=(epoch+1)*steps,loss=loss_sum/len(fit),seconds=duration,
                            mean_EQ_eligible_classes=float(eligible_counts.sum()/batches),EQ_eligible_batches_per_class=eligible_counts.tolist(),
                            batches=batches,EQ_active=method=='rebal' and epoch>=int(.8*args.epochs)))
        print(json.dumps(dict(seed=seed,method=method,**history[-1])),flush=True)
        if (epoch+1)%args.checkpoint_every==0 or epoch+1==args.epochs:
            common.save_torch(checkpoint,dict(identity=identity,method=method,initial_state_sha256=initial,model=model.state_dict(),optimizer=optimizer.state_dict(),rng=common.rng_state(),
                    completed_epochs=epoch+1,history=history,training_seconds=elapsed))
            write_json(target/'history.json',history)
    effective=counts/counts.sum() if method=='ce' else counts*weights/np.sum(counts*weights)
    return model,dict(initial_state_sha256=initial,completed_epochs=args.epochs,optimizer_updates=total,training_seconds=elapsed,beta=beta,
            class_weights=weights.tolist(),expected_sample_weight=float(np.sum(counts*weights)/counts.sum()),LA_prior=effective.tolist(),
            LA_prior_note='natural prior for CE; final deferred objective weighted prior for CB/REBAL, an explicit approximation to a changing objective')


def encode(model,x,indices,kind,args):
    model.eval(); logits=[]; features=[]
    with torch.inference_mode():
        for offset in range(0,len(indices),args.batch_size):
            l,f=model(input_batch(x,indices[offset:offset+args.batch_size],kind,args.device)); logits.append(l.cpu().numpy()); features.append(f.cpu().numpy())
    l,f=np.concatenate(logits),np.concatenate(features)
    if not np.isfinite(l).all() or not np.isfinite(f).all(): raise ValueError('Nonfinite representation')
    return l,f


def neighbor_graph(features,labels,classes,k=5,device='cpu',query_block=256,reference_block=8192):
    counts=np.bincount(labels,minlength=classes); actual_k=min(k,int(counts.min())-1)
    if actual_k<1: raise ValueError('Insufficient class examples for latent SMOTE')
    graph=np.full((len(labels),actual_k),-1,dtype=np.int64)
    for c in range(classes):
        ids=np.flatnonzero(labels==c)
        if len(ids)==counts.max(): continue  # No synthetic draws for an already maximal class.
        class_features=np.ascontiguousarray(features[ids])
        if device=='cpu':
            index=NearestNeighbors(n_neighbors=actual_k+1,algorithm='brute',metric='euclidean',n_jobs=1).fit(class_features)
            for q in range(0,len(ids),query_block):
                found=index.kneighbors(class_features[q:q+query_block],return_distance=False)
                for i,neighbors in enumerate(found): graph[ids[q+i]]=ids[neighbors[neighbors!=q+i][:actual_k]]
        else:
            reference=torch.as_tensor(class_features,device=device)
            with torch.inference_mode():
                for q in range(0,len(ids),query_block):
                    query=reference[q:q+query_block]; best=torch.full((len(query),actual_k),float('inf'),device=device); best_ids=torch.full((len(query),actual_k),-1,device=device,dtype=torch.long)
                    for r in range(0,len(ids),reference_block):
                        dist=torch.cdist(query,reference[r:r+reference_block]); ri=torch.arange(r,min(r+reference_block,len(ids)),device=device).expand(len(query),-1)
                        qi=torch.arange(q,q+len(query),device=device)[:,None]; dist.masked_fill_(ri==qi,float('inf'))
                        distances=torch.cat((best,dist),1); candidates=torch.cat((best_ids,ri),1)
                        best,where=torch.topk(distances,actual_k,dim=1,largest=False,sorted=True); best_ids=candidates.gather(1,where)
                    graph[ids[q:q+len(query)]]=ids[best_ids.cpu().numpy()]
    needed=counts[counts<counts.max()]; largest=int(needed.max()) if len(needed) else 0
    return graph,dict(k_neighbors=actual_k,metric='raw Euclidean',search='exhaustive blockwise nearest neighbors; floating-point ties backend-dependent',
                      query_block=query_block,reference_block=reference_block,device=device,excluded_self=True,
                      largest_oversampled_class=largest,graph_bytes=int(graph.nbytes),
                      estimated_GPU_distance_block_bytes=min(query_block,largest)*min(reference_block,largest)*4,
                      estimated_GPU_candidate_index_block_bytes=min(query_block,largest)*min(reference_block,largest)*8,
                      estimated_GPU_reference_feature_bytes=largest*features.shape[1]*4)


class LatentSampler:
    def __init__(self,features,labels,graph,device,seed):
        self.features=torch.as_tensor(features,dtype=torch.float32,device=device); self.labels=np.asarray(labels)
        self.members=[np.flatnonzero(labels==c) for c in range(int(labels.max())+1)]; self.counts=np.array([len(m) for m in self.members]); self.graph=graph
        self.rng=np.random.default_rng(seed); self.device=device
    def sample(self,kind,size):
        u,v,w,z,alpha=self.rng.random((5,size))
        if kind=='natural': base=(u*len(self.labels)).astype(np.int64); classes=self.labels[base]
        else:
            classes=(u*len(self.members)).astype(np.int64); base=np.empty(size,dtype=np.int64)
            for c,members in enumerate(self.members):
                mask=classes==c; base[mask]=members[(v[mask]*len(members)).astype(np.int64)]
        synthetic=(w>=self.counts[classes]/self.counts.max()) if kind=='smote' else np.zeros(size,dtype=bool)
        neighbor=base.copy()
        if synthetic.any(): neighbor[synthetic]=self.graph[base[synthetic],(z[synthetic]*self.graph.shape[1]).astype(np.int64)]
        if (neighbor<0).any() or not np.array_equal(self.labels[neighbor],classes): raise ValueError('Invalid cross-class SMOTE neighbor')
        base_t=torch.as_tensor(base,device=self.device); neighbor_t=torch.as_tensor(neighbor,device=self.device)
        amount=torch.as_tensor(alpha*synthetic,dtype=torch.float32,device=self.device)[:,None]
        sample=self.features[base_t]+amount*(self.features[neighbor_t]-self.features[base_t])
        return sample,torch.as_tensor(classes,device=self.device),dict(base=base,neighbor=neighbor,alpha=alpha*synthetic,synthetic=synthetic)


def train_head(features,labels,graph,kind,seed,args,classes):
    common.set_seed(seed+90909); head=nn.Linear(128,classes).to(args.device); common.initialize(head)
    initial=common.state_hash(head.state_dict()); optimizer=torch.optim.SGD(head.parameters(),lr=.01,momentum=.9)
    sampler=LatentSampler(features,labels,graph,args.device,seed+60606); history=[]; started=time.perf_counter()
    for step in range(args.head_steps):
        x,y,_=sampler.sample(kind,args.head_batch_size); optimizer.zero_grad(set_to_none=True)
        loss=F.cross_entropy(head(x),y)+.0005*head.weight.square().sum()
        if not torch.isfinite(loss): raise RuntimeError('Nonfinite head loss')
        loss.backward(); optimizer.step()
        if step==0 or (step+1)%100==0 or step+1==args.head_steps: history.append(dict(update=step+1,loss=float(loss.detach().cpu())))
    prior=np.bincount(labels,minlength=classes).astype(float) if kind=='natural' else np.ones(classes)
    prior/=prior.sum()
    return head,dict(initial_state_sha256=initial,optimizer_updates=args.head_steps,seconds=time.perf_counter()-started,losses=history,training_prior=prior.tolist(),
                    sampling='natural uniform rows' if kind=='natural' else 'class-uniform streaming '+kind,
                    smote_variant='fresh within-class interpolations on exact kNN graph; real probability n_class/n_max' if kind=='smote' else None)


def scores(y,pred,rank):
    c=len(rank); output=common.metrics(y,pred,rank)
    output['per_class_precision']=precision_score(y,pred,labels=np.arange(c),average=None,zero_division=0).tolist()
    output['confusion_matrix']=confusion_matrix(y,pred,labels=np.arange(c)).tolist(); return output


def adjust(cal_logits,cal_labels,test_logits,prior):
    prior=np.asarray(prior,dtype=np.float32); log_prior=np.log(prior); curve=[]
    if np.ptp(prior)==0:
        value=float(f1_score(cal_labels,cal_logits.argmax(1),average='macro',zero_division=0))
        return cal_logits.copy(),test_logits.copy(),dict(tau=0.,curve=[dict(tau=t,macro_f1=value) for t in TAUS],prior=prior.tolist(),
                selection='uniform training prior changes all logits by one common constant; tau fixed0 to avoid numerical tie artifacts')
    for tau in TAUS:
        pred=(cal_logits-tau*log_prior).argmax(1); value=float(f1_score(cal_labels,pred,average='macro',zero_division=0))
        curve.append(dict(tau=tau,macro_f1=value))
    tau=max(curve,key=lambda v:v['macro_f1'])['tau']
    return cal_logits-tau*log_prior,test_logits-tau*log_prior,dict(tau=tau,curve=curve,prior=list(map(float,prior)),selection='calibration macro-F1; smallest tau breaks ties')


def required_methods():
    base=set(TRUNKS)|{m+'_crt_'+h for m in TRUNKS for h in HEADS}
    return base|{m+'_la' for m in base}|{m+'_ncm' for m in TRUNKS}


def validate_result(result,path,identity,args):
    if result['identity']!=identity: raise ValueError('Completed cell identity mismatch')
    if set(result['metrics'])!=required_methods() or set(result['trunks'])!=set(TRUNKS): raise ValueError('Incomplete methods')
    if len({r['initial_state_sha256'] for r in result['trunks'].values()})!=1: raise ValueError('Unmatched backbone initialization')
    for record in result['trunks'].values():
        if record['completed_epochs']!=args.epochs or record['optimizer_updates']!=args.epochs*math.ceil(result['fit_count']/args.batch_size): raise ValueError('Incorrect trunk budget')
    for name,record in result['metrics'].items():
        if not np.isfinite([v for v in record.values() if isinstance(v,(int,float))]+record['per_class_f1']+record['per_class_recall']).all(): raise ValueError('Nonfinite metrics')
        if '_crt_' in name and not name.endswith('_la') and record['training']['optimizer_updates']!=args.head_steps: raise ValueError('Incorrect head budget')
    required={'predictions.npz','fit_indices.npy'}|{f'{m}/{n}' for m in TRUNKS for n in ('checkpoint.pt','history.json','knn_graph.npy','knn.json')}|{f'{m}/head_{h}.pt' for m in TRUNKS for h in HEADS}
    if set(result['artifacts'])!=required: raise ValueError('Incomplete artifact inventory')
    for name,digest in result['artifacts'].items():
        if sha(path/name)!=digest: raise ValueError('Completed artifact checksum mismatch')


def run_seed(condition,args,protocol,seed):
    directory,meta,y,split=condition; key=identifier(meta); target=args.out/key; target.mkdir(parents=True,exist_ok=True)
    x,fit,transforms=prepare_transforms(directory,meta,y,split,args,target/'transforms')
    identity=dict(protocol=protocol['fingerprint'],condition=key,seed=seed,transforms=transforms['fingerprint'])
    cell=target/f'seed_{seed}'; cell.mkdir(exist_ok=True); result_path=cell/'metrics.json'
    if result_path.exists():
        result=json.loads(result_path.read_text()); validate_result(result,cell,identity,args); return result
    np.save(cell/'fit_indices.npy',fit); artifacts=['fit_indices.npy']; c=meta['n_classes']; rank=split['rank']; kind=meta['task_type']
    vocab=transforms['detail'].get('vocabulary_size'); cal_y=np.asarray(y[split['calibration']]); test_y=np.asarray(y[split['test']]); fit_y=np.asarray(y[fit])
    results={}; trunks={}; predictions=dict(test_labels=test_y,calibration_labels=cal_y,class_rank=rank)
    def record(name,cal_logits,test_logits,prior=None):
        results[name]=scores(test_y,test_logits.argmax(1),rank); predictions[name+'__test_logits']=test_logits; predictions[name+'__calibration_logits']=cal_logits
        if prior is not None:
            cl,tl,choice=adjust(cal_logits,cal_y,test_logits,prior); record(name+'_la',cl,tl); results[name+'_la']['adjustment']=choice
    for method in TRUNKS:
        folder=cell/method; model,trunk=train_trunk(x,y,fit,kind,c,vocab,method,seed,args,folder,identity); trunks[method]=trunk
        _,fit_features=encode(model,x,fit,kind,args); cl,cf=encode(model,x,split['calibration'],kind,args); tl,tf=encode(model,x,split['test'],kind,args)
        record(method,cl,tl,trunk['LA_prior']); record(method+'_ncm',common.ncm_logits(fit_features,fit_y,cf,c),common.ncm_logits(fit_features,fit_y,tf,c))
        graph_identity=dict(identity=identity,trunk=method,features_sha256=hashlib.sha256(fit_features.tobytes()).hexdigest())
        if (folder/'knn.json').exists():
            graph_meta=json.loads((folder/'knn.json').read_text())
            if graph_meta['identity']!=graph_identity or sha(folder/'knn_graph.npy')!=graph_meta['graph_sha256']: raise ValueError('Neighbor graph identity mismatch')
            graph=np.load(folder/'knn_graph.npy')
        else:
            baseline_memory=None
            if args.knn_device=='cuda': torch.cuda.reset_peak_memory_stats(); baseline_memory=torch.cuda.memory_allocated()
            started=time.perf_counter(); graph,graph_meta=neighbor_graph(fit_features,fit_y,c,args.smote_k,args.knn_device,args.knn_query_block,args.knn_reference_block)
            graph_meta.update(seconds=time.perf_counter()-started,feature_standardization='none; shared raw representation geometry for all heads',identity=graph_identity)
            if baseline_memory is not None: graph_meta.update(CUDA_baseline_allocated_bytes=baseline_memory,CUDA_process_peak_allocated_bytes=torch.cuda.max_memory_allocated())
            np.save(folder/'knn_graph.npy',graph); graph_meta['graph_sha256']=sha(folder/'knn_graph.npy'); write_json(folder/'knn.json',graph_meta)
        for h in HEADS:
            head,training=train_head(fit_features,fit_y,graph,h,seed,args,c); name=method+'_crt_'+h
            cal_logits=common.head_logits(head,cf,args); test_logits=common.head_logits(head,tf,args)
            record(name,cal_logits,test_logits,training['training_prior']); results[name]['training']=training
            common.save_torch(folder/f'head_{h}.pt',dict(identity=identity,method=name,model=head.state_dict(),training=training)); artifacts.append(f'{method}/head_{h}.pt'); del head
        artifacts.extend(f'{method}/{n}' for n in ('checkpoint.pt','history.json','knn_graph.npy','knn.json')); del model,fit_features,cf,tf,graph
    if len({v['initial_state_sha256'] for v in trunks.values()})!=1: raise RuntimeError('Unmatched backbone initialization')
    save_npz(cell/'predictions.npz',**predictions); artifacts.append('predictions.npz')
    result=dict(identity=identity,completed_utc=common.now(),fit_count=len(fit),calibration_count=len(cal_y),test_count=len(test_y),fit_class_counts=np.bincount(fit_y,minlength=c).tolist(),
                metrics=results,trunks=trunks,artifacts={name:sha(cell/name) for name in artifacts})
    write_json(result_path,result); validate_result(result,cell,identity,args); return result


def aggregate(conditions,args,protocol):
    rows=[]; cells=[]; comparisons=[]
    for directory,meta,y,split in conditions:
        key=identifier(meta); marker=args.out/key/'transforms'/'transforms.json'
        if not marker.exists(): continue
        transforms=json.loads(marker.read_text())
        verify_transforms(marker.parent,transforms)
        for seed in args.seeds:
            cell=args.out/key/f'seed_{seed}'; path=cell/'metrics.json'
            if not path.exists(): continue
            result=json.loads(path.read_text()); identity=dict(protocol=protocol['fingerprint'],condition=key,seed=seed,transforms=transforms['fingerprint'])
            validate_result(result,cell,identity,args); cells.append(result)
            for method,m in result['metrics'].items(): rows.append(dict(condition=key,seed=seed,method=method,**{k:v for k,v in m.items() if isinstance(v,(int,float))}))
        subset=[r for r in cells if r['identity']['condition']==key]
        if len(subset)!=len(args.seeds): continue
        for a,b in [('rebal_crt_smote','rebal_crt_ros'),('rebal_crt_natural','cb_crt_natural')]:
            subset.sort(key=lambda r:r['identity']['seed']); d=np.array([r['metrics'][a]['macro_f1']-r['metrics'][b]['macro_f1'] for r in subset]); n=len(d)
            avg=float(d.mean()); sd=float(d.std(ddof=1)) if n>1 else None
            half=float(stats.t.ppf(.975,n-1)*sd/math.sqrt(n)) if n>1 else None
            exact=float(np.mean(np.abs((np.asarray(list(itertools.product((-1.,1.),repeat=n)))*d).mean(1))>=abs(avg)-1e-12))
            p=float(stats.ttest_1samp(d,0).pvalue) if n>1 and np.any(d!=d[0]) else 1. if np.all(d==0) else None
            comparisons.append(dict(condition=key,a=a,b=b,endpoint='macro_f1',seeds=[r['identity']['seed'] for r in subset],differences=d.tolist(),mean=avg,ci95=None if half is None else [avg-half,avg+half],signflip_p=exact,t_p=p))
    family=2*len(conditions)
    if len(comparisons)==family:
        for source,destination in [('signflip_p','signflip_holm'),('t_p','t_holm')]:
            order=sorted(range(family),key=lambda i:comparisons[i][source] if comparisons[i][source] is not None else 1.); running=0.
            for rank,i in enumerate(order):
                p=comparisons[i][source]; running=max(running,min(1.,(family-rank)*(p if p is not None else 1.))); comparisons[i][destination]=running if p is not None else None
    summary=[]
    for key,method in sorted({(r['condition'],r['method']) for r in rows}):
        group=[r for r in rows if r['condition']==key and r['method']==method]; metrics={}
        for metric in ('accuracy','macro_f1','macro_recall','head_f1','mid_f1','tail_f1'):
            values=np.array([r[metric] for r in group]); n=len(values); avg=float(values.mean()); sd=float(values.std(ddof=1)) if n>1 else None
            half=float(stats.t.ppf(.975,n-1)*sd/math.sqrt(n)) if n>1 else None
            metrics[metric]=dict(mean=avg,sd=sd,ci95=None if half is None else [avg-half,avg+half])
        summary.append(dict(condition=key,method=method,n=len(group),metrics=metrics))
    result=dict(protocol=protocol['fingerprint'],completed_cells=len(cells),expected_cells=len(conditions)*len(args.seeds),evaluations=len(rows),summary=summary,rows=rows,paired_comparisons=comparisons,
            inference_note='Fixed two-comparison family per condition; no correction until whole family complete. Five-seed two-sided exact signflip minimum p=.0625.')
    write_json(args.out/'aggregate.json',result); return result


def throughput_pilot(condition,args,protocol):
    """Measure train-only device support/throughput; never encode or score held-out rows."""
    directory,meta,y,split=condition; target=args.out/identifier(meta); target.mkdir(parents=True,exist_ok=True)
    x,fit,transform=prepare_transforms(directory,meta,y,split,args,target/'transforms')
    classes=meta['n_classes']; kind=meta['task_type']; seed=args.seeds[0]; records=[]
    weights,beta=cb_weights(np.bincount(y[fit],minlength=classes)); cw=torch.tensor(weights,device=args.device)
    for method in TRUNKS:
        common.set_seed(seed); rng=np.random.default_rng(np.random.SeedSequence([seed,818]))
        model=Encoder(kind,classes,transform['detail'].get('vocabulary_size')).to(args.device)
        initial=common.state_hash(model.state_dict()); optimizer=torch.optim.SGD(model.parameters(),lr=args.lr,momentum=.9,nesterov=True)
        model.train(); started=time.perf_counter(); losses=[]; processed=0
        for step in range(args.pilot_steps):
            ids=rng.choice(fit,size=min(args.batch_size,len(fit)),replace=False)
            batch=input_batch(x,ids,kind,args.device); label=torch.as_tensor(np.array(y[ids]),device=args.device)
            optimizer.zero_grad(set_to_none=True); logits,features=model(batch); loss=F.cross_entropy(logits,label,reduction='none')
            if method!='ce': loss=loss*cw[label]
            loss=loss.mean()+.0005*linear_l2(model)
            if method=='rebal': loss=loss+.05*common.equalization(features,label,classes)
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite pilot loss')
            loss.backward(); gradients=[p.grad for p in model.parameters() if p.grad is not None]; norms=torch.stack([g.norm() for g in gradients])
            if not torch.isfinite(norms).all(): raise RuntimeError('Nonfinite pilot gradient')
            for gradient,norm in zip(gradients,norms): gradient.mul_((1./(norm+1e-6)).clamp(max=1.))
            optimizer.step(); losses.append(float(loss.detach().cpu())); processed+=len(ids)
        if args.device=='cuda': torch.cuda.synchronize()
        elif args.device=='mps': torch.mps.synchronize()
        seconds=time.perf_counter()-started
        records.append(dict(method=method,optimizer_updates=args.pilot_steps,examples=processed,seconds=seconds,examples_per_second=processed/seconds,
                            initial_state_sha256=initial,final_state_sha256=common.state_hash(model.state_dict()),first_loss=losses[0],last_loss=losses[-1]))
        print(json.dumps(dict(stage='train_only_pilot',condition=identifier(meta),**records[-1])),flush=True); del model
    report=dict(protocol=protocol['fingerprint'],condition=identifier(meta),completed_utc=common.now(),records=records,
                calibration_evaluated=False,test_evaluated=False,neighbor_graph_built=False,readouts_trained=False,
                sampling='independent fit-only minibatches without replacement within batch',
                scope='throughput/operator smoke only; CB/EQ immediately active; constantLR; weights discarded; not final accuracy evidence',
                fit_count=len(fit),CB_beta=beta)
    write_json(target/'pilot.json',report); return report


def make_protocol(conditions,args):
    names=('torch','numpy','scipy','scikit-learn')+ (('pyod',) if args.ecod else ())
    settings={k:v for k,v in vars(args).items() if k not in ('data','out','command','aggregate_only','worker_index','worker_count','defer_aggregate')}
    p=dict(version=VERSION,execution='train_only_pilot' if args.pilot_steps else 'full_study',scope='from-scratch non-image REBAL adaptation; neither original image architecture nor a transformer benchmark',settings=settings,
           source_sha256=sha(__file__),shared_helpers_sha256=sha(Path(__file__).with_name('train_tiny_fullscale.py')),
           inputs={identifier(m):sha(d/'metadata.json') for d,m,_,_ in conditions},software={name:importlib.metadata.version(name) for name in names},
           runtime=dict(python=sys.version.split()[0],OS=platform.system(),OS_release=platform.release(),architecture=platform.machine(),
                        device=args.device,GPU_name=torch.cuda.get_device_name(0) if args.device=='cuda' else None,CUDA=torch.version.cuda,
                        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),determinism_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
                        cudnn_deterministic=torch.backends.cudnn.deterministic,matmul_TF32=torch.backends.cuda.matmul.allow_tf32,
                        repeatability_limit='CUDA EmbeddingBag backward and floating-point top-k ties may vary; seeds do not guarantee bitwise cross-device reproduction'),
           architecture='tabular54->256->256->128; text random128-dimensional embedding mean->256->128; ReLU,dropout.1,sharedlinearhead',
           optimizer=dict(name='SGD',momentum=.9,nesterov=True,linear_kernel_l2=.0005,per_tensor_clipnorm=1.,defer_fraction=.8,eq_lambda=.05),
           CB='beta=1-1/n_max; stableexpm1; weights normalized to mean1 over fit examples; intentionallarge-dataadaptation',
           sampling='paired natural row order for every trunk; fit-only transformations; fixedsplitandranks across optimizationseeds',
           SMOTE='latentstreaming freshsynthetics withprobability1-n_class/n_max; exactblockwise kNN on retainedfitrepresentations; neverinterpolatetokenIDs orrawcategories',
           feature_geometry='raw learned128D; no extra standardization for any head',
           head_budget=dict(updates=args.head_steps,batch_size=args.head_batch_size,lr=.01,momentum=.9,nesterov=False,linear_kernel_l2=.0005),
           LA='calibrationmacroF1; taus'+str(TAUS)+'; use head-specific trainingprior, uniform forROS/SMOTE; CBnativeusesfinalweightedobjectiveprior',
           primary_endpoint='macro_f1',primary_comparisons=['rebal_crt_smote minus rebal_crt_ros','rebal_crt_natural minus cb_crt_natural'],
           selection='fixed final epochs and updates; no test-set model or epoch selection',statistical_family=2*len(conditions))
    p['fingerprint']=common.fingerprint(p); return p


def main():
    parser=argparse.ArgumentParser(description=__doc__); sub=parser.add_subparsers(dest='command',required=True); run=sub.add_parser('run')
    run.add_argument('--data',type=Path,nargs='+',required=True); run.add_argument('--out',type=Path,required=True)
    run.add_argument('--seeds',type=int,nargs='+',default=[0,1,2,3,4]); run.add_argument('--epochs',type=int,default=50)
    run.add_argument('--warmup-epochs',type=int,default=5); run.add_argument('--batch-size',type=int,default=1024); run.add_argument('--lr',type=float,default=.05)
    run.add_argument('--head-steps',type=int,default=2000); run.add_argument('--head-batch-size',type=int,default=2048)
    run.add_argument('--max-words',type=int,default=50000); run.add_argument('--max-tokens',type=int,default=256)
    run.add_argument('--device',choices=['cpu','cuda','mps'],default='cpu'); run.add_argument('--knn-device',choices=['cpu','cuda','mps'],default=None)
    run.add_argument('--knn-query-block',type=int,default=256); run.add_argument('--knn-reference-block',type=int,default=8192); run.add_argument('--smote-k',type=int,default=5)
    run.add_argument('--ecod',action='store_true'); run.add_argument('--threads',type=int,default=4); run.add_argument('--checkpoint-every',type=int,default=5)
    run.add_argument('--pilot-steps',type=int,default=0,help='Positive value runs only train-set throughput/operator checks; no held-out scores, kNN, or final readouts')
    run.add_argument('--aggregate-only',action='store_true'); run.add_argument('--worker-index',type=int,default=0); run.add_argument('--worker-count',type=int,default=1); run.add_argument('--defer-aggregate',action='store_true')
    args=parser.parse_args()
    if min(args.epochs,args.batch_size,args.head_steps,args.head_batch_size,args.max_words,args.max_tokens,args.checkpoint_every,args.smote_k,args.knn_query_block,args.knn_reference_block)<1: parser.error('Positive sizes and budgets required')
    if args.warmup_epochs<0 or args.lr<=0 or len(set(args.seeds))!=len(args.seeds) or len(args.seeds)>20: parser.error('Invalid optimizer/seeds; exhaustive signflip supports at most20seeds')
    if args.worker_count<1 or not 0<=args.worker_index<args.worker_count or (args.worker_count>1 and not args.defer_aggregate and not args.aggregate_only): parser.error('Invalid worker configuration')
    if args.pilot_steps<0 or (args.pilot_steps and (args.aggregate_only or args.worker_count!=1)): parser.error('Train-only pilot requires one worker and no aggregation')
    if args.device=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    if args.device=='mps' and not torch.backends.mps.is_available(): raise RuntimeError('MPS unavailable')
    if args.knn_device is None: args.knn_device=args.device
    if args.knn_device=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA kNN requested but unavailable')
    if args.knn_device=='mps' and not torch.backends.mps.is_available(): raise RuntimeError('MPS kNN requested but unavailable')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.set_num_threads(args.threads); torch.use_deterministic_algorithms(True,warn_only=True)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True; torch.backends.cuda.matmul.allow_tf32=False
    conditions=[load_condition(p) for p in args.data]
    if len({identifier(m) for _,m,_,_ in conditions})!=len(conditions): raise ValueError('Duplicate condition identifiers')
    args.out.mkdir(parents=True,exist_ok=True); protocol=make_protocol(conditions,args); ppath=args.out/'protocol.json'
    if ppath.exists():
        if json.loads(ppath.read_text())['fingerprint']!=protocol['fingerprint']: raise ValueError('Protocol mismatch; use distinct output directory')
    else:
        write_json(ppath,protocol); write_json(args.out/'study_plan.json',dict(created_utc=common.now(),protocol=protocol['fingerprint'],execution=protocol['execution'],planned_conditions=list(protocol['inputs']),seeds=args.seeds,
                expected_cells=0 if args.pilot_steps else len(conditions)*len(args.seeds),primary_family=0 if args.pilot_steps else 2*len(conditions)))
    if args.pilot_steps:
        for condition in conditions: throughput_pilot(condition,args,protocol)
        return
    if not args.aggregate_only:
        grid=[(condition,seed) for condition in conditions for seed in args.seeds]
        for index,(condition,seed) in enumerate(grid):
            if index%args.worker_count!=args.worker_index: continue
            run_seed(condition,args,protocol,seed)
            if not args.defer_aggregate: aggregate(conditions,args,protocol)
    if args.aggregate_only or not args.defer_aggregate: aggregate(conditions,args,protocol)


if __name__=='__main__': main()
