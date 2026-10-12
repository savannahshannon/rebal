#!/usr/bin/env python3
"""Validation-selected equalization factorial on frozen DeiT features.

Run phases in order: prepare, pilot, tune, select, evaluate, aggregate, verify.
Confirmation inference refuses to run until all tuning cells have finished and
selection.json is frozen. No final scores enter hyperparameter selection.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import time
import numpy as np
from scipy import stats
import torch
from torch import nn
from torch.nn import functional as F

SPEC=importlib.util.spec_from_file_location('followup_vit',Path(__file__).with_name('benchmark_vit_rebal.py'))
v=importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(v)
common, nonimage=v.common,v.nonimage
sha,fingerprint,write_json,save_npz=v.sha,v.fingerprint,v.write_json,v.save_npz
require=v.require
GRID=[dict(onset=o,coefficient=l) for o in (.2,.5,.8) for l in (.01,.05)]
REFERENCE=dict(onset=.8,coefficient=.05)
BCL_GRID=[dict(coefficient=l,temperature=t) for l in (.05,.2,.5) for t in (.07,.1)]
LR_GRID=(.01,.025,.05,.075,.1,.2)


def equalization(features,labels,classes):
    """Same within-minus-between formula without allocating K×K×D tensor."""
    f=F.normalize(features,p=2,dim=1); counts=torch.bincount(labels,minlength=classes)
    present=torch.nonzero(counts>=2).flatten()
    if not len(present): return features.sum()*0.
    sums=torch.zeros(classes,f.shape[1],device=f.device,dtype=f.dtype).index_add_(0,labels,f)
    centers=sums/counts.clamp_min(1).to(f.dtype)[:,None]
    spread=torch.zeros(classes,device=f.device,dtype=f.dtype).index_add_(0,labels,(f-centers[labels]).square().sum(1))
    within=(spread[present]/counts[present]).mean()
    if len(present)<2: return within
    selected=centers[present]; n=len(present)
    between=2*(n*selected.square().sum()-selected.sum(0).square().sum())/(n*(n-1))
    return within-between


def balanced_contrastive(features,labels,prototypes,temperature):
    """BCL class averaging + class complement; two views and one center/class.

    Independent implementation of Zhu et al. CVPR2022 BalSCL formulation:
    negative-class terms average by total instances including their prototype;
    same-class terms use count minus one. Self similarities are excluded.
    The adapter variant uses dropout views and normalized classifier weights.
    """
    require(features.ndim==3 and features.shape[1]==2 and len(labels)==len(features),'Invalid two-view features')
    require(temperature>0 and prototypes.shape[1]==features.shape[2],'Invalid BCL temperature/centers')
    classes=len(prototypes); anchors=F.normalize(torch.cat((features[:,0],features[:,1])),dim=1)
    allfeatures=torch.cat((anchors,F.normalize(prototypes,dim=1)))
    all_labels=torch.cat((labels,labels,torch.arange(classes,device=labels.device)))
    counts=torch.bincount(all_labels,minlength=classes).to(features.dtype)
    positive=(all_labels[:len(anchors),None]==all_labels[None,:])
    active=torch.ones_like(positive); active[torch.arange(len(anchors)),torch.arange(len(anchors))]=False
    positive=positive&active
    divisors=counts[all_labels][None,:]-positive.to(features.dtype)
    similarities=anchors@allfeatures.T/temperature
    denominator=torch.logsumexp((similarities-torch.log(divisors)).masked_fill(~active,-torch.inf),dim=1)
    peranchor=-(similarities-denominator[:,None]).masked_fill(~positive,0.).sum(1)/positive.sum(1)
    return peranchor.mean()


def make_partition(labels,eligible,classes=200,split_seed=20261013,tune_per_class=20,
                   cal_per_class=20,confirmation_per_class=100,rho=100):
    """Partition canonical development rows before fixed long-tail thinning."""
    labels=np.asarray(labels); eligible=np.asarray(eligible,dtype=np.int64)
    require(len(np.unique(eligible))==len(eligible),'Repeated eligible row')
    require(np.array_equal(np.unique(labels[eligible]),np.arange(classes)),'Missing class')
    rank=np.random.default_rng(np.random.SeedSequence([split_seed,901])).permutation(classes).astype(np.int64)
    per_class=[np.flatnonzero(labels[eligible]==c) for c in range(classes)]
    hold=tune_per_class+cal_per_class+confirmation_per_class
    nmax=min(map(len,per_class))-hold
    require(nmax>=3 and rho>=1,'Too few fit rows after independent holdouts')
    counts=np.maximum(1,np.floor(nmax*np.power(float(rho),-np.arange(classes)/max(1,classes-1))).astype(np.int64))
    fit=[]; pool=[]; tune=[]; cal=[]; confirmation=[]
    for c,positions in enumerate(per_class):
        ids=np.random.default_rng(np.random.SeedSequence([split_seed,c,902])).permutation(eligible[positions])
        tune.extend(ids[:tune_per_class]); cal.extend(ids[tune_per_class:tune_per_class+cal_per_class])
        confirmation.extend(ids[tune_per_class+cal_per_class:hold]); pool.extend(ids[hold:])
        r=int(np.flatnonzero(rank==c)[0]); fit.extend(ids[hold:hold+counts[r]])
    result={k:np.asarray(a,dtype=np.int64) for k,a in dict(fit=fit,fit_pool=pool,tune=tune,calibration=cal,confirmation=confirmation).items()}
    result['class_order']=rank
    inverse=np.empty(classes,dtype=np.int64); inverse[rank]=np.arange(classes)
    result['class_rank']=inverse
    validate_partition(result,labels,eligible,classes)
    return result


def validate_partition(split,labels,eligible,classes):
    names=('fit_pool','tune','calibration','confirmation')
    sets=[set(map(int,split[n])) for n in names]
    require(all(len(s)==len(split[n]) for s,n in zip(sets,names)),'Partition repeats row')
    require(all(not a&b for a,b in itertools.combinations(sets,2)),'Partition overlap')
    require(set.union(*sets)==set(map(int,eligible)),'Partition does not cover clean development')
    require(set(map(int,split['fit']))<=sets[0] and len(np.unique(split['fit']))==len(split['fit']),'Invalid thinned fit')
    require(sorted(split['class_rank'].tolist())==list(range(classes)),'Invalid class rank')
    require(np.array_equal(split['class_rank'][split['class_order']],np.arange(classes)),'Rank inverse differs')
    for n in ('fit','tune','calibration','confirmation'):
        require(np.bincount(labels[split[n]],minlength=classes).min()>0,'Partition missing class')


def importance_weights(counts,sampler):
    """p*(row)=1/(C*n_c); natural q(row)=1/N; balanced q=p*."""
    counts=np.asarray(counts,dtype=np.float64)
    require(np.all(counts>0) and sampler in ('natural','balanced'),'Invalid sampling counts')
    return (counts.sum()/(len(counts)*counts) if sampler=='natural' else np.ones(len(counts))).astype(np.float32)


def epoch_batches(fit,labels,classes,seed,epoch,batch_size,sampler):
    rng=np.random.default_rng(np.random.SeedSequence([seed,epoch,707]))
    members=[np.flatnonzero(labels[fit]==c) for c in range(classes)]
    offsets=np.concatenate(([0],np.cumsum([len(m) for m in members[:-1]])))
    flat=np.concatenate(members); counts=np.array([len(m) for m in members])
    for _ in range(math.ceil(len(fit)/batch_size)):
        u,vv=rng.random((2,batch_size))
        if sampler=='natural': local=(u*len(fit)).astype(np.int64)
        else:
            c=(u*classes).astype(np.int64); local=flat[offsets[c]+(vv*counts[c]).astype(np.int64)]
        yield fit[local]


def configuration_name(config):
    if config['objective'] in ('ce','balanced_softmax'):
        return config['objective']+'__lr'+format(config.get('peak_lr',.05),'.3g')
    if config['objective']=='bcl_dropout':
        return 'bcl_dropout__lambda'+format(config['coefficient'],'.2g')+'__temperature'+format(config['temperature'],'.2g')
    return config['objective']+'__'+config['sampler']+('__off' if not config['coefficient'] else '__onset'+format(config['onset'],'.2g')+'__lambda'+format(config['coefficient'],'.2g'))


def configurations():
    result=[]
    for sampler in ('natural','balanced'):
        result.append(dict(objective='class_balanced',sampler=sampler,onset=0.,coefficient=0.))
        result.extend(dict(objective='class_balanced',sampler=sampler,**g) for g in GRID)
    result.extend(dict(objective=objective,sampler='natural',onset=0.,coefficient=0.,peak_lr=lr) for objective in ('ce','balanced_softmax') for lr in LR_GRID)
    result.extend(dict(objective='bcl_dropout',sampler='natural',onset=0.,**g) for g in BCL_GRID)
    return result


def loss_for(logits,features,label,counts,config,epoch,epochs):
    if config['objective'] in ('balanced_softmax','bcl_dropout'):
        loss=F.cross_entropy(logits+torch.log(torch.as_tensor(counts,device=logits.device,dtype=logits.dtype)),label)
    else:
        losses=F.cross_entropy(logits,label,reduction='none')
        if config['objective']=='class_balanced':
            weights=torch.as_tensor(importance_weights(counts,config['sampler']),device=logits.device)
            losses=losses*weights[label]
        loss=losses.mean()
    if config['objective']=='class_balanced' and config['coefficient'] and epoch>=int(config['onset']*epochs):
        loss=loss+config['coefficient']*equalization(features,label,len(counts))
    return loss


def train_adapter(x,y,fit,classes,config,seed,args,target,identity,stop_after=None):
    target.mkdir(parents=True,exist_ok=True); cp=target/'checkpoint.pt'
    model=v.Adapter(x.shape[1],classes,seed).to(args.device)
    peak_lr=config.get('peak_lr',args.lr)
    initial=common.state_hash(model.state_dict()); optimizer=torch.optim.SGD(model.parameters(),lr=peak_lr,momentum=.9,nesterov=True)
    counts=np.bincount(y[fit],minlength=classes); steps=math.ceil(len(fit)/args.batch_size); total=steps*args.epochs
    start=0; history=[]; elapsed=0.
    if cp.exists():
        ck=torch.load(cp,map_location=args.device,weights_only=False)
        require(ck['identity']==identity and ck['config']==config and ck['initial']==initial,'Checkpoint identity mismatch')
        require(0<=ck['completed_epochs']<=args.epochs and len(ck['history'])==ck['completed_epochs'],'Checkpoint budget mismatch')
        model.load_state_dict(ck['model']); optimizer.load_state_dict(ck['optimizer']); common.restore_rng(ck['rng'])
        start,history,elapsed=ck['completed_epochs'],ck['history'],ck['seconds']
    for epoch in range(start,args.epochs if stop_after is None else min(args.epochs,stop_after)):
        model.train(); began=time.perf_counter(); eligible=np.zeros(classes,dtype=np.int64); losses=[]
        for b,ids in enumerate(epoch_batches(fit,y,classes,seed,epoch,args.batch_size,config['sampler'])):
            batch=torch.as_tensor(np.array(x[ids]),device=args.device); label=torch.as_tensor(np.array(y[ids]),device=args.device)
            for group in optimizer.param_groups: group['lr']=common.learning_rate(epoch*steps+b,total,args.warmup_epochs*steps,peak_lr)
            optimizer.zero_grad(set_to_none=True); logits,features=model(batch)
            loss=loss_for(logits,features,label,counts,config,epoch,args.epochs)+.0005*nonimage.linear_l2(model)
            if config['objective']=='bcl_dropout':
                contrastive_batch=batch[:64]; contrastive_labels=label[:64]
                _,view1=model(contrastive_batch); _,view2=model(contrastive_batch)
                loss=loss+config['coefficient']*balanced_contrastive(torch.stack((view1,view2),dim=1),contrastive_labels,model.head.weight,config['temperature'])
            require(bool(torch.isfinite(loss)),'Nonfinite loss'); loss.backward()
            for p in model.parameters():
                if p.grad is not None:
                    norm=p.grad.norm(); require(bool(torch.isfinite(norm)),'Nonfinite gradient'); p.grad.mul_((1./(norm+1e-6)).clamp(max=1.))
            optimizer.step(); losses.append(float(loss.detach().cpu())); eligible+=np.bincount(y[ids],minlength=classes)>=2
        duration=time.perf_counter()-began; elapsed+=duration
        history.append(dict(epoch=epoch+1,optimizer_updates=(epoch+1)*steps,mean_loss=float(np.mean(losses)),seconds=duration,
        EQ_active=bool(config['objective']=='class_balanced' and config['coefficient'] and epoch>=int(config['onset']*args.epochs)),
        BCL_active=config['objective']=='bcl_dropout',BCL_contrastive_examples_per_update=min(64,args.batch_size) if config['objective']=='bcl_dropout' else 0,
        batches=steps,eligible_batches_per_class=eligible.tolist()))
        if (epoch+1)%args.checkpoint_every==0 or epoch+1==args.epochs or epoch+1==stop_after:
            common.save_torch(cp,dict(identity=identity,config=config,initial=initial,model=model.state_dict(),optimizer=optimizer.state_dict(),rng=common.rng_state(),completed_epochs=epoch+1,history=history,seconds=elapsed))
            write_json(target/'history.json',history)
    write_json(target/'history.json',history)
    info=dict(initial=initial,final=common.state_hash(model.state_dict()),completed_epochs=len(history),optimizer_updates=len(history)*steps,
        seconds=elapsed,config=config,importance_weights=importance_weights(counts,config['sampler']).tolist() if config['objective']=='class_balanced' else None,
        target_prior=(counts/counts.sum()).tolist() if config['objective']=='ce' else (np.ones(classes)/classes).tolist())
    return model,info


def load_inputs(args):
    cache=Path(args.features); m=v.read_json(cache/'manifest.json'); y=np.load(cache/'labels.npy',allow_pickle=False)
    for name,digest in m['artifacts'].items(): require(sha(cache/name)==digest,'Cache checksum differs')
    require(m['encoder_id']=='deit_tiny' and m['ntrain']==100000 and m['n_eval']==10000 and m['feature_dim']==192,'Unexpected official DeiT cache')
    x=np.load(cache/'features.npy',mmap_mode='r',allow_pickle=False)
    eligible,hygiene=v.split_helper.clean_training_indices(m,y[:m['ntrain']])
    require(x.shape==(len(y),192) and np.isfinite(x).all(),'Invalid features')
    return m,x,y,eligible,hygiene


def prepare(args):
    m,raw,y,eligible,hygiene=load_inputs(args); args.out.mkdir(parents=True,exist_ok=True)
    split=make_partition(y[:m['ntrain']],eligible,split_seed=args.split_seed,rho=args.rho)
    p=dict(version=1,source_sha256=sha(__file__),helper_sha256={'benchmark_vit_rebal.py':sha(Path(__file__).with_name('benchmark_vit_rebal.py')),
        'train_tiny_fullscale.py':sha(Path(__file__).with_name('train_tiny_fullscale.py')),'train_nonimage.py':sha(Path(__file__).with_name('train_nonimage.py'))},
        cache_manifest_sha256=sha(Path(args.features)/'manifest.json'),source_archive_sha256=m['archive_sha256'],hygiene=hygiene,
        settings={k:getattr(args,k) for k in ('seeds','split_seed','rho','epochs','warmup_epochs','batch_size','lr','head_steps','head_batch_size','device','threads','checkpoint_every')},
        holdouts_per_class=dict(tune=20,calibration=20,confirmation=100),grid=GRID,reference=REFERENCE,bcl_grid=BCL_GRID,control_lr_grid=LR_GRID,configurations=configurations(),
        objective='Same class-balanced expected cross-entropy across factorial samplers from epoch1, natural importance N/(C*n_c), balanced weight1; separate CE and BalancedSoftmax controls',
        batch_policy='Fixed full batches, replacement sampling, ceil(N/B) updates per epoch for every arm; paired initialization/dropout streams',
        selection='Global mean tuning macroF1 over10pairedseeds; equal6candidate budgets for each EQsampler, BCL, CE and BalancedSoftmax. CE uses calibration-selected LA before tuning-score selection. Exact ties choose smaller coefficient/onset, temperature or LR. Factorial off remains matched fixedLR.05',
        BCL='Class-averaged contrastive denominator, all-class classifier-weight prototypes, two independent adapter-dropout views of first64random batch examples/update, plus BalancedSoftmax; frozen-feature adaptation, not original image-augmentation/two-projection-branch reproduction',
        BCL_source='https://github.com/FlamieZhu/Balanced-Contrastive-Learning/blob/main/loss/contrastive.py; https://openaccess.thecvf.com/content/CVPR2022/html/Zhu_Balanced_Contrastive_Learning_for_Long-Tailed_Visual_Recognition_CVPR_2022_paper.html',
        confirmation_note='New seed-independent canonical dev holdout for this followup; rows were potentially used to train historical adapters; all new models restart from identical random adapter initialization',
        primary_family=['natural_EQ_effect','balanced_EQ_effect','sampling_effect_EQoff','factorial_interaction'],primary_endpoint='native_macro_f1',
        evaluation='Native plus2000-update natural/ROS heads, with calibration-only logit adjustment; no new SMOTE graphs or officialeval scores')
    p['fingerprint']=fingerprint(p)
    pp=args.out/'protocol.json'; sp=args.out/'partition.npz'
    if pp.exists():
        require(v.read_json(pp)['fingerprint']==p['fingerprint'],'Protocol changed; use distinct output')
        with np.load(sp,allow_pickle=False) as saved: require(all(np.array_equal(saved[k],a) for k,a in split.items()),'Partition changed')
        require(sha(sp)==v.read_json(args.out/'partition_identity.json')['sha256'],'Partition checksum differs')
    else:
        save_npz(sp,**split); write_json(pp,p); write_json(args.out/'partition_identity.json',dict(sha256=sha(sp),protocol=p['fingerprint'],fit_count=len(split['fit']),
            fit_counts=np.bincount(y[split['fit']],minlength=200).tolist(),actual_ratio=float(np.bincount(y[split['fit']],minlength=200).max()/np.bincount(y[split['fit']],minlength=200).min())))
    x,_=v.prepare_scaler(raw,split['fit'],args.out/'scaler',dict(protocol=p['fingerprint'],partition_sha256=sha(sp)))
    return p,m,x,y,split


def tune(args):
    p,m,x,y,s=prepare(args); grid=list(itertools.product(args.seeds,configurations()))
    for i,(seed,config) in enumerate(grid):
        if i%args.worker_count!=args.worker_index: continue
        name=configuration_name(config); target=args.out/'tuning'/f'seed_{seed}'/name; marker=target/'tune.json'
        identity=dict(protocol=p['fingerprint'],seed=seed,configuration=name)
        if marker.exists():
            r=v.read_json(marker); require(r['identity']==identity and sha(target/'tune_logits.npz')==r['predictions_sha256'] and sha(target/'checkpoint.pt')==r['checkpoint_sha256'],'Tuning artifact changed'); continue
        model,info=train_adapter(x,y,s['fit'],200,config,seed,args,target,identity)
        require(info['completed_epochs']==args.epochs,'Incomplete tuning training')
        logits,_=nonimage.encode(model,x,s['tune'],'tabular',args); raw_macro_f1=common.metrics(y[s['tune']],logits.argmax(1),s['class_rank'])['macro_f1']
        arrays=dict(logits=logits,labels=y[s['tune']]); adjustment=None; selected_logits=logits
        if config['objective']=='ce':
            cal_logits,_=nonimage.encode(model,x,s['calibration'],'tabular',args)
            arrays.update(calibration_logits=cal_logits,calibration_labels=y[s['calibration']])
            _,selected_logits,adjustment=nonimage.adjust(cal_logits,y[s['calibration']],logits,info['target_prior'])
        save_npz(target/'tune_logits.npz',**arrays)
        r=dict(identity=identity,training=info,macro_f1=common.metrics(y[s['tune']],selected_logits.argmax(1),s['class_rank'])['macro_f1'],raw_macro_f1=raw_macro_f1,adjustment=adjustment,
            predictions_sha256=sha(target/'tune_logits.npz'),checkpoint_sha256=sha(target/'checkpoint.pt'),history_sha256=sha(target/'history.json'))
        write_json(marker,r); print(json.dumps(dict(stage='tune_complete',seed=seed,configuration=name,macro_f1=r['macro_f1'])),flush=True)


def select(args):
    p,_,_,y,s=prepare(args); rows=[]
    for seed,config in itertools.product(args.seeds,configurations()):
        folder=args.out/'tuning'/f'seed_{seed}'/configuration_name(config); r=v.read_json(folder/'tune.json')
        require(r['identity']==dict(protocol=p['fingerprint'],seed=seed,configuration=configuration_name(config)),'Tuning identity changed')
        require(r['training']['completed_epochs']==args.epochs and r['training']['optimizer_updates']==args.epochs*math.ceil(len(s['fit'])/args.batch_size) and sha(folder/'checkpoint.pt')==r['checkpoint_sha256'] and sha(folder/'tune_logits.npz')==r['predictions_sha256'] and sha(folder/'history.json')==r['history_sha256'],'Incomplete tuning artifact')
        with np.load(folder/'tune_logits.npz',allow_pickle=False) as z:
            require(np.array_equal(z['labels'],y[s['tune']]),'Tuning labels/order differ')
            logits=z['logits']
            if config['objective']=='ce':
                require(np.array_equal(z['calibration_labels'],y[s['calibration']]),'Tuning calibration labels/order differ')
                _,logits,adj=nonimage.adjust(z['calibration_logits'],z['calibration_labels'],logits,r['training']['target_prior'])
                require(fingerprint(adj)==fingerprint(r['adjustment']),'Tuning calibration selection differs')
            actual=common.metrics(z['labels'],logits.argmax(1),s['class_rank'])['macro_f1']
            require(abs(actual-r['macro_f1'])<1e-12,'Tuning score differs from stored logits')
        rows.append(dict(seed=seed,config=config,macro_f1=r['macro_f1'],tune_json_sha256=sha(folder/'tune.json')))
    selected={}; candidates={}
    for sampler in ('natural','balanced'):
        candidates[sampler]=[dict(config=dict(objective='class_balanced',sampler=sampler,**g),mean_macro_f1=float(np.mean([r['macro_f1'] for r in rows if r['config']==dict(objective='class_balanced',sampler=sampler,**g)]))) for g in GRID]
        winner=sorted(candidates[sampler],key=lambda r:(-r['mean_macro_f1'],r['config']['coefficient'],r['config']['onset']))[0]
        selected[sampler]=winner['config']
    candidates['bcl_dropout']=[dict(config=dict(objective='bcl_dropout',sampler='natural',onset=0.,**g),mean_macro_f1=float(np.mean([r['macro_f1'] for r in rows if r['config']==dict(objective='bcl_dropout',sampler='natural',onset=0.,**g)]))) for g in BCL_GRID]
    selected['bcl_dropout']=sorted(candidates['bcl_dropout'],key=lambda r:(-r['mean_macro_f1'],r['config']['coefficient'],r['config']['temperature']))[0]['config']
    for objective in ('ce','balanced_softmax'):
        candidates[objective]=[dict(config=dict(objective=objective,sampler='natural',onset=0.,coefficient=0.,peak_lr=lr),mean_macro_f1=float(np.mean([r['macro_f1'] for r in rows if r['config']==dict(objective=objective,sampler='natural',onset=0.,coefficient=0.,peak_lr=lr)]))) for lr in LR_GRID]
        selected[objective]=sorted(candidates[objective],key=lambda r:(-r['mean_macro_f1'],r['config']['peak_lr']))[0]['config']
    result=dict(protocol=p['fingerprint'],completed_tuning_cells=len(rows),selected=selected,candidates=candidates,tuning_inventory=rows,
        confirmation_access_before_freeze=False,selection_created_utc=common.now())
    # Timestamp is excluded from identity; reruns must produce exactly the same decision/inventory.
    result['fingerprint']=fingerprint({k:a for k,a in result.items() if k!='selection_created_utc'})
    target=args.out/'selection.json'
    if target.exists(): require(v.read_json(target)['fingerprint']==result['fingerprint'],'Frozen selection changed')
    else: write_json(target,result)
    print(json.dumps(dict(stage='selection_frozen',fingerprint=result['fingerprint'],selected=selected)),flush=True)
    return result


def selected_arms(selection):
    result={}
    for sampler in ('natural','balanced'):
        result[sampler+'_off']=dict(objective='class_balanced',sampler=sampler,onset=0.,coefficient=0.)
        result[sampler+'_on']=selection['selected'][sampler]
        result[sampler+'_reference']=dict(objective='class_balanced',sampler=sampler,**REFERENCE)
    for objective in ('ce','balanced_softmax'):
        result[objective]=selection['selected'][objective]
        result[objective+'_reference']=dict(objective=objective,sampler='natural',onset=0.,coefficient=0.,peak_lr=.05)
    result['bcl_dropout']=selection['selected']['bcl_dropout']
    return result


def evaluate(args):
    selection=select(args); p,m,x,y,s=prepare(args)
    for i,seed in enumerate(args.seeds):
        if i%args.worker_count!=args.worker_index: continue
        cell=args.out/'confirmation'/f'seed_{seed}'; cell.mkdir(parents=True,exist_ok=True); final=cell/'metrics.json'
        if final.exists():
            r=v.read_json(final); verify_cell(r,cell,p,selection,args); continue
        predictions=dict(confirmation_labels=y[s['confirmation']],calibration_labels=y[s['calibration']],class_rank=s['class_rank'])
        metrics={}; training={}; views={}; calibrated={}
        for arm,config in selected_arms(selection).items():
            source=args.out/'tuning'/f'seed_{seed}'/configuration_name(config)
            info=v.read_json(source/'tune.json')['training']; identity=dict(protocol=p['fingerprint'],seed=seed,configuration=configuration_name(config))
            model,_=train_adapter(x,y,s['fit'],200,config,seed,args,source,identity)
            cl,cf=nonimage.encode(model,x,s['calibration'],'tabular',args); tl,tf=nonimage.encode(model,x,s['confirmation'],'tabular',args)
            _,ff=nonimage.encode(model,x,s['fit'],'tabular',args)
            training[arm]=info
            def record(name,cal_logits,confirm_logits,prior):
                predictions[name+'__calibration_logits']=cal_logits; predictions[name+'__confirmation_logits']=confirm_logits
                metrics[name]=common.metrics(y[s['confirmation']],confirm_logits.argmax(1),s['class_rank']); views[name]=dict(base=name,operation='stored')
                _,adjusted,adjustment=nonimage.adjust(cal_logits,y[s['calibration']],confirm_logits,prior)
                metrics[name+'_la']=common.metrics(y[s['confirmation']],adjusted.argmax(1),s['class_rank']); metrics[name+'_la']['adjustment']=adjustment
                calibrated[name]=adjustment; views[name+'_la']=dict(base=name,operation='logit_adjustment')
            record(arm+'__native',cl,tl,info['target_prior'])
            for kind in ('natural','ros'):
                headidentity=dict(protocol=p['fingerprint'],selection=selection['fingerprint'],seed=seed,arm=arm,features_sha256=hashlib.sha256(ff.tobytes()).hexdigest())
                head,hinfo=v.train_head(ff,y[s['fit']],None,kind,seed,args,200,cell/(arm+'__head_'+kind+'.pt'),headidentity)
                record(arm+'__'+kind,common.head_logits(head,cf,args),common.head_logits(head,tf,args),hinfo['training_prior']); training[arm+'__'+kind]=hinfo
            print(json.dumps(dict(stage='confirmation_arm_complete',seed=seed,arm=arm)),flush=True)
        save_npz(cell/'predictions.npz',**predictions)
        artifacts={f.name:sha(f) for f in cell.iterdir() if f.name!='metrics.json' and f.is_file()}
        r=dict(protocol=p['fingerprint'],selection=selection['fingerprint'],seed=seed,completed_utc=common.now(),metrics=metrics,training=training,prediction_views=views,artifacts=artifacts)
        verify_cell(r,cell,p,selection,args); write_json(final,r)


def verify_cell(r,cell,p,selection,args):
    require(r['protocol']==p['fingerprint'] and r['selection']==selection['fingerprint'],'Confirmation identity differs')
    expected={arm+'__'+kind+suffix for arm in selected_arms(selection) for kind in ('native','natural','ros') for suffix in ('','_la')}
    require(set(r['metrics'])==set(r['prediction_views'])==expected,'Missing confirmation readout')
    for name,digest in r['artifacts'].items(): require(sha(cell/name)==digest,'Confirmation artifact differs: '+name)
    for arm in selected_arms(selection): require(r['training'][arm]['completed_epochs']==args.epochs,'Incomplete adapter budget')
    for arm in selected_arms(selection):
        for kind in ('natural','ros'): require(r['training'][arm+'__'+kind]['optimizer_updates']==args.head_steps,'Incomplete head budget')


def paired(values,name):
    values=np.array(values,dtype=float); n=len(values); mean=float(values.mean()); sd=float(values.std(ddof=1))
    half=float(stats.t.ppf(.975,n-1)*sd/math.sqrt(n)); signs=np.array(list(itertools.product((-1.,1.),repeat=n)))
    ep=float(np.mean(np.abs((signs*values).mean(1))>=abs(mean)-1e-12))
    tp=float(stats.ttest_1samp(values,0).pvalue) if np.std(values)>0 else (1. if np.all(values==0) else 0.)
    return dict(name=name,differences=values.tolist(),mean=mean,sd=sd,ci95=[mean-half,mean+half],signflip_p=ep,t_p=tp)


def holm(rows,key):
    order=sorted(range(len(rows)),key=lambda i:rows[i][key]); maximum=0.
    for rank,i in enumerate(order): maximum=max(maximum,min(1.,(len(rows)-rank)*rows[i][key])); rows[i][key+'_holm']=maximum


def aggregate(args,verify=False):
    p,_,_,y,s=prepare(args); selection=select(args); cells=[]
    for seed in args.seeds:
        cell=args.out/'confirmation'/f'seed_{seed}'; r=v.read_json(cell/'metrics.json'); verify_cell(r,cell,p,selection,args)
        if verify:
            with np.load(cell/'predictions.npz',allow_pickle=False) as z:
                require(np.array_equal(z['confirmation_labels'],y[s['confirmation']]) and np.array_equal(z['calibration_labels'],y[s['calibration']]) and np.array_equal(z['class_rank'],s['class_rank']),'Prediction labels/order differ')
                for name,view in r['prediction_views'].items():
                    logits=z[view['base']+'__confirmation_logits']
                    if view['operation']=='logit_adjustment':
                        cal=z[view['base']+'__calibration_logits']; a=r['metrics'][name]['adjustment']; _,logits,actual=nonimage.adjust(cal,z['calibration_labels'],logits,a['prior'])
                        require(fingerprint(actual)==fingerprint(a),'Calibration selection differs')
                    actual=common.metrics(z['confirmation_labels'],logits.argmax(1),z['class_rank'])
                    for metric,value in actual.items(): require(np.allclose(value,r['metrics'][name][metric],rtol=0,atol=1e-12),'Stored metric differs: '+name+'/'+metric)
        cells.append(r)
    summary=[]
    for name in sorted(cells[0]['metrics']):
        metrics={}
        for endpoint in v.METRICS:
            vals=np.array([r['metrics'][name][endpoint] for r in cells]); mean=float(vals.mean()); sd=float(vals.std(ddof=1)); half=float(stats.t.ppf(.975,len(vals)-1)*sd/math.sqrt(len(vals)))
            metrics[endpoint]=dict(mean=mean,sd=sd,ci95=[mean-half,mean+half])
        summary.append(dict(method=name,n=len(cells),metrics=metrics))
    contrasts=[]
    for name,a,b in [('natural_EQ_effect','natural_on','natural_off'),('balanced_EQ_effect','balanced_on','balanced_off'),('sampling_effect_EQoff','balanced_off','natural_off')]:
        contrasts.append(paired([r['metrics'][a+'__native']['macro_f1']-r['metrics'][b+'__native']['macro_f1'] for r in cells],name))
    contrasts.append(paired([(r['metrics']['balanced_on__native']['macro_f1']-r['metrics']['balanced_off__native']['macro_f1'])-(r['metrics']['natural_on__native']['macro_f1']-r['metrics']['natural_off__native']['macro_f1']) for r in cells],'factorial_interaction'))
    holm(contrasts,'signflip_p'); holm(contrasts,'t_p')
    result=dict(status='verified_complete' if verify else 'complete',protocol=p['fingerprint'],selection=selection['fingerprint'],seeds=args.seeds,completed_tuning_cells=selection['completed_tuning_cells'],
        confirmation_cells=len(cells),readouts=sum(len(r['metrics']) for r in cells),summary=summary,paired_comparisons=contrasts,
        selection_frozen_before_confirmation=True,primary_family_size=4,official_evaluation_scored=False,
        verification='Numeric logits, labels, calibration selections, every scalar/per-class metric, artifact hashes and epoch/update budgets replayed; model states not re-executed' if verify else None)
    write_json(args.out/('verification.json' if verify else 'aggregate.json'),result)
    print(json.dumps(dict(stage=result['status'],confirmation_cells=len(cells),readouts=result['readouts'])),flush=True)
    return result


def pilot(args):
    p,_,x,y,s=prepare(args); config=dict(objective='class_balanced',sampler='balanced',onset=0.,coefficient=.05)
    identity=dict(protocol=p['fingerprint'],seed=args.seeds[0],configuration='train_only_pilot')
    began=time.perf_counter(); _,info=train_adapter(x,y,s['fit'],200,config,args.seeds[0],args,args.out/'pilot',identity,stop_after=2)
    write_json(args.out/'pilot.json',dict(training=info,wall_seconds=time.perf_counter()-began,heldout_access=False))
    print(json.dumps(dict(stage='train_only_pilot',training=info)),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('phase',choices=['prepare','pilot','tune','select','evaluate','aggregate','verify'])
    parser.add_argument('--features',type=Path,required=True); parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--seeds',nargs='+',type=int,default=list(range(10,20))); parser.add_argument('--split-seed',type=int,default=20261013); parser.add_argument('--rho',type=float,default=100)
    parser.add_argument('--epochs',type=int,default=50); parser.add_argument('--warmup-epochs',type=int,default=5); parser.add_argument('--batch-size',type=int,default=1024); parser.add_argument('--lr',type=float,default=.05)
    parser.add_argument('--head-steps',type=int,default=2000); parser.add_argument('--head-batch-size',type=int,default=2048); parser.add_argument('--head-checkpoint-every',type=int,default=500)
    parser.add_argument('--device',choices=['cpu','mps','cuda'],default='cpu'); parser.add_argument('--threads',type=int,default=4); parser.add_argument('--checkpoint-every',type=int,default=5)
    parser.add_argument('--worker-count',type=int,default=1); parser.add_argument('--worker-index',type=int,default=0)
    args=parser.parse_args()
    require(len(args.seeds)>=2 and len(set(args.seeds))==len(args.seeds) and len(args.seeds)<=20,'Invalid paired seeds')
    require(min(args.epochs,args.batch_size,args.head_steps,args.head_batch_size,args.threads,args.checkpoint_every)>0 and args.warmup_epochs>=0 and args.lr>0,'Invalid training budget')
    require(args.worker_count>0 and 0<=args.worker_index<args.worker_count,'Invalid worker assignment')
    torch.set_num_threads(args.threads); torch.use_deterministic_algorithms(True,warn_only=True)
    if args.phase=='prepare': prepare(args)
    elif args.phase=='pilot': pilot(args)
    elif args.phase=='tune': tune(args)
    elif args.phase=='select': select(args)
    elif args.phase=='evaluate': evaluate(args)
    else: aggregate(args,verify=args.phase=='verify')

if __name__=='__main__': main()
