#!/usr/bin/env python3
"""From-scratch Tiny ImageNet REBAL-F adaptation, explicitly a PyTorch port.

This is not a bitwise reproduction of the original TensorFlow implementation.
The port preserves the source ResNet32/objective/cRT structure, applies ECOD only
to the fit partition, and gives all trunks the same retained images. It adds
matched natural/ROS/SMOTE readouts and an isolated train-derived LA holdout.

  python scripts/train_tiny_fullscale.py prepare --archive tiny-imagenet-200.zip --cache-dir data/tiny_pixels
  python scripts/train_tiny_fullscale.py run --cache-dir data/tiny_pixels --out experiments/tiny_fullscale --device cuda

The default grid is 3 ratios x 5 seeds x 3 independently trained 200-epoch trunks.
Checkpoints resume interrupted epochs; completed cells require intact artifacts.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import io
import itertools
import json
import math
import os
import platform
import random
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import stats
from sklearn.metrics import f1_score, recall_score
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

VERSION = 1
TRUNKS = ('ce', 'cb', 'rebal')
TAUS = (0., .25, .5, .75, 1., 1.25, 1.5, 2.)
OFFICIAL_SHA = '6198c8ae015e2b3e007c7841da39ec069199b9aa3bfa943a462022fe5e43c821'
DEVIATIONS = [
    'PyTorch port, not a bitwise TensorFlow replication; backend kernels and BatchNorm running-variance semantics differ.',
    'PyTorch symmetric stride-2 convolution padding differs from TensorFlow SAME asymmetric padding on even-sized inputs.',
    'PyTorch SGD momentum stores gradient buffers; TensorFlow velocity includes the historical learning rate, so changing-LR trajectories differ.',
    'Native 64x64 Tiny ImageNet replaces CIFAR32; labels/ranks and explicit splits are saved.',
    'All trunks share fit-only ECOD-retained images; original framework filtered before its development split.',
    'Plain CE explicitly disables equalization; original bundled baseline inherited it.',
    'Matching initialization and batches; exact ceil(N/B) schedule replaces source floor/ceil mismatch.',
    'Balanced train-derived 50/class calibration holdout; not the original imbalanced 10-percent holdout.',
    'PyTorch affine-grid bilinear reflective translation and horizontal flip implement the source augmentation intent.',
    'Natural/ROS/SMOTE cRT heads have matched update budgets; original code only trained a SMOTE cRT head.',
    'cRT uses uniform sampling with replacement on each pool; the source Keras fit uses shuffled passes without replacement.',
    'Initial baseline feature SMOTE discarded by the source is omitted; final-trunk features alone train cRT.',
    'No cGAN: this is the no-generator REBAL-F variant, not full generator-augmented REBAL.',
]


def now(): return datetime.now(timezone.utc).isoformat()

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(b)
    return h.hexdigest()


def fingerprint(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    tmp.replace(path)


def save_torch(path, obj):
    tmp = Path(str(path) + '.tmp'); torch.save(obj, tmp); tmp.replace(path)


def save_npz(path, **arrays):
    with open(str(path) + '.tmp', 'wb') as f: np.savez_compressed(f, **arrays)
    Path(str(path) + '.tmp').replace(path)


def state_hash(state):
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        h.update(key.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def rng_state():
    return dict(torch=torch.get_rng_state(), numpy=np.random.get_state(), python=random.getstate(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                mps=torch.mps.get_rng_state() if torch.backends.mps.is_available() else None)


def restore_rng(state):
    torch.set_rng_state(state['torch'].cpu()); np.random.set_state(state['numpy']); random.setstate(state['python'])
    if state['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])
    if state.get('mps') is not None and torch.backends.mps.is_available(): torch.mps.set_rng_state(state['mps'].cpu())


def prepare(archive, cache_dir, expected_classes=200):
    archive, cache_dir = Path(archive), Path(cache_dir)
    archive_hash = sha(archive)
    if expected_classes == 200 and archive_hash != OFFICIAL_SHA:
        raise ValueError('Archive differs from the verified official Tiny ImageNet archive')
    cache_dir.mkdir(parents=True, exist_ok=True)
    if (cache_dir / 'manifest.json').exists():
        old = json.loads((cache_dir / 'manifest.json').read_text())
        if old['archive_sha256'] != archive_hash: raise ValueError('Existing cache belongs to another archive')
        load_cache(cache_dir)
        return old
    prefix = 'tiny-imagenet-200/'
    with zipfile.ZipFile(archive) as z:
        classes = sorted(z.read(prefix + 'wnids.txt').decode().split())
        if len(classes) != expected_classes or len(set(classes)) != expected_classes: raise ValueError('Class inventory mismatch')
        ci = {c: i for i, c in enumerate(classes)}
        tr = sorted(n for n in z.namelist() if n.startswith(prefix+'train/') and n.endswith('.JPEG'))
        ev = sorted(n for n in z.namelist() if n.startswith(prefix+'val/images/') and n.endswith('.JPEG'))
        annotations = dict(line.split('\t')[:2] for line in z.read(prefix+'val/val_annotations.txt').decode().splitlines())
        names = tr + ev
        labels = np.asarray([ci[n.split('/')[2]] for n in tr] + [ci[annotations[Path(n).name]] for n in ev], dtype=np.int64)
        if expected_classes == 200 and (len(tr) != 100000 or len(ev) != 10000): raise ValueError('Official dataset size mismatch')
        images = np.lib.format.open_memmap(cache_dir/'images.npy.tmp', mode='w+', dtype=np.uint8, shape=(len(names),64,64,3))
        digests = []
        for i, name in enumerate(names):
            with Image.open(io.BytesIO(z.read(name))) as im: a = np.asarray(im.convert('RGB'))
            if a.shape != (64,64,3): raise ValueError(f'Unexpected image shape {name}: {a.shape}')
            images[i] = a; digests.append(hashlib.sha256(a.tobytes()).hexdigest())
            if (i+1) % 10000 == 0: print(f'prepared {i+1}/{len(names)} images', flush=True)
        images.flush(); del images
    (cache_dir/'images.npy.tmp').replace(cache_dir/'images.npy')
    np.save(cache_dir/'labels.npy', labels)
    manifest = dict(created_utc=now(), source_url='https://cs231n.stanford.edu/tiny-imagenet-200.zip',
                    archive_sha256=archive_hash, classes=classes, train_count=len(tr), eval_count=len(ev),
                    files=names, pixel_sha256=digests, image_shape=[64,64,3],
                    files_sha256={n:sha(cache_dir/n) for n in ('images.npy','labels.npy')})
    write_json(cache_dir/'manifest.json', manifest)
    return manifest


def load_cache(cache_dir):
    cache_dir = Path(cache_dir); manifest = json.loads((cache_dir/'manifest.json').read_text())
    for name, digest in manifest['files_sha256'].items():
        if sha(cache_dir/name) != digest: raise ValueError(f'Cache checksum mismatch: {name}')
    images = np.load(cache_dir/'images.npy', mmap_mode='r'); labels = np.load(cache_dir/'labels.npy')
    n = manifest['train_count'] + manifest['eval_count']
    if images.shape != (n,64,64,3) or labels.shape != (n,): raise ValueError('Cache dimensions do not match manifest')
    if len(manifest['files']) != n or len(set(manifest['files'])) != n or len(manifest['pixel_sha256']) != n:
        raise ValueError('Manifest identity mismatch')
    return images, labels, manifest


def make_split(labels, manifest, seed, rho, holdout):
    ntrain = manifest['train_count']; c = len(manifest['classes'])
    hashes = manifest['pixel_sha256']; evaluation = set(hashes[ntrain:]); seen = set()
    eligible, overlaps, duplicates = [], [], []
    for i, h in enumerate(hashes[:ntrain]):
        if h in evaluation: overlaps.append(i)
        elif h in seen: duplicates.append(i)
        else: eligible.append(i); seen.add(h)
    eligible = np.asarray(eligible, dtype=np.int64)
    if not np.array_equal(np.unique(labels[eligible]), np.arange(c)): raise ValueError('Missing class after deduplication')
    max_count = int(np.bincount(labels[eligible], minlength=c).min() - holdout)
    if max_count < 2 or rho < 1 or holdout < 1: raise ValueError('Invalid split parameters')
    rank_order = np.random.default_rng(np.random.SeedSequence([seed,202])).permutation(c)
    rank = np.empty(c,dtype=np.int64); rank[rank_order] = np.arange(c)
    counts = np.floor(max_count * float(rho)**(-rank/(c-1))).astype(np.int64)
    if counts.min() < 2: raise ValueError('Each fit class must retain >=2 examples for SMOTE')
    hr = np.random.default_rng(np.random.SeedSequence([seed,101])); tr = np.random.default_rng(np.random.SeedSequence([seed,303]))
    fit, dev = [], []
    for k in range(c):
        order = hr.permutation(eligible[labels[eligible] == k]); dev.extend(order[:holdout])
        fit.extend(tr.permutation(order[holdout:])[:counts[k]])
    fit = np.asarray(fit,dtype=np.int64); dev = np.asarray(dev,dtype=np.int64)
    ev = np.arange(ntrain,len(labels),dtype=np.int64)
    assert not (set(hashes[i] for i in fit) & set(hashes[i] for i in dev))
    assert not (set(hashes[i] for i in fit) & evaluation)
    return dict(fit_before_ecod=fit, holdout=dev, evaluation=ev, class_rank=rank,
                requested_counts=counts, excluded_overlap=np.asarray(overlaps,dtype=np.int64),
                excluded_duplicate=np.asarray(duplicates,dtype=np.int64))


def ecod_filter(images, labels, indices, enabled=True):
    if not enabled: return indices.copy(), np.empty(0,dtype=np.int64)
    from pyod.models.ecod import ECOD
    keep, excluded = [], []
    for k in np.unique(labels[indices]):
        ids = indices[labels[indices] == k]
        if len(ids) < 10: keep.extend(ids); continue
        # Source uses flattened pixels mapped to[-1,1]; affine monotonic scaling preserves ECOD ranks.
        flat = np.asarray(images[ids],dtype=np.float32).reshape(len(ids),-1) / 127.5 - 1.
        detector = ECOD(contamination=.05,n_jobs=1); detector.fit(flat)
        keep.extend(ids[detector.labels_ == 0]); excluded.extend(ids[detector.labels_ != 0])
    return np.asarray(keep,dtype=np.int64), np.asarray(excluded,dtype=np.int64)


class PixelDataset(Dataset):
    def __init__(self, images, labels, indices): self.images,self.labels,self.indices=images,labels,np.asarray(indices)
    def __len__(self): return len(self.indices)
    def __getitem__(self,i):
        j = int(self.indices[i]); x = torch.from_numpy(np.array(self.images[j],copy=True)).permute(2,0,1).float().div_(255.)
        return x, int(self.labels[j])


def loader(images, labels, indices, batch, workers, shuffle=False, seed=0):
    return DataLoader(PixelDataset(images,labels,indices),batch_size=batch,shuffle=shuffle,num_workers=workers,
                      generator=torch.Generator().manual_seed(seed),persistent_workers=workers>0,pin_memory=False)


class Block(nn.Module):
    def __init__(self, incoming, outgoing, stride=1):
        super().__init__(); self.conv1=nn.Conv2d(incoming,outgoing,3,stride,1,bias=False)
        self.bn1=nn.BatchNorm2d(outgoing,eps=.001,momentum=.01)
        self.conv2=nn.Conv2d(outgoing,outgoing,3,1,1,bias=False); self.bn2=nn.BatchNorm2d(outgoing,eps=.001,momentum=.01)
        self.shortcut=nn.Identity() if stride==1 and incoming==outgoing else nn.Sequential(
            nn.Conv2d(incoming,outgoing,1,stride,bias=False),nn.BatchNorm2d(outgoing,eps=.001,momentum=.01))
    def forward(self,x): return F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x)))))+self.shortcut(x))


class ResNet32(nn.Module):
    def __init__(self, classes, blocks=5):
        super().__init__(); self.conv=nn.Conv2d(3,16,3,1,1,bias=False); self.bn=nn.BatchNorm2d(16,eps=.001,momentum=.01)
        seq=[]; incoming=16
        for stage,width in enumerate((16,32,64)):
            for i in range(blocks):
                stride=2 if stage>0 and i==0 else 1; seq.append(Block(incoming,width,stride)); incoming=width
        self.blocks=nn.Sequential(*seq); self.feat=nn.Linear(64,128); self.dropout=nn.Dropout(.3); self.logits=nn.Linear(128,classes)
        self.apply(initialize)
    def forward(self,x):
        x=self.blocks(F.relu(self.bn(self.conv(x)))); x=x.mean((2,3)); feature=self.dropout(F.relu(self.feat(x)))
        return self.logits(feature),feature


def initialize(module):
    if isinstance(module,(nn.Conv2d,nn.Linear)):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None: nn.init.zeros_(module.bias)
    elif isinstance(module,nn.BatchNorm2d): nn.init.ones_(module.weight); nn.init.zeros_(module.bias)


def augment(x, generator):
    n=len(x); flip=torch.rand(n,generator=generator)<.5
    shift=(torch.rand(n,2,generator=generator)*2.-1.)*.2
    theta=torch.zeros(n,2,3); theta[:,0,0]=torch.where(flip,-1.,1.); theta[:,1,1]=1.; theta[:,:,2]=shift
    grid=F.affine_grid(theta.to(x.device),x.shape,align_corners=False)
    return F.grid_sample(x,grid,mode='bilinear',padding_mode='reflection',align_corners=False)


def equalization(features,labels,classes):
    f=F.normalize(features,p=2,dim=1); counts=torch.bincount(labels,minlength=classes)
    present=torch.nonzero(counts>=2).flatten()
    if len(present)==0: return features.sum()*0.
    sums=torch.zeros(classes,f.shape[1],device=f.device,dtype=f.dtype).index_add_(0,labels,f)
    centers=sums/counts.clamp_min(1).to(f.dtype)[:,None]
    spread=torch.zeros(classes,device=f.device,dtype=f.dtype).index_add_(0,labels,(f-centers[labels]).square().sum(1))
    within=(spread[present]/counts[present]).mean()
    if len(present)<2: return within
    selected=centers[present]; d2=(selected[:,None,:]-selected[None,:,:]).square().sum(2)
    between=d2.sum()/(len(present)*(len(present)-1))
    return within-between


def kernel_l2(model):
    return sum((m.weight.square().sum() for m in model.modules() if isinstance(m,(nn.Conv2d,nn.Linear))),start=torch.zeros((),device=next(model.parameters()).device))


def cb_weights(counts):
    values=(1.-.999)/(1.-np.power(.999,np.asarray(counts,dtype=np.float64)))
    return (values/values.mean()).astype(np.float32)


def learning_rate(step,total,warmup,peak):
    if step < warmup: return peak * step / max(1,warmup)
    return peak * .5 * (1.+math.cos(math.pi*min(1.,(step-warmup)/max(1,total-warmup))))


def predict(model, images, labels, indices, args):
    model.eval(); logits=[]; features=[]
    with torch.inference_mode():
        for x,_ in loader(images,labels,indices,args.batch_size,args.workers):
            l,f=model(x.to(args.device)); logits.append(l.cpu().numpy()); features.append(f.cpu().numpy())
    l,f=np.concatenate(logits),np.concatenate(features)
    if not np.isfinite(l).all() or not np.isfinite(f).all(): raise ValueError('Nonfinite model outputs')
    return l,f


def train_trunk(method,images,labels,split,args,cell_dir,identity):
    path=cell_dir/method; path.mkdir(parents=True,exist_ok=True); checkpoint=path/'checkpoint.pt'
    seed=identity['seed']; set_seed(seed); classes=len(split['class_rank']); model=ResNet32(classes,args.blocks).to(args.device)
    initial_sha=state_hash(model.state_dict()); optimizer=torch.optim.SGD(model.parameters(),lr=args.lr,momentum=.9,nesterov=True)
    counts=np.bincount(labels[split['fit']],minlength=classes); weights=torch.tensor(cb_weights(counts),device=args.device)
    steps=math.ceil(len(split['fit'])/args.batch_size); total=steps*args.epochs; warmup=min(args.warmup_epochs,args.epochs)*steps
    start_epoch=0; history=[]; elapsed=0.
    if checkpoint.exists():
        ck=torch.load(checkpoint,map_location=args.device,weights_only=False)
        if ck['identity']!=identity or ck['method']!=method or ck['initial_state_sha256']!=initial_sha: raise ValueError('Checkpoint identity mismatch')
        model.load_state_dict(ck['model']); optimizer.load_state_dict(ck['optimizer']); restore_rng(ck['rng'])
        start_epoch=ck['completed_epochs']; history=ck['history']; elapsed=ck['training_seconds']
    defer=int(.8*args.epochs)
    for epoch in range(start_epoch,args.epochs):
        model.train(); aug_rng=torch.Generator().manual_seed(seed*1000003+epoch+4141)
        data=loader(images,labels,split['fit'],args.batch_size,args.workers,True,seed*1000003+epoch+5151)
        started=time.perf_counter(); sumloss=0.; seen=0
        for batch,(x,y) in enumerate(data):
            x=augment(x.to(args.device),aug_rng); y=y.to(args.device); optimizer.zero_grad(set_to_none=True)
            step=epoch*steps+batch; lr=learning_rate(step,total,warmup,args.lr)
            for group in optimizer.param_groups: group['lr']=lr
            logits,features=model(x); per_sample=F.cross_entropy(logits,y,reduction='none')
            if method in ('cb','rebal') and epoch>=defer: per_sample=per_sample*weights[y]
            loss=per_sample.mean()+.0005*kernel_l2(model)
            if method=='rebal' and epoch>=defer: loss=loss+.05*equalization(features,y,classes)
            if not torch.isfinite(loss): raise RuntimeError(f'Nonfinite loss: {method},epoch{epoch},batch{batch}')
            loss.backward()
            # tf.keras optimizer clipnorm clips each gradient tensor separately.
            gradients=[p.grad for p in model.parameters() if p.grad is not None]
            norms=torch.stack([g.norm(2) for g in gradients])
            if not torch.isfinite(norms).all(): raise RuntimeError('Nonfinite gradient norm')
            for gradient,norm in zip(gradients,norms): gradient.mul_((1./(norm+1e-6)).clamp(max=1.))
            optimizer.step(); sumloss+=float(loss.detach().cpu())*len(y); seen+=len(y)
        if args.device=='cuda': torch.cuda.synchronize()
        elif args.device=='mps': torch.mps.synchronize()
        duration=time.perf_counter()-started; elapsed+=duration
        row=dict(epoch=epoch+1,optimizer_updates=(epoch+1)*steps,training_loss=sumloss/seen,seconds=duration,images_per_second=seen/duration,final_batch_lr=lr)
        history.append(row); print(json.dumps(dict(seed=seed,rho=identity['rho'],trunk=method,**row)),flush=True)
        if (epoch+1)%args.checkpoint_every==0 or epoch+1==args.epochs:
            save_torch(checkpoint,dict(identity=identity,method=method,initial_state_sha256=initial_sha,model=model.state_dict(),
                        optimizer=optimizer.state_dict(),rng=rng_state(),completed_epochs=epoch+1,history=history,training_seconds=elapsed))
            write_json(path/'training_history.json',history)
    return model,dict(initial_state_sha256=initial_sha,completed_epochs=args.epochs,optimizer_updates=total,training_seconds=elapsed,
                      checkpoint_sha256=sha(checkpoint),fit_counts=counts.tolist())


def metrics(y,pred,rank):
    c=len(rank); pc=f1_score(y,pred,labels=np.arange(c),average=None,zero_division=0)
    rc=recall_score(y,pred,labels=np.arange(c),average=None,zero_division=0)
    out=dict(accuracy=float(np.mean(y==pred)),macro_f1=float(pc.mean()),macro_recall=float(rc.mean()),worst_f1=float(pc.min()),
             per_class_f1=pc.tolist(),per_class_recall=rc.tolist())
    for name,mask in [('head',rank<c//3),('mid',(rank>=c//3)&(rank<2*c//3)),('tail',rank>=2*c//3)]:
        out[name+'_f1']=float(pc[mask].mean()); out[name+'_recall']=float(rc[mask].mean())
    out['head_tail_f1_gap']=out['head_f1']-out['tail_f1']; return out


def choose_tau(dev_logits,dev_labels,counts):
    prior=np.log(counts/counts.sum()).astype(np.float32); curve=[]
    for tau in TAUS:
        pred=(dev_logits-tau*prior).argmax(1)
        score=float(recall_score(dev_labels,pred,labels=np.arange(len(counts)),average='macro',zero_division=0))
        curve.append(dict(tau=tau,macro_recall=score))
    best=max(curve,key=lambda v:v['macro_recall']); return best['tau'],curve,prior


def train_crt(features,labels,kind,seed,args,classes):
    from imblearn.over_sampling import RandomOverSampler,SMOTE
    counts=np.bincount(labels,minlength=classes); target=classes*int(counts.max())
    meta=dict(method=kind,real_examples=len(labels),target_per_class=int(counts.max()),feature_geometry='raw Euclidean')
    if kind=='ros':
        sampler=RandomOverSampler(random_state=seed); x,y=sampler.fit_resample(features,labels); indices=sampler.sample_indices_
    elif kind=='smote':
        k=min(5,int(counts.min())-1)
        if k<1: raise ValueError('SMOTE needs >=2 samples in every class')
        sampler=SMOTE(random_state=seed,k_neighbors=k); x,y=sampler.fit_resample(features,labels); indices=None; meta['k_neighbors']=k
    else: x,y,indices=features.copy(),labels.copy(),np.arange(len(labels))
    set_seed(seed+90909); head=nn.Linear(features.shape[1],classes).to(args.device); initialize(head)
    initial=state_hash(head.state_dict()); optimizer=torch.optim.SGD(head.parameters(),lr=.01,momentum=.9)
    x=torch.as_tensor(x,dtype=torch.float32,device=args.device); y=torch.as_tensor(y,dtype=torch.long,device=args.device)
    # A fixed total batch budget avoids giving SMOTE/ROS more updates than the natural control.
    steps=math.ceil(target/args.crt_batch_size); generator=torch.Generator().manual_seed(seed+60606)
    losses=[]; start=time.perf_counter()
    for step in range(args.crt_epochs*steps):
        ids=torch.randint(len(y),(args.crt_batch_size,),generator=generator).to(args.device)
        optimizer.zero_grad(set_to_none=True); loss=F.cross_entropy(head(x[ids]),y[ids])+.0005*head.weight.square().sum()
        if not torch.isfinite(loss): raise RuntimeError('Nonfinite cRT loss')
        loss.backward(); optimizer.step()
        if step==0 or (step+1)%100==0 or step+1==args.crt_epochs*steps: losses.append(dict(update=step+1,loss=float(loss.detach().cpu())))
    meta.update(pool_examples=len(y),optimizer_updates=args.crt_epochs*steps,initial_state_sha256=initial,seconds=time.perf_counter()-start,losses=losses,
                sampling='uniform with replacement, identical index RNG stream across controls')
    return head,meta,indices


def head_logits(head,features,args):
    head.eval(); out=[]
    with torch.inference_mode():
        for i in range(0,len(features),2048): out.append(head(torch.as_tensor(features[i:i+2048],device=args.device)).cpu().numpy())
    return np.concatenate(out)


def ncm_logits(features,labels,query,classes):
    f=features/np.maximum(np.linalg.norm(features,axis=1,keepdims=True),1e-8)
    centers=np.stack([f[labels==k].mean(0) for k in range(classes)]); centers/=np.maximum(np.linalg.norm(centers,axis=1,keepdims=True),1e-8)
    q=query/np.maximum(np.linalg.norm(query,axis=1,keepdims=True),1e-8); return q@centers.T


def validate_cell(result, directory, args, protocol, seed, rho):
    expected=dict(protocol_fingerprint=protocol['fingerprint'],seed=seed,rho=rho,split_sha256=sha(directory/'split.npz'))
    if result.get('identity')!=expected: raise ValueError('Existing completed cell identity mismatch')
    methods=set(TRUNKS)|{m+'_ncm' for m in TRUNKS}|{'ce_la','rebal_crt_smote_la'}|{m+'_crt_'+h for m in TRUNKS for h in ('natural','ros','smote')}
    if set(result.get('metrics',{}))!=methods or set(result.get('trunks',{}))!=set(TRUNKS): raise ValueError('Incomplete method inventory')
    for record in result['trunks'].values():
        if record['completed_epochs']!=args.epochs or record['optimizer_updates']!=math.ceil(result['data']['fit']/args.batch_size)*args.epochs:
            raise ValueError('Incomplete backbone budget')
    if len({r['initial_state_sha256'] for r in result['trunks'].values()})!=1: raise ValueError('Unmatched trunk initialization')
    for method,record in result['metrics'].items():
        numeric=[value for value in record.values() if isinstance(value,(float,int))]
        numeric.extend(record['per_class_f1']); numeric.extend(record['per_class_recall'])
        if not np.isfinite(numeric).all(): raise ValueError('Nonfinite metrics')
        if '_crt_' in method and not method.endswith('_la'):
            expected_updates=args.crt_epochs*math.ceil(len(result['data']['post_ecod_counts'])*max(result['data']['post_ecod_counts'])/args.crt_batch_size)
            if record['training']['optimizer_updates']!=expected_updates: raise ValueError('Incomplete cRT budget')
    required={'split.npz','split_identity.json','predictions.npz'}|{f'{m}/checkpoint.pt' for m in TRUNKS}|{f'{m}/training_history.json' for m in TRUNKS}|{f'{m}/crt_{h}.pt' for m in TRUNKS for h in ('natural','ros','smote')}
    if set(result.get('artifacts',{}))!=required: raise ValueError('Incomplete artifact inventory')
    for name,digest in result['artifacts'].items():
        if sha(directory/name)!=digest: raise ValueError(f'Completed cell artifact mismatch: {name}')


def run_cell(images,labels,manifest,args,protocol,seed,rho):
    out=args.out/f'rho_{rho:g}'/f'seed_{seed}'; out.mkdir(parents=True,exist_ok=True)
    identity=dict(protocol_fingerprint=protocol['fingerprint'],seed=seed,rho=rho)
    result_path=out/'metrics.json'
    if result_path.exists():
        existing=json.loads(result_path.read_text())
        validate_cell(existing,out,args,protocol,seed,rho)
        print(f'verified completed cell {out}',flush=True); return existing
    split_path=out/'split.npz'; split_meta=out/'split_identity.json'
    if split_path.exists():
        if json.loads(split_meta.read_text())!=dict(identity,split_sha256=sha(split_path)): raise ValueError('Split identity mismatch')
        with np.load(split_path) as z: split={k:z[k] for k in z.files}
    else:
        split=make_split(labels,manifest,seed,rho,args.holdout_per_class)
        retained,excluded=ecod_filter(images,labels,split['fit_before_ecod'],not args.no_ecod)
        split.update(fit=retained,excluded_ecod=excluded,post_ecod_counts=np.bincount(labels[retained],minlength=len(split['class_rank'])))
        if split['post_ecod_counts'].min()<2: raise ValueError('ECOD left fewer than2fit examples in a class')
        save_npz(split_path,**split); write_json(split_meta,dict(identity,split_sha256=sha(split_path)))
    split_digest=sha(split_path); identity=dict(identity,split_sha256=split_digest)
    results={}; records={}; arrays={}; classes=len(split['class_rank']); rank=split['class_rank']; counts=split['post_ecod_counts']
    ev_y,dev_y,fit_y=labels[split['evaluation']],labels[split['holdout']],labels[split['fit']]
    arrays.update(eval_labels=ev_y,holdout_labels=dev_y,fit_labels=fit_y,class_rank=rank)
    artifacts=['split.npz','split_identity.json']
    for method in TRUNKS:
        model,record=train_trunk(method,images,labels,split,args,out,identity); records[method]=record
        fit_logits,fit_features=predict(model,images,labels,split['fit'],args)
        dev_logits,dev_features=predict(model,images,labels,split['holdout'],args)
        eval_logits,eval_features=predict(model,images,labels,split['evaluation'],args)
        def register(name,el,dl):
            arrays[name+'__eval_logits']=el; arrays[name+'__holdout_logits']=dl
            results[name]=metrics(ev_y,el.argmax(1),rank)
        register(method,eval_logits,dev_logits)
        register(method+'_ncm',ncm_logits(fit_features,fit_y,eval_features,classes),ncm_logits(fit_features,fit_y,dev_features,classes))
        if method=='ce':
            tau,curve,prior=choose_tau(dev_logits,dev_y,counts); register('ce_la',eval_logits-tau*prior,dev_logits-tau*prior)
            results['ce_la']['adjustment']=dict(tau=tau,curve=curve,log_prior=prior.tolist())
        if method in TRUNKS:
            for kind in ('natural','ros','smote'):
                head,meta,sample_indices=train_crt(fit_features,fit_y,kind,seed,args,classes)
                name=method+'_crt_'+kind; el=head_logits(head,eval_features,args); dl=head_logits(head,dev_features,args)
                register(name,el,dl); results[name]['training']=meta
                head_path=out/method/f'crt_{kind}.pt'; save_torch(head_path,dict(identity=identity,method=name,model=head.state_dict(),training=meta))
                artifacts.append(str(head_path.relative_to(out)))
                if sample_indices is not None: arrays[name+'__sample_indices']=sample_indices
                if method=='rebal' and kind=='smote':
                    tau,curve,prior=choose_tau(dl,dev_y,counts); register('rebal_crt_smote_la',el-tau*prior,dl-tau*prior)
                    results['rebal_crt_smote_la']['adjustment']=dict(tau=tau,curve=curve,log_prior=prior.tolist())
                del head
        artifacts.extend([f'{method}/checkpoint.pt',f'{method}/training_history.json']); del model
    if len(set(v['initial_state_sha256'] for v in records.values()))!=1: raise RuntimeError('Trunks did not share initialization')
    save_npz(out/'predictions.npz',**arrays); artifacts.append('predictions.npz')
    result=dict(identity=identity,completed_utc=now(),scope='from-scratch controlled PyTorch REBAL-F adaptation',
                protocol=protocol['fingerprint'],trunks=records,metrics=results,
                data=dict(fit=len(split['fit']),holdout=len(dev_y),evaluation=len(ev_y),post_ecod_counts=counts.tolist(),
                          nominal_rho=rho,actual_rho=float(counts.max()/counts.min()),ecod_excluded=len(split['excluded_ecod'])),
                artifacts={n:sha(out/n) for n in artifacts})
    write_json(result_path,result); return result


def aggregate(args,protocol):
    rows=[]; grouped={}
    for rho in args.rhos:
        for seed in args.seeds:
            path=args.out/f'rho_{rho:g}'/f'seed_{seed}'/'metrics.json'
            if not path.exists(): continue
            r=json.loads(path.read_text())
            validate_cell(r,path.parent,args,protocol,seed,rho)
            for name,m in r['metrics'].items():
                scalars={k:v for k,v in m.items() if isinstance(v,(int,float))}
                row=dict(rho=rho,seed=seed,method=name,**scalars); rows.append(row); grouped.setdefault((rho,name),[]).append(row)
    summary=[]
    for (rho,name),group in sorted(grouped.items()):
        entry=dict(rho=rho,method=name,n=len(group),seeds=[g['seed'] for g in group],metrics={})
        for k in group[0]:
            if k in ('rho','seed','method'): continue
            x=np.asarray([g[k] for g in group],dtype=float); sd=float(x.std(ddof=1)) if len(x)>1 else None
            half=float(stats.t.ppf(.975,len(x)-1)*sd/math.sqrt(len(x))) if len(x)>1 else None
            entry['metrics'][k]=dict(mean=float(x.mean()),sd=sd,ci95=None if half is None else [float(x.mean()-half),float(x.mean()+half)])
        summary.append(entry)
    comparisons=[]
    for rho in args.rhos:
        for a,b in [('rebal_crt_smote','ce_crt_smote'),('rebal_crt_smote','cb_crt_smote'),('rebal_crt_smote','rebal_crt_ros')]:
            da={g['seed']:g['tail_f1'] for g in grouped.get((rho,a),[])}; db={g['seed']:g['tail_f1'] for g in grouped.get((rho,b),[])}
            seeds=sorted(set(da)&set(db)); d=np.asarray([da[s]-db[s] for s in seeds])
            if len(d)!=len(args.seeds): continue
            n=len(d); mean=float(d.mean()); sd=float(d.std(ddof=1)) if n>1 else None
            signs=np.asarray(list(itertools.product((-1.,1.),repeat=n))); exact=float(np.mean(np.abs((signs*d).mean(1))>=abs(mean)-1e-12))
            half=float(stats.t.ppf(.975,n-1)*sd/math.sqrt(n)) if n>1 else None
            p=float(stats.ttest_1samp(d,0).pvalue) if n>1 and np.any(d!=d[0]) else (1. if np.all(d==0) else None)
            comparisons.append(dict(rho=rho,a=a,b=b,endpoint='tail_f1',seeds=seeds,differences=d.tolist(),mean_difference=mean,
                                    ci95=None if half is None else [mean-half,mean+half],t_pvalue=p,signflip_pvalue=exact))
    # Correct only after the whole prespecified family is available.
    family_size=3*len(args.rhos)
    if len(comparisons)==family_size:
        for source,dest in [('signflip_pvalue','signflip_holm'),('t_pvalue','t_holm')]:
            ordered=sorted(range(len(comparisons)),key=lambda i:comparisons[i][source] if comparisons[i][source] is not None else 1.)
            running=0.
            for rank,i in enumerate(ordered):
                p=comparisons[i][source]; running=max(running,min(1.,(family_size-rank)*(p if p is not None else 1.)))
                comparisons[i][dest]=running if p is not None else None
    result=dict(created_utc=now(),protocol_fingerprint=protocol['fingerprint'],completed_evaluations=len(rows),
                expected_cells=len(args.rhos)*len(args.seeds),summary=summary,paired_comparisons=comparisons,rows=rows,
                multiplicity_note='Holm across3comparisons timesallratios, onlycomputedwhenfamilycomplete; n5exacttwo-sidedminimum p.0625.')
    write_json(args.out/'aggregate.json',result); return result


def make_protocol(args,manifest):
    import importlib.metadata
    versions={name:importlib.metadata.version(name) for name in ('torch','numpy','scipy','scikit-learn','imbalanced-learn','Pillow')}
    if not args.no_ecod: versions['pyod']=importlib.metadata.version('pyod')
    settings={k:v for k,v in vars(args).items() if k not in ('out','cache_dir','command','aggregate_only','worker_index','worker_count','defer_aggregate')}
    p=dict(version=VERSION,scope='from-scratch PyTorch REBAL-F adaptation',source_sha256=sha(__file__),
           cache_manifest_sha256=sha(args.cache_dir/'manifest.json'),archive_sha256=manifest['archive_sha256'],
           settings=settings,software=versions,platform=platform.platform(),trunks=list(TRUNKS),deviations=DEVIATIONS,
           optimizer=dict(name='SGD',momentum=.9,nesterov=True,kernel_l2=.0005,per_tensor_clipnorm=1.,
                          cb_beta=.999,eq_lambda=.05,defer_fraction=.8),
           crt=dict(epochs=args.crt_epochs,lr=.01,momentum=.9,nesterov=False,kernel_l2=.0005,
                    steps_per_epoch='ceil(number_of_classes * largest_post_ECOD_fit_count / crt_batch_size)',
                    sampling='uniform with replacement on natural/ROS/SMOTE pool; matched updates and initialization'),
           primary_endpoint='tail macro F1',primary_comparisons=['rebal_crt_smote minus ce_crt_smote','rebal_crt_smote minus cb_crt_smote','rebal_crt_smote minus rebal_crt_ros'],
           checkpoint_selection='final fixed epoch, no eval/holdout checkpoint selection',LA_tau_grid=list(TAUS),
           calibration='balanced train-derived holdout; provides extra labeled calibration examples beyond long-tail fit counts',
           hardware=dict(device=args.device,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0) if args.device=='cuda' else None))
    p['fingerprint']=fingerprint(p); return p


def main():
    parser=argparse.ArgumentParser(description=__doc__); sub=parser.add_subparsers(dest='command',required=True)
    prep=sub.add_parser('prepare'); prep.add_argument('--archive',type=Path,required=True); prep.add_argument('--cache-dir',type=Path,required=True)
    prep.add_argument('--expected-classes',type=int,default=200)
    run=sub.add_parser('run'); run.add_argument('--cache-dir',type=Path,required=True); run.add_argument('--out',type=Path,required=True)
    run.add_argument('--rhos',type=float,nargs='+',default=[100.,50.,10.]); run.add_argument('--seeds',type=int,nargs='+',default=[0,1,2,3,4])
    run.add_argument('--epochs',type=int,default=200); run.add_argument('--crt-epochs',type=int,default=20)
    run.add_argument('--warmup-epochs',type=int,default=10); run.add_argument('--lr',type=float,default=.1)
    run.add_argument('--batch-size',type=int,default=128); run.add_argument('--crt-batch-size',type=int,default=512)
    run.add_argument('--holdout-per-class',type=int,default=50); run.add_argument('--workers',type=int,default=4)
    run.add_argument('--device',choices=['cpu','cuda','mps'],default='cuda'); run.add_argument('--checkpoint-every',type=int,default=5)
    run.add_argument('--threads',type=int,default=4); run.add_argument('--no-ecod',action='store_true',help='Explicit sensitivity/smoke mode; primary retains fit-only shared ECOD')
    run.add_argument('--blocks',type=int,default=5,help='5=ResNet32; smaller values are smoke-only and recorded')
    run.add_argument('--aggregate-only',action='store_true')
    run.add_argument('--worker-index',type=int,default=0,help='Stride through cells with this zero-based worker index; does not change scientific protocol')
    run.add_argument('--worker-count',type=int,default=1,help='Number of independent GPU processes sharing this output folder')
    run.add_argument('--defer-aggregate',action='store_true',help='Workers write separate cells; run one --aggregate-only process after workers finish')
    args=parser.parse_args()
    if args.command=='prepare': prepare(args.archive,args.cache_dir,args.expected_classes); return
    if min(args.epochs,args.crt_epochs,args.batch_size,args.crt_batch_size,args.holdout_per_class,args.checkpoint_every,args.blocks)<1: parser.error('positive integer settings required')
    if args.warmup_epochs<0 or args.workers<0 or args.lr<=0: parser.error('invalid optimization/loader settings')
    if len(set(args.rhos))!=len(args.rhos) or len(set(args.seeds))!=len(args.seeds): parser.error('duplicate grid cells')
    if args.worker_count<1 or not 0<=args.worker_index<args.worker_count: parser.error('Invalid worker assignment')
    if args.worker_count>1 and not args.defer_aggregate and not args.aggregate_only: parser.error('Parallel workers require --defer-aggregate')
    if args.device=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable; choose an available device explicitly')
    if args.device=='mps' and not torch.backends.mps.is_available(): raise RuntimeError('MPS unavailable')
    torch.set_num_threads(args.threads); torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    images,labels,manifest=load_cache(args.cache_dir); protocol=make_protocol(args,manifest); args.out.mkdir(parents=True,exist_ok=True)
    ppath=args.out/'protocol.json'
    if ppath.exists():
        if json.loads(ppath.read_text())['fingerprint']!=protocol['fingerprint']: raise ValueError('Output folder belongs to another protocol/source/environment')
    else:
        write_json(ppath,protocol); write_json(args.out/'study_plan.json',dict(created_utc=now(),protocol_fingerprint=protocol['fingerprint'],status='specified before this full-scale study produces evaluation scores',
                   required_cells=len(args.rhos)*len(args.seeds),required_backbones=3*len(args.rhos)*len(args.seeds),source_previous_transfer_results_already_seen=True))
    if not args.aggregate_only:
        cells=[(rho,seed) for rho in args.rhos for seed in args.seeds]
        for index,(rho,seed) in enumerate(cells):
            if index%args.worker_count!=args.worker_index: continue
            run_cell(images,labels,manifest,args,protocol,seed,rho)
            if not args.defer_aggregate: aggregate(args,protocol)
    if args.aggregate_only or not args.defer_aggregate: aggregate(args,protocol)


if __name__=='__main__': main()
