#!/usr/bin/env python3
"""Frozen DINO features and actual last-two-block DeiT tuning on protected splits.

All architecture/hyperparameter choices use tuning rows. Confirmation inference
requires a frozen selection receipt and complete final-seed training. Cached
DeiT inputs are outputs of the first ten frozen blocks, not final embeddings.
"""
from __future__ import annotations
import argparse,copy,csv,hashlib,importlib.util,importlib.metadata,itertools,json,math,os,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy import stats
from sklearn.metrics import confusion_matrix

def module(name):
    spec=importlib.util.spec_from_file_location('vision_'+name,Path(__file__).with_name(name+'.py'))
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result
extractor=module('extract_vit_comparison_features')
sha,write_json=extractor.sha,extractor.write_json
fingerprint=extractor.fingerprint
require=extractor.require
VERSION=1
DINO_URL='https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth'
DINO_SHA='b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9'
ARMS=[(m,o) for m in ('deit_frozen','deit_last2','dino_frozen') for o in ('ce','balanced_softmax')]

def load_dino(path):
    import timm
    from timm.models.vision_transformer import checkpoint_filter_fn
    require(sha(path)==DINO_SHA,'Official DINO bytes changed')
    model=timm.create_model('vit_small_patch14_dinov2.lvd142m',pretrained=False,img_size=224,num_classes=0)
    state=torch.load(path,map_location='cpu',weights_only=True)
    adapted=checkpoint_filter_fn(state,model,interpolation='bicubic',antialias=True)
    model.load_state_dict(adapted,strict=True);model.eval()
    for p in model.parameters():p.requires_grad_(False)
    return model

def deit_prefix(model,x):
    x=model.patch_embed(x);x=model._pos_embed(x);x=model.patch_drop(x);x=model.norm_pre(x)
    for block in model.blocks[:10]:x=block(x)
    return x

class LastBlocks(nn.Module):
    def __init__(self,backbone,classes,seed,scaler=None,train_blocks=True):
        super().__init__();self.blocks=copy.deepcopy(backbone.blocks[10:]);self.norm=copy.deepcopy(backbone.norm)
        self.train_blocks=train_blocks
        self.register_buffer('feature_mean',torch.from_numpy(scaler[0]) if scaler is not None else torch.zeros(192))
        self.register_buffer('feature_sd',torch.from_numpy(scaler[1]) if scaler is not None else torch.ones(192))
        for p in itertools.chain(self.blocks.parameters(),self.norm.parameters()):p.requires_grad_(train_blocks)
        torch.manual_seed(seed+901);self.head=nn.Linear(192,classes)
        nn.init.trunc_normal_(self.head.weight,std=.02);nn.init.zeros_(self.head.bias)
    def train(self,mode=True):
        super().train(mode)
        if not self.train_blocks:self.blocks.eval();self.norm.eval()
        return self
    def forward(self,x):
        x=self.blocks[0](x)
        if hasattr(self.blocks[1],'attn'):x=final_cls_block(self.blocks[1],x)
        else:x=self.blocks[1](x)[:,:1]
        features=(self.norm(x)[:,0]-self.feature_mean)/self.feature_sd
        return self.head(features)

def final_cls_block(block,x):
    """Exact class-token output; other final-block outputs are unused by head.

    Q/K/V retain original parameters and all source keys/values. The classifier
    only requires query0 and its token-wise residual/MLP. Dropout is zero in the
    official instantiated DeiT; nonzero attention dropout is rejected here.
    """
    attention=block.attn
    require(attention.attn_drop.p==0 and attention.proj_drop.p==0,'CLS optimization requires deterministic DeiT attention')
    z=block.norm1(x);batch,tokens,width=z.shape
    qkv=attention.qkv(z).reshape(batch,tokens,3,attention.num_heads,attention.head_dim).permute(2,0,3,1,4)
    q,k,v=qkv.unbind(0);q=attention.q_norm(q[:,:,:1]);k=attention.k_norm(k)
    context=F.scaled_dot_product_attention(q,k,v,dropout_p=0.,scale=attention.scale)
    context=context.transpose(1,2).reshape(batch,1,width);context=attention.proj_drop(attention.proj(context))
    cls=x[:,:1]+block.drop_path1(block.ls1(context))
    return cls+block.drop_path2(block.ls2(block.mlp(block.norm2(cls))))

class FrozenHead(nn.Module):
    def __init__(self,width,classes,seed):
        super().__init__();torch.manual_seed(seed+901);self.head=nn.Linear(width,classes)
        nn.init.trunc_normal_(self.head.weight,std=.02);nn.init.zeros_(self.head.bias)
    def forward(self,x):return self.head(x)

def objective(logits,labels,counts,name):
    if name=='balanced_softmax':logits=logits+torch.log(torch.as_tensor(counts,device=logits.device,dtype=logits.dtype))
    return F.cross_entropy(logits,labels)

def metric(logits,labels,order):
    require(logits.shape==(len(labels),len(order)) and np.isfinite(logits).all(),'Invalid or nonfinite logits')
    pred=np.argmax(logits,axis=1);c=len(order);cm=confusion_matrix(labels,pred,labels=np.arange(c))
    tp=np.diag(cm).astype(float);support=cm.sum(1);f1=np.divide(2*tp,support+cm.sum(0),out=np.zeros(c),where=(support+cm.sum(0))>0)
    recall=np.divide(tp,support,out=np.zeros(c),where=support>0)
    groups={'head':order[:c//3],'mid':order[c//3:2*c//3],'tail':order[2*c//3:]}
    return dict(accuracy=float(tp.sum()/cm.sum()),macro_f1=float(f1.mean()),macro_recall=float(recall.mean()),
        **{name+'_f1':float(f1[ids].mean()) for name,ids in groups.items()},per_class_f1=f1.tolist(),per_class_support=support.tolist())

def calibration(logits,labels,prior,order):
    # All methods use the same explicitly calibration-selected offset grid.
    grid=np.linspace(0,2,41);offset=np.log(np.maximum(prior,1e-12))
    scores=[metric(logits-t*offset,labels,order)['macro_f1'] for t in grid]
    index=int(np.argmax(scores));return float(grid[index]),[dict(tau=float(t),macro_f1=float(s)) for t,s in zip(grid,scores)]

def load_split(args):
    split={k:a.copy() for k,a in np.load(args.partition,allow_pickle=False).items()}
    y=np.load(Path(args.deit_cache)/'labels.npy',allow_pickle=False)
    require(all(not set(split[a])&set(split[b]) for a,b in itertools.combinations(('fit','tune','calibration','confirmation'),2)),'Overlapping partitions')
    require(len(split['confirmation'])==20000 and len(np.unique(split['fit']))==len(split['fit']),'Invalid split')
    order=split['class_order'];require(sorted(order.tolist())==list(range(200)) and np.array_equal(split['class_rank'][order],np.arange(200)),'Invalid class order/rank')
    return split,y,order

def prepare(args):
    args.out.mkdir(parents=True,exist_ok=True);split,y,order=load_split(args)
    manifest=json.loads((Path(args.deit_cache)/'manifest.json').read_text())
    for name,digest in manifest['artifacts'].items():require(sha(Path(args.deit_cache)/name)==digest,'Original feature cache bytes changed')
    ids=np.unique(np.concatenate([split[n] for n in ('fit','tune','calibration','confirmation')]+[np.arange(100000,110000)]))
    settings=dict(seeds=list(range(10,20)),epochs=args.epochs,batch_size=args.batch_size,
        head_lr_candidates=[.001,.003],encoder_lr_candidates=[1e-5,3e-5],partial_head_lr_multiplier=100,tuning_seed=901,
        optimizer='AdamW',weight_decay=.01,warmup_fraction=.1,gradient_clip=1.,arms=ARMS,
        preprocessing=extractor.PREPROCESSING,device=args.device,threads=args.threads,
        augmentation='deterministic image transform; first10blocks frozen/eval; last2 trainable without new image augmentation',
        cache_precision='float16 prefix with prespecified mean absolute feature error<=0.001 and max<=0.02 checked on train-only rows',
        last_layer_computation='Original firsttrainable block evaluatesall197tokens; final block evaluatesCLSquery andCLSresidual/MLP only withallkeys/values, mathematically equivalent atzeroDeiTdropout; verified outputs/input/parameter gradients againstfullblock',
        selection='best tuning-macroF1 epoch after calibration-selected logit adjustment; two LR candidates per arm; no confirmation or officialeval scoring until freeze',
        inference='raw and calibration-selected logit adjustment, tau0..2 by.05; uniform BalancedSoftmax prior => repeated tau0 view',
        class_order_semantics='partition class_order is rank-to-class, class_rank its inverse; headfirst66,midnext67,taillast67',
        primary_family=['partial_vs_frozen_ce','dino_vs_deit_ce','balanced_softmax_vs_ce_deit_frozen','balanced_softmax_vs_ce_deit_last2','balanced_softmax_vs_ce_dino'],
        endpoint='calibrated confirmation macroF1',confirmation_scope='New holdout for this study; historical models may have seen these source rows; same-dataset followup, not external replication')
    protocol=dict(version=VERSION,source_sha256=sha(__file__),source_helper_sha256=sha(extractor.__file__),packages={k:importlib.metadata.version(k) for k in ('torch','torchvision','timm','numpy')},partition_sha256=sha(args.partition),deit_cache_manifest_sha256=sha(Path(args.deit_cache)/'manifest.json'),
        official_archive_sha256=extractor.OFFICIAL_ARCHIVE_SHA256,dino_weight_sha256=DINO_SHA,deit_weight_sha256=extractor.ENCODERS['deit_tiny']['expected_sha256'],
        source_rows=110000,prefix_rows=len(ids),fit_rows=len(split['fit']),tune_rows=len(split['tune']),cal_rows=len(split['calibration']),confirmation_rows=len(split['confirmation']),settings=settings)
    protocol['fingerprint']=fingerprint(protocol)
    path=args.out/'protocol.json'
    if path.exists():
        require(json.loads(path.read_text())==protocol,'Protocol changed, use new output')
        require(np.array_equal(np.load(args.out/'prefix_row_ids.npy',allow_pickle=False),ids),'Prefix source-row identities changed')
    else:write_json(path,protocol);np.save(args.out/'prefix_row_ids.npy',ids,allow_pickle=False)
    return protocol,split,y,order

def extract(args,pilot=False):
    protocol,split,y,order=prepare(args)
    corpus=extractor.Corpus(args.archive,args.original_manifest)
    deit,_=extractor.load_encoder('deit_tiny',Path(args.deit_weights));deit.to(args.device)
    dino=load_dino(args.dino_weights).to(args.device)
    if pilot:
        representatives=np.array([split['fit'][np.flatnonzero(y[split['fit']]==c)[0]] for c in range(200)])
        rest=np.setdiff1d(split['fit'],representatives);extra=np.random.default_rng(20261013).choice(rest,56,replace=False);ids=np.concatenate([representatives,extra])
    else:ids=np.load(args.out/'prefix_row_ids.npy',allow_pickle=False)
    if pilot:
        began=time.perf_counter();error=[];cls_error=[]
        for start in range(0,len(ids),args.extract_batch):
            x=torch.stack([corpus[int(i)][0] for i in ids[start:start+args.extract_batch]]).to(args.device)
            with torch.inference_mode():
                prefix=deit_prefix(deit,x);exact=deit.forward_features(x);replayed=prefix.to(torch.float16).to(torch.float32)
                efficient=deit.blocks[10](replayed);efficient=deit.norm(final_cls_block(deit.blocks[11],efficient))[:,0]
                for block in deit.blocks[10:]:replayed=block(replayed)
                replayed=deit.norm(replayed);error.append((efficient-exact[:,0]).abs().cpu().numpy());cls_error.append((efficient-replayed[:,0]).abs().cpu().numpy());dino(x)
        extractor.synchronize(args.device);delta=np.concatenate(error)
        require(float(delta.mean())<=.001 and float(delta.max())<=.02,'Prefix precision check failed');cls_delta=np.concatenate(cls_error);require(float(cls_delta.max())<=.0001,'CLS-only numerical equivalence check failed')
        write_json(args.out/'train_only_pilot.json',dict(rows=len(ids),seconds=time.perf_counter()-began,
            train_only=True,confirmation_scored=False,mean_absolute_prefix_feature_error=float(delta.mean()),max_absolute_prefix_feature_error=float(delta.max()),cls_only_max_absolute_error_vs_full_final_block=float(cls_delta.max())))
        return
    pilot_record=json.loads((args.out/'train_only_pilot.json').read_text());require(pilot_record['train_only'],'Pilot missing')
    prefix_path=args.out/'deit_prefix.npy';progress_path=args.out/'prefix_progress.json'
    if prefix_path.exists():
        prefix=np.load(prefix_path,mmap_mode='r+');progress=json.loads(progress_path.read_text());require(progress['protocol']==protocol['fingerprint'],'Cache identity changed')
    else:
        prefix=np.lib.format.open_memmap(prefix_path,mode='w+',dtype=np.float16,shape=(len(ids),197,192));progress=dict(protocol=protocol['fingerprint'],completed=0,blocks=[]);write_json(progress_path,progress)
    verify_progress(progress,len(ids));require(prefix.shape==(len(ids),197,192) and prefix.dtype==np.float16,'Prefix dimensions changed')
    for r in progress['blocks']:require(hashlib.sha256(prefix[r['start']:r['end']].tobytes()).hexdigest()==r['sha256'],'Prefix block changed')
    started=time.perf_counter()
    for start in range(progress['completed'],len(ids),args.extract_batch):
        batchids=ids[start:start+args.extract_batch];x=torch.stack([corpus[int(i)][0] for i in batchids]).to(args.device)
        with torch.inference_mode():a=deit_prefix(deit,x).cpu().numpy().astype(np.float16)
        require(np.isfinite(a).all(),'Nonfinite prefix');prefix[start:start+len(a)]=a
        if start//args.extract_batch%20==19 or start+len(a)==len(ids):
            prefix.flush();old=progress['completed'];end=start+len(a);progress['blocks'].append(dict(start=old,end=end,sha256=hashlib.sha256(prefix[old:end].tobytes()).hexdigest()));progress['completed']=end;write_json(progress_path,progress)
            print(json.dumps(dict(stage='prefix',completed=end,expected=len(ids))),flush=True)
    del prefix;deit.cpu();torch.mps.empty_cache() if args.device=='mps' else None
    dino_path=args.out/'dino_features.npy';dprogress=args.out/'dino_progress.json'
    if dino_path.exists():
        features=np.load(dino_path,mmap_mode='r+');p=json.loads(dprogress.read_text());require(p['protocol']==protocol['fingerprint'],'DINO cache identity changed')
    else:features=np.lib.format.open_memmap(dino_path,mode='w+',dtype=np.float32,shape=(110000,384));p=dict(protocol=protocol['fingerprint'],completed=0,blocks=[]);write_json(dprogress,p)
    verify_progress(p,110000);require(features.shape==(110000,384) and features.dtype==np.float32,'DINO dimensions changed')
    for r in p['blocks']:require(hashlib.sha256(features[r['start']:r['end']].tobytes()).hexdigest()==r['sha256'],'DINO block changed')
    for start in range(p['completed'],110000,args.extract_batch):
        x=torch.stack([corpus[i][0] for i in range(start,min(start+args.extract_batch,110000))]).to(args.device)
        with torch.inference_mode():a=dino(x).cpu().numpy()
        require(np.isfinite(a).all(),'Nonfinite DINO');features[start:start+len(a)]=a
        if start//args.extract_batch%20==19 or start+len(a)==110000:
            features.flush();old=p['completed'];end=start+len(a);p['blocks'].append(dict(start=old,end=end,sha256=hashlib.sha256(features[old:end].tobytes()).hexdigest()));p['completed']=end;write_json(dprogress,p)
            print(json.dumps(dict(stage='dino',completed=end,expected=110000)),flush=True)
    features.flush();write_json(args.out/'extraction_complete.json',dict(protocol=protocol['fingerprint'],seconds=time.perf_counter()-started,prefix_rows=len(ids),dino_rows=110000,
        prefix_sha256=sha(prefix_path),dino_sha256=sha(dino_path),ids_sha256=sha(args.out/'prefix_row_ids.npy'),weight_sha256=DINO_SHA,source_reader_sha256=sha(Path(extractor.__file__)),pilot=pilot_record,
        dino_positions='Official518px positional grid resized by reviewed timm bicubic+antialias to224; common224transform controls resolution',encoders_frozen_during_extraction=True))

def inputs(args,name,split):
    if name in ('deit_last2','deit_frozen'):
        a=np.load(args.out/'deit_prefix.npy',mmap_mode='r');ids=np.load(args.out/'prefix_row_ids.npy');lookup=np.full(110000,-1);lookup[ids]=np.arange(len(ids));return a,lookup,None
    a=np.load((args.out/'dino_features.npy') if name=='dino_frozen' else Path(args.deit_cache)/'features.npy',mmap_mode='r');lookup=np.arange(len(a));fit=np.asarray(a[split['fit']],dtype=np.float64);mean=fit.mean(0).astype(np.float32);sd=fit.std(0).astype(np.float32);sd[sd<1e-8]=1
    return a,lookup,(mean,sd)

def batch(a,lookup,ids,scaler,device):
    require((lookup[ids]>=0).all(),'Missing cache row')
    x=np.array(a[lookup[ids]],dtype=np.float32)
    if scaler is not None:x=(x-scaler[0])/scaler[1]
    return torch.from_numpy(x).to(device)

def logits_for(model,a,lookup,ids,scaler,args):
    model.eval();result=[]
    with torch.inference_mode():
        for start in range(0,len(ids),args.eval_batch):result.append(model(batch(a,lookup,ids[start:start+args.eval_batch],scaler,args.device)).cpu().numpy())
    return np.concatenate(result)

def create_training_model(args,split,name,seed):
    a,lookup,scaler=inputs(args,name,split);backbone,_=extractor.load_encoder('deit_tiny',Path(args.deit_weights))
    if name in ('deit_last2','deit_frozen'):
        original=np.load(Path(args.deit_cache)/'features.npy',mmap_mode='r');f=np.asarray(original[split['fit']],dtype=np.float64);mean=f.mean(0).astype(np.float32);sd=f.std(0).astype(np.float32);sd[sd<1e-8]=1
        model=LastBlocks(backbone,200,seed,(mean,sd),train_blocks=name=='deit_last2')
    else:model=FrozenHead(a.shape[-1],200,seed)
    model.to(args.device)
    return model,a,lookup,scaler

def verify_progress(record,rows):
    require(0<=record['completed']<=rows,'Cache completion range invalid');cursor=0
    for block in record['blocks']:
        require(block['start']==cursor and cursor<block['end']<=rows,'Cache blocks are not contiguous');cursor=block['end']
    require(cursor==record['completed'],'Cache coverage differs from completion')

def validate_extraction(args,protocol):
    record=json.loads((args.out/'extraction_complete.json').read_text());require(record['protocol']==protocol['fingerprint'],'Extraction protocol mismatch')
    ids=np.load(args.out/'prefix_row_ids.npy',allow_pickle=False);require(len(ids)==protocol['prefix_rows'] and len(np.unique(ids))==len(ids),'Prefix row identities invalid')
    for file,key,shape,dtype in [('deit_prefix.npy','prefix_sha256',(len(ids),197,192),np.float16),('dino_features.npy','dino_sha256',(110000,384),np.float32),('prefix_row_ids.npy','ids_sha256',(len(ids),),np.int64)]:
        array=np.load(args.out/file,mmap_mode='r',allow_pickle=False);require(array.shape==shape and array.dtype==dtype and sha(args.out/file)==record[key],'Extraction artifact mismatch: '+file)
    require(record['prefix_rows']==len(ids) and record['dino_rows']==110000 and record['weight_sha256']==DINO_SHA,'Extraction completeness mismatch')

def validate_training_receipt(args,folder,identity,steps):
    record=json.loads((folder/'training_complete.json').read_text())
    require(record['identity']==identity and record['epochs']==args.epochs and record['optimizer_updates']==args.epochs*steps,'Incomplete or mismatched final training')
    require(sha(folder/'checkpoint.pt')==record['checkpoint_sha256'],'Final checkpoint changed')
    history=json.loads((folder/'history.json').read_text());require(len(history)==args.epochs and all(r['epoch']==i+1 and r['optimizer_updates']==(i+1)*steps for i,r in enumerate(history)),'Final history budgets changed')
    require(1<=record['best_epoch']<=args.epochs and record['best_epoch']==int(np.argmax([r['tuning_macro_f1'] for r in history]))+1,'Checkpoint selection changed')
    return record

def validate_checkpoint(saved,model,identity,initial,epochs,steps):
    require(saved['identity']==identity and saved['initial']==initial,'Checkpoint identity differs')
    history=saved['history'];completed=saved['completed_epochs']
    require(0<=completed<=epochs and len(history)==completed and all(r['epoch']==i+1 and r['optimizer_updates']==(i+1)*steps for i,r in enumerate(history)),'Checkpoint training budget invalid')
    require(np.isfinite(saved['best']) and 0<=saved['best']<=1 and np.isfinite(saved['seconds']) and saved['seconds']>=0,'Checkpoint selection statistic invalid')
    expected=model.state_dict()
    for key in ('model','best_state'):
        state=saved[key];require(set(state)==set(expected),'Checkpoint state keys differ')
        require(all(state[k].shape==expected[k].shape and state[k].dtype==expected[k].dtype and bool(torch.isfinite(state[k]).all()) for k in state),'Checkpoint state shape, precision or values invalid')

def train(args,split,y,order,name,obj,seed,lr,folder,protocol):
    folder.mkdir(parents=True,exist_ok=True);cp=folder/'checkpoint.pt';identity=dict(protocol=protocol['fingerprint'],name=name,objective=obj,seed=seed,lr=lr)
    model,a,lookup,scaler=create_training_model(args,split,name,seed)
    initial=extractor.state_hash(model);head=[p for n,p in model.named_parameters() if n.startswith('head.')];encoder=[p for n,p in model.named_parameters() if not n.startswith('head.') and p.requires_grad]
    head_lr=100*lr if name=='deit_last2' else lr
    groups=[dict(params=head,lr=head_lr,base_lr=head_lr)]
    if encoder:groups.append(dict(params=encoder,lr=lr,base_lr=lr))
    optimizer=torch.optim.AdamW(groups,weight_decay=.01);steps=math.ceil(len(split['fit'])/args.batch_size);total=steps*args.epochs
    counts=np.bincount(y[split['fit']],minlength=200);history=[];start=0;best=-1.;best_state=None;elapsed=0.
    if cp.exists():
        saved=torch.load(cp,map_location=args.device,weights_only=False);validate_checkpoint(saved,model,identity,initial,args.epochs,steps);history=saved['history'];start=saved['completed_epochs'];model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);best=saved['best'];best_state=saved['best_state'];elapsed=saved['seconds']
    for epoch in range(start,args.epochs):
        torch.manual_seed(seed+epoch*100003);model.train();ids=np.random.default_rng(np.random.SeedSequence([seed,epoch,905])).permutation(split['fit']);began=time.perf_counter();loss_sum=0.
        for start_batch in range(0,len(ids),args.batch_size):
            b=ids[start_batch:start_batch+args.batch_size];step=epoch*steps+start_batch//args.batch_size;factor=min(1.,(step+1)/max(1,.1*total))*.5*(1+math.cos(math.pi*max(0,step-.1*total)/max(1,.9*total)))
            for group in optimizer.param_groups:group['lr']=group['base_lr']*factor
            optimizer.zero_grad(set_to_none=True);z=model(batch(a,lookup,b,scaler,args.device));loss=objective(z,torch.from_numpy(y[b]).to(args.device),counts,obj);require(bool(torch.isfinite(loss)),'Nonfinite loss');loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step();loss_sum+=float(loss.detach().cpu())*len(b)
        tune=logits_for(model,a,lookup,split['tune'],scaler,args);raw_score=metric(tune,y[split['tune']],order)['macro_f1']
        prior=counts/counts.sum() if obj=='ce' else np.ones(200)/200
        if obj=='ce':
            cal=logits_for(model,a,lookup,split['calibration'],scaler,args);tau,_=calibration(cal,y[split['calibration']],prior,order)
        else:tau=0.
        score=metric(tune-tau*np.log(prior),y[split['tune']],order)['macro_f1'];duration=time.perf_counter()-began;elapsed+=duration
        if score>best:best=score;best_state={k:t.detach().cpu().clone() for k,t in model.state_dict().items()};best_epoch=epoch+1
        else:best_epoch=history[-1]['best_epoch'] if history else 1
        history.append(dict(epoch=epoch+1,loss=loss_sum/len(ids),tuning_macro_f1=score,tuning_raw_macro_f1=raw_score,calibration_selected_tau=tau,best_epoch=best_epoch,seconds=duration,optimizer_updates=(epoch+1)*steps))
        torch.save(dict(identity=identity,initial=initial,model=model.state_dict(),optimizer=optimizer.state_dict(),completed_epochs=epoch+1,history=history,best=best,best_state=best_state,seconds=elapsed),cp.with_suffix('.partial'));cp.with_suffix('.partial').replace(cp)
        write_json(folder/'history.json',history);print(json.dumps(dict(stage='train',arm=name+'__'+obj,seed=seed,epoch=epoch+1,tune_macro_f1=score)),flush=True)
    model.load_state_dict(best_state);write_json(folder/'training_complete.json',dict(identity=identity,initial=initial,best_tune=best,best_epoch=history[-1]['best_epoch'],epochs=len(history),optimizer_updates=args.epochs*steps,trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),total_model_parameters=sum(p.numel() for p in model.parameters()),seconds=elapsed,
        scaler_fitted_only_on_fit=True,checkpoint_sha256=sha(cp)))
    return model,a,lookup,scaler

def tune(args):
    protocol,split,y,order=prepare(args);validate_extraction(args,protocol)
    for name,obj in ARMS:
        for lr in ([1e-5,3e-5] if name=='deit_last2' else [.001,.003]):train(args,split,y,order,name,obj,901,lr,args.out/'tuning'/(name+'__'+obj)/str(lr),protocol)
    selection={}
    for name,obj in ARMS:
        candidates=[]
        for lr in ([1e-5,3e-5] if name=='deit_last2' else [.001,.003]):
            p=args.out/'tuning'/(name+'__'+obj)/str(lr)/'training_complete.json';r=json.loads(p.read_text());candidates.append(dict(lr=lr,score=r['best_tune'],training_sha256=sha(p)))
        selection[name+'__'+obj]=sorted(candidates,key=lambda r:(-r['score'],r['lr']))[0]
    record=dict(protocol=protocol['fingerprint'],selection=selection,selection_source='Development data: calibration rows select CE offsets, tuning rows select learning rate and best epoch; dedicated tuning seed901 and epoch selection repeated for final seeds. Confirmation and officialeval excluded.',confirmation_scored=False)
    p=args.out/'selection.json'
    if p.exists():require(json.loads(p.read_text())==record,'Selection changed')
    else:write_json(p,record)

def final_train(args):
    protocol,split,y,order=prepare(args);validate_extraction(args,protocol);selection=json.loads((args.out/'selection.json').read_text());require(selection['protocol']==protocol['fingerprint'] and not selection['confirmation_scored'],'Invalid selection')
    require(set(selection['selection'])=={a+'__'+b for a,b in ARMS},'Selection incomplete')
    for name,obj in ARMS:
        lr=selection['selection'][name+'__'+obj]['lr']
        for seed in range(10,20):train(args,split,y,order,name,obj,seed,lr,args.out/'final'/(name+'__'+obj)/str(seed),protocol)
    receipts={};steps=math.ceil(len(split['fit'])/args.batch_size)
    for name,obj in ARMS:
        for seed in range(10,20):
            folder=args.out/'final'/(name+'__'+obj)/str(seed);identity=dict(protocol=protocol['fingerprint'],name=name,objective=obj,seed=seed,lr=selection['selection'][name+'__'+obj]['lr']);validate_training_receipt(args,folder,identity,steps);receipts[name+'__'+obj+'/'+str(seed)]=sha(folder/'training_complete.json')
    write_json(args.out/'final_training_complete.json',dict(protocol=protocol['fingerprint'],selection_sha256=sha(args.out/'selection.json'),models=60,epochs=args.epochs,training_receipt_sha256=receipts))

def evaluate(args):
    protocol,split,y,order=prepare(args);validate_extraction(args,protocol);require((args.out/'final_training_complete.json').exists(),'All final training must precede confirmation')
    complete=json.loads((args.out/'final_training_complete.json').read_text());require(complete['protocol']==protocol['fingerprint'] and complete['selection_sha256']==sha(args.out/'selection.json') and complete['models']==60 and complete['epochs']==args.epochs,'Final training gate mismatch')
    selection=json.loads((args.out/'selection.json').read_text());records=[];steps=math.ceil(len(split['fit'])/args.batch_size)
    for name,obj in ARMS:
        for seed in range(10,20):
            folder=args.out/'final'/(name+'__'+obj)/str(seed);identity=dict(protocol=protocol['fingerprint'],name=name,objective=obj,seed=seed,lr=selection['selection'][name+'__'+obj]['lr']);validate_training_receipt(args,folder,identity,steps);require(sha(folder/'training_complete.json')==complete['training_receipt_sha256'][name+'__'+obj+'/'+str(seed)],'Training receipt changed after gate')
    for name,obj in ARMS:
        for seed in range(10,20):
            folder=args.out/'final'/(name+'__'+obj)/str(seed);model,a,lookup,scaler=create_training_model(args,split,name,seed);saved=torch.load(folder/'checkpoint.pt',map_location=args.device,weights_only=False);identity=dict(protocol=protocol['fingerprint'],name=name,objective=obj,seed=seed,lr=selection['selection'][name+'__'+obj]['lr']);validate_checkpoint(saved,model,identity,extractor.state_hash(model),args.epochs,steps);model.load_state_dict(saved['best_state'])
            counts=np.bincount(y[split['fit']],minlength=200);prior=counts/counts.sum() if obj=='ce' else np.ones(200)/200
            cal=logits_for(model,a,lookup,split['calibration'],scaler,args);tau,curve=calibration(cal,y[split['calibration']],prior,order)
            conf=logits_for(model,a,lookup,split['confirmation'],scaler,args);official=logits_for(model,a,lookup,np.arange(100000,110000),scaler,args)
            np.savez_compressed(folder/'predictions.npz',calibration_logits=cal,confirmation_logits=conf,official_logits=official)
            record=dict(arm=name+'__'+obj,seed=seed,selected_tau=tau,calibration_curve=curve,prior=prior.tolist(),confirmation=metric(conf,y[split['confirmation']],order),confirmation_la=metric(conf-tau*np.log(prior),y[split['confirmation']],order),official_exploratory=metric(official-tau*np.log(prior),y[100000:],order),predictions_sha256=sha(folder/'predictions.npz'))
            write_json(folder/'results.json',record);records.append(record)
    write_json(args.out/'results.json',dict(protocol=protocol['fingerprint'],selection_sha256=sha(args.out/'selection.json'),rows=records))

def constant_safe_t_p(d):
    d=np.asarray(d,dtype=float)
    require(d.shape==(10,) and np.isfinite(d).all(),'Invalid paired statistic input')
    return float(stats.ttest_1samp(d,0).pvalue) if np.any(d!=d[0]) else (0. if d[0] else 1.)

def aggregate(args):
    protocol,split,y,order=prepare(args);result=json.loads((args.out/'results.json').read_text());rows=result['rows'];require(result['protocol']==protocol['fingerprint'] and result['selection_sha256']==sha(args.out/'selection.json'),'Results identity changed');require(len(rows)==60 and {(r['arm'],r['seed']) for r in rows}=={(a+'__'+b,s) for a,b in ARMS for s in range(10,20)},'Missing or duplicated final model')
    # Verify every stored raw prediction independently from the model objects.
    for r in rows:
        folder=args.out/'final'/r['arm']/str(r['seed']);p=folder/'predictions.npz';require(sha(p)==r['predictions_sha256'],'Predictions changed')
        require(json.loads((folder/'results.json').read_text())==r,'Per-model result identity mismatch')
        with np.load(p,allow_pickle=False) as saved:
            prior=np.asarray(r['prior']);tau,curve=calibration(saved['calibration_logits'],y[split['calibration']],prior,order)
            require(tau==r['selected_tau'] and curve==r['calibration_curve'],'Calibration mismatch')
            for key,z,labels in [('confirmation',saved['confirmation_logits'],y[split['confirmation']]),('confirmation_la',saved['confirmation_logits']-tau*np.log(prior),y[split['confirmation']]),('official_exploratory',saved['official_logits']-tau*np.log(prior),y[100000:])]:require(metric(z,labels,order)==r[key],'Independent metric mismatch')
    summary=[]
    for name,obj in ARMS:
        arm=name+'__'+obj;rs=[r for r in rows if r['arm']==arm]
        for endpoint in ('confirmation','confirmation_la','official_exploratory'):
            for k in ('accuracy','macro_f1','macro_recall','head_f1','mid_f1','tail_f1'):
                a=np.array([r[endpoint][k] for r in rs]);summary.append(dict(arm=arm,endpoint=endpoint,metric=k,n=len(a),mean=float(a.mean()),sd=float(a.std(ddof=1))))
    contrasts=[];pairs=[('partial_vs_frozen_ce','deit_last2__ce','deit_frozen__ce'),('dino_vs_deit_ce','dino_frozen__ce','deit_frozen__ce')]+[('balanced_softmax_vs_ce_'+name,name+'__balanced_softmax',name+'__ce') for name in ('deit_frozen','deit_last2','dino_frozen')]
    for name,a,b in pairs:
        av={r['seed']:r['confirmation_la']['macro_f1'] for r in rows if r['arm']==a};bv={r['seed']:r['confirmation_la']['macro_f1'] for r in rows if r['arm']==b};d=np.array([av[s]-bv[s] for s in range(10,20)]);err=float(stats.t.ppf(.975,9)*d.std(ddof=1)/math.sqrt(10));null=np.array([np.mean(d*np.array(sign)) for sign in itertools.product((-1,1),repeat=10)]);p=float(np.mean(np.abs(null)>=abs(d.mean())-1e-14))
        contrasts.append(dict(name=name,mean=float(d.mean()),ci95_low=float(d.mean()-err),ci95_high=float(d.mean()+err),exact_p=p,t_p=constant_safe_t_p(d),paired_differences=d.tolist()))
    for field in ('exact_p','t_p'):
        indices=sorted(range(len(contrasts)),key=lambda i:contrasts[i][field]);previous=0.
        for rank,i in enumerate(indices):previous=max(previous,min(1.,(len(indices)-rank)*contrasts[i][field]));contrasts[i][field+'_holm']=previous
    record=dict(status='complete_and_prediction_verified',protocol=protocol['fingerprint'],models=60,readouts=180,summary=summary,primary_contrasts=contrasts,selection_sha256=sha(args.out/'selection.json'),results_sha256=sha(args.out/'results.json'))
    write_json(args.out/'aggregate.json',record)
    for name,data in [('summary.csv',summary),('paired_comparisons.csv',contrasts)]:
        with open(args.out/name,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    print(json.dumps(dict(status=record['status'],models=60,readouts=180)),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','pilot','extract','tune','train','evaluate','aggregate']);p.add_argument('--out',type=Path,required=True);p.add_argument('--partition',type=Path,required=True);p.add_argument('--deit-cache',type=Path,required=True);p.add_argument('--archive',type=Path,required=True);p.add_argument('--original-manifest',type=Path,required=True);p.add_argument('--deit-weights',type=Path,required=True);p.add_argument('--dino-weights',type=Path,required=True);p.add_argument('--device',choices=['cpu','mps','cuda'],default='mps');p.add_argument('--threads',type=int,default=1);p.add_argument('--epochs',type=int,default=15);p.add_argument('--batch-size',type=int,default=128);p.add_argument('--eval-batch',type=int,default=256);p.add_argument('--extract-batch',type=int,default=64);args=p.parse_args()
    torch.set_num_threads(args.threads);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True,warn_only=args.device!='cpu');require(args.device!='mps' or torch.backends.mps.is_available(),'MPS unavailable')
    if args.phase=='pilot':extract(args,True)
    elif args.phase=='extract':extract(args)
    else:globals()[{'train':'final_train'}.get(args.phase,args.phase)](args)
if __name__=='__main__':main()
