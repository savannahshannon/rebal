#!/usr/bin/env python3
"""Train and independently recheck fresh-confirmation DBpedia BERT controls.

Frozen BERT, random mean-pooled embeddings and last1/last2-block BERT tuning
share source partitions. Tune selects learning rate; calibration selects logit
adjustment; confirmation is scored only after both choices are frozen.
"""
from __future__ import annotations
import argparse, importlib.util, itertools, json, math, os, platform, time
from pathlib import Path
import numpy as np
from scipy import stats
from sklearn.metrics import f1_score
import torch
from torch import nn
from torch.nn import functional as F


def sibling(name):
    spec=importlib.util.spec_from_file_location('text_'+name,Path(__file__).with_name(name+'.py'))
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
extract=sibling('extract_text_followup_features');nonimage=sibling('train_nonimage');common=nonimage.common
sha,write_json,fingerprint=extract.sha,extract.write_json,extract.fingerprint
METHODS=('bert_ce','bert_bsoftmax','bert_natural_eq','bert_balanced_ce','bert_balanced_eq',
         'mean_ce','mean_bsoftmax','last0_ce','last0_bsoftmax','last1_ce','last1_bsoftmax','last2_ce','last2_bsoftmax')
TAUS=(0.,.25,.5,.75,1.,1.25,1.5,2.)
PRIMARY=(('bert_ce_la','mean_ce_la'),('bert_bsoftmax','bert_ce_la'),
         ('bert_balanced_eq','bert_balanced_ce'),('bert_natural_eq_la','bert_ce_la'),
         ('last1_ce_la','last0_ce_la'),('last2_ce_la','last0_ce_la'),
         ('last1_bsoftmax','last0_bsoftmax'),('last2_bsoftmax','last0_bsoftmax'))


def balanced_softmax(logits,labels,counts):
    counts=torch.as_tensor(counts,dtype=logits.dtype,device=logits.device)
    if counts.ndim!=1 or counts.shape[0]!=logits.shape[1] or not torch.all(counts>0):raise ValueError('Invalid positive class counts')
    return F.cross_entropy(logits+counts.log(),labels)


def sampling_ids(fit,y,seed,epoch,balanced=False):
    rng=np.random.default_rng(np.random.SeedSequence([seed,epoch,91201]));fit=np.asarray(fit,dtype=np.int64)
    if not balanced:return rng.permutation(fit)
    members=[fit[y[fit]==c] for c in range(14)]
    if any(len(x)==0 for x in members):raise ValueError('Missing fit class')
    classes=rng.integers(0,14,len(fit));u=rng.random(len(fit));out=np.empty(len(fit),dtype=np.int64)
    for c,rows in enumerate(members):
        mask=classes==c;out[mask]=rows[np.minimum((u[mask]*len(rows)).astype(np.int64),len(rows)-1)]
    return out


def epoch_batches(fit,y,seed,epoch,batch_size,balanced=False,lengths=None):
    if batch_size<=0:raise ValueError('Batch size must be positive')
    order=sampling_ids(fit,y,seed,epoch,balanced)
    if lengths is None:return [order[i:i+batch_size] for i in range(0,len(order),batch_size)]
    bucket=((np.asarray(lengths)+31)//32)*32;batches=[]
    for length in (32,64,96,128):
        selected=order[bucket[order]==length]
        batches.extend(selected[i:i+batch_size] for i in range(0,len(selected),batch_size))
    if sum(len(x) for x in batches)!=len(order):raise ValueError('Length buckets do not cover every sampled row')
    rng=np.random.default_rng(np.random.SeedSequence([seed,epoch,94231]))
    return [batches[i] for i in rng.permutation(len(batches))]


def validate_history(history,completed,epochs,steps):
    if not isinstance(completed,int) or not 0<=completed<=epochs or len(history)!=completed:raise ValueError('Invalid completed epoch/history budget')
    for i,row in enumerate(history,1):
        if row['epoch']!=i or row['optimizer_updates']!=i*steps or row['batches']!=steps or not math.isfinite(row['loss']) or not math.isfinite(row['seconds']) or row['seconds']<0:raise ValueError('History update/epoch budget mismatch')
        if len(row['EQ_eligible_batches_per_class'])!=14 or any(not 0<=x<=steps for x in row['EQ_eligible_batches_per_class']):raise ValueError('Invalid EQ coverage counts')


def validate_training_artifact(folder,selection=None):
    folder=Path(folder);budget=json.loads((folder/'budget.json').read_text());history=json.loads((folder/'history.json').read_text())
    for name,digest in budget['files'].items():
        if sha(folder/name)!=digest:raise ValueError('Training budget artifact integrity mismatch')
    validate_history(history,budget['completed_epochs'],budget['planned_epochs'],budget['steps_per_epoch'])
    if selection:
        if budget['identity']!=selection['identity'] or budget['completed_epochs']!=selection['epochs'] or selection['updates']!=budget['completed_epochs']*budget['steps_per_epoch'] or selection['checkpoint_sha256']!=budget['files']['checkpoint.pt'] or selection['initial_state_sha256']!=budget['initial_state_sha256']:raise ValueError('Selection/training budget mismatch')
    return budget


def validate_hyperparameters(out,method,choice,protocol,tune_labels):
    if choice['protocol_fingerprint']!=protocol['fingerprint'] or choice['tuning_seed']!=9 or choice['confirmation_scored_at_freeze'] is not False:raise ValueError('Hyperparameter freeze protocol mismatch')
    expected=protocol['partial_lrs'] if method.startswith('last') else protocol['adapter_lrs']
    if sorted(c['identity']['lr'] for c in choice['candidates'])!=sorted(expected):raise ValueError('Incomplete or extra LR tuning candidates')
    for candidate in choice['candidates']:
        folder=Path(out)/method/'seed_9'/f"lr_{candidate['identity']['lr']:g}"
        if candidate['identity']!={'protocol':protocol['fingerprint'],'method':method,'seed':9,'lr':candidate['identity']['lr']}:raise ValueError('Tuning candidate identity mismatch')
        budget=validate_training_artifact(folder,candidate)
        expected_epochs=protocol['partial_epochs'] if method.startswith('last') else protocol['adapter_epochs'];expected_steps=protocol['partial_steps_per_epoch'] if method.startswith('last') else protocol['adapter_steps_per_epoch']
        if budget['planned_epochs']!=expected_epochs or budget['steps_per_epoch']!=expected_steps or candidate['epochs']!=expected_epochs:raise ValueError('Training candidate budget differs from declared protocol')
        logits=np.load(folder/'tune_logits.npy',allow_pickle=False);validate_logits(logits,tune_labels)
        if sha(folder/'tune_logits.npy')!=candidate['tune_logits_sha256'] or float(f1_score(tune_labels,logits.argmax(1),labels=np.arange(14),average='macro',zero_division=0))!=candidate['tune_macro_f1']:raise ValueError('Tuning candidate score/integrity mismatch')
    winner=max(choice['candidates'],key=lambda c:c['tune_macro_f1'])
    if choice['selected_lr']!=winner['identity']['lr']:raise ValueError('Frozen global LR differs from tune-only winner')


class Adapter(nn.Module):
    def __init__(self,dimension=768):
        super().__init__();self.body=nn.Sequential(nn.Linear(dimension,256),nn.ReLU(),nn.Dropout(.1),nn.Linear(256,128),nn.ReLU(),nn.Dropout(.1));self.head=nn.Linear(128,14);self.apply(common.initialize)
    def forward(self,x):
        features=self.body(x);return self.head(features),features


def cls_only_layer(layer,hidden,attention_mask):
    """Exact encoder BERT CLS output at dropout0, preserving all-token K/V.

    The final layer's other query outputs cannot influence CLS because its
    attention-output projection, residual, layer norms and FFN are per token.
    Training dropout preserves the marginal distribution, not RNG realization.
    """
    attention=layer.attention.self
    if attention.position_embedding_type!='absolute' or layer.is_decoder or layer.add_cross_attention or layer.chunk_size_feed_forward!=0:raise ValueError('CLS optimization requires plain absolute-position encoder BERT')
    batch,length,_=hidden.shape;heads=attention.num_attention_heads;width=attention.attention_head_size
    query=attention.query(hidden[:,:1]).view(batch,1,heads,width).transpose(1,2)
    key=attention.key(hidden).view(batch,length,heads,width).transpose(1,2)
    value=attention.value(hidden).view(batch,length,heads,width).transpose(1,2)
    probability=attention.dropout_prob if hasattr(attention,'dropout_prob') else attention.dropout.p
    context=F.scaled_dot_product_attention(query,key,value,attn_mask=attention_mask,dropout_p=probability if attention.training else 0.,is_causal=False)
    context=context.transpose(1,2).reshape(batch,1,heads*width)
    attended=layer.attention.output(context,hidden[:,:1])
    return layer.output(layer.intermediate(attended),attended)


class PartialBert(nn.Module):
    def __init__(self,bert,depth,mean=None,std=None):
        super().__init__()
        if depth not in (1,2):raise ValueError('Only last1/last2 supported')
        dimension=bert.config.hidden_size;self.layers=nn.ModuleList(list(bert.encoder.layer[10:]));self.depth=depth;self.adapter=Adapter(dimension)
        for p in self.layers.parameters():p.requires_grad_(False)
        for layer in self.layers[2-depth:]:
            for p in layer.parameters():p.requires_grad_(True)
        self.register_buffer('feature_mean',torch.zeros(dimension) if mean is None else torch.as_tensor(mean,dtype=torch.float32));self.register_buffer('feature_std',torch.ones(dimension) if std is None else torch.as_tensor(std,dtype=torch.float32))
        self.train()
    def train(self,mode=True):
        super().train(mode)
        # Frozen layer10 dropout remains off when only layer11 is tuned.
        for layer in self.layers[:2-self.depth]:layer.eval()
        return self
    def forward(self,hidden,mask):
        extended=(1.-mask[:,None,None,:].to(torch.float32))*torch.finfo(torch.float32).min
        hidden=self.layers[0](hidden,attention_mask=extended)[0]
        cls=cls_only_layer(self.layers[1],hidden,extended)[:,0]
        return self.adapter((cls.to(torch.float32)-self.feature_mean)/self.feature_std)


def validate_logits(logits,labels):
    if logits.shape!=(len(labels),14) or not np.issubdtype(logits.dtype,np.floating) or not np.isfinite(logits).all():raise ValueError('Invalid finite logit matrix')


def adjust(cal,cal_y,test,prior):
    validate_logits(cal,cal_y);prior=np.asarray(prior,dtype=np.float64)
    if prior.shape!=(14,) or not np.all(prior>0) or not np.isclose(prior.sum(),1):raise ValueError('Invalid positive prior')
    logprior=np.log(prior).astype(np.float32);curve=[]
    for tau in TAUS:
        pred=(cal-tau*logprior).argmax(1);curve.append(dict(tau=tau,macro_f1=float(f1_score(cal_y,pred,labels=np.arange(14),average='macro',zero_division=0))))
    chosen=max(curve,key=lambda r:r['macro_f1'])['tau']
    return test-chosen*logprior,dict(tau=chosen,curve=curve,prior=prior.tolist(),tie_break='first smallest tau; calibration only')


def load_data(data,cache=None,verify=True):
    data=Path(data);meta=json.loads((data/'dataset.json').read_text());y=np.load(data/'labels.npy',mmap_mode='r')
    if verify:
        for name,digest in meta['files'].items():
            if sha(data/name)!=digest:raise ValueError('Dataset hash mismatch')
    with np.load(data/'split.npz') as z:split={k:z[k] for k in z.files}
    extract.validate_split(y,split);rank=np.empty(14,dtype=np.int64);rank[split['class_rank']]=np.arange(14);split['rank']=rank
    result=dict(meta=meta,y=y,split=split,data=data)
    if cache:
        cache=Path(cache);cm=json.loads((cache/'features.json').read_text())
        if not cm['complete'] or cm['dataset_fingerprint']!=meta['fingerprint'] or cm['pilot']:raise ValueError('Feature cache identity mismatch')
        if verify:
            if sha(cache/'features.npy')!=cm['feature_sha256'] or sha(cache/'row_ids.npy')!=cm['row_ids_sha256']:raise ValueError('Frozen cache integrity mismatch')
        ids=np.load(cache/'row_ids.npy')
        if not np.array_equal(ids,np.arange(len(y),dtype=np.int64)):raise ValueError('Frozen feature row alignment invalid')
        result.update(cache=cache,cache_meta=cm,features=np.load(cache/'features.npy',mmap_mode='r'))
        cls_path=cache/'cls'/'cls_cache.json'
        if cls_path.exists():
            cls_meta=json.loads(cls_path.read_text())
            if not cls_meta['complete'] or cls_meta['source_sha256']!=sha(__file__) or cls_meta['dataset_fingerprint']!=meta['fingerprint'] or cls_meta['prefix_sha256']!=cm['prefix_sha256']:raise ValueError('CLS cache identity mismatch')
            if verify and (sha(cls_path.parent/'features.npy')!=cls_meta['feature_sha256'] or sha(cls_path.parent/'row_ids.npy')!=cls_meta['row_ids_sha256']):raise ValueError('CLS cache integrity mismatch')
            ids=np.load(cls_path.parent/'row_ids.npy');required=np.sort(np.concatenate([split[k] for k in extract.PARTS[:-1]]))
            if not np.array_equal(ids,required):raise ValueError('CLS cache row coverage mismatch')
            lookup=np.full(len(y),-1,dtype=np.int64);lookup[ids]=np.arange(len(ids))
            result.update(cls_meta=cls_meta,cls_features=np.load(cls_path.parent/'features.npy',mmap_mode='r'),cls_lookup=lookup)
    return result


def mean_inputs(d,args):
    target=args.out/'mean_transforms';target.mkdir(parents=True,exist_ok=True);marker=target/'transforms.json'
    identity=dict(dataset_fingerprint=d['meta']['fingerprint'],vocab_fit_ids_sha256=sha(d['data']/'split.npz'),max_words=args.max_words,max_tokens=128)
    if marker.exists():
        record=json.loads(marker.read_text())
        if record['identity']!=identity:raise ValueError('Mean input resume protocol mismatch')
        for name,digest in record['files'].items():
            if sha(target/name)!=digest:raise ValueError('Mean input hash mismatch')
        return np.load(target/'inputs.npy',mmap_mode='r'),record['vocab_size']
    vocab=nonimage.build_vocabulary(d['data']/'texts.jsonl',d['split']['fit'],args.max_words);write_json(target/'vocabulary.json',vocab)
    x=np.lib.format.open_memmap(target/'inputs.npy',mode='w+',dtype=np.int32,shape=(len(d['y']),128));x[:]=0
    with open(d['data']/'texts.jsonl',encoding='utf-8') as stream:
        rows=0
        for i,line in enumerate(stream):
            row=json.loads(line);words=nonimage.tokens(row['title'],row['content'])[:128];encoded=[vocab.get(t,1) for t in words] or [1]
            x[i,:len(encoded)]=encoded;rows+=1
    if rows!=len(d['y']):raise ValueError('Mean text alignment mismatch')
    x.flush();write_json(marker,dict(identity=identity,vocab_size=len(vocab),files={n:sha(target/n) for n in ('vocabulary.json','inputs.npy')},fitted_only_on='fit IDs',max_tokens=128))
    return x,len(vocab)


def prefix_inputs(d,verify=True):
    cache=d['cache'];meta=d['cache_meta']
    if 'prefix_sha256' not in meta:raise ValueError('Partial tuning needs frozen prefix cache')
    if verify and(sha(cache/'prefix.npy')!=meta['prefix_sha256'] or sha(cache/'prefix_row_ids.npy')!=meta['prefix_row_ids_sha256']):raise ValueError('Prefix cache hash mismatch')
    ids=np.load(cache/'prefix_row_ids.npy');required=np.sort(np.concatenate([d['split'][k] for k in extract.PARTS[:-1]]))
    if not np.array_equal(ids,required):raise ValueError('Prefix cache split coverage mismatch')
    lookup=np.full(len(d['y']),-1,dtype=np.int64);lookup[ids]=np.arange(len(ids))
    return np.load(cache/'prefix.npy',mmap_mode='r'),lookup,np.load(d['data']/'attention_mask.npy',mmap_mode='r')


def get_rng():
    record=common.rng_state()
    if torch.backends.mps.is_available():record['mps']=torch.mps.get_rng_state()
    return record

def restore_rng(record):
    common.restore_rng(record)
    if 'mps' in record and torch.backends.mps.is_available():torch.mps.set_rng_state(record['mps'])


def train_model(d,method,seed,lr,args,inputs,vocab=None,prefix=None):
    target=args.out/method/f'seed_{seed}'/f'lr_{lr:g}';target.mkdir(parents=True,exist_ok=True)
    identity=dict(protocol=args.protocol_fingerprint,method=method,seed=seed,lr=lr)
    common.set_seed(seed);kind='mean' if method.startswith('mean_') else 'matched' if method.startswith('last0') else 'partial' if method.startswith('last') else 'frozen'
    if kind=='mean':model=nonimage.Encoder('text',14,vocab).to(args.device)
    elif kind in ('frozen','matched'):model=Adapter().to(args.device)
    else:
        bert=extract.load_bert(args.modeldir,'cpu',attention=args.attention);common.set_seed(seed);depth=int(method[4]);model=PartialBert(bert,depth,*d.get('_cls_scaler',(None,None))).to(args.device);del bert
    initial=common.state_hash(model.state_dict());initial_head=common.state_hash(model.adapter.state_dict() if kind=='partial' else model.state_dict());fit=d['split']['fit'];counts=np.bincount(d['y'][fit],minlength=14)
    if kind in ('partial','matched'):
        if kind=='partial':
            enc=[p for p in model.layers.parameters() if p.requires_grad];head=list(model.adapter.parameters());groups=[dict(params=enc,lr=lr),dict(params=head,lr=lr*10)]
        else:groups=[dict(params=model.parameters(),lr=lr*10)]
        optimizer=torch.optim.AdamW(groups,weight_decay=.01)
        epochs=args.partial_epochs;batch_size=args.partial_batch_size
    else:optimizer=torch.optim.SGD(model.parameters(),lr=lr,momentum=.9,nesterov=True);epochs=args.adapter_epochs;batch_size=args.batch_size
    balanced=method.startswith('bert_balanced');eq=method.endswith('_eq');bs=method.endswith('_bsoftmax')
    lengths=d.get('_lengths') if kind in ('partial','matched') and args.partial_length_buckets else None
    steps=len(epoch_batches(fit,d['y'],seed,0,batch_size,balanced,lengths));total=epochs*steps;history=[];start=0;checkpoint=target/'checkpoint.pt';amp=(kind=='partial' and args.partial_precision=='amp_float16' and args.device=='mps');scaler=torch.amp.GradScaler('mps',enabled=amp,init_scale=1024.)
    if checkpoint.exists():
        budget=validate_training_artifact(target)
        if budget['planned_epochs']!=epochs or budget['steps_per_epoch']!=steps or budget['identity']!=identity:raise ValueError('Checkpoint budget protocol mismatch')
        ck=torch.load(checkpoint,map_location=args.device,weights_only=False)
        if ck['identity']!=identity or ck['initial_state_sha256']!=initial:raise ValueError('Checkpoint identity mismatch')
        model.load_state_dict(ck['model']);optimizer.load_state_dict(ck['optimizer']);restore_rng(ck['rng']);history=ck['history'];start=ck['completed_epochs']
        validate_history(history,start,epochs,steps)
        if start!=budget['completed_epochs']:raise ValueError('Checkpoint/sidecar epoch mismatch')
        if amp:scaler.load_state_dict(ck['scaler'])
    def forward(ids):
        if kind=='partial':
            hidden,lookup,mask=prefix;loc=lookup[ids]
            if np.any(loc<0):raise ValueError('Attempted uncached/held-out prefix row')
            length=int(((mask[ids].sum(1).max()+31)//32)*32) if args.partial_length_buckets else 128
            h=torch.from_numpy(np.ascontiguousarray(hidden[loc,:length])).to(args.device)
            if not amp:h=h.to(torch.float32)
            m=torch.tensor(np.array(mask[ids,:length],copy=True),dtype=torch.long,device=args.device)
            with torch.autocast(device_type=args.device,dtype=torch.float16,enabled=amp):return model(h,m)
        if kind=='mean':x=torch.tensor(np.array(inputs[ids],copy=True),dtype=torch.long,device=args.device)
        else:x=torch.tensor(np.array(inputs[ids],copy=True),dtype=torch.float32,device=args.device)
        return model(x)
    for epoch in range(start,epochs):
        model.train();batches_ids=epoch_batches(fit,d['y'],seed,epoch,batch_size,balanced,lengths);loss_sum=0.;eligible=np.zeros(14,dtype=np.int64);t=time.perf_counter();batches=0
        for batch_index,ids in enumerate(batches_ids):
            labels=torch.tensor(np.array(d['y'][ids],copy=True),device=args.device)
            step=epoch*steps+batch_index;scale=common.learning_rate(step,total,min(1,epochs)*steps,1.)
            for i,group in enumerate(optimizer.param_groups):group['lr']=lr*scale*(10 if kind=='matched' or kind=='partial' and i==1 else 1)
            optimizer.zero_grad(set_to_none=True);logits,features=forward(ids)
            loss=balanced_softmax(logits,labels,counts) if bs else F.cross_entropy(logits,labels)
            if kind not in ('partial','matched'):loss=loss+.0005*nonimage.linear_l2(model)
            if eq:loss=loss+args.eq_weight*common.equalization(features,labels,14)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            scaler.scale(loss).backward();scaler.unscale_(optimizer);norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
            if not torch.isfinite(norm):raise RuntimeError('Nonfinite gradient')
            scaler.step(optimizer);scaler.update();loss_sum+=float(loss.detach().cpu())*len(ids);eligible+=(np.bincount(d['y'][ids],minlength=14)>=2);batches+=1
        duration=time.perf_counter()-t;history.append(dict(epoch=epoch+1,optimizer_updates=(epoch+1)*steps,loss=loss_sum/len(fit),seconds=duration,batches=batches,EQ_active=eq,EQ_eligible_batches_per_class=eligible.tolist()))
        common.save_torch(checkpoint,dict(identity=identity,initial_state_sha256=initial,model=model.state_dict(),optimizer=optimizer.state_dict(),rng=get_rng(),completed_epochs=epoch+1,history=history,scaler=scaler.state_dict()))
        write_json(target/'history.json',history)
        write_json(target/'budget.json',dict(identity=identity,initial_state_sha256=initial,planned_epochs=epochs,completed_epochs=epoch+1,steps_per_epoch=steps,fit_examples_per_epoch=len(fit),files={n:sha(target/n) for n in ('checkpoint.pt','history.json')}))
        print(json.dumps(dict(stage='train',method=method,seed=seed,lr=lr,**history[-1])),flush=True)
    def predict(ids):
        model.eval();logits=np.empty((len(ids),14),dtype=np.float32)
        if kind=='partial' and args.partial_length_buckets:
            evaluation_buckets=((d['_lengths'][ids]+31)//32)*32
            groups=[np.flatnonzero(evaluation_buckets==k) for k in (32,64,96,128)]
        else:groups=[np.arange(len(ids),dtype=np.int64)]
        with torch.inference_mode():
            for positions in groups:
                for i in range(0,len(positions),batch_size):
                    chosen=positions[i:i+batch_size];logits[chosen]=forward(ids[chosen])[0].to(torch.float32).cpu().numpy()
        validate_logits(logits,d['y'][ids]);return logits
    if start==epochs and (target/'selection.json').exists():
        selection=json.loads((target/'selection.json').read_text());validate_training_artifact(target,selection)
        if selection['identity']!=identity or selection['final_state_sha256']!=common.state_hash(model.state_dict()) or sha(target/'tune_logits.npy')!=selection['tune_logits_sha256']:raise ValueError('Completed fit selection integrity mismatch')
        return selection,predict,model
    tune=predict(d['split']['tune']);np.save(target/'tune_logits.npy',tune,allow_pickle=False)
    score=float(f1_score(d['y'][d['split']['tune']],tune.argmax(1),labels=np.arange(14),average='macro',zero_division=0))
    selection=dict(identity=identity,tune_macro_f1=score,initial_state_sha256=initial,initial_head_state_sha256=initial_head,final_state_sha256=common.state_hash(model.state_dict()),epochs=epochs,updates=total,
        checkpoint_sha256=sha(checkpoint),tune_logits_sha256=sha(target/'tune_logits.npy'),budget_sha256=sha(target/'budget.json'),train_seconds=sum(r['seconds'] for r in history),device=args.device,precision='amp_float16' if amp else 'fp32',master_parameter_dtype='float32',fit_examples=len(fit),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),total_parameters=sum(p.numel() for p in model.parameters()))
    write_json(target/'selection.json',selection)
    return selection,predict,model


def prepare_protocol(d,args):
    if any(m.startswith('last') for m in args.methods) and (d.get('cls_meta',{}).get('precision')!=args.partial_precision or d.get('cls_meta',{}).get('attention')!=args.attention):raise ValueError('CLS cache/backend precision differs from matched tuning protocol')
    lengths=np.asarray(np.load(d['data']/'attention_mask.npy',mmap_mode='r').sum(1),dtype=np.int64);bucket=((lengths[d['split']['fit']]+31)//32)*32
    counts={str(k):int(np.sum(bucket==k)) for k in (32,64,96,128)}
    partial_steps=sum(math.ceil(v/args.partial_batch_size) for v in counts.values()) if args.partial_length_buckets else math.ceil(len(d['split']['fit'])/args.partial_batch_size)
    record=dict(version=1,dataset_fingerprint=d['meta']['fingerprint'],cache_fingerprint=d.get('cache_meta',{}).get('fingerprint'),cls_cache_fingerprint=d.get('cls_meta',{}).get('fingerprint'),
        methods=list(args.methods),seeds=list(args.seeds),fit_examples=len(d['split']['fit']),fit_length_bucket_counts=counts,adapter_steps_per_epoch=math.ceil(len(d['split']['fit'])/args.batch_size),partial_steps_per_epoch=partial_steps,adapter_epochs=args.adapter_epochs,partial_epochs=args.partial_epochs,batch_size=args.batch_size,
        partial_batch_size=args.partial_batch_size,max_words=args.max_words,attention=args.attention,partial_length_buckets=args.partial_length_buckets,inference_order='LengthbucketpartialpredictionsscatteredbacktooriginalsourceIDorder',adapter_lrs=list(args.adapter_lrs),partial_lrs=list(args.partial_lrs),eq_weight=args.eq_weight,
        sampling='natural shuffled rows or class-uniform replacement, equal updates/examples budget',
        objective='CE; Balanced Softmax is CE(logits+log fit counts), sampled naturally; balanced methods use CE without extra class weights',
        tuning='two final-epoch LR candidates per method at dedicated seed9; freeze one global method LR before ten final seeds10..19; macro-F1 on tune only; deterministic smaller-LR tie break',
        calibration='CE/naturalEQ natural training prior corrected by tau grid; BS and class-balanced objectives use uniform effective prior and no duplicate LA readout',
        TAUS=list(TAUS),primary_comparisons=[list(x) for x in PRIMARY],source_sha256=sha(__file__),extractor_sha256=sha(Path(extract.__file__)),nonimage_helper_sha256=sha(Path(nonimage.__file__)),numeric_helper_sha256=sha(Path(common.__file__)),preparation_helper_sha256=sha(Path(extract.source.__file__)),
        evaluation='balanced fresh train-derived confirmation primary; native 70k test exploratory only for frozen/mean arms',
        mean_baseline='fit-only50k vocabulary;128token cap; random128DEmbeddingBag mean followed by256→128adapter',
        frozen_baseline='Fit-only768Dmean/std sharedscaler; BERT first10 frozen eval, FP16 hidden cache roundtrip, last2 frozen eval;768→256→128adapter',
        matched='FrozenBERTCLS cache, fit-onlyCLS scaler; same3epochs1024batchbucketorder/AdamWheadLR10×referenceLR aspartial; last2blocks frozen eval; samefit scaler; identicalheadinitialization; transformer dropout difference makespartialtraining-regime comparison, notgradient-only',
        partial='CLS pooling withfit-onlyCLSscaler matchedlast0; finalBERTblock computesCLSquery/output only,K/Valltokens; exactdropout0,marginaldropoutlaw withchangedRNGordering; same prefix/adapter; last1 orlast2 blocks train; frozen layer10 stays eval forlast1;AdamWencoderLR,adapter10×LR',
        device_by_kind={'frozen':'cpu','mean':'cpu','matched':'cpu','partial':'mps'},partial_precision=args.partial_precision,grad_scaler_init_scale=1024.,prefix_transfer='native float16 direct toMPS underAMP; FP32masterweights',threads=args.threads,torch_version=torch.__version__,
        pretraining_overlap_risk='BERT pretrained Wikipedia; DBpedia abstracts derive from Wikipedia, so fresh supervised partitions do not establish independence from pretraining texts; this is a transfer/model-package comparison')
    record['fingerprint']=fingerprint(record);path=args.out/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=record:raise ValueError('Output protocol mismatch')
    write_json(path,record);args.protocol_fingerprint=record['fingerprint'];return record


def run(args):
    if len(set(args.seeds))!=len(args.seeds) or 9 in args.seeds or len(args.seeds)<2:raise ValueError('Final seeds must be unique, exclude tuning seed9 and contain at least2pairs')
    if any(not math.isfinite(v) or v<=0 for v in list(args.adapter_lrs)+list(args.partial_lrs)):raise ValueError('Learning rates must be finite and positive')
    if any(x<=0 for x in (args.threads,args.batch_size,args.partial_batch_size,args.adapter_epochs,args.partial_epochs,args.max_words)):raise ValueError('Invalid positive training budget')
    torch.set_num_threads(args.threads);args.out.mkdir(parents=True,exist_ok=True);d=load_data(args.data,args.cache)
    worker_methods=args.only_methods if args.only_methods else args.methods
    if any(args.device!=('mps' if m.startswith('last1') or m.startswith('last2') else 'cpu') for m in worker_methods):raise ValueError('Worker device violates frozen device_by_kind policy')
    protocol=prepare_protocol(d,args);frozen=np.asarray(d['features']);fit=d['split']['fit']
    # Fit-only scaling reused in all frozen BERT arms; partial encoders retain the same frozen fit-only scaler while their
    # output distribution evolves during tuning, an explicit shared control.
    mean=frozen[fit].mean(0,dtype=np.float64);std=frozen[fit].std(0,dtype=np.float64);std[std==0]=1
    standardized=(frozen-mean)/std;standardized=standardized.astype(np.float32);d['_mean_scaler']=(mean,std);d['_lengths']=np.asarray(np.load(d['data']/'attention_mask.npy',mmap_mode='r').sum(1),dtype=np.int64)
    common.save_npz(args.out/'mean_scaler.npz',mean=mean,std=std)
    if any(m.startswith('last') for m in args.methods):
        if 'cls_features' not in d:raise ValueError('Matched/fine-tunedCLS arms require the prospective separateCLS cache')
        cls_raw=np.asarray(d['cls_features']);lookup=d['cls_lookup'];cls_mean=cls_raw[lookup[fit]].mean(0,dtype=np.float64);cls_std=cls_raw[lookup[fit]].std(0,dtype=np.float64);cls_std[cls_std==0]=1
        cls_standardized=np.zeros((len(d['y']),768),dtype=np.float32);cls_standardized[np.flatnonzero(lookup>=0)]=((cls_raw-cls_mean)/cls_std).astype(np.float32);d['_cls_scaler']=(cls_mean,cls_std)
        common.save_npz(args.out/'cls_scaler.npz',mean=cls_mean,std=cls_std)
    common.save_npz(args.out/'partition.npz',**{k+'_ids':d['split'][k] for k in extract.PARTS},**{k+'_labels':np.array(d['y'][d['split'][k]]) for k in extract.PARTS},rank=d['split']['rank'])
    mean_x=None;vocab=None;prefix=None
    if any(m.startswith('mean_') for m in worker_methods):mean_x,vocab=mean_inputs(d,args)
    if any(m.startswith('last1') or m.startswith('last2') for m in worker_methods):prefix=prefix_inputs(d)
    worker_methods=args.only_methods if args.only_methods else args.methods
    if any(m not in args.methods for m in worker_methods):raise ValueError('Worker method outside declared full protocol')
    for method in worker_methods:
        cell=args.out/method/'seed_9';cell.mkdir(parents=True,exist_ok=True);global_choice=args.out/method/'hyperparameters.json'
        if global_choice.exists():
            choice=json.loads(global_choice.read_text())
            validate_hyperparameters(args.out,method,choice,protocol,np.array(d['y'][d['split']['tune']]))
            continue
        if args.stage=='evaluate':raise ValueError('Evaluation requires all hyperparameter choices frozen first')
        candidates=[];lrs=args.partial_lrs if method.startswith('last') else args.adapter_lrs;inputs=mean_x if method.startswith('mean_') else cls_standardized if method.startswith('last0') else standardized
        for lr in sorted(lrs):
            selection,predict,model=train_model(d,method,9,lr,args,inputs,vocab,prefix);candidates.append(selection);del predict,model
            if args.device=='mps':torch.mps.empty_cache()
        best=max(candidates,key=lambda r:r['tune_macro_f1'])
        write_json(global_choice,dict(protocol_fingerprint=protocol['fingerprint'],tuning_seed=9,candidates=candidates,selected_lr=best['identity']['lr'],confirmation_scored_at_freeze=False))
    if args.stage=='tune':return dict(status='global_method_hyperparameters_frozen_without_confirmation',methods=worker_methods,tuning_fits=2*len(worker_methods),protocol_fingerprint=protocol['fingerprint'])
    if args.stage!='evaluate':
        for method,seed in itertools.product(worker_methods,args.seeds):
            inputs=mean_x if method.startswith('mean_') else cls_standardized if method.startswith('last0') else standardized;choice=json.loads((args.out/method/'hyperparameters.json').read_text())
            selection,predict,model=train_model(d,method,seed,choice['selected_lr'],args,inputs,vocab,prefix);del predict,model
            if args.device=='mps':torch.mps.empty_cache()
        for method in worker_methods:
            records=[]
            for seed in args.seeds:
                choice=json.loads((args.out/method/'hyperparameters.json').read_text());folder=args.out/method/f'seed_{seed}'/f"lr_{choice['selected_lr']:g}";selection=json.loads((folder/'selection.json').read_text());validate_training_artifact(folder,selection)
                records.append(dict(seed=seed,selection_sha256=sha(folder/'selection.json'),budget_sha256=sha(folder/'budget.json')))
            write_json(args.out/method/'completed_fits.json',dict(protocol_fingerprint=protocol['fingerprint'],final_seed_count=len(records),records=records,confirmation_inferred=False))
        if args.stage=='train':return dict(status='trained_waiting_for_whole_study_confirmation_gate',methods=worker_methods,final_fits=len(worker_methods)*len(args.seeds),protocol_fingerprint=protocol['fingerprint'])
    frozen_choices={};trained_fits=[]
    for method in args.methods:
        choice_path=args.out/method/'hyperparameters.json'
        if not choice_path.exists():raise ValueError('Whole-study confirmation gate awaits all global method choices')
        choice=json.loads(choice_path.read_text());validate_hyperparameters(args.out,method,choice,protocol,np.array(d['y'][d['split']['tune']]))
        frozen_choices[method]=sha(choice_path)
        for seed in args.seeds:
            folder=args.out/method/f'seed_{seed}'/f"lr_{choice['selected_lr']:g}"
            if not (folder/'selection.json').exists():raise ValueError('Whole-study confirmation gate awaits all final training fits')
            selection=json.loads((folder/'selection.json').read_text());budget=validate_training_artifact(folder,selection)
            expected_epochs=protocol['partial_epochs'] if method.startswith('last') else protocol['adapter_epochs'];expected_steps=protocol['partial_steps_per_epoch'] if method.startswith('last') else protocol['adapter_steps_per_epoch']
            if budget['planned_epochs']!=expected_epochs or budget['completed_epochs']!=expected_epochs or budget['steps_per_epoch']!=expected_steps or selection['identity']!={'protocol':protocol['fingerprint'],'method':method,'seed':seed,'lr':choice['selected_lr']}:raise ValueError('Whole-study gate rejects a short or wrong-identity final fit')
            trained_fits.append(dict(method=method,seed=seed,selection_sha256=sha(folder/'selection.json')))
    global_gate=dict(protocol_fingerprint=protocol['fingerprint'],hyperparameter_choices_sha256=frozen_choices,trained_fits=trained_fits,confirmation_inferred_at_freeze=False)
    gate_path=args.out/'whole_study_confirmation_gate.json'
    if gate_path.exists() and json.loads(gate_path.read_text())!=global_gate:raise ValueError('Whole-study immutable confirmation gate differs')
    write_json(gate_path,global_gate)
    if any(args.out.glob('*/seed_*/metrics.json')):verify(args.out)
    for method,seed in itertools.product(worker_methods,args.seeds):
        cell=args.out/method/f'seed_{seed}';result_path=cell/'metrics.json'
        if result_path.exists():
            r=json.loads(result_path.read_text())
            if r['protocol_fingerprint']!=protocol['fingerprint']:raise ValueError('Completed cell protocol mismatch')
            if sha(cell/'predictions.npz')!=r['predictions_sha256']:raise ValueError('Completed predictions hash mismatch')
            continue
        choice=json.loads((args.out/method/'hyperparameters.json').read_text());inputs=mean_x if method.startswith('mean_') else cls_standardized if method.startswith('last0') else standardized
        if (cell/'frozen_choice.json').exists() and json.loads((cell/'frozen_choice.json').read_text())!=choice:raise ValueError('Preserving inconsistent earlier cell choice')
        write_json(cell/'frozen_choice.json',choice)
        selection,predict,model=train_model(d,method,seed,choice['selected_lr'],args,inputs,vocab,prefix)
        # This stage begins only after hyperparameters are saved and frozen.
        cal=predict(d['split']['calibration']);adjustment=None
        needs_adjustment=(method.endswith('_ce') and 'balanced' not in method or method=='bert_natural_eq')
        if needs_adjustment:
            counts=np.bincount(d['y'][fit],minlength=14);_,adjustment=adjust(cal,np.array(d['y'][d['split']['calibration']]),np.empty((0,14),dtype=np.float32),counts/counts.sum())
        write_json(cell/'calibration_choice.json',dict(protocol_fingerprint=protocol['fingerprint'],method=method,seed=seed,adjustment=adjustment,confirmation_inferred_at_freeze=False))
        confirm=predict(d['split']['confirmation'])
        raw={'calibration_logits':cal,'confirmation_logits':confirm,'calibration_labels':np.array(d['y'][d['split']['calibration']]),
            'confirmation_labels':np.array(d['y'][d['split']['confirmation']]),'calibration_ids':d['split']['calibration'],'confirmation_ids':d['split']['confirmation'],'rank':d['split']['rank']}
        metrics={method:common.metrics(raw['confirmation_labels'],confirm.argmax(1),d['split']['rank'])}
        if adjustment:
            adjusted=confirm-adjustment['tau']*np.log(adjustment['prior']).astype(np.float32);metrics[method+'_la']=common.metrics(raw['confirmation_labels'],adjusted.argmax(1),d['split']['rank'])
        native=None
        if not method.startswith('last'):
            nl=predict(d['split']['native_test']);raw['native_test_logits']=nl;raw['native_test_labels']=np.array(d['y'][d['split']['native_test']]);raw['native_test_ids']=d['split']['native_test']
            native={method:common.metrics(raw['native_test_labels'],nl.argmax(1),d['split']['rank'])}
            if adjustment:
                nl=nl-adjustment['tau']*np.log(adjustment['prior']).astype(np.float32);native[method+'_la']=common.metrics(raw['native_test_labels'],nl.argmax(1),d['split']['rank'])
        common.save_npz(cell/'predictions.npz',**raw)
        result=dict(protocol_fingerprint=protocol['fingerprint'],method=method,seed=seed,selection=selection,adjustment=adjustment,metrics=metrics,native_exploratory=native,
            predictions_sha256=sha(cell/'predictions.npz'),completed_utc=common.now())
        write_json(result_path,result);print(json.dumps(dict(stage='cell_complete',method=method,seed=seed,metrics=metrics)),flush=True);del predict,model
        if args.device=='mps':torch.mps.empty_cache()
    return verify(args.out)


def cls_cache(args):
    """Derive prospective frozen CLS representations without evaluating labels."""
    torch.set_num_threads(args.threads);d=load_data(args.data,args.cache);hidden,lookup,mask=prefix_inputs(d)
    ids=np.load(args.cache/'prefix_row_ids.npy');lengths=mask[ids].sum(1);buckets=((lengths+31)//32)*32;order=np.argsort(buckets,kind='stable')
    target=args.cache/'cls';target.mkdir(parents=True,exist_ok=True)
    identity=dict(version=1,dataset_fingerprint=d['meta']['fingerprint'],prefix_sha256=d['cache_meta']['prefix_sha256'],prefix_row_ids_sha256=d['cache_meta']['prefix_row_ids_sha256'],
        model_assets_sha256=sha(args.modeldir/'assets.json'),source_sha256=sha(__file__),extractor_sha256=sha(Path(extract.__file__)),rows=len(ids),shape=[len(ids),768],
        pooling='Final hidden CLS token; full frozen block10 then exact CLS-query-only frozen block11',attention=args.attention,precision=args.partial_precision,
        input_precision='Shared first10 FP16 cache transferred directly; FP32modelmasterweights',device=args.device,batch_size=args.partial_batch_size,length_buckets=[32,64,96,128],
        dropout='All pretrained blocks eval; no labels read for scoring or selection')
    fp=fingerprint(identity);reservation=target/'reservation.json';marker=target/'progress.json';path=target/'features.npy'
    if reservation.exists():
        if json.loads(reservation.read_text())!={'fingerprint':fp,'identity':identity}:raise ValueError('Preserving CLS cache with another identity')
    else:
        if path.exists() or marker.exists():raise ValueError('CLS arrays lack immutable reservation')
        write_json(reservation,dict(fingerprint=fp,identity=identity))
    if marker.exists():
        state=json.loads(marker.read_text());features=np.lib.format.open_memmap(path,mode='r+');begin=extract.verify_block_journal(state,fp,order,ids,features);blocks=state['blocks'];times=state['batch_seconds']
    else:features=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(len(ids),768));begin=0;blocks=[];times=[]
    if features.shape!=(len(ids),768) or features.dtype!=np.float32:raise ValueError('CLS cache array shape/dtype mismatch')
    if (target/'cls_cache.json').exists():
        receipt=json.loads((target/'cls_cache.json').read_text())
        if begin!=len(ids) or receipt['fingerprint']!=fp or sha(path)!=receipt['feature_sha256'] or sha(target/'row_ids.npy')!=receipt['row_ids_sha256']:raise ValueError('Completed CLS cache integrity mismatch')
        return receipt
    model=extract.load_bert(args.modeldir,args.device,attention=args.attention);layers=model.encoder.layer[10:];started=time.perf_counter();i=begin;committed=begin;amp=args.partial_precision=='amp_float16'
    with torch.inference_mode(),torch.autocast(device_type=args.device,dtype=torch.float16,enabled=amp):
        while i<len(ids):
            j=min(i+args.partial_batch_size,len(ids));positions=order[i:j];length=int(buckets[positions[0]]);positions=positions[buckets[positions]==length];j=i+len(positions)
            raw_ids=ids[positions];xx=torch.from_numpy(np.ascontiguousarray(hidden[positions,:length])).to(args.device)
            if not amp:xx=xx.to(torch.float32)
            mm=torch.tensor(np.array(mask[raw_ids,:length],copy=True),dtype=torch.long,device=args.device)
            if not torch.all(mm[:,0]==1):raise ValueError('CLS token is not attended')
            extended=(1.-mm[:,None,None,:].to(torch.float32))*torch.finfo(torch.float32).min;t=time.perf_counter()
            xx=layers[0](xx,attention_mask=extended)[0];value=cls_only_layer(layers[1],xx,extended)[:,0].to(torch.float32).cpu().numpy()
            if value.shape!=(j-i,768) or not np.isfinite(value).all():raise ValueError('Invalid finite CLS features')
            features[positions]=value;times.append(time.perf_counter()-t)
            if len(times)%10==0 or j==len(ids):
                features.flush();positions_committed=order[committed:j];blocks.append(dict(begin=committed,end=j,features_sha256=extract.array_digest(features,positions_committed)));committed=j
                write_json(marker,dict(fingerprint=fp,completed_rows=j,blocks=blocks,batch_seconds=times));print(json.dumps(dict(stage='frozen_CLS_cache',rows=j,total=len(ids),elapsed_seconds=time.perf_counter()-started)),flush=True)
            i=j
    features.flush();np.save(target/'row_ids.npy',ids,allow_pickle=False)
    receipt=dict(**identity,fingerprint=fp,complete=True,feature_sha256=sha(path),row_ids_sha256=sha(target/'row_ids.npy'),journal_sha256=sha(marker),committed_blocks=len(blocks),inference_seconds=sum(times),wall_seconds=time.perf_counter()-started)
    write_json(target/'cls_cache.json',receipt);return receipt


def paired_stats(differences):
    d=np.asarray(differences,dtype=np.float64);n=len(d);sd=float(d.std(ddof=1));half=float(stats.t.ppf(.975,n-1)*sd/math.sqrt(n))
    observed=abs(float(d.mean()));exact=float(np.mean(np.abs((np.asarray(list(itertools.product((-1.,1.),repeat=n)))*d).mean(1))>=observed-1e-12))
    tp=float(stats.ttest_1samp(d,0).pvalue) if np.any(d!=d[0]) else (1. if d[0]==0 else 0.)
    return dict(n=n,differences=d.tolist(),mean=float(d.mean()),sd=sd,ci95=[float(d.mean())-half,float(d.mean())+half],signflip_p=exact,t_p=tp)


def verify(out):
    out=Path(out);protocol=json.loads((out/'protocol.json').read_text());rows=[];summary=[];verified=[]
    with np.load(out/'partition.npz',allow_pickle=False) as z:partition={k:z[k] for k in z.files}
    gate_path=out/'whole_study_confirmation_gate.json'
    if any(out.glob('*/seed_*/metrics.json')) and not gate_path.exists():raise ValueError('Scored cells exist without whole-study confirmation freeze')
    if gate_path.exists():
        gate=json.loads(gate_path.read_text())
        if gate['protocol_fingerprint']!=protocol['fingerprint'] or gate['confirmation_inferred_at_freeze'] is not False:raise ValueError('Invalid whole-study confirmation gate')
        if set(gate['hyperparameter_choices_sha256'])!=set(protocol['methods']):raise ValueError('Incomplete or extra global choices in confirmation gate')
        expected_grid={(m,s) for m in protocol['methods'] for s in protocol['seeds']};actual_grid=[(r['method'],r['seed']) for r in gate['trained_fits']]
        if set(actual_grid)!=expected_grid or len(actual_grid)!=len(expected_grid):raise ValueError('Confirmation gate lacks exact unique full final-fit grid')
        for method,digest in gate['hyperparameter_choices_sha256'].items():
            if sha(out/method/'hyperparameters.json')!=digest:raise ValueError('Global hyperparameter choice changed after confirmation freeze')
        for record in gate['trained_fits']:
            method,seed=record['method'],record['seed'];choice=json.loads((out/method/'hyperparameters.json').read_text());folder=out/method/f'seed_{seed}'/f"lr_{choice['selected_lr']:g}"
            if sha(folder/'selection.json')!=record['selection_sha256']:raise ValueError('Training selection changed after confirmation freeze')
    all_ids=np.concatenate([partition[k+'_ids'] for k in extract.PARTS])
    if len(all_ids)!=len(np.unique(all_ids)):raise ValueError('Saved partition overlap')
    for method,seed in itertools.product(protocol['methods'],protocol['seeds']):
        cell=out/method/f'seed_{seed}';path=cell/'metrics.json'
        if not path.exists():continue
        result=json.loads(path.read_text());p=cell/'predictions.npz'
        if result['protocol_fingerprint']!=protocol['fingerprint'] or result['method']!=method or result['seed']!=seed or sha(p)!=result['predictions_sha256']:raise ValueError('Cell hash/protocol mismatch')
        chosen=json.loads((cell/'frozen_choice.json').read_text())
        if chosen!=json.loads((out/method/'hyperparameters.json').read_text()):raise ValueError('Cell/global hyperparameter freeze differs')
        validate_hyperparameters(out,method,chosen,protocol,partition['tune_labels'])
        calibration_choice=json.loads((cell/'calibration_choice.json').read_text())
        if calibration_choice!={'protocol_fingerprint':protocol['fingerprint'],'method':method,'seed':seed,'adjustment':result['adjustment'],'confirmation_inferred_at_freeze':False}:raise ValueError('Calibration freeze differs from scored choice')
        best=max(chosen['candidates'],key=lambda r:r['tune_macro_f1'])
        if chosen['selected_lr']!=best['identity']['lr'] or result['selection']['identity']!={'protocol':protocol['fingerprint'],'method':method,'seed':seed,'lr':chosen['selected_lr']}:raise ValueError('LR choice inconsistent with frozen candidates')
        finalfolder=cell/f"lr_{chosen['selected_lr']:g}"
        budget=validate_training_artifact(finalfolder,result['selection'])
        if budget['planned_epochs']!=(protocol['partial_epochs'] if method.startswith('last') else protocol['adapter_epochs']) or budget['steps_per_epoch']!=(protocol['partial_steps_per_epoch'] if method.startswith('last') else protocol['adapter_steps_per_epoch']):raise ValueError('Final budget differs from protocol')
        if sha(finalfolder/'budget.json')!=result['selection']['budget_sha256']:raise ValueError('Final training budget receipt mismatch')
        if sha(finalfolder/'checkpoint.pt')!=result['selection']['checkpoint_sha256'] or sha(finalfolder/'tune_logits.npy')!=result['selection']['tune_logits_sha256']:raise ValueError('Final candidate integrity mismatch')
        for candidate in chosen['candidates']:
            folder=out/method/'seed_9'/f"lr_{candidate['identity']['lr']:g}";tune=np.load(folder/'tune_logits.npy');validate_logits(tune,np.arange(len(tune)))
            if sha(folder/'tune_logits.npy')!=candidate['tune_logits_sha256'] or sha(folder/'checkpoint.pt')!=candidate['checkpoint_sha256']:raise ValueError('Candidate integrity failure')
            tune_score=float(f1_score(partition['tune_labels'],tune.argmax(1),labels=np.arange(14),average='macro',zero_division=0))
            if tune_score!=candidate['tune_macro_f1']:raise ValueError('Tune score replay mismatch')
        with np.load(p,allow_pickle=False) as z:
            for part in ('calibration','confirmation'):
                validate_logits(z[part+'_logits'],z[part+'_labels'])
                if not np.array_equal(z[part+'_labels'],partition[part+'_labels']) or not np.array_equal(z[part+'_ids'],partition[part+'_ids']):raise ValueError('Logit source partition alignment mismatch')
            if not np.array_equal(z['rank'],partition['rank']):raise ValueError('Class rank mismatch')
            raw=common.metrics(z['confirmation_labels'],z['confirmation_logits'].argmax(1),z['rank'])
            if raw!=result['metrics'][method]:raise ValueError('Raw score replay mismatch')
            if result['adjustment']:
                counts=np.bincount(partition['fit_labels'],minlength=14);expected=counts/counts.sum()
                if result['adjustment']['prior']!=expected.tolist():raise ValueError('Calibration prior is not fit-only frequency')
                adjusted,choice=adjust(z['calibration_logits'],z['calibration_labels'],z['confirmation_logits'],result['adjustment']['prior'])
                if choice!=result['adjustment'] or common.metrics(z['confirmation_labels'],adjusted.argmax(1),z['rank'])!=result['metrics'][method+'_la']:raise ValueError('Calibrated score replay mismatch')
            if result['native_exploratory']:
                validate_logits(z['native_test_logits'],z['native_test_labels'])
                if not np.array_equal(z['native_test_ids'],partition['native_test_ids']) or not np.array_equal(z['native_test_labels'],partition['native_test_labels']):raise ValueError('Native logit alignment mismatch')
                if common.metrics(z['native_test_labels'],z['native_test_logits'].argmax(1),z['rank'])!=result['native_exploratory'][method]:raise ValueError('Native raw score replay mismatch')
                if result['adjustment']:
                    nl=z['native_test_logits']-result['adjustment']['tau']*np.log(result['adjustment']['prior']).astype(np.float32)
                    if common.metrics(z['native_test_labels'],nl.argmax(1),z['rank'])!=result['native_exploratory'][method+'_la']:raise ValueError('Native adjusted score replay mismatch')
            for key,m in result['metrics'].items():rows.append(dict(method=key,seed=seed,**m))
        verified.append(dict(method=method,seed=seed,predictions_sha256=result['predictions_sha256'],initial_head_state_sha256=result['selection']['initial_head_state_sha256']))
    for seed in protocol['seeds']:
        heads=[r['initial_head_state_sha256'] for r in verified if r['seed']==seed and not r['method'].startswith('mean_')]
        if len(set(heads))>1:raise ValueError('Unmatched BERT classifier head initializations')
    for method in sorted({r['method'] for r in rows}):
        group=[r for r in rows if r['method']==method];metrics={}
        for key in ('accuracy','macro_f1','macro_recall','head_f1','mid_f1','tail_f1'):
            v=np.array([r[key] for r in group]);metrics[key]=dict(mean=float(v.mean()),sd=float(v.std(ddof=1)) if len(v)>1 else None)
        summary.append(dict(method=method,n=len(group),metrics=metrics))
    comparisons=[]
    for a,b in PRIMARY:
        ga={r['seed']:r for r in rows if r['method']==a};gb={r['seed']:r for r in rows if r['method']==b}
        if set(ga)==set(gb)==set(protocol['seeds']):comparisons.append(dict(a=a,b=b,endpoint='macro_f1',**paired_stats([ga[s]['macro_f1']-gb[s]['macro_f1'] for s in protocol['seeds']])))
    if len(comparisons)==len(PRIMARY):
        for field in ('signflip_p','t_p'):
            order=sorted(range(len(comparisons)),key=lambda i:comparisons[i][field]);running=0.
            for k,i in enumerate(order):running=max(running,min(1.,(len(comparisons)-k)*comparisons[i][field]));comparisons[i][field+'_holm']=running
    result=dict(status='verified_complete' if len(verified)==len(protocol['methods'])*len(protocol['seeds']) else 'incomplete',
        protocol_fingerprint=protocol['fingerprint'],completed_cells=len(verified),expected_cells=len(protocol['methods'])*len(protocol['seeds']),
        readouts=len(rows),verified_cells=verified,summary=summary,paired_comparisons=comparisons,rows=rows,
        verification_scope='Saved numeric logits/labels and hashes replayed, plus candidate choices; no model weights loaded by verifier',
        family='Eight declared macro-F1 comparisons; Holm applied only when complete')
    write_json(out/'aggregate.json',result);return result


def pilot(args):
    torch.set_num_threads(args.threads);d=load_data(args.data,args.cache);prefix=prefix_inputs(d);fit=d['split']['fit'];records=[]
    for depth in (1,2):
        common.set_seed(10);bert=extract.load_bert(args.modeldir,'cpu');model=PartialBert(bert,depth,*d.get('_scaler',(None,None))).to(args.device);del bert
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=2e-5)
        for bs in args.pilot_batches:
            t=time.perf_counter();losses=[]
            for step in range(args.pilot_steps):
                ids=fit[step*bs:(step+1)*bs];hidden,lookup,mask=prefix
                x=torch.tensor(np.array(hidden[lookup[ids]],copy=True),dtype=torch.float32,device=args.device);m=torch.tensor(np.array(mask[ids],copy=True),dtype=torch.long,device=args.device);labels=torch.tensor(np.array(d['y'][ids],copy=True),device=args.device)
                optimizer.zero_grad(set_to_none=True);logits,_=model(x,m);loss=F.cross_entropy(logits,labels);loss.backward();optimizer.step();losses.append(float(loss.detach().cpu()))
            elapsed=time.perf_counter()-t;records.append(dict(depth=depth,batch_size=bs,steps=args.pilot_steps,seconds=elapsed,examples_per_second=bs*args.pilot_steps/elapsed,losses=losses,only_fit_rows=True))
            print(json.dumps(records[-1]),flush=True)
        del model
        if args.device=='mps':torch.mps.empty_cache()
    write_json(args.out/'train_only_pilot.json',dict(records=records,confirmation_evaluated=False,tune_evaluated=False,weights_discarded=True));return records


def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
    for command in ('run','pilot','cls-cache'):
        q=s.add_parser(command);q.add_argument('--data',type=Path,required=True);q.add_argument('--cache',type=Path,required=True);q.add_argument('--modeldir',type=Path,required=True);q.add_argument('--out',type=Path,required=True)
        q.add_argument('--stage',choices=['tune','train','evaluate','all'],default='train');q.add_argument('--device',choices=['cpu','mps','cuda'],default='cpu');q.add_argument('--threads',type=int,default=2)
        q.add_argument('--methods',nargs='+',choices=METHODS,default=list(METHODS));q.add_argument('--seeds',nargs='+',type=int,default=list(range(10,20)));q.add_argument('--only-methods',nargs='+',choices=METHODS)
        q.add_argument('--adapter-epochs',type=int,default=50);q.add_argument('--partial-epochs',type=int,default=3);q.add_argument('--batch-size',type=int,default=1024);q.add_argument('--partial-batch-size',type=int,default=1024)
        q.add_argument('--adapter-lrs',nargs='+',type=float,default=[.01,.05]);q.add_argument('--partial-lrs',nargs='+',type=float,default=[2e-5,5e-5]);q.add_argument('--attention',choices=['eager','sdpa'],default='sdpa');q.add_argument('--no-partial-length-buckets',dest='partial_length_buckets',action='store_false',default=True);q.add_argument('--partial-precision',choices=['fp32','amp_float16'],default='amp_float16');q.add_argument('--eq-weight',type=float,default=.05);q.add_argument('--max-words',type=int,default=50000)
        q.add_argument('--pilot-batches',nargs='+',type=int,default=[32,64]);q.add_argument('--pilot-steps',type=int,default=5)
    q=s.add_parser('verify');q.add_argument('--out',type=Path,required=True)
    args=p.parse_args();result=verify(args.out) if args.command=='verify' else cls_cache(args) if args.command=='cls-cache' else globals()[args.command](args);print(json.dumps(result,sort_keys=True))
if __name__=='__main__':main()
