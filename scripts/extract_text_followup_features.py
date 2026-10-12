#!/usr/bin/env python3
"""Pinned, resumable BERT acquisition and DBpedia feature preparation.

Only acquisition accesses the network. No trust_remote_code or pickle weight
loading is used. All labels/text row IDs retain their original source alignment.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, math, os, platform, re, shutil, time, urllib.request
from pathlib import Path
import numpy as np
import torch

_spec=importlib.util.spec_from_file_location('text_source',Path(__file__).with_name('prepare_nonimage.py'))
source=importlib.util.module_from_spec(_spec); _spec.loader.exec_module(source)
sha,fingerprint=source.sha,source.fingerprint

def write_json(path,value):
    path=Path(path);temporary=path.with_name(path.name+f'.tmp-{os.getpid()}')
    with open(temporary,'w',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)
REPO='google-bert/bert-base-uncased'
ASSETS=('config.json','model.safetensors','tokenizer.json','tokenizer_config.json','vocab.txt','README.md')
PARTS=('fit','tune','calibration','confirmation','native_test')


def masked_mean(hidden,mask):
    if hidden.ndim!=3 or mask.shape!=hidden.shape[:2]: raise ValueError('Hidden/mask alignment mismatch')
    if not torch.all(mask.sum(1)>0): raise ValueError('Empty attention mask')
    value=hidden.to(torch.float32)
    return (value*mask.unsqueeze(-1).to(value.dtype)).sum(1)/mask.sum(1,keepdim=True).to(value.dtype)


def safe_name(name):
    if name not in ASSETS: raise ValueError('Unapproved asset')
    return name


def acquire(out):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    api=f'https://huggingface.co/api/models/{REPO}?blobs=true'
    with urllib.request.urlopen(api,timeout=60) as stream: info=json.load(stream)
    revision=info.get('sha','')
    if not re.fullmatch('[0-9a-f]{40}',revision): raise ValueError('Revision is not pinned')
    license=info.get('cardData',{}).get('license')
    if license!='apache-2.0': raise ValueError('Unexpected model license')
    siblings={r['rfilename']:r for r in info['siblings']}; files={}
    for name in ASSETS:
        safe_name(name); spec=siblings[name]; url=f'https://huggingface.co/{REPO}/resolve/{revision}/{name}'
        path=out/name; expected=spec.get('lfs',{}).get('sha256')
        valid=False
        if path.exists() and path.stat().st_size==spec['size']:
            if expected:valid=sha(path)==expected
            else:
                raw=path.read_bytes();valid=hashlib.sha1(f'blob {len(raw)}\0'.encode()+raw).hexdigest()==spec['blobId']
        if not valid:
            temporary=path.with_suffix(path.suffix+'.partial')
            with urllib.request.urlopen(url,timeout=120) as stream,open(temporary,'wb') as target:
                n=0
                for block in iter(lambda:stream.read(1024**2),b''):
                    n+=len(block)
                    if n>spec['size']: raise ValueError('Asset exceeds official size')
                    target.write(block)
            if temporary.stat().st_size!=spec['size']: raise ValueError('Incomplete asset')
            if expected and sha(temporary)!=expected: raise ValueError('Safetensor asset SHA mismatch')
            # Ordinary Git blobs have a verifiable Git object hash.
            if not expected:
                raw=temporary.read_bytes(); digest=hashlib.sha1(f'blob {len(raw)}\0'.encode()+raw).hexdigest()
                if digest!=spec['blobId']: raise ValueError('Git blob digest mismatch')
            temporary.replace(path)
        files[name]=dict(bytes=path.stat().st_size,sha256=sha(path),public_url=url,
                         official_lfs_sha256=expected,official_git_blob=spec['blobId'])
        print(json.dumps(dict(asset=name,**files[name])),flush=True)
    record=dict(model_repo=REPO,revision=revision,license=license,metadata_url=api,files=files,
                loading='local safetensors only; trust_remote_code=False',script_sha256=sha(__file__))
    write_json(out/'assets.json',record); return record


def fresh_splits(labels,canonical,rho=100.,split_seed=20261012,rank_seed=20261013,confirmation=4000,tune=2000,calibration=2000):
    labels=np.asarray(labels); canonical=np.asarray(canonical,dtype=np.int64)
    if not np.isfinite(rho) or rho<1: raise ValueError('Invalid rho')
    if len(np.unique(canonical))!=len(canonical) or np.any(canonical>=560000): raise ValueError('Invalid canonical development IDs')
    rng=np.random.default_rng(split_seed); order=np.random.default_rng(rank_seed).permutation(14)
    byclass=[rng.permutation(canonical[labels[canonical]==c]) for c in range(14)]
    reserve=confirmation+tune+calibration; nmax=min(len(v)-reserve for v in byclass)
    if nmax<2: raise ValueError('Insufficient canonical records')
    parts={n:[] for n in PARTS[:-1]}
    for rank,c in enumerate(order):
        rows=byclass[int(c)]; nf=max(2,int(nmax*rho**(-rank/13)))
        parts['confirmation'].extend(rows[:confirmation]); parts['tune'].extend(rows[confirmation:confirmation+tune])
        parts['calibration'].extend(rows[confirmation+tune:reserve]); parts['fit'].extend(rows[reserve:reserve+nf])
    split={k:np.asarray(v,dtype=np.int64) for k,v in parts.items()}; split['native_test']=np.arange(560000,len(labels),dtype=np.int64)
    split['class_rank']=order.astype(np.int64)
    validate_split(labels,split)
    return split,dict(split_seed=split_seed,rank_seed=rank_seed,nominal_rho=rho,fit_nmax=nmax,
        split_counts={k:len(split[k]) for k in PARTS},
        split_class_counts={k:np.bincount(labels[split[k]],minlength=14).tolist() for k in PARTS},
        confirmation_note='Fresh train-derived confirmation with a new fixed split; no model fitting or hyperparameter selection on confirmation. Native benchmark test is exploratory after earlier analyses.',
        duplicate_note='Canonical normalized exact/content-only groups are assigned as a unit by retaining one representative before splitting.')


def validate_split(labels,split):
    seen=set()
    for name in PARTS:
        ids=split[name]
        if ids.ndim!=1 or ids.dtype!=np.int64 or len(ids)!=len(np.unique(ids)) or np.any(ids<0) or np.any(ids>=len(labels)): raise ValueError('Invalid split IDs')
        if not np.array_equal(np.unique(labels[ids]),np.arange(14)): raise ValueError('Split omits classes')
        vals=set(ids.tolist())
        if vals & seen: raise ValueError('Split overlap')
        seen |= vals
    if not np.array_equal(np.sort(split['class_rank']),np.arange(14)): raise ValueError('Class rank is not a permutation')


def prepare(source_path,out,rho=100.):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    if (out/'dataset.json').exists():
        info=json.loads((out/'dataset.json').read_text())
        for n,h in info['files'].items():
            if sha(out/n)!=h: raise ValueError('Existing prepared dataset corrupted')
        return info
    source_info=source._source_record(source_path,'dbpedia')
    labels,canonical,audit,boundary=source._text(source_path,out,source_info)
    if boundary!=560000: raise ValueError('Unexpected native boundary')
    split,details=fresh_splits(labels,canonical,rho=rho)
    np.save(out/'labels.npy',labels,allow_pickle=False); np.savez_compressed(out/'split.npz',**split)
    np.save(out/'canonical.npy',canonical,allow_pickle=False)
    files={n:sha(out/n) for n in ('texts.jsonl','labels.npy','split.npz','canonical.npy')}
    record=dict(dataset='dbpedia14',source=source_info,rows=len(labels),classes=source.DATASETS['dbpedia']['classes'],
                dedup_audit=audit,files=files,preparation_script_sha256=sha(__file__),**details)
    record['fingerprint']=fingerprint(record); write_json(out/'dataset.json',record); return record


def verify_assets(modeldir):
    modeldir=Path(modeldir); record=json.loads((modeldir/'assets.json').read_text())
    for name,row in record['files'].items():
        safe_name(name)
        if sha(modeldir/name)!=row['sha256']: raise ValueError('Model/tokenizer integrity mismatch')
    return record


def tokenize(data,modeldir,max_length=128,batch_size=1024):
    from transformers import BertTokenizerFast
    data=Path(data); verify_assets(modeldir)
    tokenizer=BertTokenizerFast.from_pretrained(str(modeldir),local_files_only=True)
    info=json.loads((data/'dataset.json').read_text()); n=info['rows']
    marker=data/'tokens.json'
    if marker.exists():
        old=json.loads(marker.read_text())
        if old['max_length']!=max_length or old['dataset_fingerprint']!=info['fingerprint'] or old['model_assets_sha256']!=sha(Path(modeldir)/'assets.json'): raise ValueError('Tokenization protocol mismatch')
        for name,digest in old['files'].items():
            if sha(data/name)!=digest: raise ValueError('Token cache corrupted')
        return old
    arrays={name:np.lib.format.open_memmap(data/(name+'.npy'),mode='w+',dtype=dtype,shape=(n,max_length))
        for name,dtype in [('input_ids',np.int32),('attention_mask',np.uint8),('token_type_ids',np.uint8)]}
    titles=[]; contents=[]; processed=0
    def flush():
        nonlocal processed,titles,contents
        if not titles:return
        batch=tokenizer(titles,contents,padding='max_length',truncation='longest_first',max_length=max_length,return_tensors='np')
        for name,arr in arrays.items(): arr[processed:processed+len(titles)]=batch[name]
        processed+=len(titles); titles=[];contents=[]
    with open(data/'texts.jsonl',encoding='utf-8') as stream:
        for line in stream:
            row=json.loads(line); titles.append(row['title']); contents.append(row['content'])
            if len(titles)==batch_size:flush()
    flush()
    if processed!=n:raise ValueError('Text rows differ from labels')
    for arr in arrays.values():arr.flush()
    record=dict(rows=n,max_length=max_length,pair_order='title, content',truncation='longest_first',padding='right,max_length',
        model_assets_sha256=sha(Path(modeldir)/'assets.json'),dataset_fingerprint=info['fingerprint'],
        files={name+'.npy':sha(data/(name+'.npy')) for name in arrays},
        lengths=dict(min=int(arrays['attention_mask'].sum(1).min()),max=int(arrays['attention_mask'].sum(1).max()),mean=float(arrays['attention_mask'].sum(1).mean())))
    record['fingerprint']=fingerprint(record);write_json(marker,record);return record


def load_bert(modeldir,device,attention='eager'):
    from transformers import BertModel
    verify_assets(modeldir)
    model=BertModel.from_pretrained(str(modeldir),local_files_only=True,use_safetensors=True,add_pooling_layer=False,attn_implementation=attention)
    model.eval().to(device)
    for p in model.parameters():p.requires_grad_(False)
    return model


def batch_tokens(arrays,ids,device):
    return {k:torch.tensor(np.array(v[ids],copy=True),dtype=torch.long,device=device) for k,v in arrays.items()}


def array_digest(array,positions):
    return hashlib.sha256(np.ascontiguousarray(array[positions]).tobytes()).hexdigest()


def verify_block_journal(state,fp,order,ids,features,prefix=None,lookup=None):
    if state['fingerprint']!=fp:raise ValueError('Resume identity mismatch')
    completed=state['completed_rows']
    if not isinstance(completed,int) or not 0<=completed<=len(order):raise ValueError('Invalid completed row count')
    previous=0
    for block in state['blocks']:
        if block['begin']!=previous or not previous<block['end']<=completed:raise ValueError('Noncontiguous committed journal')
        positions=order[previous:block['end']]
        if array_digest(features,positions)!=block['features_sha256']:raise ValueError('Committed feature block integrity mismatch')
        if prefix is not None:
            locations=lookup[ids[positions]];locations=locations[locations>=0]
            if block['prefix_rows']!=len(locations) or array_digest(prefix,locations)!=block['prefix_sha256']:raise ValueError('Committed prefix block integrity mismatch')
        previous=block['end']
    if previous!=completed:raise ValueError('Journal does not cover all completed rows')
    return completed


def extract(data,modeldir,out,device='cpu',batch_size=32,threads=2,pilot_rows=0,prefix_cache=False,precision='fp32',attention='eager',length_buckets=False):
    if batch_size<=0 or threads<=0:raise ValueError('Batch size and thread count must be positive')
    torch.set_num_threads(threads); data=Path(data);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    meta=json.loads((data/'dataset.json').read_text());tok=json.loads((data/'tokens.json').read_text())
    if meta['fingerprint']!=fingerprint({k:v for k,v in meta.items() if k!='fingerprint'}):raise ValueError('Dataset manifest fingerprint mismatch')
    if tok['dataset_fingerprint']!=meta['fingerprint'] or tok['fingerprint']!=fingerprint({k:v for k,v in tok.items() if k!='fingerprint'}):raise ValueError('Token manifest dataset mismatch')
    for name,digest in meta['files'].items():
        if sha(data/name)!=digest:raise ValueError('Prepared source file integrity mismatch')
    for name,digest in tok['files'].items():
        if sha(data/name)!=digest:raise ValueError('Token checksum mismatch')
    arrays={k:np.load(data/(k+'.npy'),mmap_mode='r') for k in ('input_ids','attention_mask','token_type_ids')}
    with np.load(data/'split.npz') as z:
        split={k:z[k] for k in z.files};fit=split['fit'];selected=np.sort(np.concatenate([split[k] for k in PARTS[:-1]]))
    validate_split(np.load(data/'labels.npy',mmap_mode='r'),split)
    if pilot_rows:
        y=np.load(data/'labels.npy',mmap_mode='r');rng=np.random.default_rng(20261014);chosen=[]
        for c in range(14):
            members=fit[y[fit]==c];take=pilot_rows//14+(c<pilot_rows%14)
            if take>len(members):raise ValueError('Train-only pilot exceeds per-class feasible size')
            chosen.extend(rng.choice(members,take,replace=False))
        ids=np.asarray(chosen,dtype=np.int64)
    else:ids=np.arange(meta['rows'],dtype=np.int64)
    lengths=np.asarray(arrays['attention_mask'][ids].sum(1),dtype=np.int64);buckets=((lengths+31)//32)*32
    process_order=np.argsort(buckets,kind='stable') if length_buckets else np.arange(len(ids),dtype=np.int64)
    model=load_bert(modeldir,device,attention)
    identity=dict(dataset_fingerprint=meta['fingerprint'],tokens_fingerprint=tok['fingerprint'],assets_sha256=sha(Path(modeldir)/'assets.json'),
        pooling='attention-masked last-layer mean, includes CLS/SEP; excludes PAD',source_sha256=sha(__file__),rows=len(ids),pilot=bool(pilot_rows),
        prefix_cache=prefix_cache,prefix_layers=10,prefix_dtype='float16',precision=precision,attention=attention,batch_size=batch_size,threads=threads,device=device,torch_version=torch.__version__,length_buckets=length_buckets,bucket_lengths=[32,64,96,128],features_note='Computed after first10 hidden states roundtrip float16 to float32, matching partial encoder cache precision')
    fp=fingerprint(identity);marker=out/'progress.json';begin=0;times=[];blocks=[]
    reservation=out/'reservation.json'
    if reservation.exists():
        if json.loads(reservation.read_text())!={'fingerprint':fp,'identity':identity}:raise ValueError('Preserving preexisting cache with another identity')
    else:
        if marker.exists() or (out/'features.npy').exists() or (out/'prefix.npy').exists():raise ValueError('Cache arrays exist without an identity reservation; preserve and inspect them')
        write_json(reservation,dict(fingerprint=fp,identity=identity))
    path=out/'features.npy'
    if marker.exists():
        state=json.loads(marker.read_text())
        if state['fingerprint']!=fp:raise ValueError('Resume identity mismatch')
        begin=state['completed_rows'];times=state['batch_seconds'];blocks=state['blocks'];features=np.lib.format.open_memmap(path,mode='r+')
    else:features=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(len(ids),768))
    prefix=None; prefix_ids=ids if pilot_rows else selected
    if prefix_cache:
        ppath=out/'prefix.npy'
        prefix=np.lib.format.open_memmap(ppath,mode='r+' if marker.exists() else 'w+',dtype=np.float16,shape=(len(prefix_ids),tok['max_length'],768))
        lookup=np.full(meta['rows'],-1,dtype=np.int64);lookup[prefix_ids]=np.arange(len(prefix_ids))
    if features.shape!=(len(ids),768) or features.dtype!=np.float32:raise ValueError('Feature array shape/dtype mismatch')
    if prefix is not None and (prefix.shape!=(len(prefix_ids),tok['max_length'],768) or prefix.dtype!=np.float16):raise ValueError('Prefix array shape/dtype mismatch')
    if marker.exists():begin=verify_block_journal(state,fp,process_order,ids,features,prefix,lookup if prefix is not None else None)
    if (out/'features.json').exists():
        complete=json.loads((out/'features.json').read_text())
        if begin!=len(ids) or complete['fingerprint']!=fp or sha(path)!=complete['feature_sha256'] or sha(out/'row_ids.npy')!=complete['row_ids_sha256']:raise ValueError('Completed cache integrity mismatch')
        if prefix is not None and (sha(out/'prefix.npy')!=complete['prefix_sha256'] or sha(out/'prefix_row_ids.npy')!=complete['prefix_row_ids_sha256']):raise ValueError('Completed prefix integrity mismatch')
        return complete
    committed=begin;quantization=[]
    started=time.perf_counter()
    with torch.inference_mode(), torch.autocast(device_type=device,dtype=torch.float16,enabled=precision=='amp_float16'):
        i=begin
        while i<len(ids):
            j=min(i+batch_size,len(ids));positions=process_order[i:j]
            if length_buckets:
                current=buckets[positions[0]];same=positions[buckets[positions]==current];j=i+len(same);positions=same
            batch_ids=ids[positions];batch=batch_tokens(arrays,batch_ids,device)
            length=int(buckets[positions].max()) if length_buckets else tok['max_length']
            batch={k:v[:,:length] for k,v in batch.items()}
            t=time.perf_counter();hidden=model.embeddings(input_ids=batch['input_ids'],token_type_ids=batch['token_type_ids'])
            extended=model.get_extended_attention_mask(batch['attention_mask'],batch['input_ids'].shape)
            for layer in model.encoder.layer[:10]: hidden=layer(hidden,attention_mask=extended)[0]
            quantized=hidden.to(torch.float16).to(torch.float32)
            if pilot_rows:quantization.append(float((hidden-quantized).abs().max().cpu()))
            if prefix is not None:
                locations=lookup[batch_ids]; kept=locations>=0
                if kept.any():prefix[locations[kept],:length]=hidden[torch.as_tensor(kept,device=device)].cpu().numpy().astype(np.float16)
            hidden=quantized
            for layer in model.encoder.layer[10:]:hidden=layer(hidden,attention_mask=extended)[0]
            f=masked_mean(hidden,batch['attention_mask']).cpu().numpy()
            if f.shape!=(j-i,768) or not np.isfinite(f).all():raise ValueError('Nonfinite or wrong-shaped features')
            features[positions]=f;times.append(time.perf_counter()-t)
            if len(times)%20==0 or j==len(ids):
                features.flush()
                if prefix is not None:prefix.flush()
                positions_committed=process_order[committed:j];block=dict(begin=committed,end=j,features_sha256=array_digest(features,positions_committed))
                if prefix is not None:
                    locations_committed=lookup[ids[positions_committed]];locations_committed=locations_committed[locations_committed>=0]
                    block.update(prefix_rows=len(locations_committed),prefix_sha256=array_digest(prefix,locations_committed))
                blocks.append(block);committed=j
                write_json(marker,dict(fingerprint=fp,completed_rows=j,batch_seconds=times,device=device,threads=threads,blocks=blocks))
                print(json.dumps(dict(rows=j,total=len(ids),elapsed_seconds=time.perf_counter()-started,batches=len(times),device=device)),flush=True)
            i=j
    features.flush();np.save(out/'row_ids.npy',ids,allow_pickle=False)
    record=dict(**identity,fingerprint=fp,complete=True,
        inference_seconds=float(sum(times)),wall_seconds=time.perf_counter()-started,
        rows_per_second=len(ids)/sum(times),full630k_inference_eta_hours=630000/(len(ids)/sum(times))/3600,
        feature_sha256=sha(path),row_ids_sha256=sha(out/'row_ids.npy'),feature_shape=list(features.shape),versions={'torch':torch.__version__},
        hardware=platform.machine(),committed_block_count=len(blocks),journal_sha256=sha(marker),resume_verification='Every previously committed feature and prefix block rehashed before any resumed inference')
    if prefix is not None:
        prefix.flush();np.save(out/'prefix_row_ids.npy',prefix_ids,allow_pickle=False)
        record.update(prefix_shape=list(prefix.shape),prefix_bytes=(out/'prefix.npy').stat().st_size,prefix_sha256=sha(out/'prefix.npy'),prefix_row_ids_sha256=sha(out/'prefix_row_ids.npy'))
    if quantization:record['train_only_max_hidden_roundtrip_abs_error']=max(quantization)
    write_json(out/'features.json',record);return record


def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
    a=s.add_parser('acquire');a.add_argument('--out',type=Path,required=True)
    a=s.add_parser('prepare');a.add_argument('--source',type=Path,required=True);a.add_argument('--out',type=Path,required=True);a.add_argument('--rho',type=float,default=100.)
    a=s.add_parser('tokenize');a.add_argument('--data',type=Path,required=True);a.add_argument('--modeldir',type=Path,required=True);a.add_argument('--max-length',type=int,default=128)
    a=s.add_parser('extract');a.add_argument('--data',type=Path,required=True);a.add_argument('--modeldir',type=Path,required=True);a.add_argument('--out',type=Path,required=True);a.add_argument('--device',choices=['cpu','mps','cuda'],default='cpu');a.add_argument('--batch-size',type=int,default=32);a.add_argument('--threads',type=int,default=2);a.add_argument('--pilot-rows',type=int,default=0);a.add_argument('--prefix-cache',action='store_true');a.add_argument('--precision',choices=['fp32','amp_float16'],default='fp32');a.add_argument('--attention',choices=['eager','sdpa'],default='eager');a.add_argument('--length-buckets',action='store_true')
    args=p.parse_args();kw=vars(args);command=kw.pop('command')
    if command=='prepare': kw['source_path']=kw.pop('source')
    result=globals()[command](**kw);print(json.dumps(result,sort_keys=True,indent=2))
if __name__=='__main__':main()
