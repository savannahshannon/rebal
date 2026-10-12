#!/usr/bin/env python3
"""Independent weight-free replay of the frozen EQ study's original analysis.

Replays saved logits, source rows, selection, calibration, metrics, sampling
eligibility and the historical four-comparison family. It never imports the
scientific runner or loads model states. Run apply_followup_amendment.py next to
reproduce the prospective six-primary/two-secondary analysis. Original state
verification is a separately hash-bound receipt, not portable state coverage.
"""
from pathlib import Path
import argparse,ast,hashlib,importlib.metadata,itertools,json,math,platform,time
from datetime import datetime
import numpy as np
from scipy import stats
from sklearn.metrics import f1_score,recall_score

PARTS=('fit','fit_pool','tune','calibration','confirmation')
SUMMARY_METRICS=('accuracy','head_f1','macro_f1','macro_recall','mid_f1','tail_f1')
PRIMARY_NAMES=('natural_EQ_effect','balanced_EQ_effect','sampling_effect_EQoff','factorial_interaction')
GROUPS=('natural','balanced','bcl_dropout','ce','balanced_softmax')

def require(ok,message):
    if not ok:raise ValueError(message)

def read(path):return json.loads(Path(path).read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()
def fingerprint(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def digest(value):return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)
def utc(value):
    result=datetime.fromisoformat(value.replace('Z','+00:00'));require(result.tzinfo is not None,'Timezone-aware selection/confirmation ordering required');return result

class Comparator:
    def __init__(self):self.maximum=0.
    def check(self,actual,expected,message):
        a=np.asarray(actual,dtype=float);e=np.asarray(expected,dtype=float)
        require(a.shape==e.shape and np.isfinite(a).all() and np.isfinite(e).all(),message+' shape/finite guard')
        difference=float(np.max(np.abs(a-e))) if a.size else 0.;self.maximum=max(self.maximum,difference)
        require(difference<=1e-12,message)

def metrics(y,pred,rank,crosscheck=True):
    c=len(rank);require(y.shape==pred.shape and y.ndim==1 and y.dtype.kind in 'iu' and np.all((y>=0)&(y<c)) and np.all((pred>=0)&(pred<c)),'Invalid label/prediction vector')
    matrix=np.bincount(c*y.astype(np.int64)+pred.astype(np.int64),minlength=c*c).reshape(c,c)
    correct=np.diag(matrix).astype(float);support=matrix.sum(1);predicted=matrix.sum(0)
    f1=np.divide(2*correct,support+predicted,out=np.zeros(c),where=support+predicted>0)
    recall=np.divide(correct,support,out=np.zeros(c),where=support>0)
    if crosscheck:
        require(np.max(np.abs(f1-f1_score(y,pred,labels=np.arange(c),average=None,zero_division=0)))<=1e-12,'Independent sklearn F1 crosscheck differs')
        require(np.max(np.abs(recall-recall_score(y,pred,labels=np.arange(c),average=None,zero_division=0)))<=1e-12,'Independent sklearn recall crosscheck differs')
    result=dict(accuracy=float(correct.sum()/len(y)),macro_f1=float(f1.mean()),macro_recall=float(recall.mean()),worst_f1=float(f1.min()),per_class_f1=f1.tolist(),per_class_recall=recall.tolist())
    for name,mask in (('head',rank<c//3),('mid',(rank>=c//3)&(rank<2*c//3)),('tail',rank>=2*c//3)):
        require(mask.any(),'Rank region is empty');result[name+'_f1']=float(f1[mask].mean());result[name+'_recall']=float(recall[mask].mean())
    result['head_tail_f1_gap']=result['head_f1']-result['tail_f1'];return result

def macro(y,logits,c):
    pred=logits.argmax(1);matrix=np.bincount(c*y.astype(np.int64)+pred,minlength=c*c).reshape(c,c)
    denominator=matrix.sum(0)+matrix.sum(1)
    return float(np.divide(2*np.diag(matrix),denominator,out=np.zeros(c),where=denominator>0).mean())

def adjustment(calibration,y,confirmation,prior,taus):
    prior=np.asarray(prior,dtype=np.float32);require(prior.shape==(calibration.shape[1],) and np.isfinite(prior).all() and np.all(prior>0) and abs(float(prior.sum())-1)<1e-6,'Invalid effective training prior')
    if np.ptp(prior)==0:return confirmation.copy(),0.,[macro(y,calibration,len(prior))]*len(taus)
    log_prior=np.log(prior);curve=[macro(y,calibration-tau*log_prior,len(prior)) for tau in taus];tau=taus[int(np.argmax(curve))]
    return confirmation-tau*log_prior,tau,curve

def configuration_name(config):
    if config['objective'] in ('ce','balanced_softmax'):return config['objective']+'__lr'+format(config['peak_lr'],'.3g')
    if config['objective']=='bcl_dropout':return 'bcl_dropout__lambda'+format(config['coefficient'],'.2g')+'__temperature'+format(config['temperature'],'.2g')
    require(config['objective']=='class_balanced','Unknown adapter objective')
    return config['objective']+'__'+config['sampler']+('__off' if not config['coefficient'] else '__onset'+format(config['onset'],'.2g')+'__lambda'+format(config['coefficient'],'.2g'))

def arms(selection,reference):
    result={}
    for sampler in ('natural','balanced'):
        result[sampler+'_off']=dict(objective='class_balanced',sampler=sampler,onset=0.,coefficient=0.)
        result[sampler+'_on']=selection['selected'][sampler]
        result[sampler+'_reference']=dict(objective='class_balanced',sampler=sampler,**reference)
    for objective in ('ce','balanced_softmax'):
        result[objective]=selection['selected'][objective]
        result[objective+'_reference']=dict(objective=objective,sampler='natural',onset=0.,coefficient=0.,peak_lr=.05)
    result['bcl_dropout']=selection['selected']['bcl_dropout'];return result

def tau_grid(helper):
    tree=ast.parse(Path(helper).read_text())
    records=[ast.literal_eval(node.value) for node in tree.body if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='TAUS' for t in node.targets)]
    require(len(records)==1,'Hash-bound helper must declare one literal TAUS grid')
    result=list(records[0]);require(result==[0.,.25,.5,.75,1.,1.25,1.5,2.],'Unexpected frozen tau grid');return result

def clean_development(cache):
    """Reconstruct image-identity exclusions without importing the runner."""
    train=cache['train_pixel_sha256'];evaluation=set(cache['eval_pixel_sha256']);seen=set();keep=[];overlap=[];repeated=[]
    require(len(train)==cache['ntrain'] and len(cache['eval_pixel_sha256'])==cache['n_eval'] and all(digest(h) for h in train+cache['eval_pixel_sha256']),'Pixel identity manifest dimensions/schema differ')
    for i,h in enumerate(train):
        if h in evaluation:overlap.append(i)
        elif h in seen:repeated.append(i)
        else:keep.append(i);seen.add(h)
    return np.array(keep,np.int64),overlap,repeated

def regenerate_partition(labels,eligible,c,seed,holdouts,rho):
    order=np.random.default_rng(np.random.SeedSequence([seed,901])).permutation(c).astype(np.int64)
    rank=np.argsort(order).astype(np.int64);hold=sum(holdouts.values())
    nmax=min(np.bincount(labels[eligible],minlength=c))-hold
    capacity=np.maximum(1,np.floor(nmax*float(rho)**(-np.arange(c)/max(1,c-1))).astype(np.int64))
    output={name:[] for name in PARTS}
    for cls in range(c):
        rows=eligible[labels[eligible]==cls];rows=np.random.default_rng(np.random.SeedSequence([seed,cls,902])).permutation(rows)
        t=holdouts['tune'];cal=t+holdouts['calibration']
        output['tune'].extend(rows[:t]);output['calibration'].extend(rows[t:cal]);output['confirmation'].extend(rows[cal:hold])
        output['fit_pool'].extend(rows[hold:]);output['fit'].extend(rows[hold:hold+capacity[rank[cls]]])
    return {**{k:np.array(v,np.int64) for k,v in output.items()},'class_order':order,'class_rank':rank}

def declared_configurations(p):
    eq=[dict(onset=o,coefficient=l) for o in (.2,.5,.8) for l in (.01,.05)]
    bcl=[dict(coefficient=l,temperature=t) for l in (.05,.2,.5) for t in (.07,.1)]
    lrs=[.01,.025,.05,.075,.1,.2]
    require(p['grid']==eq and p['bcl_grid']==bcl and p['control_lr_grid']==lrs and p['reference']==dict(onset=.8,coefficient=.05),'Frozen EQ/BCL/control grid declarations differ')
    result=[]
    for sampler in ('natural','balanced'):
        result.append(dict(objective='class_balanced',sampler=sampler,onset=0.,coefficient=0.))
        result.extend(dict(objective='class_balanced',sampler=sampler,**g) for g in eq)
    result.extend(dict(objective=obj,sampler='natural',onset=0.,coefficient=0.,peak_lr=lr) for obj in ('ce','balanced_softmax') for lr in lrs)
    result.extend(dict(objective='bcl_dropout',sampler='natural',onset=0.,**g) for g in bcl)
    require(p['configurations']==result,'Exact declared 32-configuration scientific grid differs')
    return result

def source_inputs(args):
    run=args.run;protocol=read(run/'protocol.json');selection=read(run/'selection.json');aggregate=read(run/'aggregate.json');original=read(args.original_state_audit)
    require(fingerprint({k:v for k,v in protocol.items() if k!='fingerprint'})==protocol['fingerprint'],'Protocol fingerprint differs')
    require(sha(args.source)==protocol['source_sha256'],'Scientific source changed')
    for name,h in protocol['helper_sha256'].items():require(sha(args.helper_dir/name)==h,'Frozen numeric helper changed: '+name)
    require(protocol['fingerprint']==selection['protocol']==aggregate['protocol']==original['protocol'],'Protocol identities differ')
    cache=read(args.cache_manifest)
    require(sha(args.cache_manifest)==protocol['cache_manifest_sha256'] and cache['archive_sha256']==protocol['source_archive_sha256'] and cache['artifacts']['labels.npy']==sha(args.labels),'Frozen source-label/cache identity differs')
    require(fingerprint({k:v for k,v in selection.items() if k not in ('fingerprint','selection_created_utc')})==selection['fingerprint']==aggregate['selection']==original['selection'],'Selection identity differs')
    require(original['status']==('independently_verified_fixture' if args.fixture else 'independently_verified_complete') and original['adapter_checkpoint_states_verified']==320 and original['head_checkpoint_states_verified']==220 and original['aggregate_sha256']==sha(run/'aggregate.json'),'Separately audited original actual-state scope required')
    require(args.portable_without_weights and not list(run.rglob('*.pt')),'Explicit zero-weight portable mode requires all checkpoints omitted')
    with np.load(run/'partition.npz',allow_pickle=False) as z:
        require(set(z.files)==set(PARTS)|{'class_rank','class_order'},'Unexpected partition-array schema')
        split={k:z[k] for k in z.files}
    labels=np.load(args.labels,allow_pickle=False);require(labels.ndim==1 and labels.dtype.kind in 'iu','Source labels must be an integer vector')
    c=len(split['class_rank']);rank=split['class_rank'];require(c>=3 and np.array_equal(np.sort(rank),np.arange(c)) and np.array_equal(rank[split['class_order']],np.arange(c)),'Class rank/order inversion differs')
    for name,a in split.items():require(a.ndim==1 and a.dtype.kind in 'iu' and len(set(a.tolist()))==len(a),'Invalid or duplicate partition indices')
    for name in PARTS:require(np.all((split[name]>=0)&(split[name]<len(labels))),'Partition index outside source')
    require(np.all((labels>=0)&(labels<c)),'Source class range differs')
    sets=[set(split[k].tolist()) for k in ('fit_pool','tune','calibration','confirmation')]
    require(all(not a&b for a,b in itertools.combinations(sets,2)) and set(split['fit'].tolist())<=sets[0],'Partition overlap or fit outside pool')
    eligible,overlap,repeated=clean_development(cache)
    require(set.union(*sets)==set(eligible.tolist()),'Partition does not exactly cover deduplicated training rows or includes official-evaluation rows')
    identity=read(run/'partition_identity.json');require(identity['sha256']==sha(run/'partition.npz') and identity['protocol']==protocol['fingerprint'] and identity['fit_count']==len(split['fit']),'Partition receipt differs')
    counts=np.bincount(labels[split['fit']],minlength=c);require(np.all(counts>0) and identity['fit_counts']==counts.tolist(),'Fit class supports differ')
    for part,n in protocol['holdouts_per_class'].items():require(np.array_equal(np.bincount(labels[split[part]],minlength=c),np.full(c,n)),'Holdout support differs')
    if not args.fixture:
        require(c==200 and len(split['fit'])==15268 and counts.max()==355 and counts.min()==3 and len(labels)==110000 and cache['ntrain']==100000 and cache['n_eval']==10000 and len(eligible)==99973,'Production source/fit scope differs')
        require(p_hygiene_matches(protocol['hygiene'],labels,eligible,overlap,repeated,c),'Deduplication receipt differs')
        regenerated=regenerate_partition(labels,eligible,c,protocol['settings']['split_seed'],protocol['holdouts_per_class'],protocol['settings']['rho'])
        require(all(np.array_equal(split[k],regenerated[k]) for k in split),'Seeded partition/thinning/rank replay differs')
    scale=read(run/'scaler/scaler.json');require(scale['artifact_sha256']==sha(run/'scaler/scaler.npz') and scale['identity']==dict(partition_sha256=sha(run/'partition.npz'),protocol=protocol['fingerprint']) and scale['fit_count']==len(split['fit']) and scale['feature_dim']==192,'Fit-scaler receipt identity differs')
    with np.load(run/'scaler/scaler.npz',allow_pickle=False) as z:require(set(z.files)=={'mean','scale'} and z['mean'].shape==z['scale'].shape==(192,) and z['mean'].dtype==z['scale'].dtype==np.float64 and np.isfinite(z['mean']).all() and np.isfinite(z['scale']).all() and np.all(z['scale']>0),'Fit-scaler numeric schema differs')
    return protocol,selection,aggregate,original,labels,split,counts,tau_grid(args.helper_dir/'train_nonimage.py')

def p_hygiene_matches(record,labels,eligible,overlap,repeated,c):
    return (record['excluded_indices']==sorted(overlap+repeated) and record['train_eval_overlap_excluded']==overlap and record['duplicate_train_excluded']==repeated and
            record['retained_examples']==len(eligible) and record['clean_class_counts']==np.bincount(labels[eligible],minlength=c).tolist())

def eligibility(labels,fit,counts,seed,sampler,epochs,steps,batch):
    c=len(counts);members=[np.flatnonzero(labels[fit]==k) for k in range(c)];flat=np.concatenate(members);offsets=np.concatenate(([0],np.cumsum(counts[:-1])))
    result=[]
    for epoch in range(epochs):
        rng=np.random.default_rng(np.random.SeedSequence([seed,epoch,707]));observed=np.zeros(c,dtype=np.int64)
        for step in range(steps):
            u,w=rng.random((2,batch))
            if sampler=='natural':loc=(u*len(fit)).astype(np.int64)
            else:
                classes=(u*c).astype(np.int64);loc=flat[offsets[classes]+(w*counts[classes]).astype(np.int64)]
            observed+=np.bincount(labels[fit[loc]],minlength=c)>=2
        result.append(observed)
    return np.array(result)

def paired(d):
    d=np.asarray(d,float);require(d.shape==(10,) and np.isfinite(d).all(),'Ten finite paired differences required')
    mean=float(d.mean());sd=float(d.std(ddof=1));half=float(stats.t.ppf(.975,9)*sd/math.sqrt(10))
    signs=np.array(list(itertools.product((-1.,1.),repeat=10)))
    exact=float(np.mean(np.abs((signs*d).mean(1))>=abs(mean)-1e-12))
    t=float(stats.ttest_1samp(d,0).pvalue) if np.std(d)>0 else (1. if np.all(d==0) else 0.)
    return dict(differences=d.tolist(),mean=mean,sd=sd,ci95=[mean-half,mean+half],signflip_p=exact,t_p=t)

def holm(values):
    order=np.argsort(values,kind='stable');out=np.empty(len(values));running=0.
    for rank,i in enumerate(order):running=max(running,min(1.,(len(values)-rank)*values[i]));out[i]=running
    return out.tolist()

def verify(args):
    p,s,a,original,y,split,counts,taus=source_inputs(args);cmp=Comparator();c=len(counts);rank=split['class_rank'];settings=p['settings'];seeds=settings['seeds']
    require(seeds==list(range(10,20)) and len(declared_configurations(p))==32,'Exact 32-configuration/ten-seed grid required')
    epochs=settings['epochs'];steps=math.ceil(len(split['fit'])/settings['batch_size']);updates=epochs*steps
    if not args.fixture:require((epochs,updates,settings['head_steps'])==(50,750,2000),'Production optimizer budgets differ')
    expected={(seed,fingerprint(config)) for seed in seeds for config in p['configurations']}
    inventory=s['tuning_inventory'];observed=[(r['seed'],fingerprint(r['config'])) for r in inventory]
    require(len(observed)==320 and len(set(observed))==320 and set(observed)==expected and s['completed_tuning_cells']==320 and s['confirmation_access_before_freeze'] is False,'Complete unique tuning grid/freeze required')
    actual_folders={file.parent for file in args.run.glob('tuning/seed_*/*/tune.json')}
    require(actual_folders=={args.run/'tuning'/f"seed_{r['seed']}"/configuration_name(r['config']) for r in inventory},'Extra or missing tuning cells')
    initials={};training={};eligible_cache={};eligible_summaries={};tune_records=[]
    for item in inventory:
        seed,config=item['seed'],item['config'];name=configuration_name(config);folder=args.run/'tuning'/f'seed_{seed}'/name;r=read(folder/'tune.json');hist=read(folder/'history.json');info=r['training']
        require(sha(folder/'tune.json')==item['tune_json_sha256'] and r['identity']==dict(protocol=p['fingerprint'],seed=seed,configuration=name),'Tuning receipt/source identity differs')
        require(sha(folder/'history.json')==r['history_sha256'] and sha(folder/'tune_logits.npz')==r['predictions_sha256'] and digest(r['checkpoint_sha256']),'Tuning raw/history/checkpoint-reference hash differs')
        require(info['config']==config and info['completed_epochs']==epochs and info['optimizer_updates']==updates and len(hist)==epochs and all(digest(info[k]) for k in ('initial','final')),'Tuning completed metadata budget differs')
        initials.setdefault(seed,set()).add(info['initial']);training[(seed,fingerprint(config))]=info
        require(all(row['epoch']==i+1 and row['optimizer_updates']==(i+1)*steps and row['batches']==steps and math.isfinite(row['mean_loss']) and math.isfinite(row['seconds']) and row['seconds']>=0 for i,row in enumerate(hist)),'Tuning history budget/finite guard differs')
        cmp.check(sum(row['seconds'] for row in hist),info['seconds'],'Tuning recorded time sum differs')
        weights=(counts.sum()/(c*counts)).astype(np.float32) if config['objective']=='class_balanced' and config['sampler']=='natural' else np.ones(c,dtype=np.float32)
        if config['objective']=='class_balanced':require(np.array_equal(np.asarray(info['importance_weights'],np.float32),weights),'Objective importance weights differ')
        else:require(info['importance_weights'] is None,'Non-factorial objective must not carry importance weights')
        prior=(counts/counts.sum()).astype(np.float32) if config['objective']=='ce' else np.full(c,1/c,dtype=np.float32)
        require(np.array_equal(np.asarray(info['target_prior'],np.float32),prior),'Adapter effective target prior differs')
        key=(seed,config['sampler'])
        if key not in eligible_cache:eligible_cache[key]=eligibility(y,split['fit'],counts,seed,config['sampler'],epochs,steps,settings['batch_size'])
        require(np.array_equal(np.asarray([row['eligible_batches_per_class'] for row in hist]),eligible_cache[key]),'Sampling eligibility replay differs')
        require(all(row['EQ_active']==bool(config['objective']=='class_balanced' and config['coefficient'] and i>=int(config['onset']*epochs)) and row['BCL_active']==(config['objective']=='bcl_dropout') and row['BCL_contrastive_examples_per_update']==(min(64,settings['batch_size']) if config['objective']=='bcl_dropout' else 0) for i,row in enumerate(hist)),'EQ/BCL history eligibility/activation differs')
        if config['objective']=='class_balanced' and not config['coefficient']:eligible_summaries.setdefault(config['sampler'],[]).append(eligible_cache[key].sum(0)/updates)
        with np.load(folder/'tune_logits.npz',allow_pickle=False) as z:
            fields={'labels','logits'}|({'calibration_labels','calibration_logits'} if config['objective']=='ce' else set())
            require(set(z.files)==fields,'Unexpected tuning NPZ fields')
            require(z['labels'].dtype.kind in 'iu' and np.array_equal(z['labels'],y[split['tune']]),'Tuning source label order differs');logits=z['logits']
            require(logits.shape==(len(split['tune']),c) and logits.dtype==np.float32 and np.isfinite(logits).all(),'Tuning logits shape/dtype/finite guard failed')
            cmp.check(macro(z['labels'],logits,c),r['raw_macro_f1'],'Raw tuning macro-F1 differs')
            if config['objective']=='ce':
                calibration=z['calibration_logits'];require(z['calibration_labels'].dtype.kind in 'iu' and np.array_equal(z['calibration_labels'],y[split['calibration']]) and calibration.shape==(len(split['calibration']),c) and calibration.dtype==np.float32 and np.isfinite(calibration).all(),'CE tuning calibration source/schema differs')
                logits,tau,curve=adjustment(calibration,z['calibration_labels'],logits,prior,taus);adj=r['adjustment']
                require(adj['tau']==tau and [row['tau'] for row in adj['curve']]==taus,'CE tuning selected tau/grid differs');cmp.check(curve,[row['macro_f1'] for row in adj['curve']],'CE tuning calibration curve differs');cmp.check(adj['prior'],prior,'CE tuning adjustment prior differs')
            else:require(r['adjustment'] is None,'Non-CE tuning cannot select an extra offset')
            value=macro(z['labels'],logits,c);cmp.check(value,r['macro_f1'],'Tuning macro-F1 differs');cmp.check(value,item['macro_f1'],'Inventory tuning score differs')
        tune_records.append(dict(seed=seed,configuration=name,tune_json_sha256=sha(folder/'tune.json'),predictions_sha256=r['predictions_sha256']))
    require(all(len(v)==1 for v in initials.values()),'Unpaired adapter initial-state reference hashes')
    require(set(s['candidates'])==set(s['selected'])==set(GROUPS) and all(len(v)==6 for v in s['candidates'].values()),'Exact six-candidate groups required')
    for group,candidates in s['candidates'].items():
        require(len({fingerprint(v['config']) for v in candidates})==6,'Duplicate candidate configuration')
        declared_group=[config for config in p['configurations'] if (config['objective']=='class_balanced' and config['sampler']==group and config['coefficient']>0) if group in ('natural','balanced')] if group in ('natural','balanced') else [config for config in p['configurations'] if config['objective']==group]
        require({fingerprint(v['config']) for v in candidates}=={fingerprint(config) for config in declared_group},'Candidate group membership differs from declared scientific grid')
        for candidate in candidates:
            values=[r['macro_f1'] for r in inventory if r['config']==candidate['config']];require(len(values)==10,'Candidate pairing incomplete');cmp.check(np.mean(values),candidate['mean_macro_f1'],'Candidate mean differs')
        key=(lambda v:(-v['mean_macro_f1'],v['config']['coefficient'],v['config']['onset'])) if group in ('natural','balanced') else (lambda v:(-v['mean_macro_f1'],v['config']['coefficient'],v['config']['temperature'])) if group=='bcl_dropout' else (lambda v:(-v['mean_macro_f1'],v['config']['peak_lr']))
        require(sorted(candidates,key=key)[0]['config']==s['selected'][group],'Frozen global winner differs')
    declared=arms(s,p['reference']);expected_views={arm+'__'+head+suffix for arm in declared for head in ('native','natural','ros') for suffix in ('','_la')};cells=[];calcurves=0;readouts=0;pred_records=[]
    require({f.parent.name for f in args.run.glob('confirmation/seed_*/metrics.json')}=={f'seed_{seed}' for seed in seeds},'Extra or missing confirmation cells')
    for seed in seeds:
        folder=args.run/'confirmation'/f'seed_{seed}';r=read(folder/'metrics.json')
        require(r['seed']==seed and r['protocol']==p['fingerprint'] and r['selection']==s['fingerprint'] and utc(r['completed_utc'])>=utc(s['selection_created_utc']),'Confirmation cell identity/freeze ordering differs')
        require(set(r['metrics'])==set(r['prediction_views'])==expected_views,'Confirmation readout grid differs')
        expected_training=set(declared)|{arm+'__'+head for arm in declared for head in ('natural','ros')}
        require(set(r['training'])==expected_training,'Adapter/head training metadata grid differs')
        omitted={arm+'__head_'+head+'.pt' for arm in declared for head in ('natural','ros')}
        require(set(r['artifacts'])==omitted|{'predictions.npz'} and all(digest(h) for h in r['artifacts'].values()),'Raw/head artifact-reference schema differs')
        require(sha(folder/'predictions.npz')==r['artifacts']['predictions.npz'],'Confirmation raw NPZ hash differs')
        bases={arm+'__'+head for arm in declared for head in ('native','natural','ros')}
        for arm,config in declared.items():require(r['training'][arm]==training[(seed,fingerprint(config))],'Selected adapter metadata differs from frozen tuning record')
        for name,info in r['training'].items():
            if '__' not in name:continue
            require(info['optimizer_updates']==settings['head_steps'] and digest(info['initial_state_sha256']) and digest(info['final_state_sha256']),'Head metadata budget/state-reference differs')
            expected_loss_steps=[1]+list(range(100,settings['head_steps']+1,100))
            require([v['update'] for v in info['losses']]==expected_loss_steps and all(math.isfinite(v['loss']) for v in info['losses']),'Head loss history budget/finite guard differs')
            expected_prior=(counts/counts.sum()).astype(np.float32) if name.endswith('__natural') else np.full(c,1/c,dtype=np.float32)
            require(np.array_equal(np.asarray(info['training_prior'],np.float32),expected_prior),'Head effective target prior differs')
        require(len({info['initial_state_sha256'] for name,info in r['training'].items() if '__' in name})==1,'Unpaired natural/ROS head initial-state hash references')
        with np.load(folder/'predictions.npz',allow_pickle=False) as z:
            require(set(z.files)=={'confirmation_labels','calibration_labels','class_rank'}|{base+'__'+part+'_logits' for base in bases for part in ('calibration','confirmation')},'Unexpected confirmation NPZ fields')
            require(all(z[name].dtype.kind in 'iu' for name in ('confirmation_labels','calibration_labels','class_rank')) and np.array_equal(z['confirmation_labels'],y[split['confirmation']]) and np.array_equal(z['calibration_labels'],y[split['calibration']]) and np.array_equal(z['class_rank'],rank),'Confirmation/calibration source-row label or rank order differs')
            for base in sorted(bases):
                tl=z[base+'__confirmation_logits'];cl=z[base+'__calibration_logits']
                require(tl.shape==(len(split['confirmation']),c) and cl.shape==(len(split['calibration']),c) and tl.dtype==cl.dtype==np.float32 and np.isfinite(tl).all() and np.isfinite(cl).all(),'Prediction logits shape/dtype/finite guard failed')
                arm,head=base.rsplit('__',1);prior=r['training'][arm]['target_prior'] if head=='native' else r['training'][base]['training_prior']
                adjusted,tau,curve=adjustment(cl,z['calibration_labels'],tl,prior,taus)
                for name,logits,operation in ((base,tl,'stored'),(base+'_la',adjusted,'logit_adjustment')):
                    require(r['prediction_views'][name]==dict(base=base,operation=operation),'Prediction view operation/source differs')
                    actual=metrics(z['confirmation_labels'],logits.argmax(1),rank)
                    record=r['metrics'][name];require(set(record)==set(actual)|({'adjustment'} if operation=='logit_adjustment' else set()),'Readout metric schema differs')
                    for key,value in actual.items():cmp.check(value,record[key],'Readout metric differs: '+name+'/'+key)
                    if operation=='logit_adjustment':
                        adj=record['adjustment'];require(adj['tau']==tau and [v['tau'] for v in adj['curve']]==taus,'Confirmation calibration tau/grid differs');cmp.check(curve,[v['macro_f1'] for v in adj['curve']],'Confirmation calibration curve differs');cmp.check(np.asarray(adj['prior'],np.float32),np.asarray(prior,np.float32),'Confirmation adjustment prior differs');calcurves+=1
                    readouts+=1
        cells.append(r);pred_records.append(dict(seed=seed,metrics_sha256=sha(folder/'metrics.json'),predictions_sha256=r['artifacts']['predictions.npz']))
        print(json.dumps(dict(stage='confirmation_numeric_replayed',seed=seed,readouts=66)),flush=True)
    require((readouts,calcurves)==(660,330),'Complete readout/calibration scope required')
    require(a['status'] in ('complete','verified_complete') and a['seeds']==seeds and a['completed_tuning_cells']==320 and a['confirmation_cells']==10 and a['readouts']==660 and a['primary_family_size']==4 and a['selection_frozen_before_confirmation'] is True and a['official_evaluation_scored'] is False,'Complete historical aggregate scope differs')
    require(len(a['summary'])==66 and {r['method'] for r in a['summary']}==expected_views,'Historical summary grid differs')
    for row in a['summary']:
        require(row['n']==10 and set(row['metrics'])==set(SUMMARY_METRICS),'Summary count/endpoint schema differs')
        for endpoint,record in row['metrics'].items():
            values=np.array([r['metrics'][row['method']][endpoint] for r in cells]);mean=values.mean();sd=values.std(ddof=1);half=stats.t.ppf(.975,9)*sd/math.sqrt(10)
            cmp.check([mean,sd,mean-half,mean+half],[record['mean'],record['sd'],*record['ci95']],'Historical summary moments differ')
    require(len(a['paired_comparisons'])==4 and {r['name'] for r in a['paired_comparisons']}==set(PRIMARY_NAMES),'Historical four-contrast family differs')
    for row in a['paired_comparisons']:
        name=row['name'];values=[]
        for r in cells:
            m=r['metrics']
            if name=='factorial_interaction':value=(m['balanced_on__native']['macro_f1']-m['balanced_off__native']['macro_f1'])-(m['natural_on__native']['macro_f1']-m['natural_off__native']['macro_f1'])
            else:
                first,second={'natural_EQ_effect':('natural_on','natural_off'),'balanced_EQ_effect':('balanced_on','balanced_off'),'sampling_effect_EQoff':('balanced_off','natural_off')}[name]
                value=m[first+'__native']['macro_f1']-m[second+'__native']['macro_f1']
            values.append(value)
        for field,value in paired(values).items():cmp.check(value,row[field],'Historical paired statistic differs: '+field)
    for field in ('signflip_p','t_p'):
        actual=holm([r[field] for r in a['paired_comparisons']]);cmp.check(actual,[r[field+'_holm'] for r in a['paired_comparisons']],'Historical Holm family differs')
    return dict(status='independently_verified_fixture' if args.fixture else 'independently_verified_complete_portable_numeric',protocol=p['fingerprint'],selection=s['fingerprint'],
        scientific_source_sha256=p['source_sha256'],scientific_source_file_verified=True,auditor_sha256=sha(__file__),aggregate_sha256=sha(args.run/'aggregate.json'),selection_sha256=sha(args.run/'selection.json'),
        source_labels_sha256=sha(args.labels),source_cache_manifest_sha256=sha(args.cache_manifest),partition_sha256=sha(args.run/'partition.npz'),original_state_audit_sha256=sha(args.original_state_audit),
        tuning_cells_recomputed=320,confirmation_cells=10,readouts_recomputed=660,calibration_curves_recomputed=330,primary_family_size=4,summary_rows_recomputed=66,
        completed_adapter_updates=updates,completed_head_updates=settings['head_steps'],adapter_checkpoint_states_loaded=0,head_checkpoint_states_loaded=0,checkpoint_states_omitted=540,
        original_actual_adapter_states_verified=original['adapter_checkpoint_states_verified'],original_actual_head_states_verified=original['head_checkpoint_states_verified'],
        matched_adapter_and_head_initial_state_hash_references=True,actual_initial_state_tensors_reconstructed=False,sampling_eligibility_histories_replayed=320,
        clean_development_rows_verified=sum(len(split[k]) for k in ('fit_pool','tune','calibration','confirmation')),official_evaluation_rows_excluded=True,
        seeded_partition_thinning_rank_replayed=not args.fixture,
        eligibility={sampler:{group:float(np.array(values)[:,mask].mean()) for group,mask in (('head',rank<c//3),('mid',(rank>=c//3)&(rank<2*c//3)),('tail',rank>=2*c//3))} for sampler,values in eligible_summaries.items()},
        tune_capture_records=tune_records,confirmation_capture_records=pred_records,max_numeric_discrepancy=cmp.maximum,
        environment=dict(python=platform.python_version(),packages={name:importlib.metadata.version(name) for name in ('numpy','scipy','scikit-learn')}),
        omission_mode='explicit portable archive mode; all model and optimizer states omitted',
        limitations='Independent numeric replay of stored logits/labels, metadata budgets, priors, selection, eligibility and historical statistics; no checkpoint tensors, optimizer moments, encoder/head inference or scaler-estimation recomputation. Original actual-state verification is a separate receipt. Historical four contrasts require the prospective amendment helper for the current six-plus-two analysis. Source-row order is bound by labels and hash-bound partition indices; prediction NPZ has no explicit source-ID arrays.',
        verified_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run','labels','cache-manifest','source','helper-dir','original-state-audit','receipt'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--portable-without-weights',action='store_true');p.add_argument('--fixture',action='store_true');args=p.parse_args()
    result=verify(args);args.receipt.parent.mkdir(parents=True,exist_ok=True);args.receipt.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('status','tuning_cells_recomputed','readouts_recomputed','calibration_curves_recomputed','checkpoint_states_omitted','max_numeric_discrepancy')}))

if __name__=='__main__':main()
