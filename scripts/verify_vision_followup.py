"""Independent prediction/statistic audit; no model-state loading or inference."""
from pathlib import Path
import argparse,json,hashlib,itertools,math
import numpy as np
from scipy import stats
from sklearn.metrics import f1_score

NAMES=('deit_frozen','deit_last2','dino_frozen')
ARMS=tuple(name+'__'+obj for name in NAMES for obj in ('ce','balanced_softmax'))
SEEDS=tuple(range(10,20))
ENDPOINTS=('confirmation','confirmation_la','official_exploratory')
SCALARS=('accuracy','macro_f1','macro_recall','head_f1','mid_f1','tail_f1')
PAIRS=(('partial_vs_frozen_ce','deit_last2__ce','deit_frozen__ce'),
       ('dino_vs_deit_ce','dino_frozen__ce','deit_frozen__ce'))+tuple(
       ('balanced_softmax_vs_ce_'+name,name+'__balanced_softmax',name+'__ce') for name in NAMES)

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def require(value,message):
 if not value:raise ValueError(message)
def fp(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def verify_sources(protocol,source,helper):
 require(sha(source)==protocol['source_sha256'],'Frozen scientific source hash differs')
 require(sha(helper)==protocol['source_helper_sha256'],'Frozen scientific helper hash differs')

def verify_scope(protocol,aggregate,results,complete,selection_sha):
 require(aggregate['status']=='complete_and_prediction_verified' and aggregate['models']==60 and aggregate['readouts']==180,
         'Complete 60-model/180-readout aggregate required')
 require(aggregate['protocol']==results['protocol']==complete['protocol']==protocol['fingerprint'],'Complete protocol identities differ')
 require(aggregate['selection_sha256']==results['selection_sha256']==complete['selection_sha256']==selection_sha,'Complete selection identities differ')
 require(protocol['settings']['epochs']==complete['epochs']==15 and protocol['settings']['batch_size']==128 and
         protocol['settings']['seeds']==list(SEEDS),'Prespecified training settings differ')
 expected={(arm,seed) for arm in ARMS for seed in SEEDS}
 require(len(results['rows'])==60 and {(r['arm'],r['seed']) for r in results['rows']}==expected,'Exact final grid')
 require(complete['models']==60 and set(complete['training_receipt_sha256'])=={arm+'/'+str(seed) for arm,seed in expected},
         'Exact 60-receipt all-model completion gate required')
 summaries=aggregate['summary'];expected_summary={(arm,endpoint,metric) for arm in ARMS for endpoint in ENDPOINTS for metric in SCALARS}
 require(len(summaries)==108 and {(r['arm'],r['endpoint'],r['metric']) for r in summaries}==expected_summary and
         all(r['n']==10 and math.isfinite(r['mean']) and 0<=r['mean']<=1 and math.isfinite(r['sd']) and r['sd']>=0 for r in summaries),
         'Exact 108 finite ten-seed summaries required')
 contrasts=aggregate['primary_contrasts']
 require(len(contrasts)==5 and [r['name'] for r in contrasts]==[p[0] for p in PAIRS],
         'Exact five primary contrasts required')
 for row in contrasts:
  require(len(row['paired_differences'])==10 and np.isfinite(row['paired_differences']).all(),'Finite ten-seed contrast differences required')
  require(all(math.isfinite(row[k]) for k in ('mean','ci95_low','ci95_high','exact_p','t_p','exact_p_holm','t_p_holm')),
          'Finite contrast statistics required')

def paired_statistic(name,d):
 d=np.asarray(d,dtype=float);require(d.shape==(10,) and np.isfinite(d).all(),'Finite ten-seed paired statistic input required')
 mean=float(d.mean());half=float(stats.t.ppf(.975,9)*d.std(ddof=1)/math.sqrt(10))
 null=np.array([np.mean(d*np.array(signs)) for signs in itertools.product([-1,1],repeat=10)])
 ep=float(np.mean(np.abs(null)>=abs(mean)-1e-14))
 tp=float(stats.ttest_1samp(d,0).pvalue) if np.any(d!=d[0]) else (0. if d[0] else 1.)
 return dict(name=name,mean=mean,ci95_low=mean-half,ci95_high=mean+half,exact_p=ep,t_p=tp,paired_differences=d.tolist())
def metrics(z,y,order):
 require(z.shape==(len(y),200) and np.isfinite(z).all(),'Prediction shape/values')
 pred=z.argmax(1);cm=np.bincount(y*200+pred,minlength=40000).reshape(200,200);tp=np.diag(cm).astype(float);support=cm.sum(1);den=support+cm.sum(0)
 f=np.divide(2*tp,den,out=np.zeros(200),where=den>0);r=np.divide(tp,support,out=np.zeros(200),where=support>0)
 require(abs(float(f.mean())-f1_score(y,pred,labels=np.arange(200),average='macro',zero_division=0))<1e-12,'Independent sklearn F1 disagrees')
 return dict(accuracy=float(tp.sum()/len(y)),macro_f1=float(f.mean()),macro_recall=float(r.mean()),head_f1=float(f[order[:66]].mean()),mid_f1=float(f[order[66:133]].mean()),tail_f1=float(f[order[133:]].mean()),per_class_f1=f.tolist(),per_class_support=support.tolist())

def main(a):
 run=Path(a.run);cache=Path(a.cache);protocol=json.loads((run/'protocol.json').read_text());selection=json.loads((run/'selection.json').read_text());aggregate=json.loads((run/'aggregate.json').read_text());results=json.loads((run/'results.json').read_text())
 require(fp({k:v for k,v in protocol.items() if k!='fingerprint'})==protocol['fingerprint'],'Protocol fingerprint')
 verify_sources(protocol,a.source,a.source_helper)
 require(aggregate['results_sha256']==sha(run/'results.json'),'Aggregate result bytes differ')
 complete=json.loads((run/'final_training_complete.json').read_text())
 verify_scope(protocol,aggregate,results,complete,sha(run/'selection.json'))
 require(sha(a.partition)==protocol['partition_sha256'],'Partition hash')
 m=json.loads((cache/'manifest.json').read_text());labels=np.load(cache/'labels.npy',allow_pickle=False);require(sha(cache/'manifest.json')==protocol['deit_cache_manifest_sha256'] and sha(cache/'labels.npy')==m['artifacts']['labels.npy'],'Original label identities')
 split={k:v for k,v in np.load(a.partition,allow_pickle=False).items()};order=split['class_order'];require(np.array_equal(split['class_rank'][order],np.arange(200)),'Class-rank inverse')
 sets=[set(split[k]) for k in ['fit_pool','tune','calibration','confirmation']];require(all(not x&y for x,y in itertools.combinations(sets,2)),'Partition overlap');require(set(split['fit'])<=sets[0],'Fit rows outside fit pool')
 require(len(split['fit'])==15268 and len(split['tune'])==4000 and len(split['calibration'])==4000 and len(split['confirmation'])==20000,'Declared partitions')
 counts=np.bincount(labels[split['fit']],minlength=200);require(counts.max()==355 and counts.min()==3,'Thinning counts');require(np.array_equal(np.bincount(labels[split['confirmation']],minlength=200),np.full(200,100)),'Confirmation class counts')
 require(selection['protocol']==protocol['fingerprint'] and not selection['confirmation_scored'],'Selection identity');require(aggregate['selection_sha256']==sha(run/'selection.json') and results['selection_sha256']==sha(run/'selection.json'),'Selection hash')
 names=NAMES;expected={(name+'__'+obj,s) for name in names for obj in ['ce','balanced_softmax'] for s in range(10,20)}
 rows=results['rows'];require(len(rows)==60 and {(r['arm'],r['seed']) for r in rows}==expected,'Exact final grid');require(results['protocol']==protocol['fingerprint'] and aggregate['protocol']==protocol['fingerprint'],'Result protocol')
 extraction=json.loads((run/'extraction_complete.json').read_text());require(extraction['protocol']==protocol['fingerprint'],'Extraction protocol')
 # Portable archive excludes dense feature arrays; check them when available locally.
 cache_checks=0
 for name,key in [('deit_prefix.npy','prefix_sha256'),('dino_features.npy','dino_sha256'),('prefix_row_ids.npy','ids_sha256')]:
  if (run/name).exists():require(sha(run/name)==extraction[key],'Dense cache changed');cache_checks+=1
 require(set(selection['selection'])==set(ARMS),'Exact six-arm selection required')
 development=json.loads((run/'development_capture_complete.json').read_text());require(development['protocol']==protocol['fingerprint'] and development['models']==72 and not development['confirmation_accessed'] and development['selection_sha256']==sha(run/'selection.json'),'Development capture gate')
 development_records={};tuning_candidates={};development_curves=0
 for name in names:
  for obj in ['ce','balanced_softmax']:
   arm=name+'__'+obj;prior=counts/counts.sum() if obj=='ce' else np.ones(200)/200;candidates=[]
   configurations=[(901,lr,run/'tuning'/arm/str(lr)) for lr in ([1e-5,3e-5] if name=='deit_last2' else [.001,.003])]+[(seed,selection['selection'][arm]['lr'],run/'final'/arm/str(seed)) for seed in range(10,20)]
   for seed,lr,folder in configurations:
    record=json.loads((folder/'development_capture.json').read_text());identity=dict(protocol=protocol['fingerprint'],name=name,objective=obj,seed=seed,lr=lr);require(record['identity']==identity,'Development model identity')
    require(sha(folder/'selected_development_logits.npz')==record['predictions_sha256'],'Development prediction bytes changed')
    receipt=json.loads((folder/'training_complete.json').read_text());history=json.loads((folder/'history.json').read_text());require(receipt['identity']==identity and len(history)==15 and receipt['epochs']==15 and receipt['optimizer_updates']==1800,'Development training budget')
    require(all(h['epoch']==i+1 and h['optimizer_updates']==(i+1)*120 for i,h in enumerate(history)),'Development history updates');require(record['best_epoch']==receipt['best_epoch']==int(np.argmax([h['tuning_macro_f1'] for h in history]))+1,'Development selected epoch')
    with np.load(folder/'selected_development_logits.npz',allow_pickle=False) as captured:
     require(captured['tune_logits'].shape==(4000,200) and captured['calibration_logits'].shape==(4000,200),'Development prediction shapes')
     scores=[metrics(captured['calibration_logits']-t*np.log(prior),labels[split['calibration']],order)['macro_f1'] for t in np.linspace(0,2,41)];tau=float(np.linspace(0,2,41)[np.argmax(scores)])
     require(tau==record['selected_tau'] and len(record['calibration_curve'])==41,'Development offset grid');require(all(abs(v-r['macro_f1'])<1e-12 and abs(float(t)-r['tau'])<1e-12 for v,t,r in zip(scores,np.linspace(0,2,41),record['calibration_curve'])),'Development calibration replay')
     score=metrics(captured['tune_logits']-tau*np.log(prior),labels[split['tune']],order)['macro_f1'];require(abs(score-record['tuning_macro_f1'])<1e-12 and abs(score-receipt['best_tune'])<1e-12 and abs(score-history[record['best_epoch']-1]['tuning_macro_f1'])<1e-12,'Selected checkpoint development score')
    development_records[(arm,seed,lr)]=record;development_curves+=1
    if seed==901:candidates.append(dict(lr=lr,score=score,training_sha256=sha(folder/'training_complete.json')))
   winner=sorted(candidates,key=lambda r:(-r['score'],r['lr']))[0];require(winner==selection['selection'][arm],'Learning-rate winner disagrees with retained development predictions');tuning_candidates[arm]=candidates
 require(len(development_records)==72 and len(development['records'])==72,'Development record count');require({fp(r) for r in development['records']}=={fp(r) for r in development_records.values()},'Development gate records changed')
 discrepancies=[];verified={};scalar_count=0;cal_count=0;history_count=0;checkpoint_hashes=0
 for record in rows:
  arm=record['arm'];seed=record['seed'];folder=run/'final'/arm/str(seed);name,obj=arm.rsplit('__',1)
  receipt=json.loads((folder/'training_complete.json').read_text());require(receipt['identity']==dict(protocol=protocol['fingerprint'],name=name,objective=obj,seed=seed,lr=selection['selection'][arm]['lr']),'Training identity');require(sha(folder/'training_complete.json')==complete['training_receipt_sha256'][arm+'/'+str(seed)],'Training gate receipt')
  history=json.loads((folder/'history.json').read_text());require(len(history)==15 and receipt['epochs']==15 and receipt['optimizer_updates']==1800,'Training budget')
  require(all(h['epoch']==i+1 and h['optimizer_updates']==(i+1)*120 for i,h in enumerate(history)),'History update budget');require(receipt['best_epoch']==int(np.argmax([h['tuning_macro_f1'] for h in history]))+1,'Tuning-only epoch choice');history_count+=len(history)
  if (folder/'checkpoint.pt').exists():require(sha(folder/'checkpoint.pt')==receipt['checkpoint_sha256'],'Checkpoint hash');checkpoint_hashes+=1
  predictions=folder/'predictions.npz';require(sha(predictions)==record['predictions_sha256'],'Raw predictions changed')
  prior=counts/counts.sum() if obj=='ce' else np.ones(200)/200;require(np.allclose(prior,record['prior'],rtol=0,atol=1e-15),'Training prior')
  with np.load(predictions,allow_pickle=False) as saved:
   cal=saved['calibration_logits'];conf=saved['confirmation_logits'];official=saved['official_logits'];require(cal.shape==(4000,200) and conf.shape==(20000,200) and official.shape==(10000,200),'Raw prediction dimensions')
   with np.load(folder/'selected_development_logits.npz',allow_pickle=False) as development_saved:require(np.array_equal(cal,development_saved['calibration_logits']),'Confirmation evaluator changed selected checkpoint calibration logits')
   scores=[metrics(cal-t*np.log(prior),labels[split['calibration']],order)['macro_f1'] for t in np.linspace(0,2,41)];tau=float(np.linspace(0,2,41)[np.argmax(scores)]);require(tau==record['selected_tau'],'Calibration-only tau');cal_count+=1
   require(all(abs(v-r['macro_f1'])<1e-12 and abs(float(t)-r['tau'])<1e-12 for v,t,r in zip(scores,np.linspace(0,2,41),record['calibration_curve'])),'Calibration curve')
   views={'confirmation':metrics(conf,labels[split['confirmation']],order),'confirmation_la':metrics(conf-tau*np.log(prior),labels[split['confirmation']],order),'official_exploratory':metrics(official-tau*np.log(prior),labels[100000:],order)}
   verified[(arm,seed)]=views
   for endpoint,metric in views.items():
    for key,value in metric.items():
     actual=np.asarray(value);claimed=np.asarray(record[endpoint][key]);delta=float(np.max(np.abs(actual-claimed)));require(delta<1e-12,'Metric mismatch');discrepancies.append(delta);scalar_count+=actual.size
 for r in aggregate['summary']:
  values=np.array([verified[(r['arm'],s)][r['endpoint']][r['metric']] for s in range(10,20)]);require(abs(values.mean()-r['mean'])<1e-12 and abs(values.std(ddof=1)-r['sd'])<1e-12 and r['n']==10,'Summary mismatch')
 pairs=PAIRS;calculated=[]
 for name,left,right in pairs:
  d=np.array([verified[(left,s)]['confirmation_la']['macro_f1']-verified[(right,s)]['confirmation_la']['macro_f1'] for s in range(10,20)])
  calculated.append(paired_statistic(name,d))
 for field in ['exact_p','t_p']:
  indices=sorted(range(5),key=lambda i:calculated[i][field]);prev=0
  for rank,i in enumerate(indices):prev=max(prev,min(1,(5-rank)*calculated[i][field]));calculated[i][field+'_holm']=prev
 for actual,claimed in zip(calculated,aggregate['primary_contrasts']):
  require(actual['name']==claimed['name'],'Contrast order')
  for k in actual:
   if k!='name':require(np.max(np.abs(np.asarray(actual[k])-np.asarray(claimed[k])))<1e-12,'Contrast statistic mismatch')
 report=dict(status='verified_complete',source='Independent NumPy-bincount metrics crosschecked with sklearn F1; SciPy statistics; no model objects loaded',models_verified=60,readouts_verified=180,summary_rows_verified=108,calibration_curves_verified=cal_count,development_models_verified=72,development_calibration_curves_verified=development_curves,tuning_winners_verified=6,history_epochs_verified=history_count,scalar_metrics_verified=scalar_count,primary_contrasts_verified=5,max_metric_discrepancy=max(discrepancies),dense_cache_hashes_checked=cache_checks,checkpoint_hashes_checked=checkpoint_hashes,checkpoint_states_loaded=0,scientific_source_file_verified=True,scientific_source_sha256=sha(a.source),scientific_helper_file_verified=True,scientific_helper_sha256=sha(a.source_helper),protocol=protocol['fingerprint'],aggregate_sha256=sha(run/'aggregate.json'),results_sha256=sha(run/'results.json'),labels_sha256=sha(cache/'labels.npy'),partition_sha256=sha(a.partition),verifier_sha256=sha(__file__),limitations='Does not rerun frozen encoders or replay every historical epoch prediction. Budgets replayed from histories/receipts; optimizer parameter shapes and actual per-parameter step counters are not loaded or independently verified here. Portable archives omit dense caches, weights and model/optimizer states; stored identities are retained. Logit row binding relies on frozen inference code and preserved partition order; archives do not store explicit row IDs.')
 Path(a.out).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--cache',required=True);p.add_argument('--partition',required=True);p.add_argument('--source',required=True);p.add_argument('--source-helper',required=True);p.add_argument('--out',required=True);main(p.parse_args())
