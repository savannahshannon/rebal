"""Synthetic small-array 320-cell EQ archive and adversarial portable replay."""
from pathlib import Path
import argparse,copy,importlib.util,json,math,tempfile,unittest
import numpy as np
from scipy import stats
from sklearn.metrics import f1_score,recall_score

REPO=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('eq_numeric_verifier',REPO/'scripts/verify_eq_followup.py')
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)

def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data,indent=2)+'\n')
def npz(path,**arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**arrays)

def metric(y,logits,rank):
    pred=logits.argmax(1);c=len(rank);f=f1_score(y,pred,labels=np.arange(c),average=None,zero_division=0);r=recall_score(y,pred,labels=np.arange(c),average=None,zero_division=0)
    result=dict(accuracy=float(np.mean(y==pred)),macro_f1=float(f.mean()),macro_recall=float(r.mean()),worst_f1=float(f.min()),per_class_f1=f.tolist(),per_class_recall=r.tolist())
    for name,mask in (('head',rank<c//3),('mid',(rank>=c//3)&(rank<2*c//3)),('tail',rank>=2*c//3)):
        result[name+'_f1']=float(f[mask].mean());result[name+'_recall']=float(r[mask].mean())
    result['head_tail_f1_gap']=result['head_f1']-result['tail_f1'];return result

def adjust(cal,y,logits,prior):
    prior=np.asarray(prior,np.float32);taus=[0.,.25,.5,.75,1.,1.25,1.5,2.]
    if np.ptp(prior)==0:curve=[float(f1_score(y,cal.argmax(1),average='macro',zero_division=0))]*len(taus);tau=0.;out=logits.copy()
    else:
        lp=np.log(prior);curve=[float(f1_score(y,(cal-t*lp).argmax(1),average='macro',zero_division=0)) for t in taus];tau=taus[int(np.argmax(curve))];out=logits-tau*lp
    return out,dict(tau=tau,curve=[dict(tau=t,macro_f1=m) for t,m in zip(taus,curve)],prior=prior.tolist())

def configurations():
    result=[]
    for sampler in ('natural','balanced'):
        result.append(dict(objective='class_balanced',sampler=sampler,onset=0.,coefficient=0.))
        result += [dict(objective='class_balanced',sampler=sampler,onset=o,coefficient=l) for o in (.2,.5,.8) for l in (.01,.05)]
    for objective in ('ce','balanced_softmax'):
        result += [dict(objective=objective,sampler='natural',onset=0.,coefficient=0.,peak_lr=lr) for lr in (.01,.025,.05,.075,.1,.2)]
    result += [dict(objective='bcl_dropout',sampler='natural',onset=0.,coefficient=l,temperature=t) for l in (.05,.2,.5) for t in (.07,.1)]
    return result

def fixture(folder):
    folder=Path(folder);run=folder/'run';data=folder/'data';run.mkdir();data.mkdir();seeds=list(range(10,20));rank=np.arange(3,dtype=np.int64)
    labels=np.array([0]*6+[1]*3+[2]+[0,1,2]*4,dtype=np.int64);np.save(data/'labels.npy',labels)
    split=dict(fit=np.arange(10),fit_pool=np.arange(10),tune=np.arange(10,13),calibration=np.arange(13,16),confirmation=np.arange(16,22),class_rank=rank,class_order=rank)
    npz(run/'partition.npz',**split);counts=np.bincount(labels[split['fit']],minlength=3)
    cache=dict(archive_sha256='a'*64,artifacts={'labels.npy':v.sha(data/'labels.npy')},ntrain=len(labels),n_eval=0,
        train_pixel_sha256=[v.fingerprint(dict(row=i)) for i in range(len(labels))],eval_pixel_sha256=[]);write(data/'cache_manifest.json',cache)
    p=dict(source_sha256=v.sha(REPO/'scripts/followup_eq.py'),helper_sha256={n:v.sha(REPO/'scripts'/n) for n in ('benchmark_vit_rebal.py','train_nonimage.py','train_tiny_fullscale.py')},
        cache_manifest_sha256=v.sha(data/'cache_manifest.json'),source_archive_sha256=cache['archive_sha256'],configurations=configurations(),reference=dict(onset=.8,coefficient=.05),
        settings=dict(seeds=seeds,epochs=2,batch_size=4,head_steps=100),holdouts_per_class=dict(tune=1,calibration=1,confirmation=2),
        grid=[dict(onset=o,coefficient=l) for o in (.2,.5,.8) for l in (.01,.05)],
        bcl_grid=[dict(coefficient=l,temperature=t) for l in (.05,.2,.5) for t in (.07,.1)],control_lr_grid=[.01,.025,.05,.075,.1,.2])
    p['fingerprint']=v.fingerprint(p);write(run/'protocol.json',p)
    write(run/'partition_identity.json',dict(sha256=v.sha(run/'partition.npz'),protocol=p['fingerprint'],fit_count=10,fit_counts=counts.tolist()))
    npz(run/'scaler/scaler.npz',mean=np.zeros(192,np.float64),scale=np.ones(192,np.float64));write(run/'scaler/scaler.json',dict(artifact_sha256=v.sha(run/'scaler/scaler.npz'),identity=dict(protocol=p['fingerprint'],partition_sha256=v.sha(run/'partition.npz')),fit_count=10,feature_dim=192))
    base=np.array([[2,0,-1],[1.5,1,-1],[2,0,.5]],np.float32);inventory=[];infos={};eligible={}
    for seed in seeds:
        for config in p['configurations']:
            name=v.configuration_name(config);path=run/'tuning'/f'seed_{seed}'/name;path.mkdir(parents=True)
            prior=(counts/counts.sum()).astype(np.float32) if config['objective']=='ce' else np.full(3,1/3,np.float32)
            key=(seed,config['sampler'])
            if key not in eligible:eligible[key]=v.eligibility(labels,split['fit'],counts,seed,config['sampler'],2,3,4)
            history=[dict(epoch=i+1,optimizer_updates=(i+1)*3,batches=3,mean_loss=.5,seconds=.1,eligible_batches_per_class=eligible[key][i].tolist(),
                EQ_active=bool(config['objective']=='class_balanced' and config['coefficient'] and i>=int(config['onset']*2)),BCL_active=config['objective']=='bcl_dropout',BCL_contrastive_examples_per_update=4 if config['objective']=='bcl_dropout' else 0) for i in range(2)]
            write(path/'history.json',history)
            info=dict(config=config,completed_epochs=2,optimizer_updates=6,seconds=.2,initial=v.fingerprint(dict(seed=seed)),final=v.fingerprint(config),
                importance_weights=((counts.sum()/(3*counts)).astype(np.float32).tolist() if config['sampler']=='natural' else [1.,1.,1.]) if config['objective']=='class_balanced' else None,target_prior=prior.tolist())
            arrays=dict(logits=base,labels=labels[split['tune']]);selected=base;adj=None
            if config['objective']=='ce':arrays.update(calibration_logits=base,calibration_labels=labels[split['calibration']]);selected,adj=adjust(base,arrays['calibration_labels'],base,prior)
            npz(path/'tune_logits.npz',**arrays)
            r=dict(identity=dict(protocol=p['fingerprint'],seed=seed,configuration=name),training=info,macro_f1=metric(arrays['labels'],selected,rank)['macro_f1'],
                raw_macro_f1=metric(arrays['labels'],base,rank)['macro_f1'],adjustment=adj,predictions_sha256=v.sha(path/'tune_logits.npz'),history_sha256=v.sha(path/'history.json'),checkpoint_sha256='c'*64)
            write(path/'tune.json',r);inventory.append(dict(seed=seed,config=config,macro_f1=r['macro_f1'],tune_json_sha256=v.sha(path/'tune.json')));infos[(seed,v.fingerprint(config))]=info
    candidates={};selected={}
    for group in v.GROUPS:
        configs=[config for config in p['configurations'] if (config['objective']=='class_balanced' and config['sampler']==group and config['coefficient']>0) if group in ('natural','balanced')] if group in ('natural','balanced') else [config for config in p['configurations'] if config['objective']==group]
        candidates[group]=[dict(config=config,mean_macro_f1=float(np.mean([r['macro_f1'] for r in inventory if r['config']==config]))) for config in configs]
        key=(lambda x:(-x['mean_macro_f1'],x['config']['coefficient'],x['config']['onset'])) if group in ('natural','balanced') else (lambda x:(-x['mean_macro_f1'],x['config']['coefficient'],x['config']['temperature'])) if group=='bcl_dropout' else (lambda x:(-x['mean_macro_f1'],x['config']['peak_lr']))
        selected[group]=sorted(candidates[group],key=key)[0]['config']
    s=dict(protocol=p['fingerprint'],completed_tuning_cells=320,tuning_inventory=inventory,candidates=candidates,selected=selected,confirmation_access_before_freeze=False,selection_created_utc='2026-10-12T00:00:00+00:00')
    s['fingerprint']=v.fingerprint({k:x for k,x in s.items() if k!='selection_created_utc'});write(run/'selection.json',s);declared=v.arms(s,p['reference']);cells=[]
    for seed in seeds:
        path=run/'confirmation'/f'seed_{seed}';path.mkdir(parents=True);record={};views={};training={};prediction=dict(confirmation_labels=labels[split['confirmation']],calibration_labels=labels[split['calibration']],class_rank=rank)
        for arm,config in declared.items():
            training[arm]=infos[(seed,v.fingerprint(config))]
            for head in ('native','natural','ros'):
                name=arm+'__'+head;prior=training[arm]['target_prior'] if head=='native' else (counts/counts.sum()).astype(np.float32).tolist() if head=='natural' else [float(np.float32(1/3))]*3
                if head!='native':training[name]=dict(optimizer_updates=100,initial_state_sha256='d'*64,final_state_sha256='e'*64,losses=[dict(update=1,loss=.5),dict(update=100,loss=.3)],training_prior=prior)
                cl=base;tl=base[labels[split['confirmation']]];prediction[name+'__calibration_logits']=cl;prediction[name+'__confirmation_logits']=tl
                record[name]=metric(labels[split['confirmation']],tl,rank);views[name]=dict(base=name,operation='stored')
                adjusted,adj=adjust(cl,labels[split['calibration']],tl,prior);record[name+'_la']=metric(labels[split['confirmation']],adjusted,rank);record[name+'_la']['adjustment']=adj;views[name+'_la']=dict(base=name,operation='logit_adjustment')
        npz(path/'predictions.npz',**prediction);artifacts={'predictions.npz':v.sha(path/'predictions.npz')}
        artifacts.update({arm+'__head_'+head+'.pt':'f'*64 for arm in declared for head in ('natural','ros')})
        r=dict(protocol=p['fingerprint'],selection=s['fingerprint'],seed=seed,completed_utc='2026-10-12T00:01:00+00:00',metrics=record,prediction_views=views,training=training,artifacts=artifacts);write(path/'metrics.json',r);cells.append(r)
    summary=[]
    for name in sorted(cells[0]['metrics']):
        summary.append(dict(method=name,n=10,metrics={key:dict(mean=cells[0]['metrics'][name][key],sd=0.,ci95=[cells[0]['metrics'][name][key]]*2) for key in v.SUMMARY_METRICS}))
    contrasts=[dict(name=name,**v.paired([0.]*10),signflip_p_holm=1.,t_p_holm=1.) for name in v.PRIMARY_NAMES]
    aggregate=dict(status='verified_complete',protocol=p['fingerprint'],selection=s['fingerprint'],seeds=seeds,completed_tuning_cells=320,confirmation_cells=10,readouts=660,primary_family_size=4,
        selection_frozen_before_confirmation=True,official_evaluation_scored=False,summary=summary,paired_comparisons=contrasts);write(run/'aggregate.json',aggregate)
    audit=folder/'original_state_audit.json';write(audit,dict(status='independently_verified_fixture',protocol=p['fingerprint'],selection=s['fingerprint'],adapter_checkpoint_states_verified=320,head_checkpoint_states_verified=220,aggregate_sha256=v.sha(run/'aggregate.json')))
    return argparse.Namespace(run=run,labels=data/'labels.npy',cache_manifest=data/'cache_manifest.json',source=REPO/'scripts/followup_eq.py',helper_dir=REPO/'scripts',original_state_audit=audit,portable_without_weights=True,fixture=True)

def resign_selection(args):
    s=v.read(args.run/'selection.json');s['fingerprint']=v.fingerprint({k:x for k,x in s.items() if k not in ('fingerprint','selection_created_utc')});write(args.run/'selection.json',s)
    a=v.read(args.run/'aggregate.json');a['selection']=s['fingerprint'];write(args.run/'aggregate.json',a)
    audit=v.read(args.original_state_audit);audit.update(selection=s['fingerprint'],aggregate_sha256=v.sha(args.run/'aggregate.json'));write(args.original_state_audit,audit)

class PortableEQTests(unittest.TestCase):
    def setUp(self):self.temp=tempfile.TemporaryDirectory();self.args=fixture(Path(self.temp.name))
    def tearDown(self):self.temp.cleanup()
    def test_complete_fixture_replays_all_320_660_330_with_zero_states(self):
        r=v.verify(self.args);self.assertEqual((r['tuning_cells_recomputed'],r['readouts_recomputed'],r['calibration_curves_recomputed']),(320,660,330));self.assertEqual(r['adapter_checkpoint_states_loaded']+r['head_checkpoint_states_loaded'],0);self.assertEqual(r['checkpoint_states_omitted'],540);self.assertLessEqual(r['max_numeric_discrepancy'],1e-12)
    def test_missing_raw_tuning_capture_rejected(self):
        next(self.args.run.glob('tuning/*/*/tune_logits.npz')).unlink()
        with self.assertRaises(FileNotFoundError):v.verify(self.args)
    def test_candidate_group_cannot_be_swapped_after_resigning_selection(self):
        s=v.read(self.args.run/'selection.json');s['candidates']['ce'],s['candidates']['balanced_softmax']=s['candidates']['balanced_softmax'],s['candidates']['ce']
        write(self.args.run/'selection.json',s);resign_selection(self.args)
        with self.assertRaisesRegex(ValueError,'Candidate group membership'):v.verify(self.args)
    def test_clean_training_union_excludes_official_and_duplicate_rows(self):
        cache=dict(ntrain=5,n_eval=1,train_pixel_sha256=['a'*64,'b'*64,'a'*64,'c'*64,'d'*64],eval_pixel_sha256=['b'*64])
        rows,overlap,repeated=v.clean_development(cache);np.testing.assert_array_equal(rows,[0,3,4]);self.assertEqual(overlap,[1]);self.assertEqual(repeated,[2])
        path=self.args.run/'partition.npz'
        with np.load(path,allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
        arrays['fit_pool']=arrays['fit_pool'][:-1];arrays['fit']=arrays['fit'][:-1];npz(path,**arrays)
        with self.assertRaisesRegex(ValueError,'exactly cover deduplicated'):v.verify(self.args)
    def test_unpaired_head_initial_references_are_rejected(self):
        folder=self.args.run/'confirmation/seed_10';r=v.read(folder/'metrics.json');r['training']['ce__natural']['initial_state_sha256']='a'*64;write(folder/'metrics.json',r)
        with self.assertRaisesRegex(ValueError,'Unpaired natural/ROS head'):v.verify(self.args)
    def test_changed_raw_confirmation_bytes_rejected(self):
        path=self.args.run/'confirmation/seed_10/predictions.npz';path.write_bytes(path.read_bytes()+b'changed')
        with self.assertRaisesRegex(ValueError,'raw NPZ hash'):v.verify(self.args)
    def test_hidden_text_array_is_rejected_even_after_rehash(self):
        folder=self.args.run/'confirmation/seed_10';path=folder/'predictions.npz'
        with np.load(path,allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
        npz(path,**arrays,raw_text=np.array(['forbidden content']))
        r=v.read(folder/'metrics.json');r['artifacts']['predictions.npz']=v.sha(path);write(folder/'metrics.json',r)
        with self.assertRaisesRegex(ValueError,'Unexpected confirmation NPZ fields'):v.verify(self.args)
    def test_calibration_choice_and_metric_tampering_are_rejected(self):
        folder=self.args.run/'confirmation/seed_10';r=v.read(folder/'metrics.json');original=copy.deepcopy(r)
        for change in ('tau','metric'):
            r=copy.deepcopy(original)
            if change=='tau':r['metrics']['ce__native_la']['adjustment']['tau']=99.
            else:r['metrics']['ce__native']['macro_f1']+=.01
            write(folder/'metrics.json',r)
            with self.assertRaises(ValueError):v.verify(self.args)
    def test_wrong_global_winner_rejected_after_receipts_resigned(self):
        s=v.read(self.args.run/'selection.json');s['selected']['ce']=s['candidates']['ce'][-1]['config'];write(self.args.run/'selection.json',s);resign_selection(self.args)
        with self.assertRaisesRegex(ValueError,'Frozen global winner'):v.verify(self.args)
    def test_checkpoint_presence_or_false_original_scope_rejected(self):
        path=self.args.run/'checkpoint.pt';path.write_bytes(b'Forbidden state placeholder')
        with self.assertRaisesRegex(ValueError,'checkpoints omitted'):v.verify(self.args)
        path.unlink();audit=v.read(self.args.original_state_audit);audit['adapter_checkpoint_states_verified']=0;write(self.args.original_state_audit,audit)
        with self.assertRaisesRegex(ValueError,'original actual-state scope'):v.verify(self.args)
    def test_independent_confusion_math_and_holm_known_values(self):
        y=np.array([0,0,1,1,2,2]);pred=np.array([0,1,1,1,0,2]);rank=np.array([2,0,1]);m=v.metrics(y,pred,rank)
        self.assertAlmostEqual(m['macro_f1'],(0.5+0.8+2/3)/3);self.assertAlmostEqual(m['head_f1'],.8);self.assertAlmostEqual(m['tail_f1'],.5)
        self.assertEqual(v.holm([.01,.02,.2]),[.03,.04,.2]);self.assertEqual(v.paired([0.]*10)['signflip_p'],1.)

if __name__=='__main__':unittest.main()
