#!/usr/bin/env python3
"""Assemble the combined report payload only from four complete audited bundles.

This formats verified evidence; it does not train, select models, infer held-out
predictions, plot figures or author a PDF. All paths in the emitted payload are
relative to its directory. The report builder independently replays the numeric
bindings before the payload is atomically published.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import importlib.util
import json
import math
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent.parent
SEEDS = list(range(10, 20))
STUDIES = ('equalization', 'vision', 'text', 'tabular')
TEXT_METHODS = ('bert_ce', 'bert_bsoftmax', 'bert_natural_eq', 'bert_balanced_ce',
                'bert_balanced_eq', 'mean_ce', 'mean_bsoftmax', 'last0_ce',
                'last0_bsoftmax', 'last1_ce', 'last1_bsoftmax', 'last2_ce', 'last2_bsoftmax')
EQ_ROWS = (
    ('ce__native_la', 'Selected CE + LA'),
    ('balanced_softmax__native_la', 'Selected Balanced Softmax'),
    ('bcl_dropout__native_la', 'Selected cached-feature BCL adaptation'),
    ('natural_off__native', 'Natural sampler, EQ off; balanced objective'),
    ('natural_reference__native', 'Natural sampler, fixed EQ .8/.05'),
    ('balanced_off__native', 'Class-uniform sampler, EQ off'),
    ('balanced_reference__native', 'Class-uniform sampler, fixed EQ .8/.05'),
)
VISION_ROWS = tuple((encoder + '__' + loss, 'confirmation_la', label + ' / ' + objective)
    for encoder, label in (('deit_frozen', 'Frozen DeiT'), ('deit_last2', 'Last-two-block DeiT'),
                           ('dino_frozen', 'Frozen DINOv2'))
    for loss, objective in (('ce', 'CE + LA'), ('balanced_softmax', 'Balanced Softmax')))
TEXT_LABELS = dict(bert_ce='Frozen BERT mean / CE + LA',
    bert_bsoftmax='Frozen BERT mean / Balanced Softmax',
    bert_natural_eq='Frozen BERT mean / natural EQ + LA',
    bert_balanced_ce='Frozen BERT mean / class-uniform CE',
    bert_balanced_eq='Frozen BERT mean / class-uniform EQ',
    mean_ce='Random mean-pooled text / CE + LA', mean_bsoftmax='Random mean-pooled text / Balanced Softmax',
    last0_ce='Matched frozen BERT CLS / CE + LA', last0_bsoftmax='Matched frozen BERT CLS / Balanced Softmax',
    last1_ce='Last-one-block BERT CLS / CE + LA', last1_bsoftmax='Last-one-block BERT CLS / Balanced Softmax',
    last2_ce='Last-two-block BERT CLS / CE + LA', last2_bsoftmax='Last-two-block BERT CLS / Balanced Softmax')
TABULAR_ROWS = (('mlp_ce_la', 'Natural CE MLP + LA'), ('mlp_cb_la', 'Effective-number MLP + LA'),
                ('ft_transformer_la', 'FT-Transformer + LA'), ('xgboost_la', 'XGBoost + LA'))
LABELS = {
    'fixed_natural_EQ_effect': 'Fixed EQ on minus off, natural sampler',
    'fixed_balanced_EQ_effect': 'Fixed EQ on minus off, class-uniform sampler',
    'sampling_effect_EQoff': 'Class-uniform minus natural sampler, EQ off',
    'fixed_factorial_interaction': 'Fixed EQ effect: class-uniform minus natural',
    'selected_balanced_softmax_vs_selected_ce_la': 'Selected Balanced Softmax minus selected CE + LA',
    'selected_bcl_vs_selected_balanced_softmax': 'Selected BCL adaptation minus selected Balanced Softmax',
    'partial_vs_frozen_ce': 'Last-two-block DeiT CE + LA minus frozen DeiT CE + LA',
    'dino_vs_deit_ce': 'Frozen DINO CE + LA minus frozen DeiT CE + LA',
    'balanced_softmax_vs_ce_deit_frozen': 'Balanced Softmax minus CE + LA, frozen DeiT',
    'balanced_softmax_vs_ce_deit_last2': 'Balanced Softmax minus CE + LA, last-two-block DeiT',
    'balanced_softmax_vs_ce_dino': 'Balanced Softmax minus CE + LA, frozen DINO',
    'bert_ce_la__minus__mean_ce_la': 'Frozen BERT mean CE + LA minus random mean CE + LA',
    'bert_bsoftmax__minus__bert_ce_la': 'Frozen BERT mean Balanced Softmax minus CE + LA',
    'bert_balanced_eq__minus__bert_balanced_ce': 'EQ minus CE, class-uniform frozen BERT mean',
    'bert_natural_eq_la__minus__bert_ce_la': 'Natural EQ + LA minus CE + LA, frozen BERT mean',
    'last1_ce_la__minus__last0_ce_la': 'Last-one-block CE + LA minus matched frozen CLS CE + LA',
    'last2_ce_la__minus__last0_ce_la': 'Last-two-block CE + LA minus matched frozen CLS CE + LA',
    'last1_bsoftmax__minus__last0_bsoftmax': 'Last-one-block Balanced Softmax minus matched frozen CLS',
    'last2_bsoftmax__minus__last0_bsoftmax': 'Last-two-block Balanced Softmax minus matched frozen CLS',
    'ft_transformer_la__minus__mlp_ce_la': 'FT-Transformer + LA minus natural CE MLP + LA',
    'xgboost_la__minus__mlp_ce_la': 'XGBoost + LA minus natural CE MLP + LA',
    'mlp_cb_la__minus__mlp_ce_la': 'Effective-number MLP + LA minus natural CE MLP + LA',
}
LAYOUT = {
 'equalization': dict(marker='manifest.json', status='EQcomplete_independently_verified', roles={
    'aggregate':'aggregate.json', 'protocol':'protocol.json', 'independent_verification':'independent_verification.json',
    'amendment_verification':'amendment_verification.json', 'completion':'completion.json',
    'selection':'public_amendment_inputs/selection.json', 'analysis_amendment':'analysis_amendment.json',
    'historical_aggregate':'historical_aggregate.json', 'portable_manifest':'portable/manifest.json',
    'portable_verification':'portable/audit/portable_numeric_verification.json',
    'portable_amendment_reproduction':'portable/audit/amendment_reproduction.json',
    'scientific_source':'portable/scripts/followup_eq.py'}),
 'vision': dict(marker='manifest.json', status='complete_weight_free_bundle_independently_replayed', roles={
    'aggregate':'run/aggregate.json', 'protocol':'run/protocol.json',
    'independent_verification':'audit/independent_verification.json',
    'checkpoint_verification':'audit/actual_checkpoint_verification.json',
    'portable_verification':'audit/portable_numeric_verification.json', 'selection':'run/selection.json',
    'scientific_source':'scripts/benchmark_vision_followup.py', 'partition':'data/partition.npz',
    'documentation':'docs/vision_followup.md'}),
 'text': dict(marker='manifest.json', status='portable_numeric_archive_verified', roles={
    'aggregate':'experiment/aggregate.json', 'protocol':'experiment/protocol.json',
    'independent_verification':'audits/original_independent_verification.json',
    'portable_verification':'audits/portable_numeric_verification.json', 'metadata':'data/dataset.json',
    'partition':'experiment/partition.npz', 'source_partition':'data/split.npz',
    'scientific_source':'code/scripts/benchmark_text_followup.py', 'documentation':'code/docs/text_followup.md'}),
 'tabular': dict(marker='bundle_manifest.json', status='weight_free_bundle_independently_replayed', roles={
    'aggregate':'aggregate.json', 'protocol':'protocol.json',
    'independent_verification':'production_model_provenance/verification.json',
    'checkpoint_verification':'production_model_provenance/checkpoint_audit_complete.json',
    'portable_verification':'portable_verification.json', 'statistical_verification':'statistical_replay.json',
    'independent_prediction_verification':'independent_prediction_verification.json',
    'independent_prediction_helper':'scripts/verify_tabular_followup_predictions.py',
    'selection':'frozen_selection.json', 'metadata':'data/metadata.json', 'partition':'data/split.npz',
    'source_labels':'data/labels.npy', 'scaler':'data/scaler.npz',
    'scientific_source':'scripts/benchmark_tabular_followup.py', 'documentation':'docs/tabular_followup.md'}),
}

def require(condition, message):
    if not condition: raise ValueError(message)

def read(path):
    return json.loads(Path(path).read_text())

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024**2), b''): h.update(block)
    return h.hexdigest()

def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def safe_relative(value):
    p = Path(value)
    require(not p.is_absolute() and '..' not in p.parts and str(p) not in ('', '.'), 'Unsafe bundle evidence path')
    return p

def manifest_files(manifest):
    files = manifest['files']
    if isinstance(files, list):
        require(len({r['path'] for r in files}) == len(files), 'Duplicate manifest path')
        return {r['path']:r for r in files}
    require(isinstance(files, dict), 'Invalid bundle manifest files')
    return files

def complete_markers(root):
    """Read completion metadata first; incomplete studies never reach outcomes."""
    missing = [name for name in STUDIES if not (root/name/LAYOUT[name]['marker']).is_file()]
    require(not missing, 'All four complete bundles required; missing: '+', '.join(missing))
    markers = {name:read(root/name/LAYOUT[name]['marker']) for name in STUDIES}
    for name, m in markers.items():
        require(m['status']==LAYOUT[name]['status'], name+' bundle completion marker did not pass')
    eq=read(root/'equalization/completion.json')
    require(eq['status']=='complete_independently_verified_reported_and_visually_checked' and
            (eq['tuning_adapters'],eq['confirmation_seeds'],eq['retrained_heads'],eq['confirmation_readouts'],eq['primary_comparisons'])==(320,10,220,660,6),
            'EQ completion counts differ')
    eq_portable=read(root/'equalization/portable/manifest.json')
    require(eq_portable['status']=='complete_local_weight_free_EQ_numeric_archive_independently_replayed' and
            eq_portable['identity']['counts']==dict(tuning_cells=320,confirmation_cells=10,confirmation_readouts=660,calibration_curves=330,primary_comparisons=6,secondary_comparisons=2) and
            (eq_portable['identity']['portable_checkpoint_states_loaded'],eq_portable['identity']['checkpoint_states_omitted'])==(0,540),
            'EQ full local raw numeric archive is incomplete')
    v=markers['vision']
    require((v['models'],v['confirmation_readouts'],v['official_exploratory_readouts'],v['development_captures'],
             v['production_checkpoint_states_verified'],v['portable_checkpoint_states_loaded'])==(60,120,60,72,72,0),
            'Vision complete numeric/state scopes differ')
    t=markers['text']
    require(t['counts']==dict(models=130,tuning_fits=26,confirmation_readouts=190,native_readouts=100) and
            (t['checkpoint_states_verified_in_original_audit'],t['checkpoint_states_verified_in_portable_replay'],t['checkpoint_states_omitted'])==(156,0,156),
            'Text complete numeric/state scopes differ')
    tab=markers['tabular']
    require((tab['prediction_files'],tab['tuning_prediction_files'],tab['readouts'],tab['model_weights_present'])==(40,8,80,0),
            'Tabular complete scope differs')
    return markers

def load_bundles(root, markers):
    bundles={}
    for name in STUDIES:
        folder=root/name;catalog=manifest_files(markers[name])
        for relative,record in catalog.items():
            file=folder/safe_relative(relative)
            require(file.is_file() and file.resolve().is_relative_to(folder.resolve()), 'Bundle file missing or outside bundle')
            require(file.stat().st_size==record['bytes'] and sha(file)==record['sha256'], name+' bundle manifest file changed: '+relative)
        paths=dict(LAYOUT[name]['roles']);paths['package_manifest']=LAYOUT[name]['marker']
        require(set(paths.values())-set(catalog)<= {LAYOUT[name]['marker']}, name+' required role missing from completed manifest')
        records=[dict(role=role,path=str(Path(name)/relative),sha256=sha(folder/relative)) for role,relative in paths.items()]
        objects={role:read(folder/relative) for role,relative in paths.items() if Path(relative).suffix=='.json'}
        protocol=objects['protocol'];require(fingerprint({k:v for k,v in protocol.items() if k!='fingerprint'})==protocol['fingerprint'], name+' frozen protocol differs')
        if 'scientific_source' in paths:require(sha(folder/paths['scientific_source'])==protocol['source_sha256'], name+' frozen scientific source differs')
        bundles[name]=dict(records=records,objects=objects,paths=paths)
    return bundles

def summary_row(aggregate, method, label):
    matches=[r for r in aggregate['summary'] if r.get('method')==method]
    require(len(matches)==1, 'Main result method missing or duplicated: '+method)
    return dict(label=label,source_selector=dict(method=method),
                metrics={k:{f:matches[0]['metrics'][k][f] for f in ('mean','sd')} for k in ('macro_f1','tail_f1','accuracy')})

def vision_row(aggregate, arm, endpoint, label):
    rows=[r for r in aggregate['summary'] if r['arm']==arm and r['endpoint']==endpoint]
    require(len(rows)==6 and len({r['metric'] for r in rows})==6,'Vision result view missing or duplicated')
    return dict(label=label,source_selector=dict(arm=arm,endpoint=endpoint),
                metrics={r['metric']:dict(mean=r['mean'],sd=r['sd']) for r in rows if r['metric'] in ('macro_f1','tail_f1','accuracy')})

def displayed_rows(bundles):
    eq=bundles['equalization']['objects']['aggregate'];v=bundles['vision']['objects']['aggregate']
    t=bundles['text']['objects']['aggregate'];tab=bundles['tabular']['objects']['aggregate']
    return dict(equalization=[summary_row(eq,*r) for r in EQ_ROWS],
                vision=[vision_row(v,*r) for r in VISION_ROWS],
                text=[summary_row(t,m+('_la' if (m.endswith('_ce') and 'balanced' not in m) or m=='bert_natural_eq' else ''),TEXT_LABELS[m]) for m in TEXT_METHODS],
                tabular=[summary_row(tab,*r) for r in TABULAR_ROWS])

def result_statement(title, rows):
    best=max(rows,key=lambda r:r['metrics']['macro_f1']['mean']);m=best['metrics']['macro_f1']
    return (f"{title}: among the listed confirmation readouts, {best['label']} has the largest mean macro-F1, "
            f"{100*m['mean']:.2f}% (sample SD {100*m['sd']:.2f} percentage points). This ranking is descriptive; "
            'the declared paired comparisons supply the inferential evidence.')

def contrast_statement(name, analysis):
    rows=[r for r in analysis['comparisons'] if r['study']==name]
    detectable=[r for r in rows if r['signflip_p_study_holm']<=.05]
    positive=sum(r['mean']>0 for r in detectable);negative=sum(r['mean']<0 for r in detectable)
    return (f"The declared family contains {len(rows)} paired comparisons. {len(detectable)} reach exact sign-flip "
            f"Holm p <= 0.05 within this study ({positive} positive and {negative} negative contrasts in the stated "
            'subtraction direction). The table retains every comparison, including undetected effects.')

def secondary_eq(aggregate):
    names=('selected_natural_EQ_pipeline_effect','selected_balanced_EQ_pipeline_effect')
    source={r['name']:r for r in aggregate['secondary_comparisons']}
    require(set(source)==set(names),'Selected EQ secondary comparisons differ')
    rows=[]
    for name,label in zip(names,('Selected natural EQ on minus off','Selected class-uniform EQ on minus off')):
        r=source[name];rows.append([label,f"{100*r['mean']:+.4f}",
            f"[{100*r['ci95'][0]:+.4f}, {100*r['ci95'][1]:+.4f}]",f"{r['signflip_p']:.6f}"])
    pipeline=dict(source_kind='selected_EQ_secondary',source_selectors=[dict(comparison=n) for n in names],
        headers=['Adaptive EQ pipeline contrast','Effect (pp)','Paired 95% CI (pp)','Exact p'],rows=rows,widths=[.44,.15,.27,.14],
        caption='Secondary adaptive-pipeline effects use separately selected EQ coefficients/onsets. Exact p-values are unadjusted; these two effects are outside the six-member primary family. They do not define the fixed-treatment factorial interaction.')
    summaries={r['method']:r for r in aggregate['summary']};arms=('ce','balanced_softmax','bcl_dropout');rows=[]
    for arm,label in zip(arms,('Selected CE + LA','Selected Balanced Softmax','Selected BCL adaptation')):
        values=[]
        for head in ('native','natural','ros'):
            m=summaries[arm+'__'+head+'_la']['metrics']['macro_f1'];values.append(f"{100*m['mean']:.2f} +/- {100*m['sd']:.2f}")
        rows.append([label,*values])
    heads=dict(source_kind='EQ_head_diagnostic',source_selectors=[dict(arm=a) for a in arms],
        headers=['Selected representation','Native head','Natural refit head','ROS refit head'],rows=rows,widths=[.34,.22,.22,.22],
        caption='Secondary confirmation macro-F1 (%), mean +/- sample SD. Natural and ROS readouts fit new linear classifiers for 2,000 updates on each fixed representation; ROS resamples actual fit vectors. Head changes can alter descriptive model ranking. These readouts are not additional independent encoder fits.')
    return [pipeline,heads]

def study_payloads(bundles, analysis):
    rows=displayed_rows(bundles);eq=bundles['equalization']['objects'];v=bundles['vision']['objects']
    t=bundles['text']['objects'];tab=bundles['tabular']['objects'];ep=eq['protocol'];vp=v['protocol'];tp=t['protocol'];bp=tab['protocol']
    tm=t['metadata'];bm=tab['metadata'];ts=tm['split_counts'];bs=bm['split_counts']
    require(tm['fingerprint']==tp['dataset_fingerprint'] and fingerprint({k:x for k,x in tm.items() if k!='fingerprint'})==tm['fingerprint'],'Text source metadata differs')
    require(bm['fingerprint']==bp['data_fingerprint'] and fingerprint({k:x for k,x in bm.items() if k!='fingerprint'})==bm['fingerprint'],'Tabular source metadata differs')
    require(tp['methods']==list(TEXT_METHODS) and tp['seeds']==bp['seeds']==vp['settings']['seeds']==ep['settings']['seeds']==SEEDS,'Frozen study method/seed grid differs')
    require(ep['source_archive_sha256']==vp['official_archive_sha256'] and ep['cache_manifest_sha256']==vp['deit_cache_manifest_sha256'],'Image studies source/cache identities differ')
    classes=len(ep['hygiene']['clean_class_counts']);holdouts={k:n*classes for k,n in ep['holdouts_per_class'].items()}
    require((holdouts['tune'],holdouts['calibration'],holdouts['confirmation'])==(vp['tune_rows'],vp['cal_rows'],vp['confirmation_rows']), 'Image holdout sizes differ')
    clean=ep['hygiene']['retained_examples'];fit_pool=clean-sum(holdouts.values());fit=vp['fit_rows']
    eq_updates=ep['settings']['epochs']*math.ceil(fit/ep['settings']['batch_size'])
    require((len(ep['configurations'])*len(SEEDS),eq_updates,ep['settings']['head_steps'])==(320,750,2000),'EQ declared budgets differ')
    require(tp['fit_examples']==ts['fit'] and (tp['adapter_epochs']*tp['adapter_steps_per_epoch'],tp['partial_epochs']*tp['partial_steps_per_epoch'])==(5200,318),'Text declared fit/update budgets differ')
    require(vp['settings']['epochs']*math.ceil(fit/vp['settings']['batch_size'])==1800,'Vision declared update budget differs')
    tab_updates={family:bp['neural_epochs']*math.ceil(bs['fit']/bp[family+'_batch_size']) for family in ('mlp','ft')}
    require(sum(bs.values())==bm['source_rows'] and bm['scaler_fit_part']=='fit','Tabular source partition/scaler scope differs')
    require((ep['settings']['epochs'],ep['settings']['batch_size'])==(50,1024) and
            (vp['settings']['epochs'],vp['settings']['batch_size'])==(15,128) and
            (tp['adapter_epochs'],tp['partial_epochs'],tp['batch_size'],tp['partial_batch_size'],tp['adapter_steps_per_epoch'],tp['partial_steps_per_epoch'])==(50,3,1024,1024,104,106) and
            tp['eq_weight']==0.05 and
            (bp['neural_epochs'],bp['mlp_batch_size'],bp['ft_batch_size'])==(50,2048,1024),
            'The narrative applies only to the frozen production budget settings')
    require(set(bp['grids'])=={'mlp_ce','mlp_cb','ft_transformer','xgboost'} and
            all(len(grid)==2 for grid in bp['grids'].values()) and
            all(c['n_estimators']==600 for c in bp['grids']['xgboost']), 'Tabular fixed search/budget scope differs')
    require(v['portable_verification']['checkpoint_states_loaded']==0 and
            v['portable_verification']['checkpoint_hashes_checked']==0,'Vision portable audit must omit actual checkpoint states')
    require(eq['portable_verification']['status']=='independently_verified_complete_portable_numeric' and
            (eq['portable_verification']['tuning_cells_recomputed'],eq['portable_verification']['readouts_recomputed'],eq['portable_verification']['calibration_curves_recomputed'])==(320,660,330) and
            (eq['portable_verification']['adapter_checkpoint_states_loaded'],eq['portable_verification']['head_checkpoint_states_loaded'],eq['portable_verification']['checkpoint_states_omitted'])==(0,0,540),
            'EQ portable replay must cover all stored raw scores and omit actual states')
    require((t['independent_verification']['actual_checkpoint_states_verified'],t['independent_verification']['checkpoint_states_omitted'])==(156,0) and
            (t['portable_verification']['actual_checkpoint_states_verified'],t['portable_verification']['checkpoint_states_omitted'])==(0,156),
            'Text original/portable actual-state audit scopes differ')
    require(tab['checkpoint_verification']['tuning_trials_verified']==8 and
            tab['checkpoint_verification']['final_fits_verified']==40 and
            tab['statistical_verification']['summaries_verified']==64 and
            tab['statistical_verification']['paired_contrasts_verified']==3 and
            tab['portable_verification']['model_weight_artifacts_present_verified']==0 and
            tab['portable_verification']['neural_checkpoint_states_present_verified']==0,
            'Tabular original state/statistical versus portable scope differs')
    ip=tab['independent_prediction_verification']
    require(ip['status']=='independent_final_prediction_replay_passed' and
            (ip['completed_fits'],ip['readouts_recomputed'],ip['calibration_curves_recomputed'],ip['tau_candidates_per_curve'],ip['calibration_candidate_scores_recomputed'],
             ip['scalar_metrics_recomputed'],ip['per_class_metric_values_recomputed'],ip['confusion_entries_recomputed'],ip['class_supports_verified'])==(40,80,40,8,320,640,1680,3920,560) and
            ip['model_weights_loaded']==0 and ip['scientific_runner_imported'] is False and ip['prediction_model_inference'] is False,
            'Independent tabular final prediction replay is incomplete')
    titles=dict(equalization='Tiny ImageNet | EQ and stronger loss controls',vision='Tiny ImageNet | encoder and partial-tuning controls',
                text='DBpedia-14 | contextual and selective-tuning controls',tabular='Covertype | tabular model and objective controls')
    base={name:dict(status='independently_verified_complete',title=titles[name],source_sha256=bundles[name]['objects']['protocol']['source_sha256'],
                   evidence_files=bundles[name]['records'],summary=rows[name],result_notes=[result_statement(titles[name],rows[name])],
                   contrast_notes=[contrast_statement(name,analysis)]) for name in STUDIES}
    base['equalization'].update(source_rows=vp['source_rows'],fit_rows=fit,confirmation_rows=holdouts['confirmation'],
        tuning_rows=holdouts['tune'],calibration_rows=holdouts['calibration'],canonical_development_rows=clean,fit_pool_rows=fit_pool,
        tuning_fits=320,final_fits=220,final_fit_scope='220 new linear heads; selected/reference adapters reused from tuning',
        confirmation_seeds=10,confirmation_readouts=660,budget_label=f"{ep['settings']['epochs']} epochs/{eq_updates} adapter updates; {ep['settings']['head_steps']:,} head updates",
        method_notes=[
            'The full cached image source contains 100,000 development and 10,000 official-evaluation images. EQ uses the canonicalized development partitions and does not score the official 10,000-image endpoint in this round. The 110,000-source count therefore includes an unused evaluation source; canonical development is reported separately.',
            'This GAN-free study holds the frozen DeiT feature cache fixed and compares EQ, sampling and stronger loss controls. Natural and class-uniform factorial sampling target the same expected class-balanced cross-entropy from epoch one. Natural samples receive importance weight N/(C*n_c); class-uniform samples receive weight one.',
            'The same expected loss does not imply identical stochastic optimizer updates: sample variance, momentum and clipping can differ. The fixed primary EQ treatment begins at fraction 0.8 of training and uses coefficient 0.05 in both samplers, with matched learning rate 0.05.',
            'CE + logit adjustment, Balanced Softmax, and a BCL adaptation use six prespecified candidates each. BCL implements class-averaged contrastive denominators and all-class classifier-weight prototypes with two adapter-dropout views of cached vectors. It is a frozen-feature adaptation, not a reproduction of BCL image augmentation or its complete original architecture.'
        ],partition_table=[['Canonical development',clean,'Deduplicate and exclude official-evaluation content matches before splitting'],
            ['Fit pool before thinning',fit_pool,'Separate pool; nominal 100:1 thinning follows holdout construction'],['Imbalanced fit',fit,'Gradient updates; fit-only feature standardization'],
            ['Tuning',holdouts['tune'],'Global candidate selection across ten seeds'],['Calibration',holdouts['calibration'],'Decision-offset selection; also used in CE tuning policy'],
            ['Confirmation',holdouts['confirmation'],'100 rows per class; accessed after selection freeze']],
        selection_and_compute=[
            f"The {len(ep['configurations'])} adapter configurations each run all ten seeds, giving 320 complete {ep['settings']['epochs']}-epoch fits and {eq_updates} updates per adapter. Mean adjusted tuning macro-F1 selects CE, while the other rules use their declared effective priors. Calibration is development selection data, not untouched data reserved until after global configuration freeze.",
            'Each EQ sampler searches six onset/coefficient settings; CE and Balanced Softmax each search six learning rates; BCL searches six coefficient/temperature settings. EQ-off factorial controls remain fixed and matched. Equal candidate counts do not make the selected EQ-on pipeline an unchanged-treatment factorial intervention.',
            'After all configuration choices freeze, eleven named adapter policies per seed supply native readouts and two newly fitted linear heads. This gives 220 new heads and 660 named raw/adjusted readouts. Uniform-prior adjusted aliases repeat predictions. The CPU backend/thread count was chosen by fit-only throughput measurements.',
            'A hash-bound analysis amendment was recorded before any confirmation scores. It retains four fixed-treatment factorial contrasts and adds two selected stronger-control contrasts to one six-member primary family; two selected EQ pipeline effects are secondary. Supplied timestamps and a preserved zero-confirmation counter establish recorded ordering, without external time attestation.',
            'The original audit verifies all 320 adapter and 220 linear-head states. A separate local weight-free replay verifies all 320 saved tuning captures, 660 confirmation readouts and 330 decision-offset curves with zero model-state loads. It independently regenerates the clean development split, rank and thinning; it does not recompute feature extraction or fit-scaler estimation.'
        ],secondary_tables=secondary_eq(eq['aggregate']))
    base['equalization']['contrast_notes'].append('The stronger-control contrasts compare complete validation-selected pipelines; their learning rates can differ. All fixed factorial contrasts use the same loss target and the same prespecified EQ treatment. These statements do not establish a full GAN-versus-ViT causal ablation.')
    base['vision'].update(source_rows=vp['source_rows'],fit_rows=fit,confirmation_rows=vp['confirmation_rows'],
        tuning_rows=vp['tune_rows'],calibration_rows=vp['cal_rows'],tuning_fits=12,final_fits=60,
        confirmation_readouts=120,official_exploratory_readouts=60,budget_label=f"{vp['settings']['epochs']} epochs; 1,800 AdamW updates per fit",
        method_notes=[
            'Six arms combine frozen DeiT-tiny, last-two-block DeiT-tiny tuning, or frozen DINOv2 ViT-S/14 with CE or Balanced Softmax. Frozen models train linear heads. Partial DeiT tunes the final two blocks, final norm and classifier; the first ten blocks stay frozen in evaluation mode.',
            'All original 64x64 images use the same deterministic RGB, bicubic resize/crop to 224, and ImageNet normalization. Enlarging images adds no observed detail. This study introduces no stochastic image augmentation.',
            'The FP16 frozen DeiT prefix has a train-only reconstruction-error gate. Its final trainable block computes only the CLS query/output while retaining all keys/values; zero-dropout outputs and gradients are independently checked against the complete block. DINO uses its own fit-only feature scaler.'
        ],partition_table=[['Full source',vp['source_rows'],'100,000 development plus 10,000 official evaluation images'],['Imbalanced fit',fit,'Same documented fixed image partition as EQ'],
            ['Tuning',vp['tune_rows'],'Learning-rate and checkpoint selection'],['Calibration',vp['cal_rows'],'Decision adjustment within checkpoint/LR selection'],
            ['Confirmation',vp['confirmation_rows'],'Primary calibrated macro-F1 after complete global freeze'],['Official evaluation',vp['source_rows']-100000,'Exploratory only']],
        selection_and_compute=[
            f"Dedicated tuning seed {vp['settings']['tuning_seed']} evaluates two learning rates for each of six arms, followed by ten final seeds. Every fit completes {vp['settings']['epochs']} epochs, batch {vp['settings']['batch_size']}, 120 updates per epoch and 1,800 total updates. AdamW uses warmup/cosine decay, weight decay 0.01 and clipping at 1.",
            'CE selects a calibration-set decision offset at every epoch and ranks checkpoints by adjusted tuning macro-F1. Consequently calibration is development data in this study. All six learning rates and all sixty final fit receipts are verified before any confirmation or official-evaluation outcome is opened.',
            'Frozen head learning rates are 0.001/0.003; partial encoder rates are 0.00001/0.00003 with a 100-fold head multiplier. Completed training budgets are retained even when an earlier checkpoint wins. Balanced Softmax has a uniform effective prior, so its confirmation LA view repeats raw predictions.',
            'Independent numeric replay covers sixty final models, 180 named readouts and 72 tuning/final development captures. A separate read-only audit loads all 72 actual checkpoint states and verifies tensor inventories, trainable/frozen eligibility, paired initialization, AdamW moments/groups and all 1,800 steps. The portable bundle loads zero states and does not reconstruct encoder inference.'
        ])
    base['vision']['contrast_notes'].append('DINO versus DeiT compares model packages that differ in pretraining, width, patch size and capacity. Partial versus frozen DeiT changes trainable parameters and optimization. Neither contrast isolates architecture alone; the partial model is not full end-to-end fine-tuning. Reported wrapper parameter counts do not represent the whole pretrained backbone.')
    base['text'].update(source_rows=tm['rows'],fit_rows=ts['fit'],confirmation_rows=ts['confirmation'],
        tuning_rows=ts['tune'],calibration_rows=ts['calibration'],canonical_development_rows=tm['dedup_audit']['canonical_development_rows'],
        tuning_fits=26,final_fits=130,confirmation_readouts=190,native_exploratory_readouts=100,
        budget_label=f"{tp['adapter_epochs']} epochs/5,200 updates or matched {tp['partial_epochs']} epochs/318 updates",
        method_notes=[
            'Thirteen text arms retrain random mean-pooled controls and official pretrained BERT controls on the new partitions. Seven predictive frozen/mean arms use fifty epochs; matched frozen CLS, last-one-block and last-two-block BERT each pair CE and Balanced Softmax under the same three-epoch batch lists.',
            'The fifty-epoch BERT representation is attention-masked contextual mean pooling with a fit-only scaler. The matched/partial group uses final CLS and its own shared fit-only scaler. The random mean baseline uses a fit-only 50,000-word vocabulary, a 128-word cap and a randomly initialized embedding-bag model. Word caps and WordPiece positions are different text units.',
            'Natural sampling uses the empirical fit distribution; class-uniform sampling uses replacement without an extra class weight. Balanced Softmax uses naturally sampled CE(logits + log(fit counts)) and evaluates raw logits with uniform effective prior. Natural CE/EQ alone receive calibration-selected LA. The text EQ coefficient is 0.05 throughout all fifty adapter epochs; the image fixed-reference treatment starts at fraction 0.8.',
            'The matched frozen CLS control uses CPU float32; selective tuning uses MPS float16 autocast with float32 master weights. Activating transformer dropout is part of the partial regime. The prospective CLS-only final-layer computation preserves all K/V tokens and is output/gradient equivalent at zero dropout; training dropout has the same marginal law with changed RNG ordering.'
        ],partition_table=[['Full native source',tm['rows'],'560,000 development and 70,000 native test records'],['Canonical development',tm['dedup_audit']['canonical_development_rows'],'Normalized full-record/content exact deduplication before partitioning'],
            ['Long-tail fit',ts['fit'],'14 classes; nominal 100:1 thinning; fit-only vocabulary/scalers'],['Tuning',ts['tune'],'2,000 records per class; global LR selection only'],
            ['Calibration',ts['calibration'],'2,000 per class; freeze CE decision offset before confirmation inference'],['Confirmation',ts['confirmation'],'4,000 per class; primary balanced train-derived holdout'],
            ['Native test',ts['native_test'],'Exploratory for frozen/mean arms only']],
        selection_and_compute=[
            'Two full-budget learning-rate candidates per arm at dedicated seed 9 give 26 tuning fits. One global method learning rate is frozen before final seeds 10-19. Tune-only macro-F1 selects final-epoch candidates; all thirteen choices and all 130 final fits pass a whole-study gate before confirmation.',
            f"The fifty-epoch methods complete {tp['adapter_epochs']*tp['adapter_steps_per_epoch']:,} updates; the matched/partial methods complete {tp['partial_epochs']*tp['partial_steps_per_epoch']} updates on identical three-epoch fit coverage. Length buckets remove masked padding while keeping source IDs/valid tokens. Batch/backend scheduling was selected with fit-only throughput pilots.",
            'Each CE calibration choice is persisted before its confirmation inference. All 190 named confirmation readouts and 100 exploratory native readouts retain exact labels, row IDs/ranks, logits, decision priors/tau and histories. Named raw/adjusted views are not additional independent encoder training fits.',
            'Independent replay verifies 26 global tuning captures, 130 final cells, source-row binding, all metrics and the eight-member primary family. Its original audit loads all 156 candidate/final checkpoint states; the portable audit loads zero and explicitly declares 156 omitted states.'
        ])
    base['text']['contrast_notes'].append('Each text arm selects its own learning rate. EQ-on/off and Balanced Softmax/CE contrasts therefore compare selected pipelines and can include learning-rate differences; they do not isolate a loss toggle. Selective tuning is compared to the matched three-epoch frozen CLS group. Its contrast includes dropout, precision and backend effects. The fifty-epoch mean-pooled BERT baseline is a predictive reference; different pooling and budgets preclude attributing that difference solely to encoder tuning. Text sampler objectives differ, so their EQ contrasts do not establish an unchanged-objective factorial interaction.')
    base['tabular'].update(source_rows=bm['source_rows'],fit_rows=bs['fit'],confirmation_rows=bs['confirmation'],
        tuning_rows=bs['tune'],calibration_rows=bs['calibration'],tuning_fits=8,final_fits=40,confirmation_readouts=80,
        budget_label=f"{bp['neural_epochs']} epochs; MLP {tab_updates['mlp']:,}/FT {tab_updates['ft']:,} updates; trees 600 rounds",
        method_notes=[
            'Four families compare a natural CE MLP, effective-number-weighted MLP, a genuine FT-Transformer implementation and XGBoost histogram trees. This study retains the natural Covertype class frequencies and adds no synthetic observations.',
            'Fit-only statistics standardize ten continuous variables. MLP/trees use all 54 original predictors; FT uses ten numeric feature tokens plus two categorical tokens that losslessly encode the four wilderness and forty soil indicators. Exact one-hot reconstruction is audited.',
            'The FT implementation has numerical affine feature tokens, categorical embeddings/biases, CLS, and two attention/ReGLU blocks. It is an architecture-family implementation with disclosed settings, not the unmodified official RTDL code. Effective-number weighting applies throughout MLP fitting and is normalized to expected fit sample weight one.'
        ],partition_table=[['Full source',bm['source_rows'],'All canonical UCI records; original row IDs retained'],['Fit',bs['fit'],'Class-stratified 70% floor; transform and gradients'],
            ['Tuning',bs['tune'],'10% floor; raw-logit checkpoint/configuration selection'],['Calibration',bs['calibration'],'10% floor; decision-offset selection after global configuration freeze'],
            ['Confirmation',bs['confirmation'],'Per-class remainder; primary natural-prior evaluation']],
        selection_and_compute=[
            f"Dedicated tuning seed {bp['tuning_seed']} evaluates two configurations per family, giving eight full-budget trials; ten final seeds give forty models and eighty raw/adjusted readouts. Neural models complete {bp['neural_epochs']} AdamW epochs with five-epoch warmup/cosine decay; MLP batch {bp['mlp_batch_size']} gives {tab_updates['mlp']:,} updates and FT batch {bp['ft_batch_size']} gives {tab_updates['ft']:,} updates.",
            'XGBoost completes 600 multiclass boosting rounds (4,200 trees); the tuning set selects the reported iteration among 50-round intervals through 600. Equal two-trial searches use model-specific budgets and do not claim equal FLOPs or wall time.',
            'Raw-logit tuning macro-F1 selects configurations/checkpoints. All eight tuning captures and four selected family configurations freeze before calibration/confirmation. Macro-F1 decision offsets use the effective weighted prior for the weighted MLP and the fit-frequency prior for the other families.',
            f"The fresh confirmation partition overlaps {bm['prior_study_test_overlap_with_new_confirmation']:,} rows of the previous study's test partition. Shared CPU/MPS load and checkpoint resumption affect invocation timings; retained training time and actual phase wall time are separate records. Runtime records do not establish isolated efficiency.",
            'The original metadata/checkpoint audit includes the scientific runner’s numeric self-replay. A separate verifier imports no scientific runner and recomputes all forty final prediction cells, eighty raw/adjusted readouts, forty eight-candidate decision-offset curves, scalar and per-class metrics with NumPy confusion matrices crosschecked against scikit-learn. The portable final prediction replay loads zero model weights and does not reconstruct inference.'
        ])
    base['tabular']['contrast_notes'].append('The three primary effects compare adjusted FT, XGBoost and effective-number MLP to adjusted natural CE MLP. Natural-prior confirmation macro-F1 is primary. Uniform-prior reweighting is a secondary diagnostic on the same records, not another independent dataset.')
    return base

def assemble(root, global_file, references_file):
    root=Path(root).resolve();global_file=Path(global_file).resolve();references_file=Path(references_file).resolve()
    require(global_file.is_relative_to(root),'Global analysis must be inside evidence root')
    markers=complete_markers(root)
    bundles=load_bundles(root,markers)
    require(global_file.is_file(),'Complete saved global analysis required')
    global_source=read(global_file);analysis=copy.deepcopy(global_source)
    require(all('label' not in r for r in analysis['comparisons']),'Saved analysis must contain numeric fields only; labels belong in report payload')
    for r in analysis['comparisons']:
        require(r['name'] in LABELS,'Undeclared comparison label');r['label']=LABELS[r['name']]
    builder=module('followup_evidence_builder',ROOT/'work/build_followup_report.py')
    aggregates={n:bundles[n]['objects']['aggregate'] for n in STUDIES}
    digests={n:next(r['sha256'] for r in bundles[n]['records'] if r['role']=='aggregate') for n in STUDIES}
    builder.bind_global(analysis,aggregates,digests)
    studies=study_payloads(bundles,analysis)
    refs=read(references_file)['primary_sources']
    require(len({r['key'] for r in refs})==len(refs) and any(r['key']=='dbpedia_14' for r in refs),'Primary dataset/method reference catalog incomplete')
    positive=sum(r['mean']>0 and r['signflip_p_study_holm']<=.05 for r in analysis['comparisons'])
    negative=sum(r['mean']<0 and r['signflip_p_study_holm']<=.05 for r in analysis['comparisons'])
    pooled=sum(r['signflip_p_global_holm']<=.05 for r in analysis['comparisons'])
    evidence=dict(status='all_studies_complete_independently_verified',completed_utc=datetime.now(timezone.utc).isoformat(),
        seeds=SEEDS,studies=studies,global_analysis=analysis,
        global_analysis_evidence=dict(path=str(global_file.relative_to(root)),sha256=sha(global_file)),
        abstract='This completed follow-up examines stronger long-tail classification controls across images, text and tabular data. Tiny ImageNet supports a fixed-treatment EQ/sampler study and an encoder/partial-tuning study; DBpedia-14 tests contextual BERT controls and selective tuning; Covertype tests weighted objectives, FT-Transformer and boosted trees. Every study retains ten paired optimization seeds, separate fit/tuning/calibration/confirmation partitions, full declared budgets, saved predictions and independent numeric verification. The primary families contain 6, 5, 8 and 3 paired macro-F1 comparisons. Per-study Holm tests are primary; a hash-bound 22-comparison global sensitivity is also retained. Results are conditional on these source partitions and model packages.',
        principal_findings=[result_statement(studies[n]['title'],studies[n]['summary']) for n in STUDIES]+[
            f"Across the 22 declared comparisons, {positive} positive and {negative} negative effects reach within-study exact sign-flip Holm p <= 0.05; {pooled} comparisons reach the separate global Holm sensitivity threshold. The complete tables retain signs, intervals and non-detections."],
        verification_notes=[
            'Each package passes its completion marker, complete file catalog hashes, frozen protocol/source identities and original independent audit. The assembler selects table rows by exact source identifiers and reruns the report builder binding checks. The saved global analysis file is hashed separately from the report labels.',
            'The EQ audit checks all 320 adapter and 220 head states. Vision has separate 72-state and 60-model/180-readout numeric audits. Text has 156-state and 130-model/190-confirmation/100-native coverage. Tabular retains eight tuning and forty final state/budget audits and eighty numeric readouts. These are original working-run scopes; portable archives omit states and declare their narrower replay scopes.',
            'Numeric verification recomputes metrics, global choices, decision-offset selections and paired statistics from saved predictions and labels. Actual-state audits check identities, tensors and optimizer budgets where retained. They do not independently rerun training, all encoder inference or acquisition; SHA256 binds artifacts rather than proving external replication.'
        ],selection_notes=[
            'Configuration and checkpoint selection never optimizes confirmation macro-F1. EQ averages development scores over its ten seeds; vision uses dedicated tuning seed 901; text and tabular use seed 9. Final seed pairing and fixed source partitions are prespecified.',
            'Calibration here selects a classification decision offset by macro-F1; the studies do not measure probability reliability. EQ and vision use calibrated tuning policies, so their calibration rows are development selection data. Text and tabular select configurations using raw tuning metrics and persist calibration decisions after global choice freeze and before confirmation inference.',
            'The EQ amendment precedes recorded confirmation access and changes analysis roles without changing training. Primary fixed-treatment contrasts use onset .8/coefficient .05; validation-selected onset/coefficient effects remain secondary. The native source evaluation partitions remain exploratory where earlier reports informed the direction.'
            ,'The sign-flip reference distribution is exhaustively enumerated over 1,024 assignments, but its inferential validity assumes paired sign exchangeability or symmetry under the null. Paired-t intervals use a normality approximation for the mean of ten optimization-seed differences, conditional on the fixed partitions. Neither rule supplies external-dataset uncertainty.'
        ],recommendation_notes=[
            'Use the within-study paired comparisons and their uncertainty when choosing a pipeline for these fixed partitions. Descriptive score leaders are not evidence of universal superiority. Small, undetected or negative improvements are retained rather than removed from the report.',
            'Strong calibrated CE and Balanced Softmax controls remain necessary when evaluating EQ or contrastive methods. Compare selective tuning against the matched frozen control. Evaluate tabular attention alongside boosted trees and conventional MLP objectives under the natural data distribution.'
        ],limitations=[
            'The official DeiT checkpoint is ImageNet-pretrained, while Tiny ImageNet is sampled and downsampled from ImageNet. Item-level pretraining exclusion has not been established. Disjoint experimental partitions therefore do not establish independence from pretraining; this is a source-derived limitation, not a measured exact-item overlap rate.',
            'The ten-seed intervals condition on the fixed source partitions and selected configurations. They do not cover hyperparameter-selection uncertainty or external dataset/cohort uncertainty.',
            'The image studies reuse Tiny ImageNet source rows; historical models may have seen the new confirmation images. Covertype is a previously studied, spatially dependent source. New partitions protect this round from its own selection, but do not provide an independent dataset, geographic cohort or external replication.',
            'BERT was pretrained on Wikipedia and DBpedia abstracts derive from Wikipedia. Normalized exact deduplication does not remove pretraining, entity or paraphrase overlap. Encoder-package contrasts also change pretraining/capacity, while partial tuning changes dropout, optimizer and sometimes precision/backend. Results do not isolate transformer architecture alone.',
            'Ten optimization seeds quantify initialization/training variation conditional on fixed partitions. Their intervals exclude data resampling, dependence across entities/locations and foundation-model pretraining uncertainty. Non-significance does not establish equivalence; a degenerate zero-variance paired difference uses the declared finite statistics convention.',
            'The studies do not conduct a fully matched GAN-versus-ViT intervention, full BCL image-augmentation reproduction, full backbone fine-tuning or probability-calibration reliability study. Optional SMOTE graphs and extra synthetic observations are absent from these primary follow-ups. Wrapper parameter counts must not be described as complete pretrained backbone capacity.',
            'Reported training/phase timings reflect a shared machine, selected backends and some checkpoint resumption. They document resource use, not isolated throughput superiority or equal compute across model families.'
        ],reproduction_notes=[
            'The public source and study documentation specify acquisition, canonicalization, source/model revisions, fit-only transformations, training grids, budgets and inference gates. Portable evidence retains scalar/class results, labels, source IDs/ranks, predictions, selected configuration records, histories and original audit receipts.',
            'Full raw numeric portable archives remain local in the supplied evidence archive. Git contains smaller derived results, summaries, audit receipts, catalogs, report and source; the multi-gigabyte raw prediction blobs are not uploaded to Git. Image prediction row order is bound to saved partition indices and source labels rather than explicit per-prediction source-ID arrays.',
            'To reproduce the current EQ analysis, run scripts/apply_followup_amendment.py on equalization/public_amendment_inputs with the frozen scientific source and counter snapshot. The historical runner aggregate remains archived; the amendment helper recreates the six-member primary and two secondary analyses.',
            'Run each package numeric verifier in its documented weight-free mode; it must declare zero loaded model states. Original actual-state verification requires the omitted working checkpoints, dense inputs and pinned weights. Then run scripts/analyze_followup_comparisons.py on the four complete aggregates and this assembler before the report builder.',
            'Scientific source hashes, protocol fingerprints, aggregate hashes, separate original/portable audit receipts and the global analysis identity are retained below. Credentials, private machine paths and operational logs are excluded from publication. Supplied record timestamps do not constitute external timestamp attestation.'
        ],references=[{k:r[k] for k in ('author','title','url')} for r in refs],
        reference_catalog_sha256=sha(references_file),figure_status='Awaiting root-owned scientific plotting and visual QA; no figures inserted by assembler')
    identities=[]
    for n,s in studies.items():
        identities.extend([dict(role=n+' scientific source',sha256=s['source_sha256']),
                           dict(role=n+' protocol fingerprint',sha256=bundles[n]['objects']['protocol']['fingerprint'])])
        for role in ('aggregate','independent_verification','checkpoint_verification'):
            found=[r for r in s['evidence_files'] if r['role']==role]
            if found:identities.append(dict(role=n+' '+role,sha256=found[0]['sha256']))
    identities.extend([dict(role='Global 22-comparison analysis file',sha256=sha(global_file)),
                       dict(role='Global comparison helper source',sha256=analysis['analysis_source_sha256'])])
    evidence['identity_table']=identities
    return evidence

def write_evidence(root, global_file, references_file, out, check_only=False):
    root=Path(root).resolve();out=Path(out).absolute()
    require(out.parent.resolve()==root,'Evidence payload must be directly inside evidence root')
    evidence=assemble(root,global_file,references_file)
    if out.exists():
        # Preserve the original authoring time on an exact repeat; no score or
        # source field is borrowed from the existing authoring payload.
        evidence['completed_utc']=read(out)['completed_utc']
    builder=module('followup_payload_gate',ROOT/'work/build_followup_report.py')
    temporary=out.with_name(out.name+'.pending')
    require(not temporary.exists(),'Preserve existing pending authoring payload; inspect before retry')
    try:
        temporary.write_text(json.dumps(evidence,indent=2,allow_nan=False)+'\n')
        builder.load_evidence(temporary)
        if check_only:temporary.unlink()
        elif out.exists():
            require(read(out)==evidence,'Existing evidence payload differs; use a new destination')
            temporary.unlink()
        else:temporary.replace(out)
    except BaseException:
        if temporary.exists():temporary.unlink()
        raise
    return dict(status='all_four_complete_evidence_checked' if check_only else 'all_four_complete_evidence_assembled',
                studies=4,comparisons=22,summary_rows={n:len(s['summary']) for n,s in evidence['studies'].items()},
                protocol={n:read(root/next(r['path'] for r in s['evidence_files'] if r['role']=='protocol'))['fingerprint'] for n,s in evidence['studies'].items()},
                figures_inserted=0,pdf_authored=False,partial_report_emitted=False,
                **({} if check_only else dict(evidence_sha256=sha(out))))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence-root',type=Path,default=ROOT/'outputs/REBAL_Next_Round_Evidence')
    p.add_argument('--global-analysis',type=Path)
    p.add_argument('--references',type=Path,default=ROOT/'work/next_round/report_references.json')
    p.add_argument('--out',type=Path);p.add_argument('--check-only',action='store_true')
    args=p.parse_args();root=args.evidence_root
    print(json.dumps(write_evidence(root,args.global_analysis or root/'global_analysis.json',args.references,
                                    args.out or root/'evidence.json',args.check_only),indent=2))

if __name__=='__main__':main()
