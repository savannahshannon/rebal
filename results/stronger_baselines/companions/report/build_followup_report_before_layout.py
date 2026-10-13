#!/usr/bin/env python3
"""Author the combined follow-up PDF only from complete audited evidence.

The PDF skill authoring marker was run once before this builder was created.
This file is prepared while the runs proceed; no partial-results PDF is emitted.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import math
from datetime import datetime, timezone
from xml.sax.saxutils import escape

def require(condition, message):
    if not condition:
        raise ValueError(message)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def numeric_match(claim, expected, message):
    if isinstance(expected,dict):
        require(isinstance(claim,dict) and set(claim)==set(expected),message)
        for key in expected:numeric_match(claim[key],expected[key],message)
    elif isinstance(expected,list):
        require(isinstance(claim,list) and len(claim)==len(expected),message)
        for actual,wanted in zip(claim,expected):numeric_match(actual,wanted,message)
    elif isinstance(expected,(float,int)) and not isinstance(expected,bool):
        require(isinstance(claim,(float,int)) and not isinstance(claim,bool) and math.isfinite(claim) and
                math.isfinite(expected) and abs(claim-expected)<=1e-12,message)
    else:require(claim==expected,message)

def bind_summary(name,study,aggregate):
    """A display label is authored; every displayed numeric field has a selector."""
    source=aggregate['summary'];keys=[];display_keys=[]
    for row in source:
        if name=='vision':
            keys.append((row['arm'],row['endpoint'],row['metric']))
            require(row['n']==10,'Vision summary source needs ten seeds')
        else:
            keys.append(row['method'])
            require(row.get('n',len(row.get('seeds',[])))==10,'Summary source needs ten seeds')
            if name=='tabular':require(row['seeds']==list(range(10,20)),'Tabular summary seed order differs')
    require(len(keys)==len(set(keys)),name+' duplicate aggregate summary identity')
    require(study['summary'],name+' display summary is empty')
    for displayed in study['summary']:
        selector=displayed['source_selector'];display_keys.append(fingerprint(selector))
        if name=='vision':
            require(set(selector)=={'arm','endpoint'},'Vision summary selector requires arm and endpoint')
            matches=[r for r in source if r['arm']==selector['arm'] and r['endpoint']==selector['endpoint']]
            require(len(matches)==6 and len({r['metric'] for r in matches})==6,'Vision display source is incomplete')
            actual={r['metric']:{'mean':r['mean'],'sd':r['sd']} for r in matches}
        else:
            require(set(selector)=={'method'},'Summary selector requires method')
            matches=[r for r in source if r['method']==selector['method']]
            require(len(matches)==1,'Display source selector is missing or ambiguous')
            actual=matches[0]['metrics']
        require({'macro_f1','tail_f1','accuracy'}<=set(displayed['metrics'])<=set(actual),'Display metric names differ')
        for metric,claim in displayed['metrics'].items():
            require({'mean','sd'}<=set(claim)<=set(actual[metric]),'Display metric fields differ')
            numeric_match(claim,{k:actual[metric][k] for k in claim},'Displayed summary differs from hashed aggregate')
    require(len(display_keys)==len(set(display_keys)),name+' duplicate display source selector')

def bind_global(claim,aggregates,input_digests):
    """Recompute normalization/Holm from hashed aggregates; labels alone are free."""
    helper=Path(__file__).parent/'rebal/scripts/analyze_followup_comparisons.py'
    spec=importlib.util.spec_from_file_location('report_global_analysis',helper)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    expected=module.analyze(aggregates)
    require(set(claim)==set(expected)|{'input_sha256','analysis_source_sha256'},'Global authoring payload fields differ')
    require(claim['input_sha256']==input_digests and claim['analysis_source_sha256']==sha(helper),'Global analysis source/input hashes differ')
    for key in expected:
        if key!='comparisons':numeric_match(claim[key],expected[key],'Global authoring payload differs from numeric replay')
    require(len(claim['comparisons'])==22,'Exactly 22 displayed comparisons required')
    for actual,wanted in zip(claim['comparisons'],expected['comparisons']):
        require(set(actual)==set(wanted)|{'label'} and isinstance(actual['label'],str),'Comparison permits an added label only')
        numeric_match({k:actual[k] for k in wanted},wanted,'Displayed contrast differs from hashed aggregate replay')

def bind_secondary_tables(name,study,aggregate):
    """Optional EQ diagnostics retain numeric source binding in rendered cells."""
    for table in study.get('secondary_tables',[]):
        require(name=='equalization','Only prespecified EQ secondary tables are supported')
        records=table['source_selectors'];rows=table['rows']
        require(len(records)==len(rows) and records,'Secondary table row source is incomplete')
        require(len({fingerprint(record) for record in records})==len(records),'Duplicate secondary table selector')
        if table['source_kind']=='selected_EQ_secondary':
            require(len(table['headers'])==4,'Secondary EQ comparison columns differ')
            source={r['name']:r for r in aggregate['secondary_comparisons']}
            require(set(source)=={'selected_natural_EQ_pipeline_effect','selected_balanced_EQ_pipeline_effect'},'Selected EQ secondary family differs')
            for selector,row in zip(records,rows):
                require(set(selector)=={'comparison'} and selector['comparison'] in source,'Unknown secondary EQ comparison')
                r=source[selector['comparison']]
                require(r['n']==10 and r['seeds']==list(range(10,20)),'Secondary EQ source needs ten paired seeds')
                expected=[f"{100*r['mean']:+.4f}",f"[{100*r['ci95'][0]:+.4f}, {100*r['ci95'][1]:+.4f}]",f"{r['signflip_p']:.6f}"]
                require(len(row)==4 and row[1:]==expected,'Displayed secondary contrast differs from hashed aggregate')
        elif table['source_kind']=='EQ_head_diagnostic':
            require(len(table['headers'])==4,'Secondary head comparison columns differ')
            source={r['method']:r for r in aggregate['summary']}
            for selector,row in zip(records,rows):
                require(set(selector)=={'arm'} and selector['arm'] in {'ce','balanced_softmax','bcl_dropout'},'Unknown head diagnostic arm')
                expected=[]
                for head in ('native','natural','ros'):
                    r=source[selector['arm']+'__'+head+'_la'];m=r['metrics']['macro_f1']
                    require(r['n']==10,'Head diagnostic source needs ten seeds')
                    expected.append(f"{100*m['mean']:.2f} +/- {100*m['sd']:.2f}")
                require(len(row)==4 and row[1:]==expected,'Displayed head diagnostic differs from hashed aggregate')
        else:require(False,'Unknown secondary table source kind')

def partition_lengths(path, fields):
    """Read only numeric ID-array shapes; prediction arrays are not accessed."""
    import numpy as np
    with np.load(path,allow_pickle=False) as arrays:
        result={}
        for role,key in fields.items():
            ids=arrays[key]
            require(ids.ndim==1 and np.issubdtype(ids.dtype,np.integer),
                    'Displayed partition counts require one-dimensional integer IDs')
            result[role]=len(ids)
    return result

def bind_display_scope(studies, protocols, metadata, partitions, audits, aggregates):
    """Bind all displayed count fields and fixed-budget labels to hashed inputs.

    The reported source is larger than the long-tail fit partition. EQ shares
    the vision partition and source/cache identities; it reuses tuning adapters
    and separately fits linear heads, whose count comes from the bound audit.
    """
    ep,vp,tp,bp=(protocols[n] for n in ('equalization','vision','text','tabular'))
    es,vs=ep['settings'],vp['settings']
    require(ep['source_archive_sha256']==vp['official_archive_sha256'] and
            ep['cache_manifest_sha256']==vp['deit_cache_manifest_sha256'],
            'Displayed image scope requires the same source/cache identities')
    image=partitions['vision']
    require({k:image[k] for k in ('fit','tune','calibration','confirmation')}==dict(fit=vp['fit_rows'],tune=vp['tune_rows'],calibration=vp['cal_rows'],confirmation=vp['confirmation_rows']),
            'Image partition counts differ from hashed protocol')
    classes=len(ep['hygiene']['clean_class_counts'])
    holdouts={k:classes*v for k,v in ep['holdouts_per_class'].items()}
    require(holdouts=={k:image[k] for k in ('tune','calibration','confirmation')},
            'EQ holdout counts differ from the shared hashed partition')
    clean=ep['hygiene']['retained_examples'];pool=clean-sum(holdouts.values())
    require(image['fit_pool']==pool,'Shared image fit-pool count differs from hashed EQ source hygiene')
    expected={}
    eq_updates=es['epochs']*math.ceil(image['fit']/es['batch_size'])
    expected['equalization']=dict(source_rows=vp['source_rows'],fit_rows=image['fit'],
        confirmation_rows=image['confirmation'],tuning_rows=image['tune'],calibration_rows=image['calibration'],
        canonical_development_rows=clean,fit_pool_rows=pool,tuning_fits=len(ep['configurations'])*len(es['seeds']),
        final_fits=audits['equalization']['head_checkpoint_states_verified'],confirmation_seeds=len(es['seeds']),
        confirmation_readouts=aggregates['equalization']['readouts'],
        budget_label=f"{es['epochs']} epochs/{eq_updates} adapter updates; {es['head_steps']:,} head updates")
    arms=vs['arms'];vision_updates=vs['epochs']*math.ceil(image['fit']/vs['batch_size'])
    tuning=sum(len(vs['encoder_lr_candidates'] if arm=='deit_last2' else vs['head_lr_candidates']) for arm,_ in arms)
    expected['vision']=dict(source_rows=vp['source_rows'],fit_rows=image['fit'],confirmation_rows=image['confirmation'],
        tuning_rows=image['tune'],calibration_rows=image['calibration'],tuning_fits=tuning,final_fits=len(arms)*len(vs['seeds']),
        confirmation_readouts=2*len(arms)*len(vs['seeds']),official_exploratory_readouts=len(arms)*len(vs['seeds']),
        budget_label=f"{vs['epochs']} epochs; {vision_updates:,} AdamW updates per fit")
    for name,protocol,key in [('text',tp,'dataset_fingerprint'),('tabular',bp,'data_fingerprint')]:
        meta=metadata[name]
        require(fingerprint({k:v for k,v in meta.items() if k!='fingerprint'})==meta['fingerprint'] and
                meta['fingerprint']==protocol[key],name+' displayed source metadata fingerprint differs')
        require(partitions[name]=={k:meta['split_counts'][k] for k in ('fit','tune','calibration','confirmation')},
                name+' partition counts differ from hashed metadata')
    tm,bm=metadata['text'],metadata['tabular'];text=partitions['text'];tab=partitions['tabular']
    require(text['fit']==tp['fit_examples'],'Text declared fit count differs from hashed partition')
    text_tunes=sum(len(tp['partial_lrs'] if method.startswith('last') else tp['adapter_lrs']) for method in tp['methods'])
    expected['text']=dict(source_rows=tm['rows'],fit_rows=text['fit'],confirmation_rows=text['confirmation'],
        tuning_rows=text['tune'],calibration_rows=text['calibration'],canonical_development_rows=tm['dedup_audit']['canonical_development_rows'],
        tuning_fits=text_tunes,final_fits=len(tp['methods'])*len(tp['seeds']),
        confirmation_readouts=audits['text']['readouts'],native_exploratory_readouts=audits['text']['native_readouts'],
        budget_label=f"{tp['adapter_epochs']} epochs/{tp['adapter_epochs']*tp['adapter_steps_per_epoch']:,} updates or matched {tp['partial_epochs']} epochs/{tp['partial_epochs']*tp['partial_steps_per_epoch']} updates")
    require(sum(bm['split_counts'].values())==bm['source_rows'],'Tabular complete source partition counts differ')
    rounds={c['n_estimators'] for c in bp['grids']['xgboost']}
    require(len(rounds)==1,'A single completed boosting budget is required for the displayed label')
    updates={family:bp['neural_epochs']*math.ceil(tab['fit']/bp[family+'_batch_size']) for family in ('mlp','ft')}
    expected['tabular']=dict(source_rows=bm['source_rows'],fit_rows=tab['fit'],confirmation_rows=tab['confirmation'],
        tuning_rows=tab['tune'],calibration_rows=tab['calibration'],tuning_fits=sum(len(grid) for grid in bp['grids'].values()),
        final_fits=len(bp['grids'])*len(bp['seeds']),confirmation_readouts=aggregates['tabular']['readouts'],
        budget_label=f"{bp['neural_epochs']} epochs; MLP {updates['mlp']:,}/FT {updates['ft']:,} updates; trees {next(iter(rounds))} rounds")
    table_counts={
        'equalization':{'Canonical development':clean,'Fit pool before thinning':pool,'Imbalanced fit':image['fit'],
                       'Tuning':image['tune'],'Calibration':image['calibration'],'Confirmation':image['confirmation']},
        'vision':{'Full source':vp['source_rows'],'Imbalanced fit':image['fit'],'Tuning':image['tune'],
                  'Calibration':image['calibration'],'Confirmation':image['confirmation'],
                  'Official evaluation':vp['prefix_rows']-sum(image[k] for k in ('fit','tune','calibration','confirmation'))},
        'text':{'Full native source':tm['rows'],'Canonical development':tm['dedup_audit']['canonical_development_rows'],
                'Long-tail fit':text['fit'],'Tuning':text['tune'],'Calibration':text['calibration'],
                'Confirmation':text['confirmation'],'Native test':tm['split_counts']['native_test']},
        'tabular':{'Full source':bm['source_rows'],'Fit':tab['fit'],'Tuning':tab['tune'],
                   'Calibration':tab['calibration'],'Confirmation':tab['confirmation']},
    }
    for name,fields in expected.items():
        for key,value in fields.items():
            claim=studies[name].get(key)
            require((type(claim) is int if type(value) is int else isinstance(claim,str)) and claim==value,
                    'Displayed scope field differs from hashed evidence: '+name+' '+key)
        rows=studies[name].get('partition_table')
        require(isinstance(rows,list) and all(isinstance(r,list) and len(r)==3 and isinstance(r[0],str) for r in rows),
                name+' displayed partition table is missing or malformed')
        require(len(rows)==len(table_counts[name]) and {r[0] for r in rows}==set(table_counts[name]),
                name+' displayed partition table labels are missing or duplicated')
        for label,count,_ in rows:
            require(type(count) is int and count==table_counts[name][label],
                    'Displayed partition table count differs from hashed evidence: '+name+' '+label)
    return expected

def load_evidence(path):
    path=Path(path);e=json.loads(path.read_text())
    require(e['status']=='all_studies_complete_independently_verified','Report requires all completed studies')
    require(set(e['studies'])=={'equalization','vision','text','tabular'},'Four studies required')
    require(e['seeds']==list(range(10,20)),'Ten paired seeds required')
    aggregates={};input_digests={};protocols={};audits={};metadata={};partitions={}
    for name,study in e['studies'].items():
        require(study['status']=='independently_verified_complete',name+' audit incomplete')
        roles={r['role']:r for r in study['evidence_files']}
        require(len(roles)==len(study['evidence_files']) and {'aggregate','independent_verification','protocol'}<=set(roles),'Required evidence roles missing or duplicated')
        for identity in study['evidence_files']:
            relative=Path(identity['path'])
            require(not relative.is_absolute() and '..' not in relative.parts,'Unsafe evidence path')
            require(sha(path.parent/relative)==identity['sha256'],'Evidence file changed')
        aggregate=json.loads((path.parent/roles['aggregate']['path']).read_text())
        audit=json.loads((path.parent/roles['independent_verification']['path']).read_text())
        protocol=json.loads((path.parent/roles['protocol']['path']).read_text())
        protocols[name]=protocol;audits[name]=audit
        require(fingerprint({k:v for k,v in protocol.items() if k!='fingerprint'})==protocol['fingerprint'],name+' protocol fingerprint differs')
        require(study['source_sha256']==protocol['source_sha256'],name+' scientific source identity differs')
        if 'scientific_source' in roles:
            require(roles['scientific_source']['sha256']==protocol['source_sha256'],name+' scientific source file differs')
        aggregates[name]=aggregate;input_digests[name]=roles['aggregate']['sha256']
        acceptable={
            'equalization':{'independently_verified_complete'},
            'vision':{'verified_complete'},
            'text':{'independently_verified_complete','verified_complete'},
            'tabular':{'independent_integrity_selection_and_numeric_replay_passed'},
        }
        require(audit['status'] in acceptable[name],name+' independent receipt did not pass')
        if name=='equalization':
            require(aggregate['status']=='complete_independently_verified_amended_analysis' and aggregate['primary_family_size']==6 and aggregate['confirmation_cells']==10 and aggregate['readouts']==660 and aggregate['completed_tuning_cells']==320 and audit['tuning_cells_recomputed']==320 and audit['confirmation_cells']==10 and audit['readouts_recomputed']==660,'EQ complete amended result required')
            require('amendment_verification' in roles,'Independent amendment replay required')
            amended_audit=json.loads((path.parent/roles['amendment_verification']['path']).read_text())
            require(amended_audit['status']=='independently_verified_amended_expressions' and
                    amended_audit['family_size']==6 and amended_audit['seed_order']==list(range(10,20)) and
                    amended_audit['amended_aggregate_sha256']==roles['aggregate']['sha256'] and
                    amended_audit['independent_raw_logit_audit_sha256']==roles['independent_verification']['sha256'], 'Amended EQ audit identities differ')
            require(audit['protocol']==aggregate['protocol']==protocol['fingerprint'],'EQ protocol differs')
            require((audit['adapter_checkpoint_states_verified'],audit['head_checkpoint_states_verified'],audit['calibration_curves_recomputed'])==(320,220,330),
                    'Original EQ actual-state/calibration scope differs')
            require({'portable_verification','portable_manifest','portable_amendment_reproduction','historical_aggregate'}<=set(roles),'Complete local EQ raw archive replay required')
            portable=json.loads((path.parent/roles['portable_verification']['path']).read_text())
            catalog=json.loads((path.parent/roles['portable_manifest']['path']).read_text())
            reproduction=json.loads((path.parent/roles['portable_amendment_reproduction']['path']).read_text())
            require(portable['status']=='independently_verified_complete_portable_numeric' and
                    (portable['tuning_cells_recomputed'],portable['confirmation_cells'],portable['readouts_recomputed'],portable['calibration_curves_recomputed'])==(320,10,660,330) and
                    (portable['adapter_checkpoint_states_loaded'],portable['head_checkpoint_states_loaded'],portable['checkpoint_states_omitted'])==(0,0,540) and
                    portable['protocol']==protocol['fingerprint'] and portable['scientific_source_sha256']==protocol['source_sha256'] and
                    portable['aggregate_sha256']==roles['historical_aggregate']['sha256'] and portable['original_state_audit_sha256']==roles['independent_verification']['sha256'],
                    'EQ raw portable numeric audit binding differs')
            require(catalog['status']=='complete_local_weight_free_EQ_numeric_archive_independently_replayed' and catalog['identity_fingerprint']==fingerprint(catalog['identity']) and
                    catalog['identity']['protocol']==protocol['fingerprint'] and catalog['identity']['scientific_source_sha256']==protocol['source_sha256'] and
                    catalog['identity']['counts']==dict(tuning_cells=320,confirmation_cells=10,confirmation_readouts=660,calibration_curves=330,primary_comparisons=6,secondary_comparisons=2) and
                    catalog['identity']['portable_numeric_verification_sha256']==roles['portable_verification']['sha256'] and
                    catalog['identity']['amendment_replay_sha256']==roles['portable_amendment_reproduction']['sha256'] and
                    (catalog['identity']['portable_checkpoint_states_loaded'],catalog['identity']['checkpoint_states_omitted'])==(0,540),
                    'EQ complete local archive manifest binding differs')
            require(reproduction['status']=='prospective_amendment_semantically_reproduced' and reproduction['frozen_amended_aggregate_sha256']==roles['aggregate']['sha256'] and
                    (reproduction['primary_comparisons'],reproduction['secondary_comparisons'],reproduction['maximum_numeric_discrepancy'])==(6,2,0),
                    'EQ raw archive amendment replay differs')
        elif name=='vision':
            require('partition' in roles and roles['partition']['sha256']==protocol['partition_sha256'],
                    'Vision displayed scope requires its protocol-bound partition')
            partitions[name]=partition_lengths(path.parent/roles['partition']['path'],{k:k for k in ('fit','fit_pool','tune','calibration','confirmation')})
            require(aggregate['models']==60 and aggregate['readouts']==180 and audit['models_verified']==60 and audit['readouts_verified']==180,'Vision final grid incomplete')
            require(audit['aggregate_sha256']==roles['aggregate']['sha256'] and audit['protocol']==aggregate['protocol']==protocol['fingerprint'] and audit['primary_contrasts_verified']==5 and audit['summary_rows_verified']==108 and audit['scientific_source_file_verified'] is True and audit['scientific_source_sha256']==protocol['source_sha256'],'Vision audit identities differ')
            require('checkpoint_verification' in roles,'Separate actual vision checkpoint audit required')
            states=json.loads((path.parent/roles['checkpoint_verification']['path']).read_text())
            require(states['status']=='independently_verified_complete_checkpoint_states' and states['checkpoint_states_verified']==72 and states['tuning_models_verified']==12 and states['final_models_verified']==60 and states['actual_optimizer_steps_per_parameter_verified']==1800 and states['epochs_per_model_verified']==15 and states['protocol']==protocol['fingerprint'] and states['source_sha256']==protocol['source_sha256'] and states['confirmation_prediction_files_read']==0,'Vision actual checkpoint audit differs')
        elif name=='text':
            require({'metadata','partition'}<=set(roles),'Text displayed scope requires metadata and partition')
            metadata[name]=json.loads((path.parent/roles['metadata']['path']).read_text())
            partitions[name]=partition_lengths(path.parent/roles['partition']['path'],{k:k+'_ids' for k in ('fit','tune','calibration','confirmation')})
            require(aggregate['completed_cells']==130 and aggregate['expected_cells']==130 and aggregate['readouts']==190,'Text final grid incomplete')
            require(audit['aggregate_sha256']==roles['aggregate']['sha256'] and audit['protocol']==aggregate['protocol_fingerprint']==protocol['fingerprint'] and audit['models']==130 and audit['readouts']==190 and audit['native_readouts']==100 and audit['primary_family_size']==8 and audit['source_sha256']==protocol['source_sha256'],'Text audit identities differ')
            require((audit['actual_checkpoint_states_verified'],audit['checkpoint_states_omitted'])==(156,0),'Original text actual-state scope differs')
        else:
            require({'metadata','partition'}<=set(roles),'Tabular displayed scope requires metadata and partition')
            metadata[name]=json.loads((path.parent/roles['metadata']['path']).read_text())
            partitions[name]=partition_lengths(path.parent/roles['partition']['path'],{k:k for k in ('fit','tune','calibration','confirmation')})
            require(aggregate['completed_fits']==40 and aggregate['readouts']==80,'Tabular final grid incomplete')
            require(audit['protocol']==protocol['fingerprint'] and audit['numeric_replay']['aggregate_sha256']==roles['aggregate']['sha256'] and audit['numeric_replay']['source_sha256']==protocol['source_sha256'] and audit['numeric_replay']['completed_fits']==40 and audit['numeric_replay']['readouts']==80 and audit['tuning_trials_verified']==8 and audit['final_fit_configs_and_budgets_verified']==40,'Tabular audit identities differ')
            require({'independent_prediction_verification','independent_prediction_helper','source_labels','partition','scaler'}<=set(roles),'Independent final tabular prediction replay required')
            ip=json.loads((path.parent/roles['independent_prediction_verification']['path']).read_text())
            require(ip['status']=='independent_final_prediction_replay_passed' and
                    (ip['completed_fits'],ip['readouts_recomputed'],ip['calibration_curves_recomputed'],ip['tau_candidates_per_curve'],ip['calibration_candidate_scores_recomputed'],
                     ip['scalar_metrics_recomputed'],ip['per_class_metric_values_recomputed'],ip['confusion_entries_recomputed'],ip['class_supports_verified'])==(40,80,40,8,320,640,1680,3920,560) and
                    ip['model_weights_loaded']==0 and ip['scientific_runner_imported'] is False and ip['prediction_model_inference'] is False and
                    ip['protocol']==protocol['fingerprint'] and ip['scientific_source_sha256']==protocol['source_sha256'] and
                    ip['aggregate_sha256']==roles['aggregate']['sha256'] and ip['helper_source_sha256']==roles['independent_prediction_helper']['sha256'] and
                    ip['labels_sha256']==roles['source_labels']['sha256'] and ip['split_sha256']==roles['partition']['sha256'] and ip['scaler_sha256']==roles['scaler']['sha256'],
                    'Independent final tabular prediction identities or scope differ')
        require(e['global_analysis']['input_sha256'][name]==roles['aggregate']['sha256'],'Global analysis uses a different study aggregate')
        bind_summary(name,study,aggregate)
        bind_secondary_tables(name,study,aggregate)
        if study.get('figure'):
            figure=Path(study['figure'])
            require(not figure.is_absolute() and '..' not in figure.parts and
                    any(r['path']==study['figure'] for r in study['evidence_files']),'Displayed figure requires hashed evidence role')
    bind_display_scope(e['studies'],protocols,metadata,partitions,audits,aggregates)
    require(e['global_analysis']['status']=='complete_replayed_global_sensitivity' and
            e['global_analysis']['family_size']==22,'Global analysis missing')
    bind_global(e['global_analysis'],aggregates,input_digests)
    require(set(e.get('global_analysis_evidence',{}))=={'path','sha256'},'Hashed global analysis file required')
    global_record=e['global_analysis_evidence'];global_path=Path(global_record['path'])
    require(not global_path.is_absolute() and '..' not in global_path.parts,'Unsafe global analysis path')
    require(sha(path.parent/global_path)==global_record['sha256'],'Global analysis evidence file changed')
    global_source=json.loads((path.parent/global_path).read_text())
    global_claim={k:v for k,v in e['global_analysis'].items() if k!='comparisons'}
    global_claim['comparisons']=[{k:v for k,v in row.items() if k!='label'} for row in e['global_analysis']['comparisons']]
    numeric_match(global_claim,global_source,'Global analysis display differs from hashed analysis file')
    return e

def create_pdf(e, output):
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
    navy=colors.HexColor('#18364F');teal=colors.HexColor('#087E8B');light=colors.HexColor('#EDF3F6')
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='ReportTitle',fontName='Helvetica-Bold',fontSize=24,leading=29,textColor=navy,spaceAfter=18))
    styles.add(ParagraphStyle(name='ReportSection',fontName='Helvetica-Bold',fontSize=17,leading=21,textColor=navy,spaceAfter=14))
    styles.add(ParagraphStyle(name='ReportBody',fontName='Helvetica',fontSize=10,leading=14,spaceAfter=9))
    styles.add(ParagraphStyle(name='ReportSmall',fontName='Helvetica',fontSize=8,leading=10,spaceAfter=5))
    styles.add(ParagraphStyle(name='ReportCell',fontName='Helvetica',fontSize=8,leading=10))
    styles.add(ParagraphStyle(name='ReportHeader',fontName='Helvetica-Bold',fontSize=8,leading=10,textColor=colors.white))
    story=[];width=6.9*inch
    def paragraph(text, style='ReportBody'):
        story.append(Paragraph(escape(str(text)),styles[style]))
    def title(text):paragraph(text,'ReportSection')
    def newpage(text):story.append(PageBreak());title(text)
    def table(headers,rows,widths):
        def cell(x,header=False):return Paragraph(escape(str(x)),styles['ReportHeader' if header else 'ReportCell'])
        tab=Table([[cell(x,True) for x in headers]]+[[cell(x) for x in r] for r in rows],colWidths=[width*x for x in widths],repeatRows=1,hAlign='LEFT')
        tab.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),navy),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,light]),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),('LINEBELOW',(0,0),(-1,0),.8,teal)]))
        story.append(tab);story.append(Spacer(1,12))
    def metric(value):return f"{100*value['mean']:.2f} +/- {100*value['sd']:.2f}"
    def effect(c):
        precision=4 if c['study']=='equalization' and c['name'] in ('fixed_natural_EQ_effect','fixed_balanced_EQ_effect','sampling_effect_EQoff','fixed_factorial_interaction') else 2
        return f"{100*c['mean']:+.{precision}f} [{100*c['ci95'][0]:+.{precision}f}, {100*c['ci95'][1]:+.{precision}f}]"
    def pvalue(value):return f'{value:.6f}'
    def scores(study):
        table(['Method / readout','Macro-F1 (%)','Tail F1 (%)','Accuracy (%)'],[[r['label'],metric(r['metrics']['macro_f1']),metric(r['metrics']['tail_f1']),metric(r['metrics']['accuracy'])] for r in study['summary']],[.43,.19,.19,.19])
        paragraph('Entries are mean +/- sample SD over ten optimization seeds. Tail groups use the fixed fit-frequency rank. Readout aliases and differing target priors are disclosed in the study notes.','ReportSmall')
    def contrasts(name):
        rows=[r for r in e['global_analysis']['comparisons'] if r['study']==name]
        table(['Paired contrast','Effect and 95% CI (pp)','Exact p','Study Holm p'],[[r['label'],effect(r),pvalue(r['signflip_p']),pvalue(r['signflip_p_study_holm'])] for r in rows],[.45,.29,.13,.13])
        paragraph('Intervals use the paired-t rule and are not simultaneous confidence intervals. Exact p-values enumerate all 1,024 sign assignments. Holm adjustments apply to the declared primary family.','ReportSmall')
    def notes(study,kind):
        for text in study[kind]:paragraph(text)
    def footer(canvas,doc):
        canvas.saveState();canvas.setStrokeColor(light);canvas.line(42,39,570,39);canvas.setFont('Helvetica',8);canvas.setFillColor(navy);canvas.drawString(42,26,'REBAL | Stronger baselines and protected confirmation');canvas.drawRightString(570,26,str(doc.page));canvas.restoreState()
    paragraph('REBAL: stronger baselines and protected confirmation','ReportTitle')
    paragraph('Experimental report | '+e['completed_utc'][:10]+' | Ten paired optimization seeds','ReportSmall')
    paragraph(e['abstract'])
    for statement in e['principal_findings']:paragraph(statement)
    paragraph('Every result in this report comes from a completed fixed-budget run and independent numeric replay. Fresh confirmation partitions protect this round from its own selection, but historical models may have seen these source rows. Foundation-model pretraining overlap and source dependence remain limitations.')
    table(['Study','Full source','Current fit','Confirmation'],[[s['title'],f"{s['source_rows']:,}",f"{s['fit_rows']:,}",f"{s['confirmation_rows']:,}"] for s in e['studies'].values()],[.43,.19,.19,.19])
    newpage('Completed scope and evidence')
    table(['Study','Tuning fits','Final fitted models','Budget'],[[s['title'],s['tuning_fits'],s['final_fits'],s['budget_label']] for s in e['studies'].values()],[.34,.15,.18,.33])
    paragraph('EQ trains 320 tuning adapters over the ten paired seeds, then reuses its selected and reference adapters for confirmation. It separately trains 220 linear classifiers for 2,000 updates each. Vision, text and tabular selection use separate dedicated tuning seeds. Counts of named readouts do not represent independent fitted models.')
    notes(e,'verification_notes')
    newpage('Selection, confirmation and statistical inference')
    notes(e,'selection_notes')
    paragraph('The primary endpoint is confirmation macro-F1. Accuracy, tail F1, macro recall, class-level results and distribution diagnostics remain secondary. A one percentage-point macro-F1 effect is a practical reference; it does not establish a universal deployment threshold.')
    paragraph('Ten seeds quantify optimizer and initialization variation on fixed source partitions. Paired intervals do not include partition, entity, geographic or foundation-model pretraining uncertainty. A statistically detectable improvement can remain practically small. Non-significance does not prove equivalence.')
    paragraph('The four prespecified primary families contain 6 EQ, 5 vision, 8 text and 3 tabular comparisons. Per-study Holm correction is primary. A separately labeled global Holm sensitivity adjusts all 22 comparisons. These corrections remain valid without assuming independence among hypotheses.')
    for name in ('equalization','vision','text','tabular'):
        s=e['studies'][name]
        newpage(s['title']+' | methodology')
        notes(s,'method_notes')
        table(['Partition','Rows','Use'],s['partition_table'],[.25,.18,.57])
        notes(s,'selection_and_compute')
        newpage(s['title']+' | confirmation results')
        scores(s)
        notes(s,'result_notes')
        newpage(s['title']+' | paired comparisons')
        contrasts(name)
        notes(s,'contrast_notes')
        for extra in s.get('secondary_tables',[]):
            table(extra['headers'],extra['rows'],extra['widths'])
            paragraph(extra['caption'],'ReportSmall')
        if s.get('figure'):
            image_path=Path(e['_evidence_root'])/s['figure'];require(image_path.exists(),'Result figure missing')
            im=Image(str(image_path));im.drawHeight=min(3.2*inch,width*im.imageHeight/im.imageWidth);im.drawWidth=im.drawHeight*im.imageWidth/im.imageHeight;story.append(im)
    all_comparisons=e['global_analysis']['comparisons']
    for start in (0,11):
        newpage('Global multiplicity sensitivity | '+('1 of 2' if start==0 else '2 of 2'))
        table(['Study / contrast','Effect and 95% CI (pp)','Study Holm p','Global Holm p'],[[r['study']+' / '+r['label'],effect(r),pvalue(r['signflip_p_study_holm']),pvalue(r['signflip_p_global_holm'])] for r in all_comparisons[start:start+11]],[.43,.29,.14,.14])
        paragraph('Both adjusted p columns use the exact sign-flip test. Global correction is sensitivity analysis, not replacement of the prespecified study families. The paired-t p-values, all individual differences and raw p-values are retained in machine-readable evidence.','ReportSmall')
    newpage('Practical interpretation and limitations')
    notes(e,'recommendation_notes');notes(e,'limitations')
    newpage('Reproducibility and artifact identities')
    notes(e,'reproduction_notes')
    table(['Artifact role','SHA256'],[[r['role'],r['sha256']] for r in e['identity_table']],[.32,.68])
    paragraph('Dense inputs, pretrained feature caches, model/optimizer states, credentials and private operational logs are excluded from public evidence. Stored prediction records and split labels permit independent numeric replay; omitted training assets retain hash identities. The report does not claim to replay all encoder inference from the portable archive.')
    newpage('Primary method and dataset references')
    for r in e['references']:
        paragraph(r['author']+'. '+r['title']+'.','ReportBody')
        paragraph(r['url'],'ReportSmall')
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    doc=SimpleDocTemplate(str(output),pagesize=(612,792),leftMargin=42,rightMargin=42,topMargin=42,bottomMargin=51,title='REBAL: stronger baselines and protected confirmation',author='REBAL experimental research',subject='Audited multimodal long-tail classification experiments')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)

def main(args):
    evidence=load_evidence(args.evidence)
    if args.check_only:
        print(json.dumps({'status':'all_evidence_gates_passed','studies':4,'comparisons':22}))
        return
    evidence['_evidence_root']=str(Path(args.evidence).parent)
    create_pdf(evidence,args.out)
    print(json.dumps({'status':'created_pending_render_and_numeric_QA','pdf_sha256':sha(args.out)}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--evidence',required=True);p.add_argument('--out',required=True);p.add_argument('--check-only',action='store_true');main(p.parse_args())
