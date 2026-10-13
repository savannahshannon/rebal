"""Render scientific figures from the completed, audited report evidence."""
from pathlib import Path
import argparse
import importlib.util
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

spec=importlib.util.spec_from_file_location('report_gate',Path(__file__).resolve().parents[1]/'build_followup_report.py')
gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)
PALETTE={'equalization':'#355A81','vision':'#007C83','text':'#8770A6','tabular':'#A97400'}

def main(args):
    e=gate.load_evidence(args.evidence);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
    comparisons=e['global_analysis']['comparisons'];fig,ax=plt.subplots(figsize=(12,10.5))
    for i,r in enumerate(comparisons):
        mean=100*r['mean'];low,high=np.asarray(r['ci95'])*100;color=PALETTE[r['study']]
        ax.errorbar(mean,len(comparisons)-1-i,xerr=[[mean-low],[high-mean]],fmt='o',color=color,capsize=3,markersize=5)
    ax.axvline(0,color='#53616E',lw=1);ax.set_yticks(np.arange(len(comparisons)));ax.set_yticklabels([r['study'].capitalize()+': '+r['label'] for r in comparisons][::-1]);ax.set_xlabel('Confirmation macro-F1 difference (percentage points)');ax.set_title('Prespecified paired effects across four studies',loc='left',weight='bold');ax.grid(axis='x',alpha=.18)
    fig.text(.02,.012,'95% paired-t intervals; ten optimizer seeds on fixed source partitions. Effects do not rank architectures across datasets.',fontsize=9,color='#53616E');fig.tight_layout(rect=(0,.035,1,1));fig.savefig(out/'rebal_followup_paired_effects.png',dpi=220);plt.close(fig)
    for name,study in e['studies'].items():
        rows=study['summary'];positions=np.arange(len(rows));fig,axes=plt.subplots(1,2,figsize=(10,max(4.5,.28*len(rows)+1)),sharey=True)
        for ax,key,title in zip(axes,['macro_f1','tail_f1'],['Macro-F1','Tail F1']):
            mean=np.array([r['metrics'][key]['mean'] for r in rows])*100;sd=np.array([r['metrics'][key]['sd'] for r in rows])*100
            ax.barh(positions,mean,xerr=sd,color=PALETTE[name],alpha=.83,error_kw={'capsize':2,'lw':.8});ax.set_xlabel(title+' (%)');ax.set_xlim(0,100);ax.grid(axis='x',alpha=.16);ax.set_axisbelow(True)
        axes[0].set_yticks(positions);axes[0].set_yticklabels([r['label'] for r in rows]);axes[0].invert_yaxis();fig.suptitle(study['title']+' | confirmation',weight='bold');fig.text(.02,.015,'Mean +/- sample SD over ten optimizer seeds; tail groups fixed by fit rank. Named readouts can share fitted models.',fontsize=9.5,color='#53616E');fig.tight_layout(rect=(0,.04,1,.96));fig.savefig(out/('rebal_followup_'+name+'_performance.png'),dpi=220);plt.close(fig)
    print(json.dumps({'status':'rendered_from_complete_verified_evidence','figures':5}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--evidence',required=True);p.add_argument('--out',required=True);main(p.parse_args())
