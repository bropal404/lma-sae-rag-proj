"""Regenerate the final-report vector plots from the team's committed CSV exports.
Usage: python generate_figures.py --results-root /path/to/repository
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
p=argparse.ArgumentParser();p.add_argument('--results-root',type=Path,default=Path(__file__).resolve().parents[2]);p.add_argument('--out',type=Path,default=Path(__file__).parent/'figures'); a=p.parse_args();R=a.results_root;O=a.out;O.mkdir(parents=True,exist_ok=True)
NAVY='#17324D';TEAL='#008E82';BLUE='#3974A3';ORANGE='#D8754C';SILVER='#7B8B9A';LIGHT='#EBF0F3';CHARCOAL='#2C3F4A';PURPLE='#8D6DA4'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.2,'axes.titlesize':9.3,'axes.labelsize':8.2,'xtick.labelsize':7.8,'ytick.labelsize':7.8,'axes.edgecolor':'#CBD6DE','axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','text.color':NAVY,'axes.labelcolor':NAVY,'xtick.color':CHARCOAL,'ytick.color':CHARCOAL,'pdf.fonttype':42,'savefig.facecolor':'white','figure.facecolor':'white','axes.facecolor':'white','grid.color':'#DFE8ED','axes.axisbelow':True})
def save(fig,name):fig.savefig(O/name,format='pdf',bbox_inches='tight',pad_inches=.09);plt.close(fig)
# 1. pooled accuracy / retrieval-call Pareto scatter
d=pd.read_csv(R/'results/final/gate2_aggregate.csv')
names={'b1_always_retrieve':'Always retrieve','b2_never_retrieve':'Never retrieve','b4_verbalised':'Verbalised','b8_tfidf_lr':'TF-IDF','gemma3-4b-it__miss_as_retrieve__sae16k_last@L29__l2':'SAE L29','gemma3-4b-it__miss_as_retrieve__resid_last@L22__l2':'Dense L22'}
colors={'Always retrieve':NAVY,'Never retrieve':SILVER,'Verbalised':PURPLE,'TF-IDF':ORANGE,'SAE L29':TEAL,'Dense L22':BLUE}
fig,ax=plt.subplots(figsize=(4.25,2.55));ax.axhline(float(d.loc[d.router=='b1_always_retrieve','accuracy'].iloc[0])*100-2,c='#BFCBD4',ls='--',lw=1,zorder=0);ax.axhline(float(d.loc[d.router=='b1_always_retrieve','accuracy'].iloc[0])*100,c='#BFCBD4',ls=':',lw=1,zorder=0)
shifts={'Never retrieve':(4,-13),'Always retrieve':(-45,-16),'Verbalised':(-51,-16),'TF-IDF':(-14,-16),'SAE L29':(-42,8),'Dense L22':(-38,8)}
for _,r in d.iterrows():
    label=names.get(r.router)
    if not label:continue
    x,y=r.retrieval_call_rate*100,r.accuracy*100
    ax.scatter(x,y,s=100 if label=='SAE L29' else 64,c=colors[label],edgecolors='white',linewidth=1.05,zorder=4)
    dx,dy=shifts[label];ax.annotate(label,(x,y),xytext=(dx,dy),textcoords='offset points',fontsize=7.5,fontweight='bold' if label=='SAE L29' else 'normal',color=colors[label])
ax.set(xlim=(-5,115),ylim=(24,36),xlabel='BM25 retrieval calls (% of questions)',ylabel='End-to-end answer accuracy (%)');ax.xaxis.set_major_locator(MultipleLocator(20));ax.yaxis.set_major_locator(MultipleLocator(2));ax.grid(axis='y',lw=.65);save(fig,'accuracy_retrieval.pdf')
# 2. accuracy per dataset
d=pd.read_csv(R/'results/final/gate2_by_dataset.csv');ds=['nq_test','triviaqa_test','popqa_test','hotpotqa_test'];base=['NQ','TriviaQA','PopQA','HotpotQA'];models=['b1_always_retrieve','gemma3-4b-it__miss_as_retrieve__sae16k_last@L29__l2','gemma3-4b-it__miss_as_retrieve__resid_last@L22__l2'];clrs=[NAVY,TEAL,BLUE];leg=['Always retrieve','SAE L29','Dense L22'];fig,ax=plt.subplots(figsize=(6.9,2.3));x=np.arange(4);w=.24
for j,m in enumerate(models):
    values=[float(d[(d.dataset==ds_) & (d.router==m)].accuracy.iloc[0])*100 for ds_ in ds]
    bars=ax.bar(x+(j-1)*w,values,width=w,color=clrs[j],label=leg[j],zorder=4)
    for b,val in zip(bars,values):ax.annotate(f'{val:.1f}',(b.get_x()+b.get_width()/2,b.get_height()),xytext=(0,3),textcoords='offset points',ha='center',fontsize=7,color=CHARCOAL)
ax.set(xticks=x,xticklabels=base,ylim=(0,62),ylabel='Answer accuracy (%)');ax.grid(axis='y',lw=.7);ax.legend(frameon=False,ncol=3,loc='upper right',fontsize=7.8);save(fig,'dataset_accuracy.pdf')
# 3. model choice validation curves
d=pd.read_csv(R/'results/final/model_selection.csv');fig,ax=plt.subplots(figsize=(4.0,2.3));layers=[9,17,22,29]
for pre,label,c,ls in [('sae16k_last@L','SAE (16k)',TEAL,'-'),('resid_last@L','Dense',BLUE,'-')]:
    ys=[float(d.loc[d.model==f'{pre}{L}','validation_macro_f1'].iloc[0])*100 for L in layers]
    ax.plot(layers,ys,marker='o',markersize=5,lw=2,c=c,label=label)
    # Keep labels on the *outer* sides of the two curves at each layer.
    # The upper/lower series swap at L29; fixed SAE-above/dense-below
    # annotations caused the L17 and L22 values to overlap.
    for z,y in zip(layers,ys):
        upper = (pre.startswith('sae') and z == 29) or (not pre.startswith('sae') and z != 29)
        offset_y = 10 if upper else -13
        ax.annotate(f'{y:.1f}', (z,y), xytext=(0,offset_y),
                    textcoords='offset points', ha='center', va='center',
                    fontsize=7, color=c,
                    bbox=dict(boxstyle='square,pad=0.1', facecolor='white',
                              edgecolor='none', alpha=0.85))
ax.set(xticks=layers,xlim=(7,31),ylim=(37,57),xlabel='Residual-stream layer',ylabel='Validation macro-F1 (%)');ax.grid(axis='y',lw=.7);ax.legend(frameon=False,loc='lower right',fontsize=7.7);save(fig,'layer_validation.pdf')
# 4 label budgets
D=pd.read_csv(R/'results/final/h2_label_efficiency.csv');ds=['nq_test','triviaqa_test','popqa_test','hotpotqa_test'];fig,ax=plt.subplots(figsize=(4.0,2.25));budgets=['50','100','200','500','1000','2000','5000','full']
for model,label,color in [('sae16k_last@L29','SAE L29',TEAL),('resid_last@L22','Dense L22',BLUE)]:
    sub=D[(D.model==model)&(D['eval'].isin(ds))].groupby('budget').macro_f1.mean()
    vals=[sub[b]*100 for b in budgets];ax.plot(range(len(budgets)),vals,marker='o',lw=2,c=color,label=label,markersize=4.2)
ax.set(xticks=range(len(budgets)),xticklabels=['50','100','200','500','1k','2k','5k','8k'],ylim=(34,52),xlabel='Training-label budget',ylabel='Mean test macro-F1 (%)');ax.grid(axis='y',lw=.7);ax.legend(frameon=False,loc='lower right',fontsize=7.5);save(fig,'label_budget.pdf')
# 5. hole risk
H=pd.read_csv(R/'results/final/h3_hole_risk.csv');z=H.groupby('router').apply(lambda g:pd.Series({'wrong':(g.n*g.wrong_per_hole_question).sum()/g.n.sum()*100,'cover':(g.n*g.coverage).sum()/g.n.sum()*100}),include_groups=False)
keys=[v for v in z.index if '__2way' not in v]+[v for v in z.index if '__2way' in v]
fig,ax=plt.subplots(figsize=(4.0,2.25));xx=np.arange(2);ww=.31
for j,(field,label,color) in enumerate([('wrong','Wrong / all holes',ORANGE),('cover','Answer coverage',TEAL)]):
    vals=[z.loc[k,field] for k in keys];bars=ax.bar(xx+(j-.5)*ww,vals,width=ww,color=color,label=label)
    for b,v in zip(bars,vals):ax.text(b.get_x()+b.get_width()/2,v+1.5,f'{v:.1f}%',ha='center',fontsize=7.7,color=CHARCOAL)
ax.set(xticks=xx,xticklabels=['Three-way SAE','Two-way SAE'],ylim=(0,113),ylabel='Share of hole questions (%)');ax.grid(axis='y',lw=.7);ax.legend(loc='upper left',frameon=False,fontsize=7.1);save(fig,'hole_risk.pdf')
# 6 latency
L=pd.read_csv(R/'results/sagemaker/latency/routed_l29.csv');mapping=[('b1_always_retrieve','Always retrieve'),('gemma3-4b-it__miss_as_retrieve__sae16k_last@L29__l2','SAE L29'),('gemma3-4b-it__miss_as_retrieve__resid_last@L22__l2','Dense L22'),('b8_tfidf_lr','TF-IDF'),('b4_verbalised','Verbalised')]
fig,ax=plt.subplots(figsize=(6.9,2.15));y=np.arange(len(mapping));w=.29
p50=[float(L[L.router==r].p50_ms.iloc[0]) for r,_ in mapping];p95=[float(L[L.router==r].p95_ms.iloc[0]) for r,_ in mapping]
ax.barh(y+w/2,p50,height=w,color=TEAL,label='p50');ax.barh(y-w/2,p95,height=w,color=BLUE,alpha=.78,label='p95')
ax.set(yticks=y,yticklabels=[t for _,t in mapping],xlabel='Composed response latency (ms)',xlim=(0,830));ax.invert_yaxis();ax.grid(axis='x',lw=.65);ax.legend(frameon=False,ncol=2,loc='lower right',fontsize=8)
for yy,v in zip(y,p50):ax.text(v+5,yy+w/2,f'{v:.0f}',va='center',fontsize=7.3,color=TEAL)
for yy,v in zip(y,p95):ax.text(v+5,yy-w/2,f'{v:.0f}',va='center',fontsize=7.3,color=BLUE)
save(fig,'latency.pdf')
# 7 ablation (test macro F1) from separate summaries at SAE L22.
base_glob=list((R/'results/sagemaker/probes/gemma3-4b-it/miss_as_retrieve').rglob('summary.csv'));categories=[('Standard L2','__l2/summary.csv'),('L1 penalty','__l1__l1/summary.csv'),('Binarised SAE','__l2_bin__binarized/summary.csv'),('Mean pooling','__l2__mean_pool/summary.csv'),('262k-wide SAE','__l2__wide/summary.csv')];ab=[]
for label,suf in categories:
    matches=[f for f in base_glob if str(f).endswith(suf)];assert len(matches)==1,(label,matches)
    df=pd.read_csv(matches[0]);m='sae262k_last@L22' if '262k' in label else 'sae16k_mean@L22' if 'Mean' in label else 'sae16k_last@L22';z=df[(df.model==m)&(df.budget=='full')&df['eval'].isin(ds)].macro_f1.mean();ab.append((label,z*100))
fig,ax=plt.subplots(figsize=(4.0,2.3));y=np.arange(len(ab));vals=[v for _,v in ab];bars=ax.barh(y,vals,height=.58,color=[TEAL]+[SILVER]*4);ax.invert_yaxis();ax.set(yticks=y,yticklabels=[u for u,_ in ab],xlim=(0,59),xlabel='Mean test macro-F1 (%)');ax.grid(axis='x',lw=.65)
for b,v in zip(bars,vals):ax.text(v+1,b.get_y()+b.get_height()/2,f'{v:.1f}',va='center',fontsize=8,color=CHARCOAL)
save(fig,'ablation.pdf')
# 8 top feature removal validation L22
top=list((R/'results/sagemaker/probes/gemma3-4b-it/miss_as_retrieve').glob('analysis__sae16k_last@L22/drop_top_k.csv'))[0];D=pd.read_csv(top);fig,ax=plt.subplots(figsize=(4.0,2.15));ax.plot(D.k,D.f1_zeroed*100,c=ORANGE,marker='o',lw=1.8,label='Zero coefficients');ax.plot(D.k,D.f1_retrained*100,c=TEAL,marker='s',lw=1.8,label='Retrain without features');ax.set(xlim=(-2,103),ylim=(13,58),xticks=[0,5,10,25,50,100],xlabel='Top-k SAE features removed per class',ylabel='Probe macro-F1 (%)');ax.grid(axis='y',lw=.7);ax.legend(frameon=False,fontsize=7.2,loc='upper right');save(fig,'feature_removal.pdf')
print('WROTE',len(list(O.glob('*.pdf'))),'figures',[(x.name,x.stat().st_size) for x in O.glob('*.pdf')])
# 9. compact per-dataset accuracy for a one-column float
fig,ax=plt.subplots(figsize=(3.8,2.25));xx=np.arange(4);ww=.24
for j,m in enumerate(models):
    D2=pd.read_csv(R/'results/final/gate2_by_dataset.csv')
    vs=[float(D2[(D2.dataset==ds_) & (D2.router==m)].accuracy.iloc[0])*100 for ds_ in ds]
    ax.bar(xx+(j-1)*ww,vs,width=ww,color=clrs[j],label=leg[j]);
ax.set(xticks=xx,xticklabels=base,ylim=(0,62),ylabel='Accuracy (%)');ax.grid(axis='y',lw=.6);ax.legend(frameon=False,ncol=3,fontsize=6.8,loc='upper center');save(fig,'dataset_accuracy_compact.pdf')
# 10. compact latency for a one-column float
fig,ax=plt.subplots(figsize=(3.85,2.35));pos=np.arange(len(mapping));w=.31
ax.barh(pos+w/2,p50,height=w,color=TEAL,label='Median');ax.barh(pos-w/2,p95,height=w,color=BLUE,alpha=.77,label='p95');ax.set(yticks=pos,yticklabels=['Always BM25','SAE L29','Dense L22','TF-IDF','Verbalised'],xlabel='Composed latency (ms)',xlim=(0,805));ax.invert_yaxis();ax.grid(axis='x',lw=.6);ax.legend(frameon=False,ncol=2,fontsize=7.1,loc='lower right');save(fig,'latency_compact.pdf')
