#!/usr/bin/env python3
"""Recreate Block-1 research figures from exported tables (no model downloads)."""
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
MAIN=pd.read_csv(ROOT/'results/main/all_methods_summary.csv')
PARA=pd.read_csv(ROOT/'results/paraphrase/paraphrase_summary.csv')
import argparse

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--out',type=Path,default=ROOT/'local_checks/figures')
OUT=parser.parse_args().out
OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.size':11,'figure.dpi':150})
def save(name):
    plt.tight_layout();plt.savefig(OUT/name,dpi=180,bbox_inches='tight');plt.close()

# BGE strict & soft (same selection of 500 original questions)
for metric in ('strict_hit','soft_hit'):
    fig,ax=plt.subplots(figsize=(8.1,4.8))
    for label,c in [('Small','bge_small_full'),('Base','bge_base_full'),('Large','bge_large_full')]:
        a=MAIN[(MAIN.condition==c)&(MAIN.is_paraphrase==False)].sort_values('k')
        ax.plot(a.k,100*a[metric],marker='o',label=f'BGE {label}')
    ax.set(title=f'BGE models: {metric.replace("_"," ").title()} on SQuAD train (n=500)',xlabel='k',ylabel='Hit@k, %',xticks=[1,3,5,10],ylim=(50,100));ax.grid(alpha=.22);ax.legend()
    save(f'01_bge_models_{metric}.png')

base=MAIN[(MAIN.condition=='bge_small_full')&(MAIN.is_paraphrase==False)&(MAIN.k==5)].iloc[0]
fig,ax=plt.subplots(figsize=(8.1,4.8))
series=[]
for d in [64,128,192,256]:
    a=MAIN[(MAIN.condition.str.match(f'bge_small_pca_{d}_seed(42|43|44)$'))&(MAIN.is_paraphrase==False)&(MAIN.k==5)]
    series.append([d,100*a.strict_hit.mean(),100*a.strict_hit.min(),100*a.strict_hit.max()])
series=np.array(series)
ax.errorbar(series[:,0],series[:,1],yerr=[series[:,1]-series[:,2],series[:,3]-series[:,1]],fmt='o-',capsize=4,label='PCA mean and min–max (3 seeds)')
ax.axhline(100*base.strict_hit,ls='--',label='BGE Small, full 384d')
ax.set(title='BGE Small: PCA vs full embeddings (strict Hit@5)',xlabel='Dimensions',ylabel='Hit@5, %',xticks=[64,128,192,256,384],ylim=(50,90));ax.legend();ax.grid(alpha=.22)
save('02_bge_pca_stability.png')

comp=pd.read_csv(ROOT/'results/main/nomic_mrl_vs_pca_same_dim.csv')
fig,ax=plt.subplots(figsize=(8.1,4.8))
for name,col in [('Matryoshka','matryoshka_hit5'),('PCA (seed 42)','pca_hit5_seed42')]:
    ax.plot(comp.dimension,100*comp[col],marker='o',label=name)
full=MAIN[(MAIN.condition=='nomic_full') & (MAIN.is_paraphrase==False)&(MAIN.k==5)].iloc[0]
ax.axhline(100*full.strict_hit,ls='--',label='Nomic full 768d')
ax.set(title='Same Nomic embeddings: Matryoshka vs PCA',xlabel='Dimensions',ylabel='Strict Hit@5, %',xticks=[64,128,256,512,768],ylim=(55,90));ax.legend();ax.grid(alpha=.22)
save('03_nomic_mrl_vs_pca.png')

p5=PARA[PARA.k==5].copy();mapping={'bge_small':'Small','bge_base':'Base','bge_large':'Large'}
p5['model']=pd.Categorical(p5.model,categories=['bge_small','bge_base','bge_large'],ordered=True)
p5=p5.sort_values('model');xs=np.arange(3)
fig,ax=plt.subplots(figsize=(8,4.8))
ax.bar(xs-.19,100*p5.strict_original,.36,label='Original')
ax.bar(xs+.19,100*p5.strict_paraphrase,.36,label='Paraphrased')
ax.set(title='Same 500 pairs: change in strict Hit@5',ylabel='Hit@5, %',xticks=xs,xticklabels=['BGE '+mapping[str(x)] for x in p5.model],ylim=(0,100));ax.legend();ax.grid(axis='y',alpha=.2)
save('04_paraphrase_hit5.png')

fig,ax=plt.subplots(figsize=(8,4.8))
ax.bar(xs-.18,p5.strict_lost_after_paraphrase,.36,label='Lost successes')
ax.bar(xs+.18,p5.strict_gained_after_paraphrase,.36,label='Gained successes')
ax.set(title='Paired response changes after paraphrasing (n=500)',ylabel='Number of questions',xticks=xs,xticklabels=['BGE '+mapping[str(x)] for x in p5.model]);ax.legend();ax.grid(axis='y',alpha=.2)
save('05_paraphrase_lost_gained.png')
print('Saved figures to',OUT)
