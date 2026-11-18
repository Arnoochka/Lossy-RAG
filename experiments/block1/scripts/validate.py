#!/usr/bin/env python3
"""Validate all published Block-1 CSV outputs independently of models and caches.
Run from any cwd: python experiments/block1/scripts/validate.py
"""
from pathlib import Path
import sys, json, math, hashlib
import numpy as np
import pandas as pd
from scipy.stats import binomtest
HERE=Path(__file__).resolve().parents[1]
MAIN=HERE/'results/main'
PARA=HERE/'results/paraphrase'
OUT=HERE/'local_checks'
OUT.mkdir(exist_ok=True)
checks=[]
failures=[]
def check(ok, message, detail=None):
    record={'ok':bool(ok),'test':message}
    if detail is not None:record['detail']=detail
    checks.append(record)
    if not ok:failures.append(message)

def near(a,b,atol=1e-10):
    return np.isclose(float(a),float(b),atol=atol,rtol=0)

def asbool(s):
    return s.astype(str).str.lower().isin(['true','1'])

try:
    main=pd.read_csv(MAIN/'all_methods_summary.csv')
    selected=pd.read_csv(MAIN/'selected_train_questions.csv')
    check(len(selected)==500 and selected.qid.is_unique,'main: 500 unique train questions')
    check((selected.role=='train').all() and not asbool(selected.is_paraphrase).any(),'main: train original questions only')
    check(main.condition.nunique()==32,'main: 32 configurations',int(main.condition.nunique()))
    check(len(main)==32*2*4,'main: 256 summary rows')
    check(set(main.k)=={1,3,5,10},'main: k=1,3,5,10')
    check(main.doc_hash.nunique()==1 and main.query_hash.nunique()==1,'main: one corpus+question fingerprint')
    check(set(main.n_questions)=={11,500},'main: 500 original and 11 pilot paraphrases')
    check(main.strict_hit.le(main.soft_hit+1e-12).all(),'main: strict <= soft at every summary entry')
    check(not main.duplicated(['condition','is_paraphrase','k']).any(),'main: unique condition/role/k summary')
    for _,group in main.groupby(['condition','is_paraphrase']):
        arr=group.sort_values('k')
        check((arr.strict_hit.diff().fillna(0)>=-1e-10).all() and (arr.soft_hit.diff().fillna(0)>=-1e-10).all(),f"main: monotonic k {arr.condition.iloc[0]} / {arr.is_paraphrase.iloc[0]}")
    all_detail=[]
    for name in sorted(main.condition.unique()):
        path=MAIN/'by_question'/f'{name}_by_question.csv'
        if not path.is_file():check(False,'missing detail: '+str(path));continue
        d=pd.read_csv(path)
        check(len(d)==(500+11)*4,f'detail: {name} has 511*4 rows')
        check(d[['doc_hash','query_hash']].nunique().max()==1,f'detail: {name} fixed fingerprint')
        check((d.strict_hit<=d.soft_hit).all(),f'detail: {name} strict <= soft')
        check(d.loc[~asbool(d.is_paraphrase),'qid'].nunique()==500,f'detail: {name} 500 original qids')
        check(set(d.loc[~asbool(d.is_paraphrase),'qid'].unique())==set(selected.qid),f'detail: {name} same original qids')
        sums=(d.groupby(['is_paraphrase','k'],as_index=False)
              .agg(strict_hit=('strict_hit','mean'),soft_hit=('soft_hit','mean'),n_questions=('qid','size')))
        sums['is_paraphrase']=asbool(sums.is_paraphrase)
        for row in sums.itertuples(index=False):
            m=main[(main.condition==name) & (asbool(main.is_paraphrase)==row.is_paraphrase) & (main.k==row.k)]
            check(len(m)==1 and near(m.strict_hit.iloc[0],row.strict_hit) and near(m.soft_hit.iloc[0],row.soft_hit) and int(m.n_questions.iloc[0])==int(row.n_questions), f'metrics: {name}, para={row.is_paraphrase}, k={row.k}')
        all_detail.append(d)
    detail=pd.concat(all_detail,ignore_index=True)
    # Nomic paired wins and McNemar - same questions, seed42 PCA
    comp=pd.read_csv(MAIN/'nomic_mrl_vs_pca_same_dim.csv')
    for x in comp.itertuples(index=False):
        m=detail[(detail.condition==f'nomic_matryoshka_{int(x.dimension)}') & (~asbool(detail.is_paraphrase)) & (detail.k==5)].set_index('qid').sort_index()
        p=detail[(detail.condition==f'nomic_pca_{int(x.dimension)}_seed42') & (~asbool(detail.is_paraphrase)) & (detail.k==5)].set_index('qid').sort_index()
        check(m.index.equals(p.index) and len(m)==500,f'nomic: paired ids dim={x.dimension}')
        a=m.strict_hit.astype(int);c=p.strict_hit.astype(int)
        wins=int(((a==1)&(c==0)).sum());losses=int(((a==0)&(c==1)).sum())
        pv=binomtest(wins,wins+losses,0.5).pvalue if wins+losses else 1.
        check(wins==x.mrl_wins and losses==x.pca_wins and near(pv,x.mcnemar_exact_p) and near(a.mean(),x.matryoshka_hit5) and near(c.mean(),x.pca_hit5_seed42),f'nomic: MRL/PCA dim={x.dimension} independently matches')
    # Paired comparisons vs BGE small
    ps=pd.read_csv(MAIN/'paired_vs_bge_small.csv')
    base=detail[(detail.condition=='bge_small_full') & (~asbool(detail.is_paraphrase)) & (detail.k==5)].set_index('qid').strict_hit
    for x in ps.itertuples(index=False):
        other=detail[(detail.condition==x.condition)&(~asbool(detail.is_paraphrase))&(detail.k==5)].set_index('qid').strict_hit
        if not other.index.equals(base.index): other=other.reindex(base.index)
        gains=int(((base==0)&(other==1)).sum()); losses=int(((base==1)&(other==0)).sum())
        pv=binomtest(gains,gains+losses,.5).pvalue if (gains+losses)>0 else 1.
        check(near(other.mean()-base.mean(),x.delta_hit5) and gains==x.gains and losses==x.losses and near(pv,x.mcnemar_exact_p),f'paired vs Small: {x.condition}')
    # Paraphrase validations
    pairs=pd.read_csv(PARA/'selected_500_pairs.csv')
    detail_p=pd.read_csv(PARA/'paraphrase_by_pair.csv')
    sums=pd.read_csv(PARA/'paraphrase_summary.csv')
    check(len(pairs)==500 and pairs.orig_qid.is_unique and pairs.paraphrase_qid.is_unique,'paraphrase: 500 distinct original and paraphrased question IDs')
    check((pairs.original_question!=pairs.paraphrase_question).all(),'paraphrase: source and rewritten text differ')
    check(len(detail_p)==500*3*4,'paraphrase: 6000 per-pair per-k evaluations')
    check(set(detail_p.model)=={'bge_small','bge_base','bge_large'},'paraphrase: all 3 BGE models')
    check(set(detail_p.k)=={1,3,5,10},'paraphrase: k=1,3,5,10')
    check(detail_p.run_hash.nunique()==1,'paraphrase: common run fingerprint')
    check(set(detail_p.orig_qid)==set(pairs.orig_qid),'paraphrase: same 500 original ids in details')
    check(detail_p.strict_original.le(detail_p.soft_original).all() and detail_p.strict_paraphrase.le(detail_p.soft_paraphrase).all(),'paraphrase: strict <= soft')
    for (model,k),d in detail_p.groupby(['model','k']):
        s=sums[(sums.model==model)&(sums.k==k)]
        if len(s)!=1:check(False,f'paraphrase: summary row missing {model} k={k}');continue
        s=s.iloc[0]
        ao=d.strict_original.astype(int);ap=d.strict_paraphrase.astype(int)
        loss=int(((ao==1)&(ap==0)).sum());gain=int(((ao==0)&(ap==1)).sum())
        p=binomtest(loss,loss+gain,.5).pvalue if loss+gain else 1.
        check(near(s.strict_original,ao.mean()) and near(s.strict_paraphrase,ap.mean()) and near(s.soft_original,d.soft_original.mean()) and near(s.soft_paraphrase,d.soft_paraphrase.mean()) and loss==s.strict_lost_after_paraphrase and gain==s.strict_gained_after_paraphrase and near(p,s.mcnemar_exact_p),f'paraphrase: {model} k={k} agrees with details')
    # Inspect committed repository source paraphrases; no internet required
    repo=HERE.parents[1]
    upstream=repo/'data/questions_paraphrased.csv'
    if upstream.exists():
        u=pd.read_csv(upstream).set_index('qid')
        check(set(pairs.paraphrase_qid).issubset(set(u.index)),'paraphrase: all 500 qids exist in repository snapshot')
        aligned=u.reindex(pairs.paraphrase_qid)
        check((aligned.orig_qid.to_numpy()==pairs.orig_qid.to_numpy()).all() and (aligned.question.to_numpy()==pairs.paraphrase_question.to_numpy()).all(),'paraphrase: qid/orig_qid/text match repository snapshot')
    else:check(False,'repository questions_paraphrased.csv not present')
    # Verify topk arrays still match 1000 queries and are IDs not positions
    for model in ['bge_small','bge_base','bge_large']:
        idx=np.load(PARA/'rankings'/f'{model}_topk_ids.npy',mmap_mode='r')
        check(idx.shape==(1000,10),f'paraphrase: {model} topk shape 1000x10')
        check(np.all(idx>=0),f'paraphrase: {model} nonnegative chunk ids')
        check(np.all([len(set(row.tolist()))==10 for row in idx]),f'paraphrase: {model} no duplicate chunks in top10')
    # Save summaries derived from source
    core=main[(~asbool(main.is_paraphrase))&(main.k==5)].copy()
    core=core[['condition','dimension','strict_hit','soft_hit','n_questions']].sort_values('condition')
    core.to_csv(OUT / 'main_strict_soft_hit5.csv',index=False)
    pca=core[core.condition.str.contains('_pca_')].copy()
    pca['model']=pca.condition.str.extract(r'^(bge_small|nomic)_pca_')[0]
    stab=pca.groupby(['model','dimension'],as_index=False).agg(hit5_mean=('strict_hit','mean'),hit5_min=('strict_hit','min'),hit5_max=('strict_hit','max'),n_seeds=('strict_hit','size'))
    stab.to_csv(OUT / 'pca_stability_hit5.csv',index=False)
    status='PASS' if not failures else 'FAIL'
except Exception as exc:
    import traceback
    check(False,'Unexpected validation exception',traceback.format_exc())
    status='FAIL'
report={'status':status,'checks_total':len(checks),'checks_passed':sum(c['ok'] for c in checks),'failures':failures,
        'completed_configurations':32,'main_original_questions':500,'main_pilot_paraphrases':11,
        'paraphrase_pairs':500, 'scope':'CSV cross-checks and available paraphrase topk; does not authenticate missing document vectors or human equivalence of paraphrases'}
(OUT/'validation_summary.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
(OUT/'validation_details.json').write_text(json.dumps(checks,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps(report,indent=2,ensure_ascii=False))
sys.exit(0 if not failures else 1)
