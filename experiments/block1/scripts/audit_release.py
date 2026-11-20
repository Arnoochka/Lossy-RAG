#!/usr/bin/env python3
"""Read-only audit of code, results, corpus and the supplied BGE Small vectors.
Default: full audit. --tables-only: no dataset/model/vector download required.
Outputs go to ignored local_checks/; archived reports and results stay immutable.
"""
import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from retrieval import exact_topk, signature

BLOCK = Path(__file__).resolve().parents[1]
ROOT = BLOCK.parents[1]
EXPECTED_DOC = '15704f7b9c5a2a21f5484613e837786e00afc7d596014eb82832dfb0732fb4af'
EXPECTED_QUERY = 'e7fe7426f54dc7f2c41f53f54fbf3376dbc40305f0052070f7f65b0f331e6d3b'
checks = []
def check(ok, label, detail=None):
    record = {'ok': bool(ok), 'test': label}
    if detail is not None: record['detail'] = detail
    checks.append(record)
    if not ok: print('FAIL:', label, flush=True)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()

def boolean(column):
    strings = column.astype(str).str.lower()
    if not strings.isin(['true','false','0','1']).all():
        raise ValueError('Invalid boolean values')
    return strings.isin(['true','1'])

def holm(p):
    p = np.asarray(p, dtype=float); order = np.argsort(p)
    out = np.empty_like(p)
    out[order] = np.minimum(1, np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p)))))
    return out

def check_frame(a, b, cols, label):
    check(a[cols].reset_index(drop=True).equals(b[cols].reset_index(drop=True)), label)

def code_and_tables():
    import nbformat
    for p in sorted(ROOT.rglob('*.py')):
        if any(x in p.parts for x in ('.venv','__pycache__','.git')): continue
        compile(p.read_text(encoding='utf-8'),str(p),'exec')
    nbs = list(ROOT.rglob('*.ipynb'))
    for p in nbs:
        nb = nbformat.read(p, as_version=4); nbformat.validate(nb)
        for i,cell in enumerate(nb.cells):
            if cell.cell_type != 'code': continue
            # Upstream cells can have IPython magics; validate these with IPython.
            src = cell.source
            if re.search(r'^\s*[%!]',src,re.MULTILINE):
                from IPython.core.inputtransformer2 import TransformerManager
                src = TransformerManager().transform_cell(src)
            compile(src,f'{p.name}:cell{i}','exec')
    check(True, 'All Python files and notebook code cells compile', {'notebooks':len(nbs)})
    run = subprocess.run([sys.executable,str(BLOCK/'scripts/validate.py')],capture_output=True,text=True)
    check(run.returncode == 0, 'Original 559 result checks pass',run.stdout.strip())
    selected = pd.read_csv(BLOCK/'results/main/selected_train_questions.csv')
    pilot = pd.read_csv(BLOCK/'results/main/selected_paraphrases.csv')
    combined = pd.concat([selected,pilot],ignore_index=True)
    check(signature(combined.qid,combined.question)==EXPECTED_QUERY,'Frozen 511-query fingerprint')
    summary = pd.read_csv(BLOCK/'results/main/all_methods_summary.csv')
    check(summary.doc_hash.eq(EXPECTED_DOC).all() and summary.query_hash.eq(EXPECTED_QUERY).all(),'Main fingerprints match the actual frozen inputs')
    for name in summary.condition.unique():
        d = pd.read_csv(BLOCK/'results/main/by_question'/f'{name}_by_question.csv')
        check(not d.duplicated(['qid','k']).any(),'No duplicate question/k: '+name)
        check(d.qid.value_counts().eq(4).all() and set(d.qid)==set(combined.qid),'Complete question/k grid: '+name)
        check(d[['strict_hit','soft_hit']].isin([0,1]).all().all(),'Binary hits: '+name)
        check(d.doc_hash.eq(EXPECTED_DOC).all() and d.query_hash.eq(EXPECTED_QUERY).all(),'Per-question fingerprints: '+name)
        goldq=combined.set_index('qid').loc[d.qid]
        check(np.array_equal(goldq.orig_qid,d.orig_qid) and np.array_equal(boolean(goldq.is_paraphrase),boolean(d.is_paraphrase)),'Question identity and stratum: '+name)
        check(all(np.all(np.diff(g.sort_values('k').strict_hit)>=0) and np.all(np.diff(g.sort_values('k').soft_hit)>=0) for _,g in d.groupby('qid')),'Individual Hit@k monotonicity: '+name)
        ss = pd.read_csv(BLOCK/'results/main/by_question'/f'{name}_summary.csv')
        ss=ss.sort_values(['is_paraphrase','k']);ref=summary[summary.condition==name].sort_values(['is_paraphrase','k'])
        cols=['condition','dimension','is_paraphrase','k','n_questions','doc_hash','query_hash']
        check_frame(ss,ref,cols,'Individual summary identity: '+name)
        check(np.allclose(ss[['strict_hit','soft_hit']],ref[['strict_hit','soft_hit']],atol=1e-12),'Individual summary values: '+name)
    for path,pcol,hcol in [(BLOCK/'results/main/paired_vs_bge_small.csv','mcnemar_exact_p','holm_p'),(BLOCK/'results/paraphrase/paired_tests_hit5.csv','mcnemar_exact_p','holm_p_strict_hit5'),(BLOCK/'results/paraphrase/model_robustness_comparison.csv','paired_permutation_p','holm_p')]:
        df=pd.read_csv(path);check(np.allclose(holm(df[pcol]),df[hcol],atol=1e-12),'Holm correction: '+path.name)
    df=pd.read_csv(BLOCK/'results/main/nomic_mrl_vs_pca_same_dim.csv')
    check((holm(df.mcnemar_exact_p)>0.05).all(),'Nomic: no significant comparison after Holm',holm(df.mcnemar_exact_p).tolist())
    pairs=pd.read_csv(BLOCK/'results/paraphrase/selected_500_pairs.csv')
    qids=np.r_[pairs.original_qid,pairs.paraphrase_qid]
    texts=pd.concat([pairs.original_question,pairs.paraphrase_question],ignore_index=True)
    qhash=signature(qids,texts);runhash=hashlib.sha256((EXPECTED_DOC+qhash).encode()).hexdigest()
    detail=pd.read_csv(BLOCK/'results/paraphrase/paraphrase_by_pair.csv')
    check(not detail.duplicated(['model','orig_qid','k']).any(),'Unique paired model/question/k')
    check(detail.run_hash.eq(runhash).all(),'Paired run fingerprint rebuilt from texts')
    for model,dim in [('bge_small',384),('bge_base',768),('bge_large',1024)]:
        q=np.load(BLOCK/f'results/paraphrase/query_embeddings/{model}_paired_queries.npy',mmap_mode='r')
        m=json.loads((BLOCK/f'results/paraphrase/query_embeddings/{model}_paired_queries.json').read_text())
        check(q.shape==(1000,dim) and q.dtype==np.float32 and np.isfinite(q).all(),'Paired query matrix: '+model)
        check(np.max(np.abs(np.linalg.norm(q,axis=1)-1))<1e-5,'Query norms: '+model)
        check(m['query_hash']==qhash,'Query row fingerprint: '+model)
        top=np.load(BLOCK/f'results/paraphrase/rankings/{model}_topk_ids.npy')
        state=json.loads((BLOCK/f'results/paraphrase/rankings/{model}_topk_state.json').read_text())
        check(state['completed']==1000 and state['config']['run_hash']==runhash,'Complete ranking state: '+model)
        d=detail[detail.model==model]
        per=pd.read_csv(BLOCK/f'results/paraphrase/tables/{model}_500pairs.csv')
        check_frame(d,per,list(d.columns),'Individual pair table: '+model)
        for k in [1,3,5,10]:
            ss=d[d.k==k].set_index('orig_qid').loc[pairs.orig_qid]
            overlap=[len(set(a[:k])&set(b[:k]))/k for a,b in zip(top[:500],top[500:])]
            check(np.allclose(overlap,ss.topk_overlap_fraction),'Top-k overlap: '+model+f' k={k}')
    return selected,pilot,pairs

def data_and_vectors(data_dir, selected, pilot, pairs):
    c=pd.read_csv(data_dir/'contexts.csv',keep_default_na=False)
    d=pd.read_csv(data_dir/'chunks.csv',keep_default_na=False)
    q=pd.read_csv(data_dir/'questions.csv',keep_default_na=False)
    a=pd.read_csv(data_dir/'answers.csv',keep_default_na=False)
    ac=pd.read_csv(data_dir/'answer_chunks.csv',keep_default_na=False)
    q['is_paraphrase']=boolean(q.is_paraphrase);ac['is_annotated']=boolean(ac.is_annotated)
    counts={k:len(v) for k,v in [('contexts',c),('chunks',d),('questions',q),('answers',a),('answer_chunks',ac)]}
    check(len(c)==20958 and len(d)==82222 and len(ac)==160386 and int(ac.is_annotated.sum())==146780,'Expected dataset sizes',counts)
    check(c.pid.is_unique and d.chunk_id.is_unique and q.qid.is_unique,'Unique primary keys')
    check(not ac.duplicated(['orig_qid','chunk_id']).any() and not a.duplicated(['orig_qid','text','start']).any(),'Unique gold rows')
    check(signature(d.chunk_id,d.text)==EXPECTED_DOC,'Rebuilt corpus equals historical document fingerprint')
    check(np.array_equal(d.chunk_id,np.arange(len(d))),'Canonical chunk_id row order')
    check(d.pid.isin(c.pid).all() and q.pid.isin(c.pid).all() and ac.chunk_id.isin(d.chunk_id).all(),'Document foreign keys')
    originals=q[~q.is_paraphrase].set_index('qid')
    check(q.orig_qid.isin(originals.index).all() and a.orig_qid.isin(originals.index).all() and ac.orig_qid.isin(originals.index).all(),'Question foreign keys')
    check(q.role.isin(['train','valid']).all() and np.array_equal(q.role,originals.loc[q.orig_qid].role),'Train/valid roles inherited correctly')
    check(np.array_equal(q.pid,originals.loc[q.orig_qid].pid),'Original/paraphrase paragraph identity')
    context=c.set_index('pid').context.to_dict()
    check(all(0<=r.start<r.end<=len(context[r.pid]) and r.text==context[r.pid][r.start:r.end] for r in d.itertuples()),'Every chunk matches its paragraph span')
    check(c.set_index('pid').n_chunks.equals(d.groupby('pid').size().rename('n_chunks')),'Paragraph chunk counts')
    check(all(r.text==context[int(originals.loc[r.orig_qid,'pid'])][r.start:r.start+len(r.text)] for r in a.itertuples()),'Every annotated answer matches its paragraph span')
    check(set(originals.index)==set(ac.loc[ac.is_annotated,'orig_qid']),'Every original has strict gold')
    for tab,label in [(selected,'main'),(pilot,'pilot')]:
        aligned=q.set_index('qid').loc[tab.qid]
        check(np.array_equal(aligned.question,tab.question) and np.array_equal(aligned.orig_qid,tab.orig_qid) and aligned.role.eq('train').all(),'Frozen questions match full dataset: '+label)
    for field,textcol in [('original_qid','original_question'),('paraphrase_qid','paraphrase_question')]:
        aligned=q.set_index('qid').loc[pairs[field]]
        check(np.array_equal(aligned.question,pairs[textcol]) and np.array_equal(aligned.orig_qid,pairs.orig_qid) and aligned.role.eq('train').all(),'500 pairs match full dataset: '+field)
    # Rebuild the complete strict/soft relation from spans, independently of saved CSV.
    achunks=defaultdict(list);answers=defaultdict(list)
    for r in d.itertuples():achunks[r.pid].append((r.chunk_id,r.start,r.end))
    for r in a.itertuples():answers[r.orig_qid].append((r.text,r.start))
    expected=set()
    for oid,r in originals.iterrows():
        ann={(start,len(text)) for text,start in answers[oid]}
        found=set()
        for text in {text for text,_ in answers[oid] if text}:
            found.update((m.start(),len(text)) for m in re.finditer(r'(?<!\w)'+re.escape(text)+r'(?!\w)',context[r.pid]))
        for cid,start,end in achunks[r.pid]:
            strict=any(start<=s and s+length<=end for s,length in ann)
            soft=any(start<=s and s+length<=end for s,length in found-ann)
            if strict or soft:expected.add((int(oid),cid,bool(strict)))
    actual=set(ac.itertuples(index=False,name=None))
    check(actual==expected,'Entire strict/soft gold relation regenerated',{'missing':len(expected-actual),'extra':len(actual-expected)})
    strict=ac[ac.is_annotated].groupby('orig_qid').chunk_id.agg(set).to_dict()
    soft=ac.groupby('orig_qid').chunk_id.agg(set).to_dict()
    detail=pd.read_csv(BLOCK/'results/paraphrase/paraphrase_by_pair.csv')
    for model in ['bge_small','bge_base','bge_large']:
        top=np.load(BLOCK/f'results/paraphrase/rankings/{model}_topk_ids.npy')
        check(np.isin(top,d.chunk_id).all(),'All ranked IDs exist: '+model)
        for k in [1,3,5,10]:
            ss=detail[(detail.model==model)&(detail.k==k)].set_index('orig_qid').loc[pairs.orig_qid]
            for offset,kind in [(0,'original'),(500,'paraphrase')]:
                for gold,label in [(strict,'strict'),(soft,'soft')]:
                    hits=[int(bool(set(row[:k])&gold[int(oid)])) for row,oid in zip(top[offset:offset+500],pairs.orig_qid)]
                    check(np.array_equal(hits,ss[f'{label}_{kind}']),'Gold -> ranking -> hits: '+model+f' {kind}/{label}@{k}')
    docs=np.load(data_dir/'embeddings_bge-small-en-v1.5/chunk_embeddings.npy',mmap_mode='r')
    ids=np.load(data_dir/'embeddings_bge-small-en-v1.5/chunk_ids.npy')
    queries=np.load(data_dir/'embeddings_bge-small-en-v1.5/query_embeddings.npy')
    qids=np.load(data_dir/'embeddings_bge-small-en-v1.5/query_ids.npy')
    check(docs.shape==(82222,384) and docs.dtype==np.float32 and np.isfinite(docs).all(),'Full document matrix finite float32 82222x384')
    check(np.max(np.abs(np.linalg.norm(docs,axis=1)-1))<1e-5,'Full document matrix L2 norms')
    check(np.array_equal(ids,d.chunk_id) and np.array_equal(qids,selected.qid),'Block 2 document/query row order')
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=4):
        top,scores=exact_topk(queries,docs,ids)
        ptop,_=exact_topk(np.load(BLOCK/'results/paraphrase/query_embeddings/bge_small_paired_queries.npy'),docs,ids)
    old=np.load(BLOCK/'results/intermediate/bge_full_topk.npz')
    check(np.array_equal(top,old['chunk_id']) and np.allclose(scores,old['similarity'],atol=1e-6),'Exact BGE Small replay: 500 questions, all top-10 IDs and scores')
    check(np.array_equal(ptop,np.load(BLOCK/'results/paraphrase/rankings/bge_small_topk_ids.npy')),'Exact BGE Small replay: all 1000 paired query top-10 IDs')
    check(np.array_equal(top,np.load(data_dir/'embeddings_bge-small-en-v1.5/exact_top10_chunk_ids.npy')),'Block 2 exact reference matches replay')
    main=pd.read_csv(BLOCK/'results/main/by_question/bge_small_full_by_question.csv')
    for k in [1,3,5,10]:
        ss=main[(main.k==k)&(~boolean(main.is_paraphrase))].set_index('qid').loc[qids]
        for gold,label in [(strict,'strict'),(soft,'soft')]:
            hits=[int(bool(set(row[:k])&gold[int(oid)])) for row,oid in zip(top,qids)]
            check(np.array_equal(hits,ss[label+'_hit']),'Replayed Small matches FINAL main metrics: '+label+f'@{k}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tables-only',action='store_true')
    parser.add_argument('--data-dir',type=Path,default=ROOT/'data')
    args=parser.parse_args()
    try:
        selected,pilot,pairs=code_and_tables()
        if not args.tables_only:data_and_vectors(args.data_dir,selected,pilot,pairs)
    except Exception:
        import traceback
        check(False,'Audit exception',traceback.format_exc())
    report={'status':'PASS' if all(x['ok'] for x in checks) else 'FAIL','scope':'tables_and_code' if args.tables_only else 'tables_code_full_dataset_BGE_Small_retrieval','checks':len(checks),'passed':sum(x['ok'] for x in checks),'details':checks,'limitations':['No new neural model encoding; Base/Large/Nomic document matrices not supplied.','Human semantic review of paraphrases remains pending.','Historical exact software/model revisions of encoding are unknown.']}
    out=BLOCK/'local_checks';out.mkdir(exist_ok=True)
    (out/'audit_release.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='details'},ensure_ascii=False,indent=2))
    return 0 if report['status']=='PASS' else 1
if __name__=='__main__':sys.exit(main())
