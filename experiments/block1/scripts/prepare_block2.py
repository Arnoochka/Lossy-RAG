#!/usr/bin/env python3
"""Switch only the BGE Small query selection used by upstream stage2 notebooks.
main: published 500 original train questions (default in the supplied data pack).
paraphrase: all 500 originals, followed by their 500 paraphrases. No valid queries.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from retrieval import exact_topk, signature
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[3];BLOCK=ROOT/'experiments/block1'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for x in iter(lambda:f.read(8*1024*1024),b''):h.update(x)
    return h.hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--sample',choices=['main','paraphrase'],required=True);a=p.parse_args()
    dest=ROOT/'data/embeddings_bge-small-en-v1.5';d=pd.read_csv(ROOT/'data/chunks.csv');q=pd.read_csv(ROOT/'data/questions.csv')
    meta=json.loads((dest/'manifest.json').read_text())
    if signature(d.chunk_id,d.text)!=meta['doc_fingerprint']:raise ValueError('Corpus fingerprint mismatch')
    docs=np.load(dest/'chunk_embeddings.npy',mmap_mode='r');ids=np.load(dest/'chunk_ids.npy')
    if not np.array_equal(ids,d.chunk_id) or sha(dest/'chunk_embeddings.npy')!=meta['files']['chunk_embeddings.npy']['sha256']:raise ValueError('Document vectors or IDs changed')
    if a.sample=='main':
        selected=pd.read_csv(BLOCK/'results/main/selected_train_questions.csv');qids=selected.qid.to_numpy(dtype=np.int64)
        vectors=np.load(BLOCK/'results/intermediate/embeddings/bge_small_selected_queries.npy')
        reference=np.load(BLOCK/'results/intermediate/bge_full_topk.npz')['chunk_id']
    else:
        selected=pd.read_csv(BLOCK/'results/paraphrase/selected_500_pairs.csv');qids=np.r_[selected.original_qid,selected.paraphrase_qid].astype(np.int64)
        vectors=np.load(BLOCK/'results/paraphrase/query_embeddings/bge_small_paired_queries.npy')
        reference=np.load(BLOCK/'results/paraphrase/rankings/bge_small_topk_ids.npy')
    if not q.set_index('qid').loc[qids].role.eq('train').all():raise ValueError('Expected train questions only')
    with threadpool_limits(limits=4):top,scores=exact_topk(vectors,docs,ids)
    if not np.array_equal(top,reference):raise ValueError('Vectors do not reproduce saved ranking; no output changed')
    outputs={'query_ids.npy':qids,'query_embeddings.npy':vectors,'exact_top10_chunk_ids.npy':top,'exact_top10_scores.npy':scores}
    # Remove other exact-k caches, whose query selection could otherwise become stale.
    for name in dest.glob('exact_top*.npy'):
        if name.name not in outputs:name.unlink()
    for name,array in outputs.items():np.save(dest/name,array)
    meta['n_queries']=len(qids);meta['query_scope']=a.sample+'; TRAIN only'
    for name in outputs:meta['files'][name]={'bytes':(dest/name).stat().st_size,'sha256':sha(dest/name)}
    (dest/'manifest.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
    print(f'Ready for stage2 notebooks: {len(docs)} documents, {len(qids)} train queries; sample={a.sample}')
if __name__=='__main__':main()
