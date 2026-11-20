#!/usr/bin/env python3
"""Export verified document matrices and corpus to a NEW directory.
Default uses the supplied BGE Small vectors. --include-vectors all also requires
Base/Large/Nomic final caches on the original experiment computer.
"""
import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from retrieval import signature


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda:file.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--include-vectors',choices=['small','all'],default='small')
    args=parser.parse_args();root=args.root.resolve();out=args.out.resolve()
    if out.exists():parser.error('Destination already exists; choose a new export directory')
    for protected in [root/'data',root/'lossy_rag_final_results_safe',root/'experiments/block1']:
        if out==protected or protected in out.parents:parser.error('Destination overlaps source data/cache')
    data=root/'data';research=Path(__file__).resolve().parents[1]
    chunks=pd.read_csv(data/'chunks.csv');fingerprint=signature(chunks.chunk_id,chunks.text)
    expected=pd.read_csv(research/'results/main/all_methods_summary.csv').doc_hash.iloc[0]
    if len(chunks)!=82222 or not chunks.chunk_id.is_unique or fingerprint!=expected:raise ValueError('Corpus does not match the experiment')
    models={'bge_small':('BAAI/bge-small-en-v1.5',384)}
    if args.include_vectors=='all':models.update({'bge_base':('BAAI/bge-base-en-v1.5',768),'bge_large':('BAAI/bge-large-en-v1.5',1024),'nomic':('nomic-ai/nomic-embed-text-v1.5',768)})
    validated={}
    for slug,(model_id,dim) in models.items():
        if slug=='bge_small' and (data/'embeddings_bge-small-en-v1.5/manifest.json').exists():
            base=data/'embeddings_bge-small-en-v1.5';info=json.loads((base/'manifest.json').read_text())
            path=base/'chunk_embeddings.npy'
            if info['model_id']!=model_id or info['doc_fingerprint']!=fingerprint or sha(path)!=info['files'][path.name]['sha256'] or not np.array_equal(np.load(base/'chunk_ids.npy'),chunks.chunk_id):raise ValueError('BGE Small handoff checksum/order mismatch')
        else:
            base=root/'lossy_rag_final_results_safe/cache';meta=base/f'{slug}_docs.json'
            info=json.loads(meta.read_text());path=Path(info.get('source',str(base/f'{slug}_docs.npy')))
            if not path.is_file():path=base/f'{slug}_docs.npy'
            if info.get('fingerprint')!=fingerprint or info.get('model_id')!=model_id or info.get('kind')!='docs':raise ValueError('Wrong document cache: '+slug)
        a=np.load(path,mmap_mode='r')
        if a.shape!=(82222,dim) or a.dtype!=np.float32:raise ValueError('Wrong shape/dtype: '+slug)
        for start in range(0,len(a),4096):
            x=np.asarray(a[start:start+4096]);norm=np.linalg.norm(x,axis=1)
            if not np.isfinite(x).all() or (norm<1e-8).any():raise ValueError('Invalid vectors: '+slug)
            if slug!='nomic' and np.max(np.abs(norm-1))>1e-4:raise ValueError('Expected L2-normalized BGE vectors')
        validated[slug]=path
    # Validate all sources before creating any export files.
    required=['contexts.csv','chunks.csv','questions.csv','answers.csv','answer_chunks.csv']
    for name in required:
        if not (data/name).is_file():raise FileNotFoundError(data/name)
    out.parent.mkdir(parents=True,exist_ok=True)
    temporary=Path(tempfile.mkdtemp(prefix='.block1-export-',dir=out.parent))
    try:
        for name in required:shutil.copy2(data/name,temporary/name)
        for source in [research/'results/main/selected_train_questions.csv',research/'results/paraphrase/selected_500_pairs.csv']:shutil.copy2(source,temporary/source.name)
        for slug,path in validated.items():shutil.copy2(path,temporary/f'{slug}_docs.npy')
        manifest={'doc_fingerprint':fingerprint,'corpus_rows':len(chunks),'vector_row_order':'chunks.csv, unsorted','models':{slug:{'model_id':models[slug][0],'dimension':models[slug][1],'file':f'{slug}_docs.npy'} for slug in validated},'files':{p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in temporary.iterdir()}}
        (temporary/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        temporary.rename(out)
    except BaseException:
        shutil.rmtree(temporary);raise
    print('Export validated:',out)
if __name__=='__main__':main()
