import ast
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

BLOCK=Path(__file__).resolve().parents[1];ROOT=BLOCK.parents[1]
sys.path.insert(0,str(BLOCK/'scripts'));sys.path.insert(0,str(ROOT))
from retrieval import array_hash, exact_topk, stable_topk_positions


def function_from_notebook(name,function,ns):
    nb=json.loads((BLOCK/'notebooks'/name).read_text())
    for cell in nb['cells']:
        if cell['cell_type']!='code':continue
        tree=ast.parse(''.join(cell['source']))
        for node in tree.body:
            if isinstance(node,ast.FunctionDef) and node.name==function:
                exec(compile(ast.Module(body=[node],type_ignores=[]),name,'exec'),ns)
                return ns[function]
    raise AssertionError('Missing notebook function '+function)

class SearchTests(unittest.TestCase):
    def test_ties_across_partition_boundary(self):
        ids=np.arange(40,0,-1);scores=np.ones((2,40),np.float32)
        pos=stable_topk_positions(scores,ids,10)
        np.testing.assert_array_equal(ids[pos],np.tile(np.arange(1,11),(2,1)))
    def test_blocks_match_full_sort(self):
        rng=np.random.default_rng(42);q=rng.standard_normal((9,7)).astype('float32');d=rng.standard_normal((45,7)).astype('float32');ids=rng.permutation(45)*3+100
        found,_=exact_topk(q,d,ids,k=10,doc_block=6,query_block=3)
        scores=q@d.T
        expected=np.array([ids[np.lexsort((ids,-row))[:10]] for row in scores])
        np.testing.assert_array_equal(found,expected)
    def test_reject_nonfinite_duplicate_and_large_k(self):
        q=np.ones((1,2),np.float32);d=np.ones((3,2),np.float32)
        with self.assertRaises(ValueError):exact_topk(q,d,[0,0,1],k=1)
        with self.assertRaises(ValueError):exact_topk(q,d,[0,1,2],k=4)
        d[0,0]=np.nan
        with self.assertRaises(ValueError):exact_topk(q,d,[0,1,2],k=1)
    def test_notebook_resume_and_changed_vectors(self):
        for name in ['01_BGE_и_PCA.ipynb','02_Nomic_Matryoshka_PCA.ipynb']:
            with self.subTest(name=name),tempfile.TemporaryDirectory() as tmp:
                result=Path(tmp);(result/'rankings').mkdir()
                import time
                ns={'np':np,'hashlib':hashlib,'json':json,'time':time,'K_VALUES':[1,3], 'QUERY_BLOCK':1,'DOC_BLOCK':5,'THREADS':1,'PRINT_EVERY_BLOCKS':1,'RESULT_DIR':result,'log':lambda *x:None,'threadpool_limits':threadpool_limits,'stable_topk_positions':stable_topk_positions,'json_atomic':lambda p,o:p.write_text(json.dumps(o)),'file_sha256':lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()}
                fn=function_from_notebook(name,'exact_topk_resume',ns)
                ids=np.arange(30,0,-1);docs=np.ones((30,2),np.float32);q=np.array([[1,0]],np.float32)
                a,_=fn('case',q,docs,ids,'q','d');np.testing.assert_array_equal(a,[[1,2,3]])
                a,_=fn('case',q,docs,ids,'q','d');np.testing.assert_array_equal(a,[[1,2,3]])
                docs[0,0]=2
                a,_=fn('case',q,docs,ids,'q','d');np.testing.assert_array_equal(a,[[30,1,2]])
    def test_notebook_rejects_corrupt_vectors(self):
        for name in ['01_BGE_и_PCA.ipynb','02_Nomic_Matryoshka_PCA.ipynb']:
            ns={'np':np,'Path':Path};fn=function_from_notebook(name,'checked_vectors',ns)
            with tempfile.TemporaryDirectory() as t:
                p=Path(t)/'x.npy';np.save(p,np.ones((4,3),np.float32));self.assertIsNotNone(fn(p,4,3))
                x=np.ones((4,3),np.float32);x[3,2]=np.nan;np.save(p,x);self.assertIsNone(fn(p,4,3))
    def test_pca_cache_uses_actual_source_vectors(self):
        import time

        from sklearn.decomposition import PCA
        for name in ['01_BGE_и_PCA.ipynb','02_Nomic_Matryoshka_PCA.ipynb']:
            with self.subTest(name=name),tempfile.TemporaryDirectory() as tmp:
                result=Path(tmp);(result/'pca').mkdir()
                ns={'np':np,'json':json,'time':time,'PCA':PCA,'PCA_FIT_SAMPLE':32,'DOC_HASH':'same-texts','chunk_ids':np.arange(32),'THREADS':1,'RESULT_DIR':result,'log':lambda *x:None,'threadpool_limits':threadpool_limits,'array_hash':array_hash,'json_atomic':lambda p,o:p.write_text(json.dumps(o))}
                fn=function_from_notebook(name,'get_pca_basis',ns)
                docs=np.random.default_rng(42).normal(size=(32,8)).astype(np.float32)
                mean1,_=fn(docs,'test',42,[2]);mean2,_=fn(docs+3,'test',42,[2])
                np.testing.assert_allclose(mean2,mean1+3,atol=1e-6)

class BuilderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec=importlib.util.spec_from_file_location('builder',ROOT/'data/build_data.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);cls.Builder=m.SquadDatasetBuilder
    def test_bad_window(self):
        with self.assertRaises(ValueError):self.Builder(window=0)
    def test_span_windows(self):
        builder=self.Builder.__new__(self.Builder);builder.window=2;builder.step=1
        self.assertEqual(builder._windows([(0,3),(4,8),(9,13)],13),[(0,8),(4,13)])
        self.assertEqual(builder._windows([(1,3)],5),[(0,5)])
    def test_paraphrase_identity_checks(self):
        import pandas as pd
        q=pd.DataFrame([{'qid':1,'squad_id':'id','orig_qid':1,'role':'train','is_paraphrase':False,'pid':3,'question':'One?'}]);p=q.copy();p['qid']=2;p['question']='One rewritten?'
        self.assertEqual(len(self.Builder._add_paraphrases(q,p)),2)
        p['role']='valid'
        with self.assertRaises(ValueError):self.Builder._add_paraphrases(q,p)
        p['role']='train';p['qid']=1
        with self.assertRaises(ValueError):self.Builder._add_paraphrases(q,p)

class UpstreamIntegration(unittest.TestCase):
    def test_stage2_without_model_libraries(self):
        # These imports previously required torch even for cached embeddings.
        from src.encoder import DummyEncoder
        from src.quant import BaseQuant, BqQuant, PqQuant, SqQuant
        from src.search import BaseSearch, Bm25Search, HnswSearch, IvfSearch
        rng=np.random.default_rng(42);docs=rng.normal(size=(100,8)).astype(np.float32);docs/=np.linalg.norm(docs,axis=1,keepdims=True)
        texts=[f'item {i} token{i}' for i in range(len(docs))];queries=['query0','query1'];q=docs[:2]
        enc=DummyEncoder(texts,docs,queries,q);ids=np.arange(len(docs))*2+10
        baseline=BaseSearch(enc,BaseQuant());baseline.fit(texts);baseline.add(texts,ids);found,_=baseline.search(queries,k=5)
        np.testing.assert_array_equal(found[:,0],ids[:2])
        for quant in [SqQuant(8),BqQuant(),PqQuant(2,2)]:
            with threadpool_limits(limits=1):quant.fit(docs);restored=quant.dequant(quant.quant(docs))
            self.assertEqual(restored.shape,docs.shape);self.assertTrue(np.isfinite(restored).all())
        for search in [IvfSearch(enc,BaseQuant(),nlist=4,nprobe=4),HnswSearch(enc,BaseQuant()),Bm25Search(enc,BaseQuant())]:
            with threadpool_limits(limits=1):search.fit(texts);search.add(texts,ids);found,score=search.search(queries,k=5)
            self.assertEqual(found.shape,(2,5));self.assertTrue(np.isin(found,ids).all())
            if not isinstance(search,Bm25Search):np.testing.assert_array_equal(found[:,0],ids[:2])

if __name__=='__main__':unittest.main(verbosity=2)
