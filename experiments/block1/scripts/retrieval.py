"""Exact float32 dot-product search; ties use ascending chunk_id, including at k."""
import hashlib

import numpy as np


def array_hash(array):
    h = hashlib.sha256()
    h.update(str(array.shape).encode())
    h.update(str(array.dtype).encode())
    for start in range(0, len(array), 4096):
        h.update(np.ascontiguousarray(array[start:start+4096]).tobytes())
    return h.hexdigest()


def signature(ids, texts):
    if len(ids) != len(texts):
        raise ValueError('IDs and texts have different lengths')
    h = hashlib.sha256()
    for i, text in zip(ids, texts):
        h.update(str(int(i)).encode())
        h.update(b'\0')
        h.update(str(text).encode('utf-8'))
        h.update(b'\n')
    return h.hexdigest()


def stable_topk_positions(scores, ids, k):
    """Select all boundary ties before choosing the smallest document IDs."""
    scores = np.asarray(scores)
    ids = np.asarray(ids)
    if scores.ndim != 2 or ids.ndim != 1 or scores.shape[1] != len(ids):
        raise ValueError('Expected scores[queries, documents] and ids[documents]')
    if not 1 <= k <= scores.shape[1] or not np.isfinite(scores).all():
        raise ValueError('Invalid k or nonfinite similarities')
    partial = np.argpartition(-scores, k-1, axis=1)[:, :k]
    result = np.empty_like(partial)
    for i, row in enumerate(scores):
        cutoff = row[partial[i]].min()
        above = np.flatnonzero(row > cutoff)
        ties = np.flatnonzero(row == cutoff)
        ties = ties[np.argsort(ids[ties], kind='stable')[:k-len(above)]]
        selected = np.concatenate((above, ties))
        result[i] = selected[np.lexsort((ids[selected], -row[selected]))]
    return result


def exact_topk(queries, documents, ids, k=10, query_block=32, doc_block=4096):
    if queries.ndim != 2 or documents.ndim != 2 or queries.shape[1] != documents.shape[1]:
        raise ValueError('Incompatible embedding shapes')
    ids = np.asarray(ids, dtype=np.int64)
    if len(ids) != len(documents) or len(np.unique(ids)) != len(ids):
        raise ValueError('Document IDs must be unique and aligned with rows')
    if not 1 <= k <= len(ids) or query_block < 1 or doc_block < 1:
        raise ValueError('Invalid search size')
    found = np.empty((len(queries), k), dtype=np.int64)
    values = np.empty((len(queries), k), dtype=np.float32)
    for qb in range(0, len(queries), query_block):
        q = np.ascontiguousarray(queries[qb:qb+query_block], dtype=np.float32)
        best_s = np.empty((len(q), 0), dtype=np.float32)
        best_i = np.empty((len(q), 0), dtype=np.int64)
        for db in range(0, len(documents), doc_block):
            d = np.ascontiguousarray(documents[db:db+doc_block], dtype=np.float32)
            scores = q @ d.T
            pos = stable_topk_positions(scores, ids[db:db+len(d)], min(k, len(d)))
            ss = np.concatenate((best_s, np.take_along_axis(scores, pos, axis=1)), axis=1)
            ii = np.concatenate((best_i, ids[db+pos]), axis=1)
            order = np.lexsort((ii, -ss), axis=1)[:, :k]
            best_s = np.take_along_axis(ss, order, axis=1)
            best_i = np.take_along_axis(ii, order, axis=1)
        found[qb:qb+len(q)], values[qb:qb+len(q)] = best_i, best_s
    return found, values
