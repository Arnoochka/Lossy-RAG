import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from .base import BaseSearch
from src.encoder import BaseEncoder
from src.quant.base import BaseQuant


class Bm25Search(BaseSearch):
    def __init__(self, encoder: BaseEncoder, quantizer: BaseQuant, k1: float = 1.2, b: float = 0.75) -> None:
        super().__init__(encoder, quantizer)
        self.k1 = k1
        self.b = b
        
    def fit(self, texts: list[str]) -> "Bm25Search":
        return self

    def add(self, chunk_texts: list[str], chunk_ids: np.ndarray) -> "Bm25Search":
        
        self.chunk_ids = chunk_ids
        self.vectorizer = CountVectorizer(
            token_pattern=r"(?u)\b\w+\b", dtype=np.float32,
        )
        counts = self.vectorizer.fit_transform(chunk_texts).tocsr()
        lengths = np.asarray(counts.sum(axis=1)).ravel()
        document_frequency = np.bincount(
            counts.indices, minlength=counts.shape[1],
        )
        idf = np.log1p(
            (len(chunk_ids) - document_frequency + 0.5)
            / (document_frequency + 0.5)
        )
        length_norm = self.k1 * (
            1 - self.b + self.b * lengths / lengths.mean()
        )
        counts.data = (
            idf[counts.indices] * counts.data * (self.k1 + 1)
            / (counts.data + np.repeat(length_norm, np.diff(counts.indptr)))
        ).astype(np.float32)
        self.bm25_weights = counts
        return self

    def search(self, query_texts: list[str], k: int = 10, batch_size: int = 64) -> tuple[np.ndarray, np.ndarray]:
        
        top_ids = np.full((len(query_texts), k), -1, dtype=self.chunk_ids.dtype)
        top_scores = np.full((len(query_texts), k), -np.inf, dtype=np.float32)
        count = min(k, len(self.chunk_ids))

        for start in range(0, len(query_texts), batch_size):
            queries = self.vectorizer.transform(query_texts[start:start + batch_size])
            queries.data.fill(1)
            batch_scores = (queries @ self.bm25_weights.T).tocsr()

            for i in range(batch_scores.shape[0]):
                scores = batch_scores[i:i + 1].toarray().ravel()
                candidates = np.argpartition(scores, -count)[-count:]
                order = candidates[np.argsort(-scores[candidates], kind="stable")]
                top_ids[start + i, :count] = self.chunk_ids[order]
                top_scores[start + i, :count] = scores[order]

        return top_ids, top_scores


class HybridBm25Search(Bm25Search):
    def __init__(
        self,
        vector_search: BaseSearch,
        alpha: float = 0.5,
        num_candidates: int = 100,
        k1: float = 1.2,
        b: float = 0.75,
    ) -> None:
        super().__init__(
            encoder=vector_search.encoder,
            quantizer=vector_search.quantizer,
            k1=k1,
            b=b,
        )
        self.vector_search = vector_search
        self.alpha = alpha
        self.num_candidates = num_candidates

    def fit(self, texts: list[str]) -> "HybridBm25Search":
        self.vector_search.fit(texts)
        return self

    def add(
        self,
        chunk_texts: list[str],
        chunk_ids: np.ndarray,
    ) -> "HybridBm25Search":
        super().add(chunk_texts, chunk_ids)
        self.vector_search.add(chunk_texts, chunk_ids)
        return self

    @staticmethod
    def _normalize(scores: np.ndarray) -> np.ndarray:
        if len(scores) == 0 or scores.max() == scores.min():
            return np.zeros_like(scores)
        return (scores - scores.min()) / (scores.max() - scores.min())

    def search(
        self,
        query_texts: list[str],
        k: int = 10,
        batch_size: int = 64,
    ) -> tuple[np.ndarray, np.ndarray]:
        
        count = min(max(self.num_candidates, k), len(self.chunk_ids))
        bm25_ids, bm25_scores = super().search(query_texts, count, batch_size)
        vector_ids, vector_scores = self.vector_search.search(
            query_texts, count, batch_size,
        )
        top_ids = np.full((len(query_texts), k), -1, dtype=self.chunk_ids.dtype)
        top_scores = np.full((len(query_texts), k), -np.inf, dtype=np.float32)

        for i in range(len(query_texts)):
            valid = np.isfinite(vector_scores[i])
            vec_ids = vector_ids[i, valid]
            vec_scores = self._normalize(vector_scores[i, valid])
            lexical_scores = self._normalize(bm25_scores[i])

            if self.alpha == 0:
                ids, scores = bm25_ids[i], lexical_scores
            elif self.alpha == 1:
                ids, scores = vec_ids, vec_scores
            else:
                ids, positions = np.unique(
                    np.concatenate((vec_ids, bm25_ids[i])), return_inverse=True,
                )
                scores = np.zeros(len(ids), dtype=np.float32)
                contributions = np.concatenate((
                    self.alpha * vec_scores,
                    (1 - self.alpha) * lexical_scores,
                ))
                np.add.at(scores, positions, contributions)

            order = np.argsort(-scores, kind="stable")[:k]
            top_ids[i, :len(order)] = ids[order]
            top_scores[i, :len(order)] = scores[order]

        return top_ids, top_scores
