import numpy as np

from src.encoder import BaseEncoder
from src.quant.base import BaseQuant


class BaseSearch:
    def __init__(self, encoder: BaseEncoder, quantizer: BaseQuant) -> None:
        
        self.encoder = encoder
        self.quantizer = quantizer

    def _encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        return self.encoder.encode(texts, is_query=is_query)

    def fit(self, texts: list[str]) -> "BaseSearch":
        embeddings = self._encode(texts)
        self.quantizer.fit(embeddings)
        reconstructed = self.quantizer.dequant(self.quantizer.quant(embeddings))
        return self._fit(reconstructed)

    def _fit(self, X: np.ndarray) -> "BaseSearch":
        return self

    def add(self, chunk_texts: list[str], chunk_ids: np.ndarray) -> "BaseSearch":
        embeddings = self._encode(chunk_texts)
        reconstructed = self.quantizer.dequant(self.quantizer.quant(embeddings))
        return self._add(reconstructed, chunk_ids)

    def _add(self, chunk_embeddings: np.ndarray, chunk_ids: np.ndarray) -> "BaseSearch":
        self.chunk_embeddings = chunk_embeddings
        self.chunk_ids = chunk_ids
        return self

    def search(self, query_texts: list[str], k: int = 10, batch_size: int = 64) -> tuple[np.ndarray, np.ndarray]:
        query_embeddings = self._encode(query_texts, is_query=True)
        return self._search(query_embeddings, k, batch_size)

    def _search(self, query_embeddings: np.ndarray, k: int = 10, batch_size: int = 64) -> tuple[np.ndarray, np.ndarray]:
        top_ids = []
        top_scores = []

        for start in range(0, len(query_embeddings), batch_size):
            scores = (
                query_embeddings[start:start + batch_size]
                @ self.chunk_embeddings.T
            )
            candidates = np.argpartition(scores, -k, axis=1)[:, -k:]
            candidate_scores = np.take_along_axis(scores, candidates, axis=1)
            order = np.argsort(-candidate_scores, axis=1)
            positions = np.take_along_axis(candidates, order, axis=1)

            top_ids.append(self.chunk_ids[positions])
            top_scores.append(np.take_along_axis(scores, positions, axis=1))

        return np.concatenate(top_ids), np.concatenate(top_scores)
