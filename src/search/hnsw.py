import hnswlib
import numpy as np
from .base import BaseSearch
from src.encoder import BaseEncoder
from src.quant.base import BaseQuant


class HnswSearch(BaseSearch):
    def __init__(
        self,
        encoder: BaseEncoder,
        quantizer: BaseQuant,
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 100,
        random_state: int = 42) -> None:
        
        super().__init__(encoder, quantizer)
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.random_state = random_state

    def _add(self, chunk_embeddings: np.ndarray, chunk_ids: np.ndarray) -> "HnswSearch":
        
        super()._add(chunk_embeddings, chunk_ids)
        self.index = hnswlib.Index(space="ip", dim=chunk_embeddings.shape[1])
        self.index.init_index(
            max_elements=len(chunk_embeddings),
            M=self.M,
            ef_construction=self.ef_construction,
            random_seed=self.random_state,
        )
        self.index.add_items(chunk_embeddings, np.arange(len(chunk_embeddings)))
        
        return self

    def _search(self, query_embeddings: np.ndarray, k: int = 10, batch_size: int = 64) -> tuple[np.ndarray, np.ndarray]:
        
        self.index.set_ef(max(self.ef_search, k))
        top_ids = []
        top_scores = []

        for start in range(0, len(query_embeddings), batch_size):
            positions, distances = self.index.knn_query(
                query_embeddings[start:start + batch_size], k=k,
            )
            top_ids.append(self.chunk_ids[positions])
            # В режиме ip библиотека возвращает расстояние 1 - dot product.
            top_scores.append(1 - distances)

        return np.concatenate(top_ids), np.concatenate(top_scores)
