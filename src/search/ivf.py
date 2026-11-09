import numpy as np
from heapq import heappush, heapreplace
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances_argmin
from .base import BaseSearch
from src.encoder import BaseEncoder
from src.quant.base import BaseQuant
from src.quant.bq import RaBitqQuant

class IvfSearch(BaseSearch):
    def __init__(
        self,
        encoder: BaseEncoder,
        quantizer: BaseQuant,
        nlist: int = 1024,
        nprobe: int = 16,
        random_state: int = 42) -> None:
        
        super().__init__(encoder, quantizer)
        self.nlist = nlist
        self.nprobe = nprobe
        self.random_state = random_state

    def _fit(self, X: np.ndarray) -> "IvfSearch":
        kmeans = MiniBatchKMeans(
            n_clusters=self.nlist,
            batch_size=4096,
            n_init=1,
            max_iter=50,
            compute_labels=False,
            random_state=self.random_state,
        )
        kmeans.fit(X)
        self.centroids = kmeans.cluster_centers_.astype(np.float32)
        return self

    def _add(self, chunk_embeddings: np.ndarray, chunk_ids: np.ndarray) -> "IvfSearch":
        super()._add(chunk_embeddings, chunk_ids)
        labels = pairwise_distances_argmin(chunk_embeddings, self.centroids)
        self.lists = [np.flatnonzero(labels == i) for i in range(self.nlist)]
        return self

    def _search(self, query_embeddings: np.ndarray, k: int = 10, batch_size: int = 64) -> tuple[np.ndarray, np.ndarray]:
        
        top_ids = np.full((len(query_embeddings), k), -1, dtype=self.chunk_ids.dtype)
        top_scores = np.full(
            (len(query_embeddings), k), -np.inf, dtype=self.chunk_embeddings.dtype,
        )
        centroid_norms = np.sum(self.centroids ** 2, axis=1)

        for start in range(0, len(query_embeddings), batch_size):
            queries = query_embeddings[start:start + batch_size]
            # ||q||² одинаково для всех центроидов и не влияет на их порядок. ||q-c_norms||^2
            distances = centroid_norms - 2 * queries @ self.centroids.T
            clusters = np.argpartition(distances, self.nprobe - 1, axis=1)[:, :self.nprobe]

            for i, (query, cluster_ids) in enumerate(zip(queries, clusters)):
                positions = np.concatenate([self.lists[j] for j in cluster_ids])
                count = min(k, len(positions))
                if count == 0:
                    continue
                scores = self.chunk_embeddings[positions] @ query
                candidates = np.argpartition(scores, -count)[-count:]
                order = candidates[np.argsort(-scores[candidates])]
                top_ids[start + i, :count] = self.chunk_ids[positions[order]]
                top_scores[start + i, :count] = scores[order]

        return top_ids, top_scores


class IvfRaBitqSearch(IvfSearch):
    def __init__(
        self,
        encoder: BaseEncoder,
        quantizer: BaseQuant,
        nlist: int = 1024,
        nprobe: int = 16,
        random_state: int = 42,
        epsilon: float = 1.9) -> None:
        
        super().__init__(
            encoder=encoder,
            quantizer=quantizer,
            nlist=nlist,
            nprobe=nprobe,
            random_state=random_state,
        )
        self.epsilon = epsilon

    def _fit(self, X: np.ndarray) -> "IvfRaBitqSearch":
        super()._fit(X)
        self.rabitq_quantizer = RaBitqQuant(random_state=self.random_state).fit(X)
        self.dimension = self.rabitq_quantizer.dimension
        return self

    def _add(self, chunk_embeddings: np.ndarray, chunk_ids: np.ndarray) -> "IvfRaBitqSearch":
        super()._add(chunk_embeddings, chunk_ids)
        self.labels = np.empty(len(chunk_embeddings), dtype=np.int32)
        for cluster_id, positions in enumerate(self.lists):
            self.labels[positions] = cluster_id

        residuals = chunk_embeddings - self.centroids[self.labels]
        self.codes = self.rabitq_quantizer.quant(residuals)
        norm_squared = np.sum(residuals * residuals, axis=1)
        # ||r_hat - r||² = d * scale² - ||r||²; делим на d - 1 для границы ошибки.
        self.error_scales = np.sqrt(
            np.maximum(self.dimension * self.codes["scale"] ** 2 - norm_squared, 0)
            / max(self.dimension - 1, 1)
        )
        return self

    def _search(self, query_embeddings: np.ndarray, k: int = 10, batch_size: int = 64) -> tuple[np.ndarray, np.ndarray]:
        
        top_ids = np.full((len(query_embeddings), k), -1, dtype=self.chunk_ids.dtype)
        top_scores = np.full(
            (len(query_embeddings), k), -np.inf, dtype=self.chunk_embeddings.dtype,
        )
        centroid_norms = np.sum(self.centroids ** 2, axis=1)

        for start in range(0, len(query_embeddings), batch_size):
            queries = query_embeddings[start:start + batch_size]
            centroid_scores = queries @ self.centroids.T
            distances = centroid_norms - 2 * centroid_scores
            clusters = np.argpartition(distances, self.nprobe - 1, axis=1)[:, :self.nprobe]
            rotated_queries = queries @ self.rabitq_quantizer.rotation

            for i, (query, cluster_ids) in enumerate(zip(queries, clusters)):
                positions = np.concatenate([self.lists[j] for j in cluster_ids])
                codes = self.codes[positions]
                bits = np.unpackbits(codes["bits"], axis=1, count=self.dimension)
                signs = bits.astype(np.float32) * 2 - 1
                estimates = codes["scale"] * (signs @ rotated_queries[i])
                # x = c + r, поэтому <q, x> = <q, c> + <q, r>.
                estimates += centroid_scores[i, self.labels[positions]]
                margins = self.epsilon * np.linalg.norm(query) * self.error_scales[positions]
                upper_scores = estimates + margins
                best: list[tuple[float, int]] = []

                # При таком порядке после первого отсева остальные границы ещё ниже.
                for candidate in np.argsort(-upper_scores):
                    if len(best) == k and upper_scores[candidate] < best[0][0]:
                        break
                    position = int(positions[candidate])
                    score = float(self.chunk_embeddings[position] @ query)
                    if len(best) < k:
                        heappush(best, (score, position))
                    elif score > best[0][0]:
                        heapreplace(best, (score, position))

                for rank, (score, position) in enumerate(sorted(best, reverse=True)):
                    top_ids[start + i, rank] = self.chunk_ids[position]
                    top_scores[start + i, rank] = score

        return top_ids, top_scores
