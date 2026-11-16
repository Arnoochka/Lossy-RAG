import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances_argmin
from .base import BaseQuant

class AqQuant(BaseQuant):
    
    def __init__(
        self,
        num_codebooks: int,
        num_bits: int,
        num_iters: int = 5,
        num_search_iters: int = 3,
        random_state: int = 42,
    ) -> None:
        self.num_codebooks = num_codebooks
        self.num_bits = num_bits
        self.L = 2 ** num_bits
        self.num_iters = num_iters
        self.num_search_iters = num_search_iters
        self.random_state = random_state

    def fit(self, X: np.ndarray) -> "AqQuant":
        self.codebooks: list[np.ndarray] = []
        codes = np.empty(
            (len(X), self.num_codebooks), dtype=np.min_scalar_type(self.L - 1),
        )
        residual = X.copy()
        for i in range(self.num_codebooks):
            kmeans = MiniBatchKMeans(
                n_clusters=self.L,
                batch_size=4096,
                n_init=1,
                max_iter=50,
                compute_labels=False,
                random_state=self.random_state,
            )
            kmeans.fit(residual)
            codebook = kmeans.cluster_centers_.astype(np.float32)
            self.codebooks.append(codebook)
            codes[:, i] = pairwise_distances_argmin(residual, codebook)
            residual -= codebook[codes[:, i]]

        for _ in range(self.num_iters):
            codes = self._refine_codes(X, codes)
            residual = X - self.dequant(codes)
            for i, codebook in enumerate(self.codebooks):
                targets = residual + codebook[codes[:, i]]
                counts = np.bincount(codes[:, i], minlength=self.L)
                sums = np.zeros_like(codebook)
                np.add.at(sums, codes[:, i], targets)
                nonempty = counts > 0
                codebook[nonempty] = sums[nonempty] / counts[nonempty, None]
                residual = targets - codebook[codes[:, i]]
        return self

    def quant(self, X: np.ndarray) -> np.ndarray:
        codes = np.empty(
            (len(X), self.num_codebooks), dtype=np.min_scalar_type(self.L - 1),
        )
        residual = X.copy()
        for i, codebook in enumerate(self.codebooks):
            codes[:, i] = pairwise_distances_argmin(residual, codebook)
            residual -= codebook[codes[:, i]]
        return self._refine_codes(X, codes)

    def _refine_codes(self, X: np.ndarray, codes: np.ndarray) -> np.ndarray:
        residual = X - self.dequant(codes)
        for _ in range(self.num_search_iters):
            for i, codebook in enumerate(self.codebooks):
                residual += codebook[codes[:, i]]
                codes[:, i] = pairwise_distances_argmin(residual, codebook)
                residual -= codebook[codes[:, i]]
        return codes

    def dequant(self, codes: np.ndarray) -> np.ndarray:
        reconstructed = np.zeros(
            (len(codes), self.codebooks[0].shape[1]), dtype=np.float32,
        )
        for i, codebook in enumerate(self.codebooks):
            reconstructed += codebook[codes[:, i]]
        return reconstructed
