import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances_argmin
from .base import BaseQuant

class PqQuant(BaseQuant):
    def __init__(
        self, num_parts: int, num_bits: int, random_state: int = 42) -> None:
        self.num_parts = num_parts
        self.num_bits = num_bits
        self.L = 2 ** num_bits
        self.random_state = random_state

    def fit(self, X: np.ndarray) -> "PqQuant":
        self.codebooks: list[np.ndarray] = []
        for part in np.split(X, self.num_parts, axis=1):
            kmeans = MiniBatchKMeans(
                n_clusters=self.L,
                batch_size=4096,
                n_init=1,
                max_iter=50,
                compute_labels=False,
                random_state=self.random_state,
            )
            kmeans.fit(part)
            self.codebooks.append(kmeans.cluster_centers_.astype(np.float32))
        return self

    def quant(self, X: np.ndarray) -> np.ndarray:
        parts = np.split(X, self.num_parts, axis=1)
        codes = [
            pairwise_distances_argmin(part, codebook)
            for part, codebook in zip(parts, self.codebooks)
        ]
        return np.column_stack(codes).astype(np.min_scalar_type(self.L - 1))

    def dequant(self, codes: np.ndarray) -> np.ndarray:
        parts = [
            codebook[codes[:, i]]
            for i, codebook in enumerate(self.codebooks)
        ]
        return np.concatenate(parts, axis=1)


class OpqQuant(PqQuant):
    def __init__(
        self,
        num_parts: int,
        num_bits: int,
        num_iters: int = 5,
        random_state: int = 42,
    ) -> None:
        super().__init__(num_parts, num_bits, random_state)
        self.num_iters = num_iters

    def fit(self, X: np.ndarray) -> "OpqQuant":
        self.rotation = np.eye(X.shape[1], dtype=np.float32)
        super().fit(X)
        for _ in range(self.num_iters):
            rotated = X @ self.rotation
            reconstructed = super().dequant(super().quant(rotated))
            # Для векторов-строк решаем min ||X @ R - reconstructed||².
            U, _, Vt = np.linalg.svd(X.T @ reconstructed, full_matrices=False)
            self.rotation = (U @ Vt).astype(np.float32)
            super().fit(X @ self.rotation)
        return self

    def quant(self, X: np.ndarray) -> np.ndarray:
        return super().quant(X @ self.rotation)

    def dequant(self, codes: np.ndarray) -> np.ndarray:
        return super().dequant(codes) @ self.rotation.T
