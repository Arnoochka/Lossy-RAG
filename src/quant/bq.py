import numpy as np
from .base import BaseQuant

class BqQuant(BaseQuant):
    def fit(self, X: np.ndarray) -> "BqQuant":
        self.dimension = X.shape[1]
        return self

    def quant(self, X: np.ndarray) -> np.ndarray:
        return np.packbits(X >= 0, axis=1)

    def dequant(self, codes: np.ndarray) -> np.ndarray:
        bits = np.unpackbits(codes, axis=1, count=self.dimension)
        return bits.astype(np.float32) * 2 - 1


class RaBitqQuant(BqQuant):
    def __init__(self, random_state: int = 42) -> None:
        super().__init__()
        self.random_state = random_state

    def fit(self, X: np.ndarray) -> "RaBitqQuant":
        self.dimension = X.shape[1]
        rng = np.random.default_rng(self.random_state)
        Q, R = np.linalg.qr(rng.standard_normal((self.dimension, self.dimension)))
        Q *= np.where(np.diag(R) < 0, -1, 1)
        self.rotation = Q.astype(np.float32)
        return self

    def quant(self, X: np.ndarray) -> np.ndarray:
        rotated = X @ self.rotation
        codes = np.empty(len(X), dtype=[
            ("bits", np.uint8, ((self.dimension + 7) // 8,)),
            ("scale", np.float32),
        ])
        codes["bits"] = np.packbits(rotated >= 0, axis=1)
        # Поправка RaBitQ: ||x||² / ||x @ rotation||₁.
        norm_squared = np.sum(X * X, axis=1)
        denominator = np.sum(np.abs(rotated), axis=1)
        codes["scale"] = np.divide(
            norm_squared, denominator,
            out=np.zeros_like(denominator), where=denominator > 0,
        )
        return codes

    def dequant(self, codes: np.ndarray) -> np.ndarray:
        bits = np.unpackbits(codes["bits"], axis=1, count=self.dimension)
        signs = bits.astype(np.float32) * 2 - 1
        return (signs * codes["scale"][:, None]) @ self.rotation.T
