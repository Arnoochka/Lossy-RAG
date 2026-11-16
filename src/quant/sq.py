import numpy as np
from .base import BaseQuant

class SqQuant(BaseQuant):
    def __init__(self, num_bits: int) -> None:
        self.num_bits = num_bits
        self.L = 2 ** num_bits

    def fit(self, X: np.ndarray) -> "SqQuant":
        self.min_value: float = float(np.min(X))
        self.max_value: float = float(np.max(X))
        self.step: float = (
            (self.max_value - self.min_value) / (self.L - 1)
            if self.max_value > self.min_value else 1.0
        )
        return self

    def quant(self, X: np.ndarray) -> np.ndarray:
        codes = np.clip(
            np.round((X - self.min_value) / self.step), 0, self.L - 1,
        )
        return codes.astype(np.min_scalar_type(self.L - 1))

    def dequant(self, codes: np.ndarray) -> np.ndarray:
        return codes.astype(np.float32) * self.step + self.min_value
