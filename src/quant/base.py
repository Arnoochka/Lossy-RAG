import numpy as np

class BaseQuant:
    def fit(self, X: np.ndarray) -> "BaseQuant":
        return self

    def quant(self, X: np.ndarray) -> np.ndarray:
        return X

    def dequant(self, codes: np.ndarray) -> np.ndarray:
        return codes