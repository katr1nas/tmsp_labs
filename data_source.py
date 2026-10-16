from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class DataSource(ABC):
    @abstractmethod
    def load(self) -> Tuple[np.ndarray, np.ndarray]:
        ...

    @property
    @abstractmethod
    def label(self) -> str:
        ...


class SyntheticPairSource(DataSource):
    def __init__(self, seed: int, n: int = 1500, beta0: float = 1.0, beta_vol: float = 0.003) -> None:
        self._seed = seed
        self._n = n
        self._beta0 = beta0
        self._beta_vol = beta_vol

    @property
    def label(self) -> str:
        return f"synthetic-{self._seed}"

    def load(self) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(self._seed)
        x = np.cumsum(rng.normal(0.0003, 0.01, self._n)) + 4.0
        beta = self._beta0 + np.cumsum(rng.normal(0.0, self._beta_vol, self._n))
        noise = np.zeros(self._n)
        for t in range(1, self._n):
            noise[t] = 0.93 * noise[t - 1] + rng.normal(0.0, 0.01)
        y = 0.2 + beta * x + noise
        return y, x


class CsvPairSource(DataSource):
    def __init__(self, path: str, column_a: str, column_b: str) -> None:
        self._path = path
        self._a = column_a
        self._b = column_b

    @property
    def label(self) -> str:
        return f"{self._a}/{self._b}"

    def load(self) -> Tuple[np.ndarray, np.ndarray]:
        import csv

        rows = list(csv.DictReader(open(self._path, newline="")))
        y = np.log(np.array([float(r[self._a]) for r in rows]))
        x = np.log(np.array([float(r[self._b]) for r in rows]))
        return y, x
