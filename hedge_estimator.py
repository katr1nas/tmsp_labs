from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Observation:
    spread: float
    scale: float
    beta: float


class HedgeEstimator(ABC):
    @abstractmethod
    def fit(self, y: np.ndarray, x: np.ndarray) -> None:
        ...

    @abstractmethod
    def update(self, y: float, x: float) -> Observation:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...
