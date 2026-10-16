import numpy as np

from hedge_estimator import HedgeEstimator, Observation


class StaticOLSHedge(HedgeEstimator):
    def __init__(self) -> None:
        self._alpha = 0.0
        self._beta = 1.0
        self._scale = 1.0

    @property
    def name(self) -> str:
        return "Static OLS"

    def fit(self, y: np.ndarray, x: np.ndarray) -> None:
        design = np.column_stack([np.ones_like(x), x])
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        self._alpha, self._beta = float(coef[0]), float(coef[1])
        residual = y - self._alpha - self._beta * x
        self._scale = float(residual.std(ddof=1))

    def update(self, y: float, x: float) -> Observation:
        spread = y - self._alpha - self._beta * x
        return Observation(spread, self._scale, self._beta)
