from collections import deque

import numpy as np

from hedge_estimator import HedgeEstimator, Observation


class RollingOLSHedge(HedgeEstimator):
    def __init__(self, window: int = 120) -> None:
        self._window = window
        self._ys: deque = deque(maxlen=window)
        self._xs: deque = deque(maxlen=window)
        self._alpha = 0.0
        self._beta = 1.0
        self._scale = 1.0

    @property
    def name(self) -> str:
        return f"Rolling OLS({self._window})"

    def fit(self, y: np.ndarray, x: np.ndarray) -> None:
        for yt, xt in zip(y, x):
            self._ys.append(float(yt))
            self._xs.append(float(xt))
        self._refit()

    def update(self, y: float, x: float) -> Observation:
        spread = y - self._alpha - self._beta * x
        scale, beta = self._scale, self._beta
        self._ys.append(y)
        self._xs.append(x)
        self._refit()
        return Observation(spread, scale, beta)

    def _refit(self) -> None:
        ys = np.array(self._ys)
        xs = np.array(self._xs)
        design = np.column_stack([np.ones_like(xs), xs])
        coef, *_ = np.linalg.lstsq(design, ys, rcond=None)
        self._alpha, self._beta = float(coef[0]), float(coef[1])
        self._scale = float((ys - design @ coef).std(ddof=1))
