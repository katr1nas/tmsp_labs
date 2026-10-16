import numpy as np

from hedge_estimator import HedgeEstimator, Observation


class KalmanHedge(HedgeEstimator):
    def __init__(self, process_noise: float = 1e-5, observation_noise: float = 1e-3) -> None:
        self._q = process_noise
        self._r = observation_noise
        self._theta = np.array([0.0, 1.0])
        self._p = np.eye(2)

    @property
    def name(self) -> str:
        return "Kalman"

    def fit(self, y: np.ndarray, x: np.ndarray) -> None:
        design = np.column_stack([np.ones_like(x), x])
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        self._theta = coef.astype(float)
        self._p = np.eye(2) * 1e-2
        for yt, xt in zip(y, x):
            self.update(float(yt), float(xt))

    def update(self, y: float, x: float) -> Observation:
        h = np.array([1.0, x])
        p_pred = self._p + np.eye(2) * self._q
        innovation = y - h @ self._theta
        s = float(h @ p_pred @ h + self._r)
        gain = p_pred @ h / s
        self._theta = self._theta + gain * innovation
        self._p = p_pred - np.outer(gain, h) @ p_pred
        return Observation(float(innovation), float(np.sqrt(s)), float(self._theta[1]))
