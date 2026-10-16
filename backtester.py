import numpy as np

from config import BacktestConfig
from data_source import DataSource
from hedge_estimator import HedgeEstimator
from metrics import Performance, compute_performance
from signal_strategy import SignalStrategy
from trade_logger import TradeLogger


class Backtester:
    def __init__(
        self,
        estimator: HedgeEstimator,
        strategy: SignalStrategy,
        config: BacktestConfig,
        logger: TradeLogger,
    ) -> None:
        self._estimator = estimator
        self._strategy = strategy
        self._config = config
        self._logger = logger

    def run(self, source: DataSource) -> Performance:
        y, x = source.load()
        split = int(len(y) * self._config.train_ratio)
        self._estimator.fit(y[:split], x[:split])

        n_test = len(y) - split
        positions = np.zeros(n_test, dtype=int)
        betas = np.zeros(n_test)
        current = 0
        for i in range(n_test):
            obs = self._estimator.update(float(y[split + i]), float(x[split + i]))
            z = obs.spread / obs.scale
            current = self._strategy.next_position(z, current)
            positions[i] = current
            betas[i] = obs.beta

        ret_y = np.diff(y[split - 1:])
        ret_x = np.diff(x[split - 1:])
        prev_pos = np.concatenate([[0], positions[:-1]])
        prev_beta = np.concatenate([[betas[0]], betas[:-1]])
        gross = prev_pos * (ret_y - prev_beta * ret_x)
        turnover = np.abs(positions - prev_pos) * (1.0 + np.abs(betas))
        pnl = gross - self._config.transaction_cost * turnover

        result = compute_performance(pnl, positions)
        self._logger.info(
            f"{source.label} | {self._estimator.name} | sharpe={result.sharpe:.2f} trades={result.trades}"
        )
        return result
