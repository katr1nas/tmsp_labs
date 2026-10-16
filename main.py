import numpy as np

from backtester import Backtester
from config import BacktestConfig
from data_source import SyntheticPairSource
from hedge_estimator import HedgeEstimator
from kalman_hedge import KalmanHedge
from rolling_hedge import RollingOLSHedge
from signal_strategy import ZScoreStrategy
from static_hedge import StaticOLSHedge
from trade_logger import TradeLogger


def build_estimators(config: BacktestConfig) -> list:
    return [
        StaticOLSHedge(),
        KalmanHedge(config.kalman_process_noise, config.kalman_observation_noise),
        RollingOLSHedge(config.rolling_window),
    ]


def main() -> None:
    config = BacktestConfig()
    logger = TradeLogger()
    strategy = ZScoreStrategy(config.entry_z, config.exit_z, config.stop_z)

    scores: dict = {}
    for seed in config.pair_seeds:
        source = SyntheticPairSource(seed)
        for estimator in build_estimators(config):
            result = Backtester(estimator, strategy, config, logger).run(source)
            scores.setdefault(estimator.name, []).append(result.sharpe)

    print()
    print(f"{'Method':<18}{'Mean Sharpe':>12}")
    for name, values in scores.items():
        print(f"{name:<18}{np.mean(values):>12.2f}")

    static = scores["Static OLS"]
    kalman = scores["Kalman"]
    wins = sum(k > s for k, s in zip(kalman, static))
    print(f"\nKalman beats Static on {wins} of {len(static)} pairs")


if __name__ == "__main__":
    main()
