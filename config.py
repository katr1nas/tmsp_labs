from dataclasses import dataclass


@dataclass(frozen=True)
class BacktestConfig:
    train_ratio: float = 0.6
    transaction_cost: float = 0.0005
    entry_z: float = 2.0
    exit_z: float = 0.5
    stop_z: float = 4.0
    kalman_process_noise: float = 1e-6
    kalman_observation_noise: float = 5e-4
    rolling_window: int = 120
    pair_seeds: tuple = (1, 2, 3, 4, 5)
