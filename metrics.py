from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Performance:
    sharpe: float
    total_return: float
    max_drawdown: float
    trades: int


def compute_performance(pnl: np.ndarray, positions: np.ndarray) -> Performance:
    std = pnl.std(ddof=1)
    sharpe = float(pnl.mean() / std * np.sqrt(252)) if std > 0 else 0.0
    equity = np.cumsum(pnl)
    drawdown = float((equity - np.maximum.accumulate(equity)).min())
    entries = int(np.sum((positions[1:] != 0) & (positions[:-1] == 0)) + (positions[0] != 0))
    return Performance(sharpe, float(equity[-1]), drawdown, entries)
