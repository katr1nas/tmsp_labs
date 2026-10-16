# Adaptive Hedge Ratio Engine - SOLID Principles Implementation

Laboratory Work 1

## Table of Contents

- Introduction
- Features
- SOLID Principles Applied
  - Single Responsibility Principle (SRP)
  - Open/Closed Principle (OCP)
  - Dependency Inversion Principle (DIP)
- Project Structure
  - Directory Breakdown
- Usage
  - Running the Application
- Results
- Limitations
- Conclusion
- Author

## Introduction

This project is developed as part of Laboratory Work 1 and demonstrates three SOLID principles - Single Responsibility (SRP), Open/Closed (OCP) and Dependency Inversion (DIP) - on a pairs-trading backtesting engine taken from my research project `adaptive-hedge-ratio`.

The engine models the relationship between two log-price series:

    log(A_t) = alpha_t + beta_t * log(B_t) + epsilon_t

and compares different ways of estimating the hedge ratio beta: a static OLS estimate, a Kalman filter and a rolling OLS window. The trading spread is the residual (innovation) of the model, traded with a z-score entry/exit/stop rule.

## Features

- Static OLS Hedge: beta estimated once on the training period and kept fixed.
- Kalman Hedge: alpha and beta are latent states updated on every observation.
- Rolling OLS Hedge: beta re-estimated on a sliding window.
- Extensibility: a new estimator, signal rule or data source is added without modifying existing code.
- Modularity: data loading, estimation, signal generation, backtesting, metrics and logging are separate modules.
- Logging: every backtest run is logged with its pair, method, Sharpe ratio and trade count.
- Configuration Management: all thresholds, costs and parameters live in one `BacktestConfig`.

## SOLID Principles Applied

### Single Responsibility Principle (SRP)

- Definition: A class should have only one reason to change.
- Implementation in the Project:
  - `StaticOLSHedge`, `KalmanHedge`, `RollingOLSHedge`: each only estimates the hedge ratio with its own method.
  - `ZScoreStrategy`: only converts a z-score into a position.
  - `Backtester`: only orchestrates the train/test run and PnL calculation.
  - `compute_performance` (metrics.py): only computes Sharpe, return, drawdown and trade count.
  - `TradeLogger`: only handles logging.
  - `SyntheticPairSource`, `CsvPairSource`: only load data.

```python
class StaticOLSHedge(HedgeEstimator):
    def fit(self, y, x):
        design = np.column_stack([np.ones_like(x), x])
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        self._alpha, self._beta = float(coef[0]), float(coef[1])
        residual = y - self._alpha - self._beta * x
        self._scale = float(residual.std(ddof=1))

    def update(self, y, x):
        spread = y - self._alpha - self._beta * x
        return Observation(spread, self._scale, self._beta)
```

### Open/Closed Principle (OCP)

- Definition: Software entities should be open for extension but closed for modification.
- Implementation in the Project:
  - `HedgeEstimator` interface defines `fit`, `update` and `name`.
  - `RollingOLSHedge` was added after `StaticOLSHedge` and `KalmanHedge` by only implementing this interface; `Backtester`, `ZScoreStrategy` and the metrics were not touched.
  - The same applies to `SignalStrategy` (new trading rules) and `DataSource` (new datasets).

```python
class HedgeEstimator(ABC):
    @abstractmethod
    def fit(self, y, x): ...

    @abstractmethod
    def update(self, y, x) -> Observation: ...

    @property
    @abstractmethod
    def name(self) -> str: ...
```

### Dependency Inversion Principle (DIP)

- Definition: High-level modules should depend on abstractions, not on low-level modules.
- Implementation in the Project:
  - `Backtester` (high-level) depends only on `HedgeEstimator`, `SignalStrategy`, `DataSource` and `TradeLogger`, never on `KalmanHedge` or `ZScoreStrategy` directly.
  - Dependencies are injected through the constructor; `main.py` is the only place where concrete classes are chosen.

```python
class Backtester:
    def __init__(self, estimator: HedgeEstimator, strategy: SignalStrategy,
                 config: BacktestConfig, logger: TradeLogger) -> None:
        self._estimator = estimator
        self._strategy = strategy
        self._config = config
        self._logger = logger
```

## Project Structure

### Directory Breakdown

```
Lab1/
├── hedge_estimator.py     # Abstract HedgeEstimator + Observation, follows DIP/OCP
├── static_hedge.py        # SRP-compliant, static OLS estimator
├── kalman_hedge.py        # SRP-compliant, Kalman filter estimator
├── rolling_hedge.py       # SRP-compliant, rolling OLS estimator (added via OCP)
├── signal_strategy.py     # Abstract SignalStrategy + ZScoreStrategy
├── data_source.py         # Abstract DataSource + synthetic and CSV sources
├── backtester.py          # High-level backtest logic, depends on abstractions
├── metrics.py             # Performance metrics
├── trade_logger.py        # Logging
├── config.py              # Centralized configuration
├── main.py                # Entry point, composition of concrete classes
└── requirements.txt       # Dependencies
```

## Usage

### Running the Application

```bash
pip install -r requirements.txt
python main.py
```

Example output:

```
10:49:05 | INFO | synthetic-1 | Static OLS | sharpe=-0.24 trades=1
10:49:05 | INFO | synthetic-1 | Kalman | sharpe=0.07 trades=14
10:49:05 | INFO | synthetic-1 | Rolling OLS(120) | sharpe=0.38 trades=10
...
Method             Mean Sharpe
Static OLS                0.05
Kalman                    0.20
Rolling OLS(120)          0.48

Kalman beats Static on 4 of 5 pairs
```

To use real data instead of the synthetic generator, replace `SyntheticPairSource(seed)` in `main.py` with `CsvPairSource("prices.csv", "UPS", "FDX")`.

## Results

Setup: 1500 observations per pair, 60/40 train/test split, entry |z| > 2, exit |z| < 0.5, stop |z| > 4, transaction cost 0.0005, 5 synthetic pairs in which the true beta drifts as a random walk.

| Method           | Mean out-of-sample Sharpe |
| ---------------- | ------------------------- |
| Static OLS       | 0.05                      |
| Kalman           | 0.20                      |
| Rolling OLS(120) | 0.48                      |

The Kalman estimator beat the static one on 4 of 5 pairs. The rolling window was the best method on this data.

## Limitations

- The data is synthetic (the sandbox used for generation had no market data access), so the numbers demonstrate the architecture, not a trading edge.
- Only 5 pairs, so the differences between methods are not statistically established.
- Kalman noise parameters (process 1e-6, observation 5e-4) were chosen so that the innovation z-score has a standard deviation close to 1. With the initial values (1e-5, 1e-3) the Kalman strategy opened no trades at all, which shows how sensitive the method is to these parameters.

## Conclusion

The project shows how SOLID principles structure a quantitative research codebase. Each estimator, the signal rule, the data source, the metrics and the logger have a single responsibility. The rolling OLS estimator was added without changing any existing class, which confirms the Open/Closed Principle. The backtester depends only on abstractions injected from `main.py`, so estimators and strategies can be swapped freely. The experiment also shows that an adaptive hedge ratio is not automatically better: its result depends strongly on the chosen parameters and must be validated on real data.

## Author

Serghei Barladean FAF-241
