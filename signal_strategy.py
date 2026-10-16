from abc import ABC, abstractmethod


class SignalStrategy(ABC):
    @abstractmethod
    def next_position(self, z: float, current: int) -> int:
        ...


class ZScoreStrategy(SignalStrategy):
    def __init__(self, entry: float = 2.0, exit_: float = 0.5, stop: float = 4.0) -> None:
        self._entry = entry
        self._exit = exit_
        self._stop = stop

    def next_position(self, z: float, current: int) -> int:
        if abs(z) > self._stop:
            return 0
        if current == 0:
            if z > self._entry:
                return -1
            if z < -self._entry:
                return 1
            return 0
        if abs(z) < self._exit:
            return 0
        return current
