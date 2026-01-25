from abc import ABC, abstractmethod
from typing import Any

from .rollout import RolloutBufferArray


class RolloutCallback(ABC):
    @staticmethod
    def make_statistics(others: list[Any]) -> Any:
        """statistics from multiple rollouts

        Args:
            others (list[Any]): _list of results from multiple rollouts_

        Raises:
            NotImplementedError: _if not implemented_

        Returns:
            Any: _statistics_
        """
        raise NotImplementedError

    @abstractmethod
    def on_rollout_end(self, rollout: RolloutBufferArray) -> Any:
        raise NotImplementedError


class EpisodeLengthCallback(RolloutCallback):
    @staticmethod
    def make_statistics(others: list[int]) -> Any:
        return sum(others) / len(others)

    def on_rollout_end(self, rollout: RolloutBufferArray) -> int:
        return rollout.reward_array.shape[0]


class TotalRewardCallback(RolloutCallback):
    @staticmethod
    def make_statistics(others: list[float]) -> Any:
        return sum(others) / len(others)

    def on_rollout_end(self, rollout: RolloutBufferArray) -> float:
        return rollout.reward_array.sum()


class SuccessRateCallback(RolloutCallback):
    @staticmethod
    def make_statistics(others: list[bool]) -> Any:
        return sum(others) / len(others)

    def on_rollout_end(self, rollout: RolloutBufferArray) -> Any:
        return rollout.terminated_array[-1]
