from typing import Callable, Optional, Type
from dataclasses import dataclass

import gymnasium as gym

from gymnasium.spaces import Space

from .actor_critic import ActorCritic


@dataclass
class WorkerConfig:
    obs_space: Space

    def build_model(self) -> ActorCritic: ...

    def build_env(self, worker_id: int) -> gym.Env: ...
