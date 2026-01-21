from typing import Callable
from dataclasses import dataclass

import gymnasium as gym

from gymnasium.spaces import Space

from .actor_critic import ActorCritic

from gym_wrap import ActionParser, ObsBuilder, TruncateCondition, DoneCondition, RewardFn


@dataclass
class WorkerConfig:
    model: ActorCritic
    model_kwargs: dict
    obs_space: Space

    obs_builder: ObsBuilder
    action_parser: ActionParser
    done_condition: DoneCondition
    truncate_condition: TruncateCondition
    reward_fn: RewardFn

    def build_env(self, worker_id: int) -> gym.Env: ...
