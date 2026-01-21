from typing import Any, Callable
from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory

import numpy as np
import torch as th

from gymnasium.spaces import Space

from .utils import (
    create_obs_buffer,
    append_to_obs_buffer,
    obs_buffer_to_array,
    get_obs_bytes,
    get_obs_shape,
    make_shm_from_obs_array,
    make_obs_with_shm,
    make_obs_with_shape_dtype_shm,
    copy_obs,
    close_obs_shm,
    unlink_obs_shm,
    slice_obs_array,
    obs_to_tensor,
    obs_to_device,
    ObsBuffer,
    ObsType,
    ObsBufferArray,
    ObsArrayBytes,
    ObsSharedMemory,
    ObsArrayShape,
    ObsDtype,
    ObsTensor,
)
from .actor_critic import ActorCritic
from .ppo_batch import PPOBatch


@dataclass
class RolloutDtypes:
    obs: ObsDtype
    act: np.dtype
    reward: np.dtype
    truncated: np.dtype
    done: np.dtype
    logp: np.dtype


@dataclass
class RolloutShape:
    obs: ObsArrayShape
    act: tuple[int, ...]
    reward: tuple[int, ...]
    truncated: tuple[int, ...]
    done: tuple[int, ...]
    logp: tuple[int, ...]


class RolloutBuffer:
    """
    A buffer to store rollouts
    """

    def __init__(
        self,
        obs_space: Space,
    ):
        self.obs_space = obs_space

    def init(self):
        """Initialize the buffer"""
        self.obs_buffer = create_obs_buffer(self.obs_space)
        self.act_buffer: list[int | np.ndarray] = []
        self.reward_buffer = []
        self.truncated_buffer = []
        self.done_buffer = []
        self.logp_buffer = []

    def add(
        self,
        obs: ObsType,
        act: int | np.ndarray,
        reward: float,
        truncated: bool,
        done: bool,
        logp: float,
    ):
        append_to_obs_buffer(self.obs_buffer, obs)
        self.act_buffer.append(act)
        self.reward_buffer.append(reward)
        self.truncated_buffer.append(truncated)
        self.done_buffer.append(done)
        self.logp_buffer.append(logp)

    def add_last_obs(self, next_obs: ObsType):
        """Add the last observation"""
        append_to_obs_buffer(self.obs_buffer, next_obs)

    @property
    def size(self) -> int:
        return len(self.reward_buffer)

    def clear(self):
        """Clear the buffer"""
        del self.obs_buffer
        del self.act_buffer
        del self.reward_buffer
        del self.truncated_buffer
        del self.done_buffer
        del self.logp_buffer

        self.init()

    def __len__(self) -> int:
        return self.size

    def to_numpy(self) -> tuple[ObsBufferArray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Convert the buffer to numpy arrays"""
        return (
            obs_buffer_to_array(self.obs_buffer),
            np.array(self.act_buffer),
            np.array(self.reward_buffer, dtype=np.float32),
            np.array(self.truncated_buffer, dtype=bool),
            np.array(self.done_buffer, dtype=bool),
            np.array(self.logp_buffer, dtype=np.float32),
        )


class RolloutShmHandles:
    """
    A container for shared memory handles of a RolloutBufferArray
    """

    def __init__(
        self,
        obs_shm: ObsSharedMemory,
        act_shm: SharedMemory,
        reward_shm: SharedMemory,
        truncated_shm: SharedMemory,
        done_shm: SharedMemory,
        logp_shm: SharedMemory,
    ):
        self.obs_shm = obs_shm
        self.act_shm = act_shm
        self.reward_shm = reward_shm
        self.truncated_shm = truncated_shm
        self.done_shm = done_shm
        self.logp_shm = logp_shm

    def close(self):
        """Close the shared memory handles"""
        close_obs_shm(self.obs_shm)
        self.act_shm.close()
        self.reward_shm.close()
        self.truncated_shm.close()
        self.done_shm.close()
        self.logp_shm.close()

    def unlink(self):
        """Unlink the shared memory handles"""
        unlink_obs_shm(self.obs_shm)
        self.act_shm.unlink()
        self.reward_shm.unlink()
        self.truncated_shm.unlink()
        self.done_shm.unlink()
        self.logp_shm.unlink()


class RolloutBufferArray:
    """
    A buffer to store rollouts in numpy arrays
    """

    def __init__(
        self,
        obs_array: ObsBufferArray,
        act_array: np.ndarray,
        reward_array: np.ndarray,
        truncated_array: np.ndarray,
        done_array: np.ndarray,
        logp_array: np.ndarray,
    ):
        self.obs_array = obs_array
        self.act_array = act_array
        self.reward_array = reward_array
        self.truncated_array = truncated_array
        self.done_array = done_array
        self.logp_array = logp_array

    @property
    def size(self) -> int:
        return len(self.reward_array)

    def __len__(self) -> int:
        return self.size

    def shmemory_size(self) -> tuple[ObsArrayBytes, int, int, int, int, int]:
        """Calculate the size of the buffer in shared memory (in bytes)"""
        return (
            get_obs_bytes(self.obs_array),
            self.act_array.nbytes,
            self.reward_array.nbytes,
            self.truncated_array.nbytes,
            self.done_array.nbytes,
            self.logp_array.nbytes,
        )

    def get_shapes(
        self,
    ) -> tuple[ObsArrayShape, tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        """Get the shapes of the buffer arrays"""
        return (
            get_obs_shape(self.obs_array),
            self.act_array.shape,
            self.reward_array.shape,
            self.truncated_array.shape,
            self.done_array.shape,
            self.logp_array.shape,
        )

    def get_dtypes(
        self,
    ) -> tuple[ObsDtype, np.dtype, np.dtype, np.dtype, np.dtype, np.dtype]:
        """Get the dtypes of the buffer arrays"""

        def get_obs_dtype(buffer: ObsBufferArray) -> ObsDtype:
            if isinstance(buffer, dict):
                return {key: get_obs_dtype(subbuffer) for key, subbuffer in buffer.items()}
            elif isinstance(buffer, tuple):
                return tuple(get_obs_dtype(subbuffer) for subbuffer in buffer)
            else:
                return buffer.dtype

        return (
            get_obs_dtype(self.obs_array),
            self.act_array.dtype,
            self.reward_array.dtype,
            self.truncated_array.dtype,
            self.done_array.dtype,
            self.logp_array.dtype,
        )

    def make_shmemory(self):
        """Create shared memory versions of the buffer arrays"""
        obs_shm = make_shm_from_obs_array(get_obs_bytes(self.obs_array))
        act_shm = SharedMemory(create=True, size=self.act_array.nbytes)
        reward_shm = SharedMemory(create=True, size=self.reward_array.nbytes)
        truncated_shm = SharedMemory(create=True, size=self.truncated_array.nbytes)
        done_shm = SharedMemory(create=True, size=self.done_array.nbytes)
        logp_shm = SharedMemory(create=True, size=self.logp_array.nbytes)

        shm_obs_array = make_obs_with_shm(self.obs_array, obs_shm)
        shm_act_array = np.ndarray(self.act_array.shape, dtype=self.act_array.dtype, buffer=act_shm.buf)
        shm_reward_array = np.ndarray(self.reward_array.shape, dtype=self.reward_array.dtype, buffer=reward_shm.buf)
        shm_truncated_array = np.ndarray(self.truncated_array.shape, dtype=self.truncated_array.dtype, buffer=truncated_shm.buf)
        shm_done_array = np.ndarray(self.done_array.shape, dtype=self.done_array.dtype, buffer=done_shm.buf)
        shm_logp_array = np.ndarray(self.logp_array.shape, dtype=self.logp_array.dtype, buffer=logp_shm.buf)

        copy_obs(self.obs_array, shm_obs_array)
        np.copyto(shm_act_array, self.act_array)
        np.copyto(shm_reward_array, self.reward_array)
        np.copyto(shm_truncated_array, self.truncated_array)
        np.copyto(shm_done_array, self.done_array)
        np.copyto(shm_logp_array, self.logp_array)

        return (
            obs_shm,
            act_shm,
            reward_shm,
            truncated_shm,
            done_shm,
            logp_shm,
        )

    def clear(self):
        """Clear the buffer arrays from memory"""
        del self.obs_array
        del self.act_array
        del self.reward_array
        del self.truncated_array
        del self.done_array
        del self.logp_array

    @classmethod
    def from_shm(
        cls,
        shms: RolloutShmHandles,
        shapes: RolloutShape,
        dtypes: RolloutDtypes,
    ):
        """Create a RolloutBufferArray from shared memory"""
        obs_array = make_obs_with_shape_dtype_shm(shapes.obs, dtypes.obs, shms.obs_shm)
        act_array = np.ndarray(shapes.act, dtype=dtypes.act, buffer=shms.act_shm.buf)
        reward_array = np.ndarray(shapes.reward, dtype=dtypes.reward, buffer=shms.reward_shm.buf)
        truncated_array = np.ndarray(shapes.truncated, dtype=dtypes.truncated, buffer=shms.truncated_shm.buf)
        done_array = np.ndarray(shapes.done, dtype=dtypes.done, buffer=shms.done_shm.buf)
        logp_array = np.ndarray(shapes.logp, dtype=dtypes.logp, buffer=shms.logp_shm.buf)
        return cls(
            obs_array,
            act_array,
            reward_array,
            truncated_array,
            done_array,
            logp_array,
        )


def clone_obs(obs: ObsTensor) -> ObsTensor:
    """Clone an observation array."""
    if isinstance(obs, dict):
        return {key: clone_obs(subobs) for key, subobs in obs.items()}
    elif isinstance(obs, tuple):
        return tuple(clone_obs(subobs) for subobs in obs)
    else:
        return obs.clone()


def compute_gae(
    rollout: RolloutBufferArray,
    gamma: float,
    lam: float,
    value_batch_size: int,
    ac: ActorCritic,
) -> PPOBatch:
    value = np.zeros(rollout.size + 1, dtype=np.float32)
    with th.no_grad():
        obs = rollout.obs_array
        for start in range(0, rollout.size + 1, value_batch_size):
            end = min(start + value_batch_size, rollout.size)
            obs_batch = slice_obs_array(obs, start, end)
            i = obs_to_device(obs_to_tensor(obs_batch), ac.device)
            value[start:end] = ac.only_value(i).cpu().numpy()

    adv = np.zeros(rollout.size, dtype=np.float32)
    last_gae_lam = 0.0
    for t in reversed(range(rollout.size)):
        nonterminal = 1.0 - float(rollout.done_array[t])
        delta = rollout.reward_array[t] + gamma * value[t + 1] * nonterminal - value[t]
        adv[t] = last_gae_lam = delta + gamma * lam * nonterminal * last_gae_lam
    ret = adv + value[:-1]

    return PPOBatch(
        obs=clone_obs(obs_to_tensor(slice_obs_array(obs, 0, rollout.size))),
        act=th.from_numpy(rollout.act_array).clone(),
        logp_old=th.from_numpy(rollout.logp_array).clone(),
        adv=th.from_numpy(adv).clone(),
        ret=th.from_numpy(ret).clone(),
        val_old=th.from_numpy(value[:-1]).clone(),
    )


Callback = (
    Callable[[RolloutBufferArray], Any]
    | list[Callable[[RolloutBufferArray], Any]]
    | dict[str, Callable[[RolloutBufferArray], Any]]
)


def log_from_array(rollout: RolloutBufferArray, callback: Callback) -> Any:
    if isinstance(callback, list):
        return [log_from_array(rollout, cb) for cb in callback]
    elif isinstance(callback, dict):
        return {key: log_from_array(rollout, cb) for key, cb in callback.items()}
    else:
        return callback(rollout)
