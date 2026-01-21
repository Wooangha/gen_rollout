from typing import Any, Callable

import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.queues import Queue
from multiprocessing.sharedctypes import Synchronized

from dataclasses import dataclass

import numpy as np
import torch as th

import gymnasium as gym

from .rollout import RolloutBuffer, RolloutBufferArray, RolloutShmHandles, compute_gae
from .ppo_batch import PPOBatch
from .actor_critic import ActorCritic
from .config import WorkerConfig
from .utils import ObsArrayShape, ObsShmName, ObsSharedMemory, ObsDtype, get_obs_shm_name
from .rollout import RolloutShape, RolloutDtypes


@dataclass
class RolloutShmName:
    obs: ObsShmName
    act: str
    reward: str
    truncated: str
    done: str
    logp: str


def make_rollout_shm_name(rollout_shm: RolloutShmHandles) -> RolloutShmName:
    """Convert RolloutShmHandles to RolloutShmName."""
    return RolloutShmName(
        obs=get_obs_shm_name(rollout_shm.obs_shm),
        act=rollout_shm.act_shm.name,
        reward=rollout_shm.reward_shm.name,
        truncated=rollout_shm.truncated_shm.name,
        done=rollout_shm.done_shm.name,
        logp=rollout_shm.logp_shm.name,
    )


Callback = (
    Callable[[RolloutBufferArray], Any]
    | list[Callable[[RolloutBufferArray], Any]]
    | dict[str, Callable[[RolloutBufferArray], Any]]
)
Log = Any | list["Log"] | dict[str, "Log"]


def log_from_array(rollout: RolloutBufferArray, callback: Callback) -> Log:
    if isinstance(callback, list):
        return [log_from_array(rollout, cb) for cb in callback]
    elif isinstance(callback, dict):
        return {key: log_from_array(rollout, cb) for key, cb in callback.items()}
    else:
        return callback(rollout)


@dataclass
class RolloutMetadata:
    version: int
    rollout_shm_names: RolloutShmName
    rollout_shapes: RolloutShape
    rollout_dtypes: RolloutDtypes
    info: Log


def obs_shm_name2obs_shm(name: ObsShmName) -> ObsSharedMemory:
    """Convert shared memory names to SharedMemory handles."""
    if isinstance(name, dict):
        return {key: obs_shm_name2obs_shm(subname) for key, subname in name.items()}
    elif isinstance(name, tuple):
        return tuple(obs_shm_name2obs_shm(subname) for subname in name)
    else:
        return SharedMemory(name=name)


def make_shm_from_shm_name(rollout_shm_name: RolloutShmName) -> RolloutShmHandles:
    """Convert RolloutShmName to RolloutShmHandles."""
    return RolloutShmHandles(
        obs_shm=obs_shm_name2obs_shm(rollout_shm_name.obs),
        act_shm=SharedMemory(name=rollout_shm_name.act),
        reward_shm=SharedMemory(name=rollout_shm_name.reward),
        truncated_shm=SharedMemory(name=rollout_shm_name.truncated),
        done_shm=SharedMemory(name=rollout_shm_name.done),
        logp_shm=SharedMemory(name=rollout_shm_name.logp),
    )


def gen_rollout(
    worker_id: int,
    queue: Queue,
    model: ActorCritic,
    config: WorkerConfig,
    log_callback: Callback,
    version: Synchronized[int],
):
    env = config.build_env(worker_id)
    model_copy: ActorCritic = config.model(**config.model_kwargs)
    now_model_version = -1
    while True:
        try:
            with version.get_lock():
                if now_model_version < version.value:
                    now_model_version = version.value
                    model_copy.load_state_dict(model.state_dict())

            model_copy.eval()
            rollout_buffer = RolloutBuffer(config.obs_space)
            with th.no_grad():
                obs, info = env.reset()
                done = False
                while not done:
                    a, logp, _ = model_copy.act(obs)
                    a = a.cpu().numpy()
                    next_obs, reward, terminated, truncated, info = env.step(a)
                    done = terminated or truncated
                    rollout_buffer.add(obs, a, reward, truncated, terminated, logp.cpu().numpy())  # type: ignore
                    obs = next_obs

                rollout_buffer.add_last_obs(next_obs)

            # it is possible that model is updated during rollout generation
            with version.get_lock():
                if now_model_version < version.value:
                    continue

            rollout_array = RolloutBufferArray(*rollout_buffer.to_numpy())
            rsh = RolloutShmHandles(*rollout_array.make_shmemory())
            metadata = RolloutMetadata(
                now_model_version,
                make_rollout_shm_name(rsh),
                RolloutShape(*rollout_array.get_shapes()),
                RolloutDtypes(*rollout_array.get_dtypes()),
                log_from_array(rollout_array, log_callback),
            )
            queue.put(metadata)
            rsh.close()
        except Exception as e:
            print(f"Worker {worker_id} encountered an error: {e}")


class RolloutGenerator:
    """
    gpu_model has to be training model reference!
    """

    def __init__(
        self,
        n_worker: int,
        cpu_model: ActorCritic,
        gpu_model: ActorCritic,
        max_queue: int,
        log_callback: Callback,
        config: WorkerConfig,
    ) -> None:
        workers = []
        cpu_model.share_memory()
        version = mp.Value("i", 0)
        queue: Queue[RolloutMetadata] = mp.Queue(max_queue)
        for worker_id in range(n_worker):
            p = mp.Process(target=gen_rollout, args=(worker_id, queue, cpu_model, config, log_callback, version))
            p.start()
            workers.append(p)

        self.cpu_model = cpu_model
        self.gpu_model = gpu_model
        self.version = version
        self.queue = queue
        self.workers = workers

    def collect_rollout(
        self,
        ts_per_iteration: int,
        gamma: float,
        lam: float,
        value_batch_size: int,
    ) -> PPOBatch:
        timestep = 0

        self.gpu_model.eval()

        ppo_batches = []

        while timestep < ts_per_iteration:
            metadata = self.queue.get()
            shm_handler = make_shm_from_shm_name(metadata.rollout_shm_names)
            with self.version.get_lock():
                cur_ver = self.version.value

            if metadata.version < cur_ver:
                shm_handler.close()
                shm_handler.unlink()
                continue
            timestep += metadata.rollout_shapes.act[0]
            try:
                ppo_batches.append(
                    compute_gae(
                        RolloutBufferArray.from_shm(
                            shm_handler,
                            metadata.rollout_shapes,
                            metadata.rollout_dtypes,
                        ),
                        gamma=gamma,
                        lam=lam,
                        value_batch_size=value_batch_size,
                        ac=self.gpu_model,
                    )
                )
            finally:
                shm_handler.close()
                shm_handler.unlink()

        return PPOBatch.concat(ppo_batches)

    def update_model(self):
        with self.version.get_lock():
            self.version.value += 1
            self.cpu_model.load_state_dict(self.gpu_model.state_dict())
            self.cpu_model.to("cpu")
