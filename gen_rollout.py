import os
from typing import Any, Callable
import time
import random

import torch.multiprocessing as mp
from torch.multiprocessing.queue import Queue
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.sharedctypes import Synchronized

from dataclasses import dataclass

import numpy as np
import torch as th
import tqdm

import gymnasium as gym

from .rollout import RolloutBuffer, RolloutBufferArray, RolloutShmHandles, compute_gae
from .ppo_batch import PPOBatch
from .actor_critic import ActorCritic
from .config import WorkerConfig
from .utils import ObsArrayShape, ObsShmName, ObsSharedMemory, ObsDtype, get_obs_shm_name
from .rollout import RolloutShape, RolloutDtypes
from .callback import RolloutCallback
from stable_baselines3.common.vec_env import SubprocVecEnv


@dataclass
class RolloutShmName:
    obs: ObsShmName
    act: str
    reward: str
    truncated: str
    terminated: str
    logp: str


def make_rollout_shm_name(rollout_shm: RolloutShmHandles) -> RolloutShmName:
    """Convert RolloutShmHandles to RolloutShmName."""
    return RolloutShmName(
        obs=get_obs_shm_name(rollout_shm.obs_shm),
        act=rollout_shm.act_shm.name,
        reward=rollout_shm.reward_shm.name,
        truncated=rollout_shm.truncated_shm.name,
        terminated=rollout_shm.terminated_shm.name,
        logp=rollout_shm.logp_shm.name,
    )


LogCallback = RolloutCallback | dict[str, RolloutCallback]
Log = Any | dict[str, Any]


def log_from_array(rollout: RolloutBufferArray, callback: LogCallback) -> Log:
    if isinstance(callback, dict):
        return {key: log_from_array(rollout, cb) for key, cb in callback.items()}
    else:
        return callback.on_rollout_end(rollout)


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
        terminated_shm=SharedMemory(name=rollout_shm_name.terminated),
        logp_shm=SharedMemory(name=rollout_shm_name.logp),
    )


def set_seed(seed: int):
    random.seed(seed)

    np.random.seed(seed)

    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)


def gen_rollout(
    worker_id: int,
    queue: Queue,
    model: ActorCritic,
    config: WorkerConfig,
    log_callback: LogCallback,
    version: Synchronized,
    pause: Synchronized,
):
    base_seed = int(time.time())
    seed = base_seed + worker_id * 10000
    set_seed(seed)

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    th.set_num_threads(1)
    th.set_num_interop_threads(1)
    env = config.build_env(worker_id)
    model_copy: ActorCritic = config.build_model()
    now_model_version = -1

    env_reset_flag = False
    while True:
        t_act = 0.0
        t_step = 0.0
        break_rollout = False
        pause_flag = False
        try:
            if not env_reset_flag:
                obs, info = env.reset()
                env_reset_flag = True
            with version.get_lock():
                # end condition for workers
                if version.value == -1:
                    break
                if now_model_version < version.value:
                    now_model_version = version.value
                    model_copy.load_state_dict(model.state_dict())
                    model_copy.eval()
            if pause.value:
                pause_flag = True
                time.sleep(1)
                continue

            rollout_buffer = RolloutBuffer(config.obs_space)
            with th.inference_mode():
                done = False
                while not done:
                    t0 = time.perf_counter()
                    a, logp, _ = model_copy.act(obs)
                    t_act += time.perf_counter() - t0

                    a = a.cpu().numpy().item()

                    t1 = time.perf_counter()
                    next_obs, reward, terminated, truncated, info = env.step(a)
                    t_step += time.perf_counter() - t1

                    done = terminated or truncated
                    rollout_buffer.add(obs, a, reward, truncated, terminated, logp.cpu().numpy().item())  # type: ignore
                    obs = next_obs
                    if pause.value:
                        break_rollout = True
                        break
                if break_rollout:
                    print(f"[W{worker_id}] Rollout generation paused, discarding incomplete rollout.", flush=True)
                    rollout_buffer.clear()
                    env_reset_flag = False
                    continue

                rollout_buffer.add_last_obs(next_obs)
                print(f"[W{worker_id}] act {t_act:.3f}s step {t_step:.3f}s", flush=True)

            # it is possible that model is updated during rollout generation
            with version.get_lock():
                if now_model_version < version.value:
                    env_reset_flag = False
                    continue

            rollout_array = RolloutBufferArray(*rollout_buffer.to_numpy())
            rsh = RolloutShmHandles(*rollout_array.make_shmemory())
            try:
                metadata = RolloutMetadata(
                    now_model_version,
                    make_rollout_shm_name(rsh),
                    RolloutShape(*rollout_array.get_shapes()),
                    RolloutDtypes(*rollout_array.get_dtypes()),
                    log_from_array(rollout_array, log_callback),
                )
                queue.put(metadata)
            except Exception as e:
                print(f"Worker {worker_id} failed to put rollout to queue: {e}")
                raise e
            finally:
                env_reset_flag = False
                rsh.close()
        except Exception as e:
            env_reset_flag = False
            print(f"Worker {worker_id} encountered an error: {e}")
    env.close()


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
        log_callback: LogCallback,
        config: WorkerConfig,
    ) -> None:
        workers = []
        cpu_model.share_memory()
        version = mp.Value("i", 0)
        pause = mp.Value("i", 1)
        queue = mp.Queue(max_queue)
        for worker_id in range(n_worker):
            p = mp.Process(target=gen_rollout, args=(worker_id, queue, cpu_model, config, log_callback, version, pause))
            p.start()
            workers.append(p)

        self.cpu_model = cpu_model
        self.gpu_model = gpu_model
        self.version = version
        self.queue = queue
        self.workers = workers
        self.log_callback = log_callback
        self.pause = pause

    def collect_rollout(
        self,
        ts_per_iteration: int,
        gamma: float,
        lam: float,
        value_batch_size: int,
    ) -> tuple[PPOBatch, Log]:
        with self.pause.get_lock():
            self.pause.value = 0

        timestep = 0

        self.gpu_model.eval()

        ppo_batches = []
        logs = []

        progress = tqdm.tqdm(total=ts_per_iteration, desc="Collecting Rollout")

        while timestep < ts_per_iteration:
            metadata: RolloutMetadata = self.queue.get()
            shm_handler = make_shm_from_shm_name(metadata.rollout_shm_names)
            with self.version.get_lock():
                cur_ver = self.version.value

            if metadata.version < cur_ver:
                shm_handler.close()
                shm_handler.unlink()
                continue

            log = metadata.info
            logs.append(log)

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
                timestep += metadata.rollout_shapes.act[0]
                progress.update(metadata.rollout_shapes.act[0])
            finally:
                shm_handler.close()
                shm_handler.unlink()
        progress.close()
        with self.pause.get_lock():
            self.pause.value = 1

        def merge_logs(logs: list[Log], log_callback) -> Log:
            if isinstance(log_callback, dict):
                return {key: merge_logs([log[key] for log in logs], log_callback[key]) for key in log_callback.keys()}
            else:
                return log_callback.make_statistics(logs)

        return PPOBatch.concat(ppo_batches), merge_logs(logs, self.log_callback)

    def update_model(self):
        with self.version.get_lock():
            self.version.value += 1
            self.cpu_model.load_state_dict(self.gpu_model.state_dict())
            self.cpu_model.to("cpu")

    def close(self):
        with self.version.get_lock():
            self.version.value = -1
        for p in self.workers:
            p.join()
