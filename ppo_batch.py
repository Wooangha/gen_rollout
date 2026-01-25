from typing import cast, List, Dict, Iterator
from collections import deque

from multiprocessing.shared_memory import SharedMemory

from dataclasses import dataclass
import torch as th

from .utils import make_shm_from_obs_array

ObsTensor = th.Tensor | dict[str, "ObsTensor"] | tuple["ObsTensor", ...]


def _concat_obs_tensors(obs_tensors: list[ObsTensor]) -> ObsTensor:
    """Concatenate a list of observation tensors."""
    if isinstance(obs_tensors[0], dict):
        o: list[dict[str, ObsTensor]] = cast(list[dict[str, ObsTensor]], obs_tensors)
        return {key: _concat_obs_tensors([obs[key] for obs in o]) for key in o[0].keys()}
    elif isinstance(obs_tensors[0], tuple):
        o_: list[tuple[ObsTensor, ...]] = cast(list[tuple[ObsTensor, ...]], obs_tensors)
        return tuple(_concat_obs_tensors([obs[i] for obs in o_]) for i in range(len(o_[0])))
    else:
        o__ = cast(list[th.Tensor], obs_tensors)
        return th.cat(o__, dim=0)


@dataclass
class PPOBatch:
    obs: ObsTensor
    act: th.Tensor
    logp_old: th.Tensor
    adv: th.Tensor
    ret: th.Tensor
    val_old: th.Tensor

    @staticmethod
    def concat(batches: list["PPOBatch"]) -> "PPOBatch":
        """Concatenate a list of PPOBatch into a single PPOBatch."""
        obs = _concat_obs_tensors([batch.obs for batch in batches])
        act = th.cat([batch.act for batch in batches], dim=0)
        logp_old = th.cat([batch.logp_old for batch in batches], dim=0)
        adv = th.cat([batch.adv for batch in batches], dim=0)
        ret = th.cat([batch.ret for batch in batches], dim=0)
        val_old = th.cat([batch.val_old for batch in batches], dim=0)
        return PPOBatch(obs, act, logp_old, adv, ret, val_old)


def _obs_to_batch_tensor(obs: ObsTensor, batch_idx: th.Tensor, device: th.device | str) -> ObsTensor:
    """Convert an observation tensor to a batch tensor."""
    if isinstance(obs, dict):
        return {key: _obs_to_batch_tensor(subobs, batch_idx, device) for key, subobs in obs.items()}
    elif isinstance(obs, tuple):
        return tuple(_obs_to_batch_tensor(subobs, batch_idx, device) for subobs in obs)
    else:
        return obs[batch_idx].to(device)


def queue2batch(queue: deque[PPOBatch], batch_size: int, device: th.device | str) -> Iterator[PPOBatch]:
    """Convert a deque of PPOBatch to a single PPOBatch by concatenation."""

    p = PPOBatch.concat(list(queue))
    size = p.logp_old.shape[0]
    idx = th.arange(p.logp_old.shape[0])

    idx = idx[th.randperm(size)]
    for start in range(0, size, batch_size):
        end = start + batch_size
        mb_idx = idx[start:end]
        yield PPOBatch(
            obs=_obs_to_batch_tensor(p.obs, mb_idx, device),
            act=p.act[mb_idx].to(device),
            logp_old=p.logp_old[mb_idx].to(device),
            adv=p.adv[mb_idx].to(device),
            ret=p.ret[mb_idx].to(device),
            val_old=p.val_old[mb_idx].to(device),
        )
