from typing import cast, List, Dict

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
