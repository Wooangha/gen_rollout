from typing import TypedDict, Type, Optional, TypeVar, Generic

import numpy as np
import torch as th
import torch.nn as nn

from .utils import obs_to_tensor

ObsType = th.Tensor | dict[str, "ObsType"] | tuple["ObsType", ...]
ObsNumpy = np.ndarray | dict[str, "ObsNumpy"] | tuple["ObsNumpy", ...]


def numpy2torch(obs: ObsNumpy | ObsType, device: Optional[th.device] = None) -> ObsType:
    """Convert a numpy observation to a torch tensor."""
    if isinstance(obs, dict):
        return {key: numpy2torch(subobs, device) for key, subobs in obs.items()}
    elif isinstance(obs, tuple):
        return tuple(numpy2torch(subobs, device) for subobs in obs)
    else:
        if isinstance(obs, th.Tensor):
            if obs.dtype != th.float32:
                obs = obs.float()
            return obs.to(device) if device is not None else obs
        tensor = th.from_numpy(obs).float()
        if device is not None:
            tensor = tensor.to(device)
        return tensor


class FeatureExtractor(nn.Module):
    def forward(self, obs: ObsType) -> th.Tensor:
        """Extract features from observations"""
        raise NotImplementedError


def layer_init(layer: nn.Linear | nn.Conv2d, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    if layer.bias is not None:
        nn.init.constant_(layer.bias, bias_const)


# Only discrete action space is supported for now
class ActorCritic(nn.Module):
    def __init__(
        self,
        feature_extractor_cls: Type[FeatureExtractor] | tuple[Type[FeatureExtractor], Type[FeatureExtractor]],
        policy_head_cls: Type[nn.Module],
        value_head_cls: Type[nn.Module],
        feature_extractor_kwargs: Optional[dict] = None,
        policy_head_kwargs: Optional[dict] = None,
        value_head_kwargs: Optional[dict] = None,
    ) -> None:
        """Actor-Critic model

        Args:
            feature_extractor_cls (Type[FeatureExtractorBase] | tuple[Type[FeatureExtractorBase], Type[FeatureExtractorBase]]): feature_extractor | (feature_extractor_actor, feature_extractor_critic)
            feature_extractor_kwargs: Optional[dict],
            policy_head_cls (Type[ModuleBase]): policy head
            policy_head_kwargs (dict): policy head kwargs
            value_head_cls (Type[ModuleBase]): value head
            value_head_kwargs (dict): value head kwargs
        """
        super().__init__()
        if isinstance(feature_extractor_cls, tuple):
            self.feature_extractor = nn.ModuleList(
                [
                    (
                        feature_extractor_cls[0](**feature_extractor_kwargs)
                        if feature_extractor_kwargs
                        else feature_extractor_cls[0]()
                    ),
                    (
                        feature_extractor_cls[1](**feature_extractor_kwargs)
                        if feature_extractor_kwargs
                        else feature_extractor_cls[1]()
                    ),
                ]
            )
            self.feature_extractor[0].apply(
                lambda m: layer_init(m) if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) else None
            )
            self.feature_extractor[1].apply(
                lambda m: layer_init(m) if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) else None
            )
        else:
            self.feature_extractor = (
                feature_extractor_cls(**feature_extractor_kwargs) if feature_extractor_kwargs else feature_extractor_cls()
            )
            self.feature_extractor.apply(
                lambda m: layer_init(m) if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) else None
            )
        self.policy_head = policy_head_cls(**policy_head_kwargs) if policy_head_kwargs else policy_head_cls()
        self.value_head = value_head_cls(**value_head_kwargs) if value_head_kwargs else value_head_cls()

        self.policy_head.apply(
            lambda m: layer_init(m, std=0.01) if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) else None
        )
        self.value_head.apply(lambda m: layer_init(m, std=1.00) if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) else None)

    def feature_extract(self, obs: ObsType | ObsNumpy) -> th.Tensor | tuple[th.Tensor, th.Tensor]:
        """Extract features from observations"""
        obs = numpy2torch(obs)
        if isinstance(self.feature_extractor, tuple):
            feat_actor = self.feature_extractor[0](obs)
            feat_critic = self.feature_extractor[1](obs)
            return feat_actor, feat_critic
        else:
            return self.feature_extractor(obs)

    def value_policy_dist(self, obs: ObsType) -> tuple[th.Tensor, th.distributions.Categorical]:
        feat_out = self.feature_extract(obs)
        if isinstance(feat_out, tuple):
            feat_actor, feat_critic = feat_out
            val: th.Tensor = self.value_head(feat_critic)
            logits: th.Tensor = self.policy_head(feat_actor)
        else:
            val: th.Tensor = self.value_head(feat_out)
            logits: th.Tensor = self.policy_head(feat_out)
        return val.squeeze(-1), th.distributions.Categorical(logits=logits)

    def only_policy_dist(self, obs: ObsType) -> th.distributions.Categorical:
        feat_out = self.feature_extract(obs)
        if isinstance(feat_out, tuple):
            feat_actor, _ = feat_out
            logits: th.Tensor = self.policy_head(feat_actor)
        else:
            logits: th.Tensor = self.policy_head(feat_out)
        return th.distributions.Categorical(logits=logits)

    def only_value(self, obs: ObsType) -> th.Tensor:
        feat_out = self.feature_extract(obs)
        if isinstance(feat_out, tuple):
            _, feat_critic = feat_out
            val: th.Tensor = self.value_head(feat_critic)
        else:
            val: th.Tensor = self.value_head(feat_out)
        return val.squeeze(-1)

    @th.no_grad()
    def act(self, obs: ObsType):
        dist = self.only_policy_dist(obs)
        a = dist.sample()
        logp = dist.log_prob(a)
        ent = dist.entropy()
        return a, logp, ent

    def forward(self, obs: ObsType) -> tuple[th.Tensor, th.distributions.Categorical]:
        return self.value_policy_dist(obs)
