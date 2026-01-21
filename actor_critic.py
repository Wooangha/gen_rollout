import numpy as np
import torch as th
import torch.nn as nn

ObsType = th.Tensor | dict[str, "ObsType"] | tuple["ObsType", ...]


class FeatureExtractor(nn.Module):
    def forward(self, obs: ObsType) -> th.Tensor:
        """Extract features from observations"""
        raise NotImplementedError


# Only discrete action space is supported for now
class ActorCritic(nn.Module):
    def __init__(
        self,
        feature_extractor: nn.Module | tuple[nn.Module, nn.Module],
        policy_head: nn.Module,
        value_head: nn.Module,
        device: th.device,
    ) -> None:
        """Actor-Critic model

        Args:
            feature_extractor (nn.Module | tuple[nn.Module, nn.Module]): feature_extractor | (feature_extractor_actor, feature_extractor_critic)
            policy_head (nn.Module): policy head
            value_head (nn.Module): value head
        """
        super().__init__()
        self.feature_extractor = feature_extractor
        self.policy_head = policy_head
        self.value_head = value_head
        self.device = device

    def feature_extract(self, obs: ObsType) -> th.Tensor | tuple[th.Tensor, th.Tensor]:
        """Extract features from observations"""
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
