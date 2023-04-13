from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from rl_games.algos_torch.model_builder import register_network
from rl_games.algos_torch.network_builder import NetworkBuilder

from .pointnet import PointNetEncoder


class PointActorCritic(NetworkBuilder.BaseNetwork):
    def __init__(self, params, **kwargs) -> None:
        super().__init__()

        self.load(params=params)
        print("params: ", params)
        print("kwargs: ", kwargs)

        actions_num = kwargs.pop("actions_num")
        input_shape = kwargs.pop("input_shape")
        input_shape = input_shape[0] if isinstance(input_shape, tuple) else input_shape

        self.value_size = kwargs.pop("value_size", 1)
        self.num_points = kwargs.pop("num_points", 1024)

        self.mlp_input_size = 1024 + 256
        self.mlp_output_size = 128

        input_shape = input_shape - self.num_points * 3

        self.pointnet_encoder = PointNetEncoder()

        self.shared_mlp = nn.Sequential(
            nn.Linear(input_shape, 256),
            nn.ReLU(),
        )

        mlp_args = {
            "input_size": self.mlp_input_size,
            "units": self.units,
            "activation": self.activation,
            "norm_func_name": self.normalization,
            "dense_func": torch.nn.Linear,
            "d2rl": self.is_d2rl,
            "norm_only_first_layer": self.norm_only_first_layer,
        }

        self.actor_mlp = self._build_mlp(**mlp_args)
        self.critic_mlp = self._build_mlp(**mlp_args)

        self.value = nn.Linear(self.mlp_output_size, self.value_size)
        self.value_act = self.activations_factory.create(self.value_activation)

        # create mu and sigma layers for continuous action space
        self.mu = nn.Linear(self.mlp_output_size, actions_num)
        self.mu_act = self.activations_factory.create(self.space_config["mu_activation"])

        if self.fixed_sigma:
            self.sigma = nn.Parameter(torch.zeros(actions_num), requires_grad=True)
        else:
            self.sigma = nn.Linear(self.mlp_output_size, actions_num)
        self.sigma_act = self.activations_factory.create(self.space_config["sigma_activation"])

    def load(self, params):
        self.separate = params.get("separate", False)
        self.units = params["mlp"]["units"]
        self.activation = params["mlp"]["activation"]
        self.initializer = params["mlp"]["initializer"]
        self.is_d2rl = params["mlp"].get("d2rl", False)
        self.norm_only_first_layer = params["mlp"].get("norm_only_first_layer", False)
        self.value_activation = params.get("value_activation", "None")
        self.normalization = params.get("normalization", None)
        self.has_rnn = "rnn" in params
        self.has_space = "space" in params
        self.central_value = params.get("central_value", False)
        self.joint_obs_actions_config = params.get("joint_obs_actions", None)

        if self.has_space:
            self.is_multi_discrete = "multi_discrete" in params["space"]
            self.is_discrete = "discrete" in params["space"]
            self.is_continuous = "continuous" in params["space"]
            if self.is_continuous:
                self.space_config = params["space"]["continuous"]
                self.fixed_sigma = self.space_config["fixed_sigma"]
            elif self.is_discrete:
                self.space_config = params["space"]["discrete"]
            elif self.is_multi_discrete:
                self.space_config = params["space"]["multi_discrete"]
        else:
            self.is_discrete = False
            self.is_continuous = False
            self.is_multi_discrete = False

    def is_rnn(self):
        return False

    def forward(self, obs_dict: Dict[str, Any]) -> torch.Tensor:
        assert self.is_continuous, "Only continuous action space is supported for now"
        obs = obs_dict["obs"]

        pointcloud = obs[:, : 3 * self.num_points].reshape(-1, 3, self.num_points)
        proprioception = obs[:, 3 * self.num_points :]

        # print("pointcloud: ", pointcloud.shape)
        # print("proprioception: ", proprioception.shape)

        pointcloud_feat = self.pointnet_encoder(pointcloud)[0]
        # print("pointcloud_feat: ", pointcloud_feat.shape)
        proprioception_feat = self.shared_mlp(proprioception)
        # print("proprioception_feat: ", proprioception_feat.shape)
        feat = torch.cat([pointcloud_feat, proprioception_feat], dim=1)
        # print("feat: ", feat.shape)

        actor_out = self.actor_mlp(feat)
        critic_out = self.critic_mlp(feat)

        # print("actor_out: ", actor_out.shape)
        # print("critic_out: ", critic_out.shape)

        value = self.value_act(self.value(critic_out))
        mu = self.mu_act(self.mu(actor_out))

        if self.fixed_sigma:
            sigma = mu * 0.0 + self.sigma_act(self.sigma)
        else:
            sigma = self.sigma_act(self.sigma(actor_out))

        return mu, sigma, value, None


class PointActorCriticBuilder(NetworkBuilder):
    def __init__(self, **kwargs):
        NetworkBuilder.__init__(self)

    def load(self, params):
        self.params = params

    def build(self, name, **kwargs):
        print("Building PointActorCritic ...")
        print("params: ", self.params)
        print("kwargs: ", kwargs)
        return PointActorCritic(self.params, **kwargs)

    def __call__(self, name, **kwargs):
        return self.build(name, **kwargs)


register_network("point_actor_critic", PointActorCriticBuilder)
