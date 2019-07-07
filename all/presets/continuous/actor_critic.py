# /Users/cpnota/repos/autonomous-learning-library/all/approximation/value/action/torch.py
import torch
from torch import nn
from torch.optim import Adam
from all.layers import Flatten, Linear0
from all.agents import ActorCritic
from all.approximation import VNetwork
from all.experiments import DummyWriter
from all.policies import GaussianPolicy


def fc_value(env):
    return nn.Sequential(
        Flatten(),
        nn.Linear(env.state_space.shape[0], 256),
        nn.ReLU(),
        Linear0(256, 1)
    )

def fc_policy(env):
    return nn.Sequential(
        Flatten(),
        nn.Linear(env.state_space.shape[0], 256),
        nn.ReLU(),
        Linear0(256, env.action_space.shape[0] * 2)
    )

def actor_critic(
        lr_v=2e-4,
        lr_pi=1e-4,
        entropy_loss_scaling=0.01,
        device=torch.device('cuda')
):
    def _actor_critic(env, writer=DummyWriter()):
        value_model = fc_value(env).to(device)
        value_optimizer = Adam(value_model.parameters(), lr=lr_v)
        v = VNetwork(value_model, value_optimizer, writer=writer)

        policy_model = fc_policy(env).to(device)
        policy_optimizer = Adam(policy_model.parameters(), lr=lr_pi)
        policy = GaussianPolicy(
            policy_model,
            policy_optimizer,
            env.action_space.shape[0],
            entropy_loss_scaling=entropy_loss_scaling,
            writer=writer
        )

        return ActorCritic(v, policy)
    return _actor_critic


__all__ = ["actor_critic"]
