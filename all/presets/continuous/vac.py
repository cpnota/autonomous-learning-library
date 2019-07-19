# /Users/cpnota/repos/autonomous-learning-library/all/approximation/value/action/torch.py
import torch
from torch.optim import Adam
from all import nn
from all.agents import VAC
from all.approximation import VNetwork
from all.experiments import DummyWriter
from all.policies import GaussianPolicy


def fc_value(env):
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(env.state_space.shape[0], 256),
        nn.ReLU(),
        nn.Linear0(256, 1)
    )

def fc_policy(env):
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(env.state_space.shape[0], 256),
        nn.ReLU(),
        nn.Linear0(256, env.action_space.shape[0] * 2)
    )

class EmptyFeatureNetwork():
    def __call__(self, states):
        return states

    def eval(self, states):
        return states

    def reinforce(self):
        return

def vac(
        lr_v=2e-4,
        lr_pi=1e-4,
        entropy_loss_scaling=0.01,
        discount_factor=0.99,
        device=torch.device('cuda')
):
    def _vac(env, writer=DummyWriter()):
        features = EmptyFeatureNetwork()
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

        return VAC(features, v, policy, gamma=discount_factor)
    return _vac


__all__ = ["vac"]
