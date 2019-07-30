# /Users/cpnota/repos/autonomous-learning-library/all/approximation/value/action/torch.py
import torch
from torch.optim import Adam
from all import nn
from all.agents import DDPG
from all.approximation import QContinuous, PolyakTarget
from all.logging import DummyWriter
from all.policies import DeterministicPolicy
from all.memory import ExperienceReplayBuffer


def fc_value(env):
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(env.state_space.shape[0] + env.action_space.shape[0], 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear0(64, 1)
    )

def fc_policy(env):
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(env.state_space.shape[0], 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear0(64, env.action_space.shape[0]),
        nn.TanhActionBound(env.action_space)
    )

def ddpg(
        lr_q=1e-3,
        lr_pi=1e-4,
        noise=0.1,
        replay_start_size=5000,
        replay_buffer_size=50000,
        minibatch_size=64,
        discount_factor=0.99,
        polyak_rate=0.001,
        update_frequency=1,
        device=torch.device('cuda')
):
    def _ddpg(env, writer=DummyWriter()):
        value_model = fc_value(env).to(device)
        value_optimizer = Adam(value_model.parameters(), lr=lr_q)
        q = QContinuous(
            value_model,
            value_optimizer,
            target=PolyakTarget(polyak_rate),
            writer=writer
        )

        policy_model = fc_policy(env).to(device)
        policy_optimizer = Adam(policy_model.parameters(), lr=lr_pi)
        policy = DeterministicPolicy(
            policy_model,
            policy_optimizer,
            env.action_space,
            noise,
            target=PolyakTarget(polyak_rate),
            writer=writer
        )

        replay_buffer = ExperienceReplayBuffer(
            replay_buffer_size,
            device=device
        )

        return DDPG(
            q,
            policy,
            replay_buffer,
            replay_start_size=replay_start_size,
            discount_factor=discount_factor,
            update_frequency=update_frequency,
            minibatch_size=minibatch_size
        )
    return _ddpg


__all__ = ["ddpg"]
