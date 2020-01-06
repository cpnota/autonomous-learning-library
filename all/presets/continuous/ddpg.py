# /Users/cpnota/repos/autonomous-learning-library/all/approximation/value/action/torch.py
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from all import nn
from all.agents import DDPG
from all.approximation import QContinuous, PolyakTarget
from all.bodies import TimeFeature
from all.logging import DummyWriter
from all.policies import DeterministicPolicy
from all.memory import ExperienceReplayBuffer


def fc_value(env, hidden1=400, hidden2=300):
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(env.state_space.shape[0] + env.action_space.shape[0] + 1, hidden1),
        nn.ReLU(),
        nn.Linear(hidden1, hidden2),
        nn.ReLU(),
        nn.Linear0(hidden2, 1),
    )

def fc_policy(env, hidden1=400, hidden2=300):
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(env.state_space.shape[0] + 1, hidden1),
        nn.ReLU(),
        nn.Linear(hidden1, hidden2),
        nn.ReLU(),
        nn.Linear0(hidden2, env.action_space.shape[0]),
    )

def ddpg(
        lr_q=1e-3,
        lr_pi=1e-3,
        noise=0.1,
        replay_start_size=5000,
        replay_buffer_size=1e6,
        minibatch_size=100,
        discount_factor=0.99,
        polyak_rate=0.005,
        update_frequency=1,
        final_frame=2e6, # Anneal LR and clip until here
        hidden1=400,
        hidden2=300,
        device=torch.device('cuda')
):
    final_anneal_step = (final_frame - replay_start_size) // update_frequency

    def _ddpg(env, writer=DummyWriter()):
        value_model = fc_value(env, hidden1, hidden2).to(device)
        value_optimizer = Adam(value_model.parameters(), lr=lr_q)
        q = QContinuous(
            value_model,
            value_optimizer,
            target=PolyakTarget(polyak_rate),
            scheduler=CosineAnnealingLR(
                value_optimizer,
                final_anneal_step
            ),
            writer=writer
        )

        policy_model = fc_policy(env, hidden1, hidden2).to(device)
        policy_optimizer = Adam(policy_model.parameters(), lr=lr_pi)
        policy = DeterministicPolicy(
            policy_model,
            policy_optimizer,
            env.action_space,
            target=PolyakTarget(polyak_rate),
            scheduler=CosineAnnealingLR(
                policy_optimizer,
                final_anneal_step
            ),
            writer=writer
        )

        replay_buffer = ExperienceReplayBuffer(
            replay_buffer_size,
            device=device
        )

        return TimeFeature(DDPG(
            q,
            policy,
            replay_buffer,
            env.action_space,
            noise=noise,
            replay_start_size=replay_start_size,
            discount_factor=discount_factor,
            update_frequency=update_frequency,
            minibatch_size=minibatch_size,
        ))
    return _ddpg


__all__ = ["ddpg"]
