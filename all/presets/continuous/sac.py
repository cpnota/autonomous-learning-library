# /Users/cpnota/repos/autonomous-learning-library/all/approximation/value/action/torch.py
import torch
from torch.optim import Adam
from all import nn
from all.agents import SAC
from all.approximation import QContinuous, PolyakTarget, VNetwork
from all.experiments import DummyWriter
from all.policies.soft_deterministic import SoftDeterministicPolicy
from all.memory import ExperienceReplayBuffer


def fc_q(env):
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(env.state_space.shape[0] + env.action_space.shape[0], 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    )

def fc_v(env):
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(env.state_space.shape[0], 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    )

def fc_policy(env):
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(env.state_space.shape[0], 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear0(64, env.action_space.shape[0] * 2),
    )

def sac(
        lr_q=3e-4,
        lr_v=3e-4,
        lr_pi=3e-4,
        entropy_regularizer=0.1,
        replay_start_size=5000,
        replay_buffer_size=50000,
        minibatch_size=64,
        discount_factor=0.99,
        polyak_rate=0.005,
        update_frequency=1,
        device=torch.device('cuda')
):
    def _sac(env, writer=DummyWriter()):
        q1_model = fc_q(env).to(device)
        q1_optimizer = Adam(q1_model.parameters(), lr=lr_q)
        q1 = QContinuous(
            q1_model,
            q1_optimizer,
            target=PolyakTarget(polyak_rate),
            writer=writer,
            name='q_1'
        )
        q2_model = fc_q(env).to(device)
        q2_optimizer = Adam(q2_model.parameters(), lr=lr_q)
        q2 = QContinuous(
            q2_model,
            q2_optimizer,
            target=PolyakTarget(polyak_rate),
            writer=writer,
            name='q_2'
        )

        v_model = fc_v(env).to(device)
        v_optimizer = Adam(v_model.parameters(), lr=lr_v)
        v = VNetwork(
            v_model,
            v_optimizer,
            target=PolyakTarget(polyak_rate),
            writer=writer,
            name='v',
        )

        policy_model = fc_policy(env).to(device)
        policy_optimizer = Adam(policy_model.parameters(), lr=lr_pi)
        policy = SoftDeterministicPolicy(
            policy_model,
            policy_optimizer,
            env.action_space
        )

        replay_buffer = ExperienceReplayBuffer(
            replay_buffer_size,
            device=device
        )

        return SAC(
            policy,
            q1,
            q2,
            v,
            replay_buffer,
            entropy_regularizer=entropy_regularizer,
            replay_start_size=replay_start_size,
            discount_factor=discount_factor,
            update_frequency=update_frequency,
            minibatch_size=minibatch_size,
            writer=writer
        )
    return _sac
