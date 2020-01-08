# /Users/cpnota/repos/autonomous-learning-library/all/approximation/value/action/torch.py
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from all.agents import SAC
from all.approximation import QContinuous, PolyakTarget, VNetwork
from all.bodies import TimeFeature
from all.logging import DummyWriter
from all.policies.soft_deterministic import SoftDeterministicPolicy
from all.memory import ExperienceReplayBuffer
from .models import fc_q, fc_v, fc_soft_policy


def sac(
        lr_q=1e-3,
        lr_v=1e-3,
        lr_pi=1e-4,
        lr_temperature=1e-5,
        temperature_initial=0.1,
        entropy_target_scaling=1.,
        replay_start_size=5000,
        replay_buffer_size=1e6,
        minibatch_size=100,
        discount_factor=0.98,
        polyak_rate=0.005,
        update_frequency=2,
        final_frame=2e6, # Anneal LR and clip until here
        device=torch.device('cuda')
):
    def _sac(env, writer=DummyWriter()):
        final_anneal_step = (final_frame - replay_start_size) // update_frequency

        q_1_model = fc_q(env).to(device)
        q_1_optimizer = Adam(q_1_model.parameters(), lr=lr_q)
        q_1 = QContinuous(
            q_1_model,
            q_1_optimizer,
            scheduler=CosineAnnealingLR(
                q_1_optimizer,
                final_anneal_step
            ),
            writer=writer,
            name='q_1'
        )

        q_2_model = fc_q(env).to(device)
        q_2_optimizer = Adam(q_2_model.parameters(), lr=lr_q)
        q_2 = QContinuous(
            q_2_model,
            q_2_optimizer,
            scheduler=CosineAnnealingLR(
                q_2_optimizer,
                final_anneal_step
            ),
            writer=writer,
            name='q_2'
        )

        v_model = fc_v(env).to(device)
        v_optimizer = Adam(v_model.parameters(), lr=lr_v)
        v = VNetwork(
            v_model,
            v_optimizer,
            scheduler=CosineAnnealingLR(
                v_optimizer,
                final_anneal_step
            ),
            target=PolyakTarget(polyak_rate),
            writer=writer,
            name='v',
        )

        policy_model = fc_soft_policy(env).to(device)
        policy_optimizer = Adam(policy_model.parameters(), lr=lr_pi)
        policy = SoftDeterministicPolicy(
            policy_model,
            policy_optimizer,
            env.action_space,
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

        return TimeFeature(SAC(
            policy,
            q_1,
            q_2,
            v,
            replay_buffer,
            temperature_initial=temperature_initial,
            entropy_target=(-env.action_space.shape[0] * entropy_target_scaling),
            lr_temperature=lr_temperature,
            replay_start_size=replay_start_size,
            discount_factor=discount_factor,
            update_frequency=update_frequency,
            minibatch_size=minibatch_size,
            writer=writer
        ))
    return _sac
