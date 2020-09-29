from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from all.agents import DDPG
from all.approximation import QContinuous, PolyakTarget
from all.bodies import TimeFeature
from all.logging import DummyWriter
from all.policies import DeterministicPolicy
from all.memory import ExperienceReplayBuffer
from .models import fc_q, fc_deterministic_policy


def ddpg(
        # Common settings
        device="cuda",
        discount_factor=0.98,
        last_frame=2e6,
        # Adam optimizer settings
        lr_q=1e-3,
        lr_pi=1e-3,
        # Training settings
        minibatch_size=100,
        update_frequency=1,
        polyak_rate=0.005,
        # Replay Buffer settings
        replay_start_size=5000,
        replay_buffer_size=1e6,
        # Exploration settings
        noise=0.1,
        # Model construction
        q_model_constructor=fc_q,
        policy_model_constructor=fc_deterministic_policy
):
    """
    DDPG continuous control preset.

    Args:
        device (str): The device to load parameters and buffers onto for this agent..
        discount_factor (float): Discount factor for future rewards.
        last_frame (int): Number of frames to train.
        lr_q (float): Learning rate for the Q network.
        lr_pi (float): Learning rate for the policy network.
        minibatch_size (int): Number of experiences to sample in each training update.
        update_frequency (int): Number of timesteps per training update.
        polyak_rate (float): Speed with which to update the target network towards the online network.
        replay_start_size (int): Number of experiences in replay buffer when training begins.
        replay_buffer_size (int): Maximum number of experiences to store in the replay buffer.
        noise (float): The amount of exploration noise to add.
        q_model_constructor (function): The function used to construct the neural q model.
        policy_model_constructor (function): The function used to construct the neural policy model.
    """
    def _ddpg(env, writer=DummyWriter()):
        final_anneal_step = (last_frame - replay_start_size) // update_frequency

        q_model = q_model_constructor(env).to(device)
        q_optimizer = Adam(q_model.parameters(), lr=lr_q)
        q = QContinuous(
            q_model,
            q_optimizer,
            target=PolyakTarget(polyak_rate),
            scheduler=CosineAnnealingLR(
                q_optimizer,
                final_anneal_step
            ),
            writer=writer
        )

        policy_model = policy_model_constructor(env).to(device)
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
