from torch.optim import Adam
from all.approximation import QNetwork
from all.agents import VSarsa
from all.bodies import DeepmindAtariBody
from all.logging import DummyWriter
from all.optim import LinearScheduler
from all.policies import ParallelGreedyPolicy
from .models import nature_ddqn

def vsarsa(
        # Common settings
        device="cuda",
        discount_factor=0.99,
        # Adam optimizer settings
        lr=1e-3,
        eps=1.5e-4,
        # Exploration settings
        final_exploration_frame=1000000,
        final_exploration=0.02,
        initial_exploration=1.,
        # Parallel actors
        n_envs=64,
        # Model construction
        model_constructor=nature_ddqn
):
    """
    Vanilla SARSA Atari preset.

    Args:
        device (str): The device to load parameters and buffers onto for this agent.
        discount_factor (float): Discount factor for future rewards.
        lr (float): Learning rate for the Adam optimizer.
        eps (float): Stability parameters for the Adam optimizer.
        initial_exploration (int): Initial probability of choosing a random action,
            decayed until final_exploration_frame.
        final_exploration (int): Final probability of choosing a random action.
        final_exploration_frame (int): The frame where the exploration decay stops.
        n_envs (int): Number of parallel environments.
        model_constructor (function): The function used to construct the neural model.
    """
    def _vsarsa(envs, writer=DummyWriter()):
        action_repeat = 4
        final_exploration_timestep = final_exploration_frame / action_repeat

        env = envs[0]
        model = model_constructor(env).to(device)
        optimizer = Adam(model.parameters(), lr=lr, eps=eps)
        q = QNetwork(
            model,
            optimizer,
            writer=writer
        )
        policy = ParallelGreedyPolicy(
            q,
            env.action_space.n,
            epsilon=LinearScheduler(
                initial_exploration,
                final_exploration,
                0,
                final_exploration_timestep,
                name="epsilon",
                writer=writer
            )
        )
        return DeepmindAtariBody(
            VSarsa(q, policy, discount_factor=discount_factor),
        )
    return _vsarsa, n_envs
