from torch.optim import Adam
from all.approximation import QNetwork
from all.agents import VSarsa
from all.bodies import DeepmindAtariBody
from all.logging import DummyWriter
from all.optim import LinearScheduler
from all.policies import GreedyPolicy
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
):
    def _vsarsa(envs, writer=DummyWriter()):
        action_repeat = 4
        final_exploration_timestep = final_exploration_frame / action_repeat

        env = envs[0]
        model = nature_ddqn(env).to(device)
        optimizer = Adam(model.parameters(), lr=lr, eps=eps)
        q = QNetwork(
            model,
            optimizer,
            env.action_space.n,
            writer=writer
        )
        policy = GreedyPolicy(
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
            VSarsa(q, policy, gamma=discount_factor),
        )
    return _vsarsa, n_envs
