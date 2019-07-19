# /Users/cpnota/repos/autonomous-learning-library/all/approximation/value/action/torch.py
import torch
from torch.optim import RMSprop
from torch.nn.functional import smooth_l1_loss
from all.approximation import QNetwork
from all.agents import VSarsa
from all.bodies import ParallelAtariBody
from all.experiments import DummyWriter
from all.policies import GreedyPolicy
from .models import nature_ddqn

def vsarsa(
        action_repeat=4,
        alpha=0.999, # RMSprop smoothing
        discount_factor=0.99,
        eps=1.5e-4,  # RMSprop stability
        final_exploration_frame=1000000,
        final_exploration=0.02,
        initial_exploration=1.,
        lr=1e-3,
        n_envs=64,
        device=torch.device('cpu')
):
    # counted by number of updates rather than number of frame
    final_exploration_frame /= action_repeat

    def _vsarsa(envs, writer=DummyWriter()):
        env = envs[0]
        model = nature_ddqn(env).to(device)
        optimizer = RMSprop(model.parameters(), lr=lr, alpha=alpha, eps=eps)
        q = QNetwork(
            model,
            optimizer,
            env.action_space.n,
            loss=smooth_l1_loss,
            writer=writer
        )
        policy = GreedyPolicy(q,
                              env.action_space.n,
                              annealing_time=final_exploration_frame,
                              initial_epsilon=initial_exploration,
                              final_epsilon=final_exploration
                              )
        return ParallelAtariBody(
            VSarsa(q, policy, gamma=discount_factor),
            envs,
            action_repeat=action_repeat,
        )
    return _vsarsa, n_envs
