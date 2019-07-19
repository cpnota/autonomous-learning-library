# /Users/cpnota/repos/autonomous-learning-library/all/approximation/value/action/torch.py
import torch
from torch.optim import RMSprop
from all.agents import VQN
from all.approximation import QNetwork
from all.policies import GreedyPolicy
from all.experiments import DummyWriter
from .models import fc_relu_q

def vqn(
        lr=1e-2,
        epsilon=0.1,
        gamma=0.99,
        alpha=0.999, # RMSprop smoothing
        eps=1e-5,    # RMSprop stability
        n_envs=1,
        device=torch.device('cpu')
):
    def _vqn(envs, writer=DummyWriter()):
        env = envs[0]
        model = fc_relu_q(env).to(device)
        optimizer = RMSprop(model.parameters(), lr=lr, alpha=alpha, eps=eps)
        q = QNetwork(model, optimizer, env.action_space.n, writer=writer)
        policy = GreedyPolicy(q, env.action_space.n, annealing_time=1, final_epsilon=epsilon)
        return VQN(q, policy, gamma=gamma)
    return _vqn, n_envs
 