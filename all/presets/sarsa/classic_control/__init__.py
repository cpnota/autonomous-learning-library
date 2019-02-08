# /Users/cpnota/repos/autonomous-learning-library/all/approximation/value/action/torch.py
from torch import nn
from torch.optim import Adam
from all.layers import Flatten
from all.agents import Sarsa
from all.approximation import QTabular
from all.policies import GreedyPolicy


def fc_net(env, frames=1):
    return nn.Sequential(
        Flatten(),
        nn.Linear(env.state_space.shape[0] * frames, 256),
        nn.ReLU(),
        nn.Linear(256, env.action_space.n)
    )

def sarsa_cc(
        lr=1e-3,
        annealing_time=10000,
        final_epsilon=0.02
):
    def _sarsa_cc(env):
        model = fc_net(env)
        optimizer = Adam(model.parameters(), lr=lr)
        q = QTabular(model, optimizer)
        policy = GreedyPolicy(q, annealing_time=annealing_time, final_epsilon=final_epsilon)
        return Sarsa(q, policy)
    return _sarsa_cc


__all__ = ["sarsa_cc"]
