# /Users/cpnota/repos/autonomous-learning-library/all/approximation/value/action/torch.py
from torch.optim import Adam
from all.approximation import QTabular
from all.agents import DQN
from all.policies import GreedyPolicy
from ..model import deep_q_atari

def dqn(env):
    model = deep_q_atari(env)
    optimizer = Adam(model.parameters(), lr=1e-4)
    q = QTabular(model, optimizer, target_update_frequency=250)
    policy = GreedyPolicy(q, annealing_time=250000, initial_epsilon=1.00, final_epsilon=0.02)
    return DQN(q, policy)

__all__ = ["dqn"]
