from torch.optim import Adam
from all.agents import VQN
from all.approximation import QNetwork
from all.policies import GreedyPolicy
from all.logging import DummyWriter
from .models import fc_relu_q

def vqn(
        # Common settings
        device="cpu",
        gamma=0.99,
        # Adam optimizer settings
        lr=1e-2,
        eps=1e-5,
        # Exploration settings
        epsilon=0.1,
        # Parallel actors
        n_envs=1,
):
    def _vqn(envs, writer=DummyWriter()):
        env = envs[0]
        model = fc_relu_q(env).to(device)
        optimizer = Adam(model.parameters(), lr=lr, eps=eps)
        q = QNetwork(model, optimizer, env.action_space.n, writer=writer)
        policy = GreedyPolicy(q, env.action_space.n, epsilon=epsilon)
        return VQN(q, policy, gamma=gamma)
    return _vqn, n_envs
 