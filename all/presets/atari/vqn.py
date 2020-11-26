import copy
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from all.approximation import QNetwork
from all.agents import VQN, VQNTestAgent
from all.bodies import DeepmindAtariBody
from all.logging import DummyWriter
from all.optim import LinearScheduler
from all.policies import ParallelGreedyPolicy
from .models import nature_ddqn
from ..builder import preset_builder
from ..preset import Preset


default_hyperparameters = {
    # Common settings
    "discount_factor": 0.99,
    # Adam optimizer settings
    "lr": 1e-3,
    "eps": 1.5e-4,
    # Explicit exploration
    "initial_exploration": 1.,
    "final_exploration": 0.01,
    "final_exploration_step": 250000,
    "test_exploration": 0.001,
    # Parallel actors
    "n_envs": 64,
    # Model construction
    "model_constructor": nature_ddqn
}


class VQNAtariPreset(Preset):
    """
    Vanilla Q-Network (VQN) Atari preset.

    Args:
        env (all.environments.AtariEnvironment): The environment for which to construct the agent.
        device (torch.device, optional): The device on which to load the agent.

    Keyword Args:
        discount_factor (float): Discount factor for future rewards.
        lr (float): Learning rate for the Adam optimizer.
        eps (float): Stability parameters for the Adam optimizer.
        initial_exploration (float): Initial probability of choosing a random action,
            decayed over course of training.
        final_exploration (float): Final probability of choosing a random action.
        final_exploration_step (int): The step at which exploration decay is finished
        test_exploration (float): The exploration rate of the test Agent
        n_envs (int): Number of parallel environments.
        model_constructor (function): The function used to construct the neural model.
    """

    def __init__(self, env, device="cuda", **hyperparameters):
        hyperparameters = {**default_hyperparameters, **hyperparameters}
        super().__init__(n_envs=hyperparameters['n_envs'])
        self.model = hyperparameters['model_constructor'](env).to(device)
        self.hyperparameters = hyperparameters
        self.n_actions = env.action_space.n
        self.device = device

    def agent(self, writer=DummyWriter(), train_steps=float('inf')):
        n_updates = train_steps / self.hyperparameters['n_envs']

        optimizer = Adam(
            self.model.parameters(),
            lr=self.hyperparameters['lr'],
            eps=self.hyperparameters['eps']
        )

        q = QNetwork(
            self.model,
            optimizer,
            scheduler=CosineAnnealingLR(optimizer, n_updates),
            writer=writer
        )

        policy = ParallelGreedyPolicy(
            q,
            self.n_actions,
            epsilon=LinearScheduler(
                self.hyperparameters['initial_exploration'],
                self.hyperparameters['final_exploration'],
                0,
                self.hyperparameters["final_exploration_step"] / self.hyperparameters["n_envs"],
                name="exploration",
                writer=writer
            )
        )

        return DeepmindAtariBody(
            VQN(q, policy, discount_factor=self.hyperparameters['discount_factor']),
        )

    def test_agent(self):
        q = QNetwork(copy.deepcopy(self.model))
        return DeepmindAtariBody(
            VQNTestAgent(q, self.n_actions, exploration=self.hyperparameters['test_exploration'])
        )


vqn = preset_builder('vqn', default_hyperparameters, VQNAtariPreset)
