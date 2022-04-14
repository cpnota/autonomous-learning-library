import copy
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from all.approximation import QNetwork
from all.agents import VSarsa, VSarsaTestAgent
from all.bodies import DeepmindAtariBody
from all.logging import DummyLogger
from all.optim import LinearScheduler
from all.policies import GreedyPolicy, ParallelGreedyPolicy
from all.presets.builder import ParallelPresetBuilder
from all.presets.preset import ParallelPreset
from all.presets.classic_control.models import dueling_fc_relu_q


default_hyperparameters = {
    # Common settings
    "discount_factor": 0.99,
    # Adam optimizer settings
    "lr": 1e-2,
    "eps": 1.5e-4,
    # Explicit exploration
    "initial_exploration": 1.,
    "final_exploration": 0.,
    "final_exploration_step": 10000,
    "test_exploration": 0.001,
    # Parallel actors
    "n_envs": 8,
    # Model construction
    "model_constructor": dueling_fc_relu_q
}


class VSarsaClassicControlPreset(ParallelPreset):
    """
    Vanilla SARSA (VSarsa) Classic Control Preset.

    Args:
        env (all.environments.AtariEnvironment): The environment for which to construct the agent.
        name (str): A human-readable name for the preset.
        device (torch.device): The device on which to load the agent.

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

    def __init__(self, env, name, device, **hyperparameters):
        super().__init__(name, device, hyperparameters)
        self.model = hyperparameters['model_constructor'](env).to(device)
        self.n_actions = env.action_space.n

    def agent(self, logger=DummyLogger(), train_steps=float('inf')):
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
            logger=logger
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
                logger=logger
            )
        )

        return VSarsa(q, policy, discount_factor=self.hyperparameters['discount_factor'])

    def test_agent(self):
        q = QNetwork(copy.deepcopy(self.model))
        policy = GreedyPolicy(q, self.n_actions, epsilon=self.hyperparameters["test_exploration"])
        return VSarsaTestAgent(policy)

    def parallel_test_agent(self):
        q = QNetwork(copy.deepcopy(self.model))
        policy = ParallelGreedyPolicy(q, self.n_actions, epsilon=self.hyperparameters["test_exploration"])
        return VSarsaTestAgent(policy)


vsarsa = ParallelPresetBuilder('vsarsa', default_hyperparameters, VSarsaClassicControlPreset)
