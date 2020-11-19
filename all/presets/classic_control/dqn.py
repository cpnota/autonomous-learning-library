import copy
from torch.optim import Adam
from all.agents import DQN, DQNTestAgent
from all.approximation import QNetwork, FixedTarget
from all.logging import DummyWriter
from all.memory import ExperienceReplayBuffer
from all.optim import LinearScheduler
from all.policies import GreedyPolicy
from .models import fc_relu_q
from ..builder import preset_builder
from ..preset import Preset


default_hyperparameters = {
    # Common settings
    "discount_factor": 0.99,
    # Adam optimizer settings
    "lr": 1e-3,
    # Training settings
    "minibatch_size": 64,
    "update_frequency": 1,
    "target_update_frequency": 100,
    # Replay buffer settings
    "replay_start_size": 1000,
    "replay_buffer_size": 10000,
    # Explicit exploration
    "initial_exploration": 1.,
    "final_exploration": 0.,
    "final_exploration_step": 10000,
    "test_exploration": 0.001,
    # Model construction
    "model_constructor": fc_relu_q
}

class DQNClassicControlPreset(Preset):
    """
    Deep Q-Network (DQN) Classic Control Preset.

    Args:
        env (all.environments.GymEnvironment): The environment for which to construct the agent.
        device (torch.device, optional): The device on which to load the agent.

    Keyword Args:
        discount_factor (float, optional): Discount factor for future rewards.
        lr (float): Learning rate for the Adam optimizer.
        minibatch_size (int): Number of experiences to sample in each training update.
        update_frequency (int): Number of timesteps per training update.
        target_update_frequency (int): Number of timesteps between updates the target network.
        replay_start_size (int): Number of experiences in replay buffer when training begins.
        replay_buffer_size (int): Maximum number of experiences to store in the replay buffer.
        initial_exploration (float): Initial probability of choosing a random action,
            decayed over course of training.
        final_exploration (float): Final probability of choosing a random action.
        final_exploration_step (int): The step at which exploration decay is finished
        test_exploration (float): The exploration rate of the test Agent
        model_constructor (function): The function used to construct the neural model.
    """
    def __init__(self, env, device="cuda", **hyperparameters):
        super().__init__()
        hyperparameters = {**default_hyperparameters, **hyperparameters}
        self.model = hyperparameters['model_constructor'](env).to(device)
        self.hyperparameters = hyperparameters
        self.n_actions = env.action_space.n
        self.device = device

    def agent(self, writer=DummyWriter(), train_steps=float('inf')):
        n_updates = (train_steps - self.hyperparameters['replay_start_size']) / self.hyperparameters['update_frequency']

        optimizer = Adam(self.model.parameters(), lr=self.hyperparameters['lr'])

        q = QNetwork(
            self.model,
            optimizer,
            target=FixedTarget(self.hyperparameters['target_update_frequency']),
            writer=writer
        )

        policy = GreedyPolicy(
            q,
            self.n_actions,
            epsilon=LinearScheduler(
                self.hyperparameters['initial_exploration'],
                self.hyperparameters['final_exploration'],
                self.hyperparameters['replay_start_size'],
                self.hyperparameters['final_exploration_step'] - self.hyperparameters['replay_start_size'],
                name="exploration",
                writer=writer
            )
        )

        replay_buffer = ExperienceReplayBuffer(
            self.hyperparameters['replay_buffer_size'],
            device=self.device
        )

        return DQN(
            q,
            policy,
            replay_buffer,
            discount_factor=self.hyperparameters['discount_factor'],
            minibatch_size=self.hyperparameters['minibatch_size'],
            replay_start_size=self.hyperparameters['replay_start_size'],
            update_frequency=self.hyperparameters['update_frequency'],
        )

    def test_agent(self):
        q =  QNetwork(copy.deepcopy(self.model))
        return DQNTestAgent(q, self.n_actions, exploration=self.hyperparameters['test_exploration'])

dqn = preset_builder('dqn', default_hyperparameters, DQNClassicControlPreset)
