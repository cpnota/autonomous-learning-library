import copy
from torch.optim import Adam
from all.agents import DDQN, DDQNTestAgent
from all.approximation import QNetwork, FixedTarget
from all.logging import DummyWriter
from all.memory import PrioritizedReplayBuffer
from all.optim import LinearScheduler
from all.policies import GreedyPolicy
from .models import dueling_fc_relu_q
from ..builder import preset_builder
from ..preset import Preset


default_hyperparameters = {
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
    # Exploration settings
    "initial_exploration": 1.,
    "final_exploration": 0.,
    "final_exploration_step": 10000,
    "test_exploration": 0.001,
    # Prioritized replay settings
    "alpha": 0.2,
    "beta": 0.6,
    # Model construction
    "model_constructor": dueling_fc_relu_q
}

class DDQNClassicControlPreset(Preset):
    def __init__(self, env, device="cuda", **hyperparameters):
        hyperparameters = {**default_hyperparameters, **hyperparameters}
        super().__init__()
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

            replay_buffer = PrioritizedReplayBuffer(
                self.hyperparameters['replay_buffer_size'],
                alpha=self.hyperparameters['alpha'],
                beta=self.hyperparameters['beta'],
                device=self.device
            )

            return DDQN(
                q,
                policy,
                replay_buffer,
                discount_factor=self.hyperparameters["discount_factor"],
                minibatch_size=self.hyperparameters["minibatch_size"],
                replay_start_size=self.hyperparameters["replay_start_size"],
                update_frequency=self.hyperparameters["update_frequency"],
            )

    def test_agent(self):
        q =  QNetwork(copy.deepcopy(self.model))
        return DDQNTestAgent(q, self.n_actions, exploration=self.hyperparameters['test_exploration'])

ddqn = preset_builder('ddqn', default_hyperparameters, DDQNClassicControlPreset)
