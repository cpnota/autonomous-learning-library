import copy
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from all.approximation import QDist, FixedTarget
from all.agents import Rainbow, RainbowTestAgent
from all.bodies import DeepmindAtariBody
from all.logging import DummyWriter
from all.memory import PrioritizedReplayBuffer, NStepReplayBuffer
from all.optim import LinearScheduler
from .models import nature_rainbow
from ..builder import preset_builder
from ..preset import Preset


default_hyperparameters = {
    "discount_factor": 0.99,
    "lr": 1e-4,
    "eps": 1.5e-4,
    # Training settings
    "minibatch_size": 32,
    "update_frequency": 4,
    "target_update_frequency": 1000,
    # Replay buffer settings
    "replay_start_size": 80000,
    "replay_buffer_size": 1000000,
    # Explicit exploration
    "initial_exploration": 0.02,
    "final_exploration": 0.,
    "test_exploration": 0.001,
    # Prioritized replay settings
    "alpha": 0.5,
    "beta": 0.5,
    # Multi-step learning
    "n_steps": 3,
    # Distributional RL
    "atoms": 51,
    "v_min": -10,
    "v_max": 10,
    # Noisy Nets
    "sigma": 0.5,
}

class RainbowAtariPreset(Preset):
    def __init__(self, hyperparameters, env, device='cuda'):
        super().__init__()
        self.model = nature_rainbow(env, atoms=hyperparameters["atoms"], sigma=hyperparameters["sigma"]).to(device)
        self.hyperparameters = hyperparameters
        self.n_actions = env.action_space.n
        self.device = device

    def agent(self, writer=DummyWriter(), train_steps=float('inf')):
        n_updates = (train_steps - self.hyperparameters['replay_start_size']) / self.hyperparameters['update_frequency']

        optimizer = Adam(
            self.model.parameters(),
            lr=self.hyperparameters['lr'],
            eps=self.hyperparameters['eps']
        )

        q_dist = QDist(
            self.model,
            optimizer,
            self.n_actions,
            self.hyperparameters['atoms'],
            scheduler=CosineAnnealingLR(optimizer, n_updates),
            v_min=self.hyperparameters['v_min'],
            v_max=self.hyperparameters['v_max'],
            target=FixedTarget(self.hyperparameters['target_update_frequency']),
            writer=writer,
        )

        replay_buffer = NStepReplayBuffer(
            self.hyperparameters['n_steps'],
            self.hyperparameters['discount_factor'],
            PrioritizedReplayBuffer(
                self.hyperparameters['replay_buffer_size'],
                alpha=self.hyperparameters['alpha'],
                beta=self.hyperparameters['beta'],
                device=self.device
            )
        )

        return DeepmindAtariBody(
            Rainbow(
                q_dist,
                replay_buffer,
                exploration=LinearScheduler(
                    self.hyperparameters['initial_exploration'],
                    self.hyperparameters['final_exploration'],
                    0,
                    train_steps - self.hyperparameters['replay_start_size'],
                    name="epsilon",
                    writer=writer
                ),
                discount_factor=self.hyperparameters['discount_factor'] ** self.hyperparameters["n_steps"],
                minibatch_size=self.hyperparameters['minibatch_size'],
                replay_start_size=self.hyperparameters['replay_start_size'],
                update_frequency=self.hyperparameters['update_frequency'],
                writer=writer,
            ),
            lazy_frames=True,
            episodic_lives=True
        )

    def test_agent(self):
        q_dist = QDist(
            self.model,
            None,
            self.n_actions,
            self.hyperparameters['atoms'],
            v_min=self.hyperparameters['v_min'],
            v_max=self.hyperparameters['v_max'],
        )
        return DeepmindAtariBody(RainbowTestAgent(q_dist, self.n_actions, self.hyperparameters["test_exploration"]))

rainbow = preset_builder('rainbow', default_hyperparameters, RainbowAtariPreset)
