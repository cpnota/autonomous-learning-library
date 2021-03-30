import copy
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from all.agents import PPO, PPOTestAgent
from all.approximation import VNetwork, FeatureNetwork, Identity
from all.bodies import TimeFeature
from all.logging import DummyWriter
from all.optim import LinearScheduler
from all.policies import GaussianPolicy
from all.presets.builder import ParallelPresetBuilder
from all.presets.preset import ParallelPreset
from all.presets.continuous.models import fc_policy, fc_v


default_hyperparameters = {
    # Common settings
    "discount_factor": 0.98,
    # Adam optimizer settings
    "lr": 3e-4,  # Adam learning rate
    "eps": 1e-5,  # Adam stability
    # Loss scaling
    "entropy_loss_scaling": 0.01,
    "value_loss_scaling": 0.5,
    # Training settings
    "clip_grad": 0.5,
    "clip_initial": 0.2,
    "clip_final": 0.01,
    "epochs": 20,
    "minibatches": 4,
    # Batch settings
    "n_envs": 32,
    "n_steps": 128,
    # GAE settings
    "lam": 0.95,
    # Model construction
    "value_model_constructor": fc_v,
    "policy_model_constructor": fc_policy,
}


class PPOContinuousPreset(ParallelPreset):
    """
    Proximal Policy Optimization (PPO) Continuous Control Preset.

    Args:
        env (all.environments.AtariEnvironment): The environment for which to construct the agent.
        name (str): A human-readable name for the preset.
        device (torch.device): The device on which to load the agent.

    Keyword Args:
        discount_factor (float): Discount factor for future rewards.
        lr (float): Learning rate for the Adam optimizer.
        eps (float): Stability parameters for the Adam optimizer.
        entropy_loss_scaling (float): Coefficient for the entropy term in the total loss.
        value_loss_scaling (float): Coefficient for the value function loss.
        clip_grad (float): Clips the gradient during training so that its L2 norm (calculated over all parameters)
        # is no greater than this bound. Set to 0 to disable.
        clip_initial (float): Value for epsilon in the clipped PPO objective function at the beginning of training.
        clip_final (float): Value for epsilon in the clipped PPO objective function at the end of training.
        epochs (int): Number of times to literature through each batch.
        minibatches (int): The number of minibatches to split each batch into.
        n_envs (int): Number of parallel actors.
        n_steps (int): Length of each rollout.
        lam (float): The Generalized Advantage Estimate (GAE) decay parameter.
        value_model_constructor (function): The function used to construct the neural value model.
        policy_model_constructor (function): The function used to construct the neural policy model.
    """

    def __init__(self, env, name, device, **hyperparameters):
        super().__init__(name, device, hyperparameters)
        self.value_model = hyperparameters["value_model_constructor"](env).to(device)
        self.policy_model = hyperparameters["policy_model_constructor"](env).to(device)
        self.action_space = env.action_space

    def agent(self, writer=DummyWriter(), train_steps=float('inf')):
        n_updates = train_steps * self.hyperparameters['epochs'] * self.hyperparameters['minibatches'] / (self.hyperparameters['n_steps'] * self.hyperparameters['n_envs'])

        value_optimizer = Adam(self.value_model.parameters(), lr=self.hyperparameters['lr'], eps=self.hyperparameters['eps'])
        policy_optimizer = Adam(self.policy_model.parameters(), lr=self.hyperparameters['lr'], eps=self.hyperparameters['eps'])

        features = Identity(self.device)

        v = VNetwork(
            self.value_model,
            value_optimizer,
            loss_scaling=self.hyperparameters['value_loss_scaling'],
            clip_grad=self.hyperparameters['clip_grad'],
            writer=writer,
            scheduler=CosineAnnealingLR(
                value_optimizer,
                n_updates
            ),
        )

        policy = GaussianPolicy(
            self.policy_model,
            policy_optimizer,
            self.action_space,
            clip_grad=self.hyperparameters['clip_grad'],
            writer=writer,
            scheduler=CosineAnnealingLR(
                policy_optimizer,
                n_updates
            ),
        )

        return TimeFeature(PPO(
            features,
            v,
            policy,
            epsilon=LinearScheduler(
                self.hyperparameters['clip_initial'],
                self.hyperparameters['clip_final'],
                0,
                n_updates,
                name='clip',
                writer=writer
            ),
            epochs=self.hyperparameters['epochs'],
            minibatches=self.hyperparameters['minibatches'],
            n_envs=self.hyperparameters['n_envs'],
            n_steps=self.hyperparameters['n_steps'],
            discount_factor=self.hyperparameters['discount_factor'],
            lam=self.hyperparameters['lam'],
            entropy_loss_scaling=self.hyperparameters['entropy_loss_scaling'],
            writer=writer,
        ))

    def test_agent(self):
        policy = GaussianPolicy(copy.deepcopy(self.policy_model), space=self.action_space)
        return TimeFeature(PPOTestAgent(Identity(self.device), policy))

    def parallel_test_agent(self):
        return self.test_agent()


ppo = ParallelPresetBuilder('ppo', default_hyperparameters, PPOContinuousPreset)
